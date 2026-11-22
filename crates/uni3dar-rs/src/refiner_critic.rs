use crate::format::{silu, ModelBundle};

const OE62_12_ELEM_IDS: [usize; 12] = [0, 4, 5, 6, 7, 8, 13, 14, 15, 16, 34, 52];
const DONOR_ELEM_IDS: [usize; 4] = [6, 7, 15, 14];
const ACCEPTOR_ELEM_IDS: [usize; 5] = [4, 8, 16, 34, 52];
const HALOGEN_ELEM_IDS: [usize; 4] = [8, 16, 34, 52];

#[derive(Debug, Clone)]
pub struct RefinerResult {
    pub refined_pos: Vec<[f32; 3]>,
    pub bond_logits: Vec<f32>,      // (K * K * 5)
    pub bond_order_probs: Vec<f32>, // (K * K * 5)
}

#[derive(Debug, Clone)]
pub struct CriticPrediction {
    pub mu_norm: Vec<f32>,
    pub std_norm: Vec<f32>,
    pub logit_legal: f32,
    pub prob_legal: f32,
    pub logit_constr: f32,
    pub prob_constr: f32,
    pub utility: f32,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn logsigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

/// Exact Jacobi symmetric 3x3 eigenvalue solver returning eigenvalues sorted descending `[l0 >= l1 >= l2]`.
pub fn eigvalsh_3x3_desc(mut a: [[f32; 3]; 3]) -> [f32; 3] {
    for _ in 0..24 {
        let mut p = 0;
        let mut q = 1;
        let mut max_off = a[0][1].abs();
        if a[0][2].abs() > max_off {
            max_off = a[0][2].abs();
            p = 0;
            q = 2;
        }
        if a[1][2].abs() > max_off {
            max_off = a[1][2].abs();
            p = 1;
            q = 2;
        }
        if max_off < 1e-10 {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let r = 3 - p - q;
        let arp = a[r][p];
        let arq = a[r][q];
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        a[r][p] = c * arp - s * arq;
        a[p][r] = a[r][p];
        a[r][q] = s * arp + c * arq;
        a[q][r] = a[r][q];
    }
    let mut ev = [a[0][0].max(0.0), a[1][1].max(0.0), a[2][2].max(0.0)];
    ev.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
    ev
}

/// Run `PostTreeGraphRefiner` (`< 0.3 ms` single pass) to heal octant-seam coordinate jumps and predict `bond_order_probs`.
pub fn run_post_tree_refiner(
    bundle: &ModelBundle,
    elem_ids: &[usize],
    pos_0: &[[f32; 3]],
) -> Result<Option<RefinerResult>, String> {
    let Some(rc) = &bundle.header.refiner else {
        return Ok(None);
    };
    let k_len = elem_ids.len();
    if k_len == 0 {
        return Ok(Some(RefinerResult {
            refined_pos: Vec::new(),
            bond_logits: Vec::new(),
            bond_order_probs: Vec::new(),
        }));
    }

    let d = rc.d_model;
    let n_heads = rc.n_heads;
    let hd = d / n_heads;
    let n_rbf = rc.n_rbf;
    let d_pair = rc.d_pair;
    let rbf_centers = bundle.f32_slice("ref.rbf_centers")?;
    let elem_emb = bundle.f32_slice("ref.elem_emb")?;

    let mut h = vec![0.0f32; k_len * d];
    for i in 0..k_len {
        let eid = elem_ids[i].min(rc.num_atom_classes - 1);
        h[i * d..(i + 1) * d].copy_from_slice(&elem_emb[eid * d..(eid + 1) * d]);
    }
    let mut cur_pos = pos_0.to_vec();

    if k_len > 1 {
        let mut rbf_ij = vec![0.0f32; k_len * k_len * n_rbf];
        let mut d_mat = vec![0.0f32; k_len * k_len];
        let mut u_mat = vec![[0.0f32; 3]; k_len * k_len];

        for l in 0..rc.n_layers {
            let lp = format!("ref.layers.{l}");
            // Pairwise u_ij = pos[j] - pos[i]
            for i in 0..k_len {
                for j in 0..k_len {
                    let u = [
                        cur_pos[j][0] - cur_pos[i][0],
                        cur_pos[j][1] - cur_pos[i][1],
                        cur_pos[j][2] - cur_pos[i][2],
                    ];
                    let dij = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                    u_mat[i * k_len + j] = u;
                    d_mat[i * k_len + j] = dij;
                    let dc = dij.clamp(0.0, 6.0);
                    let r_slice = &mut rbf_ij[(i * k_len + j) * n_rbf..(i * k_len + j + 1) * n_rbf];
                    for m in 0..n_rbf {
                        let diff = dc - rbf_centers[m];
                        r_slice[m] = (-rc.rbf_gamma * diff * diff).exp();
                    }
                }
            }

            // 1. Bidirectional 3D-RBF attention
            let mut xn = h.clone();
            bundle.eval_layernorm_in_place(&format!("{lp}.norm1"), &mut xn, k_len, d, 1e-5)?;
            let mut qkv = vec![0.0f32; k_len * 3 * d];
            bundle.eval_linear(&format!("{lp}.qkv"), &xn, k_len, &mut qkv)?;

            let mut bias_mid = vec![0.0f32; k_len * k_len * (n_heads * 2)];
            bundle.eval_linear(&format!("{lp}.rbf_bias.0"), &rbf_ij, k_len * k_len, &mut bias_mid)?;
            for v in &mut bias_mid {
                *v = silu(*v);
            }
            let mut rbf_bias = vec![0.0f32; k_len * k_len * n_heads];
            bundle.eval_linear(&format!("{lp}.rbf_bias.2"), &bias_mid, k_len * k_len, &mut rbf_bias)?;

            let scale = 1.0 / (hd as f32).sqrt();
            let mut attn_cat = vec![0.0f32; k_len * d];
            let mut scores = vec![0.0f32; k_len];

            for head in 0..n_heads {
                for i in 0..k_len {
                    let q_vec = &qkv[i * 3 * d + head * hd..i * 3 * d + (head + 1) * hd];
                    let mut max_s = f32::NEG_INFINITY;
                    for j in 0..k_len {
                        let k_vec = &qkv[j * 3 * d + d + head * hd..j * 3 * d + d + (head + 1) * hd];
                        let mut dot = 0.0f32;
                        for c in 0..hd {
                            dot += q_vec[c] * k_vec[c];
                        }
                        let s = dot * scale + rbf_bias[(i * k_len + j) * n_heads + head];
                        scores[j] = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }
                    let mut sum_e = 0.0f32;
                    for j in 0..k_len {
                        let e = (scores[j] - max_s).exp();
                        scores[j] = e;
                        sum_e += e;
                    }
                    let inv = 1.0 / sum_e.max(1e-12);
                    let out_s = &mut attn_cat[i * d + head * hd..i * d + (head + 1) * hd];
                    for j in 0..k_len {
                        let w = scores[j] * inv;
                        let v_vec = &qkv[j * 3 * d + 2 * d + head * hd..j * 3 * d + 2 * d + (head + 1) * hd];
                        for c in 0..hd {
                            out_s[c] += w * v_vec[c];
                        }
                    }
                }
            }

            let mut out_p = vec![0.0f32; k_len * d];
            bundle.eval_linear(&format!("{lp}.out_proj"), &attn_cat, k_len, &mut out_p)?;
            for idx in 0..(k_len * d) {
                h[idx] += out_p[idx];
            }

            let mut xn2 = h.clone();
            bundle.eval_layernorm_in_place(&format!("{lp}.norm2"), &mut xn2, k_len, d, 1e-5)?;
            let mut ffn_m = vec![0.0f32; k_len * 2 * d];
            bundle.eval_linear(&format!("{lp}.ffn.0"), &xn2, k_len, &mut ffn_m)?;
            for v in &mut ffn_m {
                *v = silu(*v);
            }
            let mut ffn_o = vec![0.0f32; k_len * d];
            bundle.eval_linear(&format!("{lp}.ffn.3"), &ffn_m, k_len, &mut ffn_o)?;
            for idx in 0..(k_len * d) {
                h[idx] += ffn_o[idx];
            }

            // 2. SE(3)-equivariant radial force update (`0.15 < d_ij < 3.80 A`)
            let mut c_src = vec![0.0f32; k_len * 64];
            let mut c_dst = vec![0.0f32; k_len * 64];
            let mut c_rbf = vec![0.0f32; k_len * k_len * 64];
            bundle.eval_linear(&format!("{lp}.coord_src"), &h, k_len, &mut c_src)?;
            bundle.eval_linear(&format!("{lp}.coord_dst"), &h, k_len, &mut c_dst)?;
            bundle.eval_linear(&format!("{lp}.coord_rbf"), &rbf_ij, k_len * k_len, &mut c_rbf)?;

            for i in 0..k_len {
                for j in 0..k_len {
                    let ef = &mut c_rbf[(i * k_len + j) * 64..(i * k_len + j + 1) * 64];
                    for c in 0..64 {
                        ef[c] = silu(c_src[i * 64 + c] + c_dst[j * 64 + c] + ef[c]);
                    }
                }
            }
            let mut cm1 = vec![0.0f32; k_len * k_len * 32];
            bundle.eval_linear(&format!("{lp}.coord_mlp.1"), &c_rbf, k_len * k_len, &mut cm1)?;
            for v in &mut cm1 {
                *v = silu(*v);
            }
            let mut w_ij = vec![0.0f32; k_len * k_len];
            bundle.eval_linear(&format!("{lp}.coord_mlp.3"), &cm1, k_len * k_len, &mut w_ij)?;

            let mut next_pos = cur_pos.clone();
            for i in 0..k_len {
                let mut acc = [0.0f32; 3];
                let mut cnt_ok = 0.0f32;
                for j in 0..k_len {
                    let dij = d_mat[i * k_len + j];
                    if i != j && dij > 0.15 && dij < 3.80 {
                        cnt_ok += 1.0;
                        let w = w_ij[i * k_len + j];
                        let inv_d = 1.0 / dij.max(1e-4);
                        let u = u_mat[i * k_len + j];
                        acc[0] += w * u[0] * inv_d;
                        acc[1] += w * u[1] * inv_d;
                        acc[2] += w * u[2] * inv_d;
                    }
                }
                let denom = cnt_ok.max(1.0);
                next_pos[i][0] += acc[0] / denom;
                next_pos[i][1] += acc[1] / denom;
                next_pos[i][2] += acc[2] / denom;
            }
            cur_pos = next_pos;
        }
    }

    // Radial-tanh SO(3)-equivariant norm bound
    let mut refined_pos = vec![[0.0f32; 3]; k_len];
    let max_d = rc.max_disp_A.max(1e-4);
    for i in 0..k_len {
        let dx = cur_pos[i][0] - pos_0[i][0];
        let dy = cur_pos[i][1] - pos_0[i][1];
        let dz = cur_pos[i][2] - pos_0[i][2];
        let dn = (dx * dx + dy * dy + dz * dz).sqrt();
        let sc = rc.max_disp_A * (dn / max_d).tanh() / dn.max(1e-6);
        refined_pos[i] = [pos_0[i][0] + dx * sc, pos_0[i][1] + dy * sc, pos_0[i][2] + dz * sc];
    }

    // Symmetric BilinearBondHead (`5` classes)
    let mut rbf_ref = vec![0.0f32; k_len * k_len * n_rbf];
    for i in 0..k_len {
        for j in 0..k_len {
            let dx = refined_pos[i][0] - refined_pos[j][0];
            let dy = refined_pos[i][1] - refined_pos[j][1];
            let dz = refined_pos[i][2] - refined_pos[j][2];
            let dc = (dx * dx + dy * dy + dz * dz).sqrt().clamp(0.0, 6.0);
            let rs = &mut rbf_ref[(i * k_len + j) * n_rbf..(i * k_len + j + 1) * n_rbf];
            for m in 0..n_rbf {
                let diff = dc - rbf_centers[m];
                rs[m] = (-rc.rbf_gamma * diff * diff).exp();
            }
        }
    }

    let mut hm = vec![0.0f32; k_len * d_pair];
    let mut ha = vec![0.0f32; k_len * d_pair];
    let mut z_ij = vec![0.0f32; k_len * k_len * d_pair];
    bundle.eval_linear("ref.bond.proj_mul", &h, k_len, &mut hm)?;
    bundle.eval_linear("ref.bond.proj_add", &h, k_len, &mut ha)?;
    bundle.eval_linear("ref.bond.proj_rbf", &rbf_ref, k_len * k_len, &mut z_ij)?;

    for i in 0..k_len {
        for j in 0..k_len {
            let zs = &mut z_ij[(i * k_len + j) * d_pair..(i * k_len + j + 1) * d_pair];
            for c in 0..d_pair {
                zs[c] += hm[i * d_pair + c] * hm[j * d_pair + c] + ha[i * d_pair + c] + ha[j * d_pair + c];
            }
        }
    }
    bundle.eval_layernorm_in_place("ref.bond.head.0", &mut z_ij, k_len * k_len, d_pair, 1e-5)?;
    for v in &mut z_ij {
        *v = silu(*v);
    }
    let mut bh_mid = vec![0.0f32; k_len * k_len * (d_pair / 2)];
    bundle.eval_linear("ref.bond.head.2", &z_ij, k_len * k_len, &mut bh_mid)?;
    for v in &mut bh_mid {
        *v = silu(*v);
    }
    let mut bond_logits = vec![0.0f32; k_len * k_len * 5];
    bundle.eval_linear("ref.bond.head.4", &bh_mid, k_len * k_len, &mut bond_logits)?;

    let mut bond_probs = vec![0.0f32; k_len * k_len * 5];
    for p in 0..(k_len * k_len) {
        let ls = &bond_logits[p * 5..(p + 1) * 5];
        let mx = ls.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0f32;
        for c in 0..5 {
            let e = (ls[c] - mx).exp();
            bond_probs[p * 5 + c] = e;
            s += e;
        }
        let inv = 1.0 / s.max(1e-12);
        for c in 0..5 {
            bond_probs[p * 5 + c] *= inv;
        }
    }

    Ok(Some(RefinerResult {
        refined_pos,
        bond_logits,
        bond_order_probs: bond_probs,
    }))
}

/// Compute `(14,)` coarse 3D voxel geometry descriptors at the `L1 -> L0` boundary.
pub fn compute_coarse_voxel_geometry(
    levels: &[i32],
    types: &[usize],
    phy_pos: &[[f32; 3]],
    scores: &[f32],
    atom_mask_id: usize,
) -> [f32; 14] {
    let mut out = [0.0f32; 14];
    let n = levels.len();
    if n == 0 {
        return out;
    }
    let mut l1_pts = Vec::new();
    let mut l2_pts = Vec::new();
    for i in 0..n {
        if types[i] == atom_mask_id {
            if levels[i] == 1 {
                l1_pts.push(phy_pos[i]);
            } else if levels[i] == 2 {
                l2_pts.push(phy_pos[i]);
            }
        }
    }
    let k1 = l1_pts.len() as f32;
    let k2 = l2_pts.len() as f32;
    out[0] = k1 / 50.0;
    out[1] = k2 / 25.0;
    out[2] = (1.0 + k1).ln();

    let pts = if !l1_pts.is_empty() { &l1_pts } else { &l2_pts };
    if pts.len() >= 2 {
        let mut p_min = [f32::INFINITY; 3];
        let mut p_max = [f32::NEG_INFINITY; 3];
        let mut mean = [0.0f32; 3];
        for &p in pts {
            for a in 0..3 {
                p_min[a] = p_min[a].min(p[a]);
                p_max[a] = p_max[a].max(p[a]);
                mean[a] += p[a];
            }
        }
        let cnt_f = pts.len() as f32;
        for a in 0..3 {
            mean[a] /= cnt_f;
        }
        let spans = [
            (p_max[0] - p_min[0]).max(0.0),
            (p_max[1] - p_min[1]).max(0.0),
            (p_max[2] - p_min[2]).max(0.0),
        ];
        let mut spans_sorted = spans;
        spans_sorted.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        out[3] = spans_sorted[0] / 10.0;
        out[4] = spans_sorted[1] / 10.0;
        out[5] = spans_sorted[2] / 10.0;
        out[6] = (spans[0] * spans[0] + spans[1] * spans[1] + spans[2] * spans[2]).sqrt() / 15.0;
        out[7] = (spans[0] * spans[1] * spans[2]) / 500.0;

        let mut sq_sum = 0.0f32;
        let mut cov = [[0.0f32; 3]; 3];
        for &p in pts {
            let c = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
            sq_sum += c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
            for i in 0..3 {
                for j in 0..3 {
                    cov[i][j] += c[i] * c[j];
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] /= cnt_f;
            }
        }
        let rg = (sq_sum / cnt_f).max(1e-8).sqrt();
        out[8] = rg / 5.0;
        let evals = eigvalsh_3x3_desc(cov);
        out[9] = evals[0] / 10.0;
        out[10] = evals[1] / 10.0;
        out[11] = evals[2] / 10.0;
        out[12] = evals[2] / (evals[0] + 1e-4);
    }

    let sc_len = scores.len().min(n);
    let mut t_sum = 0.0f32;
    let mut t_cnt = 0usize;
    for i in 0..sc_len {
        if levels[i] > 0 && scores[i] != 0.0 {
            t_sum += scores[i];
            t_cnt += 1;
        }
    }
    if t_cnt > 0 {
        out[13] = (-t_sum / (t_cnt as f32)).clamp(0.0, 5.0);
    }
    out
}

/// Evaluate Stage-1 `OctreeProcessCritic` + utility score at `L1 -> L0`.
pub fn eval_stage1_critic(
    bundle: &ModelBundle,
    s1_h_concat: &[f32],
    geom_14: &[f32; 14],
    targets_norm: &[f32],
    active_mask: &[bool],
    legal_weight: f32,
    constr_weight: f32,
) -> Result<Option<CriticPrediction>, String> {
    let Some(c1) = &bundle.header.stage1_critic else {
        return Ok(None);
    };
    let d4 = 4 * c1.emb_dim;
    let hid = c1.hidden_dim;
    let k_t = c1.num_targets;

    let mut hn = s1_h_concat[..d4].to_vec();
    bundle.eval_layernorm_in_place("c1.h_norm", &mut hn, 1, d4, 1e-5)?;
    let mut h_vec = vec![0.0f32; hid];
    bundle.eval_linear("c1.h_proj", &hn, 1, &mut h_vec)?;
    for v in &mut h_vec {
        *v = silu(*v);
    }

    let mut aux_in = Vec::with_capacity(14 + 2 * k_t);
    aux_in.extend_from_slice(geom_14);
    for k in 0..k_t {
        aux_in.push(targets_norm.get(k).copied().unwrap_or(0.0));
    }
    for k in 0..k_t {
        aux_in.push(if active_mask.get(k).copied().unwrap_or(false) { 1.0 } else { 0.0 });
    }

    let mut a0 = vec![0.0f32; 128];
    bundle.eval_linear("c1.aux_proj.0", &aux_in, 1, &mut a0)?;
    for v in &mut a0 {
        *v = silu(*v);
    }
    let mut aux_vec = vec![0.0f32; 128];
    bundle.eval_linear("c1.aux_proj.2", &a0, 1, &mut aux_vec)?;
    for v in &mut aux_vec {
        *v = silu(*v);
    }

    let mut cat = Vec::with_capacity(hid + 128);
    cat.extend_from_slice(&h_vec);
    cat.extend_from_slice(&aux_vec);

    let mut t0 = vec![0.0f32; hid];
    bundle.eval_linear("c1.trunk.0", &cat, 1, &mut t0)?;
    bundle.eval_layernorm_in_place("c1.trunk.1", &mut t0, 1, hid, 1e-5)?;
    for v in &mut t0 {
        *v = silu(*v);
    }
    let feat_d = hid / 2;
    let mut z = vec![0.0f32; feat_d];
    bundle.eval_linear("c1.trunk.4", &t0, 1, &mut z)?;
    bundle.eval_layernorm_in_place("c1.trunk.5", &mut z, 1, feat_d, 1e-5)?;
    for v in &mut z {
        *v = silu(*v);
    }

    let mut mu_norm = vec![0.0f32; k_t];
    let mut lv_raw = vec![0.0f32; k_t];
    let mut l_leg = [0.0f32; 1];
    let mut l_con = [0.0f32; 1];
    bundle.eval_linear("c1.prop_mean", &z, 1, &mut mu_norm)?;
    bundle.eval_linear("c1.prop_logvar", &z, 1, &mut lv_raw)?;
    bundle.eval_linear("c1.legal", &z, 1, &mut l_leg)?;
    bundle.eval_linear("c1.constr", &z, 1, &mut l_con)?;

    let std_norm: Vec<f32> = lv_raw.iter().map(|&v| (0.5 * v.clamp(-6.0, 3.0)).exp()).collect();
    let logit_legal = l_leg[0];
    let logit_constr = l_con[0];
    let prob_legal = sigmoid(logit_legal);
    let prob_constr = sigmoid(logit_constr);

    let mut err_sum = 0.0f32;
    let mut act_cnt = 0.0f32;
    for k in 0..k_t {
        if active_mask.get(k).copied().unwrap_or(false) {
            let tn = targets_norm.get(k).copied().unwrap_or(0.0);
            err_sum += (mu_norm[k] - tn).abs() + 0.12 * std_norm[k];
            act_cnt += 1.0;
        }
    }
    let mean_err = err_sum / act_cnt.max(1.0);
    let leg_sf = (0.70 - prob_legal).max(0.0);
    let con_sf = (0.55 - prob_constr).max(0.0);
    let mut utility = -mean_err
        - legal_weight * (3.0 * leg_sf + 0.08 * (-logsigmoid(logit_legal)))
        - constr_weight * (2.5 * con_sf + 0.10 * (-logsigmoid(logit_constr)));
    if geom_14[0] * 50.0 < 1.5 {
        utility -= 50.0;
    }

    Ok(Some(CriticPrediction {
        mu_norm,
        std_norm,
        logit_legal,
        prob_legal,
        logit_constr,
        prob_constr,
        utility,
    }))
}

/// Compute `(42,)` element-resolved 3D cloud features for Stage-2 `OctreeElementProcessCritic`.
pub fn compute_element_cloud_features(elem_ids: &[usize], pos: &[[f32; 3]], coarse_geom_14: &[f32; 14]) -> [f32; 42] {
    let mut out = [0.0f32; 42];
    out[..14].copy_from_slice(coarse_geom_14);

    let n = elem_ids.len();
    let n_tot = n as f32;
    let n_h = elem_ids.iter().filter(|&&e| e == 0).count() as f32;
    let n_heavy = (n_tot - n_h).max(0.0);
    let d_tot = n_tot.max(1.0);
    let d_hvy = n_heavy.max(1.0);

    out[14] = n_tot / 50.0;
    out[15] = n_heavy / 35.0;
    out[16] = n_h / 35.0;
    out[17] = n_h / d_hvy;
    let z_sum: usize = elem_ids.iter().map(|&e| e + 1).sum();
    out[18] = if (z_sum % 2) == 0 { 1.0 } else { 0.0 };

    for (idx, &eid) in OE62_12_ELEM_IDS.iter().enumerate() {
        let cnt = elem_ids.iter().filter(|&&e| e == eid).count() as f32;
        out[19 + idx] = cnt / d_tot;
    }

    let n_c = elem_ids.iter().filter(|&&e| e == 5).count() as f32;
    let n_n = elem_ids.iter().filter(|&&e| e == 6).count() as f32;
    let n_p = elem_ids.iter().filter(|&&e| e == 14).count() as f32;
    let n_hal = elem_ids.iter().filter(|&&e| HALOGEN_ELEM_IDS.contains(&e)).count() as f32;
    let rdbe = 0.5 * (2.0 * n_c + 2.0 + n_n + n_p - n_h - n_hal);
    out[31] = rdbe.clamp(-4.0, 40.0) / 15.0;
    out[32] = (rdbe / d_hvy).clamp(-0.5, 2.0);

    let n_don = elem_ids.iter().filter(|&&e| DONOR_ELEM_IDS.contains(&e)).count() as f32;
    let n_acc = elem_ids.iter().filter(|&&e| ACCEPTOR_ELEM_IDS.contains(&e)).count() as f32;
    out[33] = n_don / d_hvy;
    out[34] = n_acc / d_hvy;
    out[35] = (n_don - n_acc) / d_hvy;

    if n >= 2 {
        let mut clash_cnt = 0.0f32;
        let mut iso_cnt = 0.0f32;
        let mut sum_nn = 0.0f32;
        let mut sum_deg = 0.0f32;
        let mut sum_hvy_deg = 0.0f32;
        let mut over_coord_cnt = 0.0f32;

        for i in 0..n {
            let mut nn_d = f32::INFINITY;
            let mut cov_deg = 0.0f32;
            let mut hvy_deg = 0.0f32;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = pos[i][0] - pos[j][0];
                let dy = pos[i][1] - pos[j][1];
                let dz = pos[i][2] - pos[j][2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                if d < nn_d {
                    nn_d = d;
                }
                if d <= 1.95 {
                    cov_deg += 1.0;
                    if elem_ids[i] != 0 && elem_ids[j] != 0 {
                        hvy_deg += 1.0;
                    }
                }
            }
            if nn_d < 0.75 {
                clash_cnt += 1.0;
            }
            if nn_d > 2.25 {
                iso_cnt += 1.0;
            }
            sum_nn += nn_d;
            sum_deg += cov_deg;
            if elem_ids[i] != 0 {
                sum_hvy_deg += hvy_deg;
                if hvy_deg > 4.2 {
                    over_coord_cnt += 1.0;
                }
            }
        }
        out[36] = clash_cnt / d_tot;
        out[37] = iso_cnt / d_tot;
        out[38] = (sum_deg / d_tot) / 4.0;
        out[39] = (sum_hvy_deg / d_hvy) / 4.0;
        out[40] = over_coord_cnt / d_hvy;
        out[41] = (sum_nn / d_tot) / 2.0;
    }
    out
}

/// Evaluate Stage-2 `OctreeElementProcessCritic` (`_RBFElementGraphLayer` + 12-ch multipoles) + utility.
pub fn eval_stage2_critic(
    bundle: &ModelBundle,
    s2_h_stage2: &[f32],
    elem_ids: &[usize],
    pos: &[[f32; 3]],
    cloud_42: &[f32; 42],
    targets_norm: &[f32],
    active_mask: &[bool],
    legal_weight: f32,
    constr_weight: f32,
) -> Result<Option<CriticPrediction>, String> {
    let Some(c2) = &bundle.header.stage2_critic else {
        return Ok(None);
    };
    let k_len = elem_ids.len();
    if k_len == 0 {
        return Ok(Some(CriticPrediction {
            mu_norm: vec![0.0; c2.num_targets],
            std_norm: vec![1.0; c2.num_targets],
            logit_legal: -20.0,
            prob_legal: 0.0,
            logit_constr: -20.0,
            prob_constr: 0.0,
            utility: -1e9,
        }));
    }

    let d_node = c2.d_node;
    let n_rbf = c2.n_rbf;
    let n_heads = 4usize;
    let hd = d_node / n_heads;
    let rbf_centers = bundle.f32_slice("c2.rbf_centers")?;
    let elem_emb = bundle.f32_slice("c2.elem_emb")?;

    let mut x_node = vec![0.0f32; k_len * d_node];
    for i in 0..k_len {
        let eid = elem_ids[i].min(c2.num_atom_classes - 1);
        x_node[i * d_node..(i + 1) * d_node].copy_from_slice(&elem_emb[eid * d_node..(eid + 1) * d_node]);
    }

    if k_len > 0 {
        let mut dmat = vec![0.0f32; k_len * k_len];
        let mut rbf_ij = vec![0.0f32; k_len * k_len * n_rbf];
        let mut pair_env = vec![0.0f32; k_len * k_len];
        let r_cut = c2.rbf_max.max(1.0);

        for i in 0..k_len {
            for j in 0..k_len {
                let dx = pos[i][0] - pos[j][0];
                let dy = pos[i][1] - pos[j][1];
                let dz = pos[i][2] - pos[j][2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt().clamp(0.0, 8.0);
                dmat[i * k_len + j] = d;
                let rs = &mut rbf_ij[(i * k_len + j) * n_rbf..(i * k_len + j + 1) * n_rbf];
                for m in 0..n_rbf {
                    let diff = d - rbf_centers[m];
                    rs[m] = (-c2.rbf_gamma * diff * diff).exp();
                }
                if c2.edge_value_gate && i != j && d < r_cut {
                    pair_env[i * k_len + j] = 0.5 * (1.0 + (std::f32::consts::PI * (d / r_cut).clamp(0.0, 1.0)).cos());
                }
            }
        }

        let scale = 1.0 / (hd as f32).sqrt();
        for l in 0..c2.n_gnn_layers {
            let gp = format!("c2.gnn.{l}");
            let mut h = x_node.clone();
            bundle.eval_layernorm_in_place(&format!("{gp}.norm1"), &mut h, k_len, d_node, 1e-5)?;
            let mut qkv = vec![0.0f32; k_len * 3 * d_node];
            bundle.eval_linear(&format!("{gp}.qkv"), &h, k_len, &mut qkv)?;

            let mut ep0 = vec![0.0f32; k_len * k_len * 8];
            bundle.eval_linear(&format!("{gp}.edge_proj.0"), &rbf_ij, k_len * k_len, &mut ep0)?;
            for v in &mut ep0 {
                *v = silu(*v);
            }
            let mut ep2 = vec![0.0f32; k_len * k_len * n_heads];
            bundle.eval_linear(&format!("{gp}.edge_proj.2"), &ep0, k_len * k_len, &mut ep2)?;

            let mut attn_cat = vec![0.0f32; k_len * d_node];
            let mut scores = vec![0.0f32; k_len];
            for head in 0..n_heads {
                for i in 0..k_len {
                    let q_vec = &qkv[i * 3 * d_node + head * hd..i * 3 * d_node + (head + 1) * hd];
                    let mut max_s = f32::NEG_INFINITY;
                    for j in 0..k_len {
                        let k_vec = &qkv[j * 3 * d_node + d_node + head * hd..j * 3 * d_node + d_node + (head + 1) * hd];
                        let mut dot = 0.0f32;
                        for c in 0..hd {
                            dot += q_vec[c] * k_vec[c];
                        }
                        let s = dot * scale + ep2[(i * k_len + j) * n_heads + head];
                        scores[j] = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }
                    let mut sum_e = 0.0f32;
                    for j in 0..k_len {
                        let e = (scores[j] - max_s).exp();
                        scores[j] = e;
                        sum_e += e;
                    }
                    let inv = 1.0 / sum_e.max(1e-12);
                    let out_s = &mut attn_cat[i * d_node + head * hd..i * d_node + (head + 1) * hd];
                    for j in 0..k_len {
                        let w = scores[j] * inv;
                        let v_vec = &qkv[j * 3 * d_node + 2 * d_node + head * hd..j * 3 * d_node + 2 * d_node + (head + 1) * hd];
                        for c in 0..hd {
                            out_s[c] += w * v_vec[c];
                        }
                    }
                }
            }

            let mut upd = vec![0.0f32; k_len * d_node];
            bundle.eval_linear(&format!("{gp}.out_proj"), &attn_cat, k_len, &mut upd)?;

            if c2.edge_value_gate {
                let d_edge = (d_node / 4).max(32);
                let mut ef0 = vec![0.0f32; k_len * k_len * d_edge];
                bundle.eval_linear(&format!("{gp}.edge_filter.0"), &rbf_ij, k_len * k_len, &mut ef0)?;
                for v in &mut ef0 {
                    *v = silu(*v);
                }
                let mut w_ij = vec![0.0f32; k_len * k_len * d_edge];
                bundle.eval_linear(&format!("{gp}.edge_filter.2"), &ef0, k_len * k_len, &mut w_ij)?;
                let mut v_e = vec![0.0f32; k_len * d_edge];
                bundle.eval_linear(&format!("{gp}.v_edge_in"), &h, k_len, &mut v_e)?;

                let mut msg = vec![0.0f32; k_len * d_edge];
                for i in 0..k_len {
                    let mut deg_env = 0.0f32;
                    for j in 0..k_len {
                        let env = pair_env[i * k_len + j];
                        if env > 0.0 {
                            deg_env += env;
                            let ws = &w_ij[(i * k_len + j) * d_edge..(i * k_len + j + 1) * d_edge];
                            let vj = &v_e[j * d_edge..(j + 1) * d_edge];
                            let ms = &mut msg[i * d_edge..(i + 1) * d_edge];
                            for c in 0..d_edge {
                                ms[c] += ws[c] * env * vj[c];
                            }
                        }
                    }
                    let inv_deg = 1.0 / deg_env.max(1.0);
                    for c in 0..d_edge {
                        msg[i * d_edge + c] *= inv_deg;
                    }
                }
                let mut e_upd = vec![0.0f32; k_len * d_node];
                bundle.eval_linear(&format!("{gp}.v_edge_out"), &msg, k_len, &mut e_upd)?;
                for idx in 0..(k_len * d_node) {
                    upd[idx] += e_upd[idx];
                }
            }

            for idx in 0..(k_len * d_node) {
                x_node[idx] += upd[idx];
            }

            let mut hn2 = x_node.clone();
            bundle.eval_layernorm_in_place(&format!("{gp}.norm2"), &mut hn2, k_len, d_node, 1e-5)?;
            let mut ffn_m = vec![0.0f32; k_len * 2 * d_node];
            bundle.eval_linear(&format!("{gp}.ffn.0"), &hn2, k_len, &mut ffn_m)?;
            for v in &mut ffn_m {
                *v = silu(*v);
            }
            let mut ffn_o = vec![0.0f32; k_len * d_node];
            bundle.eval_linear(&format!("{gp}.ffn.3"), &ffn_m, k_len, &mut ffn_o)?;
            for idx in 0..(k_len * d_node) {
                x_node[idx] += ffn_o[idx];
            }
        }
    }

    // Readout: pool_all, pool_hvy, pool_max + optional 12-ch dipoles & quadrupoles
    let mut pool_all = vec![0.0f32; d_node];
    let mut pool_hvy = vec![0.0f32; d_node];
    let mut pool_max = vec![f32::NEG_INFINITY; d_node];
    let mut n_hvy = 0usize;
    for i in 0..k_len {
        let xr = &x_node[i * d_node..(i + 1) * d_node];
        let is_hvy = elem_ids[i] != 0;
        if is_hvy {
            n_hvy += 1;
        }
        for c in 0..d_node {
            pool_all[c] += xr[c];
            if is_hvy {
                pool_hvy[c] += xr[c];
            }
            if xr[c] > pool_max[c] {
                pool_max[c] = xr[c];
            }
        }
    }
    let inv_tot = 1.0 / (k_len.max(1) as f32);
    let inv_hvy = 1.0 / (n_hvy.max(1) as f32);
    for c in 0..d_node {
        pool_all[c] *= inv_tot;
        pool_hvy[c] *= inv_hvy;
    }

    let mut g_pool = Vec::with_capacity(3 * d_node + 24);
    g_pool.extend_from_slice(&pool_all);
    g_pool.extend_from_slice(&pool_hvy);
    g_pool.extend_from_slice(&pool_max);

    if c2.equivariant_multipole {
        let mut q_raw = vec![0.0f32; k_len * 12];
        bundle.eval_linear("c2.multipole_q", &x_node, k_len, &mut q_raw)?;
        let mut q_mean = [0.0f32; 12];
        let mut p_mean = [0.0f32; 3];
        for i in 0..k_len {
            for m in 0..12 {
                q_mean[m] += q_raw[i * 12 + m];
            }
            p_mean[0] += pos[i][0];
            p_mean[1] += pos[i][1];
            p_mean[2] += pos[i][2];
        }
        for m in 0..12 {
            q_mean[m] *= inv_tot;
        }
        p_mean[0] *= inv_tot;
        p_mean[1] *= inv_tot;
        p_mean[2] *= inv_tot;

        let mut dip = [[0.0f32; 3]; 12];
        let mut quad = [0.0f32; 12];
        for i in 0..k_len {
            let rx = pos[i][0] - p_mean[0];
            let ry = pos[i][1] - p_mean[1];
            let rz = pos[i][2] - p_mean[2];
            let r2 = rx * rx + ry * ry + rz * rz;
            for m in 0..12 {
                let qn = q_raw[i * 12 + m] - q_mean[m];
                dip[m][0] += qn * rx;
                dip[m][1] += qn * ry;
                dip[m][2] += qn * rz;
                quad[m] += 0.10 * qn * r2;
            }
        }
        for m in 0..12 {
            let dn = (dip[m][0] * dip[m][0] + dip[m][1] * dip[m][1] + dip[m][2] * dip[m][2] + 1e-8).sqrt();
            g_pool.push(dn);
        }
        for m in 0..12 {
            g_pool.push(quad[m]);
        }
    }

    let hid = c2.hidden_dim;
    let mut h_vec = vec![0.0f32; hid];
    if c2.h_gate_scale.abs() > 1e-8 {
        let d_h = c2.num_h_pools * c2.emb_dim;
        let mut hn = s2_h_stage2[..d_h].to_vec();
        bundle.eval_layernorm_in_place("c2.h_norm", &mut hn, 1, d_h, 1e-5)?;
        bundle.eval_linear("c2.h_proj", &hn, 1, &mut h_vec)?;
        for v in &mut h_vec {
            *v = silu(*v) * c2.h_gate_scale;
        }
    }

    let k_t = c2.num_targets;
    let mut aux_in = Vec::with_capacity(g_pool.len() + 42 + 2 * k_t);
    aux_in.extend_from_slice(&g_pool);
    aux_in.extend_from_slice(cloud_42);
    for k in 0..k_t {
        aux_in.push(targets_norm.get(k).copied().unwrap_or(0.0));
    }
    for k in 0..k_t {
        aux_in.push(if active_mask.get(k).copied().unwrap_or(false) { 1.0 } else { 0.0 });
    }

    let aux_len = aux_in.len();
    bundle.eval_layernorm_in_place("c2.aux_norm", &mut aux_in, 1, aux_len, 1e-5)?;
    let mut a0 = vec![0.0f32; c2.aux_hid];
    bundle.eval_linear("c2.aux_proj.0", &aux_in, 1, &mut a0)?;
    for v in &mut a0 {
        *v = silu(*v);
    }
    let mut aux_vec = vec![0.0f32; c2.aux_hid];
    bundle.eval_linear("c2.aux_proj.3", &a0, 1, &mut aux_vec)?;
    for v in &mut aux_vec {
        *v = silu(*v);
    }

    let mut cat = Vec::with_capacity(hid + c2.aux_hid);
    cat.extend_from_slice(&h_vec);
    cat.extend_from_slice(&aux_vec);

    let mut t0 = vec![0.0f32; hid];
    bundle.eval_linear("c2.trunk.0", &cat, 1, &mut t0)?;
    bundle.eval_layernorm_in_place("c2.trunk.1", &mut t0, 1, hid, 1e-5)?;
    for v in &mut t0 {
        *v = silu(*v);
    }
    let feat_d = hid / 2;
    let mut z = vec![0.0f32; feat_d];
    bundle.eval_linear("c2.trunk.4", &t0, 1, &mut z)?;
    bundle.eval_layernorm_in_place("c2.trunk.5", &mut z, 1, feat_d, 1e-5)?;
    for v in &mut z {
        *v = silu(*v);
    }

    let mut mu_norm = vec![0.0f32; k_t];
    let mut lv_raw = vec![0.0f32; k_t];
    let mut l_leg = [0.0f32; 1];
    let mut l_con = [0.0f32; 1];
    bundle.eval_linear("c2.prop_mean", &z, 1, &mut mu_norm)?;
    bundle.eval_linear("c2.prop_logvar", &z, 1, &mut lv_raw)?;
    bundle.eval_linear("c2.legal", &z, 1, &mut l_leg)?;
    bundle.eval_linear("c2.constr", &z, 1, &mut l_con)?;

    let std_norm: Vec<f32> = lv_raw.iter().map(|&v| (0.5 * v.clamp(-6.0, 3.0)).exp()).collect();
    let logit_legal = l_leg[0];
    let logit_constr = l_con[0];
    let prob_legal = sigmoid(logit_legal);
    let prob_constr = sigmoid(logit_constr);

    let mut err_sum = 0.0f32;
    let mut act_cnt = 0.0f32;
    for k in 0..k_t {
        if active_mask.get(k).copied().unwrap_or(false) {
            let tn = targets_norm.get(k).copied().unwrap_or(0.0);
            err_sum += (mu_norm[k] - tn).abs() + 0.10 * std_norm[k];
            act_cnt += 1.0;
        }
    }
    let mean_err = err_sum / act_cnt.max(1.0);
    let leg_sf = (0.75 - prob_legal).max(0.0);
    let con_sf = (0.60 - prob_constr).max(0.0);
    let mut utility = -mean_err
        - legal_weight * (3.2 * leg_sf + 0.08 * (-logsigmoid(logit_legal)))
        - constr_weight * (2.6 * con_sf + 0.12 * (-logsigmoid(logit_constr)));
    if cloud_42[18] < 0.5 {
        utility -= 25.0;
    }
    if cloud_42[14] * 50.0 < 2.5 {
        utility -= 50.0;
    }

    Ok(Some(CriticPrediction {
        mu_norm,
        std_norm,
        logit_legal,
        prob_legal,
        logit_constr,
        prob_constr,
        utility,
    }))
}
