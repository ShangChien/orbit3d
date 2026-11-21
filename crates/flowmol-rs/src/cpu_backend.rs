use crate::config::{ForwardOutput, PrecomputedModulations, SelfCondState};
use crate::format::{eval_layernorm_cpu, eval_linear_cpu, sigmoid, silu, ModelBundle};

#[derive(Debug, Clone)]
pub struct GraphTopology {
    pub n_nodes: usize,
    pub n_up: usize,
    pub n_edges: usize,
    pub src: Vec<usize>,
    pub dst: Vec<usize>,
    /// Flattened `[n_nodes * (n_nodes - 1)]`: for node `d`, incoming directed edge indices are `dst_incoming[d*(N-1)..(d+1)*(N-1)]`.
    pub dst_incoming: Vec<u32>,
}

impl GraphTopology {
    pub fn complete(n_nodes: usize) -> Self {
        let n_up = n_nodes * n_nodes.saturating_sub(1) / 2;
        let n_edges = 2 * n_up;
        let mut u_all = Vec::with_capacity(n_up);
        let mut v_all = Vec::with_capacity(n_up);
        for u in 0..n_nodes {
            for v in (u + 1)..n_nodes {
                u_all.push(u);
                v_all.push(v);
            }
        }
        let mut src = Vec::with_capacity(n_edges);
        let mut dst = Vec::with_capacity(n_edges);
        src.extend_from_slice(&u_all);
        src.extend_from_slice(&v_all);
        dst.extend_from_slice(&v_all);
        dst.extend_from_slice(&u_all);

        let deg = n_nodes.saturating_sub(1);
        let mut buckets = vec![Vec::with_capacity(deg); n_nodes];
        for (e_idx, &d) in dst.iter().enumerate() {
            buckets[d].push(e_idx as u32);
        }
        let mut dst_incoming = Vec::with_capacity(n_edges);
        for b in buckets {
            debug_assert_eq!(b.len(), deg);
            dst_incoming.extend_from_slice(&b);
        }
        Self {
            n_nodes,
            n_up,
            n_edges,
            src,
            dst,
            dst_incoming,
        }
    }
}

pub fn compute_rbf_1d(dist: f32, d_max: f32, rbf_dim: usize, out: &mut [f32]) {
    let sigma = d_max / (rbf_dim as f32);
    let denom_step = (rbf_dim.saturating_sub(1)).max(1) as f32;
    for m in 0..rbf_dim {
        let mu = (m as f32) / denom_step * d_max;
        let z = (dist - mu) / sigma;
        out[m] = (-z * z).exp();
    }
}

pub fn compute_edge_geometry(
    x: &[[f32; 3]],
    topo: &GraphTopology,
    rbf_dmax: f32,
    rbf_dim: usize,
    unit_diff: &mut [[f32; 3]],
    rbf: &mut [f32],
) {
    for e_idx in 0..topo.n_edges {
        let s = topo.src[e_idx];
        let d = topo.dst[e_idx];
        let dx = x[s][0] - x[d][0];
        let dy = x[s][1] - x[d][1];
        let dz = x[s][2] - x[d][2];
        let sq = (dx * dx + dy * dy + dz * dz).max(1e-8);
        let dist = sq.sqrt();
        let inv = 1.0 / (dist + 1e-8);
        unit_diff[e_idx] = [dx * inv, dy * inv, dz * inv];
        compute_rbf_1d(dist, rbf_dmax, rbf_dim, &mut rbf[e_idx * rbf_dim..(e_idx + 1) * rbf_dim]);
    }
}

pub fn sinusoidal_time_embedding(t: f32, time_emb_dim: usize) -> Vec<f32> {
    let half = time_emb_dim / 2;
    let mut out = Vec::with_capacity(1 + 2 * half);
    out.push(t);
    let log_1000 = 1000.0f32.ln();
    let denom = (half.saturating_sub(1)).max(1) as f32;
    for k in 0..half {
        let freq = ((k as f32) / denom * log_1000).exp() * std::f32::consts::PI;
        out.push((t * freq).sin());
    }
    for k in 0..half {
        let freq = ((k as f32) / denom * log_1000).exp() * std::f32::consts::PI;
        out.push((t * freq).cos());
    }
    out
}

pub fn build_node_and_edge_inputs(
    bundle: &ModelBundle,
    topo: &GraphTopology,
    x_t: &[[f32; 3]],
    a_t: &[usize],
    c_t: &[usize],
    e_t_upper: &[usize],
    t_val: f32,
    sc: Option<&SelfCondState>,
) -> Result<(Vec<f32>, Vec<f32>, Vec<[f32; 3]>), String> {
    let cfg = &bundle.header.config;
    let n = topo.n_nodes;
    let a_num = cfg.n_atom_types();
    let c_num = cfg.n_charges;
    let b_num = cfg.n_bond_types;
    let v_ch = cfg.n_vec_channels;
    let rbf_dim = cfg.rbf_dim;

    let ptable = bundle.f32_slice("periodic_table_feats")?; // (a_num + 1, 34)
    let sc_vec_scale = bundle.f32_slice_opt("sc_vec_scale").map_or(0.0, |s| s[0]);
    let t_emb = sinusoidal_time_embedding(t_val, cfg.time_emb_dim);

    let node_dim = cfg.node_in_dim();
    let mut node_in = vec![0.0f32; n * node_dim];
    let mut v_init = vec![[0.0f32; 3]; n * v_ch];

    for i in 0..n {
        let row = &mut node_in[i * node_dim..(i + 1) * node_dim];
        let mut off = 0;
        let ai = a_t[i].min(a_num);
        row[off + ai] = 1.0;
        off += a_num + 1;

        let ci = c_t[i].min(c_num);
        row[off + ci] = 1.0;
        off += c_num + 1;

        row[off..off + t_emb.len()].copy_from_slice(&t_emb);
        off += t_emb.len();

        if cfg.use_periodic_features {
            row[off..off + 34].copy_from_slice(&ptable[ai * 34..(ai + 1) * 34]);
            off += 34;
        }

        if cfg.self_conditioning {
            if let Some(sc_s) = sc {
                let ap = &sc_s.a_probs[i * a_num..(i + 1) * a_num];
                let cp = &sc_s.c_probs[i * c_num..(i + 1) * c_num];
                row[off..off + a_num].copy_from_slice(ap);
                off += a_num;
                row[off..off + c_num].copy_from_slice(cp);
                off += c_num;
                if cfg.use_periodic_features {
                    for d in 0..34 {
                        let mut acc = 0.0f32;
                        for z in 0..a_num {
                            acc += ap[z] * ptable[z * 34 + d];
                        }
                        row[off + d] = acc;
                    }
                    off += 34;
                }
                v_init[i * v_ch] = [
                    sc_vec_scale * (sc_s.x[i][0] - x_t[i][0]),
                    sc_vec_scale * (sc_s.x[i][1] - x_t[i][1]),
                    sc_vec_scale * (sc_s.x[i][2] - x_t[i][2]),
                ];
            } else {
                off += a_num + c_num + if cfg.use_periodic_features { 34 } else { 0 };
            }
        }
        debug_assert_eq!(off, node_dim);
    }

    let edge_dim = cfg.edge_in_dim();
    let mut edge_in = vec![0.0f32; topo.n_edges * edge_dim];
    let mut sc_rbf_buf = vec![0.0f32; rbf_dim];

    for e_idx in 0..topo.n_edges {
        let k_up = e_idx % topo.n_up;
        let row = &mut edge_in[e_idx * edge_dim..(e_idx + 1) * edge_dim];
        let mut off = 0;
        let bi = e_t_upper[k_up].min(b_num);
        row[off + bi] = 1.0;
        off += b_num + 1;

        if cfg.self_conditioning {
            if let Some(sc_s) = sc {
                let ep = &sc_s.e_probs[k_up * b_num..(k_up + 1) * b_num];
                row[off..off + b_num].copy_from_slice(ep);
                off += b_num;
                let s = topo.src[e_idx];
                let d = topo.dst[e_idx];
                let dx = sc_s.x[s][0] - sc_s.x[d][0];
                let dy = sc_s.x[s][1] - sc_s.x[d][1];
                let dz = sc_s.x[s][2] - sc_s.x[d][2];
                let dist = (dx * dx + dy * dy + dz * dz).max(1e-8).sqrt();
                compute_rbf_1d(dist, cfg.rbf_dmax, rbf_dim, &mut sc_rbf_buf);
                row[off..off + rbf_dim].copy_from_slice(&sc_rbf_buf);
                off += rbf_dim;
            } else {
                off += b_num + rbf_dim;
            }
        }
        debug_assert_eq!(off, edge_dim);
    }

    Ok((node_in, edge_in, v_init))
}

/// Precompute `LoomConditioner` `(1 + scale, shift)` modulations on CPU (< 0.1 ms, once per trajectory).
pub fn precompute_condition_modulations(
    bundle: &ModelBundle,
    cond_values: &[f32],
    cond_active: &[bool],
) -> Result<Option<PrecomputedModulations>, String> {
    let Some(cc) = &bundle.header.conditioner else {
        return Ok(None);
    };
    if cond_values.len() != cc.num_props || cond_active.len() != cc.num_props {
        return Err(format!(
            "Expected {} condition values/masks, got {}/{}",
            cc.num_props,
            cond_values.len(),
            cond_active.len()
        ));
    }
    let has_any = cond_active.iter().any(|&b| b);
    let rbf_centers = bundle.f32_slice("cond.rbf_centers")?;
    let fourier_freqs = bundle.f32_slice("cond.fourier_freqs")?;
    let prop_id_emb = bundle.f32_slice("cond.prop_id_emb")?;
    let inactive_emb = bundle.f32_slice("cond.inactive_prop_emb")?;

    let in_dim = 2 + cc.num_rbf + 2 * cc.num_freqs;
    let d = cc.emb_dim;
    let mut fused_in = vec![0.0f32; cc.num_props * d];

    for k in 0..cc.num_props {
        let act = cond_active[k];
        if !act {
            fused_in[k * d..(k + 1) * d].copy_from_slice(&inactive_emb[k * d..(k + 1) * d]);
            continue;
        }
        let v = cond_values[k];
        let mut feat = Vec::with_capacity(in_dim);
        feat.push(v);
        for &c in rbf_centers {
            let diff = v - c;
            feat.push((-cc.rbf_gamma * diff * diff).exp());
        }
        for &f in fourier_freqs {
            feat.push((v * f).sin());
        }
        for &f in fourier_freqs {
            feat.push((v * f).cos());
        }
        feat.push(1.0);

        let mut h1 = vec![0.0f32; d];
        eval_linear_cpu(&bundle.linear_view(&format!("cond.prop_enc.{k}.0"))?, &feat, 1, &mut h1);
        eval_layernorm_cpu(
            &mut h1,
            1,
            d,
            bundle.f32_slice(&format!("cond.prop_enc.{k}.1.weight"))?,
            bundle.f32_slice(&format!("cond.prop_enc.{k}.1.bias"))?,
            1e-5,
        );
        for x in &mut h1 {
            *x = silu(*x);
        }
        let mut h2 = vec![0.0f32; d];
        eval_linear_cpu(&bundle.linear_view(&format!("cond.prop_enc.{k}.3"))?, &h1, 1, &mut h2);
        eval_layernorm_cpu(
            &mut h2,
            1,
            d,
            bundle.f32_slice(&format!("cond.prop_enc.{k}.4.weight"))?,
            bundle.f32_slice(&format!("cond.prop_enc.{k}.4.bias"))?,
            1e-5,
        );
        for i in 0..d {
            fused_in[k * d + i] = silu(h2[i]) + prop_id_emb[k * d + i];
        }
    }

    let mut gm1 = vec![0.0f32; 2 * d];
    eval_linear_cpu(&bundle.linear_view("cond.global_mlp.0")?, &fused_in, 1, &mut gm1);
    eval_layernorm_cpu(
        &mut gm1,
        1,
        2 * d,
        bundle.f32_slice("cond.global_mlp.1.weight")?,
        bundle.f32_slice("cond.global_mlp.1.bias")?,
        1e-5,
    );
    for x in &mut gm1 {
        *x = silu(*x);
    }
    let mut global_c = vec![0.0f32; d];
    eval_linear_cpu(&bundle.linear_view("cond.global_mlp.3")?, &gm1, 1, &mut global_c);
    eval_layernorm_cpu(
        &mut global_c,
        1,
        d,
        bundle.f32_slice("cond.global_mlp.4.weight")?,
        bundle.f32_slice("cond.global_mlp.4.bias")?,
        1e-5,
    );
    for x in &mut global_c {
        *x = if has_any { silu(*x) } else { 0.0 };
    }

    let silu_c: Vec<f32> = global_c.iter().map(|&x| silu(x)).collect();
    let eval_mod = |prefix: &str| -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut raw = vec![0.0f32; 2 * d];
        eval_linear_cpu(&bundle.linear_view(prefix)?, &silu_c, 1, &mut raw);
        if cc.modulation_cap > 0.0 {
            for x in &mut raw {
                *x = cc.modulation_cap * (*x / cc.modulation_cap).tanh();
            }
        }
        let scale: Vec<f32> = raw[..d].iter().map(|&s| 1.0 + s).collect();
        let shift: Vec<f32> = raw[d..].to_vec();
        Ok((scale, shift))
    };

    let mut layer_scale = Vec::with_capacity(cc.num_layers);
    let mut layer_shift = Vec::with_capacity(cc.num_layers);
    for l in 0..cc.num_layers {
        let (sc, sh) = eval_mod(&format!("cond.layer_mod.{l}"))?;
        layer_scale.push(sc);
        layer_shift.push(sh);
    }
    let (out_scale, out_shift) = eval_mod("cond.out_mod")?;

    Ok(Some(PrecomputedModulations {
        layer_scale,
        layer_shift,
        out_scale,
        out_shift,
    }))
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm3_clamped(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).max(1e-8).sqrt()
}

fn eval_standard_gvp_cpu(
    bundle: &ModelBundle,
    prefix: &str,
    rows: usize,
    v_in: usize,
    v_out: usize,
    f_in: usize,
    _f_out: usize,
    n_cp: usize,
    identity_vec_act: bool,
    scalars_in: &[f32],
    vecs_in: &[[f32; 3]],
    scalars_out: &mut [f32],
    vecs_out: &mut [[f32; 3]],
) -> Result<(), String> {
    let dim_h = v_in.max(v_out);
    let wu_in = dim_h + n_cp;
    let wh = bundle.f32_slice(&format!("{prefix}.Wh"))?; // (v_in, dim_h)
    let wcp = if n_cp > 0 {
        Some(bundle.f32_slice(&format!("{prefix}.Wcp"))?) // (v_in, 2*n_cp)
    } else {
        None
    };
    let wu = bundle.f32_slice(&format!("{prefix}.Wu"))?; // (wu_in, v_out)

    let mut s_cat = vec![0.0f32; rows * (f_in + wu_in)];
    let mut vu_all = vec![[0.0f32; 3]; rows * v_out];
    let mut vh_row = vec![[0.0f32; 3]; wu_in];
    let mut cp_tmp = vec![[0.0f32; 3]; 2 * n_cp];

    for r in 0..rows {
        let vr = &vecs_in[r * v_in..(r + 1) * v_in];
        for h in 0..dim_h {
            let mut acc = [0.0f32; 3];
            for vi in 0..v_in {
                let w = wh[vi * dim_h + h];
                acc[0] += vr[vi][0] * w;
                acc[1] += vr[vi][1] * w;
                acc[2] += vr[vi][2] * w;
            }
            vh_row[h] = acc;
        }
        if let Some(wcp_s) = wcp {
            for p in 0..(2 * n_cp) {
                let mut acc = [0.0f32; 3];
                for vi in 0..v_in {
                    let w = wcp_s[vi * (2 * n_cp) + p];
                    acc[0] += vr[vi][0] * w;
                    acc[1] += vr[vi][1] * w;
                    acc[2] += vr[vi][2] * w;
                }
                cp_tmp[p] = acc;
            }
            for p in 0..n_cp {
                vh_row[dim_h + p] = cross3(cp_tmp[p], cp_tmp[n_cp + p]);
            }
        }
        let sc_dst = &mut s_cat[r * (f_in + wu_in)..(r + 1) * (f_in + wu_in)];
        sc_dst[..f_in].copy_from_slice(&scalars_in[r * f_in..(r + 1) * f_in]);
        for h in 0..wu_in {
            sc_dst[f_in + h] = norm3_clamped(vh_row[h]);
        }
        for uo in 0..v_out {
            let mut acc = [0.0f32; 3];
            for h in 0..wu_in {
                let w = wu[h * v_out + uo];
                acc[0] += vh_row[h][0] * w;
                acc[1] += vh_row[h][1] * w;
                acc[2] += vh_row[h][2] * w;
            }
            vu_all[r * v_out + uo] = acc;
        }
    }

    eval_linear_cpu(&bundle.linear_view(&format!("{prefix}.to_feats_out"))?, &s_cat, rows, scalars_out);
    for x in scalars_out.iter_mut() {
        *x = silu(*x);
    }

    let mut gates = vec![0.0f32; rows * v_out];
    eval_linear_cpu(&bundle.linear_view(&format!("{prefix}.vec_gate"))?, scalars_out, rows, &mut gates);
    for i in 0..(rows * v_out) {
        let g = if identity_vec_act { gates[i] } else { sigmoid(gates[i]) };
        vecs_out[i] = [vu_all[i][0] * g, vu_all[i][1] * g, vu_all[i][2] * g];
    }
    Ok(())
}

fn eval_gvp_layernorm_cpu(
    h: &mut [f32],
    v: &mut [[f32; 3]],
    rows: usize,
    s_dim: usize,
    v_dim: usize,
    ln_w: &[f32],
    ln_b: &[f32],
) {
    eval_layernorm_cpu(h, rows, s_dim, ln_w, ln_b, 1e-5);
    let inv_v = 1.0 / (v_dim as f32);
    for r in 0..rows {
        let vr = &mut v[r * v_dim..(r + 1) * v_dim];
        let mut mean_vn = 0.0f32;
        for vec3 in vr.iter() {
            let sq = (vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2]).max(1e-8);
            mean_vn += sq;
        }
        mean_vn *= inv_v;
        let denom = (mean_vn + 1e-5).sqrt() + 1e-5;
        let inv_d = 1.0 / denom;
        for vec3 in vr.iter_mut() {
            vec3[0] *= inv_d;
            vec3[1] *= inv_d;
            vec3[2] *= inv_d;
        }
    }
}

pub fn forward_cpu(
    bundle: &ModelBundle,
    topo: &GraphTopology,
    x_t: &[[f32; 3]],
    a_t: &[usize],
    c_t: &[usize],
    e_t_upper: &[usize],
    t_val: f32,
    sc: Option<&SelfCondState>,
    mods: Option<&PrecomputedModulations>,
    node_cond_flags: Option<&[[f32; 2]]>,
) -> Result<ForwardOutput, String> {
    let cfg = &bundle.header.config;
    let n = topo.n_nodes;
    let n_edges = topo.n_edges;
    let s_dim = cfg.n_hidden_scalars;
    let e_dim = cfg.n_hidden_edge_feats;
    let v_dim = cfg.n_vec_channels;
    let n_cp = cfg.n_cp_feats;
    let rbf_dim = cfg.rbf_dim;

    let (node_in, edge_in, mut v) =
        build_node_and_edge_inputs(bundle, topo, x_t, a_t, c_t, e_t_upper, t_val, sc)?;

    // Scalar embedding
    let mut h_tmp = vec![0.0f32; n * s_dim];
    eval_linear_cpu(&bundle.linear_view("scalar_embedding.0")?, &node_in, n, &mut h_tmp);
    for x in &mut h_tmp {
        *x = silu(*x);
    }
    let mut h = vec![0.0f32; n * s_dim];
    eval_linear_cpu(&bundle.linear_view("scalar_embedding.2")?, &h_tmp, n, &mut h);
    for x in &mut h {
        *x = silu(*x);
    }
    eval_layernorm_cpu(
        &mut h,
        n,
        s_dim,
        bundle.f32_slice("scalar_embedding.4.weight")?,
        bundle.f32_slice("scalar_embedding.4.bias")?,
        1e-5,
    );
    if let Some(flags) = node_cond_flags {
        let flat: Vec<f32> = flags.iter().flat_map(|f| [f[0], f[1]]).collect();
        eval_linear_cpu(&bundle.linear_view("fragment_cond_proj")?, &flat, n, &mut h_tmp);
        for i in 0..(n * s_dim) {
            h[i] += h_tmp[i];
        }
    }

    // Edge embedding
    let mut e_tmp = vec![0.0f32; n_edges * e_dim];
    eval_linear_cpu(&bundle.linear_view("edge_embedding.0")?, &edge_in, n_edges, &mut e_tmp);
    for x in &mut e_tmp {
        *x = silu(*x);
    }
    let mut e = vec![0.0f32; n_edges * e_dim];
    eval_linear_cpu(&bundle.linear_view("edge_embedding.2")?, &e_tmp, n_edges, &mut e);
    for x in &mut e {
        *x = silu(*x);
    }
    eval_layernorm_cpu(
        &mut e,
        n_edges,
        e_dim,
        bundle.f32_slice("edge_embedding.4.weight")?,
        bundle.f32_slice("edge_embedding.4.bias")?,
        1e-5,
    );

    let mut x = x_t.to_vec();
    let mut unit_diff = vec![[0.0f32; 3]; n_edges];
    let mut rbf = vec![0.0f32; n_edges * rbf_dim];
    compute_edge_geometry(&x, topo, cfg.rbf_dmax, rbf_dim, &mut unit_diff, &mut rbf);

    // Reusable work buffers
    let v_in0 = 1 + v_dim + if cfg.use_dst_feats { v_dim } else { 0 };
    let dim_h0 = v_in0.max(v_dim);
    let wu0 = dim_h0 + n_cp;
    let mut s_msg_a = vec![0.0f32; n_edges * s_dim];
    let mut s_msg_b = vec![0.0f32; n_edges * s_dim];
    let mut v_msg_a = vec![[0.0f32; 3]; n_edges * v_dim];
    let mut v_msg_b = vec![[0.0f32; 3]; n_edges * v_dim];
    let mut node_s_a = vec![0.0f32; n * s_dim];
    let mut node_s_b = vec![0.0f32; n * s_dim];
    let mut node_v_a = vec![[0.0f32; 3]; n * v_dim];
    let mut node_v_b = vec![[0.0f32; 3]; n * v_dim];
    let msg_norm_inv = 1.0 / cfg.message_norm_factor(n);

    for blk_idx in 0..cfg.n_molecule_updates {
        let bp = format!("blocks.{blk_idx}");
        if let Some(m) = mods {
            let sc_l = &m.layer_scale[blk_idx];
            let sh_l = &m.layer_shift[blk_idx];
            for i in 0..n {
                for c in 0..s_dim {
                    h[i * s_dim + c] = h[i * s_dim + c] * sc_l[c] + sh_l[c];
                }
            }
        }

        for conv_idx in 0..cfg.convs_per_update {
            let cp = format!("{bp}.convs.{conv_idx}");
            // 1. Node-preprojected edge_message[0]
            let wh_unit = bundle.f32_slice(&format!("{cp}.em0.Wh_unit"))?; // (1, dim_h0)
            let wh_src = bundle.f32_slice(&format!("{cp}.em0.Wh_src"))?; // (v_dim, dim_h0)
            let wcp_unit = bundle.f32_slice_opt(&format!("{cp}.em0.Wcp_unit")); // (1, 2*n_cp)
            let wcp_src = bundle.f32_slice_opt(&format!("{cp}.em0.Wcp_src")); // (v_dim, 2*n_cp)
            let wu_0 = bundle.f32_slice(&format!("{cp}.em0.Wu"))?; // (wu0, v_dim)

            // Pre-project node vectors & scalars in O(N) instead of O(N^2)!
            let mut n_vh = vec![[0.0f32; 3]; n * dim_h0];
            let mut n_vcp = vec![[0.0f32; 3]; n * 2 * n_cp];
            for i in 0..n {
                let vi = &v[i * v_dim..(i + 1) * v_dim];
                for hd in 0..dim_h0 {
                    let mut acc = [0.0f32; 3];
                    for vc in 0..v_dim {
                        let w = wh_src[vc * dim_h0 + hd];
                        acc[0] += vi[vc][0] * w;
                        acc[1] += vi[vc][1] * w;
                        acc[2] += vi[vc][2] * w;
                    }
                    n_vh[i * dim_h0 + hd] = acc;
                }
                if let Some(wcp_s) = wcp_src {
                    for p in 0..(2 * n_cp) {
                        let mut acc = [0.0f32; 3];
                        for vc in 0..v_dim {
                            let w = wcp_s[vc * (2 * n_cp) + p];
                            acc[0] += vi[vc][0] * w;
                            acc[1] += vi[vc][1] * w;
                            acc[2] += vi[vc][2] * w;
                        }
                        n_vcp[i * 2 * n_cp + p] = acc;
                    }
                }
            }
            let mut p_s_src = vec![0.0f32; n * s_dim];
            eval_linear_cpu(&bundle.linear_view(&format!("{cp}.em0.s_src"))?, &h, n, &mut p_s_src);

            let edge_sh_cols = rbf_dim + e_dim + wu0;
            let mut edge_sh_in = vec![0.0f32; n_edges * edge_sh_cols];
            let mut vu0_all = vec![[0.0f32; 3]; n_edges * v_dim];
            let mut vh_full = vec![[0.0f32; 3]; wu0];
            let mut cp_pair = vec![[0.0f32; 3]; 2 * n_cp];

            for e_i in 0..n_edges {
                let s = topo.src[e_i];
                let ud = unit_diff[e_i];
                for hd in 0..dim_h0 {
                    let nv = n_vh[s * dim_h0 + hd];
                    let wu_w = wh_unit[hd];
                    vh_full[hd] = [nv[0] + ud[0] * wu_w, nv[1] + ud[1] * wu_w, nv[2] + ud[2] * wu_w];
                }
                if let Some(wcp_u) = wcp_unit {
                    for p in 0..(2 * n_cp) {
                        let nc = n_vcp[s * 2 * n_cp + p];
                        let w = wcp_u[p];
                        cp_pair[p] = [nc[0] + ud[0] * w, nc[1] + ud[1] * w, nc[2] + ud[2] * w];
                    }
                    for p in 0..n_cp {
                        vh_full[dim_h0 + p] = cross3(cp_pair[p], cp_pair[n_cp + p]);
                    }
                }
                let row = &mut edge_sh_in[e_i * edge_sh_cols..(e_i + 1) * edge_sh_cols];
                row[..rbf_dim].copy_from_slice(&rbf[e_i * rbf_dim..(e_i + 1) * rbf_dim]);
                row[rbf_dim..rbf_dim + e_dim].copy_from_slice(&e[e_i * e_dim..(e_i + 1) * e_dim]);
                for hd in 0..wu0 {
                    row[rbf_dim + e_dim + hd] = norm3_clamped(vh_full[hd]);
                }
                for uo in 0..v_dim {
                    let mut acc = [0.0f32; 3];
                    for hd in 0..wu0 {
                        let w = wu_0[hd * v_dim + uo];
                        acc[0] += vh_full[hd][0] * w;
                        acc[1] += vh_full[hd][1] * w;
                        acc[2] += vh_full[hd][2] * w;
                    }
                    vu0_all[e_i * v_dim + uo] = acc;
                }
            }

            eval_linear_cpu(
                &bundle.linear_view(&format!("{cp}.em0.s_edge_sh"))?,
                &edge_sh_in,
                n_edges,
                &mut s_msg_a,
            );
            for e_i in 0..n_edges {
                let s = topo.src[e_i];
                for c in 0..s_dim {
                    s_msg_a[e_i * s_dim + c] = silu(s_msg_a[e_i * s_dim + c] + p_s_src[s * s_dim + c]);
                }
            }
            let mut gates0 = vec![0.0f32; n_edges * v_dim];
            eval_linear_cpu(&bundle.linear_view(&format!("{cp}.em0.vec_gate"))?, &s_msg_a, n_edges, &mut gates0);
            for idx in 0..(n_edges * v_dim) {
                let g = sigmoid(gates0[idx]);
                v_msg_a[idx] = [vu0_all[idx][0] * g, vu0_all[idx][1] * g, vu0_all[idx][2] * g];
            }

            // Remaining edge_message GVPs
            for m_idx in 1..cfg.n_message_gvps {
                eval_standard_gvp_cpu(
                    bundle,
                    &format!("{cp}.em.{m_idx}"),
                    n_edges,
                    v_dim,
                    v_dim,
                    s_dim,
                    s_dim,
                    n_cp,
                    false,
                    &s_msg_a,
                    &v_msg_a,
                    &mut s_msg_b,
                    &mut v_msg_b,
                )?;
                std::mem::swap(&mut s_msg_a, &mut s_msg_b);
                std::mem::swap(&mut v_msg_a, &mut v_msg_b);
            }

            // Incoming-edge aggregation + residual + GVPLayerNorm
            let deg = n.saturating_sub(1);
            for d in 0..n {
                let inc = &topo.dst_incoming[d * deg..(d + 1) * deg];
                for &e_u32 in inc {
                    let e_i = e_u32 as usize;
                    for c in 0..s_dim {
                        h[d * s_dim + c] += s_msg_a[e_i * s_dim + c] * msg_norm_inv;
                    }
                    for vc in 0..v_dim {
                        let vm = v_msg_a[e_i * v_dim + vc];
                        v[d * v_dim + vc][0] += vm[0] * msg_norm_inv;
                        v[d * v_dim + vc][1] += vm[1] * msg_norm_inv;
                        v[d * v_dim + vc][2] += vm[2] * msg_norm_inv;
                    }
                }
            }
            eval_gvp_layernorm_cpu(
                &mut h,
                &mut v,
                n,
                s_dim,
                v_dim,
                bundle.f32_slice(&format!("{cp}.msg_ln.weight"))?,
                bundle.f32_slice(&format!("{cp}.msg_ln.bias"))?,
            );

            // Node update GVPs
            node_s_a.copy_from_slice(&h);
            node_v_a.copy_from_slice(&v);
            for u_idx in 0..cfg.n_update_gvps {
                eval_standard_gvp_cpu(
                    bundle,
                    &format!("{cp}.nu.{u_idx}"),
                    n,
                    v_dim,
                    v_dim,
                    s_dim,
                    s_dim,
                    n_cp,
                    false,
                    &node_s_a,
                    &node_v_a,
                    &mut node_s_b,
                    &mut node_v_b,
                )?;
                std::mem::swap(&mut node_s_a, &mut node_s_b);
                std::mem::swap(&mut node_v_a, &mut node_v_b);
            }
            for i in 0..(n * s_dim) {
                h[i] += node_s_a[i];
            }
            for i in 0..(n * v_dim) {
                v[i][0] += node_v_a[i][0];
                v[i][1] += node_v_a[i][1];
                v[i][2] += node_v_a[i][2];
            }
            eval_gvp_layernorm_cpu(
                &mut h,
                &mut v,
                n,
                s_dim,
                v_dim,
                bundle.f32_slice(&format!("{cp}.upd_ln.weight"))?,
                bundle.f32_slice(&format!("{cp}.upd_ln.bias"))?,
            );
        }

        // Position updater
        node_s_a.copy_from_slice(&h);
        node_v_a.copy_from_slice(&v);
        for p_idx in 0..cfg.n_pos_update_gvps {
            let is_last = p_idx + 1 == cfg.n_pos_update_gvps;
            let v_out = if is_last { 1 } else { v_dim };
            eval_standard_gvp_cpu(
                bundle,
                &format!("{bp}.pu.{p_idx}"),
                n,
                v_dim,
                v_out,
                s_dim,
                s_dim,
                n_cp,
                is_last,
                &node_s_a,
                &node_v_a,
                &mut node_s_b,
                &mut node_v_b[..n * v_out],
            )?;
            if is_last {
                for i in 0..n {
                    x[i][0] += node_v_b[i][0];
                    x[i][1] += node_v_b[i][1];
                    x[i][2] += node_v_b[i][2];
                }
            } else {
                std::mem::swap(&mut node_s_a, &mut node_s_b);
                std::mem::swap(&mut node_v_a, &mut node_v_b);
            }
        }

        compute_edge_geometry(&x, topo, cfg.rbf_dmax, rbf_dim, &mut unit_diff, &mut rbf);

        // Edge updater (Node-Before-Gather pre-split!)
        let mut p_eu_src = vec![0.0f32; n * e_dim];
        let mut p_eu_dst = vec![0.0f32; n * e_dim];
        eval_linear_cpu(&bundle.linear_view(&format!("{bp}.eu.w_src"))?, &h, n, &mut p_eu_src);
        eval_linear_cpu(&bundle.linear_view(&format!("{bp}.eu.w_dst"))?, &h, n, &mut p_eu_dst);

        let er_cols = e_dim + rbf_dim;
        let mut er_in = vec![0.0f32; n_edges * er_cols];
        for e_i in 0..n_edges {
            er_in[e_i * er_cols..e_i * er_cols + e_dim].copy_from_slice(&e[e_i * e_dim..(e_i + 1) * e_dim]);
            er_in[e_i * er_cols + e_dim..(e_i + 1) * er_cols]
                .copy_from_slice(&rbf[e_i * rbf_dim..(e_i + 1) * rbf_dim]);
        }
        let mut eu_h = vec![0.0f32; n_edges * e_dim];
        eval_linear_cpu(&bundle.linear_view(&format!("{bp}.eu.w_er"))?, &er_in, n_edges, &mut eu_h);
        for e_i in 0..n_edges {
            let s = topo.src[e_i];
            let d = topo.dst[e_i];
            for c in 0..e_dim {
                eu_h[e_i * e_dim + c] =
                    silu(eu_h[e_i * e_dim + c] + p_eu_src[s * e_dim + c] + p_eu_dst[d * e_dim + c]);
            }
        }
        let mut eu_out = vec![0.0f32; n_edges * e_dim];
        eval_linear_cpu(&bundle.linear_view(&format!("{bp}.eu.lin1"))?, &eu_h, n_edges, &mut eu_out);
        for idx in 0..(n_edges * e_dim) {
            e[idx] += silu(eu_out[idx]);
        }
        eval_layernorm_cpu(
            &mut e,
            n_edges,
            e_dim,
            bundle.f32_slice(&format!("{bp}.eu.ln.weight"))?,
            bundle.f32_slice(&format!("{bp}.eu.ln.bias"))?,
            1e-5,
        );
    }

    if let Some(m) = mods {
        for i in 0..n {
            for c in 0..s_dim {
                h[i * s_dim + c] = h[i * s_dim + c] * m.out_scale[c] + m.out_shift[c];
            }
        }
    }

    // Output heads
    let a_num = cfg.n_atom_types();
    let c_num = cfg.n_charges;
    let b_num = cfg.n_bond_types;
    eval_linear_cpu(&bundle.linear_view("node_output_head.0")?, &h, n, &mut node_s_a);
    for v_s in &mut node_s_a {
        *v_s = silu(*v_s);
    }
    let mut node_logits = vec![0.0f32; n * (a_num + c_num)];
    eval_linear_cpu(&bundle.linear_view("node_output_head.2")?, &node_s_a, n, &mut node_logits);

    let mut a_logits = vec![0.0f32; n * a_num];
    let mut c_logits = vec![0.0f32; n * c_num];
    for i in 0..n {
        a_logits[i * a_num..(i + 1) * a_num].copy_from_slice(&node_logits[i * (a_num + c_num)..i * (a_num + c_num) + a_num]);
        c_logits[i * c_num..(i + 1) * c_num]
            .copy_from_slice(&node_logits[i * (a_num + c_num) + a_num..(i + 1) * (a_num + c_num)]);
    }

    if bundle.header.tensors.contains_key("periodic_logit_proj.weight") {
        let ptable = bundle.f32_slice("periodic_table_feats")?;
        let mut p_lat = vec![0.0f32; n * 34];
        eval_linear_cpu(&bundle.linear_view("periodic_logit_proj")?, &h, n, &mut p_lat);
        for i in 0..n {
            let pl = &p_lat[i * 34..(i + 1) * 34];
            for z in 0..a_num {
                let pt_row = &ptable[z * 34..(z + 1) * 34];
                let mut dot = 0.0f32;
                for d in 0..34 {
                    dot += pl[d] * pt_row[d];
                }
                a_logits[i * a_num + z] += dot;
            }
        }
    }

    let mut e_sym = vec![0.0f32; topo.n_up * e_dim];
    for k in 0..topo.n_up {
        let lo = topo.n_up + k;
        for c in 0..e_dim {
            e_sym[k * e_dim + c] = e[k * e_dim + c] + e[lo * e_dim + c];
        }
    }
    let mut e_h1 = vec![0.0f32; topo.n_up * e_dim];
    eval_linear_cpu(&bundle.linear_view("to_edge_logits.0")?, &e_sym, topo.n_up, &mut e_h1);
    for v_e in &mut e_h1 {
        *v_e = silu(*v_e);
    }
    let mut e_logits = vec![0.0f32; topo.n_up * b_num];
    eval_linear_cpu(&bundle.linear_view("to_edge_logits.2")?, &e_h1, topo.n_up, &mut e_logits);

    remove_com_in_place(&mut x);

    Ok(ForwardOutput {
        x,
        a_logits,
        c_logits,
        e_logits,
    })
}

pub fn remove_com_in_place(x: &mut [[f32; 3]]) {
    if x.is_empty() {
        return;
    }
    let mut mean = [0.0f32; 3];
    for p in x.iter() {
        mean[0] += p[0];
        mean[1] += p[1];
        mean[2] += p[2];
    }
    let inv = 1.0 / (x.len() as f32);
    mean[0] *= inv;
    mean[1] *= inv;
    mean[2] *= inv;
    for p in x.iter_mut() {
        p[0] -= mean[0];
        p[1] -= mean[1];
        p[2] -= mean[2];
    }
}
