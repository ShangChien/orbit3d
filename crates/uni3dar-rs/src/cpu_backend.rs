use crate::config::PrecomputedCond;
use crate::format::{gelu_exact, silu, ModelBundle};

/// Precompute `LoomConditioner` modulations, root residual, and causal prefix tokens once per prompt.
pub fn precompute_condition(
    bundle: &ModelBundle,
    raw_targets: Option<&[f32]>,
    active_mask: Option<&[bool]>,
) -> Result<PrecomputedCond, String> {
    let bb = &bundle.header.backbone;
    let d = bb.emb_dim;
    let n_layers = bb.num_layers;
    let k_props = bundle.header.target_cols.len();

    let mut t_norm = vec![0.0f32; k_props];
    let mut act = vec![false; k_props];
    if let (Some(rt), Some(am)) = (raw_targets, active_mask) {
        for k in 0..k_props.min(rt.len()).min(am.len()) {
            act[k] = am[k];
            if am[k] {
                let m = bundle.header.target_mean.get(k).copied().unwrap_or(0.0);
                let s = bundle.header.target_scale.get(k).copied().unwrap_or(1.0).max(1e-6);
                t_norm[k] = (rt[k] - m) / s;
            }
        }
    }
    let has_any = act.iter().any(|&b| b);

    let Some(cc) = &bundle.header.conditioner else {
        return Ok(PrecomputedCond {
            targets_norm: t_norm,
            active_mask: act,
            has_any: false,
            root_delta: vec![0.0; d],
            prefix_tokens: Vec::new(),
            num_prefix: 0,
            layer_scale: vec![vec![1.0; d]; n_layers],
            layer_shift: vec![vec![0.0; d]; n_layers],
            out_scale: vec![1.0; d],
            out_shift: vec![0.0; d],
        });
    };

    let rbf_centers = bundle.f32_slice("cond.rbf_centers")?;
    let fourier_freqs = bundle.f32_slice("cond.fourier_freqs")?;
    let prop_id_emb = bundle.f32_slice("cond.prop_id_emb")?;
    let inactive_emb = bundle.f32_slice("cond.inactive_prop_emb")?;

    let in_dim = 2 + cc.num_rbf + 2 * cc.num_freqs;
    let mut per_prop_cat = vec![0.0f32; cc.num_props * d];

    for k in 0..cc.num_props {
        let is_act = act.get(k).copied().unwrap_or(false);
        if !is_act {
            per_prop_cat[k * d..(k + 1) * d].copy_from_slice(&inactive_emb[k * d..(k + 1) * d]);
            continue;
        }
        let v = t_norm.get(k).copied().unwrap_or(0.0);
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
        feat.push(1.0); // active flag last!
        debug_assert_eq!(feat.len(), in_dim);

        let mut h0 = vec![0.0f32; d];
        bundle.eval_linear(&format!("cond.prop_enc.{k}.0"), &feat, 1, &mut h0)?;
        bundle.eval_layernorm_in_place(&format!("cond.prop_enc.{k}.1"), &mut h0, 1, d, 1e-5)?;
        for val in &mut h0 {
            *val = silu(*val);
        }
        let mut h1 = vec![0.0f32; d];
        bundle.eval_linear(&format!("cond.prop_enc.{k}.3"), &h0, 1, &mut h1)?;
        bundle.eval_layernorm_in_place(&format!("cond.prop_enc.{k}.4"), &mut h1, 1, d, 1e-5)?;
        for c_i in 0..d {
            per_prop_cat[k * d + c_i] = silu(h1[c_i]) + prop_id_emb[k * d + c_i];
        }
    }

    let mut global_c = vec![0.0f32; d];
    if has_any {
        let mut gm0 = vec![0.0f32; 2 * d];
        bundle.eval_linear("cond.global_mlp.0", &per_prop_cat, 1, &mut gm0)?;
        bundle.eval_layernorm_in_place("cond.global_mlp.1", &mut gm0, 1, 2 * d, 1e-5)?;
        for val in &mut gm0 {
            *val = silu(*val);
        }
        bundle.eval_linear("cond.global_mlp.3", &gm0, 1, &mut global_c)?;
        bundle.eval_layernorm_in_place("cond.global_mlp.4", &mut global_c, 1, d, 1e-5)?;
        for val in &mut global_c {
            *val = silu(*val);
        }
    }

    // 1. Step-0 root residual `root_gate * root_proj(global_c)`
    let mut root_delta = vec![0.0f32; d];
    bundle.eval_linear("cond.root_proj", &global_c, 1, &mut root_delta)?;
    for val in &mut root_delta {
        *val *= cc.root_gate;
    }

    // 2. Causal prefix tokens (if `num_prefix_tokens > 0`)
    let p_cnt = cc.num_prefix_tokens;
    let mut prefix_tokens = vec![0.0f32; p_cnt * d];
    if p_cnt > 0 && bundle.header.tensors.contains_key("cond.prefix_queries") {
        if has_any {
            let queries = bundle.f32_slice("cond.prefix_queries")?;
            let mut cat_in = vec![0.0f32; p_cnt * (2 * d)];
            for p in 0..p_cnt {
                cat_in[p * 2 * d..p * 2 * d + d].copy_from_slice(&queries[p * d..(p + 1) * d]);
                cat_in[p * 2 * d + d..(p + 1) * 2 * d].copy_from_slice(&global_c);
            }
            let mut pp0 = vec![0.0f32; p_cnt * d];
            bundle.eval_linear("cond.prefix_proj.0", &cat_in, p_cnt, &mut pp0)?;
            bundle.eval_layernorm_in_place("cond.prefix_proj.1", &mut pp0, p_cnt, d, 1e-5)?;
            for val in &mut pp0 {
                *val = silu(*val);
            }
            let mut pp1 = vec![0.0f32; p_cnt * d];
            bundle.eval_linear("cond.prefix_proj.3", &pp0, p_cnt, &mut pp1)?;
            for i in 0..(p_cnt * d) {
                prefix_tokens[i] = queries[i] + pp1[i];
            }
        } else {
            let null_p = bundle.f32_slice("cond.null_prefix")?;
            prefix_tokens.copy_from_slice(&null_p[..p_cnt * d]);
        }
    }

    // 3. Layer & output AdaLN-Zero modulations (`SiLU(global_c) -> Linear -> cap * tanh(x / cap)`)
    let c_silu: Vec<f32> = global_c.iter().map(|&v| silu(v)).collect();
    let eval_mod = |lin_name: &str| -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut raw = vec![0.0f32; 2 * d];
        bundle.eval_linear(lin_name, &c_silu, 1, &mut raw)?;
        if cc.modulation_cap > 0.0 {
            let cap = cc.modulation_cap;
            for v in &mut raw {
                *v = cap * (*v / cap).tanh();
            }
        }
        let scale: Vec<f32> = raw[..d].iter().map(|&s| 1.0 + s).collect();
        let shift: Vec<f32> = raw[d..].to_vec();
        Ok((scale, shift))
    };

    let mut layer_scale = Vec::with_capacity(n_layers);
    let mut layer_shift = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let (sc, sh) = eval_mod(&format!("cond.layer_mod.{l}"))?;
        layer_scale.push(sc);
        layer_shift.push(sh);
    }
    let (out_scale, out_shift) = eval_mod("cond.out_mod")?;

    Ok(PrecomputedCond {
        targets_norm: t_norm,
        active_mask: act,
        has_any,
        root_delta,
        prefix_tokens,
        num_prefix: p_cnt,
        layer_scale,
        layer_shift,
        out_scale,
        out_shift,
    })
}

/// Compute 6-D non-trapping global scale-free budget features and project through `budget_emb`.
pub fn eval_budget_embedding(bundle: &ModelBundle, level: i32, remaining_atoms: i32, out_add: &mut [f32]) -> Result<(), String> {
    if !bundle.header.backbone.has_budget_emb {
        return Ok(());
    }
    let ra = (remaining_atoms as f32).max(0.0);
    let has_b = if remaining_atoms > 0 { 1.0f32 } else { 0.0f32 };
    let lvl_f = (level as f32).clamp(-1.0, 8.0);
    let log_ratio = (ra.max(1.0) / 30.0).ln();
    let f = [
        has_b,
        has_b * ((1.0 + ra).ln() / 4.5),
        has_b * ((ra - 30.0) / 25.0).clamp(-2.5, 4.5),
        (lvl_f + 1.0) / 8.0,
        has_b * (log_ratio * std::f32::consts::PI).sin(),
        has_b * (log_ratio * std::f32::consts::PI).cos(),
    ];
    let mut h0 = vec![0.0f32; 256];
    bundle.eval_linear("budget_emb.0", &f, 1, &mut h0)?;
    bundle.eval_layernorm_in_place("budget_emb.1", &mut h0, 1, 256, 1e-5)?;
    for v in &mut h0 {
        *v = silu(*v);
    }
    let d = bundle.header.backbone.emb_dim;
    let mut out = vec![0.0f32; d];
    bundle.eval_linear("budget_emb.3", &h0, 1, &mut out)?;
    for c in 0..d {
        out_add[c] += out[c];
    }
    Ok(())
}

/// Embed a single octree token `(typ, xyz_bins, level, remaining_atoms)`.
pub fn embed_octree_token(
    bundle: &ModelBundle,
    typ: usize,
    xyz_bins: [usize; 3],
    level: i32,
    remaining_atoms: i32,
    root_delta: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    let bb = &bundle.header.backbone;
    let d = bb.emb_dim;
    let atom_emb = bundle.f32_slice("bb.atom_emb")?;
    let x_emb = bundle.f32_slice("bb.xyz_emb.0")?;
    let y_emb = bundle.f32_slice("bb.xyz_emb.1")?;
    let z_emb = bundle.f32_slice("bb.xyz_emb.2")?;
    let lvl_emb = bundle.f32_slice("bb.cur_level")?;

    let mut out = vec![0.0f32; d];
    let t_safe = typ.min(bb.num_atom_vocab - 1);
    let bx = xyz_bins[0].min(bb.num_xyz - 1);
    let by = xyz_bins[1].min(bb.num_xyz - 1);
    let bz = xyz_bins[2].min(bb.num_xyz - 1);
    let lv_idx = (level + 1).max(0) as usize;
    let max_lv = (lvl_emb.len() / d).saturating_sub(1);
    let lv_safe = lv_idx.min(max_lv);

    for c in 0..d {
        out[c] = atom_emb[t_safe * d + c]
            + x_emb[bx * d + c]
            + y_emb[by * d + c]
            + z_emb[bz * d + c]
            + lvl_emb[lv_safe * d + c];
    }
    eval_budget_embedding(bundle, level, remaining_atoms, &mut out)?;
    if let Some(rd) = root_delta {
        for c in 0..d {
            out[c] += rd[c];
        }
    }
    Ok(out)
}

/// Compute 3D Axial RoPE `(cos, sin)` of length `head_dim / 2` (`32` pairs: `30` active + `2` identity).
pub fn compute_rope_3d(phy_pos: [f32; 3], xyz_res: f32, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let pairs_per_axis = half / 3; // 10 for head_dim=64
    let mut cos_v = vec![1.0f32; half];
    let mut sin_v = vec![0.0f32; half];
    let gp = [phy_pos[0] / xyz_res, phy_pos[1] / xyz_res, phy_pos[2] / xyz_res];
    for axis in 0..3 {
        for p in 0..pairs_per_axis {
            let exp = (p as f32) / (pairs_per_axis as f32);
            let rate = theta.powf(-exp);
            let ang = gp[axis] * rate;
            let idx = axis * pairs_per_axis + p;
            cos_v[idx] = ang.cos();
            sin_v[idx] = ang.sin();
        }
    }
    (cos_v, sin_v)
}

#[inline]
pub fn apply_rope_in_place(vec_hd: &mut [f32], num_heads: usize, head_dim: usize, cos_v: &[f32], sin_v: &[f32]) {
    let half = head_dim / 2;
    for h in 0..num_heads {
        let base = h * head_dim;
        for p in 0..half {
            let x0 = vec_hd[base + 2 * p];
            let x1 = vec_hd[base + 2 * p + 1];
            let c = cos_v[p];
            let s = sin_v[p];
            vec_hd[base + 2 * p] = x0 * c - x1 * s;
            vec_hd[base + 2 * p + 1] = x0 * s + x1 * c;
        }
    }
}

/// Per-slot KV Cache on CPU (used for `parity` verification and WASM CPU fallback).
#[derive(Clone, Debug)]
pub struct CpuKvSlot {
    pub k: Vec<Vec<f32>>, // [layer][kv_len * emb_dim]
    pub v: Vec<Vec<f32>>, // [layer][kv_len * emb_dim]
    pub len: usize,
}

impl CpuKvSlot {
    pub fn new(num_layers: usize, cap_tokens: usize, emb_dim: usize) -> Self {
        let mut k = Vec::with_capacity(num_layers);
        let mut v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k.push(Vec::with_capacity(cap_tokens * emb_dim));
            v.push(Vec::with_capacity(cap_tokens * emb_dim));
        }
        Self { k, v, len: 0 }
    }
}

/// Run 1 step (`seq_q` tokens, e.g. `P+1` at step 0 or `2` at step >= 1) for a single branch on CPU,
/// appending rotated `K` and `V` to `kv_slot` and returning the post-`norm` + post-`out_mod` hidden state
/// of the **last** query token `(emb_dim,)`.
pub fn forward_decoder_step_cpu(
    bundle: &ModelBundle,
    x_tokens: &[f32],
    phy_positions: &[[f32; 3]],
    cond: &PrecomputedCond,
    kv_slot: &mut CpuKvSlot,
) -> Result<Vec<f32>, String> {
    let bb = &bundle.header.backbone;
    let seq_q = phy_positions.len();
    let d = bb.emb_dim;
    let n_heads = bb.num_head;
    let hd = bb.head_dim;
    let ffn_d = bb.ffn_dim;
    let scale = 1.0 / (hd as f32).sqrt();
    debug_assert_eq!(x_tokens.len(), seq_q * d);

    let ropes: Vec<(Vec<f32>, Vec<f32>)> = phy_positions
        .iter()
        .map(|&p| compute_rope_3d(p, bb.xyz_resolution, hd, bb.rope_theta))
        .collect();

    let mut h = x_tokens.to_vec();
    let mut x_norm = vec![0.0f32; seq_q * d];
    let mut q_all = vec![0.0f32; seq_q * d];
    let mut k_new = vec![0.0f32; seq_q * d];
    let mut v_new = vec![0.0f32; seq_q * d];
    let mut attn_cat = vec![0.0f32; seq_q * d];
    let mut attn_proj = vec![0.0f32; seq_q * d];
    let mut gate = vec![0.0f32; seq_q * ffn_d];
    let mut up = vec![0.0f32; seq_q * ffn_d];
    let mut ffn_out = vec![0.0f32; seq_q * d];

    let prev_len = kv_slot.len;
    let seq_k = prev_len + seq_q;

    for l in 0..bb.num_layers {
        let lp = format!("bb.layers.{l}");
        x_norm.copy_from_slice(&h);
        bundle.eval_rmsnorm_in_place(&format!("{lp}.attn_norm.weight"), &mut x_norm, seq_q, d, 1e-6)?;

        bundle.eval_linear(&format!("{lp}.attn.q_proj"), &x_norm, seq_q, &mut q_all)?;
        bundle.eval_linear(&format!("{lp}.attn.k_proj"), &x_norm, seq_q, &mut k_new)?;
        bundle.eval_linear(&format!("{lp}.attn.v_proj"), &x_norm, seq_q, &mut v_new)?;

        for qi in 0..seq_q {
            apply_rope_in_place(&mut q_all[qi * d..(qi + 1) * d], n_heads, hd, &ropes[qi].0, &ropes[qi].1);
            apply_rope_in_place(&mut k_new[qi * d..(qi + 1) * d], n_heads, hd, &ropes[qi].0, &ropes[qi].1);
        }

        kv_slot.k[l].extend_from_slice(&k_new);
        kv_slot.v[l].extend_from_slice(&v_new);
        let k_buf = &kv_slot.k[l];
        let v_buf = &kv_slot.v[l];

        // Causal attention: query `qi` attends to `kj <= prev_len + qi`
        let mut scores = vec![0.0f32; seq_k];
        for qi in 0..seq_q {
            let max_kj = prev_len + qi;
            for head in 0..n_heads {
                let q_vec = &q_all[qi * d + head * hd..qi * d + (head + 1) * hd];
                let mut max_s = f32::NEG_INFINITY;
                for kj in 0..=max_kj {
                    let k_vec = &k_buf[kj * d + head * hd..kj * d + (head + 1) * hd];
                    let mut dot = 0.0f32;
                    for c in 0..hd {
                        dot += q_vec[c] * k_vec[c];
                    }
                    let s = dot * scale;
                    scores[kj] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }
                let mut sum_e = 0.0f32;
                for kj in 0..=max_kj {
                    let e = (scores[kj] - max_s).exp();
                    scores[kj] = e;
                    sum_e += e;
                }
                let inv_e = 1.0 / sum_e.max(1e-12);
                let out_slice = &mut attn_cat[qi * d + head * hd..qi * d + (head + 1) * hd];
                out_slice.fill(0.0);
                for kj in 0..=max_kj {
                    let w = scores[kj] * inv_e;
                    let v_vec = &v_buf[kj * d + head * hd..kj * d + (head + 1) * hd];
                    for c in 0..hd {
                        out_slice[c] += w * v_vec[c];
                    }
                }
            }
        }

        bundle.eval_linear(&format!("{lp}.attn.out_proj"), &attn_cat, seq_q, &mut attn_proj)?;
        for i in 0..(seq_q * d) {
            h[i] += attn_proj[i];
        }

        x_norm.copy_from_slice(&h);
        bundle.eval_rmsnorm_in_place(&format!("{lp}.ffn_norm.weight"), &mut x_norm, seq_q, d, 1e-6)?;
        bundle.eval_linear(&format!("{lp}.ffn.w1"), &x_norm, seq_q, &mut gate)?;
        bundle.eval_linear(&format!("{lp}.ffn.w3"), &x_norm, seq_q, &mut up)?;
        for i in 0..(seq_q * ffn_d) {
            gate[i] = silu(gate[i]) * up[i];
        }
        bundle.eval_linear(&format!("{lp}.ffn.w2"), &gate, seq_q, &mut ffn_out)?;

        let l_sc = &cond.layer_scale[l];
        let l_sh = &cond.layer_shift[l];
        for qi in 0..seq_q {
            let row = &mut h[qi * d..(qi + 1) * d];
            let fo = &ffn_out[qi * d..(qi + 1) * d];
            for c in 0..d {
                row[c] = (row[c] + fo[c]) * l_sc[c] + l_sh[c];
            }
        }
    }
    kv_slot.len = seq_k;

    // Final norm + out_mod on the last query token (`seq_q - 1`)
    let mut last = h[(seq_q - 1) * d..seq_q * d].to_vec();
    bundle.eval_rmsnorm_in_place("bb.norm.weight", &mut last, 1, d, 1e-6)?;
    for c in 0..d {
        last[c] = last[c] * cond.out_scale[c] + cond.out_shift[c];
    }
    Ok(last)
}

/// Evaluate `TaskHead` (`out(gelu_exact(dense(x)))`) for `rows` hidden vectors.
pub fn eval_task_head(bundle: &ModelBundle, prefix: &str, h: &[f32], rows: usize, out_dim: usize) -> Result<Vec<f32>, String> {
    let d = bundle.header.backbone.emb_dim;
    let mut mid = vec![0.0f32; rows * d];
    bundle.eval_linear(&format!("{prefix}.dense"), h, rows, &mut mid)?;
    for v in &mut mid {
        *v = gelu_exact(*v);
    }
    let mut logits = vec![0.0f32; rows * out_dim];
    bundle.eval_linear(&format!("{prefix}.out"), &mid, rows, &mut logits)?;
    Ok(logits)
}

pub fn eval_tree_head(bundle: &ModelBundle, h: &[f32], rows: usize) -> Result<Vec<f32>, String> {
    eval_task_head(bundle, "bb.tree_head", h, rows, 255)
}

pub fn eval_atom_head_axis(bundle: &ModelBundle, axis: usize, feat: &[f32], rows: usize) -> Result<Vec<f32>, String> {
    let out_dim = if axis == 0 {
        bundle.header.backbone.num_atom_vocab
    } else {
        bundle.header.backbone.num_xyz
    };
    eval_task_head(bundle, &format!("bb.atom_head.pred.{axis}"), feat, rows, out_dim)
}

pub fn add_atom_head_step_emb(bundle: &ModelBundle, axis: usize, token_id: usize, feat: &mut [f32]) -> Result<(), String> {
    let d = bundle.header.backbone.emb_dim;
    let emb = bundle.f32_slice(&format!("bb.atom_head.emb.{axis}"))?;
    let max_id = (emb.len() / d).saturating_sub(1);
    let safe_id = token_id.min(max_id);
    let row = &emb[safe_id * d..(safe_id + 1) * d];
    for c in 0..d {
        feat[c] += row[c];
    }
    Ok(())
}
