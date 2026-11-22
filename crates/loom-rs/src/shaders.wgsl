// W3C WebGPU-compliant Compute Shaders for `loom-rs` (`Uni-3DAR` QLoRA/FP16 Forward & Backward).
// Uses standard `unpack2x16float` (zero optional extensions required) and <= 7 storage buffers per stage.

struct Params {
    p0: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    p4: u32,
    p5: u32,
    f0: f32,
    f1: f32,
}

// ============================================================================
// 1. Tiled FP16-Weight Linear Forward: `C[M, N] = A[M, K] @ W[N, K]^T + add_in[M, N]`
// p0 = M, p1 = N, p2 = K, p3 = pairs_per_row ((K+1)/2), p4 = has_add (0 or 1)
// ============================================================================
@group(0) @binding(0) var<uniform> mm_params: Params;
@group(0) @binding(1) var<storage, read> mm_a: array<f32>;
@group(0) @binding(2) var<storage, read> mm_w_f16: array<u32>;
@group(0) @binding(3) var<storage, read> mm_add: array<f32>;
@group(0) @binding(4) var<storage, read_write> mm_c: array<f32>;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

fn read_w_f16(row_n: u32, col_k: u32, pairs_per_row: u32) -> f32 {
    let word = mm_w_f16[row_n * pairs_per_row + (col_k >> 1u)];
    let pair = unpack2x16float(word);
    if ((col_k & 1u) == 0u) {
        return pair.x;
    } else {
        return pair.y;
    }
}

@compute @workgroup_size(16, 16, 1)
fn mm_f16_fwd(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m_dim = mm_params.p0;
    let n_dim = mm_params.p1;
    let k_dim = mm_params.p2;
    let pairs = mm_params.p3;

    let row_m = gid.y;
    let col_n = gid.x;
    let ly = lid.y;
    let lx = lid.x;

    var acc: f32 = 0.0;
    let num_tiles = (k_dim + 15u) / 16u;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * 16u + lx;
        if (row_m < m_dim && a_col < k_dim) {
            tile_a[ly * 16u + lx] = mm_a[row_m * k_dim + a_col];
        } else {
            tile_a[ly * 16u + lx] = 0.0;
        }

        let b_col_k = t * 16u + ly;
        if (col_n < n_dim && b_col_k < k_dim) {
            tile_b[ly * 16u + lx] = read_w_f16(col_n, b_col_k, pairs);
        } else {
            tile_b[ly * 16u + lx] = 0.0;
        }

        workgroupBarrier();

        for (var p: u32 = 0u; p < 16u; p = p + 1u) {
            acc = acc + tile_a[ly * 16u + p] * tile_b[p * 16u + lx];
        }

        workgroupBarrier();
    }

    if (row_m < m_dim && col_n < n_dim) {
        let out_idx = row_m * n_dim + col_n;
        if (mm_params.p4 != 0u) {
            acc = acc + mm_add[out_idx];
        }
        mm_c[out_idx] = acc;
    }
}

// ============================================================================
// 2. Tiled FP16-Weight Linear Backward wrt Input: `dX[M, K] = dY[M, N] @ W[N, K] + add_in[M, K]`
// p0 = M, p1 = N (out_dim), p2 = K (in_dim), p3 = pairs_per_row ((K+1)/2), p4 = has_add
// ============================================================================
@compute @workgroup_size(16, 16, 1)
fn mm_f16_bwd_dx(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m_dim = mm_params.p0;
    let n_dim = mm_params.p1;
    let k_dim = mm_params.p2;
    let pairs = mm_params.p3;

    let row_m = gid.y;
    let col_k = gid.x;
    let ly = lid.y;
    let lx = lid.x;

    var acc: f32 = 0.0;
    let num_tiles = (n_dim + 15u) / 16u;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let dy_col_n = t * 16u + lx;
        if (row_m < m_dim && dy_col_n < n_dim) {
            tile_a[ly * 16u + lx] = mm_a[row_m * n_dim + dy_col_n];
        } else {
            tile_a[ly * 16u + lx] = 0.0;
        }

        let w_row_n = t * 16u + ly;
        if (w_row_n < n_dim && col_k < k_dim) {
            tile_b[ly * 16u + lx] = read_w_f16(w_row_n, col_k, pairs);
        } else {
            tile_b[ly * 16u + lx] = 0.0;
        }

        workgroupBarrier();

        for (var p: u32 = 0u; p < 16u; p = p + 1u) {
            acc = acc + tile_a[ly * 16u + p] * tile_b[p * 16u + lx];
        }

        workgroupBarrier();
    }

    if (row_m < m_dim && col_k < k_dim) {
        let out_idx = row_m * k_dim + col_k;
        if (mm_params.p4 != 0u) {
            acc = acc + mm_add[out_idx];
        }
        mm_c[out_idx] = acc;
    }
}

// ============================================================================
// 3. GPU LoRA Forward (`XA = X @ A^T`, `Delta = scale * XA @ B^T`) & Elementwise Kernels
// ============================================================================
@group(0) @binding(0) var<uniform> el_params: Params;
@group(0) @binding(1) var<storage, read> el_in0: array<f32>;
@group(0) @binding(2) var<storage, read> el_in1: array<f32>;
@group(0) @binding(3) var<storage, read> el_in2: array<f32>;
@group(0) @binding(4) var<storage, read> el_lens: array<u32>;
@group(0) @binding(5) var<storage, read_write> el_out0: array<f32>;
@group(0) @binding(6) var<storage, read_write> el_out1: array<f32>;

// p0 = rows, p1 = rank, p2 = in_dim
@compute @workgroup_size(64, 1, 1)
fn lora_xa_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = el_params.p0;
    let rank = el_params.p1;
    let in_d = el_params.p2;
    let idx = gid.x;
    if (idx >= rows * rank) {
        return;
    }
    let r = idx % rank;
    let m = idx / rank;
    let x_base = m * in_d;
    let a_base = r * in_d;
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < in_d; k = k + 1u) {
        acc = acc + el_in0[x_base + k] * el_in1[a_base + k];
    }
    el_out0[idx] = acc;
}

// p0 = rows, p1 = out_dim, p2 = rank, f0 = scaling
@compute @workgroup_size(256, 1, 1)
fn lora_xb_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = el_params.p0;
    let out_d = el_params.p1;
    let rank = el_params.p2;
    let sc = el_params.f0;
    let idx = gid.x;
    if (idx >= rows * out_d) {
        return;
    }
    let o = idx % out_d;
    let m = idx / out_d;
    let xa_base = m * rank;
    let b_base = o * rank;
    var acc: f32 = 0.0;
    for (var r: u32 = 0u; r < rank; r = r + 1u) {
        acc = acc + el_in0[xa_base + r] * el_in1[b_base + r];
    }
    el_out0[idx] = acc * sc;
}

@compute @workgroup_size(64, 1, 1)
fn rmsnorm_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = el_params.p0;
    let cols = el_params.p1;
    let eps = el_params.f0;
    let r = gid.x;
    if (r >= rows) {
        return;
    }
    let base = r * cols;
    var sq: f32 = 0.0;
    for (var c: u32 = 0u; c < cols; c = c + 1u) {
        let v = el_in0[base + c];
        sq = sq + v * v;
    }
    let inv_rms = 1.0 / sqrt(sq / f32(cols) + eps);
    for (var c: u32 = 0u; c < cols; c = c + 1u) {
        el_out0[base + c] = el_in0[base + c] * inv_rms * el_in1[c];
    }
}

@compute @workgroup_size(64, 1, 1)
fn rope_fwd_qk(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = el_params.p0;
    let emb_d = el_params.p1;
    let n_heads = el_params.p2;
    let hd = el_params.p3;
    let half = hd / 2u;
    let r = gid.x;
    if (r >= rows) {
        return;
    }
    let cs_base = r * hd;
    let vec_base = r * emb_d;
    for (var h: u32 = 0u; h < n_heads; h = h + 1u) {
        let h_off = vec_base + h * hd;
        for (var p: u32 = 0u; p < half; p = p + 1u) {
            let cv = el_in0[cs_base + p];
            let sv = el_in0[cs_base + half + p];
            let q0 = el_out0[h_off + 2u * p];
            let q1 = el_out0[h_off + 2u * p + 1u];
            el_out0[h_off + 2u * p] = q0 * cv - q1 * sv;
            el_out0[h_off + 2u * p + 1u] = q0 * sv + q1 * cv;

            let k0 = el_out1[h_off + 2u * p];
            let k1 = el_out1[h_off + 2u * p + 1u];
            el_out1[h_off + 2u * p] = k0 * cv - k1 * sv;
            el_out1[h_off + 2u * p + 1u] = k0 * sv + k1 * cv;
        }
    }
}

@compute @workgroup_size(256, 1, 1)
fn elem_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let mode = el_params.p0;
    let b_sz = el_params.p1;
    let max_s = el_params.p2;
    let cols = el_params.p3;
    let total = b_sz * max_s * cols;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    if (mode == 0u) {
        el_out0[idx] = el_in0[idx] + el_in1[idx];
    } else if (mode == 1u) {
        let g = el_in0[idx];
        let s = g / (1.0 + exp(-g));
        el_out0[idx] = s * el_in1[idx];
    } else if (mode == 2u) {
        let c = idx % cols;
        let rem = idx / cols;
        let s = rem % max_s;
        let b = rem / max_s;
        if (s >= el_lens[b]) {
            el_out0[idx] = 0.0;
        } else {
            let gamma = el_in2[b * 2u * cols + c];
            let beta = el_in2[b * 2u * cols + cols + c];
            el_out0[idx] = (el_in0[idx] + el_in1[idx]) * (1.0 + gamma) + beta;
        }
    } else {
        let c = idx % cols;
        let rem = idx / cols;
        let s = rem % max_s;
        let b = rem / max_s;
        if (s >= el_lens[b]) {
            el_out0[idx] = 0.0;
        } else {
            let gamma = el_in2[b * 2u * cols + c];
            let beta = el_in2[b * 2u * cols + cols + c];
            el_out0[idx] = el_in0[idx] * (1.0 + gamma) + beta;
        }
    }
}

// ============================================================================
// 4. Memory-Free Causal Multi-Head Attention Forward (stores only `LSE[B, H, S]`)
// ============================================================================
@group(0) @binding(0) var<uniform> mha_params: Params;
@group(0) @binding(1) var<storage, read> mha_q: array<f32>;
@group(0) @binding(2) var<storage, read> mha_k: array<f32>;
@group(0) @binding(3) var<storage, read> mha_v: array<f32>;
@group(0) @binding(4) var<storage, read> mha_lens: array<u32>;
@group(0) @binding(5) var<storage, read_write> mha_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> mha_lse: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn causal_mha_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b_sz = mha_params.p0;
    let max_s = mha_params.p1;
    let n_heads = mha_params.p2;
    let hd = mha_params.p3;
    let emb_d = n_heads * hd;
    let scale = mha_params.f0;

    let total_rows = b_sz * n_heads * max_s;
    let idx = gid.x;
    if (idx >= total_rows) {
        return;
    }
    let qi = idx % max_s;
    let rem = idx / max_s;
    let h = rem % n_heads;
    let b = rem / n_heads;

    let seq_len = mha_lens[b];
    let q_base = (b * max_s + qi) * emb_d + h * hd;
    let lse_idx = (b * n_heads + h) * max_s + qi;

    if (qi >= seq_len) {
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            mha_out[q_base + d] = 0.0;
        }
        mha_lse[lse_idx] = 0.0;
        return;
    }

    var q_reg: array<f32, 64>;
    var out_reg: array<f32, 64>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        q_reg[d] = mha_q[q_base + d];
        out_reg[d] = 0.0;
    }

    var m_cur: f32 = -1.0e30;
    var l_cur: f32 = 0.0;

    for (var kj: u32 = 0u; kj <= qi; kj = kj + 1u) {
        let kv_base = (b * max_s + kj) * emb_d + h * hd;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dot = dot + q_reg[d] * mha_k[kv_base + d];
        }
        let s = dot * scale;
        let m_new = max(m_cur, s);
        let exp_prev = exp(m_cur - m_new);
        let exp_s = exp(s - m_new);
        l_cur = l_cur * exp_prev + exp_s;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            out_reg[d] = out_reg[d] * exp_prev + exp_s * mha_v[kv_base + d];
        }
        m_cur = m_new;
    }

    let inv_l = 1.0 / max(l_cur, 1.0e-20);
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        mha_out[q_base + d] = out_reg[d] * inv_l;
    }
    mha_lse[lse_idx] = m_cur + log(max(l_cur, 1.0e-20));
}

// ============================================================================
// 5. Memory-Free Causal MHA Backward (`dQ` and `dK, dV`, zero atomics!)
// ============================================================================
@group(0) @binding(0) var<uniform> bwd_params: Params;
@group(0) @binding(1) var<storage, read> bwd_q: array<f32>;
@group(0) @binding(2) var<storage, read> bwd_k: array<f32>;
@group(0) @binding(3) var<storage, read> bwd_v: array<f32>;
@group(0) @binding(4) var<storage, read> bwd_do: array<f32>;
@group(0) @binding(5) var<storage, read> bwd_lse_delta: array<f32>;
@group(0) @binding(6) var<storage, read> bwd_lens: array<u32>;
@group(0) @binding(7) var<storage, read_write> bwd_out: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn causal_mha_bwd_dq(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b_sz = bwd_params.p0;
    let max_s = bwd_params.p1;
    let n_heads = bwd_params.p2;
    let hd = bwd_params.p3;
    let emb_d = n_heads * hd;
    let scale = bwd_params.f0;

    let lse_stride = b_sz * n_heads * max_s;
    let idx = gid.x;
    if (idx >= lse_stride) {
        return;
    }
    let qi = idx % max_s;
    let rem = idx / max_s;
    let h = rem % n_heads;
    let b = rem / n_heads;

    let seq_len = bwd_lens[b];
    let q_base = (b * max_s + qi) * emb_d + h * hd;

    if (qi >= seq_len) {
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            bwd_out[q_base + d] = 0.0;
        }
        return;
    }

    let lse_i = bwd_lse_delta[(b * n_heads + h) * max_s + qi];
    let delta_i = bwd_lse_delta[lse_stride + (b * n_heads + h) * max_s + qi];

    var q_reg: array<f32, 64>;
    var do_reg: array<f32, 64>;
    var dq_reg: array<f32, 64>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        q_reg[d] = bwd_q[q_base + d];
        do_reg[d] = bwd_do[q_base + d];
        dq_reg[d] = 0.0;
    }

    for (var kj: u32 = 0u; kj <= qi; kj = kj + 1u) {
        let kv_base = (b * max_s + kj) * emb_d + h * hd;
        var qk: f32 = 0.0;
        var dov: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            qk = qk + q_reg[d] * bwd_k[kv_base + d];
            dov = dov + do_reg[d] * bwd_v[kv_base + d];
        }
        let p = exp(qk * scale - lse_i);
        let ds = p * (dov - delta_i) * scale;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dq_reg[d] = dq_reg[d] + ds * bwd_k[kv_base + d];
        }
    }

    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        bwd_out[q_base + d] = dq_reg[d];
    }
}

@compute @workgroup_size(64, 1, 1)
fn causal_mha_bwd_dkv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b_sz = bwd_params.p0;
    let max_s = bwd_params.p1;
    let n_heads = bwd_params.p2;
    let hd = bwd_params.p3;
    let emb_d = n_heads * hd;
    let scale = bwd_params.f0;

    let lse_stride = b_sz * n_heads * max_s;
    let kv_total_elems = b_sz * max_s * emb_d;
    let idx = gid.x;
    if (idx >= lse_stride) {
        return;
    }
    let kj = idx % max_s;
    let rem = idx / max_s;
    let h = rem % n_heads;
    let b = rem / n_heads;

    let seq_len = bwd_lens[b];
    let kv_base = (b * max_s + kj) * emb_d + h * hd;

    if (kj >= seq_len) {
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            bwd_out[kv_base + d] = 0.0;
            bwd_out[kv_total_elems + kv_base + d] = 0.0;
        }
        return;
    }

    var k_reg: array<f32, 64>;
    var v_reg: array<f32, 64>;
    var dk_reg: array<f32, 64>;
    var dv_reg: array<f32, 64>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        k_reg[d] = bwd_k[kv_base + d];
        v_reg[d] = bwd_v[kv_base + d];
        dk_reg[d] = 0.0;
        dv_reg[d] = 0.0;
    }

    for (var qi: u32 = kj; qi < seq_len; qi = qi + 1u) {
        let q_base = (b * max_s + qi) * emb_d + h * hd;
        let lse_i = bwd_lse_delta[(b * n_heads + h) * max_s + qi];
        let delta_i = bwd_lse_delta[lse_stride + (b * n_heads + h) * max_s + qi];

        var qk: f32 = 0.0;
        var dov: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            qk = qk + bwd_q[q_base + d] * k_reg[d];
            dov = dov + bwd_do[q_base + d] * v_reg[d];
        }
        let p = exp(qk * scale - lse_i);
        let ds = p * (dov - delta_i) * scale;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dv_reg[d] = dv_reg[d] + p * bwd_do[q_base + d];
            dk_reg[d] = dk_reg[d] + ds * bwd_q[q_base + d];
        }
    }

    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        bwd_out[kv_base + d] = dk_reg[d];
        bwd_out[kv_total_elems + kv_base + d] = dv_reg[d];
    }
}
