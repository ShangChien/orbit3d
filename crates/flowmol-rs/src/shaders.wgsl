struct OpParams {
    p0: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    p4: u32,
    p5: u32,
    p6: u32,
    p7: u32,
    p8: u32,
    p9: u32,
    p10: u32,
    p11: u32,
    f0: f32,
    f1: f32,
    u14: u32,
    u15: u32,
};

@group(0) @binding(0) var<uniform> u: OpParams;
@group(0) @binding(1) var<storage, read> weights_blob: array<u32>;
@group(0) @binding(2) var<storage, read> f32_params: array<f32>;
@group(0) @binding(3) var<storage, read> graph_idx: array<u32>;
@group(0) @binding(4) var<storage, read_write> ws: array<f32>;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

fn sigmoid_f(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// ============================================================================
// 1. Tiled Shared-Memory GEMM with F32 / F16 / Q8_0 on-the-fly dequantization
// ============================================================================
// p0 = rows, p1 = out_cols, p2 = orig_in_cols, p3 = pad_in_cols
// p4 = quant_mode (0=F32, 1=F16, 2=Q8_0), p5 = act_mode (0=None, 1=SiLU, 2=Sigmoid)
// p6 = add_mode (0=None, 1=add_src, 2=add_src_dst, 3=add_elem, 4=res_add_silu_to_a)
// p7 = x_off (in ws), p8 = y_off (in ws), p9 = w_u32_off (in weights_blob)
// p10 = bias_off (in f32_params, 0xFFFFFFFF = none), p11 = blocks_per_row
// u14 = add_a_off (in ws), u15 = add_b_off (in ws)
// graph_idx layout: [0..E] = src, [E..2E] = dst, [2E..3E] = dst_incoming
var<workgroup> tileA: array<f32, 1088>; // 64 rows x 17 stride (BK=16, padded for zero bank conflicts)
var<workgroup> tileB: array<f32, 1088>; // 16 k x 68 cols (BK=16, BN=64, transposed for conflict-free vec4 reads)

fn store_gemm_epilogue(row: u32, col: u32, raw_acc: f32) {
    let rows = u.p0;
    let out_cols = u.p1;
    if (row >= rows || col >= out_cols) {
        return;
    }
    var acc = raw_acc;
    if (u.p10 != 0xFFFFFFFFu) {
        acc = acc + f32_params[u.p10 + col];
    }
    let add_m = u.p6;
    if (add_m == 1u) {
        let s_idx = graph_idx[row];
        acc = acc + ws[u.u14 + s_idx * out_cols + col];
    } else if (add_m == 2u) {
        let s_idx = graph_idx[row];
        let d_idx = graph_idx[rows + row];
        acc = acc + ws[u.u14 + s_idx * out_cols + col] + ws[u.u15 + d_idx * out_cols + col];
    } else if (add_m == 3u) {
        acc = acc + ws[u.u14 + row * out_cols + col];
    }

    var val = acc;
    if (u.p5 == 1u) {
        val = silu(val);
    } else if (u.p5 == 2u) {
        val = sigmoid_f(val);
    }

    if (add_m == 4u) {
        ws[u.p8 + row * out_cols + col] = ws[u.u14 + row * out_cols + col] + val;
    } else {
        ws[u.p8 + row * out_cols + col] = val;
    }
}

@compute @workgroup_size(16, 16, 1)
fn gemm_fused(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let lx = lid.x; // 0..15 -> 4 cols per thread
    let ly = lid.y; // 0..15 -> 4 rows per thread
    let row_base = wid.y * 64u;
    let col_base = wid.x * 64u;

    let rows = u.p0;
    let out_cols = u.p1;
    let in_cols = u.p2;
    let pad_cols = u.p3;
    let qmode = u.p4;
    let w_base = u.p9;
    let bpr = u.p11;

    var c0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var c1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var c2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var c3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let n_tiles = (in_cols + 15u) / 16u;

    for (var kt: u32 = 0u; kt < n_tiles; kt = kt + 1u) {
        let k_base = kt * 16u;

        // 256 threads load 64x16 = 1024 floats of tileA (4 rows per thread: tr = ly*4 + dr, k = lx)
        let ka = lx;
        let ka_valid = (k_base + ka) < in_cols;
        for (var dr: u32 = 0u; dr < 4u; dr = dr + 1u) {
            let tr = ly * 4u + dr;
            let gr = row_base + tr;
            if (gr < rows && ka_valid) {
                tileA[tr * 17u + ka] = ws[u.p7 + gr * in_cols + k_base + ka];
            } else {
                tileA[tr * 17u + ka] = 0.0;
            }
        }

        // 256 threads load & unpack 16x64 = 1024 floats of tileB[k, col]
        for (var dc: u32 = 0u; dc < 4u; dc = dc + 1u) {
            let tc = lx * 4u + dc;
            let gc = col_base + tc;
            if (qmode == 0u) {
                let kb = ly;
                if (gc < out_cols && (k_base + kb) < in_cols) {
                    tileB[kb * 68u + tc] = bitcast<f32>(weights_blob[w_base + gc * pad_cols + k_base + kb]);
                } else {
                    tileB[kb * 68u + tc] = 0.0;
                }
            } else if (qmode == 1u) {
                if (ly < 8u) {
                    let kb0 = ly * 2u;
                    let kb1 = kb0 + 1u;
                    if (gc < out_cols) {
                        let pair_idx = (gc * pad_cols + k_base + kb0) / 2u;
                        let up = unpack2x16float(weights_blob[w_base + pair_idx]);
                        tileB[kb0 * 68u + tc] = select(0.0, up.x, (k_base + kb0) < in_cols);
                        tileB[kb1 * 68u + tc] = select(0.0, up.y, (k_base + kb1) < in_cols);
                    } else {
                        tileB[kb0 * 68u + tc] = 0.0;
                        tileB[kb1 * 68u + tc] = 0.0;
                    }
                }
            } else {
                if (ly < 4u) {
                    let j0 = ly * 4u;
                    if (gc < out_cols) {
                        let blk_idx = kt >> 1u;
                        let half_off = (kt & 1u) * 4u;
                        let blk_off = w_base + (gc * bpr + blk_idx) * 9u;
                        let scale = bitcast<f32>(weights_blob[blk_off]);
                        let packed = bitcast<i32>(weights_blob[blk_off + 1u + half_off + ly]);
                        tileB[(j0 + 0u) * 68u + tc] = select(0.0, f32(extractBits(packed, 0u, 8u)) * scale, (k_base + j0 + 0u) < in_cols);
                        tileB[(j0 + 1u) * 68u + tc] = select(0.0, f32(extractBits(packed, 8u, 8u)) * scale, (k_base + j0 + 1u) < in_cols);
                        tileB[(j0 + 2u) * 68u + tc] = select(0.0, f32(extractBits(packed, 16u, 8u)) * scale, (k_base + j0 + 2u) < in_cols);
                        tileB[(j0 + 3u) * 68u + tc] = select(0.0, f32(extractBits(packed, 24u, 8u)) * scale, (k_base + j0 + 3u) < in_cols);
                    } else {
                        tileB[(j0 + 0u) * 68u + tc] = 0.0;
                        tileB[(j0 + 1u) * 68u + tc] = 0.0;
                        tileB[(j0 + 2u) * 68u + tc] = 0.0;
                        tileB[(j0 + 3u) * 68u + tc] = 0.0;
                    }
                }
            }
        }

        workgroupBarrier();

        let ar0 = (ly * 4u) * 17u;
        let bc0 = lx * 4u;
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let a0 = tileA[ar0 + kk];
            let a1 = tileA[ar0 + 17u + kk];
            let a2 = tileA[ar0 + 34u + kk];
            let a3 = tileA[ar0 + 51u + kk];
            let bi = kk * 68u + bc0;
            let bv = vec4<f32>(tileB[bi], tileB[bi + 1u], tileB[bi + 2u], tileB[bi + 3u]);
            c0 = fma(vec4<f32>(a0), bv, c0);
            c1 = fma(vec4<f32>(a1), bv, c1);
            c2 = fma(vec4<f32>(a2), bv, c2);
            c3 = fma(vec4<f32>(a3), bv, c3);
        }

        workgroupBarrier();
    }

    let r0 = row_base + ly * 4u;
    let col0 = col_base + lx * 4u;
    store_gemm_epilogue(r0 + 0u, col0 + 0u, c0.x);
    store_gemm_epilogue(r0 + 0u, col0 + 1u, c0.y);
    store_gemm_epilogue(r0 + 0u, col0 + 2u, c0.z);
    store_gemm_epilogue(r0 + 0u, col0 + 3u, c0.w);
    store_gemm_epilogue(r0 + 1u, col0 + 0u, c1.x);
    store_gemm_epilogue(r0 + 1u, col0 + 1u, c1.y);
    store_gemm_epilogue(r0 + 1u, col0 + 2u, c1.z);
    store_gemm_epilogue(r0 + 1u, col0 + 3u, c1.w);
    store_gemm_epilogue(r0 + 2u, col0 + 0u, c2.x);
    store_gemm_epilogue(r0 + 2u, col0 + 1u, c2.y);
    store_gemm_epilogue(r0 + 2u, col0 + 2u, c2.z);
    store_gemm_epilogue(r0 + 2u, col0 + 3u, c2.w);
    store_gemm_epilogue(r0 + 3u, col0 + 0u, c3.x);
    store_gemm_epilogue(r0 + 3u, col0 + 1u, c3.y);
    store_gemm_epilogue(r0 + 3u, col0 + 2u, c3.z);
    store_gemm_epilogue(r0 + 3u, col0 + 3u, c3.w);
}

// ============================================================================
// 2. LayerNorm + optional AdaLN Modulation
// ============================================================================
// p0 = rows, p1 = cols, p2 = x_off (in-place in ws), p3 = ln_w_off (f32_params), p4 = ln_b_off
// p5 = mode (0 = LayerNorm only, 1 = AdaLN only `h = h*scale + shift`, 2 = LayerNorm then AdaLN)
// p6 = mod_scale_off (f32_params), p7 = mod_shift_off (f32_params)
var<workgroup> ln_sum: array<f32, 64>;
var<workgroup> ln_sq: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn layernorm_mod(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let r = wid.x;
    let tid = lid.x;
    let rows = u.p0;
    let cols = u.p1;
    if (r >= rows) {
        return;
    }
    let base = u.p2 + r * cols;

    if (u.p5 == 1u) {
        for (var c: u32 = tid; c < cols; c = c + 64u) {
            ws[base + c] = ws[base + c] * f32_params[u.p6 + c] + f32_params[u.p7 + c];
        }
        return;
    }

    var local_s: f32 = 0.0;
    for (var c: u32 = tid; c < cols; c = c + 64u) {
        local_s = local_s + ws[base + c];
    }
    ln_sum[tid] = local_s;
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            ln_sum[tid] = ln_sum[tid] + ln_sum[tid + s];
        }
        workgroupBarrier();
    }
    let mean = ln_sum[0] / f32(cols);

    var local_v: f32 = 0.0;
    for (var c: u32 = tid; c < cols; c = c + 64u) {
        let d = ws[base + c] - mean;
        local_v = local_v + d * d;
    }
    ln_sq[tid] = local_v;
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            ln_sq[tid] = ln_sq[tid] + ln_sq[tid + s];
        }
        workgroupBarrier();
    }
    let inv_std = 1.0 / sqrt(ln_sq[0] / f32(cols) + 1e-5);

    for (var c: u32 = tid; c < cols; c = c + 64u) {
        var v = (ws[base + c] - mean) * inv_std * f32_params[u.p3 + c] + f32_params[u.p4 + c];
        if (u.p5 == 2u) {
            v = v * f32_params[u.p6 + c] + f32_params[u.p7 + c];
        }
        ws[base + c] = v;
    }
}

// ============================================================================
// 3. Pairwise Edge Geometry + Gaussian RBF (+ optional [e, rbf] packing)
// ============================================================================
// p0 = n_edges, p1 = rbf_dim, p2 = x_off, p3 = unit_diff_off, p4 = rbf_off
// p5 = pack_er_flag (1 = also write [e, rbf] to p7), p6 = e_off, p7 = er_out_off, p8 = e_dim
// f0 = rbf_dmax
@compute @workgroup_size(64, 1, 1)
fn edge_geom_rbf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e_i = gid.x;
    let n_edges = u.p0;
    if (e_i >= n_edges) {
        return;
    }
    let rbf_dim = u.p1;
    let s = graph_idx[e_i];
    let d = graph_idx[n_edges + e_i];

    let dx = ws[u.p2 + s * 3u + 0u] - ws[u.p2 + d * 3u + 0u];
    let dy = ws[u.p2 + s * 3u + 1u] - ws[u.p2 + d * 3u + 1u];
    let dz = ws[u.p2 + s * 3u + 2u] - ws[u.p2 + d * 3u + 2u];
    let sq = max(dx * dx + dy * dy + dz * dz, 1e-8);
    let dist = sqrt(sq);
    let inv = 1.0 / (dist + 1e-8);

    ws[u.p3 + e_i * 3u + 0u] = dx * inv;
    ws[u.p3 + e_i * 3u + 1u] = dy * inv;
    ws[u.p3 + e_i * 3u + 2u] = dz * inv;

    let dmax = u.f0;
    let sigma = dmax / f32(rbf_dim);
    let denom = f32(max(rbf_dim - 1u, 1u));
    let r_base = u.p4 + e_i * rbf_dim;
    for (var m: u32 = 0u; m < rbf_dim; m = m + 1u) {
        let mu = (f32(m) / denom) * dmax;
        let z = (dist - mu) / sigma;
        ws[r_base + m] = exp(-z * z);
    }

    if (u.p5 == 1u) {
        let e_dim = u.p8;
        let er_cols = e_dim + rbf_dim;
        let dst_b = u.p7 + e_i * er_cols;
        let e_b = u.p6 + e_i * e_dim;
        for (var c: u32 = 0u; c < e_dim; c = c + 1u) {
            ws[dst_b + c] = ws[e_b + c];
        }
        for (var m: u32 = 0u; m < rbf_dim; m = m + 1u) {
            ws[dst_b + e_dim + m] = ws[r_base + m];
        }
    }
}

// ============================================================================
// 4. Fused Equivariant GVP Vector Branch (Zero-Spill Workgroup-Parallel)
// ============================================================================
var<workgroup> wg_vin: array<vec3<f32>, 48>;
var<workgroup> wg_vh: array<vec3<f32>, 48>;
var<workgroup> wg_cp: array<vec3<f32>, 16>;

// Mode A (`gvp_vec_std`, 1 workgroup per row):
//   p0 = rows, p1 = v_in, p2 = v_out, p3 = f_in, p4 = n_cp
//   p5 = s_in_off, p6 = v_in_off, p7 = s_cat_out_off, p8 = vu_out_off
//   p9 = wh_off (f32_params), p10 = wcp_off, p11 = wu_off
@compute @workgroup_size(64, 1, 1)
fn gvp_vec_std(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let r = wid.x;
    let tid = lid.x;
    if (r >= u.p0) {
        return;
    }
    let v_in = u.p1;
    let v_out = u.p2;
    let f_in = u.p3;
    let n_cp = u.p4;
    let dim_h = max(v_in, v_out);
    let wu_in = dim_h + n_cp;
    let two_cp = 2u * n_cp;

    let v_base = u.p6 + r * v_in * 3u;
    if (tid < v_in) {
        wg_vin[tid] = vec3<f32>(ws[v_base + tid * 3u], ws[v_base + tid * 3u + 1u], ws[v_base + tid * 3u + 2u]);
    }
    workgroupBarrier();

    if (tid < dim_h) {
        var acc = vec3<f32>(0.0, 0.0, 0.0);
        for (var vi: u32 = 0u; vi < v_in; vi = vi + 1u) {
            acc = acc + wg_vin[vi] * f32_params[u.p9 + vi * dim_h + tid];
        }
        wg_vh[tid] = acc;
    } else if (n_cp > 0u && tid < dim_h + two_cp) {
        let p = tid - dim_h;
        var acc = vec3<f32>(0.0, 0.0, 0.0);
        for (var vi: u32 = 0u; vi < v_in; vi = vi + 1u) {
            acc = acc + wg_vin[vi] * f32_params[u.p10 + vi * two_cp + p];
        }
        wg_cp[p] = acc;
    }

    let sc_cols = f_in + wu_in;
    let sc_base = u.p7 + r * sc_cols;
    let sin_base = u.p5 + r * f_in;
    for (var c: u32 = tid; c < f_in; c = c + 64u) {
        ws[sc_base + c] = ws[sin_base + c];
    }
    workgroupBarrier();

    if (tid < n_cp) {
        wg_vh[dim_h + tid] = cross(wg_cp[tid], wg_cp[n_cp + tid]);
    }
    workgroupBarrier();

    if (tid < wu_in) {
        let vv = wg_vh[tid];
        ws[sc_base + f_in + tid] = sqrt(max(dot(vv, vv), 1e-8));
    }

    if (tid < v_out) {
        var acc = vec3<f32>(0.0, 0.0, 0.0);
        for (var h: u32 = 0u; h < wu_in; h = h + 1u) {
            acc = acc + wg_vh[h] * f32_params[u.p11 + h * v_out + tid];
        }
        let vu_base = u.p8 + r * v_out * 3u;
        ws[vu_base + tid * 3u + 0u] = acc.x;
        ws[vu_base + tid * 3u + 1u] = acc.y;
        ws[vu_base + tid * 3u + 2u] = acc.z;
    }
}

// Pre-project node vectors for edge_message[0] in O(N), 1 workgroup per node:
@compute @workgroup_size(64, 1, 1)
fn gvp_em0_node_pre(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = wid.x;
    let tid = lid.x;
    if (i >= u.p0) {
        return;
    }
    let v_dim = u.p1;
    let dim_h0 = u.p2;
    let n_cp = u.p3;
    let two_cp = 2u * n_cp;
    let v_base = u.p4 + i * v_dim * 3u;

    if (tid < v_dim) {
        wg_vin[tid] = vec3<f32>(ws[v_base + tid * 3u], ws[v_base + tid * 3u + 1u], ws[v_base + tid * 3u + 2u]);
    }
    workgroupBarrier();

    if (tid < dim_h0) {
        var acc = vec3<f32>(0.0, 0.0, 0.0);
        for (var vc: u32 = 0u; vc < v_dim; vc = vc + 1u) {
            acc = acc + wg_vin[vc] * f32_params[u.p7 + vc * dim_h0 + tid];
        }
        let vh_base = u.p5 + i * dim_h0 * 3u;
        ws[vh_base + tid * 3u + 0u] = acc.x;
        ws[vh_base + tid * 3u + 1u] = acc.y;
        ws[vh_base + tid * 3u + 2u] = acc.z;
    } else if (n_cp > 0u && tid < dim_h0 + two_cp) {
        let p = tid - dim_h0;
        var acc = vec3<f32>(0.0, 0.0, 0.0);
        for (var vc: u32 = 0u; vc < v_dim; vc = vc + 1u) {
            acc = acc + wg_vin[vc] * f32_params[u.p8 + vc * two_cp + p];
        }
        let vcp_base = u.p6 + i * two_cp * 3u;
        ws[vcp_base + p * 3u + 0u] = acc.x;
        ws[vcp_base + p * 3u + 1u] = acc.y;
        ws[vcp_base + p * 3u + 2u] = acc.z;
    }
}

// Per-edge vector assembly + [rbf, e, sh] packing for edge_message[0], 1 workgroup per edge:
@compute @workgroup_size(64, 1, 1)
fn gvp_em0_edge(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let e_i = wid.x;
    let tid = lid.x;
    if (e_i >= u.p0) {
        return;
    }
    let v_dim = u.p1;
    let dim_h0 = u.p2;
    let n_cp = u.p3;
    let rbf_dim = u.p4;
    let e_dim = u.p5;
    let wu0 = dim_h0 + n_cp;
    let two_cp = 2u * n_cp;
    let s = graph_idx[e_i];

    let ud = vec3<f32>(ws[u.p6 + e_i * 3u], ws[u.p6 + e_i * 3u + 1u], ws[u.p6 + e_i * 3u + 2u]);
    let wh_u_off = u.u15;
    let wcp_u_off = wh_u_off + dim_h0;
    let wu_off = wcp_u_off + two_cp;

    if (tid < dim_h0) {
        let nvh_b = u.p9 + (s * dim_h0 + tid) * 3u;
        let nv = vec3<f32>(ws[nvh_b], ws[nvh_b + 1u], ws[nvh_b + 2u]);
        wg_vh[tid] = nv + ud * f32_params[wh_u_off + tid];
    } else if (n_cp > 0u && tid < dim_h0 + two_cp) {
        let p = tid - dim_h0;
        let nvcp_b = u.p10 + (s * two_cp + p) * 3u;
        let nc = vec3<f32>(ws[nvcp_b], ws[nvcp_b + 1u], ws[nvcp_b + 2u]);
        wg_cp[p] = nc + ud * f32_params[wcp_u_off + p];
    }

    let out_cols = rbf_dim + e_dim + wu0;
    let dst_b = u.p11 + e_i * out_cols;
    let r_b = u.p7 + e_i * rbf_dim;
    let e_b = u.p8 + e_i * e_dim;
    for (var m: u32 = tid; m < rbf_dim; m = m + 64u) {
        ws[dst_b + m] = ws[r_b + m];
    }
    for (var c: u32 = tid; c < e_dim; c = c + 64u) {
        ws[dst_b + rbf_dim + c] = ws[e_b + c];
    }
    workgroupBarrier();

    if (tid < n_cp) {
        wg_vh[dim_h0 + tid] = cross(wg_cp[tid], wg_cp[n_cp + tid]);
    }
    workgroupBarrier();

    if (tid < wu0) {
        let vv = wg_vh[tid];
        ws[dst_b + rbf_dim + e_dim + tid] = sqrt(max(dot(vv, vv), 1e-8));
    }

    if (tid < v_dim) {
        var acc = vec3<f32>(0.0, 0.0, 0.0);
        for (var hd: u32 = 0u; hd < wu0; hd = hd + 1u) {
            acc = acc + wg_vh[hd] * f32_params[wu_off + hd * v_dim + tid];
        }
        let vu_b = u.u14 + e_i * v_dim * 3u;
        ws[vu_b + tid * 3u + 0u] = acc.x;
        ws[vu_b + tid * 3u + 1u] = acc.y;
        ws[vu_b + tid * 3u + 2u] = acc.z;
    }
}

// ============================================================================
// 5. GVP Vector Gate Apply (+ optional direct coordinate update x += delta)
// ============================================================================
// p0 = total_vecs (rows * v_out), p1 = gate_off, p2 = vu_off, p3 = v_out_off
// p4 = mode (0 = sigmoid * Vu -> v_out, 1 = identity * Vu -> v_out, 2 = identity * Vu added to x_off=v_out_off)
@compute @workgroup_size(64, 1, 1)
fn gvp_gate_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.p0) {
        return;
    }
    let raw_g = ws[u.p1 + i];
    let g = select(sigmoid_f(raw_g), raw_g, u.p4 != 0u);
    let vx = ws[u.p2 + i * 3u + 0u] * g;
    let vy = ws[u.p2 + i * 3u + 1u] * g;
    let vz = ws[u.p2 + i * 3u + 2u] * g;
    if (u.p4 == 2u) {
        ws[u.p3 + i * 3u + 0u] = ws[u.p3 + i * 3u + 0u] + vx;
        ws[u.p3 + i * 3u + 1u] = ws[u.p3 + i * 3u + 1u] + vy;
        ws[u.p3 + i * 3u + 2u] = ws[u.p3 + i * 3u + 2u] + vz;
    } else {
        ws[u.p3 + i * 3u + 0u] = vx;
        ws[u.p3 + i * 3u + 1u] = vy;
        ws[u.p3 + i * 3u + 2u] = vz;
    }
}

// ============================================================================
// 6. Incoming-Edge Aggregation / Node Residual + GVPLayerNorm (Scalar LN + Vector RMS)
// ============================================================================
// p0 = n_nodes, p1 = s_dim, p2 = v_dim, p3 = n_edges
// p4 = h_off, p5 = v_off, p6 = res_s_off, p7 = res_v_off
// p8 = ln_w_off (f32_params), p9 = ln_b_off
// p10 = mode (0 = incoming-edge sum * f0 + GVPLayerNorm, 1 = node residual add + GVPLayerNorm, 2 = copy h->res_s and v->res_v)
// f0 = msg_norm_inv
var<workgroup> gln_s: array<f32, 64>;
var<workgroup> gln_v: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn agg_res_gvp_norm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let d = wid.x;
    let tid = lid.x;
    let n_nodes = u.p0;
    if (d >= n_nodes) {
        return;
    }
    let s_dim = u.p1;
    let v_dim = u.p2;
    let n_edges = u.p3;
    let h_base = u.p4 + d * s_dim;
    let v_base = u.p5 + d * v_dim * 3u;

    if (u.p10 == 2u) {
        let dst_s = u.p6 + d * s_dim;
        let dst_v = u.p7 + d * v_dim * 3u;
        for (var c: u32 = tid; c < s_dim; c = c + 64u) {
            ws[dst_s + c] = ws[h_base + c];
        }
        for (var k: u32 = tid; k < v_dim * 3u; k = k + 64u) {
            ws[dst_v + k] = ws[v_base + k];
        }
        return;
    }

    if (u.p10 == 0u) {
        let deg = select(0u, n_nodes - 1u, n_nodes > 0u);
        let inc_base = 2u * n_edges + d * deg;
        let scale = u.f0;
        for (var c: u32 = tid; c < s_dim; c = c + 64u) {
            var acc_s: f32 = 0.0;
            for (var j: u32 = 0u; j < deg; j = j + 1u) {
                let e_i = graph_idx[inc_base + j];
                acc_s = acc_s + ws[u.p6 + e_i * s_dim + c];
            }
            ws[h_base + c] = ws[h_base + c] + acc_s * scale;
        }
        for (var k: u32 = tid; k < v_dim * 3u; k = k + 64u) {
            var acc_v: f32 = 0.0;
            for (var j: u32 = 0u; j < deg; j = j + 1u) {
                let e_i = graph_idx[inc_base + j];
                acc_v = acc_v + ws[u.p7 + e_i * v_dim * 3u + k];
            }
            ws[v_base + k] = ws[v_base + k] + acc_v * scale;
        }
    } else {
        for (var c: u32 = tid; c < s_dim; c = c + 64u) {
            ws[h_base + c] = ws[h_base + c] + ws[u.p6 + d * s_dim + c];
        }
        for (var k: u32 = tid; k < v_dim * 3u; k = k + 64u) {
            ws[v_base + k] = ws[v_base + k] + ws[u.p7 + d * v_dim * 3u + k];
        }
    }
    workgroupBarrier();

    // Scalar LayerNorm
    var ls: f32 = 0.0;
    for (var c: u32 = tid; c < s_dim; c = c + 64u) {
        ls = ls + ws[h_base + c];
    }
    gln_s[tid] = ls;
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            gln_s[tid] = gln_s[tid] + gln_s[tid + s];
        }
        workgroupBarrier();
    }
    let mean = gln_s[0] / f32(s_dim);

    var lv: f32 = 0.0;
    for (var c: u32 = tid; c < s_dim; c = c + 64u) {
        let diff = ws[h_base + c] - mean;
        lv = lv + diff * diff;
    }
    gln_v[tid] = lv;
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            gln_v[tid] = gln_v[tid] + gln_v[tid + s];
        }
        workgroupBarrier();
    }
    let inv_std = 1.0 / sqrt(gln_v[0] / f32(s_dim) + 1e-5);
    for (var c: u32 = tid; c < s_dim; c = c + 64u) {
        ws[h_base + c] = (ws[h_base + c] - mean) * inv_std * f32_params[u.p8 + c] + f32_params[u.p9 + c];
    }
    workgroupBarrier();

    // Vector RMS Norm (`denom = sqrt(mean(max(||v||^2, 1e-8)) + 1e-5) + 1e-5`)
    var vn_sum: f32 = 0.0;
    for (var vc: u32 = tid; vc < v_dim; vc = vc + 64u) {
        let vx = ws[v_base + vc * 3u + 0u];
        let vy = ws[v_base + vc * 3u + 1u];
        let vz = ws[v_base + vc * 3u + 2u];
        vn_sum = vn_sum + max(vx * vx + vy * vy + vz * vz, 1e-8);
    }
    gln_s[tid] = vn_sum;
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            gln_s[tid] = gln_s[tid] + gln_s[tid + s];
        }
        workgroupBarrier();
    }
    let denom = sqrt(gln_s[0] / f32(v_dim) + 1e-5) + 1e-5;
    let inv_d = 1.0 / denom;
    for (var k: u32 = tid; k < v_dim * 3u; k = k + 64u) {
        ws[v_base + k] = ws[v_base + k] * inv_d;
    }
}

// ============================================================================
// 7. Edge Symmetrize (`e_sym[k] = e[k] + e[n_up + k]`)
// ============================================================================
// p0 = n_up * e_dim, p1 = e_off, p2 = e_sym_off
@compute @workgroup_size(64, 1, 1)
fn sym_upper_edges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.p0;
    if (idx >= total) {
        return;
    }
    ws[u.p2 + idx] = ws[u.p1 + idx] + ws[u.p1 + total + idx];
}
