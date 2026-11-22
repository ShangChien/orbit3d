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

// 126 MiB = 33,030,144 u32 words (< 128 MiB WebGPU default maxStorageBufferBindingSize; 4 chunks = 504 MiB!)
const W_CHUNK_U32: u32 = 33030144u;

@group(0) @binding(0) var<uniform> u: OpParams;
@group(0) @binding(1) var<storage, read> w_blob_0: array<u32>;
@group(0) @binding(2) var<storage, read> w_blob_1: array<u32>;
@group(0) @binding(3) var<storage, read> w_blob_2: array<u32>;
@group(0) @binding(4) var<storage, read> w_blob_3: array<u32>;
@group(0) @binding(5) var<storage, read> f32_params: array<f32>;
@group(0) @binding(6) var<storage, read_write> kv_buf: array<f32>;
@group(0) @binding(7) var<storage, read_write> ws: array<f32>;

fn read_w_u32(idx: u32) -> u32 {
    let c = idx / W_CHUNK_U32;
    let r = idx - c * W_CHUNK_U32;
    if (c == 0u) {
        return w_blob_0[r];
    } else if (c == 1u) {
        return w_blob_1[r];
    } else if (c == 2u) {
        return w_blob_2[r];
    }
    return w_blob_3[r];
}

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

fn erf_approx(x: f32) -> f32 {
    let s = select(1.0, -1.0, x < 0.0);
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return s * (1.0 - poly * exp(-ax * ax));
}

fn gelu_exact(x: f32) -> f32 {
    return 0.5 * x * (1.0 + erf_approx(x * 0.7071067811865475));
}

// ============================================================================
// 1. 64x64 (4x4 vec4 Register-Blocked) GEMM supporting F32 / F16 / Q8_0 / Q4_0
// ============================================================================
// p0 = rows, p1 = out_cols, p2 = orig_in_cols, p3 = pad_in_cols
// p4 = quant_mode (0=F32, 1=F16, 2=Q8_0, 3=Q4_0)
// p5 = act_mode (0=None, 1=SiLU, 2=GELU_exact)
// p6 = epilogue_mode (0=store, 1=add_residual_from_u14, 2=res_add_then_adaln)
// p7 = x_off (ws), p8 = y_off (ws), p9 = w_u32_off, p10 = bias_off (0xFFFFFFFF=none), p11 = blocks_per_row
// u14 = res_off (ws), u15 = adaln_row_mod_off (f32_params: per-token [scale[D], shift[D]] or per-branch)
var<workgroup> tileA: array<f32, 1088>; // 64 rows x 17 stride (BK=16)
var<workgroup> tileB: array<f32, 1088>; // 16 k x 68 cols

fn store_gemm_elem(row: u32, col: u32, raw_acc: f32) {
    let rows = u.p0;
    let out_cols = u.p1;
    if (row >= rows || col >= out_cols) {
        return;
    }
    var acc = raw_acc;
    if (u.p10 != 0xFFFFFFFFu) {
        acc = acc + f32_params[u.p10 + col];
    }
    var val = acc;
    if (u.p5 == 1u) {
        val = silu(val);
    } else if (u.p5 == 2u) {
        val = gelu_exact(val);
    }

    let epi = u.p6;
    if (epi == 1u) {
        val = ws[u.u14 + row * out_cols + col] + val;
    } else if (epi == 2u) {
        let sum_v = ws[u.u14 + row * out_cols + col] + val;
        // Per-row modulation index: `branch_idx = row / seq_q` where `seq_q` is stored in `u32(u.f0)`
        let seq_q = max(u32(u.f0), 1u);
        let b_idx = row / seq_q;
        let mod_base = u.u15 + b_idx * 2u * out_cols;
        val = sum_v * f32_params[mod_base + col] + f32_params[mod_base + out_cols + col];
    }
    ws[u.p8 + row * out_cols + col] = val;
}

@compute @workgroup_size(16, 16, 1)
fn gemm_fused(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let lx = lid.x;
    let ly = lid.y;
    let row_base = wid.y * 64u;
    let col_base = wid.x * 64u;

    let rows = u.p0;
    let out_cols = u.p1;
    let in_cols = u.p2;
    let pad_cols = u.p3;
    let qmode = u.p4;
    let w_base = u.p9;
    let bpr = u.p11;

    var c0 = vec4<f32>(0.0);
    var c1 = vec4<f32>(0.0);
    var c2 = vec4<f32>(0.0);
    var c3 = vec4<f32>(0.0);
    let n_tiles = (in_cols + 15u) / 16u;

    for (var kt: u32 = 0u; kt < n_tiles; kt = kt + 1u) {
        let k_base = kt * 16u;
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

        for (var dc: u32 = 0u; dc < 4u; dc = dc + 1u) {
            let tc = lx * 4u + dc;
            let gc = col_base + tc;
            if (qmode == 0u) {
                let kb = ly;
                if (gc < out_cols && (k_base + kb) < in_cols) {
                    tileB[kb * 68u + tc] = bitcast<f32>(read_w_u32(w_base + gc * pad_cols + k_base + kb));
                } else {
                    tileB[kb * 68u + tc] = 0.0;
                }
            } else if (qmode == 1u) {
                if (ly < 8u) {
                    let kb0 = ly * 2u;
                    let kb1 = kb0 + 1u;
                    if (gc < out_cols) {
                        let pair_idx = (gc * pad_cols + k_base + kb0) / 2u;
                        let up = unpack2x16float(read_w_u32(w_base + pair_idx));
                        tileB[kb0 * 68u + tc] = select(0.0, up.x, (k_base + kb0) < in_cols);
                        tileB[kb1 * 68u + tc] = select(0.0, up.y, (k_base + kb1) < in_cols);
                    } else {
                        tileB[kb0 * 68u + tc] = 0.0;
                        tileB[kb1 * 68u + tc] = 0.0;
                    }
                }
            } else if (qmode == 2u) {
                if (ly < 4u) {
                    let j0 = ly * 4u;
                    if (gc < out_cols) {
                        let blk_idx = kt >> 1u;
                        let half_off = (kt & 1u) * 4u;
                        let blk_off = w_base + (gc * bpr + blk_idx) * 9u;
                        let scale = bitcast<f32>(read_w_u32(blk_off));
                        let packed = bitcast<i32>(read_w_u32(blk_off + 1u + half_off + ly));
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
            } else {
                // Q4_0: 1 block = 32 weights = 5 x u32 ([0]=f32 scale, [1..4]=32 x signed 4-bit nibbles, 8 per u32)
                if (ly < 2u) {
                    let j0 = ly * 8u;
                    if (gc < out_cols) {
                        let blk_idx = kt >> 1u;
                        let half_off = (kt & 1u) * 2u;
                        let blk_off = w_base + (gc * bpr + blk_idx) * 5u;
                        let scale = bitcast<f32>(read_w_u32(blk_off));
                        let packed = bitcast<i32>(read_w_u32(blk_off + 1u + half_off + ly));
                        for (var nb: u32 = 0u; nb < 8u; nb = nb + 1u) {
                            let k_idx = j0 + nb;
                            let qv = f32(extractBits(packed, nb * 4u, 4u)) * scale;
                            tileB[k_idx * 68u + tc] = select(0.0, qv, (k_base + k_idx) < in_cols);
                        }
                    } else {
                        for (var nb: u32 = 0u; nb < 8u; nb = nb + 1u) {
                            tileB[(j0 + nb) * 68u + tc] = 0.0;
                        }
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
    store_gemm_elem(r0 + 0u, col0 + 0u, c0.x);
    store_gemm_elem(r0 + 0u, col0 + 1u, c0.y);
    store_gemm_elem(r0 + 0u, col0 + 2u, c0.z);
    store_gemm_elem(r0 + 0u, col0 + 3u, c0.w);
    store_gemm_elem(r0 + 1u, col0 + 0u, c1.x);
    store_gemm_elem(r0 + 1u, col0 + 1u, c1.y);
    store_gemm_elem(r0 + 1u, col0 + 2u, c1.z);
    store_gemm_elem(r0 + 1u, col0 + 3u, c1.w);
    store_gemm_elem(r0 + 2u, col0 + 0u, c2.x);
    store_gemm_elem(r0 + 2u, col0 + 1u, c2.y);
    store_gemm_elem(r0 + 2u, col0 + 2u, c2.z);
    store_gemm_elem(r0 + 2u, col0 + 3u, c2.w);
    store_gemm_elem(r0 + 3u, col0 + 0u, c3.x);
    store_gemm_elem(r0 + 3u, col0 + 1u, c3.y);
    store_gemm_elem(r0 + 3u, col0 + 2u, c3.z);
    store_gemm_elem(r0 + 3u, col0 + 3u, c3.w);
}

// ============================================================================
// 1b. High-Occupancy Small-Batch GEMV (`rows <= 8`: 4 cols/WG x 16 K-threads)
// ============================================================================
var<workgroup> gemv_red: array<f32, 512>; // [sub_c (4), tid (16), row (8)]

@compute @workgroup_size(16, 4, 1)
fn gemv_batched(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;   // 0..15
    let sub_c = lid.y; // 0..3
    let col = wid.x * 4u + sub_c;

    let rows = u.p0;
    let out_cols = u.p1;
    let in_cols = u.p2;
    let pad_cols = u.p3;
    let qmode = u.p4;
    let x_base = u.p7;
    let w_base = u.p9;
    let bpr = u.p11;

    var acc = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    if (col < out_cols) {
        if (qmode == 2u) {
            // Q8_0: 9 x u32 per 32-weight block
            for (var bi: u32 = tid; bi < bpr; bi = bi + 16u) {
                let boff = w_base + (col * bpr + bi) * 9u;
                let scale = bitcast<f32>(read_w_u32(boff));
                let k_blk = bi * 32u;
                for (var u_i: u32 = 0u; u_i < 8u; u_i = u_i + 1u) {
                    let k0 = k_blk + u_i * 4u;
                    if (k0 < in_cols) {
                        let pk = bitcast<i32>(read_w_u32(boff + 1u + u_i));
                        let wv = vec4<f32>(
                            select(0.0, f32(extractBits(pk, 0u, 8u)) * scale, (k0 + 0u) < in_cols),
                            select(0.0, f32(extractBits(pk, 8u, 8u)) * scale, (k0 + 1u) < in_cols),
                            select(0.0, f32(extractBits(pk, 16u, 8u)) * scale, (k0 + 2u) < in_cols),
                            select(0.0, f32(extractBits(pk, 24u, 8u)) * scale, (k0 + 3u) < in_cols),
                        );
                        for (var r: u32 = 0u; r < rows; r = r + 1u) {
                            let xb = x_base + r * in_cols + k0;
                            let xv = vec4<f32>(
                                ws[xb],
                                select(0.0, ws[xb + 1u], (k0 + 1u) < in_cols),
                                select(0.0, ws[xb + 2u], (k0 + 2u) < in_cols),
                                select(0.0, ws[xb + 3u], (k0 + 3u) < in_cols),
                            );
                            acc[r] = acc[r] + dot(xv, wv);
                        }
                    }
                }
            }
        } else if (qmode == 3u) {
            // Q4_0: 5 x u32 per 32-weight block (8 signed 4-bit nibbles per u32)
            for (var bi: u32 = tid; bi < bpr; bi = bi + 16u) {
                let boff = w_base + (col * bpr + bi) * 5u;
                let scale = bitcast<f32>(read_w_u32(boff));
                let k_blk = bi * 32u;
                for (var u_i: u32 = 0u; u_i < 4u; u_i = u_i + 1u) {
                    let k0 = k_blk + u_i * 8u;
                    if (k0 < in_cols) {
                        let pk = bitcast<i32>(read_w_u32(boff + 1u + u_i));
                        let wv0 = vec4<f32>(
                            select(0.0, f32(extractBits(pk, 0u, 4u)) * scale, (k0 + 0u) < in_cols),
                            select(0.0, f32(extractBits(pk, 4u, 4u)) * scale, (k0 + 1u) < in_cols),
                            select(0.0, f32(extractBits(pk, 8u, 4u)) * scale, (k0 + 2u) < in_cols),
                            select(0.0, f32(extractBits(pk, 12u, 4u)) * scale, (k0 + 3u) < in_cols),
                        );
                        let wv1 = vec4<f32>(
                            select(0.0, f32(extractBits(pk, 16u, 4u)) * scale, (k0 + 4u) < in_cols),
                            select(0.0, f32(extractBits(pk, 20u, 4u)) * scale, (k0 + 5u) < in_cols),
                            select(0.0, f32(extractBits(pk, 24u, 4u)) * scale, (k0 + 6u) < in_cols),
                            select(0.0, f32(extractBits(pk, 28u, 4u)) * scale, (k0 + 7u) < in_cols),
                        );
                        for (var r: u32 = 0u; r < rows; r = r + 1u) {
                            let xb = x_base + r * in_cols + k0;
                            let xv0 = vec4<f32>(
                                ws[xb],
                                select(0.0, ws[xb + 1u], (k0 + 1u) < in_cols),
                                select(0.0, ws[xb + 2u], (k0 + 2u) < in_cols),
                                select(0.0, ws[xb + 3u], (k0 + 3u) < in_cols),
                            );
                            var s_r = dot(xv0, wv0);
                            if ((k0 + 4u) < in_cols) {
                                let xv1 = vec4<f32>(
                                    ws[xb + 4u],
                                    select(0.0, ws[xb + 5u], (k0 + 5u) < in_cols),
                                    select(0.0, ws[xb + 6u], (k0 + 6u) < in_cols),
                                    select(0.0, ws[xb + 7u], (k0 + 7u) < in_cols),
                                );
                                s_r = s_r + dot(xv1, wv1);
                            }
                            acc[r] = acc[r] + s_r;
                        }
                    }
                }
            }
        } else if (qmode == 1u) {
            let n_pairs = (in_cols + 1u) / 2u;
            let row_w_pairs = (col * pad_cols) / 2u;
            for (var pi: u32 = tid; pi < n_pairs; pi = pi + 16u) {
                let k0 = pi * 2u;
                let up = unpack2x16float(read_w_u32(w_base + row_w_pairs + pi));
                let w0 = up.x;
                let w1 = select(0.0, up.y, (k0 + 1u) < in_cols);
                for (var r: u32 = 0u; r < rows; r = r + 1u) {
                    let xb = x_base + r * in_cols + k0;
                    acc[r] = acc[r] + ws[xb] * w0 + select(0.0, ws[xb + 1u], (k0 + 1u) < in_cols) * w1;
                }
            }
        } else {
            let row_w = col * pad_cols;
            for (var ki: u32 = tid; ki < in_cols; ki = ki + 16u) {
                let wv = bitcast<f32>(read_w_u32(w_base + row_w + ki));
                for (var r: u32 = 0u; r < rows; r = r + 1u) {
                    acc[r] = fma(ws[x_base + r * in_cols + ki], wv, acc[r]);
                }
            }
        }
    }

    let r_slot = (sub_c * 16u + tid) * 8u;
    for (var r: u32 = 0u; r < rows; r = r + 1u) {
        gemv_red[r_slot + r] = acc[r];
    }
    workgroupBarrier();

    for (var s: u32 = 8u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            let other = (sub_c * 16u + tid + s) * 8u;
            for (var r: u32 = 0u; r < rows; r = r + 1u) {
                gemv_red[r_slot + r] = gemv_red[r_slot + r] + gemv_red[other + r];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u && col < out_cols) {
        for (var r: u32 = 0u; r < rows; r = r + 1u) {
            store_gemm_elem(r, col, gemv_red[r_slot + r]);
        }
    }
}

// ============================================================================
// 2. Out-of-place RMSNorm (`eps = 1e-6`) + optional AdaLN-Zero Output Modulation
// ============================================================================
// p0 = rows, p1 = cols, p2 = x_in_off (ws), p3 = y_out_off (ws), p4 = w_off (f32_params)
// p5 = mode (0 = RMSNorm only, 1 = RMSNorm + per-branch AdaLN modulation for last token extraction)
// p6 = mod_off (f32_params), p7 = seq_q (when mode==1, row `b` reads `x_in[(b*seq_q + seq_q - 1)*cols]`)
var<workgroup> rms_sq: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn rmsnorm_mod(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let r = wid.x;
    let tid = lid.x;
    let rows = u.p0;
    let cols = u.p1;
    let src_row = select(r, r * u.p7 + (u.p7 - 1u), u.p5 == 1u);
    let in_b = u.p2 + src_row * cols;
    let out_b = u.p3 + r * cols;

    var local_sq: f32 = 0.0;
    if (r < rows) {
        for (var c: u32 = tid; c < cols; c = c + 64u) {
            let v = ws[in_b + c];
            local_sq = local_sq + v * v;
        }
    }
    rms_sq[tid] = local_sq;
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            rms_sq[tid] = rms_sq[tid] + rms_sq[tid + s];
        }
        workgroupBarrier();
    }
    if (r < rows) {
        let inv = inverseSqrt(rms_sq[0] / f32(cols) + 1e-6);
        if (u.p5 == 1u) {
            let mod_b = u.p6 + r * 2u * cols;
            for (var c: u32 = tid; c < cols; c = c + 64u) {
                let normed = ws[in_b + c] * inv * f32_params[u.p4 + c];
                ws[out_b + c] = normed * f32_params[mod_b + c] + f32_params[mod_b + cols + c];
            }
        } else {
            for (var c: u32 = tid; c < cols; c = c + 64u) {
                ws[out_b + c] = ws[in_b + c] * inv * f32_params[u.p4 + c];
            }
        }
    }
}

// ============================================================================
// 3. 3D Axial RoPE Rotation + Direct KV-Cache Slot Write
// ============================================================================
// p0 = total_tokens (B_run * seq_q), p1 = seq_q, p2 = num_heads, p3 = head_dim
// p4 = q_off (ws), p5 = k_off (ws), p6 = v_off (ws)
// p7 = rope_pos_off (ws: 3 floats `[px/res, py/res, pz/res]` per token)
// p8 = branch_meta_off (ws: 2 x u32 `[slot_id, prev_kv_len]` bitcast as f32 per branch)
// p9 = layer_idx, p10 = max_slots, p11 = max_kv_tokens
@compute @workgroup_size(64, 1, 1)
fn rope_kv_write(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tok = wid.x;
    let tid = lid.x; // 0..63 -> covers all pairs (head * (hd/2) + pair) in loop
    let total_toks = u.p0;
    if (tok >= total_toks) {
        return;
    }
    let seq_q = u.p1;
    let n_heads = u.p2;
    let hd = u.p3;
    let d = n_heads * hd;
    let half = hd / 2u;
    let pairs_axis = half / 3u; // 10

    let b_idx = tok / seq_q;
    let qi = tok - b_idx * seq_q;
    let slot_id = bitcast<u32>(ws[u.p8 + b_idx * 2u + 0u]);
    let prev_len = bitcast<u32>(ws[u.p8 + b_idx * 2u + 1u]);
    let kv_pos = prev_len + qi;

    let gp_b = u.p7 + tok * 3u;
    let gx = ws[gp_b + 0u];
    let gy = ws[gp_b + 1u];
    let gz = ws[gp_b + 2u];

    // KV layout: [layer (L), kind (0=K, 1=V), slot (max_slots), pos (max_kv), d (emb_dim)]
    let slot_stride = u.p11 * d;
    let kind_stride = u.p10 * slot_stride;
    let layer_stride = 2u * kind_stride;
    let k_dst = u.p9 * layer_stride + 0u * kind_stride + slot_id * slot_stride + kv_pos * d;
    let v_dst = u.p9 * layer_stride + 1u * kind_stride + slot_id * slot_stride + kv_pos * d;

    let total_pairs = n_heads * half;
    for (var idx: u32 = tid; idx < total_pairs; idx = idx + 64u) {
        let h_i = idx / half;
        let p_i = idx - h_i * half;
        var c_val: f32 = 1.0;
        var s_val: f32 = 0.0;
        if (p_i < 3u * pairs_axis) {
            let ax = p_i / pairs_axis;
            let pr = p_i - ax * pairs_axis;
            let rate = pow(10000.0, -f32(pr) / f32(pairs_axis));
            let coord = select(gz, select(gy, gx, ax == 0u), ax < 2u);
            let ang = coord * rate;
            c_val = cos(ang);
            s_val = sin(ang);
        }
        let c0 = h_i * hd + 2u * p_i;
        let c1 = c0 + 1u;

        let q0 = ws[u.p4 + tok * d + c0];
        let q1 = ws[u.p4 + tok * d + c1];
        ws[u.p4 + tok * d + c0] = q0 * c_val - q1 * s_val;
        ws[u.p4 + tok * d + c1] = q0 * s_val + q1 * c_val;

        let k0 = ws[u.p5 + tok * d + c0];
        let k1 = ws[u.p5 + tok * d + c1];
        kv_buf[k_dst + c0] = k0 * c_val - k1 * s_val;
        kv_buf[k_dst + c1] = k0 * s_val + k1 * c_val;

        kv_buf[v_dst + c0] = ws[u.p6 + tok * d + c0];
        kv_buf[v_dst + c1] = ws[u.p6 + tok * d + c1];
    }
}

// ============================================================================
// 4. Batched Causal KV-Cache Multi-Head Attention (`1` workgroup per `(tok, head)`)
// ============================================================================
// wid.x = head (0..H-1), wid.y = tok (0..total_toks-1)
// p0 = total_toks, p1 = seq_q, p2 = num_heads, p3 = head_dim (64)
// p4 = q_off (ws), p5 = attn_out_off (ws), p6 = branch_meta_off (ws)
// p7 = layer_idx, p8 = max_slots, p9 = max_kv_tokens, p10 = max_n_keys_uniform
var<workgroup> attn_dot: array<f32, 64>;
var<workgroup> q_shared: array<f32, 64>;
var<workgroup> out_acc: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn batched_kv_attn(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let head = wid.x;
    let tok = wid.y;
    let tid = lid.x;
    let valid_wg = (tok < u.p0) && (head < u.p2);
    let seq_q = u.p1;
    let n_heads = u.p2;
    let hd = u.p3;
    let d = n_heads * hd;

    var slot_id: u32 = 0u;
    var n_keys: u32 = 0u;
    if (valid_wg) {
        let b_idx = tok / seq_q;
        let qi = tok - b_idx * seq_q;
        slot_id = bitcast<u32>(ws[u.p6 + b_idx * 2u + 0u]);
        let prev_len = bitcast<u32>(ws[u.p6 + b_idx * 2u + 1u]);
        n_keys = prev_len + qi + 1u;
    }

    let slot_stride = u.p9 * d;
    let kind_stride = u.p8 * slot_stride;
    let layer_stride = 2u * kind_stride;
    let k_slot_base = u.p7 * layer_stride + 0u * kind_stride + slot_id * slot_stride;
    let v_slot_base = u.p7 * layer_stride + 1u * kind_stride + slot_id * slot_stride;

    if (valid_wg && tid < hd) {
        q_shared[tid] = ws[u.p4 + tok * d + head * hd + tid];
    } else {
        q_shared[tid] = 0.0;
    }
    out_acc[tid] = 0.0;
    workgroupBarrier();

    let scale = inverseSqrt(f32(hd));
    // W3C WGSL Uniformity: loop bound `u.p10` comes strictly from `@binding(0) var<uniform> u`!
    var m_prev: f32 = -1e30;
    var l_prev: f32 = 0.0;
    let uniform_max_keys = u.p10;

    for (var k_base: u32 = 0u; k_base < uniform_max_keys; k_base = k_base + 64u) {
        let kj = k_base + tid;
        var s_val: f32 = -1e30;
        if (valid_wg && kj < n_keys) {
            let kv_off = k_slot_base + kj * d + head * hd;
            var dot_v: f32 = 0.0;
            for (var c: u32 = 0u; c < hd; c = c + 1u) {
                dot_v = fma(q_shared[c], kv_buf[kv_off + c], dot_v);
            }
            s_val = dot_v * scale;
        }
        attn_dot[tid] = s_val;
        workgroupBarrier();

        if (valid_wg && k_base < n_keys) {
            let tile_lim = min(64u, n_keys - k_base);
            var m_new = m_prev;
            for (var j: u32 = 0u; j < tile_lim; j = j + 1u) {
                m_new = max(m_new, attn_dot[j]);
            }
            let alpha = exp(m_prev - m_new);
            var l_new = l_prev * alpha;
            var acc_c = out_acc[tid] * alpha;
            for (var j: u32 = 0u; j < tile_lim; j = j + 1u) {
                let p = exp(attn_dot[j] - m_new);
                l_new = l_new + p;
                if (tid < hd) {
                    let v_off = v_slot_base + (k_base + j) * d + head * hd + tid;
                    acc_c = fma(p, kv_buf[v_off], acc_c);
                }
            }
            if (tid < hd) {
                out_acc[tid] = acc_c;
            }
            m_prev = m_new;
            l_prev = l_new;
        }
        workgroupBarrier();
    }

    if (valid_wg && tid < hd) {
        ws[u.p5 + tok * d + head * hd + tid] = out_acc[tid] / max(l_prev, 1e-12);
    }
}

// ============================================================================
// 5. SwiGLU Elementwise Multiply (`gate[i] = silu(gate[i]) * up[i]`)
// ============================================================================
// p0 = n_elems, p1 = gate_off (ws, in-place), p2 = up_off (ws)
@compute @workgroup_size(64, 1, 1)
fn swiglu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.p0) {
        return;
    }
    ws[u.p1 + i] = silu(ws[u.p1 + i]) * ws[u.p2 + i];
}
