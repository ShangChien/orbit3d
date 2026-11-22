//! Exact `SeZM` (`DPA-4`: `Nano`, `Mini`, `Neo`, `Air`, `Plus`) forward & analytical
//! reverse-mode VJP (`F = -dE/dR`) engine with optional `wgpu` acceleration for `SO(2)`
//! edge mixing and `SO(3)` Lebedev-Haar grid quadratic operators.

use std::sync::Arc;

use mlip_core::{MolecularSystem, NeighborCache, PotentialBackend, PotentialEval};

use crate::gpu::Dpa4GpuContext;
use crate::model::Dpa4Weights;
use crate::wigner::{build_edge_wigner_bwd, build_edge_wigner_fwd};

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn silu_grad(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 + x * (1.0 - s))
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `C3CutoffEnvelope` forward & exact derivative for `p = 5` and `p = 7`.
#[inline]
fn c3_envelope(r: f32, rcut: f32, p: usize) -> (f32, f32) {
    if r >= rcut {
        return (0.0, 0.0);
    }
    let u = ((rcut - r) / rcut).clamp(0.0, 1.0);
    let x = 1.0 - u;
    let u3 = u * u * u;
    let u4 = u3 * u;
    match p {
        5 => {
            let poly = 1.0 + x * (4.0 + x * (10.0 + x * (20.0 + 35.0 * x)));
            let deriv = if u > 0.0 && u < 1.0 {
                let x2 = x * x;
                -(280.0 / rcut) * u3 * (x2 * x2)
            } else {
                0.0
            };
            (u4 * poly, deriv)
        }
        7 => {
            let poly =
                1.0 + x * (4.0 + x * (10.0 + x * (20.0 + x * (35.0 + x * (56.0 + 84.0 * x)))));
            let deriv = if u > 0.0 && u < 1.0 {
                let x2 = x * x;
                let x3 = x2 * x;
                -(840.0 / rcut) * u3 * (x3 * x3)
            } else {
                0.0
            };
            (u4 * poly, deriv)
        }
        _ => unreachable!("Unsupported SeZM C3 envelope exponent {p}"),
    }
}

/// `Y = X @ W + b` where `W` has `(in_dim, out_dim)` storage (`SeZM` convention!).
fn mm_in_out_fwd(
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        let xr = &x[r * in_dim..(r + 1) * in_dim];
        let yr = &mut y[r * out_dim..(r + 1) * out_dim];
        if let Some(bv) = b {
            yr.copy_from_slice(&bv[..out_dim]);
        }
        for i in 0..in_dim {
            let xv = xr[i];
            if xv == 0.0 {
                continue;
            }
            let wr = &w[i * out_dim..(i + 1) * out_dim];
            for o in 0..out_dim {
                yr[o] += xv * wr[o];
            }
        }
    }
    y
}

/// `dX += dY @ W^T` where `W` has `(in_dim, out_dim)` storage (4-lane SIMD-friendly unrolled dot).
fn mm_in_out_bwd_add(
    dy: &[f32],
    w: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    dx: &mut [f32],
) {
    let o4 = out_dim & !3;
    for r in 0..rows {
        let dyr = &dy[r * out_dim..(r + 1) * out_dim];
        let dxr = &mut dx[r * in_dim..(r + 1) * in_dim];
        for i in 0..in_dim {
            let wr = &w[i * out_dim..(i + 1) * out_dim];
            let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let mut o = 0usize;
            while o < o4 {
                s0 += dyr[o] * wr[o];
                s1 += dyr[o + 1] * wr[o + 1];
                s2 += dyr[o + 2] * wr[o + 2];
                s3 += dyr[o + 3] * wr[o + 3];
                o += 4;
            }
            let mut tail = 0.0f32;
            while o < out_dim {
                tail += dyr[o] * wr[o];
                o += 1;
            }
            dxr[i] += (s0 + s1) + (s2 + s3) + tail;
        }
    }
}

/// `FocusLinear`: `x: (rows, F, C_in)`, `w: (C_in, F * C_out)` -> `y: (rows, F, C_out)`.
fn focus_linear_fwd(
    x: &[f32],
    w: &[f32],
    rows: usize,
    n_focus: usize,
    in_c: usize,
    out_c: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * n_focus * out_c];
    let f_out = n_focus * out_c;
    for r in 0..rows {
        for f in 0..n_focus {
            let xr = &x[(r * n_focus + f) * in_c..(r * n_focus + f + 1) * in_c];
            let yr = &mut y[(r * n_focus + f) * out_c..(r * n_focus + f + 1) * out_c];
            for i in 0..in_c {
                let xv = xr[i];
                if xv == 0.0 {
                    continue;
                }
                let wr = &w[i * f_out + f * out_c..i * f_out + (f + 1) * out_c];
                for o in 0..out_c {
                    yr[o] += xv * wr[o];
                }
            }
        }
    }
    y
}

fn focus_linear_bwd_add(
    dy: &[f32],
    w: &[f32],
    rows: usize,
    n_focus: usize,
    in_c: usize,
    out_c: usize,
    dx: &mut [f32],
) {
    let f_out = n_focus * out_c;
    for r in 0..rows {
        for f in 0..n_focus {
            let dyr = &dy[(r * n_focus + f) * out_c..(r * n_focus + f + 1) * out_c];
            let dxr = &mut dx[(r * n_focus + f) * in_c..(r * n_focus + f + 1) * in_c];
            for i in 0..in_c {
                let wr = &w[i * f_out + f * out_c..i * f_out + (f + 1) * out_c];
                let mut acc = 0.0f32;
                for o in 0..out_c {
                    acc += dyr[o] * wr[o];
                }
                dxr[i] += acc;
            }
        }
    }
}

/// `SO3Linear` / `FrameExpand` / `FrameContract`:
/// `x: (N, D, F, in_c)` with degree-wise `w: (L+1, in_c, out_c)` shared across `F`.
fn degree_linear_fwd(
    x: &[f32],
    w: &[f32],
    degree_of_d: &[usize],
    n_atoms: usize,
    dim_d: usize,
    n_focus: usize,
    in_c: usize,
    out_c: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; n_atoms * dim_d * n_focus * out_c];
    for n in 0..n_atoms {
        for d in 0..dim_d {
            let l = degree_of_d[d];
            let wl = &w[l * in_c * out_c..(l + 1) * in_c * out_c];
            for f in 0..n_focus {
                let xr = &x[((n * dim_d + d) * n_focus + f) * in_c
                    ..((n * dim_d + d) * n_focus + f + 1) * in_c];
                let yr = &mut y[((n * dim_d + d) * n_focus + f) * out_c
                    ..((n * dim_d + d) * n_focus + f + 1) * out_c];
                for i in 0..in_c {
                    let xv = xr[i];
                    if xv == 0.0 {
                        continue;
                    }
                    let wr = &wl[i * out_c..(i + 1) * out_c];
                    for o in 0..out_c {
                        yr[o] += xv * wr[o];
                    }
                }
            }
        }
    }
    y
}

fn degree_linear_bwd_add(
    dy: &[f32],
    w: &[f32],
    degree_of_d: &[usize],
    n_atoms: usize,
    dim_d: usize,
    n_focus: usize,
    in_c: usize,
    out_c: usize,
    dx: &mut [f32],
) {
    let o4 = out_c & !3;
    for n in 0..n_atoms {
        for d in 0..dim_d {
            let l = degree_of_d[d];
            let wl = &w[l * in_c * out_c..(l + 1) * in_c * out_c];
            for f in 0..n_focus {
                let dyr = &dy[((n * dim_d + d) * n_focus + f) * out_c
                    ..((n * dim_d + d) * n_focus + f + 1) * out_c];
                let dxr = &mut dx[((n * dim_d + d) * n_focus + f) * in_c
                    ..((n * dim_d + d) * n_focus + f + 1) * in_c];
                for i in 0..in_c {
                    let wr = &wl[i * out_c..(i + 1) * out_c];
                    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    let mut o = 0usize;
                    while o < o4 {
                        s0 += dyr[o] * wr[o];
                        s1 += dyr[o + 1] * wr[o + 1];
                        s2 += dyr[o + 2] * wr[o + 2];
                        s3 += dyr[o + 3] * wr[o + 3];
                        o += 4;
                    }
                    let mut tail = 0.0f32;
                    while o < out_c {
                        tail += dyr[o] * wr[o];
                        o += 1;
                    }
                    dxr[i] += (s0 + s1) + (s2 + s3) + tail;
                }
            }
        }
    }
}

#[derive(Clone)]
struct SavedRmsNorm {
    x: Vec<f32>,
    inv_rms: Vec<f32>,
}

/// `RMSNorm` / `ScalarRMSNorm` (`eps = 1e-7`) on `(rows, C)` with per-row-group `scale` `(groups, C)`
/// where `rows = batch * groups`.
fn rmsnorm_grouped_fwd(
    x: &[f32],
    scale: &[f32],
    batch: usize,
    groups: usize,
    c: usize,
    eps: f32,
) -> (Vec<f32>, SavedRmsNorm) {
    let rows = batch * groups;
    let mut y = vec![0.0f32; rows * c];
    let mut inv_rms = vec![0.0f32; rows];
    let inv_c = 1.0 / (c as f32);
    for b in 0..batch {
        for g in 0..groups {
            let r = b * groups + g;
            let xr = &x[r * c..(r + 1) * c];
            let mut m2 = 0.0f32;
            for &v in xr {
                m2 += v * v;
            }
            let ir = 1.0 / (m2 * inv_c + eps).sqrt();
            inv_rms[r] = ir;
            let sc = &scale[g * c..(g + 1) * c];
            let yr = &mut y[r * c..(r + 1) * c];
            for i in 0..c {
                yr[i] = xr[i] * ir * sc[i];
            }
        }
    }
    (
        y,
        SavedRmsNorm {
            x: x.to_vec(),
            inv_rms,
        },
    )
}

fn rmsnorm_grouped_bwd_add(
    dy: &[f32],
    scale: &[f32],
    saved: &SavedRmsNorm,
    batch: usize,
    groups: usize,
    c: usize,
    dx: &mut [f32],
) {
    let inv_c = 1.0 / (c as f32);
    for b in 0..batch {
        for g in 0..groups {
            let r = b * groups + g;
            let ir = saved.inv_rms[r];
            let xr = &saved.x[r * c..(r + 1) * c];
            let dyr = &dy[r * c..(r + 1) * c];
            let sc = &scale[g * c..(g + 1) * c];
            let mut dot_gx = 0.0f32;
            for i in 0..c {
                dot_gx += dyr[i] * sc[i] * xr[i];
            }
            let coeff = (ir * ir * ir) * inv_c * dot_gx;
            let dxr = &mut dx[r * c..(r + 1) * c];
            for i in 0..c {
                dxr[i] += dyr[i] * sc[i] * ir - coeff * xr[i];
            }
        }
    }
}

#[derive(Clone)]
struct SavedEquivNorm {
    x_centered: Vec<f32>, // (N, D, C) with d=0 centered
    inv_rms: Vec<f32>,    // (N,)
}

/// `EquivariantRMSNorm` (`eps = 1e-5`, `n_focus = 1`) on `(N, D, C)`.
fn equiv_rmsnorm_fwd(
    x: &[f32],
    scale: &[f32],
    bias: &[f32],
    degree_of_d: &[usize],
    balance_weight: &[f32],
    n_atoms: usize,
    dim_d: usize,
    c: usize,
) -> (Vec<f32>, SavedEquivNorm) {
    let mut x_centered = x.to_vec();
    let mut y = vec![0.0f32; n_atoms * dim_d * c];
    let mut inv_rms = vec![0.0f32; n_atoms];
    let inv_c = 1.0 / (c as f32);

    for n in 0..n_atoms {
        let base0 = n * dim_d * c;
        let mut mean0 = 0.0f32;
        for ch in 0..c {
            mean0 += x_centered[base0 + ch];
        }
        mean0 *= inv_c;
        for ch in 0..c {
            x_centered[base0 + ch] -= mean0;
        }

        let mut sigma2 = 0.0f32;
        for d in 0..dim_d {
            let bw = balance_weight[d];
            let row = &x_centered[(n * dim_d + d) * c..(n * dim_d + d + 1) * c];
            let mut s_row = 0.0f32;
            for &v in row {
                s_row += v * v;
            }
            sigma2 += bw * s_row;
        }
        let ir = 1.0 / (sigma2 + 1e-5).sqrt();
        inv_rms[n] = ir;

        for d in 0..dim_d {
            let l = degree_of_d[d];
            let sc = &scale[l * c..(l + 1) * c];
            let xr = &x_centered[(n * dim_d + d) * c..(n * dim_d + d + 1) * c];
            let yr = &mut y[(n * dim_d + d) * c..(n * dim_d + d + 1) * c];
            if d == 0 {
                for ch in 0..c {
                    yr[ch] = xr[ch] * ir * sc[ch] + bias[ch];
                }
            } else {
                for ch in 0..c {
                    yr[ch] = xr[ch] * ir * sc[ch];
                }
            }
        }
    }

    (
        y,
        SavedEquivNorm {
            x_centered,
            inv_rms,
        },
    )
}

fn equiv_rmsnorm_bwd_add(
    dy: &[f32],
    scale: &[f32],
    degree_of_d: &[usize],
    balance_weight: &[f32],
    saved: &SavedEquivNorm,
    n_atoms: usize,
    dim_d: usize,
    c: usize,
    dx: &mut [f32],
) {
    let inv_c = 1.0 / (c as f32);
    for n in 0..n_atoms {
        let ir = saved.inv_rms[n];
        let mut dot_gx = 0.0f32;
        for d in 0..dim_d {
            let l = degree_of_d[d];
            let sc = &scale[l * c..(l + 1) * c];
            let dyr = &dy[(n * dim_d + d) * c..(n * dim_d + d + 1) * c];
            let xr = &saved.x_centered[(n * dim_d + d) * c..(n * dim_d + d + 1) * c];
            for ch in 0..c {
                dot_gx += dyr[ch] * sc[ch] * xr[ch];
            }
        }
        let d_sigma2 = -0.5 * (ir * ir * ir) * dot_gx;

        for d in 0..dim_d {
            let l = degree_of_d[d];
            let bw = balance_weight[d];
            let sc = &scale[l * c..(l + 1) * c];
            let dyr = &dy[(n * dim_d + d) * c..(n * dim_d + d + 1) * c];
            let xr = &saved.x_centered[(n * dim_d + d) * c..(n * dim_d + d + 1) * c];
            let dxr = &mut dx[(n * dim_d + d) * c..(n * dim_d + d + 1) * c];

            if d == 0 {
                let mut tmp0 = vec![0.0f32; c];
                let mut mean_tmp = 0.0f32;
                for ch in 0..c {
                    let v = ir * dyr[ch] * sc[ch] + 2.0 * bw * d_sigma2 * xr[ch];
                    tmp0[ch] = v;
                    mean_tmp += v;
                }
                mean_tmp *= inv_c;
                for ch in 0..c {
                    dxr[ch] += tmp0[ch] - mean_tmp;
                }
            } else {
                for ch in 0..c {
                    dxr[ch] += ir * dyr[ch] * sc[ch] + 2.0 * bw * d_sigma2 * xr[ch];
                }
            }
        }
    }
}

/// SO(3) grid projection `to_grid`: `(N, D, F, K * C_ch)` -> `(N, G, F, C_ch)`
/// where `to_grid_mat` has shape `(G, D * K)` (`K = 3`).
fn so3_to_grid(
    coeff: &[f32],
    to_grid_mat: &[f32],
    n_atoms: usize,
    dim_d: usize,
    n_focus: usize,
    n_grid: usize,
    ch: usize,
) -> Vec<f32> {
    let kd = dim_d * 3;
    let mut out = vec![0.0f32; n_atoms * n_grid * n_focus * ch];
    if n_focus == 1 {
        // Fast contiguous F=1 path: coeff is (N, 3*D, C_ch)
        for n in 0..n_atoms {
            let c_n = &coeff[n * kd * ch..(n + 1) * kd * ch];
            let o_n = &mut out[n * n_grid * ch..(n + 1) * n_grid * ch];
            for g in 0..n_grid {
                let tg = &to_grid_mat[g * kd..(g + 1) * kd];
                let og = &mut o_n[g * ch..(g + 1) * ch];
                for m in 0..kd {
                    let wv = tg[m];
                    if wv == 0.0 {
                        continue;
                    }
                    let cm = &c_n[m * ch..(m + 1) * ch];
                    for c in 0..ch {
                        og[c] += wv * cm[c];
                    }
                }
            }
        }
    } else {
        // General F > 1 path (Neo F=2): coeff is (N, D, F, 3, C_ch)
        for n in 0..n_atoms {
            for g in 0..n_grid {
                let tg = &to_grid_mat[g * kd..(g + 1) * kd];
                for d in 0..dim_d {
                    for k in 0..3 {
                        let wv = tg[d * 3 + k];
                        if wv == 0.0 {
                            continue;
                        }
                        for f in 0..n_focus {
                            let cm = &coeff[(((n * dim_d + d) * n_focus + f) * 3 + k) * ch
                                ..(((n * dim_d + d) * n_focus + f) * 3 + k + 1) * ch];
                            let og = &mut out[((n * n_grid + g) * n_focus + f) * ch
                                ..((n * n_grid + g) * n_focus + f + 1) * ch];
                            for c in 0..ch {
                                og[c] += wv * cm[c];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// VJP of `so3_to_grid`: `d_grid: (N, G, F, C_ch)` -> `d_coeff += T_to^T * d_grid` `(N, D, F, 3 * C_ch)`.
fn so3_to_grid_bwd_add(
    d_grid: &[f32],
    to_grid_mat: &[f32],
    n_atoms: usize,
    dim_d: usize,
    n_focus: usize,
    n_grid: usize,
    ch: usize,
    d_coeff: &mut [f32],
) {
    let kd = dim_d * 3;
    if n_focus == 1 {
        for n in 0..n_atoms {
            let dg_n = &d_grid[n * n_grid * ch..(n + 1) * n_grid * ch];
            let dc_n = &mut d_coeff[n * kd * ch..(n + 1) * kd * ch];
            for g in 0..n_grid {
                let tg = &to_grid_mat[g * kd..(g + 1) * kd];
                let dgg = &dg_n[g * ch..(g + 1) * ch];
                for m in 0..kd {
                    let wv = tg[m];
                    if wv == 0.0 {
                        continue;
                    }
                    let dcm = &mut dc_n[m * ch..(m + 1) * ch];
                    for c in 0..ch {
                        dcm[c] += wv * dgg[c];
                    }
                }
            }
        }
    } else {
        for n in 0..n_atoms {
            for g in 0..n_grid {
                let tg = &to_grid_mat[g * kd..(g + 1) * kd];
                for d in 0..dim_d {
                    for k in 0..3 {
                        let wv = tg[d * 3 + k];
                        if wv == 0.0 {
                            continue;
                        }
                        for f in 0..n_focus {
                            let dgg = &d_grid[((n * n_grid + g) * n_focus + f) * ch
                                ..((n * n_grid + g) * n_focus + f + 1) * ch];
                            let dcm = &mut d_coeff[(((n * dim_d + d) * n_focus + f) * 3 + k) * ch
                                ..(((n * dim_d + d) * n_focus + f) * 3 + k + 1) * ch];
                            for c in 0..ch {
                                dcm[c] += wv * dgg[c];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// SO(3) grid projection `from_grid`: `(N, G, F, C_ch)` -> `(N, D, F, 3 * C_ch)`
/// where `from_grid_mat` has shape `(3*D, G)`.
fn so3_from_grid(
    grid: &[f32],
    from_grid_mat: &[f32],
    n_atoms: usize,
    dim_d: usize,
    n_focus: usize,
    n_grid: usize,
    ch: usize,
) -> Vec<f32> {
    let kd = dim_d * 3;
    let mut out = vec![0.0f32; n_atoms * kd * n_focus * ch];
    if n_focus == 1 {
        for n in 0..n_atoms {
            let g_n = &grid[n * n_grid * ch..(n + 1) * n_grid * ch];
            let o_n = &mut out[n * kd * ch..(n + 1) * kd * ch];
            for m in 0..kd {
                let fg = &from_grid_mat[m * n_grid..(m + 1) * n_grid];
                let om = &mut o_n[m * ch..(m + 1) * ch];
                for g in 0..n_grid {
                    let wv = fg[g];
                    if wv == 0.0 {
                        continue;
                    }
                    let gg = &g_n[g * ch..(g + 1) * ch];
                    for c in 0..ch {
                        om[c] += wv * gg[c];
                    }
                }
            }
        }
    } else {
        for n in 0..n_atoms {
            for d in 0..dim_d {
                for k in 0..3 {
                    let fg = &from_grid_mat[(d * 3 + k) * n_grid..(d * 3 + k + 1) * n_grid];
                    for g in 0..n_grid {
                        let wv = fg[g];
                        if wv == 0.0 {
                            continue;
                        }
                        for f in 0..n_focus {
                            let gg = &grid[((n * n_grid + g) * n_focus + f) * ch
                                ..((n * n_grid + g) * n_focus + f + 1) * ch];
                            let om = &mut out[(((n * dim_d + d) * n_focus + f) * 3 + k) * ch
                                ..(((n * dim_d + d) * n_focus + f) * 3 + k + 1) * ch];
                            for c in 0..ch {
                                om[c] += wv * gg[c];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// VJP of `so3_from_grid`: `d_coeff: (N, D, F, 3 * C_ch)` -> `d_grid: (N, G, F, C_ch)`.
fn so3_from_grid_bwd(
    d_coeff: &[f32],
    from_grid_mat: &[f32],
    n_atoms: usize,
    dim_d: usize,
    n_focus: usize,
    n_grid: usize,
    ch: usize,
) -> Vec<f32> {
    let kd = dim_d * 3;
    let mut d_grid = vec![0.0f32; n_atoms * n_grid * n_focus * ch];
    if n_focus == 1 {
        for n in 0..n_atoms {
            let dc_n = &d_coeff[n * kd * ch..(n + 1) * kd * ch];
            let dg_n = &mut d_grid[n * n_grid * ch..(n + 1) * n_grid * ch];
            for m in 0..kd {
                let fg = &from_grid_mat[m * n_grid..(m + 1) * n_grid];
                let dcm = &dc_n[m * ch..(m + 1) * ch];
                for g in 0..n_grid {
                    let wv = fg[g];
                    if wv == 0.0 {
                        continue;
                    }
                    let dgg = &mut dg_n[g * ch..(g + 1) * ch];
                    for c in 0..ch {
                        dgg[c] += wv * dcm[c];
                    }
                }
            }
        }
    } else {
        for n in 0..n_atoms {
            for d in 0..dim_d {
                for k in 0..3 {
                    let fg = &from_grid_mat[(d * 3 + k) * n_grid..(d * 3 + k + 1) * n_grid];
                    for g in 0..n_grid {
                        let wv = fg[g];
                        if wv == 0.0 {
                            continue;
                        }
                        for f in 0..n_focus {
                            let dcm = &d_coeff[(((n * dim_d + d) * n_focus + f) * 3 + k) * ch
                                ..(((n * dim_d + d) * n_focus + f) * 3 + k + 1) * ch];
                            let dgg = &mut d_grid[((n * n_grid + g) * n_focus + f) * ch
                                ..((n * n_grid + g) * n_focus + f + 1) * ch];
                            for c in 0..ch {
                                dgg[c] += wv * dcm[c];
                            }
                        }
                    }
                }
            }
        }
    }
    d_grid
}

/// Apply `_apply_scalar_path` (`SwiGLU` has gate first, value second!) on `(N, D, F, 3, ch)`.
struct SavedScalarPath {
    s_pair: Vec<f32>, // (N, F, 2*ch)
    g_sc: Vec<f32>,   // (N, F, ch)
    u_pre: Vec<f32>,  // (N, D, F, 3*ch)
}

fn apply_scalar_path_fwd(
    u_in: &[f32],
    s_pair: &[f32],
    sc_gate_w: &[f32],
    n_atoms: usize,
    dim_d: usize,
    n_focus: usize,
    ch: usize,
) -> (Vec<f32>, SavedScalarPath) {
    let gate_logits = focus_linear_fwd(s_pair, sc_gate_w, n_atoms, n_focus, 2 * ch, ch);
    let g_sc: Vec<f32> = gate_logits.iter().map(|&z| sigmoid(z)).collect();
    let mut out = u_in.to_vec();

    for n in 0..n_atoms {
        for f in 0..n_focus {
            let sp = &s_pair[(n * n_focus + f) * 2 * ch..(n * n_focus + f + 1) * 2 * ch];
            let gs = &g_sc[(n * n_focus + f) * ch..(n * n_focus + f + 1) * ch];
            for d in 0..dim_d {
                for k in 0..3 {
                    let slice = &mut out[(((n * dim_d + d) * n_focus + f) * 3 + k) * ch
                        ..(((n * dim_d + d) * n_focus + f) * 3 + k + 1) * ch];
                    for c in 0..ch {
                        slice[c] *= gs[c];
                    }
                    if d == 0 && k == 0 {
                        // SwiGLU: gate = sp[..ch], value = sp[ch..]
                        for c in 0..ch {
                            slice[c] += silu(sp[c]) * sp[ch + c];
                        }
                    }
                }
            }
        }
    }

    (
        out,
        SavedScalarPath {
            s_pair: s_pair.to_vec(),
            g_sc,
            u_pre: u_in.to_vec(),
        },
    )
}

/// VJP of `_apply_scalar_path`: returns `(d_u_in, d_s_pair)`.
fn apply_scalar_path_bwd(
    d_out: &[f32],
    sc_gate_w: &[f32],
    saved: &SavedScalarPath,
    n_atoms: usize,
    dim_d: usize,
    n_focus: usize,
    ch: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut d_u_in = vec![0.0f32; d_out.len()];
    let mut d_gate_logits = vec![0.0f32; n_atoms * n_focus * ch];
    let mut d_s_pair = vec![0.0f32; n_atoms * n_focus * 2 * ch];

    for n in 0..n_atoms {
        for f in 0..n_focus {
            let gs = &saved.g_sc[(n * n_focus + f) * ch..(n * n_focus + f + 1) * ch];
            let dgl = &mut d_gate_logits[(n * n_focus + f) * ch..(n * n_focus + f + 1) * ch];
            let dsp = &mut d_s_pair[(n * n_focus + f) * 2 * ch..(n * n_focus + f + 1) * 2 * ch];
            let sp = &saved.s_pair[(n * n_focus + f) * 2 * ch..(n * n_focus + f + 1) * 2 * ch];

            for d in 0..dim_d {
                for k in 0..3 {
                    let idx0 = (((n * dim_d + d) * n_focus + f) * 3 + k) * ch;
                    let dout_s = &d_out[idx0..idx0 + ch];
                    let upre_s = &saved.u_pre[idx0..idx0 + ch];
                    let du_s = &mut d_u_in[idx0..idx0 + ch];

                    for c in 0..ch {
                        du_s[c] = dout_s[c] * gs[c];
                        dgl[c] += dout_s[c] * upre_s[c];
                    }
                    if d == 0 && k == 0 {
                        for c in 0..ch {
                            let g_val = sp[c];
                            let v_val = sp[ch + c];
                            dsp[c] += dout_s[c] * v_val * silu_grad(g_val);
                            dsp[ch + c] += dout_s[c] * silu(g_val);
                        }
                    }
                }
            }
            for c in 0..ch {
                dgl[c] *= gs[c] * (1.0 - gs[c]);
            }
        }
    }

    focus_linear_bwd_add(
        &d_gate_logits,
        sc_gate_w,
        n_atoms,
        n_focus,
        2 * ch,
        ch,
        &mut d_s_pair,
    );
    (d_u_in, d_s_pair)
}

struct So2MixLayerTape {
    y_lin_f: Option<Vec<f32>>, // (F, E, D_m * C) for k < L_mix - 1
    gates: Option<Vec<f32>>,   // (F, E, lmax * C) sigmoid values for k < L_mix - 1
}

struct BlockTape {
    x_pre: Vec<f32>,              // (N, D, C_wide)
    x_loc_rot: Vec<f32>,          // (E, D_m, C_wide) before radial degree mix
    r_red: Vec<f32>,              // (E, D_m, C_wide)
    rad_deg_dense: Vec<f32>,      // (E, D_m, D_m, R) if degree_channel
    g_focus_norm: Option<SavedRmsNorm>,
    focus_sm: Vec<f32>,           // (E, F) raw softmax probs (before 0.98 + 0.01) if F > 1
    focus_alpha: Vec<f32>,        // (E, F) if F > 1
    x_loc_post_mix_f: Vec<f32>,   // (F, E, D_m * C) before focus_alpha
    mix_tapes: Vec<So2MixLayerTape>,
    x_loc_back: Vec<f32>,         // (E, D_m, C_wide) before Dt_from_m
    x_msg: Vec<f32>,              // (E, D, C_wide)
    qk_norm: SavedRmsNorm,
    q: Vec<f32>,                  // (N, F, C)
    k: Vec<f32>,                  // (N, F, C)
    attn_alpha: Vec<f32>,         // (E, F)
    attn_exp_over_den: Vec<f32>,  // (E, F) exp(logit - m) / denom for smooth edge_env VJP
    m_agg: Vec<f32>,              // (N, D, F, C)
    out_gate_norm: SavedRmsNorm,
    g_out_attn: Vec<f32>,         // (N, F)
    mn_lg: Vec<f32>,              // (N, G, F, C)
    mn_rg: Vec<f32>,              // (N, G, F, C)
    mn_sc: SavedScalarPath,
    post_so2_norm: SavedEquivNorm,
    pre_ffn_norm: SavedEquivNorm,
    ffn_lg: Vec<f32>,             // (N, G, H)
    ffn_rg: Vec<f32>,             // (N, G, H)
    ffn_sc: SavedScalarPath,
}

pub struct Dpa4CpuEngine {
    pub weights: Arc<Dpa4Weights>,
    neighbor_cache: NeighborCache,
    label: String,
}

impl Dpa4CpuEngine {
    pub fn new(weights: Arc<Dpa4Weights>) -> Self {
        let rcut = weights.cfg.rcut;
        let label = format!("DPA-4 {} (CPU)", weights.cfg.tier);
        Self {
            weights,
            neighbor_cache: NeighborCache::new(rcut, 0.5),
            label,
        }
    }
}

impl PotentialBackend for Dpa4CpuEngine {
    fn model_name(&self) -> &str {
        &self.label
    }

    fn cutoff_a(&self) -> f64 {
        self.weights.cfg.rcut
    }

    fn eval(
        &mut self,
        system: &MolecularSystem,
        need_forces: bool,
        need_descriptor: bool,
    ) -> Result<PotentialEval, String> {
        let w = Arc::clone(&self.weights);
        block_ready(eval_dpa4_core(
            &w,
            &mut self.neighbor_cache,
            system,
            need_forces,
            need_descriptor,
            None,
            None,
        ))
    }
}

fn block_ready<F: std::future::Future>(fut: F) -> F::Output {
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    fn clone_w(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VT)
    }
    fn noop(_: *const ()) {}
    static VT: RawWakerVTable = RawWakerVTable::new(clone_w, noop, noop, noop);
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VT)) };
    let mut cx = Context::from_waker(&waker);
    let mut p = Box::pin(fut);
    match Pin::as_mut(&mut p).poll(&mut cx) {
        Poll::Ready(v) => v,
        Poll::Pending => unreachable!("CPU path never awaits"),
    }
}

pub(crate) async fn eval_dpa4_core(
    w: &Dpa4Weights,
    neighbor_cache: &mut NeighborCache,
    system: &MolecularSystem,
    need_forces: bool,
    need_descriptor: bool,
    gpu: Option<&Dpa4GpuContext>,
    custom_d_desc: Option<&[f32]>,
) -> Result<PotentialEval, String> {
    let need_forces = need_forces || custom_d_desc.is_some();
    let n = system.n_atoms();
    let c = w.cfg.channels;
    if n == 0 {
        return Ok(PotentialEval {
            energy_ev: 0.0,
            forces_ev_a: Vec::new(),
            descriptor: if need_descriptor {
                Some(Vec::new())
            } else {
                None
            },
            desc_dim: c,
        });
    }

    let atypes = w.map_atomic_numbers(&system.atomic_numbers)?;
    let lmax = w.cfg.lmax;
    let l1_cnt = lmax + 1;
    let dim_d = w.dim_d;
    let dim_m = w.dim_m;
    let n_focus = w.cfg.n_focus;
    let c_wide = n_focus * c;
    let h_ffn = w.cfg.ffn_dim;
    let n_grid = w.cfg.n_grid;
    let rcut = w.cfg.rcut as f32;
    let eps = w.cfg.eps;

    // 1. Neighbor list & EdgeFeatureCache (`dst = e.src`, `src = e.dst`, `v_e = r[src] - r[dst] = e.vec_a`)
    let nl = neighbor_cache.update(&system.coords_a, system.pbc_cell.as_ref());
    let n_edges = nl.edges.len();

    let mut edge_src = Vec::with_capacity(n_edges);
    let mut edge_dst = Vec::with_capacity(n_edges);
    let mut edge_v = Vec::with_capacity(n_edges);
    let mut r_safe = Vec::with_capacity(n_edges);
    let mut env_rbf = Vec::with_capacity(n_edges);
    let mut d_env_rbf_dr = Vec::with_capacity(n_edges);
    let mut edge_env = Vec::with_capacity(n_edges);
    let mut d_edge_env_dr = Vec::with_capacity(n_edges);
    let mut edge_rbf = vec![0.0f32; n_edges * 16];
    let mut phi_raw = vec![0.0f32; n_edges * 16];
    let mut d_full = vec![0.0f32; n_edges * dim_d * dim_d];
    let mut saved_quats = Vec::with_capacity(if need_forces { n_edges } else { 0 });

    let mut deg_dst = vec![0.0f32; n];

    for (e_idx, e) in nl.edges.iter().enumerate() {
        let dst = e.src as usize;
        let src = e.dst as usize;
        edge_dst.push(dst);
        edge_src.push(src);

        let v = [e.vec_a[0] as f32, e.vec_a[1] as f32, e.vec_a[2] as f32];
        edge_v.push(v);
        let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + eps * eps).sqrt();
        r_safe.push(r);

        let (e7, de7) = c3_envelope(r, rcut, w.cfg.env_exp[0]);
        let (e5, de5) = c3_envelope(r, rcut, w.cfg.env_exp[1]);
        env_rbf.push(e7);
        d_env_rbf_dr.push(de7);
        edge_env.push(e5);
        d_edge_env_dr.push(de5);
        deg_dst[dst] += e5 * e5;

        for k in 0..16 {
            let wn = w.bessel_freqs[k];
            let wr = wn * r;
            let p_val = if wr.abs() < 1e-4 {
                wn * (1.0 - (wr * wr) / 6.0)
            } else {
                wr.sin() / r
            };
            phi_raw[e_idx * 16 + k] = p_val;
            edge_rbf[e_idx * 16 + k] = p_val * e7;
        }

        let sq = build_edge_wigner_fwd(
            v,
            r,
            eps,
            w,
            &mut d_full[e_idx * dim_d * dim_d..(e_idx + 1) * dim_d * dim_d],
        );
        if need_forces {
            saved_quats.push(sq);
        }
    }

    let inv_sqrt_deg: Vec<f32> = deg_dst
        .iter()
        .map(|&dg| 1.0 / (dg + w.cfg.deg_norm_floor).sqrt())
        .collect();

    // 2. Initial Embeddings:
    // 2.1 Type + ChargeSpinEmbedding
    let q_idx = ((system.charge as i64) + 100).clamp(0, 199) as usize;
    let s_idx = (system.multiplicity as i64).clamp(0, 99) as usize;
    let mut cs_cat = vec![0.0f32; 2 * c];
    cs_cat[..c].copy_from_slice(&w.chg_embed[q_idx * c..(q_idx + 1) * c]);
    cs_cat[c..].copy_from_slice(&w.spin_embed[s_idx * c..(s_idx + 1) * c]);
    let mut cs_vec = mm_in_out_fwd(&cs_cat, &w.cs_mix_w, Some(&w.cs_mix_b), 1, 2 * c, c);
    for v in &mut cs_vec {
        *v = silu(*v);
    }

    let mut t_node = vec![0.0f32; n * c];
    for i in 0..n {
        let at = atypes[i];
        let trow = &w.type_embed[at * c..(at + 1) * c];
        for ch in 0..c {
            t_node[i * c + ch] = trow[ch] + cs_vec[ch];
        }
    }

    // 2.2 RadialMLP (`radial_embedding`) + version 1.2 `edge_env` gate
    let rad_h0 = mm_in_out_fwd(&edge_rbf, &w.rad_w0, None, n_edges, 16, c);
    let (rad_h1, saved_rad_norm) = rmsnorm_grouped_fwd(&rad_h0, &w.rad_norm_scale, n_edges, 1, c, 1e-7);
    let rad_h2: Vec<f32> = rad_h1.iter().map(|&v| silu(v)).collect();
    let rad_raw = mm_in_out_fwd(&rad_h2, &w.rad_w3, None, n_edges, c, l1_cnt * c);
    let mut radial_feat0 = vec![0.0f32; n_edges * l1_cnt * c];
    for e_idx in 0..n_edges {
        let env_v = edge_env[e_idx];
        for idx in 0..(l1_cnt * c) {
            radial_feat0[e_idx * l1_cnt * c + idx] = rad_raw[e_idx * l1_cnt * c + idx] * env_v;
        }
    }

    // 2.3 EnvironmentInitialEmbedding (`l = 0` FiLM)
    let d_emb = c.min(128);
    let d_type = (c / 4).clamp(8, 32);
    let d_axis = if d_emb < 64 { 4 } else { 8 };
    let d_g_in = 32 + 2 * d_type;
    let d_hid = (2 * d_emb).max(d_g_in).min(256);

    let env_p1_pre = mm_in_out_fwd(&edge_rbf, &w.env_rbf_w1, None, n_edges, 16, 32);
    let env_p1_act: Vec<f32> = env_p1_pre.iter().map(|&v| silu(v)).collect();
    let env_p2 = mm_in_out_fwd(&env_p1_act, &w.env_rbf_w2, None, n_edges, 32, 32);

    let mut env_g_in = vec![0.0f32; n_edges * d_g_in];
    for e_idx in 0..n_edges {
        let row = &mut env_g_in[e_idx * d_g_in..(e_idx + 1) * d_g_in];
        row[..32].copy_from_slice(&env_p2[e_idx * 32..(e_idx + 1) * 32]);
        let ts = atypes[edge_src[e_idx]];
        let td = atypes[edge_dst[e_idx]];
        row[32..32 + d_type].copy_from_slice(&w.env_type_embed[ts * d_type..(ts + 1) * d_type]);
        row[32 + d_type..]
            .copy_from_slice(&w.env_type_embed[td * d_type..(td + 1) * d_type]);
    }
    let env_g1_pre = mm_in_out_fwd(&env_g_in, &w.env_g_w1, None, n_edges, d_g_in, d_hid);
    let env_g1_act: Vec<f32> = env_g1_pre.iter().map(|&v| silu(v)).collect();
    let env_g = mm_in_out_fwd(&env_g1_act, &w.env_g_w2, None, n_edges, d_hid, d_emb);

    let mut r_tilde = vec![[0.0f32; 4]; n_edges];
    let mut env_m = vec![0.0f32; n * 4 * d_emb];
    for e_idx in 0..n_edges {
        let inv_r = 1.0 / r_safe[e_idx];
        let s_e = edge_env[e_idx] * inv_r;
        let v = edge_v[e_idx];
        let rt = [s_e, s_e * v[0] * inv_r, s_e * v[1] * inv_r, s_e * v[2] * inv_r];
        r_tilde[e_idx] = rt;

        let dst = edge_dst[e_idx];
        let g_row = &env_g[e_idx * d_emb..(e_idx + 1) * d_emb];
        for a in 0..4 {
            let m_row = &mut env_m[(dst * 4 + a) * d_emb..(dst * 4 + a + 1) * d_emb];
            for k in 0..d_emb {
                m_row[k] += rt[a] * g_row[k];
            }
        }
    }
    for i in 0..n {
        let isd = inv_sqrt_deg[i];
        for idx in 0..(4 * d_emb) {
            env_m[i * 4 * d_emb + idx] *= isd;
        }
    }

    // D_i = M_i^T * M_i[:, :d_axis] -> (N, d_emb, d_axis)
    let mut env_gram = vec![0.0f32; n * d_emb * d_axis];
    for i in 0..n {
        let mi = &env_m[i * 4 * d_emb..(i + 1) * 4 * d_emb];
        let di = &mut env_gram[i * d_emb * d_axis..(i + 1) * d_emb * d_axis];
        for a in 0..4 {
            let m_a = &mi[a * d_emb..(a + 1) * d_emb];
            for k in 0..d_emb {
                let v_k = m_a[k];
                for ax in 0..d_axis {
                    di[k * d_axis + ax] += v_k * m_a[ax];
                }
            }
        }
    }
    let film_raw = mm_in_out_fwd(&env_gram, &w.env_out_proj, None, n, d_emb * d_axis, 2 * c);
    let mut gamma_raw = vec![0.0f32; n * c];
    let mut beta_raw = vec![0.0f32; n * c];
    for i in 0..n {
        gamma_raw[i * c..(i + 1) * c].copy_from_slice(&film_raw[i * 2 * c..i * 2 * c + c]);
        beta_raw[i * c..(i + 1) * c].copy_from_slice(&film_raw[i * 2 * c + c..(i + 1) * 2 * c]);
    }
    let (gamma_hat, saved_gamma_norm) =
        rmsnorm_grouped_fwd(&gamma_raw, &w.film_scale_norm, n, 1, c, 1e-7);
    let (beta_hat, saved_beta_norm) =
        rmsnorm_grouped_fwd(&beta_raw, &w.film_shift_norm, n, 1, c, 1e-7);

    // 2.4 GeometricInitialEmbedding (`GIE`) & Initial State `x: (N, D, C)`
    let mut x_cur = vec![0.0f32; n * dim_d * c];
    for i in 0..n {
        for ch in 0..c {
            let idx = i * c + ch;
            x_cur[i * dim_d * c + ch] = t_node[idx]
                * (1.0 + w.film_scale_strength * gamma_hat[idx].tanh())
                + w.film_shift_strength * beta_hat[idx].tanh();
        }
    }

    // GIE: x[dst, d, c] = inv_sqrt_deg[dst] * sum_{e: dst(e)=i} D_full[e, l*(l+1), d] * radial_feat0[e, l, c]
    for e_idx in 0..n_edges {
        let dst = edge_dst[e_idx];
        let df_e = &d_full[e_idx * dim_d * dim_d..(e_idx + 1) * dim_d * dim_d];
        let rf0_e = &radial_feat0[e_idx * l1_cnt * c..(e_idx + 1) * l1_cnt * c];
        for d in 1..dim_d {
            let l = w.degree_of_d[d];
            let z_ed = df_e[l * (l + 1) * dim_d + d];
            let dst_row = &mut x_cur[(dst * dim_d + d) * c..(dst * dim_d + d + 1) * c];
            let rf_l = &rf0_e[l * c..(l + 1) * c];
            for ch in 0..c {
                dst_row[ch] += z_ed * rf_l[ch];
            }
        }
    }
    for i in 0..n {
        let isd = inv_sqrt_deg[i];
        for d in 1..dim_d {
            let row = &mut x_cur[(i * dim_d + d) * c..(i * dim_d + d + 1) * c];
            for ch in 0..c {
                row[ch] *= isd;
            }
        }
    }

    // Fuse `edge_type_feat` into `radial_feat` `(E, L+1, C)` for all interaction blocks
    let mut radial_feat = radial_feat0.clone();
    for e_idx in 0..n_edges {
        let ts = &t_node[edge_src[e_idx] * c..(edge_src[e_idx] + 1) * c];
        let td = &t_node[edge_dst[e_idx] * c..(edge_dst[e_idx] + 1) * c];
        for l in 0..l1_cnt {
            let row = &mut radial_feat[(e_idx * l1_cnt + l) * c..(e_idx * l1_cnt + l + 1) * c];
            for ch in 0..c {
                row[ch] += ts[ch] + td[ch];
            }
        }
    }

    // Precompute `D_to_m` `(E, D_m, D)`
    let mut d_to_m = vec![0.0f32; n_edges * dim_m * dim_d];
    for e_idx in 0..n_edges {
        let df_e = &d_full[e_idx * dim_d * dim_d..(e_idx + 1) * dim_d * dim_d];
        for (m_pos, &coeff_r) in w.coeff_index_m.iter().enumerate() {
            d_to_m[(e_idx * dim_m + m_pos) * dim_d..(e_idx * dim_m + m_pos + 1) * dim_d]
                .copy_from_slice(&df_e[coeff_r * dim_d..(coeff_r + 1) * dim_d]);
        }
    }

    // 3. SeZM Interaction Blocks `b = 0 .. n_blocks - 1`
    let mut block_tapes = Vec::with_capacity(if need_forces { w.cfg.n_blocks } else { 0 });

    for (b_idx, blk) in w.blocks.iter().enumerate() {
        // 3.2.1 pre_focus_mix: (N, D, C) -> (N, D, C_wide)
        let x_pre = degree_linear_fwd(
            &x_cur,
            &blk.pre_focus_w,
            &w.degree_of_d,
            n,
            dim_d,
            1,
            c,
            c_wide,
        );

        // 3.2.2 Rotate global -> reduced m-major: x_loc_rot[e] = D_to_m[e] @ x_pre[src(e)]
        let mut x_loc_rot = vec![0.0f32; n_edges * dim_m * c_wide];
        for e_idx in 0..n_edges {
            let xp_s = &x_pre[edge_src[e_idx] * dim_d * c_wide..(edge_src[e_idx] + 1) * dim_d * c_wide];
            let dtm_e = &d_to_m[e_idx * dim_m * dim_d..(e_idx + 1) * dim_m * dim_d];
            let xl_e = &mut x_loc_rot[e_idx * dim_m * c_wide..(e_idx + 1) * dim_m * c_wide];
            for m_pos in 0..dim_m {
                let d_row = &dtm_e[m_pos * dim_d..(m_pos + 1) * dim_d];
                let xl_row = &mut xl_e[m_pos * c_wide..(m_pos + 1) * c_wide];
                for d in 0..dim_d {
                    let dv = d_row[d];
                    if dv == 0.0 {
                        continue;
                    }
                    let xp_d = &xp_s[d * c_wide..(d + 1) * c_wide];
                    for ch in 0..c_wide {
                        xl_row[ch] += dv * xp_d[ch];
                    }
                }
            }
        }

        // 3.2.3 Radial projection & DynamicRadialDegreeMixer
        let mut r_red_c = vec![0.0f32; n_edges * dim_m * c];
        for e_idx in 0..n_edges {
            for (m_pos, &deg_l) in w.degree_index_m.iter().enumerate() {
                r_red_c[(e_idx * dim_m + m_pos) * c..(e_idx * dim_m + m_pos + 1) * c]
                    .copy_from_slice(
                        &radial_feat[(e_idx * l1_cnt + deg_l) * c..(e_idx * l1_cnt + deg_l + 1) * c],
                    );
            }
        }
        let r_red = if let Some(ref hid_w) = blk.rad_hid_proj_w {
            mm_in_out_fwd(&r_red_c, hid_w, None, n_edges * dim_m, c, c_wide)
        } else {
            r_red_c
        };

        let mut rad_deg_dense = Vec::new();
        let x_loc_rad = if w.cfg.radial_so2_mode == "none" {
            x_loc_rot
                .iter()
                .zip(r_red.iter())
                .map(|(&a, &b)| a * b)
                .collect::<Vec<f32>>()
        } else {
            // degree_channel with rank R
            let rank = w.cfg.radial_so2_rank;
            let k_deg = l1_cnt * l1_cnt + lmax * lmax;
            let rad_w = blk.rad_deg_w.as_ref().unwrap();
            let basis = blk.rad_deg_basis.as_ref().unwrap(); // (R, C_wide)

            let mut r_m0 = vec![0.0f32; n_edges * l1_cnt * c_wide];
            for e_idx in 0..n_edges {
                r_m0[e_idx * l1_cnt * c_wide..(e_idx + 1) * l1_cnt * c_wide]
                    .copy_from_slice(&r_red[e_idx * dim_m * c_wide..(e_idx * dim_m + l1_cnt) * c_wide]);
            }
            let compact = mm_in_out_fwd(&r_m0, rad_w, None, n_edges, l1_cnt * c_wide, k_deg * rank);

            // Scatter compact (E, K_deg, R) -> dense (E, D_m_out, D_m_in, R)
            rad_deg_dense = vec![0.0f32; n_edges * dim_m * dim_m * rank];
            for e_idx in 0..n_edges {
                let comp_e = &compact[e_idx * k_deg * rank..(e_idx + 1) * k_deg * rank];
                let den_e =
                    &mut rad_deg_dense[e_idx * dim_m * dim_m * rank..(e_idx + 1) * dim_m * dim_m * rank];
                // m = 0 block
                for l_in in 0..l1_cnt {
                    for l_out in 0..l1_cnt {
                        let c_off = (l_in * l1_cnt + l_out) * rank;
                        let d_off = (l_out * dim_m + l_in) * rank;
                        den_e[d_off..d_off + rank].copy_from_slice(&comp_e[c_off..c_off + rank]);
                    }
                }
                // |m| = 1 neg & pos blocks
                let c_base1 = l1_cnt * l1_cnt;
                let neg_s = l1_cnt;
                let pos_s = l1_cnt + lmax;
                for l_in in 0..lmax {
                    for l_out in 0..lmax {
                        let c_off = (c_base1 + l_in * lmax + l_out) * rank;
                        let d_neg = ((neg_s + l_out) * dim_m + (neg_s + l_in)) * rank;
                        let d_pos = ((pos_s + l_out) * dim_m + (pos_s + l_in)) * rank;
                        den_e[d_neg..d_neg + rank].copy_from_slice(&comp_e[c_off..c_off + rank]);
                        den_e[d_pos..d_pos + rank].copy_from_slice(&comp_e[c_off..c_off + rank]);
                    }
                }
            }

            let mut mixed_out = vec![0.0f32; n_edges * dim_m * c_wide];
            for e_idx in 0..n_edges {
                let den_e =
                    &rad_deg_dense[e_idx * dim_m * dim_m * rank..(e_idx + 1) * dim_m * dim_m * rank];
                let xl_e = &x_loc_rot[e_idx * dim_m * c_wide..(e_idx + 1) * dim_m * c_wide];
                let out_e = &mut mixed_out[e_idx * dim_m * c_wide..(e_idx + 1) * dim_m * c_wide];
                for o in 0..dim_m {
                    for i in 0..dim_m {
                        for r in 0..rank {
                            let kv = den_e[(o * dim_m + i) * rank + r];
                            if kv == 0.0 {
                                continue;
                            }
                            let b_row = &basis[r * c_wide..(r + 1) * c_wide];
                            let xl_i = &xl_e[i * c_wide..(i + 1) * c_wide];
                            let out_o = &mut out_e[o * c_wide..(o + 1) * c_wide];
                            for ch in 0..c_wide {
                                out_o[ch] += kv * b_row[ch] * xl_i[ch];
                            }
                        }
                    }
                }
            }
            mixed_out
        };

        // 3.2.4 Permute `(E, D_m, F, C)` -> focus-major `(F, E, D_m * C)`
        let dmc = dim_m * c;
        let mut x_loc_f = vec![0.0f32; n_focus * n_edges * dmc];
        for e_idx in 0..n_edges {
            for m_pos in 0..dim_m {
                for f in 0..n_focus {
                    let src_off = ((e_idx * dim_m + m_pos) * n_focus + f) * c;
                    let dst_off = (f * n_edges + e_idx) * dmc + m_pos * c;
                    x_loc_f[dst_off..dst_off + c]
                        .copy_from_slice(&x_loc_rad[src_off..src_off + c]);
                }
            }
        }

        // Save pre-mixing scalar slice for F > 1 (`Neo` focus competition): shape (E, F, C)
        let mut g_focus_pre = if n_focus > 1 {
            let mut gf = vec![0.0f32; n_edges * n_focus * c];
            for e_idx in 0..n_edges {
                for f in 0..n_focus {
                    let src_off = (f * n_edges + e_idx) * dmc;
                    let dst_off = (e_idx * n_focus + f) * c;
                    gf[dst_off..dst_off + c].copy_from_slice(&x_loc_f[src_off..src_off + c]);
                }
            }
            gf
        } else {
            Vec::new()
        };

        // 3.2.5 SO(2) Mixing Stack (`k = 0 .. mixing_layers - 1`)
        let w0 = l1_cnt * c;
        let two_w1 = 2 * lmax * c;
        let mut mix_tapes = Vec::with_capacity(if need_forces { w.cfg.mixing_layers } else { 0 });

        for (l_idx, sl) in blk.so2_layers.iter().enumerate() {
            let mut y_f = vec![0.0f32; n_focus * n_edges * dmc];
            if let Some(g) = gpu {
                let mut xm0 = vec![0.0f32; n_focus * n_edges * w0];
                let mut xm1 = vec![0.0f32; n_focus * n_edges * two_w1];
                for f in 0..n_focus {
                    for e_idx in 0..n_edges {
                        let row = &x_loc_f[(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];
                        xm0[(f * n_edges + e_idx) * w0..(f * n_edges + e_idx + 1) * w0]
                            .copy_from_slice(&row[..w0]);
                        if two_w1 > 0 {
                            xm1[(f * n_edges + e_idx) * two_w1..(f * n_edges + e_idx + 1) * two_w1]
                                .copy_from_slice(&row[w0..]);
                        }
                    }
                }
                let (ym0, ym1) = g
                    .run_so2_linear_pair(
                        b_idx, l_idx, &xm0, &xm1, n_focus, n_edges, w0, two_w1, false,
                    )
                    .await?;
                for f in 0..n_focus {
                    for e_idx in 0..n_edges {
                        let dst_r = &mut y_f[(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];
                        dst_r[..w0].copy_from_slice(
                            &ym0[(f * n_edges + e_idx) * w0..(f * n_edges + e_idx + 1) * w0],
                        );
                        if two_w1 > 0 {
                            dst_r[w0..].copy_from_slice(
                                &ym1[(f * n_edges + e_idx) * two_w1..(f * n_edges + e_idx + 1) * two_w1],
                            );
                        }
                    }
                }
            } else {
                for f in 0..n_focus {
                    let wm0 = &sl.w_m0_f[f * w0 * w0..(f + 1) * w0 * w0];
                    let wm1 = &sl.w_m1_f[f * two_w1 * two_w1..(f + 1) * two_w1 * two_w1];
                    for e_idx in 0..n_edges {
                        let xr = &x_loc_f[(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];
                        let yr = &mut y_f[(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];
                        // m = 0 block
                        for i in 0..w0 {
                            let xv = xr[i];
                            if xv != 0.0 {
                                let wr = &wm0[i * w0..(i + 1) * w0];
                                for o in 0..w0 {
                                    yr[o] += xv * wr[o];
                                }
                            }
                        }
                        // |m| = 1 block
                        let xr1 = &xr[w0..];
                        let yr1 = &mut yr[w0..];
                        for i in 0..two_w1 {
                            let xv = xr1[i];
                            if xv != 0.0 {
                                let wr = &wm1[i * two_w1..(i + 1) * two_w1];
                                for o in 0..two_w1 {
                                    yr1[o] += xv * wr[o];
                                }
                            }
                        }
                    }
                }
            }

            if let Some(ref gw) = sl.gate_w {
                // GatedActivation (`layout = "fndc"`): gw has shape (C, F * lmax * C)
                let lc = lmax * c;
                let f_lc = n_focus * lc;
                let mut gates = vec![0.0f32; n_focus * n_edges * lc];
                let mut z_f = vec![0.0f32; y_f.len()];

                for f in 0..n_focus {
                    for e_idx in 0..n_edges {
                        let yr = &y_f[(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];
                        let zr = &mut z_f[(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];
                        let gr = &mut gates[(f * n_edges + e_idx) * lc..(f * n_edges + e_idx + 1) * lc];

                        // d = 0: SiLU(y0)
                        for ch in 0..c {
                            zr[ch] = silu(yr[ch]);
                        }
                        // Gate logits from y0
                        for i in 0..c {
                            let yv = yr[i];
                            if yv != 0.0 {
                                let w_row = &gw[i * f_lc + f * lc..i * f_lc + (f + 1) * lc];
                                for o in 0..lc {
                                    gr[o] += yv * w_row[o];
                                }
                            }
                        }
                        for v in gr.iter_mut() {
                            *v = sigmoid(*v);
                        }
                        for m_pos in 1..dim_m {
                            let l_slot = w.degree_index_m[m_pos] - 1;
                            let g_l = &gr[l_slot * c..(l_slot + 1) * c];
                            for ch in 0..c {
                                zr[m_pos * c + ch] = yr[m_pos * c + ch] * g_l[ch];
                            }
                        }
                    }
                }

                for idx in 0..x_loc_f.len() {
                    x_loc_f[idx] += z_f[idx];
                }
                if need_forces {
                    mix_tapes.push(So2MixLayerTape {
                        y_lin_f: Some(y_f),
                        gates: Some(gates),
                    });
                }
            } else {
                for idx in 0..x_loc_f.len() {
                    x_loc_f[idx] += y_f[idx];
                }
                if need_forces {
                    mix_tapes.push(So2MixLayerTape {
                        y_lin_f: None,
                        gates: None,
                    });
                }
            }
        }

        // 3.2.6 Cross-focus competition (F > 1, i.e., `Neo`)
        let x_loc_post_mix_f = if need_forces && n_focus > 1 {
            x_loc_f.clone()
        } else {
            Vec::new()
        };
        let mut g_focus_norm = None;
        let mut focus_sm = Vec::new();
        let mut focus_alpha = Vec::new();
        if n_focus > 1 {
            let fc_norm = blk.focus_comp_norm.as_ref().unwrap();
            let fc_w = blk.focus_comp_w.as_ref().unwrap(); // (C, F)
            let (g_hat, saved_fn) =
                rmsnorm_grouped_fwd(&g_focus_pre, fc_norm, n_edges, n_focus, c, 1e-7);
            g_focus_norm = Some(saved_fn);
            g_focus_pre.clear();

            focus_sm = vec![0.0f32; n_edges * n_focus];
            focus_alpha = vec![0.0f32; n_edges * n_focus];
            for e_idx in 0..n_edges {
                let mut logits = vec![0.0f32; n_focus];
                let mut max_l = f32::NEG_INFINITY;
                for f in 0..n_focus {
                    let gh = &g_hat[(e_idx * n_focus + f) * c..(e_idx * n_focus + f + 1) * c];
                    let mut dot = 0.0f32;
                    for ch in 0..c {
                        dot += gh[ch] * fc_w[ch * n_focus + f];
                    }
                    logits[f] = dot;
                    if dot > max_l {
                        max_l = dot;
                    }
                }
                let mut sum_exp = 0.0f32;
                for f in 0..n_focus {
                    logits[f] = (logits[f] - max_l).exp();
                    sum_exp += logits[f];
                }
                for f in 0..n_focus {
                    let sm = logits[f] / sum_exp;
                    let alpha = 0.98 * sm + (0.02 / (n_focus as f32));
                    focus_sm[e_idx * n_focus + f] = sm;
                    focus_alpha[e_idx * n_focus + f] = alpha;
                    let row = &mut x_loc_f[(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];
                    for v in row.iter_mut() {
                        *v *= alpha;
                    }
                }
            }
        }

        // 3.2.7 Rotate back `(F, E, D_m, C)` -> `(E, D_m, C_wide)` -> `Dt_from_m` -> `(E, D, C_wide)`
        let mut x_loc_back = vec![0.0f32; n_edges * dim_m * c_wide];
        for e_idx in 0..n_edges {
            for m_pos in 0..dim_m {
                for f in 0..n_focus {
                    let src_off = (f * n_edges + e_idx) * dmc + m_pos * c;
                    let dst_off = ((e_idx * dim_m + m_pos) * n_focus + f) * c;
                    x_loc_back[dst_off..dst_off + c]
                        .copy_from_slice(&x_loc_f[src_off..src_off + c]);
                }
            }
        }

        let mut x_msg = vec![0.0f32; n_edges * dim_d * c_wide];
        for e_idx in 0..n_edges {
            let dtm_e = &d_to_m[e_idx * dim_m * dim_d..(e_idx + 1) * dim_m * dim_d];
            let xlb_e = &x_loc_back[e_idx * dim_m * c_wide..(e_idx + 1) * dim_m * c_wide];
            let xm_e = &mut x_msg[e_idx * dim_d * c_wide..(e_idx + 1) * dim_d * c_wide];
            for m_pos in 0..dim_m {
                let d_row = &dtm_e[m_pos * dim_d..(m_pos + 1) * dim_d];
                let xlb_m = &xlb_e[m_pos * c_wide..(m_pos + 1) * c_wide];
                for d in 0..dim_d {
                    let dv = d_row[d] * w.rotate_inv_rescale[d];
                    if dv == 0.0 {
                        continue;
                    }
                    let xm_d = &mut xm_e[d * c_wide..(d + 1) * c_wide];
                    for ch in 0..c_wide {
                        xm_d[ch] += dv * xlb_m[ch];
                    }
                }
            }
        }

        // 3.2.8 Envelope-Gated Attention Aggregation (`n_atten_head = 1`)
        let mut x_l0 = vec![0.0f32; n * c_wide];
        for i in 0..n {
            x_l0[i * c_wide..(i + 1) * c_wide]
                .copy_from_slice(&x_pre[i * dim_d * c_wide..i * dim_d * c_wide + c_wide]);
        }
        let (u_qk, qk_norm) = rmsnorm_grouped_fwd(&x_l0, &blk.attn_qk_norm, n, n_focus, c, 1e-7);
        let q = focus_linear_fwd(&u_qk, &blk.attn_q_w, n, n_focus, c, c);
        let k = focus_linear_fwd(&u_qk, &blk.attn_k_w, n, n_focus, c, c);

        let inv_sqrt_c = 1.0 / (c as f32).sqrt();
        let mut logits = vec![0.0f32; n_edges * n_focus];
        let mut max_shift = vec![0.0f32; n * n_focus];
        for i in 0..n {
            for f in 0..n_focus {
                max_shift[i * n_focus + f] = blk.attn_zeta[f].ln();
            }
        }

        for e_idx in 0..n_edges {
            let dst = edge_dst[e_idx];
            let src = edge_src[e_idx];
            let r_l0_e = &r_red[e_idx * dim_m * c_wide..e_idx * dim_m * c_wide + c_wide];
            let env_v = edge_env[e_idx];
            for f in 0..n_focus {
                let q_df = &q[(dst * n_focus + f) * c..(dst * n_focus + f + 1) * c];
                let k_sf = &k[(src * n_focus + f) * c..(src * n_focus + f + 1) * c];
                let r_ef = &r_l0_e[f * c..(f + 1) * c];
                let mut qk_dot = 0.0f32;
                let mut rb_dot = 0.0f32;
                for ch in 0..c {
                    qk_dot += q_df[ch] * k_sf[ch];
                    rb_dot += r_ef[ch] * blk.attn_logit_w[ch * n_focus + f];
                }
                let lgt = qk_dot * inv_sqrt_c + rb_dot;
                logits[e_idx * n_focus + f] = lgt;
                if env_v > 0.0 {
                    let cand = lgt + 2.0 * env_v.ln();
                    if cand > max_shift[dst * n_focus + f] {
                        max_shift[dst * n_focus + f] = cand;
                    }
                }
            }
        }

        let mut denom = vec![0.0f32; n * n_focus];
        for i in 0..n {
            for f in 0..n_focus {
                denom[i * n_focus + f] =
                    (blk.attn_zeta[f].ln() - max_shift[i * n_focus + f]).exp();
            }
        }
        let mut exp_shifted = vec![0.0f32; n_edges * n_focus];
        for e_idx in 0..n_edges {
            let dst = edge_dst[e_idx];
            let env2 = edge_env[e_idx] * edge_env[e_idx];
            for f in 0..n_focus {
                let es = (logits[e_idx * n_focus + f] - max_shift[dst * n_focus + f]).exp();
                exp_shifted[e_idx * n_focus + f] = es;
                denom[dst * n_focus + f] += env2 * es;
            }
        }

        let mut attn_alpha = vec![0.0f32; n_edges * n_focus];
        let mut attn_exp_over_den = vec![0.0f32; n_edges * n_focus];
        let mut m_agg = vec![0.0f32; n * dim_d * c_wide];
        for e_idx in 0..n_edges {
            let dst = edge_dst[e_idx];
            let env2 = edge_env[e_idx] * edge_env[e_idx];
            for f in 0..n_focus {
                let den = denom[dst * n_focus + f];
                let eod = exp_shifted[e_idx * n_focus + f] / den;
                let al = env2 * eod;
                attn_exp_over_den[e_idx * n_focus + f] = eod;
                attn_alpha[e_idx * n_focus + f] = al;

                for d in 0..dim_d {
                    let xm_df =
                        &x_msg[(e_idx * dim_d + d) * c_wide + f * c..(e_idx * dim_d + d) * c_wide + (f + 1) * c];
                    let ma_df =
                        &mut m_agg[(dst * dim_d + d) * c_wide + f * c..(dst * dim_d + d) * c_wide + (f + 1) * c];
                    for ch in 0..c {
                        ma_df[ch] += al * xm_df[ch];
                    }
                }
            }
        }

        let (u_gate, out_gate_norm) =
            rmsnorm_grouped_fwd(&x_l0, &blk.attn_gate_norm, n, n_focus, c, 1e-7);
        let mut g_out_attn = vec![0.0f32; n * n_focus];
        let mut attn_out = vec![0.0f32; n * dim_d * c_wide];
        for i in 0..n {
            for f in 0..n_focus {
                let ug = &u_gate[(i * n_focus + f) * c..(i * n_focus + f + 1) * c];
                let mut dot = 0.0f32;
                for ch in 0..c {
                    dot += ug[ch] * blk.attn_gate_w[ch * n_focus + f];
                }
                let go = sigmoid(dot);
                g_out_attn[i * n_focus + f] = go;
                for d in 0..dim_d {
                    let src_s = &m_agg[(i * dim_d + d) * c_wide + f * c..(i * dim_d + d) * c_wide + (f + 1) * c];
                    let dst_s =
                        &mut attn_out[(i * dim_d + d) * c_wide + f * c..(i * dim_d + d) * c_wide + (f + 1) * c];
                    for ch in 0..c {
                        dst_s[ch] = src_s[ch] * go;
                    }
                }
            }
        }

        // 3.2.9 `message_node_grid_product` (`mode = "cross"`, `query = attn_out`, `context = x_pre`)
        let mut mn_spair = vec![0.0f32; n * n_focus * 2 * c];
        for i in 0..n {
            for f in 0..n_focus {
                let sp = &mut mn_spair[(i * n_focus + f) * 2 * c..(i * n_focus + f + 1) * 2 * c];
                sp[..c].copy_from_slice(&attn_out[i * dim_d * c_wide + f * c..i * dim_d * c_wide + (f + 1) * c]);
                sp[c..].copy_from_slice(&x_pre[i * dim_d * c_wide + f * c..i * dim_d * c_wide + (f + 1) * c]);
            }
        }
        let mn_l = degree_linear_fwd(
            &attn_out,
            &blk.mn_expand_w,
            &w.degree_of_d,
            n,
            dim_d,
            n_focus,
            c,
            3 * c,
        );
        let mn_r = degree_linear_fwd(
            &x_pre,
            &blk.mn_expand_w,
            &w.degree_of_d,
            n,
            dim_d,
            n_focus,
            c,
            3 * c,
        );

        let (mn_u, mn_lg, mn_rg) = if n_focus == 1 {
            if let Some(g) = gpu {
                g.run_so3_grid_quad_f1(&mn_l, &mn_r, n, 3 * dim_d, n_grid, c)
                    .await?
            } else {
                let lg = so3_to_grid(&mn_l, &w.to_grid_mat, n, dim_d, 1, n_grid, c);
                let rg = so3_to_grid(&mn_r, &w.to_grid_mat, n, dim_d, 1, n_grid, c);
                let prod: Vec<f32> = lg.iter().zip(rg.iter()).map(|(&a, &b)| a * b).collect();
                let u = so3_from_grid(&prod, &w.from_grid_mat, n, dim_d, 1, n_grid, c);
                (u, lg, rg)
            }
        } else {
            let lg = so3_to_grid(&mn_l, &w.to_grid_mat, n, dim_d, n_focus, n_grid, c);
            let rg = so3_to_grid(&mn_r, &w.to_grid_mat, n, dim_d, n_focus, n_grid, c);
            let prod: Vec<f32> = lg.iter().zip(rg.iter()).map(|(&a, &b)| a * b).collect();
            let u = so3_from_grid(&prod, &w.from_grid_mat, n, dim_d, n_focus, n_grid, c);
            (u, lg, rg)
        };

        let (mn_u_post_sc, mn_sc) =
            apply_scalar_path_fwd(&mn_u, &mn_spair, &blk.mn_sc_gate_w, n, dim_d, n_focus, c);
        let mn_con = degree_linear_fwd(
            &mn_u_post_sc,
            &blk.mn_contract_w,
            &w.degree_of_d,
            n,
            dim_d,
            n_focus,
            3 * c,
            c,
        );
        let mut so2_out_post_proj = attn_out.clone();
        for i in 0..n {
            for d in 0..dim_d {
                for f in 0..n_focus {
                    let rs = &blk.mn_res_scale[f * c..(f + 1) * c];
                    let mc_s = &mn_con[((i * dim_d + d) * n_focus + f) * c
                        ..((i * dim_d + d) * n_focus + f + 1) * c];
                    let out_s = &mut so2_out_post_proj[(i * dim_d + d) * c_wide + f * c
                        ..(i * dim_d + d) * c_wide + (f + 1) * c];
                    for ch in 0..c {
                        out_s[ch] += mc_s[ch] * rs[ch];
                    }
                }
            }
        }

        // 3.2.10 post_focus_mix + post_so2_norm + residual
        let so2_conv_out = degree_linear_fwd(
            &so2_out_post_proj,
            &blk.post_focus_w,
            &w.degree_of_d,
            n,
            dim_d,
            1,
            c_wide,
            c,
        );
        let (u_so2, post_so2_norm) = equiv_rmsnorm_fwd(
            &so2_conv_out,
            &blk.post_so2_norm_scale,
            &blk.post_so2_norm_bias,
            &w.degree_of_d,
            &w.balance_weight,
            n,
            dim_d,
            c,
        );
        let mut x_mid = x_cur.clone();
        for idx in 0..x_mid.len() {
            x_mid[idx] += u_so2[idx];
        }

        // 3.3 Block-Internal EquivariantFFN (`GridBranch(1)`, `F = 1`, `H = h_ffn`)
        let (ffn_x_norm, pre_ffn_norm) = equiv_rmsnorm_fwd(
            &x_mid,
            &blk.pre_ffn_norm_scale,
            &blk.pre_ffn_norm_bias,
            &w.degree_of_d,
            &w.balance_weight,
            n,
            dim_d,
            c,
        );
        let ffn_h1 = degree_linear_fwd(
            &ffn_x_norm,
            &blk.ffn_l1_w,
            &w.degree_of_d,
            n,
            dim_d,
            1,
            c,
            6 * h_ffn,
        );
        let mut ffn_l = vec![0.0f32; n * 3 * dim_d * h_ffn];
        let mut ffn_r = vec![0.0f32; n * 3 * dim_d * h_ffn];
        let mut ffn_spair = vec![0.0f32; n * 2 * h_ffn];
        for i in 0..n {
            for d in 0..dim_d {
                let row = &ffn_h1[(i * dim_d + d) * 6 * h_ffn..(i * dim_d + d + 1) * 6 * h_ffn];
                ffn_l[(i * dim_d + d) * 3 * h_ffn..(i * dim_d + d + 1) * 3 * h_ffn]
                    .copy_from_slice(&row[..3 * h_ffn]);
                ffn_r[(i * dim_d + d) * 3 * h_ffn..(i * dim_d + d + 1) * 3 * h_ffn]
                    .copy_from_slice(&row[3 * h_ffn..]);
                if d == 0 {
                    ffn_spair[i * 2 * h_ffn..i * 2 * h_ffn + h_ffn]
                        .copy_from_slice(&row[..h_ffn]);
                    ffn_spair[i * 2 * h_ffn + h_ffn..(i + 1) * 2 * h_ffn]
                        .copy_from_slice(&row[3 * h_ffn..4 * h_ffn]);
                }
            }
        }
        let ffn_lp = mm_in_out_fwd(&ffn_l, &blk.ffn_left_w, None, n * 3 * dim_d, h_ffn, h_ffn);
        let ffn_rp = mm_in_out_fwd(&ffn_r, &blk.ffn_right_w, None, n * 3 * dim_d, h_ffn, h_ffn);

        let (ffn_from_g, ffn_lg, ffn_rg) = if let Some(g) = gpu {
            g.run_so3_grid_quad_f1(&ffn_lp, &ffn_rp, n, 3 * dim_d, n_grid, h_ffn)
                .await?
        } else {
            let lg = so3_to_grid(&ffn_lp, &w.to_grid_mat, n, dim_d, 1, n_grid, h_ffn);
            let rg = so3_to_grid(&ffn_rp, &w.to_grid_mat, n, dim_d, 1, n_grid, h_ffn);
            let prod: Vec<f32> = lg.iter().zip(rg.iter()).map(|(&a, &b)| a * b).collect();
            let fg = so3_from_grid(&prod, &w.from_grid_mat, n, dim_d, 1, n_grid, h_ffn);
            (fg, lg, rg)
        };
        let ffn_u = mm_in_out_fwd(&ffn_from_g, &blk.ffn_out_w, None, n * 3 * dim_d, h_ffn, h_ffn);
        let (ffn_u_post_sc, ffn_sc) =
            apply_scalar_path_fwd(&ffn_u, &ffn_spair, &blk.ffn_sc_gate_w, n, dim_d, 1, h_ffn);
        let y_ffn = degree_linear_fwd(
            &ffn_u_post_sc,
            &blk.ffn_l2_w,
            &w.degree_of_d,
            n,
            dim_d,
            1,
            3 * h_ffn,
            c,
        );

        let mut x_next = x_mid;
        for idx in 0..x_next.len() {
            x_next[idx] += y_ffn[idx];
        }

        if need_forces {
            block_tapes.push(BlockTape {
                x_pre,
                x_loc_rot,
                r_red,
                rad_deg_dense,
                g_focus_norm,
                focus_sm,
                focus_alpha,
                x_loc_post_mix_f,
                mix_tapes,
                x_loc_back,
                x_msg,
                qk_norm,
                q,
                k,
                attn_alpha,
                attn_exp_over_den,
                m_agg,
                out_gate_norm,
                g_out_attn,
                mn_lg,
                mn_rg,
                mn_sc,
                post_so2_norm,
                pre_ffn_norm,
                ffn_lg,
                ffn_rg,
                ffn_sc,
            });
        }
        x_cur = x_next;
    }

    // 4. Readout (`output_ffn` with `GridMLP` self-fused) + `SeZMEnergyFittingNet`
    let out_h1 = degree_linear_fwd(
        &x_cur,
        &w.out_ffn.l1_w,
        &w.degree_of_d,
        n,
        dim_d,
        1,
        c,
        6 * h_ffn,
    );
    // Per-frame concat of L (N, D, 3, H) and R (N, D, 3, H) -> F_fused (N, 3*D, 2*H)
    let mut out_fused = vec![0.0f32; n * 3 * dim_d * 2 * h_ffn];
    let mut out_spair = vec![0.0f32; n * 2 * h_ffn];
    for i in 0..n {
        for d in 0..dim_d {
            let row = &out_h1[(i * dim_d + d) * 6 * h_ffn..(i * dim_d + d + 1) * 6 * h_ffn];
            for k in 0..3 {
                let dst_f =
                    &mut out_fused[((i * dim_d + d) * 3 + k) * 2 * h_ffn..((i * dim_d + d) * 3 + k + 1) * 2 * h_ffn];
                dst_f[..h_ffn].copy_from_slice(&row[k * h_ffn..(k + 1) * h_ffn]);
                dst_f[h_ffn..].copy_from_slice(&row[3 * h_ffn + k * h_ffn..3 * h_ffn + (k + 1) * h_ffn]);
            }
            if d == 0 {
                out_spair[i * 2 * h_ffn..i * 2 * h_ffn + h_ffn].copy_from_slice(&row[..h_ffn]);
                out_spair[i * 2 * h_ffn + h_ffn..(i + 1) * 2 * h_ffn]
                    .copy_from_slice(&row[3 * h_ffn..4 * h_ffn]);
            }
        }
    }
    let out_lp = mm_in_out_fwd(
        &out_fused,
        &w.out_ffn.left_w,
        None,
        n * 3 * dim_d,
        2 * h_ffn,
        2 * h_ffn,
    );
    let out_rp = mm_in_out_fwd(
        &out_fused,
        &w.out_ffn.right_w,
        None,
        n * 3 * dim_d,
        2 * h_ffn,
        2 * h_ffn,
    );

    let (out_from_g, out_lg, out_rg) = if let Some(g) = gpu {
        g.run_so3_grid_quad_f1(&out_lp, &out_rp, n, 3 * dim_d, n_grid, 2 * h_ffn)
            .await?
    } else {
        let lg = so3_to_grid(&out_lp, &w.to_grid_mat, n, dim_d, 1, n_grid, 2 * h_ffn);
        let rg = so3_to_grid(&out_rp, &w.to_grid_mat, n, dim_d, 1, n_grid, 2 * h_ffn);
        let prod: Vec<f32> = lg.iter().zip(rg.iter()).map(|(&a, &b)| a * b).collect();
        let fg = so3_from_grid(&prod, &w.from_grid_mat, n, dim_d, 1, n_grid, 2 * h_ffn);
        (fg, lg, rg)
    };
    let out_u = mm_in_out_fwd(
        &out_from_g,
        &w.out_ffn.out_w,
        None,
        n * 3 * dim_d,
        2 * h_ffn,
        h_ffn,
    );
    let (out_u_post_sc, out_sc) =
        apply_scalar_path_fwd(&out_u, &out_spair, &w.out_ffn.sc_gate_w, n, dim_d, 1, h_ffn);
    let out_y_ffn = degree_linear_fwd(
        &out_u_post_sc,
        &w.out_ffn.l2_w,
        &w.degree_of_d,
        n,
        dim_d,
        1,
        3 * h_ffn,
        c,
    );

    // Extract l = 0 scalar descriptor: d_i = (x_cur + out_y_ffn)[i, 0, :] (shape N x C)
    let mut descriptor = vec![0.0f32; n * c];
    for i in 0..n {
        for ch in 0..c {
            descriptor[i * c + ch] = x_cur[i * dim_d * c + ch] + out_y_ffn[i * dim_d * c + ch];
        }
    }

    // SeZMEnergyFittingNet (`GLULayer`: VALUE FIRST, GATE SECOND!)
    let fit_y = mm_in_out_fwd(&descriptor, &w.fit_glu_w, Some(&w.fit_glu_b), n, c, 2 * h_ffn);
    let mut total_energy_ev = 0.0f64;
    for i in 0..n {
        let yr = &fit_y[i * 2 * h_ffn..(i + 1) * 2 * h_ffn];
        let mut e_net = 0.0f32;
        for h in 0..h_ffn {
            let h_act = yr[h] * silu(yr[h_ffn + h]);
            e_net += h_act * w.fit_out_w[h];
        }
        let at = atypes[i];
        let e_i = ((e_net + (w.bias_atom_e[at] as f32)) as f64) + w.out_bias[at];
        total_energy_ev += e_i;
    }

    if !need_forces {
        return Ok(PotentialEval {
            energy_ev: total_energy_ev,
            forces_ev_a: Vec::new(),
            descriptor: if need_descriptor {
                Some(descriptor)
            } else {
                None
            },
            desc_dim: c,
        });
    }

    // =========================================================================
    // 5. Analytical Reverse-Mode VJP (`F = -dE/dR` or `-dL/dR` when `custom_d_desc` is provided)
    // =========================================================================
    let mut d_desc = vec![0.0f32; n * c];
    if let Some(up) = custom_d_desc {
        d_desc.copy_from_slice(&up[..n * c]);
    } else {
        // 5.1 Fitting Net VJP (`d_e_net == 1.0` for all atoms)
        let mut d_fit_y = vec![0.0f32; n * 2 * h_ffn];
        for i in 0..n {
            let yr = &fit_y[i * 2 * h_ffn..(i + 1) * 2 * h_ffn];
            let dyr = &mut d_fit_y[i * 2 * h_ffn..(i + 1) * 2 * h_ffn];
            for h in 0..h_ffn {
                let dh = w.fit_out_w[h];
                let v_val = yr[h];
                let g_val = yr[h_ffn + h];
                dyr[h] = dh * silu(g_val);
                dyr[h_ffn + h] = dh * v_val * silu_grad(g_val);
            }
        }
        mm_in_out_bwd_add(&d_fit_y, &w.fit_glu_w, n, c, 2 * h_ffn, &mut d_desc);
    }

    // 5.2 Output Readout FFN VJP
    let mut d_x_cur = vec![0.0f32; n * dim_d * c];
    let mut d_out_y_ffn = vec![0.0f32; n * dim_d * c];
    for i in 0..n {
        d_x_cur[i * dim_d * c..i * dim_d * c + c].copy_from_slice(&d_desc[i * c..(i + 1) * c]);
        d_out_y_ffn[i * dim_d * c..i * dim_d * c + c].copy_from_slice(&d_desc[i * c..(i + 1) * c]);
    }

    let mut d_out_u_post = vec![0.0f32; n * 3 * dim_d * h_ffn];
    degree_linear_bwd_add(
        &d_out_y_ffn,
        &w.out_ffn.l2_w,
        &w.degree_of_d,
        n,
        dim_d,
        1,
        3 * h_ffn,
        c,
        &mut d_out_u_post,
    );
    let (d_out_u, d_out_spair) =
        apply_scalar_path_bwd(&d_out_u_post, &w.out_ffn.sc_gate_w, &out_sc, n, dim_d, 1, h_ffn);
    let mut d_out_from_g = vec![0.0f32; n * 3 * dim_d * 2 * h_ffn];
    mm_in_out_bwd_add(
        &d_out_u,
        &w.out_ffn.out_w,
        n * 3 * dim_d,
        2 * h_ffn,
        h_ffn,
        &mut d_out_from_g,
    );
    let d_out_prod = so3_from_grid_bwd(&d_out_from_g, &w.from_grid_mat, n, dim_d, 1, n_grid, 2 * h_ffn);
    let d_out_lg: Vec<f32> = d_out_prod
        .iter()
        .zip(out_rg.iter())
        .map(|(&dp, &rg)| dp * rg)
        .collect();
    let d_out_rg: Vec<f32> = d_out_prod
        .iter()
        .zip(out_lg.iter())
        .map(|(&dp, &lg)| dp * lg)
        .collect();
    let mut d_out_lp = vec![0.0f32; n * 3 * dim_d * 2 * h_ffn];
    let mut d_out_rp = vec![0.0f32; n * 3 * dim_d * 2 * h_ffn];
    so3_to_grid_bwd_add(&d_out_lg, &w.to_grid_mat, n, dim_d, 1, n_grid, 2 * h_ffn, &mut d_out_lp);
    so3_to_grid_bwd_add(&d_out_rg, &w.to_grid_mat, n, dim_d, 1, n_grid, 2 * h_ffn, &mut d_out_rp);

    let mut d_out_fused = vec![0.0f32; n * 3 * dim_d * 2 * h_ffn];
    mm_in_out_bwd_add(
        &d_out_lp,
        &w.out_ffn.left_w,
        n * 3 * dim_d,
        2 * h_ffn,
        2 * h_ffn,
        &mut d_out_fused,
    );
    mm_in_out_bwd_add(
        &d_out_rp,
        &w.out_ffn.right_w,
        n * 3 * dim_d,
        2 * h_ffn,
        2 * h_ffn,
        &mut d_out_fused,
    );

    let mut d_out_h1 = vec![0.0f32; n * dim_d * 6 * h_ffn];
    for i in 0..n {
        for d in 0..dim_d {
            let dst_r = &mut d_out_h1[(i * dim_d + d) * 6 * h_ffn..(i * dim_d + d + 1) * 6 * h_ffn];
            for k in 0..3 {
                let src_f =
                    &d_out_fused[((i * dim_d + d) * 3 + k) * 2 * h_ffn..((i * dim_d + d) * 3 + k + 1) * 2 * h_ffn];
                dst_r[k * h_ffn..(k + 1) * h_ffn].copy_from_slice(&src_f[..h_ffn]);
                dst_r[3 * h_ffn + k * h_ffn..3 * h_ffn + (k + 1) * h_ffn]
                    .copy_from_slice(&src_f[h_ffn..]);
            }
            if d == 0 {
                for h in 0..h_ffn {
                    dst_r[h] += d_out_spair[i * 2 * h_ffn + h];
                    dst_r[3 * h_ffn + h] += d_out_spair[i * 2 * h_ffn + h_ffn + h];
                }
            }
        }
    }
    degree_linear_bwd_add(
        &d_out_h1,
        &w.out_ffn.l1_w,
        &w.degree_of_d,
        n,
        dim_d,
        1,
        c,
        6 * h_ffn,
        &mut d_x_cur,
    );

    // Global edge-level gradient accumulators
    let mut d_radial_feat = vec![0.0f32; n_edges * l1_cnt * c];
    let mut d_d_to_m = vec![0.0f32; n_edges * dim_m * dim_d];
    let mut d_edge_env = vec![0.0f32; n_edges];

    // 5.3 Reverse Loop Over Interaction Blocks `b = n_blocks - 1 .. 0`
    for b_idx in (0..w.cfg.n_blocks).rev() {
        let blk = &w.blocks[b_idx];
        let tape = &block_tapes[b_idx];

        // 5.3.1 Block EquivariantFFN VJP
        let mut d_x_mid = d_x_cur.clone();
        let mut d_ffn_u_post = vec![0.0f32; n * 3 * dim_d * h_ffn];
        degree_linear_bwd_add(
            &d_x_cur,
            &blk.ffn_l2_w,
            &w.degree_of_d,
            n,
            dim_d,
            1,
            3 * h_ffn,
            c,
            &mut d_ffn_u_post,
        );
        let (d_ffn_u, d_ffn_spair) = apply_scalar_path_bwd(
            &d_ffn_u_post,
            &blk.ffn_sc_gate_w,
            &tape.ffn_sc,
            n,
            dim_d,
            1,
            h_ffn,
        );
        let mut d_ffn_from_g = vec![0.0f32; n * 3 * dim_d * h_ffn];
        mm_in_out_bwd_add(
            &d_ffn_u,
            &blk.ffn_out_w,
            n * 3 * dim_d,
            h_ffn,
            h_ffn,
            &mut d_ffn_from_g,
        );
        let d_ffn_prod =
            so3_from_grid_bwd(&d_ffn_from_g, &w.from_grid_mat, n, dim_d, 1, n_grid, h_ffn);
        let d_ffn_lg: Vec<f32> = d_ffn_prod
            .iter()
            .zip(tape.ffn_rg.iter())
            .map(|(&dp, &rg)| dp * rg)
            .collect();
        let d_ffn_rg: Vec<f32> = d_ffn_prod
            .iter()
            .zip(tape.ffn_lg.iter())
            .map(|(&dp, &lg)| dp * lg)
            .collect();
        let mut d_ffn_lp = vec![0.0f32; n * 3 * dim_d * h_ffn];
        let mut d_ffn_rp = vec![0.0f32; n * 3 * dim_d * h_ffn];
        so3_to_grid_bwd_add(&d_ffn_lg, &w.to_grid_mat, n, dim_d, 1, n_grid, h_ffn, &mut d_ffn_lp);
        so3_to_grid_bwd_add(&d_ffn_rg, &w.to_grid_mat, n, dim_d, 1, n_grid, h_ffn, &mut d_ffn_rp);

        let mut d_ffn_l = vec![0.0f32; n * 3 * dim_d * h_ffn];
        let mut d_ffn_r = vec![0.0f32; n * 3 * dim_d * h_ffn];
        mm_in_out_bwd_add(&d_ffn_lp, &blk.ffn_left_w, n * 3 * dim_d, h_ffn, h_ffn, &mut d_ffn_l);
        mm_in_out_bwd_add(
            &d_ffn_rp,
            &blk.ffn_right_w,
            n * 3 * dim_d,
            h_ffn,
            h_ffn,
            &mut d_ffn_r,
        );

        let mut d_ffn_h1 = vec![0.0f32; n * dim_d * 6 * h_ffn];
        for i in 0..n {
            for d in 0..dim_d {
                let dst_r = &mut d_ffn_h1[(i * dim_d + d) * 6 * h_ffn..(i * dim_d + d + 1) * 6 * h_ffn];
                dst_r[..3 * h_ffn].copy_from_slice(
                    &d_ffn_l[(i * dim_d + d) * 3 * h_ffn..(i * dim_d + d + 1) * 3 * h_ffn],
                );
                dst_r[3 * h_ffn..].copy_from_slice(
                    &d_ffn_r[(i * dim_d + d) * 3 * h_ffn..(i * dim_d + d + 1) * 3 * h_ffn],
                );
                if d == 0 {
                    for h in 0..h_ffn {
                        dst_r[h] += d_ffn_spair[i * 2 * h_ffn + h];
                        dst_r[3 * h_ffn + h] += d_ffn_spair[i * 2 * h_ffn + h_ffn + h];
                    }
                }
            }
        }
        let mut d_ffn_x_norm = vec![0.0f32; n * dim_d * c];
        degree_linear_bwd_add(
            &d_ffn_h1,
            &blk.ffn_l1_w,
            &w.degree_of_d,
            n,
            dim_d,
            1,
            c,
            6 * h_ffn,
            &mut d_ffn_x_norm,
        );
        equiv_rmsnorm_bwd_add(
            &d_ffn_x_norm,
            &blk.pre_ffn_norm_scale,
            &w.degree_of_d,
            &w.balance_weight,
            &tape.pre_ffn_norm,
            n,
            dim_d,
            c,
            &mut d_x_mid,
        );

        // 5.3.2 post_so2_norm + post_focus_mix VJP
        let mut d_x_in = d_x_mid.clone();
        let mut d_so2_conv_out = vec![0.0f32; n * dim_d * c];
        equiv_rmsnorm_bwd_add(
            &d_x_mid,
            &blk.post_so2_norm_scale,
            &w.degree_of_d,
            &w.balance_weight,
            &tape.post_so2_norm,
            n,
            dim_d,
            c,
            &mut d_so2_conv_out,
        );
        let mut d_so2_post_proj = vec![0.0f32; n * dim_d * c_wide];
        degree_linear_bwd_add(
            &d_so2_conv_out,
            &blk.post_focus_w,
            &w.degree_of_d,
            n,
            dim_d,
            1,
            c_wide,
            c,
            &mut d_so2_post_proj,
        );

        // 5.3.3 `message_node_grid_product` VJP (`query = attn_out`, `context = x_pre`)
        let mut d_attn_out = d_so2_post_proj.clone();
        let mut d_x_pre = vec![0.0f32; n * dim_d * c_wide];

        let mut d_mn_con = vec![0.0f32; n * dim_d * c_wide];
        for i in 0..n {
            for d in 0..dim_d {
                for f in 0..n_focus {
                    let rs = &blk.mn_res_scale[f * c..(f + 1) * c];
                    let dp_s = &d_so2_post_proj[(i * dim_d + d) * c_wide + f * c
                        ..(i * dim_d + d) * c_wide + (f + 1) * c];
                    let dmc_s = &mut d_mn_con[(i * dim_d + d) * c_wide + f * c
                        ..(i * dim_d + d) * c_wide + (f + 1) * c];
                    for ch in 0..c {
                        dmc_s[ch] = dp_s[ch] * rs[ch];
                    }
                }
            }
        }
        let mut d_mn_u_post = vec![0.0f32; n * dim_d * n_focus * 3 * c];
        degree_linear_bwd_add(
            &d_mn_con,
            &blk.mn_contract_w,
            &w.degree_of_d,
            n,
            dim_d,
            n_focus,
            3 * c,
            c,
            &mut d_mn_u_post,
        );
        let (d_mn_u, d_mn_spair) = apply_scalar_path_bwd(
            &d_mn_u_post,
            &blk.mn_sc_gate_w,
            &tape.mn_sc,
            n,
            dim_d,
            n_focus,
            c,
        );
        let d_mn_prod =
            so3_from_grid_bwd(&d_mn_u, &w.from_grid_mat, n, dim_d, n_focus, n_grid, c);
        let d_mn_lg: Vec<f32> = d_mn_prod
            .iter()
            .zip(tape.mn_rg.iter())
            .map(|(&dp, &rg)| dp * rg)
            .collect();
        let d_mn_rg: Vec<f32> = d_mn_prod
            .iter()
            .zip(tape.mn_lg.iter())
            .map(|(&dp, &lg)| dp * lg)
            .collect();

        let mut d_mn_l = vec![0.0f32; n * dim_d * n_focus * 3 * c];
        let mut d_mn_r = vec![0.0f32; n * dim_d * n_focus * 3 * c];
        so3_to_grid_bwd_add(&d_mn_lg, &w.to_grid_mat, n, dim_d, n_focus, n_grid, c, &mut d_mn_l);
        so3_to_grid_bwd_add(&d_mn_rg, &w.to_grid_mat, n, dim_d, n_focus, n_grid, c, &mut d_mn_r);

        degree_linear_bwd_add(
            &d_mn_l,
            &blk.mn_expand_w,
            &w.degree_of_d,
            n,
            dim_d,
            n_focus,
            c,
            3 * c,
            &mut d_attn_out,
        );
        degree_linear_bwd_add(
            &d_mn_r,
            &blk.mn_expand_w,
            &w.degree_of_d,
            n,
            dim_d,
            n_focus,
            c,
            3 * c,
            &mut d_x_pre,
        );
        for i in 0..n {
            for f in 0..n_focus {
                let dsp = &d_mn_spair[(i * n_focus + f) * 2 * c..(i * n_focus + f + 1) * 2 * c];
                for ch in 0..c {
                    d_attn_out[i * dim_d * c_wide + f * c + ch] += dsp[ch];
                    d_x_pre[i * dim_d * c_wide + f * c + ch] += dsp[c + ch];
                }
            }
        }

        // 5.3.4 Attention Aggregation & Output Gate VJP
        let mut d_m_agg = vec![0.0f32; n * dim_d * c_wide];
        let mut d_u_gate = vec![0.0f32; n * c_wide];
        for i in 0..n {
            for f in 0..n_focus {
                let go = tape.g_out_attn[i * n_focus + f];
                let mut d_go = 0.0f32;
                for d in 0..dim_d {
                    let dao = &d_attn_out[(i * dim_d + d) * c_wide + f * c
                        ..(i * dim_d + d) * c_wide + (f + 1) * c];
                    let ma = &tape.m_agg[(i * dim_d + d) * c_wide + f * c
                        ..(i * dim_d + d) * c_wide + (f + 1) * c];
                    let dma = &mut d_m_agg[(i * dim_d + d) * c_wide + f * c
                        ..(i * dim_d + d) * c_wide + (f + 1) * c];
                    for ch in 0..c {
                        dma[ch] = dao[ch] * go;
                        d_go += dao[ch] * ma[ch];
                    }
                }
                let d_logit_go = d_go * go * (1.0 - go);
                let dug = &mut d_u_gate[(i * n_focus + f) * c..(i * n_focus + f + 1) * c];
                for ch in 0..c {
                    dug[ch] += d_logit_go * blk.attn_gate_w[ch * n_focus + f];
                }
            }
        }
        let mut d_x_l0 = vec![0.0f32; n * c_wide];
        rmsnorm_grouped_bwd_add(
            &d_u_gate,
            &blk.attn_gate_norm,
            &tape.out_gate_norm,
            n,
            n_focus,
            c,
            &mut d_x_l0,
        );

        let mut d_x_msg = vec![0.0f32; n_edges * dim_d * c_wide];
        let mut d_alpha = vec![0.0f32; n_edges * n_focus];
        let mut sum_alpha_dalpha = vec![0.0f32; n * n_focus];
        for e_idx in 0..n_edges {
            let dst = edge_dst[e_idx];
            for f in 0..n_focus {
                let al = tape.attn_alpha[e_idx * n_focus + f];
                let mut dal = 0.0f32;
                for d in 0..dim_d {
                    let dma = &d_m_agg[(dst * dim_d + d) * c_wide + f * c
                        ..(dst * dim_d + d) * c_wide + (f + 1) * c];
                    let xm = &tape.x_msg[(e_idx * dim_d + d) * c_wide + f * c
                        ..(e_idx * dim_d + d) * c_wide + (f + 1) * c];
                    let dxm = &mut d_x_msg[(e_idx * dim_d + d) * c_wide + f * c
                        ..(e_idx * dim_d + d) * c_wide + (f + 1) * c];
                    for ch in 0..c {
                        dxm[ch] = al * dma[ch];
                        dal += dma[ch] * xm[ch];
                    }
                }
                d_alpha[e_idx * n_focus + f] = dal;
                sum_alpha_dalpha[dst * n_focus + f] += al * dal;
            }
        }

        let inv_sqrt_c = 1.0 / (c as f32).sqrt();
        let mut d_q = vec![0.0f32; n * c_wide];
        let mut d_k = vec![0.0f32; n * c_wide];
        let mut d_r_red = vec![0.0f32; n_edges * dim_m * c_wide];

        for e_idx in 0..n_edges {
            let dst = edge_dst[e_idx];
            let src = edge_src[e_idx];
            let env_v = edge_env[e_idx];
            for f in 0..n_focus {
                let al = tape.attn_alpha[e_idx * n_focus + f];
                let diff = d_alpha[e_idx * n_focus + f] - sum_alpha_dalpha[dst * n_focus + f];
                let d_lgt = al * diff;
                d_edge_env[e_idx] +=
                    2.0 * env_v * tape.attn_exp_over_den[e_idx * n_focus + f] * diff;

                let scale_qk = d_lgt * inv_sqrt_c;
                let dq_df = &mut d_q[(dst * n_focus + f) * c..(dst * n_focus + f + 1) * c];
                let dk_sf = &mut d_k[(src * n_focus + f) * c..(src * n_focus + f + 1) * c];
                let q_df = &tape.q[(dst * n_focus + f) * c..(dst * n_focus + f + 1) * c];
                let k_sf = &tape.k[(src * n_focus + f) * c..(src * n_focus + f + 1) * c];
                let dr_ef =
                    &mut d_r_red[e_idx * dim_m * c_wide + f * c..e_idx * dim_m * c_wide + (f + 1) * c];
                for ch in 0..c {
                    dq_df[ch] += scale_qk * k_sf[ch];
                    dk_sf[ch] += scale_qk * q_df[ch];
                    dr_ef[ch] += d_lgt * blk.attn_logit_w[ch * n_focus + f];
                }
            }
        }

        let mut d_u_qk = vec![0.0f32; n * c_wide];
        focus_linear_bwd_add(&d_q, &blk.attn_q_w, n, n_focus, c, c, &mut d_u_qk);
        focus_linear_bwd_add(&d_k, &blk.attn_k_w, n, n_focus, c, c, &mut d_u_qk);
        rmsnorm_grouped_bwd_add(
            &d_u_qk,
            &blk.attn_qk_norm,
            &tape.qk_norm,
            n,
            n_focus,
            c,
            &mut d_x_l0,
        );
        for i in 0..n {
            for ch in 0..c_wide {
                d_x_pre[i * dim_d * c_wide + ch] += d_x_l0[i * c_wide + ch];
            }
        }

        // 5.3.5 Rotate Back VJP: `x_msg[e, d, ch] = sum_m D_to_m[e, m, d] * rescale[d] * x_loc_back[e, m, ch]`
        let mut d_x_loc_back = vec![0.0f32; n_edges * dim_m * c_wide];
        for e_idx in 0..n_edges {
            let dtm_e = &d_to_m[e_idx * dim_m * dim_d..(e_idx + 1) * dim_m * dim_d];
            let d_dtm_e = &mut d_d_to_m[e_idx * dim_m * dim_d..(e_idx + 1) * dim_m * dim_d];
            let xlb_e = &tape.x_loc_back[e_idx * dim_m * c_wide..(e_idx + 1) * dim_m * c_wide];
            let d_xlb_e = &mut d_x_loc_back[e_idx * dim_m * c_wide..(e_idx + 1) * dim_m * c_wide];
            let dxm_e = &d_x_msg[e_idx * dim_d * c_wide..(e_idx + 1) * dim_d * c_wide];

            for m_pos in 0..dim_m {
                let d_row = &dtm_e[m_pos * dim_d..(m_pos + 1) * dim_d];
                let dd_row = &mut d_dtm_e[m_pos * dim_d..(m_pos + 1) * dim_d];
                let xlb_m = &xlb_e[m_pos * c_wide..(m_pos + 1) * c_wide];
                let d_xlb_m = &mut d_xlb_e[m_pos * c_wide..(m_pos + 1) * c_wide];

                for d in 0..dim_d {
                    let rsc = w.rotate_inv_rescale[d];
                    let dxm_d = &dxm_e[d * c_wide..(d + 1) * c_wide];
                    let dv = d_row[d] * rsc;
                    let mut dot_m_d = 0.0f32;
                    for ch in 0..c_wide {
                        d_xlb_m[ch] += dv * dxm_d[ch];
                        dot_m_d += xlb_m[ch] * dxm_d[ch];
                    }
                    dd_row[d] += rsc * dot_m_d;
                }
            }
        }

        // Permute `(E, D_m, F, C)` -> focus-major `(F, E, D_m * C)`
        let dmc = dim_m * c;
        let mut d_x_loc_f = vec![0.0f32; n_focus * n_edges * dmc];
        for e_idx in 0..n_edges {
            for m_pos in 0..dim_m {
                for f in 0..n_focus {
                    let src_off = ((e_idx * dim_m + m_pos) * n_focus + f) * c;
                    let dst_off = (f * n_edges + e_idx) * dmc + m_pos * c;
                    d_x_loc_f[dst_off..dst_off + c]
                        .copy_from_slice(&d_x_loc_back[src_off..src_off + c]);
                }
            }
        }

        // 5.3.6 Cross-focus competition VJP (`F > 1`, `Neo`)
        let mut d_g_focus_pre = vec![0.0f32; if n_focus > 1 { n_edges * n_focus * c } else { 0 }];
        if n_focus > 1 {
            let fc_norm = blk.focus_comp_norm.as_ref().unwrap();
            let fc_w = blk.focus_comp_w.as_ref().unwrap();
            let mut d_g_hat = vec![0.0f32; n_edges * n_focus * c];

            for e_idx in 0..n_edges {
                let mut d_alpha_f = vec![0.0f32; n_focus];
                for f in 0..n_focus {
                    let alpha = tape.focus_alpha[e_idx * n_focus + f];
                    let x_pm = &tape.x_loc_post_mix_f
                        [(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];
                    let dx_f = &mut d_x_loc_f
                        [(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];
                    let mut dot = 0.0f32;
                    for idx in 0..dmc {
                        dot += dx_f[idx] * x_pm[idx];
                        dx_f[idx] *= alpha;
                    }
                    d_alpha_f[f] = dot;
                }
                let mut sum_sm = 0.0f32;
                for f in 0..n_focus {
                    sum_sm += tape.focus_sm[e_idx * n_focus + f] * (0.98 * d_alpha_f[f]);
                }
                for f in 0..n_focus {
                    let sm = tape.focus_sm[e_idx * n_focus + f];
                    let d_logit = sm * (0.98 * d_alpha_f[f] - sum_sm);
                    let dgh = &mut d_g_hat[(e_idx * n_focus + f) * c..(e_idx * n_focus + f + 1) * c];
                    for ch in 0..c {
                        dgh[ch] = d_logit * fc_w[ch * n_focus + f];
                    }
                }
            }
            rmsnorm_grouped_bwd_add(
                &d_g_hat,
                fc_norm,
                tape.g_focus_norm.as_ref().unwrap(),
                n_edges,
                n_focus,
                c,
                &mut d_g_focus_pre,
            );
        }

        // 5.3.7 SO(2) Mixing Stack VJP (`l_idx = mixing_layers - 1 .. 0`)
        let w0 = l1_cnt * c;
        let two_w1 = 2 * lmax * c;
        for l_idx in (0..w.cfg.mixing_layers).rev() {
            let sl = &blk.so2_layers[l_idx];
            let mt = &tape.mix_tapes[l_idx];

            // d_z_f == d_x_loc_f (residual skip keeps d_x_loc_f alive!)
            let d_y_f = if let Some(ref gw) = sl.gate_w {
                let y_f = mt.y_lin_f.as_ref().unwrap();
                let gates = mt.gates.as_ref().unwrap();
                let lc = lmax * c;
                let f_lc = n_focus * lc;
                let mut dy = vec![0.0f32; d_x_loc_f.len()];

                for f in 0..n_focus {
                    for e_idx in 0..n_edges {
                        let dz = &d_x_loc_f
                            [(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];
                        let yr = &y_f[(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];
                        let gr = &gates[(f * n_edges + e_idx) * lc..(f * n_edges + e_idx + 1) * lc];
                        let dyr = &mut dy[(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];

                        let mut d_gate_logit = vec![0.0f32; lc];
                        for m_pos in 1..dim_m {
                            let l_slot = w.degree_index_m[m_pos] - 1;
                            for ch in 0..c {
                                let g_val = gr[l_slot * c + ch];
                                let dz_v = dz[m_pos * c + ch];
                                dyr[m_pos * c + ch] = dz_v * g_val;
                                d_gate_logit[l_slot * c + ch] += dz_v * yr[m_pos * c + ch];
                            }
                        }
                        for o in 0..lc {
                            let g_val = gr[o];
                            d_gate_logit[o] *= g_val * (1.0 - g_val);
                        }
                        for i in 0..c {
                            let w_row = &gw[i * f_lc + f * lc..i * f_lc + (f + 1) * lc];
                            let mut acc = dz[i] * silu_grad(yr[i]);
                            for o in 0..lc {
                                acc += d_gate_logit[o] * w_row[o];
                            }
                            dyr[i] = acc;
                        }
                    }
                }
                dy
            } else {
                d_x_loc_f.clone()
            };

            if let Some(g) = gpu {
                let mut dym0 = vec![0.0f32; n_focus * n_edges * w0];
                let mut dym1 = vec![0.0f32; n_focus * n_edges * two_w1];
                for f in 0..n_focus {
                    for e_idx in 0..n_edges {
                        let row =
                            &d_y_f[(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];
                        dym0[(f * n_edges + e_idx) * w0..(f * n_edges + e_idx + 1) * w0]
                            .copy_from_slice(&row[..w0]);
                        if two_w1 > 0 {
                            dym1[(f * n_edges + e_idx) * two_w1..(f * n_edges + e_idx + 1) * two_w1]
                                .copy_from_slice(&row[w0..]);
                        }
                    }
                }
                let (dxm0, dxm1) = g
                    .run_so2_linear_pair(
                        b_idx, l_idx, &dym0, &dym1, n_focus, n_edges, w0, two_w1, true,
                    )
                    .await?;
                for f in 0..n_focus {
                    for e_idx in 0..n_edges {
                        let dst_r = &mut d_x_loc_f
                            [(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];
                        let s0 = &dxm0[(f * n_edges + e_idx) * w0..(f * n_edges + e_idx + 1) * w0];
                        for k in 0..w0 {
                            dst_r[k] += s0[k];
                        }
                        if two_w1 > 0 {
                            let s1 = &dxm1
                                [(f * n_edges + e_idx) * two_w1..(f * n_edges + e_idx + 1) * two_w1];
                            for k in 0..two_w1 {
                                dst_r[w0 + k] += s1[k];
                            }
                        }
                    }
                }
            } else {
                for f in 0..n_focus {
                    let wm0t = &sl.w_m0_ft[f * w0 * w0..(f + 1) * w0 * w0];
                    let wm1t = &sl.w_m1_ft[f * two_w1 * two_w1..(f + 1) * two_w1 * two_w1];
                    for e_idx in 0..n_edges {
                        let dyr =
                            &d_y_f[(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];
                        let dxr = &mut d_x_loc_f
                            [(f * n_edges + e_idx) * dmc..(f * n_edges + e_idx + 1) * dmc];
                        for o in 0..w0 {
                            let gv = dyr[o];
                            if gv != 0.0 {
                                let wrt = &wm0t[o * w0..(o + 1) * w0];
                                for i in 0..w0 {
                                    dxr[i] += gv * wrt[i];
                                }
                            }
                        }
                        let dyr1 = &dyr[w0..];
                        let dxr1 = &mut dxr[w0..];
                        for o in 0..two_w1 {
                            let gv = dyr1[o];
                            if gv != 0.0 {
                                let wrt = &wm1t[o * two_w1..(o + 1) * two_w1];
                                for i in 0..two_w1 {
                                    dxr1[i] += gv * wrt[i];
                                }
                            }
                        }
                    }
                }
            }
        }

        if n_focus > 1 {
            for e_idx in 0..n_edges {
                for f in 0..n_focus {
                    let src_off = (e_idx * n_focus + f) * c;
                    let dst_off = (f * n_edges + e_idx) * dmc;
                    for ch in 0..c {
                        d_x_loc_f[dst_off + ch] += d_g_focus_pre[src_off + ch];
                    }
                }
            }
        }

        // Permute `(F, E, D_m * C)` -> `(E, D_m, C_wide)`
        let mut d_x_loc_rad = vec![0.0f32; n_edges * dim_m * c_wide];
        for e_idx in 0..n_edges {
            for m_pos in 0..dim_m {
                for f in 0..n_focus {
                    let src_off = (f * n_edges + e_idx) * dmc + m_pos * c;
                    let dst_off = ((e_idx * dim_m + m_pos) * n_focus + f) * c;
                    d_x_loc_rad[dst_off..dst_off + c]
                        .copy_from_slice(&d_x_loc_f[src_off..src_off + c]);
                }
            }
        }

        // 5.3.8 DynamicRadialDegreeMixer VJP
        let mut d_x_loc_rot = vec![0.0f32; n_edges * dim_m * c_wide];
        if w.cfg.radial_so2_mode == "none" {
            for idx in 0..d_x_loc_rad.len() {
                d_x_loc_rot[idx] = d_x_loc_rad[idx] * tape.r_red[idx];
                d_r_red[idx] += d_x_loc_rad[idx] * tape.x_loc_rot[idx];
            }
        } else {
            let rank = w.cfg.radial_so2_rank;
            let k_deg = l1_cnt * l1_cnt + lmax * lmax;
            let rad_w = blk.rad_deg_w.as_ref().unwrap();
            let basis = blk.rad_deg_basis.as_ref().unwrap();

            let mut d_compact = vec![0.0f32; n_edges * k_deg * rank];
            for e_idx in 0..n_edges {
                let den_e = &tape.rad_deg_dense
                    [e_idx * dim_m * dim_m * rank..(e_idx + 1) * dim_m * dim_m * rank];
                let xl_e = &tape.x_loc_rot[e_idx * dim_m * c_wide..(e_idx + 1) * dim_m * c_wide];
                let dy_e = &d_x_loc_rad[e_idx * dim_m * c_wide..(e_idx + 1) * dim_m * c_wide];
                let dxl_e =
                    &mut d_x_loc_rot[e_idx * dim_m * c_wide..(e_idx + 1) * dim_m * c_wide];

                let mut d_den_e = vec![0.0f32; dim_m * dim_m * rank];
                for o in 0..dim_m {
                    let dy_o = &dy_e[o * c_wide..(o + 1) * c_wide];
                    for i in 0..dim_m {
                        let xl_i = &xl_e[i * c_wide..(i + 1) * c_wide];
                        let dxl_i = &mut dxl_e[i * c_wide..(i + 1) * c_wide];
                        for r in 0..rank {
                            let kv = den_e[(o * dim_m + i) * rank + r];
                            let b_row = &basis[r * c_wide..(r + 1) * c_wide];
                            let mut dot_k = 0.0f32;
                            for ch in 0..c_wide {
                                let b_dy = b_row[ch] * dy_o[ch];
                                dot_k += b_dy * xl_i[ch];
                                if kv != 0.0 {
                                    dxl_i[ch] += kv * b_dy;
                                }
                            }
                            d_den_e[(o * dim_m + i) * rank + r] = dot_k;
                        }
                    }
                }

                // Gather dense gradients back into compact (m=0, and neg + pos summed for |m|=1!)
                let d_comp_e = &mut d_compact[e_idx * k_deg * rank..(e_idx + 1) * k_deg * rank];
                for l_in in 0..l1_cnt {
                    for l_out in 0..l1_cnt {
                        let c_off = (l_in * l1_cnt + l_out) * rank;
                        let d_off = (l_out * dim_m + l_in) * rank;
                        for r in 0..rank {
                            d_comp_e[c_off + r] += d_den_e[d_off + r];
                        }
                    }
                }
                let c_base1 = l1_cnt * l1_cnt;
                let neg_s = l1_cnt;
                let pos_s = l1_cnt + lmax;
                for l_in in 0..lmax {
                    for l_out in 0..lmax {
                        let c_off = (c_base1 + l_in * lmax + l_out) * rank;
                        let d_neg = ((neg_s + l_out) * dim_m + (neg_s + l_in)) * rank;
                        let d_pos = ((pos_s + l_out) * dim_m + (pos_s + l_in)) * rank;
                        for r in 0..rank {
                            d_comp_e[c_off + r] += d_den_e[d_neg + r] + d_den_e[d_pos + r];
                        }
                    }
                }
            }

            let mut d_r_m0 = vec![0.0f32; n_edges * l1_cnt * c_wide];
            mm_in_out_bwd_add(
                &d_compact,
                rad_w,
                n_edges,
                l1_cnt * c_wide,
                k_deg * rank,
                &mut d_r_m0,
            );
            for e_idx in 0..n_edges {
                let dst_s =
                    &mut d_r_red[e_idx * dim_m * c_wide..(e_idx * dim_m + l1_cnt) * c_wide];
                let src_s = &d_r_m0[e_idx * l1_cnt * c_wide..(e_idx + 1) * l1_cnt * c_wide];
                for idx in 0..src_s.len() {
                    dst_s[idx] += src_s[idx];
                }
            }
        }

        let d_r_red_c = if let Some(ref hid_w) = blk.rad_hid_proj_w {
            let mut tmp = vec![0.0f32; n_edges * dim_m * c];
            mm_in_out_bwd_add(&d_r_red, hid_w, n_edges * dim_m, c, c_wide, &mut tmp);
            tmp
        } else {
            d_r_red
        };
        for e_idx in 0..n_edges {
            for (m_pos, &deg_l) in w.degree_index_m.iter().enumerate() {
                let src_s = &d_r_red_c[(e_idx * dim_m + m_pos) * c..(e_idx * dim_m + m_pos + 1) * c];
                let dst_s =
                    &mut d_radial_feat[(e_idx * l1_cnt + deg_l) * c..(e_idx * l1_cnt + deg_l + 1) * c];
                for ch in 0..c {
                    dst_s[ch] += src_s[ch];
                }
            }
        }

        // 5.3.9 Rotate Global -> Reduced `m`-major VJP: `x_loc_rot[e] = D_to_m[e] @ x_pre[src(e)]`
        for e_idx in 0..n_edges {
            let src = edge_src[e_idx];
            let xp_s = &tape.x_pre[src * dim_d * c_wide..(src + 1) * dim_d * c_wide];
            let dxp_s = &mut d_x_pre[src * dim_d * c_wide..(src + 1) * dim_d * c_wide];
            let dtm_e = &d_to_m[e_idx * dim_m * dim_d..(e_idx + 1) * dim_m * dim_d];
            let d_dtm_e = &mut d_d_to_m[e_idx * dim_m * dim_d..(e_idx + 1) * dim_m * dim_d];
            let dxl_e = &d_x_loc_rot[e_idx * dim_m * c_wide..(e_idx + 1) * dim_m * c_wide];

            for m_pos in 0..dim_m {
                let d_row = &dtm_e[m_pos * dim_d..(m_pos + 1) * dim_d];
                let dd_row = &mut d_dtm_e[m_pos * dim_d..(m_pos + 1) * dim_d];
                let dxl_m = &dxl_e[m_pos * c_wide..(m_pos + 1) * c_wide];
                for d in 0..dim_d {
                    let dv = d_row[d];
                    let xp_d = &xp_s[d * c_wide..(d + 1) * c_wide];
                    let dxp_d = &mut dxp_s[d * c_wide..(d + 1) * c_wide];
                    let mut dot = 0.0f32;
                    for ch in 0..c_wide {
                        dxp_d[ch] += dv * dxl_m[ch];
                        dot += dxl_m[ch] * xp_d[ch];
                    }
                    dd_row[d] += dot;
                }
            }
        }

        // 5.3.10 `pre_focus_mix` VJP -> `d_x_in`
        degree_linear_bwd_add(
            &d_x_pre,
            &blk.pre_focus_w,
            &w.degree_of_d,
            n,
            dim_d,
            1,
            c,
            c_wide,
            &mut d_x_in,
        );
        d_x_cur = d_x_in;
    }

    // =========================================================================
    // 5.4 Initial Embeddings VJP (`GIE`, `EnvironmentInitialEmbedding`, `RadialMLP`, `WignerD`)
    // =========================================================================
    let mut d_dfull = vec![0.0f32; n_edges * dim_d * dim_d];
    for e_idx in 0..n_edges {
        let d_dtm_e = &d_d_to_m[e_idx * dim_m * dim_d..(e_idx + 1) * dim_m * dim_d];
        let d_df_e = &mut d_dfull[e_idx * dim_d * dim_d..(e_idx + 1) * dim_d * dim_d];
        for (m_pos, &coeff_r) in w.coeff_index_m.iter().enumerate() {
            let src_r = &d_dtm_e[m_pos * dim_d..(m_pos + 1) * dim_d];
            let dst_r = &mut d_df_e[coeff_r * dim_d..(coeff_r + 1) * dim_d];
            for d in 0..dim_d {
                dst_r[d] += src_r[d];
            }
        }
    }

    // GIE VJP (`d >= 1` of `d_x_cur`)
    let mut d_inv_sqrt_deg = vec![0.0f32; n];

    let mut d_radial_feat0 = d_radial_feat; // edge_type_feat is coordinate-constant!
    for e_idx in 0..n_edges {
        let dst = edge_dst[e_idx];
        let isd = inv_sqrt_deg[dst];
        let df_e = &d_full[e_idx * dim_d * dim_d..(e_idx + 1) * dim_d * dim_d];
        let d_df_e = &mut d_dfull[e_idx * dim_d * dim_d..(e_idx + 1) * dim_d * dim_d];
        let rf0_e = &radial_feat0[e_idx * l1_cnt * c..(e_idx + 1) * l1_cnt * c];
        let d_rf0_e = &mut d_radial_feat0[e_idx * l1_cnt * c..(e_idx + 1) * l1_cnt * c];

        for d in 1..dim_d {
            let l = w.degree_of_d[d];
            let z_idx = l * (l + 1) * dim_d + d;
            let z_ed = df_e[z_idx];
            let dx_row = &d_x_cur[(dst * dim_d + d) * c..(dst * dim_d + d + 1) * c];
            let rf_l = &rf0_e[l * c..(l + 1) * c];
            let drf_l = &mut d_rf0_e[l * c..(l + 1) * c];

            let mut dot_z = 0.0f32;
            let mut dot_deg = 0.0f32;
            for ch in 0..c {
                let g = dx_row[ch];
                dot_deg += g * z_ed * rf_l[ch];
                dot_z += g * isd * rf_l[ch];
                drf_l[ch] += g * isd * z_ed;
            }
            d_inv_sqrt_deg[dst] += dot_deg;
            d_df_e[z_idx] += dot_z;
        }
    }

    // EnvironmentInitialEmbedding (`l = 0` FiLM) VJP
    let mut d_gamma_hat = vec![0.0f32; n * c];
    let mut d_beta_hat = vec![0.0f32; n * c];
    for i in 0..n {
        for ch in 0..c {
            let dx0 = d_x_cur[i * dim_d * c + ch];
            let idx = i * c + ch;
            let tg = gamma_hat[idx].tanh();
            let tb = beta_hat[idx].tanh();
            d_gamma_hat[idx] = dx0 * t_node[idx] * w.film_scale_strength * (1.0 - tg * tg);
            d_beta_hat[idx] = dx0 * w.film_shift_strength * (1.0 - tb * tb);
        }
    }
    let mut d_gamma_raw = vec![0.0f32; n * c];
    let mut d_beta_raw = vec![0.0f32; n * c];
    rmsnorm_grouped_bwd_add(
        &d_gamma_hat,
        &w.film_scale_norm,
        &saved_gamma_norm,
        n,
        1,
        c,
        &mut d_gamma_raw,
    );
    rmsnorm_grouped_bwd_add(
        &d_beta_hat,
        &w.film_shift_norm,
        &saved_beta_norm,
        n,
        1,
        c,
        &mut d_beta_raw,
    );
    let mut d_film_raw = vec![0.0f32; n * 2 * c];
    for i in 0..n {
        d_film_raw[i * 2 * c..i * 2 * c + c].copy_from_slice(&d_gamma_raw[i * c..(i + 1) * c]);
        d_film_raw[i * 2 * c + c..(i + 1) * 2 * c].copy_from_slice(&d_beta_raw[i * c..(i + 1) * c]);
    }
    let mut d_env_gram = vec![0.0f32; n * d_emb * d_axis];
    mm_in_out_bwd_add(
        &d_film_raw,
        &w.env_out_proj,
        n,
        d_emb * d_axis,
        2 * c,
        &mut d_env_gram,
    );

    // D_i = M_i^T * M_i[:, :d_axis] => dM_i = M_i[:, :d_axis] * dD_i^T; dM_i[:, :d_axis] += M_i * dD_i
    let mut d_env_m = vec![0.0f32; n * 4 * d_emb];
    for i in 0..n {
        let mi = &env_m[i * 4 * d_emb..(i + 1) * 4 * d_emb];
        let ddi = &d_env_gram[i * d_emb * d_axis..(i + 1) * d_emb * d_axis];
        let dmi = &mut d_env_m[i * 4 * d_emb..(i + 1) * 4 * d_emb];

        for a in 0..4 {
            let m_a = &mi[a * d_emb..(a + 1) * d_emb];
            let dm_a = &mut dmi[a * d_emb..(a + 1) * d_emb];
            for k in 0..d_emb {
                let mut acc = 0.0f32;
                for ax in 0..d_axis {
                    acc += m_a[ax] * ddi[k * d_axis + ax];
                }
                dm_a[k] += acc;
            }
            for ax in 0..d_axis {
                let mut acc = 0.0f32;
                for k in 0..d_emb {
                    acc += m_a[k] * ddi[k * d_axis + ax];
                }
                dm_a[ax] += acc;
            }
        }
    }

    let mut d_r_tilde = vec![[0.0f32; 4]; n_edges];
    let mut d_env_g = vec![0.0f32; n_edges * d_emb];
    for e_idx in 0..n_edges {
        let dst = edge_dst[e_idx];
        let isd = inv_sqrt_deg[dst];
        let rt = r_tilde[e_idx];
        let g_row = &env_g[e_idx * d_emb..(e_idx + 1) * d_emb];
        let dg_row = &mut d_env_g[e_idx * d_emb..(e_idx + 1) * d_emb];

        let mut dot_deg = 0.0f32;
        for a in 0..4 {
            let dm_row = &d_env_m[(dst * 4 + a) * d_emb..(dst * 4 + a + 1) * d_emb];
            let mut drt_a = 0.0f32;
            for k in 0..d_emb {
                let dm_val = dm_row[k];
                dot_deg += dm_val * rt[a] * g_row[k];
                drt_a += dm_val * isd * g_row[k];
                dg_row[k] += dm_val * isd * rt[a];
            }
            d_r_tilde[e_idx][a] = drt_a;
        }
        d_inv_sqrt_deg[dst] += dot_deg;
    }

    let mut d_env_g1_act = vec![0.0f32; n_edges * d_hid];
    mm_in_out_bwd_add(&d_env_g, &w.env_g_w2, n_edges, d_hid, d_emb, &mut d_env_g1_act);
    for idx in 0..d_env_g1_act.len() {
        d_env_g1_act[idx] *= silu_grad(env_g1_pre[idx]);
    }
    let mut d_env_g_in = vec![0.0f32; n_edges * d_g_in];
    mm_in_out_bwd_add(
        &d_env_g1_act,
        &w.env_g_w1,
        n_edges,
        d_g_in,
        d_hid,
        &mut d_env_g_in,
    );
    let mut d_env_p2 = vec![0.0f32; n_edges * 32];
    for e_idx in 0..n_edges {
        d_env_p2[e_idx * 32..(e_idx + 1) * 32]
            .copy_from_slice(&d_env_g_in[e_idx * d_g_in..e_idx * d_g_in + 32]);
    }
    let mut d_env_p1_act = vec![0.0f32; n_edges * 32];
    mm_in_out_bwd_add(&d_env_p2, &w.env_rbf_w2, n_edges, 32, 32, &mut d_env_p1_act);
    for idx in 0..d_env_p1_act.len() {
        d_env_p1_act[idx] *= silu_grad(env_p1_pre[idx]);
    }
    let mut d_edge_rbf = vec![0.0f32; n_edges * 16];
    mm_in_out_bwd_add(&d_env_p1_act, &w.env_rbf_w1, n_edges, 16, 32, &mut d_edge_rbf);

    // RadialMLP (`radial_embedding`) VJP -> `d_edge_rbf` & `d_edge_env`
    let mut d_rad_raw = vec![0.0f32; n_edges * l1_cnt * c];
    for e_idx in 0..n_edges {
        let env_v = edge_env[e_idx];
        let mut dot_env = 0.0f32;
        for idx in 0..(l1_cnt * c) {
            let g = d_radial_feat0[e_idx * l1_cnt * c + idx];
            d_rad_raw[e_idx * l1_cnt * c + idx] = g * env_v;
            dot_env += g * rad_raw[e_idx * l1_cnt * c + idx];
        }
        d_edge_env[e_idx] += dot_env;
    }
    let mut d_rad_h2 = vec![0.0f32; n_edges * c];
    mm_in_out_bwd_add(&d_rad_raw, &w.rad_w3, n_edges, c, l1_cnt * c, &mut d_rad_h2);
    for idx in 0..d_rad_h2.len() {
        d_rad_h2[idx] *= silu_grad(rad_h1[idx]);
    }
    let mut d_rad_h0 = vec![0.0f32; n_edges * c];
    rmsnorm_grouped_bwd_add(
        &d_rad_h2,
        &w.rad_norm_scale,
        &saved_rad_norm,
        n_edges,
        1,
        c,
        &mut d_rad_h0,
    );
    mm_in_out_bwd_add(&d_rad_h0, &w.rad_w0, n_edges, 16, c, &mut d_edge_rbf);

    // `inv_sqrt_deg` VJP -> `d_edge_env`
    for e_idx in 0..n_edges {
        let dst = edge_dst[e_idx];
        let isd = inv_sqrt_deg[dst];
        let d_deg = -0.5 * (isd * isd * isd) * d_inv_sqrt_deg[dst];
        d_edge_env[e_idx] += 2.0 * edge_env[e_idx] * d_deg;
    }

    // 5.5 Assemble Total Per-Edge Coordinate Gradient `g_e = dE / d(v_e)` & Scatter Forces!
    let mut forces_ev_a = vec![[0.0f64; 3]; n];
    for e_idx in 0..n_edges {
        let v = edge_v[e_idx];
        let r = r_safe[e_idx];
        let inv_r = 1.0 / r;

        // (a) WignerD -> build_edge_quaternion -> d_v
        let mut gv = build_edge_wigner_bwd(
            &d_dfull[e_idx * dim_d * dim_d..(e_idx + 1) * dim_d * dim_d],
            &saved_quats[e_idx],
            w,
        );

        // (b) EnvironmentInitialEmbedding `r_tilde = [s_e, s_e * v_x / r, s_e * v_y / r, s_e * v_z / r]`
        //     where `s_e = edge_env * inv_r`
        let drt = d_r_tilde[e_idx];
        let s_e = edge_env[e_idx] * inv_r;
        let ds_e = drt[0] + (drt[1] * v[0] + drt[2] * v[1] + drt[3] * v[2]) * inv_r;
        d_edge_env[e_idx] += ds_e * inv_r;

        // Direct v contribution from s_e * v * inv_r:
        gv[0] += drt[1] * s_e * inv_r;
        gv[1] += drt[2] * s_e * inv_r;
        gv[2] += drt[3] * s_e * inv_r;

        // `r` contribution from `s_e = edge_env / r` and `v / r`:
        let mut d_r_total = -ds_e * edge_env[e_idx] * inv_r * inv_r
            - s_e * (drt[1] * v[0] + drt[2] * v[1] + drt[3] * v[2]) * inv_r * inv_r;

        // (c) `edge_env` -> `r`
        d_r_total += d_edge_env[e_idx] * d_edge_env_dr[e_idx];

        // (d) Bessel `edge_rbf = phi(r) * env_rbf(r)` -> `r`
        let e7 = env_rbf[e_idx];
        let de7 = d_env_rbf_dr[e_idx];
        let mut d_e7 = 0.0f32;
        for k in 0..16 {
            let drbf = d_edge_rbf[e_idx * 16 + k];
            let p_val = phi_raw[e_idx * 16 + k];
            d_e7 += drbf * p_val;

            let wn = w.bessel_freqs[k];
            let wr = wn * r;
            let dphi_dr = if wr.abs() < 1e-4 {
                (wn * wn * wn * r) * (-1.0 / 3.0 + (wr * wr) / 30.0)
            } else {
                (wr * wr.cos() - wr.sin()) / (r * r)
            };
            d_r_total += drbf * e7 * dphi_dr;
        }
        d_r_total += d_e7 * de7;

        // `dr / dv = v / r`
        gv[0] += d_r_total * v[0] * inv_r;
        gv[1] += d_r_total * v[1] * inv_r;
        gv[2] += d_r_total * v[2] * inv_r;

        // Scatter to physical forces (`v_e = r_src - r_dst` => `F[dst] += gv, F[src] -= gv`)
        let dst = edge_dst[e_idx];
        let src = edge_src[e_idx];
        forces_ev_a[dst][0] += gv[0] as f64;
        forces_ev_a[dst][1] += gv[1] as f64;
        forces_ev_a[dst][2] += gv[2] as f64;
        forces_ev_a[src][0] -= gv[0] as f64;
        forces_ev_a[src][1] -= gv[1] as f64;
        forces_ev_a[src][2] -= gv[2] as f64;
    }

    Ok(PotentialEval {
        energy_ev: total_energy_ev,
        forces_ev_a,
        descriptor: if need_descriptor {
            Some(descriptor)
        } else {
            None
        },
        desc_dim: c,
    })
}
