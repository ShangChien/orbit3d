//! Exact CPU forward & analytical reverse-mode VJP (`F = -dE/dR`) engine for `HyFF-L2`
//! (`Presto`, `Vivace`, `Moderato`), implementing `mlip_core::PotentialBackend`.

use std::sync::Arc;

use mlip_core::{MolecularSystem, NeighborCache, PotentialBackend, PotentialEval};

use crate::gpu::HyffGpuContext;
use crate::model::HyffWeights;

const SQRT_2: f32 = std::f32::consts::SQRT_2;
const INV_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
const INV_SQRT_6: f32 = 0.408_248_3;

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn silu_grad(x: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-x).exp());
    sig * (1.0 + x * (1.0 - sig))
}

#[inline]
fn smooth_switch(dist: f32, cutoff: f32, width: f32) -> f32 {
    let x = ((cutoff - dist) / width).clamp(0.0, 1.0);
    x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)
}

#[inline]
fn smooth_switch_deriv(dist: f32, cutoff: f32, width: f32) -> f32 {
    let raw = (cutoff - dist) / width;
    if raw <= 0.0 || raw >= 1.0 {
        0.0
    } else {
        let omx = 1.0 - raw;
        -(30.0 / width) * raw * raw * omx * omx
    }
}

/// Orthonormal 5-coordinate symmetric-traceless quadrupole `uu^T - I/3`.
#[inline]
fn quadrupole(u: [f32; 3]) -> [f32; 5] {
    let (x, y, z) = (u[0], u[1], u[2]);
    [
        (x * x - y * y) * INV_SQRT_2,
        SQRT_2 * x * y,
        SQRT_2 * x * z,
        SQRT_2 * y * z,
        (x * x + y * y - 2.0 * z * z) * INV_SQRT_6,
    ]
}

/// Matrix-vector product `T(t) * v` for 5-coord symmetric traceless tensor `t` and 3-vector `v`.
#[inline]
fn stf_matvec(t: [f32; 5], v: [f32; 3]) -> [f32; 3] {
    let (a, b, c, d, e) = (t[0], t[1], t[2], t[3], t[4]);
    let txx = a * INV_SQRT_2 + e * INV_SQRT_6;
    let tyy = -a * INV_SQRT_2 + e * INV_SQRT_6;
    let tzz = -2.0 * e * INV_SQRT_6;
    let txy = b * INV_SQRT_2;
    let txz = c * INV_SQRT_2;
    let tyz = d * INV_SQRT_2;
    [
        txx * v[0] + txy * v[1] + txz * v[2],
        txy * v[0] + tyy * v[1] + tyz * v[2],
        txz * v[0] + tyz * v[1] + tzz * v[2],
    ]
}

/// Symmetric-traceless outer product `STV(l, r)` in the 5-coord orthonormal basis.
#[inline]
fn stv_outer(l: [f32; 3], r: [f32; 3]) -> [f32; 5] {
    let vxx = l[0] * r[0];
    let vyy = l[1] * r[1];
    let vzz = l[2] * r[2];
    [
        (vxx - vyy) * INV_SQRT_2,
        (l[0] * r[1] + l[1] * r[0]) * INV_SQRT_2,
        (l[0] * r[2] + l[2] * r[0]) * INV_SQRT_2,
        (l[1] * r[2] + l[2] * r[1]) * INV_SQRT_2,
        (vxx + vyy - 2.0 * vzz) * INV_SQRT_6,
    ]
}

/// Symmetric-traceless square `coords_5(STF(T(t)^2))` matching `_channel_tensor_square`.
#[inline]
fn stf_square(t: [f32; 5]) -> [f32; 5] {
    let (a, b, c, d, e) = (t[0], t[1], t[2], t[3], t[4]);
    let xx = a * INV_SQRT_2 + e * INV_SQRT_6;
    let yy = -a * INV_SQRT_2 + e * INV_SQRT_6;
    let zz = -2.0 * e * INV_SQRT_6;
    let xy = b * INV_SQRT_2;
    let xz = c * INV_SQRT_2;
    let yz = d * INV_SQRT_2;

    let qxx = xx * xx + xy * xy + xz * xz;
    let qyy = yy * yy + xy * xy + yz * yz;
    let qzz = zz * zz + xz * xz + yz * yz;
    let qxy = xy * (xx + yy) + xz * yz;
    let qxz = xz * (xx + zz) + xy * yz;
    let qyz = yz * (yy + zz) + xy * xz;

    [
        (qxx - qyy) * INV_SQRT_2,
        SQRT_2 * qxy,
        SQRT_2 * qxz,
        SQRT_2 * qyz,
        (qxx + qyy - 2.0 * qzz) * INV_SQRT_6,
    ]
}

#[derive(Clone)]
struct SavedLayerNorm {
    x_hat: Vec<f64>, // (N * H)
    rstd: Vec<f64>,  // (N,)
}

/// Exact `layernorm_centered_fp64` forward (`base_model.py:forward_fp64`).
fn layernorm_fp64_fwd(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    rows: usize,
    dim: usize,
) -> (Vec<f32>, SavedLayerNorm) {
    let mut out = vec![0.0f32; rows * dim];
    let mut x_hat = vec![0.0f64; rows * dim];
    let mut rstd = vec![0.0f64; rows];
    let inv_dim = 1.0 / (dim as f64);

    for i in 0..rows {
        let base = i * dim;
        let mut mean = 0.0f64;
        for c in 0..dim {
            mean += x[base + c] as f64;
        }
        mean *= inv_dim;

        let mut var = 0.0f64;
        for c in 0..dim {
            let d = (x[base + c] as f64) - mean;
            var += d * d;
        }
        var *= inv_dim;
        let rho = 1.0 / (var + 1e-5f64).sqrt();
        rstd[i] = rho;

        for c in 0..dim {
            let xh = ((x[base + c] as f64) - mean) * rho;
            x_hat[base + c] = xh;
            out[base + c] = (xh * (gamma[c] as f64) + (beta[c] as f64)) as f32;
        }
    }
    (out, SavedLayerNorm { x_hat, rstd })
}

/// Exact `layernorm_centered_fp64` backward (`layernorm_fp64.py`).
fn layernorm_fp64_bwd_add(
    dy: &[f32],
    gamma: &[f32],
    saved: &SavedLayerNorm,
    rows: usize,
    dim: usize,
    dx_accum: &mut [f32],
) {
    let inv_dim = 1.0 / (dim as f64);
    for i in 0..rows {
        let base = i * dim;
        let rho = saved.rstd[i];
        let mut mean_g = 0.0f64;
        let mut mean_g_xh = 0.0f64;
        for c in 0..dim {
            let g = (dy[base + c] as f64) * (gamma[c] as f64);
            mean_g += g;
            mean_g_xh += g * saved.x_hat[base + c];
        }
        mean_g *= inv_dim;
        mean_g_xh *= inv_dim;
        for c in 0..dim {
            let g = (dy[base + c] as f64) * (gamma[c] as f64);
            let dx = (g - mean_g - saved.x_hat[base + c] * mean_g_xh) * rho;
            dx_accum[base + c] += dx as f32;
        }
    }
}

/// `Y = X * W^T + b` where `X: (rows, in_dim)`, `W: (out_dim, in_dim)`.
fn linear_fwd(
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        let xr = &x[r * in_dim..(r + 1) * in_dim];
        let orow = &mut out[r * out_dim..(r + 1) * out_dim];
        for o in 0..out_dim {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = b.map_or(0.0, |bv| bv[o]);
            for i in 0..in_dim {
                acc += xr[i] * wr[i];
            }
            orow[o] = acc;
        }
    }
    out
}

/// `dX += dY * W` where `dY: (rows, out_dim)`, `W: (out_dim, in_dim)`.
fn linear_bwd_add(
    dy: &[f32],
    w: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    dx: &mut [f32],
) {
    for r in 0..rows {
        let dyr = &dy[r * out_dim..(r + 1) * out_dim];
        let dxr = &mut dx[r * in_dim..(r + 1) * in_dim];
        for o in 0..out_dim {
            let g = dyr[o];
            if g == 0.0 {
                continue;
            }
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            for i in 0..in_dim {
                dxr[i] += g * wr[i];
            }
        }
    }
}

struct BlockForwardTape {
    ln_scalar: SavedLayerNorm,
    sm_pre: Vec<f32>,   // (N, H)
    m_node: Vec<f32>,   // (N, M_b * C)
    rf_pre: Vec<f32>,   // (E, R_H)
    w_rad: Vec<f32>,    // (E, M_b * C)
    env_b: Vec<f32>,    // (E,)
    g_edge: Vec<f32>,   // (E, M_b * C)
    v_msg: Vec<f32>,    // (N * 3, C) if b > 0
    t_msg: Vec<f32>,    // (N * 5, C) if b > 0
    w_tv: Vec<f32>,     // (E * 3, C) if b > 0
    p_t: Vec<f32>,      // (E, C) if b > 0
    b_vt: Vec<f32>,     // (E * 5, C) if b > 0
    p_v: Vec<f32>,      // (E, C) if b > 0
    v_u: Vec<f32>,      // (N * 3, T)
    v_v: Vec<f32>,      // (N * 3, T)
    t_u: Vec<f32>,      // (N * 5, T)
    t_v: Vec<f32>,      // (N * 5, T)
    inv1: Vec<f32>,     // (N, T)
    inv2: Vec<f32>,     // (N, T)
    r_v: Vec<f32>,      // (N, T)
    r_t: Vec<f32>,      // (N, T)
    v_hat: Vec<f32>,    // (N * 3, T)
    t_hat: Vec<f32>,    // (N * 5, T)
    w_hat: Vec<f32>,    // (N * 3, T)
    q2_hat: Vec<f32>,   // (N * 5, T)
    ln_update: SavedLayerNorm,
    exp_pre: Vec<f32>,  // (N, E_H)
    u_out: Vec<f32>,    // (N, O_b)
    d_v: Vec<f32>,      // (N, T)
    d_t: Vec<f32>,      // (N, T)
    c_v: Vec<f32>,      // (N * 3, T) if local_cross
    c_t: Vec<f32>,      // (N * 5, T) if local_cross
    ln_readout: SavedLayerNorm,
    ro_pre: Vec<f32>,   // (N, H)
}

pub struct HyffCpuEngine {
    pub weights: Arc<HyffWeights>,
    neighbor_cache: NeighborCache,
    model_label: String,
}

impl HyffCpuEngine {
    pub fn new(weights: Arc<HyffWeights>) -> Self {
        let cutoff = weights.cfg.cutoff;
        let model_label = format!("hyff-{} (CPU)", weights.cfg.tier);
        Self {
            weights,
            neighbor_cache: NeighborCache::new(cutoff, 0.5),
            model_label,
        }
    }
}

impl PotentialBackend for HyffCpuEngine {
    fn model_name(&self) -> &str {
        &self.model_label
    }

    fn cutoff_a(&self) -> f64 {
        self.weights.cfg.cutoff
    }

    fn eval(
        &mut self,
        system: &MolecularSystem,
        need_forces: bool,
        need_descriptor: bool,
    ) -> Result<PotentialEval, String> {
        // Sync execution with gpu = None never suspends!
        let w = Arc::clone(&self.weights);
        block_ready(eval_hyff_core(
            &w,
            &mut self.neighbor_cache,
            system,
            need_forces,
            need_descriptor,
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

pub(crate) async fn eval_hyff_core(
    w: &HyffWeights,
    neighbor_cache: &mut NeighborCache,
    system: &MolecularSystem,
    need_forces: bool,
    need_descriptor: bool,
    gpu: Option<&HyffGpuContext>,
) -> Result<PotentialEval, String> {
        let n = system.n_atoms();
        if n == 0 {
            return Ok(PotentialEval {
                energy_ev: 0.0,
                forces_ev_a: Vec::new(),
                descriptor: if need_descriptor {
                    Some(Vec::new())
                } else {
                    None
                },
                desc_dim: w.cfg.hidden,
            });
        }

        let experts = w.validate_elements(&system.atomic_numbers)?;
        let experts_u32: Vec<u32> = if gpu.is_some() {
            experts.iter().map(|&k| k as u32).collect()
        } else {
            Vec::new()
        };
        let h = w.cfg.hidden;
        let t = w.cfg.tensor_hidden;
        let c_dim = w.cfg.edge_hidden;
        let k_rbf = w.cfg.n_rbf;
        let r_h = w.cfg.radial_hidden;
        let e_h = w.cfg.expert_hidden;
        let n_int = w.cfg.interactions;
        let rc = w.cfg.cutoff as f32;
        let sw = w.cfg.switch_width as f32;

        // 1. Build directed neighbor list via Verlet skin cache.
        let nl = neighbor_cache.update(&system.coords_a, system.pbc_cell.as_ref());
        let n_edges = nl.edges.len();

        let mut sender = Vec::with_capacity(n_edges);
        let mut receiver = Vec::with_capacity(n_edges);
        let mut dist_e = Vec::with_capacity(n_edges);
        let mut raw_norm_gt_eps = Vec::with_capacity(n_edges);
        let mut u_dir = Vec::with_capacity(n_edges);
        let mut q_quad = Vec::with_capacity(n_edges);
        let mut env_global = Vec::with_capacity(n_edges);
        let mut phi_rbf = vec![0.0f32; n_edges * k_rbf];

        for (e_idx, edge) in nl.edges.iter().enumerate() {
            let j_src = edge.dst as usize;
            let i_dst = edge.src as usize;
            sender.push(j_src);
            receiver.push(i_dst);

            let dx = edge.vec_a[0] as f32;
            let dy = edge.vec_a[1] as f32;
            let dz = edge.vec_a[2] as f32;
            let raw_r = (dx * dx + dy * dy + dz * dz).sqrt();
            raw_norm_gt_eps.push(raw_r > 1e-9);
            let r = raw_r.max(1e-9);
            dist_e.push(r);

            let u = [dx / r, dy / r, dz / r];
            u_dir.push(u);
            q_quad.push(quadrupole(u));
            env_global.push(smooth_switch(r, rc, sw));

            if r < rc {
                let cos_env = 0.5 * ((std::f32::consts::PI * r / rc).cos() + 1.0);
                let base = e_idx * k_rbf;
                for k in 0..k_rbf {
                    let delta = r - w.rbf_centers[k];
                    phi_rbf[base + k] = (-w.rbf_gamma * delta * delta).exp() * cos_env;
                }
            }
        }

        // 2. Initial intensive charge & spin state features + element embedding: s^(0) (N, H)
        let inv_n = 1.0 / (n as f32);
        let f_charge = system.charge * inv_n;
        let f_spin = ((system.multiplicity.max(1) - 1) as f32) * inv_n;
        let mut state_bias = vec![0.0f32; h];
        for c in 0..h {
            state_bias[c] = w.state_proj_w[c * 2] * f_charge + w.state_proj_w[c * 2 + 1] * f_spin;
        }

        let mut s_cur = vec![0.0f32; n * h];
        for i in 0..n {
            let z = system.atomic_numbers[i] as usize;
            let emb_row = &w.embedding_w[z * h..(z + 1) * h];
            for c in 0..h {
                s_cur[i * h + c] = emb_row[c] + state_bias[c];
            }
        }
        let mut v_cur = vec![0.0f32; n * 3 * t];
        let mut t_cur = vec![0.0f32; n * 5 * t];
        let mut e_atom = vec![0.0f32; n];
        let inv_sqrt_i = 1.0f32 / (n_int as f32).sqrt();

        let mut tapes: Vec<BlockForwardTape> = if need_forces {
            Vec::with_capacity(n_int)
        } else {
            Vec::new()
        };
        let mut final_descriptor: Option<Vec<f32>> = None;

        // If GPU is active, pre-evaluate ALL blocks' radial filter MLPs in ONE wgpu submit!
        let mut gpu_rf = if let Some(g) = gpu {
            let (pres, outs) = g.eval_all_radial_filters_fwd(w, &phi_rbf, n_edges).await?;
            Some((pres.into_iter(), outs.into_iter()))
        } else {
            None
        };

        // 3. Forward interaction blocks b = 0 .. I - 1
        for b in 0..n_int {
            let blk = &w.blocks[b];
            let m_b = blk.msg_chunks;
            let mc = m_b * c_dim;

            // Step 1: Bottleneck input (b > 0)
            let (v_msg, t_msg) = if b > 0 {
                let ev_in = blk.edge_vec_in_w.as_ref().unwrap();
                let et_in = blk.edge_ten_in_w.as_ref().unwrap();
                (
                    linear_fwd(&v_cur, ev_in, None, n * 3, t, c_dim),
                    linear_fwd(&t_cur, et_in, None, n * 5, t, c_dim),
                )
            } else {
                (Vec::new(), Vec::new())
            };

            // Step 2: Scalar source message + radial filter + edge gate
            let (s_ln, ln_scalar) =
                layernorm_fp64_fwd(&s_cur, &blk.scalar_norm_w, &blk.scalar_norm_b, n, h);
            let sm_pre = linear_fwd(&s_ln, &blk.sm0_w, Some(&blk.sm0_b), n, h, h);
            let sm_act: Vec<f32> = sm_pre.iter().map(|&x| silu(x)).collect();
            let m_node = linear_fwd(&sm_act, &blk.sm2_w, Some(&blk.sm2_b), n, h, mc);

            let (rf_pre, w_rad) = if let Some((ref mut pres_it, ref mut outs_it)) = gpu_rf {
                (pres_it.next().unwrap(), outs_it.next().unwrap())
            } else {
                let rfp = linear_fwd(&phi_rbf, &blk.rf0_w, Some(&blk.rf0_b), n_edges, k_rbf, r_h);
                let rfa: Vec<f32> = rfp.iter().map(|&x| silu(x)).collect();
                let wr = linear_fwd(&rfa, &blk.rf2_w, Some(&blk.rf2_b), n_edges, r_h, mc);
                (rfp, wr)
            };

            let mut env_b = vec![0.0f32; n_edges];
            let mut g_edge = vec![0.0f32; n_edges * mc];
            for e_idx in 0..n_edges {
                let env_val = smooth_switch(dist_e[e_idx], blk.layer_cutoff, sw);
                env_b[e_idx] = env_val;
                let scale = 0.25 * env_val;
                let j = sender[e_idx];
                let mn = &m_node[j * mc..(j + 1) * mc];
                let wr = &w_rad[e_idx * mc..(e_idx + 1) * mc];
                let ge = &mut g_edge[e_idx * mc..(e_idx + 1) * mc];
                for k in 0..mc {
                    ge[k] = mn[k] * wr[k] * scale;
                }
            }

            // Step 3 & 4: Equivariant edge contractions & receiver scatter-sum
            let mut agg_s = vec![0.0f32; n * c_dim];
            let mut agg_v = vec![0.0f32; n * 3 * c_dim];
            let mut agg_t = vec![0.0f32; n * 5 * c_dim];

            let mut w_tv = if b > 0 && need_forces {
                vec![0.0f32; n_edges * 3 * c_dim]
            } else {
                Vec::new()
            };
            let mut p_t = if b > 0 && need_forces {
                vec![0.0f32; n_edges * c_dim]
            } else {
                Vec::new()
            };
            let mut b_vt = if b > 0 && need_forces {
                vec![0.0f32; n_edges * 5 * c_dim]
            } else {
                Vec::new()
            };
            let mut p_v = if b > 0 && need_forces {
                vec![0.0f32; n_edges * c_dim]
            } else {
                Vec::new()
            };

            if blk.input_is_scalar {
                // M_b = 3: [g_s, g_vr, g_tr]
                for e_idx in 0..n_edges {
                    let i = receiver[e_idx];
                    let u = u_dir[e_idx];
                    let q = q_quad[e_idx];
                    let ge = &g_edge[e_idx * 3 * c_dim..(e_idx + 1) * 3 * c_dim];
                    let gs = &ge[0..c_dim];
                    let gvr = &ge[c_dim..2 * c_dim];
                    let gtr = &ge[2 * c_dim..3 * c_dim];

                    for c in 0..c_dim {
                        agg_s[i * c_dim + c] += gs[c];
                        for a in 0..3 {
                            agg_v[(i * 3 + a) * c_dim + c] += gvr[c] * u[a];
                        }
                        for m in 0..5 {
                            agg_t[(i * 5 + m) * c_dim + c] += gtr[c] * q[m];
                        }
                    }
                }
            } else {
                // M_b = 7: [g_s, g_vr, g_v, g_tv, g_tr, g_t, g_vt]
                for e_idx in 0..n_edges {
                    let j = sender[e_idx];
                    let i = receiver[e_idx];
                    let u = u_dir[e_idx];
                    let q = q_quad[e_idx];
                    let ge = &g_edge[e_idx * 7 * c_dim..(e_idx + 1) * 7 * c_dim];
                    let gs = &ge[0..c_dim];
                    let gvr = &ge[c_dim..2 * c_dim];
                    let gv = &ge[2 * c_dim..3 * c_dim];
                    let gtv = &ge[3 * c_dim..4 * c_dim];
                    let gtr = &ge[4 * c_dim..5 * c_dim];
                    let gt = &ge[5 * c_dim..6 * c_dim];
                    let gvt = &ge[6 * c_dim..7 * c_dim];

                    for c in 0..c_dim {
                        let ve = [
                            v_msg[(j * 3) * c_dim + c],
                            v_msg[(j * 3 + 1) * c_dim + c],
                            v_msg[(j * 3 + 2) * c_dim + c],
                        ];
                        let te = [
                            t_msg[(j * 5) * c_dim + c],
                            t_msg[(j * 5 + 1) * c_dim + c],
                            t_msg[(j * 5 + 2) * c_dim + c],
                            t_msg[(j * 5 + 3) * c_dim + c],
                            t_msg[(j * 5 + 4) * c_dim + c],
                        ];
                        let wtv = stf_matvec(te, u);
                        let pt = u[0] * wtv[0] + u[1] * wtv[1] + u[2] * wtv[2];
                        let bvt = stv_outer(ve, u);
                        let pv = ve[0] * u[0] + ve[1] * u[1] + ve[2] * u[2];

                        if need_forces {
                            for a in 0..3 {
                                w_tv[(e_idx * 3 + a) * c_dim + c] = wtv[a];
                            }
                            p_t[e_idx * c_dim + c] = pt;
                            for m in 0..5 {
                                b_vt[(e_idx * 5 + m) * c_dim + c] = bvt[m];
                            }
                            p_v[e_idx * c_dim + c] = pv;
                        }

                        agg_s[i * c_dim + c] += gs[c] + gvt[c] * pv + gtv[c] * pt;
                        for a in 0..3 {
                            agg_v[(i * 3 + a) * c_dim + c] +=
                                gv[c] * ve[a] + gvr[c] * u[a] + gtv[c] * wtv[a];
                        }
                        for m in 0..5 {
                            agg_t[(i * 5 + m) * c_dim + c] +=
                                gt[c] * te[m] + gtr[c] * q[m] + gvt[c] * bvt[m];
                        }
                    }
                }
            }

            let ds_msg = linear_fwd(&agg_s, &blk.edge_s_out_w, None, n, c_dim, h);
            let dv_msg = linear_fwd(&agg_v, &blk.edge_v_out_w, None, n * 3, c_dim, t);
            let dt_msg = linear_fwd(&agg_t, &blk.edge_t_out_w, None, n * 5, c_dim, t);

            let mut s_tilde = s_cur;
            for idx in 0..s_tilde.len() {
                s_tilde[idx] += ds_msg[idx];
            }
            let mut v_tilde = v_cur;
            for idx in 0..v_tilde.len() {
                v_tilde[idx] += dv_msg[idx];
            }
            let mut t_tilde = t_cur;
            for idx in 0..t_tilde.len() {
                t_tilde[idx] += dt_msg[idx];
            }

            // Step 5: Local channel mixing & 5 equivariant invariants
            let v_u = linear_fwd(&v_tilde, &blk.vector_u_w, None, n * 3, t, t);
            let v_v = linear_fwd(&v_tilde, &blk.vector_v_w, None, n * 3, t, t);
            let t_u = linear_fwd(&t_tilde, &blk.tensor_u_w, None, n * 5, t, t);
            let t_v = linear_fwd(&t_tilde, &blk.tensor_v_w, None, n * 5, t, t);

            let (s_uln, ln_update) =
                layernorm_fp64_fwd(&s_tilde, &blk.update_norm_w, &blk.update_norm_b, n, h);

            let d_in = h + 5 * t;
            let mut z_expert_in = vec![0.0f32; n * d_in];
            let mut inv1 = vec![0.0f32; n * t];
            let mut inv2 = vec![0.0f32; n * t];
            let mut r_v = vec![0.0f32; n * t];
            let mut r_t = vec![0.0f32; n * t];
            let mut v_hat = vec![0.0f32; n * 3 * t];
            let mut t_hat = vec![0.0f32; n * 5 * t];
            let mut w_hat = vec![0.0f32; n * 3 * t];
            let mut q2_hat = vec![0.0f32; n * 5 * t];

            for i in 0..n {
                z_expert_in[i * d_in..i * d_in + h].copy_from_slice(&s_uln[i * h..(i + 1) * h]);
                for c in 0..t {
                    let vv = [
                        v_v[(i * 3) * t + c],
                        v_v[(i * 3 + 1) * t + c],
                        v_v[(i * 3 + 2) * t + c],
                    ];
                    let tv = [
                        t_v[(i * 5) * t + c],
                        t_v[(i * 5 + 1) * t + c],
                        t_v[(i * 5 + 2) * t + c],
                        t_v[(i * 5 + 3) * t + c],
                        t_v[(i * 5 + 4) * t + c],
                    ];
                    let v2 = vv[0] * vv[0] + vv[1] * vv[1] + vv[2] * vv[2];
                    let t2 = tv[0] * tv[0]
                        + tv[1] * tv[1]
                        + tv[2] * tv[2]
                        + tv[3] * tv[3]
                        + tv[4] * tv[4];
                    let i1 = (v2 + 1e-8).sqrt();
                    let i2 = (t2 + 1e-8).sqrt();
                    let rv = 1.0 / (1.0 + v2).sqrt();
                    let rt = 1.0 / (1.0 + t2).sqrt();
                    let vh = [vv[0] * rv, vv[1] * rv, vv[2] * rv];
                    let th = [tv[0] * rt, tv[1] * rt, tv[2] * rt, tv[3] * rt, tv[4] * rt];

                    let wh = stf_matvec(th, vh);
                    let q2h = stf_square(th);

                    let i3 = q2h[0] * th[0]
                        + q2h[1] * th[1]
                        + q2h[2] * th[2]
                        + q2h[3] * th[3]
                        + q2h[4] * th[4];
                    let i4 = vh[0] * wh[0] + vh[1] * wh[1] + vh[2] * wh[2];
                    let i5 = wh[0] * wh[0] + wh[1] * wh[1] + wh[2] * wh[2];

                    inv1[i * t + c] = i1;
                    inv2[i * t + c] = i2;
                    r_v[i * t + c] = rv;
                    r_t[i * t + c] = rt;
                    for a in 0..3 {
                        v_hat[(i * 3 + a) * t + c] = vh[a];
                        w_hat[(i * 3 + a) * t + c] = wh[a];
                    }
                    for m in 0..5 {
                        t_hat[(i * 5 + m) * t + c] = th[m];
                        q2_hat[(i * 5 + m) * t + c] = q2h[m];
                    }

                    z_expert_in[i * d_in + h + c] = i1;
                    z_expert_in[i * d_in + h + t + c] = i2;
                    z_expert_in[i * d_in + h + 2 * t + c] = i3;
                    z_expert_in[i * d_in + h + 3 * t + c] = i4;
                    z_expert_in[i * d_in + h + 4 * t + c] = i5;
                }
            }

            // Step 6: 83-element routed Expert MLP (GPU wgpu shader or CPU)
            let o_b = blk.out_dim;
            let (exp_pre, u_out) = if let Some(g) = gpu {
                g.eval_expert_fwd(w, b, &experts_u32, &z_expert_in, n).await?
            } else {
                let mut ep = vec![0.0f32; n * e_h];
                let mut uo = vec![0.0f32; n * o_b];
                for i in 0..n {
                    let k_exp = experts[i];
                    let z_row = &z_expert_in[i * d_in..(i + 1) * d_in];
                    let w0 = &blk.expert_w0[k_exp * e_h * d_in..(k_exp + 1) * e_h * d_in];
                    let b0 = &blk.expert_b0[k_exp * e_h..(k_exp + 1) * e_h];
                    let w2 = &blk.expert_w2[k_exp * o_b * e_h..(k_exp + 1) * o_b * e_h];
                    let b2 = &blk.expert_b2[k_exp * o_b..(k_exp + 1) * o_b];

                    let mut hidden_act = vec![0.0f32; e_h];
                    for eh in 0..e_h {
                        let w0_row = &w0[eh * d_in..(eh + 1) * d_in];
                        let mut sum = b0[eh];
                        for d in 0..d_in {
                            sum += z_row[d] * w0_row[d];
                        }
                        ep[i * e_h + eh] = sum;
                        hidden_act[eh] = silu(sum);
                    }
                    for o in 0..o_b {
                        let w2_row = &w2[o * e_h..(o + 1) * e_h];
                        let mut sum = b2[o];
                        for eh in 0..e_h {
                            sum += hidden_act[eh] * w2_row[eh];
                        }
                        uo[i * o_b + o] = sum;
                    }
                }
                (ep, uo)
            };

            // Step 7: Residual node updates & per-block readout
            let mut d_v = vec![0.0f32; n * t];
            let mut d_t = vec![0.0f32; n * t];
            let mut gv_dot_dv = vec![0.0f32; n * t];
            let mut gt_dot_dt = vec![0.0f32; n * t];
            for i in 0..n {
                let u_row = &u_out[i * o_b..(i + 1) * o_b];
                let g_vdot = &u_row[h..h + t];
                let g_tdot = if blk.terminal {
                    &u_row[h + t..h + 2 * t]
                } else {
                    &u_row[h + 2 * t..h + 3 * t]
                };
                for c in 0..t {
                    let dv = v_u[(i * 3) * t + c] * v_v[(i * 3) * t + c]
                        + v_u[(i * 3 + 1) * t + c] * v_v[(i * 3 + 1) * t + c]
                        + v_u[(i * 3 + 2) * t + c] * v_v[(i * 3 + 2) * t + c];
                    let dt = t_u[(i * 5) * t + c] * t_v[(i * 5) * t + c]
                        + t_u[(i * 5 + 1) * t + c] * t_v[(i * 5 + 1) * t + c]
                        + t_u[(i * 5 + 2) * t + c] * t_v[(i * 5 + 2) * t + c]
                        + t_u[(i * 5 + 3) * t + c] * t_v[(i * 5 + 3) * t + c]
                        + t_u[(i * 5 + 4) * t + c] * t_v[(i * 5 + 4) * t + c];
                    d_v[i * t + c] = dv;
                    d_t[i * t + c] = dt;
                    gv_dot_dv[i * t + c] = g_vdot[c] * dv;
                    gt_dot_dt[i * t + c] = g_tdot[c] * dt;
                }
            }

            let inv_v_contrib = linear_fwd(&gv_dot_dv, &blk.inv_vec_out_w, None, n, t, h);
            let inv_t_contrib = linear_fwd(&gt_dot_dt, &blk.inv_ten_out_w, None, n, t, h);

            let mut s_next = s_tilde;
            for i in 0..n {
                let u_row = &u_out[i * o_b..(i + 1) * o_b];
                for c in 0..h {
                    s_next[i * h + c] +=
                        u_row[c] + inv_v_contrib[i * h + c] + inv_t_contrib[i * h + c];
                }
            }

            let mut c_v = Vec::new();
            let mut c_t = Vec::new();
            let (v_next, t_next) = if blk.terminal {
                (v_tilde, t_tilde)
            } else if blk.local_cross {
                c_v = vec![0.0f32; n * 3 * t];
                c_t = vec![0.0f32; n * 5 * t];
                let vt_gain = blk.vector_tensor_gain.as_ref().unwrap();
                let vv_gain = blk.vector_vector_gain.as_ref().unwrap();
                let mut vn = v_tilde;
                let mut tn = t_tilde;
                for i in 0..n {
                    let u_row = &u_out[i * o_b..(i + 1) * o_b];
                    let g_v = &u_row[h + t..h + 2 * t];
                    let g_t = &u_row[h + 3 * t..h + 4 * t];
                    for c in 0..t {
                        let a_vt = vt_gain[c].tanh();
                        let a_vv = vv_gain[c].tanh();
                        let tu_c = [
                            t_u[(i * 5) * t + c],
                            t_u[(i * 5 + 1) * t + c],
                            t_u[(i * 5 + 2) * t + c],
                            t_u[(i * 5 + 3) * t + c],
                            t_u[(i * 5 + 4) * t + c],
                        ];
                        let vu_c = [
                            v_u[(i * 3) * t + c],
                            v_u[(i * 3 + 1) * t + c],
                            v_u[(i * 3 + 2) * t + c],
                        ];
                        let vv_c = [
                            v_v[(i * 3) * t + c],
                            v_v[(i * 3 + 1) * t + c],
                            v_v[(i * 3 + 2) * t + c],
                        ];
                        let cv = stf_matvec(tu_c, vv_c);
                        let ct = stv_outer(vu_c, vv_c);
                        for a in 0..3 {
                            c_v[(i * 3 + a) * t + c] = cv[a];
                            vn[(i * 3 + a) * t + c] += g_v[c] * (vu_c[a] + a_vt * cv[a]);
                        }
                        for m in 0..5 {
                            c_t[(i * 5 + m) * t + c] = ct[m];
                            tn[(i * 5 + m) * t + c] += g_t[c] * (tu_c[m] + a_vv * ct[m]);
                        }
                    }
                }
                (vn, tn)
            } else {
                let mut vn = v_tilde;
                let mut tn = t_tilde;
                for i in 0..n {
                    let u_row = &u_out[i * o_b..(i + 1) * o_b];
                    let g_v = &u_row[h + t..h + 2 * t];
                    let g_t = &u_row[h + 3 * t..h + 4 * t];
                    for c in 0..t {
                        for a in 0..3 {
                            vn[(i * 3 + a) * t + c] += g_v[c] * v_u[(i * 3 + a) * t + c];
                        }
                        for m in 0..5 {
                            tn[(i * 5 + m) * t + c] += g_t[c] * t_u[(i * 5 + m) * t + c];
                        }
                    }
                }
                (vn, tn)
            };

            let (h_ro, ln_readout) =
                layernorm_fp64_fwd(&s_next, &blk.readout_norm_w, &blk.readout_norm_b, n, h);
            if b == n_int - 1 && need_descriptor {
                final_descriptor = Some(h_ro.clone());
            }

            let ro_pre = linear_fwd(&h_ro, &blk.ro0_w, Some(&blk.ro0_b), n, h, h);
            for i in 0..n {
                let mut dot = blk.ro2_b;
                for c in 0..h {
                    dot += silu(ro_pre[i * h + c]) * blk.ro2_w[c];
                }
                e_atom[i] += inv_sqrt_i * dot;
            }

            s_cur = s_next;
            v_cur = v_next;
            t_cur = t_next;

            if need_forces {
                tapes.push(BlockForwardTape {
                    ln_scalar,
                    sm_pre,
                    m_node,
                    rf_pre,
                    w_rad,
                    env_b,
                    g_edge,
                    v_msg,
                    t_msg,
                    w_tv,
                    p_t,
                    b_vt,
                    p_v,
                    v_u,
                    v_v,
                    t_u,
                    t_v,
                    inv1,
                    inv2,
                    r_v,
                    r_t,
                    v_hat,
                    t_hat,
                    w_hat,
                    q2_hat,
                    ln_update,
                    exp_pre,
                    u_out,
                    d_v,
                    d_t,
                    c_v,
                    c_t,
                    ln_readout,
                    ro_pre,
                });
            }
        }

        // 4. Total Energy Reduction in FP64 (`stable_sum=True`: atom readouts + element_offset + symmetric pair_coeff)
        let mut total_energy_ev = 0.0f64;
        for i in 0..n {
            let z = system.atomic_numbers[i] as usize;
            total_energy_ev += (e_atom[i] as f64) + w.element_offset_f64[z];
        }

        let mut d_phi = if need_forces {
            vec![0.0f32; n_edges * k_rbf]
        } else {
            Vec::new()
        };
        let mut d_r = if need_forces {
            vec![0.0f32; n_edges]
        } else {
            Vec::new()
        };

        for e_idx in 0..n_edges {
            let cj = experts[sender[e_idx]];
            let ci = experts[receiver[e_idx]];
            let lo = cj.min(ci);
            let hi = cj.max(ci);
            let pair_idx = w.cfg.expert_count * lo + hi;
            let w_pair = &w.pair_coeff_w[pair_idx * k_rbf..(pair_idx + 1) * k_rbf];
            let phi_e = &phi_rbf[e_idx * k_rbf..(e_idx + 1) * k_rbf];

            let mut s_e = 0.0f32;
            for k in 0..k_rbf {
                s_e += w_pair[k] * phi_e[k];
            }
            let e_pair = 0.5 * s_e * env_global[e_idx];
            total_energy_ev += e_pair as f64;

            if need_forces {
                let half_env = 0.5 * env_global[e_idx];
                for k in 0..k_rbf {
                    d_phi[e_idx * k_rbf + k] = half_env * w_pair[k];
                }
                d_r[e_idx] = 0.5 * s_e * smooth_switch_deriv(dist_e[e_idx], rc, sw);
            }
        }

        if !need_forces {
            return Ok(PotentialEval {
                energy_ev: total_energy_ev,
                forces_ev_a: Vec::new(),
                descriptor: final_descriptor,
                desc_dim: h,
            });
        }

        // 5. Analytical Reverse-Mode VJP (`F = -dE/dR`)
        let mut d_u_dir = vec![[0.0f32; 3]; n_edges];
        let mut d_q_quad = vec![[0.0f32; 5]; n_edges];
        let mut d_s = vec![0.0f32; n * h];
        let mut d_v = vec![0.0f32; n * 3 * t];
        let mut d_t = vec![0.0f32; n * 5 * t];

        for b in (0..n_int).rev() {
            let blk = &w.blocks[b];
            let tape = &tapes[b];
            let m_b = blk.msg_chunks;
            let mc = m_b * c_dim;
            let o_b = blk.out_dim;
            let d_in = h + 5 * t;

            // B.1 Readout Head b VJP (`d_e_atom == 1.0` for all atoms)
            let mut d_ro_pre = vec![0.0f32; n * h];
            for i in 0..n {
                for c in 0..h {
                    let g_act = inv_sqrt_i * blk.ro2_w[c];
                    d_ro_pre[i * h + c] = g_act * silu_grad(tape.ro_pre[i * h + c]);
                }
            }
            let mut d_h_ro = vec![0.0f32; n * h];
            linear_bwd_add(&d_ro_pre, &blk.ro0_w, n, h, h, &mut d_h_ro);
            layernorm_fp64_bwd_add(
                &d_h_ro,
                &blk.readout_norm_w,
                &tape.ln_readout,
                n,
                h,
                &mut d_s,
            );

            // B.2 Residual Node Update VJP
            let ds_next = d_s;
            let dv_next = d_v;
            let dt_next = d_t;

            let mut d_s_tilde = ds_next.clone();
            let mut d_v_tilde = dv_next.clone();
            let mut d_t_tilde = dt_next.clone();

            let mut d_v_inv = vec![0.0f32; n * t];
            let mut d_t_inv = vec![0.0f32; n * t];
            linear_bwd_add(&ds_next, &blk.inv_vec_out_w, n, t, h, &mut d_v_inv);
            linear_bwd_add(&ds_next, &blk.inv_ten_out_w, n, t, h, &mut d_t_inv);

            let mut d_vu = vec![0.0f32; n * 3 * t];
            let mut d_vv = vec![0.0f32; n * 3 * t];
            let mut d_tu = vec![0.0f32; n * 5 * t];
            let mut d_tv = vec![0.0f32; n * 5 * t];
            let mut d_u_out = vec![0.0f32; n * o_b];

            for i in 0..n {
                let u_row = &tape.u_out[i * o_b..(i + 1) * o_b];
                let g_vdot = &u_row[h..h + t];
                let g_tdot = if blk.terminal {
                    &u_row[h + t..h + 2 * t]
                } else {
                    &u_row[h + 2 * t..h + 3 * t]
                };

                // d_Delta_s = ds_next
                d_u_out[i * o_b..i * o_b + h].copy_from_slice(&ds_next[i * h..(i + 1) * h]);

                for c in 0..t {
                    let dvi = d_v_inv[i * t + c];
                    let dti = d_t_inv[i * t + c];
                    let d_gvdot = dvi * tape.d_v[i * t + c];
                    let d_dv = dvi * g_vdot[c];
                    let d_gtdot = dti * tape.d_t[i * t + c];
                    let d_dt = dti * g_tdot[c];

                    d_u_out[i * o_b + h + c] = d_gvdot;
                    if blk.terminal {
                        d_u_out[i * o_b + h + t + c] = d_gtdot;
                    } else {
                        d_u_out[i * o_b + h + 2 * t + c] = d_gtdot;
                    }

                    for a in 0..3 {
                        d_vu[(i * 3 + a) * t + c] += d_dv * tape.v_v[(i * 3 + a) * t + c];
                        d_vv[(i * 3 + a) * t + c] += d_dv * tape.v_u[(i * 3 + a) * t + c];
                    }
                    for m in 0..5 {
                        d_tu[(i * 5 + m) * t + c] += d_dt * tape.t_v[(i * 5 + m) * t + c];
                        d_tv[(i * 5 + m) * t + c] += d_dt * tape.t_u[(i * 5 + m) * t + c];
                    }
                }

                if !blk.terminal {
                    let g_v = &u_row[h + t..h + 2 * t];
                    let g_t = &u_row[h + 3 * t..h + 4 * t];
                    if blk.local_cross {
                        let vt_gain = blk.vector_tensor_gain.as_ref().unwrap();
                        let vv_gain = blk.vector_vector_gain.as_ref().unwrap();
                        for c in 0..t {
                            let a_vt = vt_gain[c].tanh();
                            let a_vv = vv_gain[c].tanh();
                            let dvn = [
                                dv_next[(i * 3) * t + c],
                                dv_next[(i * 3 + 1) * t + c],
                                dv_next[(i * 3 + 2) * t + c],
                            ];
                            let dtn = [
                                dt_next[(i * 5) * t + c],
                                dt_next[(i * 5 + 1) * t + c],
                                dt_next[(i * 5 + 2) * t + c],
                                dt_next[(i * 5 + 3) * t + c],
                                dt_next[(i * 5 + 4) * t + c],
                            ];
                            let mut dg_v = 0.0f32;
                            for a in 0..3 {
                                dg_v += dvn[a]
                                    * (tape.v_u[(i * 3 + a) * t + c]
                                        + a_vt * tape.c_v[(i * 3 + a) * t + c]);
                            }
                            let mut dg_t = 0.0f32;
                            for m in 0..5 {
                                dg_t += dtn[m]
                                    * (tape.t_u[(i * 5 + m) * t + c]
                                        + a_vv * tape.c_t[(i * 5 + m) * t + c]);
                            }
                            d_u_out[i * o_b + h + t + c] = dg_v;
                            d_u_out[i * o_b + h + 3 * t + c] = dg_t;

                            let gv_tilde = [g_v[c] * dvn[0], g_v[c] * dvn[1], g_v[c] * dvn[2]];
                            let gt_tilde = [
                                g_t[c] * dtn[0],
                                g_t[c] * dtn[1],
                                g_t[c] * dtn[2],
                                g_t[c] * dtn[3],
                                g_t[c] * dtn[4],
                            ];
                            let d_cv = [
                                a_vt * gv_tilde[0],
                                a_vt * gv_tilde[1],
                                a_vt * gv_tilde[2],
                            ];
                            let d_ct = [
                                a_vv * gt_tilde[0],
                                a_vv * gt_tilde[1],
                                a_vv * gt_tilde[2],
                                a_vv * gt_tilde[3],
                                a_vv * gt_tilde[4],
                            ];

                            let vu_c = [
                                tape.v_u[(i * 3) * t + c],
                                tape.v_u[(i * 3 + 1) * t + c],
                                tape.v_u[(i * 3 + 2) * t + c],
                            ];
                            let vv_c = [
                                tape.v_v[(i * 3) * t + c],
                                tape.v_v[(i * 3 + 1) * t + c],
                                tape.v_v[(i * 3 + 2) * t + c],
                            ];
                            let tu_c = [
                                tape.t_u[(i * 5) * t + c],
                                tape.t_u[(i * 5 + 1) * t + c],
                                tape.t_u[(i * 5 + 2) * t + c],
                                tape.t_u[(i * 5 + 3) * t + c],
                                tape.t_u[(i * 5 + 4) * t + c],
                            ];

                            let t_dct_vv = stf_matvec(d_ct, vv_c);
                            let stv_dcv_vv = stv_outer(d_cv, vv_c);
                            let t_tu_dcv = stf_matvec(tu_c, d_cv);
                            let t_dct_vu = stf_matvec(d_ct, vu_c);

                            for a in 0..3 {
                                d_vu[(i * 3 + a) * t + c] += gv_tilde[a] + t_dct_vv[a];
                                d_vv[(i * 3 + a) * t + c] += t_tu_dcv[a] + t_dct_vu[a];
                            }
                            for m in 0..5 {
                                d_tu[(i * 5 + m) * t + c] += gt_tilde[m] + stv_dcv_vv[m];
                            }
                        }
                    } else {
                        for c in 0..t {
                            let mut dg_v = 0.0f32;
                            for a in 0..3 {
                                let dvn = dv_next[(i * 3 + a) * t + c];
                                dg_v += dvn * tape.v_u[(i * 3 + a) * t + c];
                                d_vu[(i * 3 + a) * t + c] += g_v[c] * dvn;
                            }
                            let mut dg_t = 0.0f32;
                            for m in 0..5 {
                                let dtn = dt_next[(i * 5 + m) * t + c];
                                dg_t += dtn * tape.t_u[(i * 5 + m) * t + c];
                                d_tu[(i * 5 + m) * t + c] += g_t[c] * dtn;
                            }
                            d_u_out[i * o_b + h + t + c] = dg_v;
                            d_u_out[i * o_b + h + 3 * t + c] = dg_t;
                        }
                    }
                }
            }

            // B.3 83-Element Routed Expert MLP VJP (GPU wgpu shader or CPU)
            let d_z_in = if let Some(g) = gpu {
                let (dz, _) = g
                    .eval_expert_and_radial_bwd(
                        w,
                        b,
                        &experts_u32,
                        &d_u_out,
                        &tape.exp_pre,
                        &[],
                        &[],
                        n,
                        0,
                    )
                    .await?;
                dz
            } else {
                let mut dz = vec![0.0f32; n * d_in];
                for i in 0..n {
                    let k_exp = experts[i];
                    let du_row = &d_u_out[i * o_b..(i + 1) * o_b];
                    let w0 = &blk.expert_w0[k_exp * e_h * d_in..(k_exp + 1) * e_h * d_in];
                    let w2 = &blk.expert_w2[k_exp * o_b * e_h..(k_exp + 1) * o_b * e_h];

                    let mut d_hidden = vec![0.0f32; e_h];
                    for o in 0..o_b {
                        let g_val = du_row[o];
                        if g_val == 0.0 {
                            continue;
                        }
                        let w2_row = &w2[o * e_h..(o + 1) * e_h];
                        for eh in 0..e_h {
                            d_hidden[eh] += g_val * w2_row[eh];
                        }
                    }
                    let dz_row = &mut dz[i * d_in..(i + 1) * d_in];
                    for eh in 0..e_h {
                        let g_pre = d_hidden[eh] * silu_grad(tape.exp_pre[i * e_h + eh]);
                        if g_pre == 0.0 {
                            continue;
                        }
                        let w0_row = &w0[eh * d_in..(eh + 1) * d_in];
                        for d in 0..d_in {
                            dz_row[d] += g_pre * w0_row[d];
                        }
                    }
                }
                dz
            };

            let mut d_uln = vec![0.0f32; n * h];
            for i in 0..n {
                d_uln[i * h..(i + 1) * h].copy_from_slice(&d_z_in[i * d_in..i * d_in + h]);
            }
            layernorm_fp64_bwd_add(
                &d_uln,
                &blk.update_norm_w,
                &tape.ln_update,
                n,
                h,
                &mut d_s_tilde,
            );

            // B.4 Norms & Higher Invariants VJP
            for i in 0..n {
                let dz = &d_z_in[i * d_in..(i + 1) * d_in];
                for c in 0..t {
                    let di1 = dz[h + c];
                    let di2 = dz[h + t + c];
                    let di3 = dz[h + 2 * t + c];
                    let di4 = dz[h + 3 * t + c];
                    let di5 = dz[h + 4 * t + c];

                    let s1 = di1 / tape.inv1[i * t + c];
                    let s2 = di2 / tape.inv2[i * t + c];
                    for a in 0..3 {
                        d_vv[(i * 3 + a) * t + c] += s1 * tape.v_v[(i * 3 + a) * t + c];
                    }
                    for m in 0..5 {
                        d_tv[(i * 5 + m) * t + c] += s2 * tape.t_v[(i * 5 + m) * t + c];
                    }

                    let vh = [
                        tape.v_hat[(i * 3) * t + c],
                        tape.v_hat[(i * 3 + 1) * t + c],
                        tape.v_hat[(i * 3 + 2) * t + c],
                    ];
                    let th = [
                        tape.t_hat[(i * 5) * t + c],
                        tape.t_hat[(i * 5 + 1) * t + c],
                        tape.t_hat[(i * 5 + 2) * t + c],
                        tape.t_hat[(i * 5 + 3) * t + c],
                        tape.t_hat[(i * 5 + 4) * t + c],
                    ];
                    let wh = [
                        tape.w_hat[(i * 3) * t + c],
                        tape.w_hat[(i * 3 + 1) * t + c],
                        tape.w_hat[(i * 3 + 2) * t + c],
                    ];
                    let q2h = [
                        tape.q2_hat[(i * 5) * t + c],
                        tape.q2_hat[(i * 5 + 1) * t + c],
                        tape.q2_hat[(i * 5 + 2) * t + c],
                        tape.q2_hat[(i * 5 + 3) * t + c],
                        tape.q2_hat[(i * 5 + 4) * t + c],
                    ];
                    let w2h = stf_matvec(th, wh);
                    let q_vv = quadrupole(vh); // STV(vh, vh) == quadrupole(vh)!
                    let b_wv = stv_outer(wh, vh);

                    let g_vh = [
                        2.0 * (di4 * wh[0] + di5 * w2h[0]),
                        2.0 * (di4 * wh[1] + di5 * w2h[1]),
                        2.0 * (di4 * wh[2] + di5 * w2h[2]),
                    ];
                    let g_th = [
                        3.0 * di3 * q2h[0] + di4 * q_vv[0] + 2.0 * di5 * b_wv[0],
                        3.0 * di3 * q2h[1] + di4 * q_vv[1] + 2.0 * di5 * b_wv[1],
                        3.0 * di3 * q2h[2] + di4 * q_vv[2] + 2.0 * di5 * b_wv[2],
                        3.0 * di3 * q2h[3] + di4 * q_vv[3] + 2.0 * di5 * b_wv[3],
                        3.0 * di3 * q2h[4] + di4 * q_vv[4] + 2.0 * di5 * b_wv[4],
                    ];

                    let dot_v = g_vh[0] * vh[0] + g_vh[1] * vh[1] + g_vh[2] * vh[2];
                    let rv = tape.r_v[i * t + c];
                    for a in 0..3 {
                        d_vv[(i * 3 + a) * t + c] += (g_vh[a] - dot_v * vh[a]) * rv;
                    }

                    let dot_t = g_th[0] * th[0]
                        + g_th[1] * th[1]
                        + g_th[2] * th[2]
                        + g_th[3] * th[3]
                        + g_th[4] * th[4];
                    let rt = tape.r_t[i * t + c];
                    for m in 0..5 {
                        d_tv[(i * 5 + m) * t + c] += (g_th[m] - dot_t * th[m]) * rt;
                    }
                }
            }

            linear_bwd_add(&d_vu, &blk.vector_u_w, n * 3, t, t, &mut d_v_tilde);
            linear_bwd_add(&d_vv, &blk.vector_v_w, n * 3, t, t, &mut d_v_tilde);
            linear_bwd_add(&d_tu, &blk.tensor_u_w, n * 5, t, t, &mut d_t_tilde);
            linear_bwd_add(&d_tv, &blk.tensor_v_w, n * 5, t, t, &mut d_t_tilde);

            // B.5 Bottleneck Output & Receiver Gather VJP
            d_s = d_s_tilde;
            d_v = d_v_tilde;
            d_t = d_t_tilde;

            let mut d_agg_s = vec![0.0f32; n * c_dim];
            let mut d_agg_v = vec![0.0f32; n * 3 * c_dim];
            let mut d_agg_t = vec![0.0f32; n * 5 * c_dim];
            linear_bwd_add(&d_s, &blk.edge_s_out_w, n, c_dim, h, &mut d_agg_s);
            linear_bwd_add(&d_v, &blk.edge_v_out_w, n * 3, c_dim, t, &mut d_agg_v);
            linear_bwd_add(&d_t, &blk.edge_t_out_w, n * 5, c_dim, t, &mut d_agg_t);

            // B.6 Edge Contractions VJP
            let mut d_g_edge = vec![0.0f32; n_edges * mc];
            if blk.input_is_scalar {
                for e_idx in 0..n_edges {
                    let i = receiver[e_idx];
                    let u = u_dir[e_idx];
                    let q = q_quad[e_idx];
                    let ge = &tape.g_edge[e_idx * 3 * c_dim..(e_idx + 1) * 3 * c_dim];
                    let gvr = &ge[c_dim..2 * c_dim];
                    let gtr = &ge[2 * c_dim..3 * c_dim];
                    let dge = &mut d_g_edge[e_idx * 3 * c_dim..(e_idx + 1) * 3 * c_dim];

                    for c in 0..c_dim {
                        let gam_s = d_agg_s[i * c_dim + c];
                        let gam_v = [
                            d_agg_v[(i * 3) * c_dim + c],
                            d_agg_v[(i * 3 + 1) * c_dim + c],
                            d_agg_v[(i * 3 + 2) * c_dim + c],
                        ];
                        let gam_t = [
                            d_agg_t[(i * 5) * c_dim + c],
                            d_agg_t[(i * 5 + 1) * c_dim + c],
                            d_agg_t[(i * 5 + 2) * c_dim + c],
                            d_agg_t[(i * 5 + 3) * c_dim + c],
                            d_agg_t[(i * 5 + 4) * c_dim + c],
                        ];
                        dge[c] = gam_s;
                        dge[c_dim + c] = gam_v[0] * u[0] + gam_v[1] * u[1] + gam_v[2] * u[2];
                        dge[2 * c_dim + c] = gam_t[0] * q[0]
                            + gam_t[1] * q[1]
                            + gam_t[2] * q[2]
                            + gam_t[3] * q[3]
                            + gam_t[4] * q[4];

                        for a in 0..3 {
                            d_u_dir[e_idx][a] += gvr[c] * gam_v[a];
                        }
                        for m in 0..5 {
                            d_q_quad[e_idx][m] += gtr[c] * gam_t[m];
                        }
                    }
                }
            } else {
                let mut d_v_msg = vec![0.0f32; n * 3 * c_dim];
                let mut d_t_msg = vec![0.0f32; n * 5 * c_dim];

                for e_idx in 0..n_edges {
                    let j = sender[e_idx];
                    let i = receiver[e_idx];
                    let u = u_dir[e_idx];
                    let q = q_quad[e_idx];
                    let ge = &tape.g_edge[e_idx * 7 * c_dim..(e_idx + 1) * 7 * c_dim];
                    let gvr = &ge[c_dim..2 * c_dim];
                    let gv = &ge[2 * c_dim..3 * c_dim];
                    let gtv = &ge[3 * c_dim..4 * c_dim];
                    let gtr = &ge[4 * c_dim..5 * c_dim];
                    let gt = &ge[5 * c_dim..6 * c_dim];
                    let gvt = &ge[6 * c_dim..7 * c_dim];
                    let dge = &mut d_g_edge[e_idx * 7 * c_dim..(e_idx + 1) * 7 * c_dim];

                    for c in 0..c_dim {
                        let gam_s = d_agg_s[i * c_dim + c];
                        let gam_v = [
                            d_agg_v[(i * 3) * c_dim + c],
                            d_agg_v[(i * 3 + 1) * c_dim + c],
                            d_agg_v[(i * 3 + 2) * c_dim + c],
                        ];
                        let gam_t = [
                            d_agg_t[(i * 5) * c_dim + c],
                            d_agg_t[(i * 5 + 1) * c_dim + c],
                            d_agg_t[(i * 5 + 2) * c_dim + c],
                            d_agg_t[(i * 5 + 3) * c_dim + c],
                            d_agg_t[(i * 5 + 4) * c_dim + c],
                        ];
                        let ve = [
                            tape.v_msg[(j * 3) * c_dim + c],
                            tape.v_msg[(j * 3 + 1) * c_dim + c],
                            tape.v_msg[(j * 3 + 2) * c_dim + c],
                        ];
                        let te = [
                            tape.t_msg[(j * 5) * c_dim + c],
                            tape.t_msg[(j * 5 + 1) * c_dim + c],
                            tape.t_msg[(j * 5 + 2) * c_dim + c],
                            tape.t_msg[(j * 5 + 3) * c_dim + c],
                            tape.t_msg[(j * 5 + 4) * c_dim + c],
                        ];
                        let wtv = [
                            tape.w_tv[(e_idx * 3) * c_dim + c],
                            tape.w_tv[(e_idx * 3 + 1) * c_dim + c],
                            tape.w_tv[(e_idx * 3 + 2) * c_dim + c],
                        ];
                        let pt = tape.p_t[e_idx * c_dim + c];
                        let bvt = [
                            tape.b_vt[(e_idx * 5) * c_dim + c],
                            tape.b_vt[(e_idx * 5 + 1) * c_dim + c],
                            tape.b_vt[(e_idx * 5 + 2) * c_dim + c],
                            tape.b_vt[(e_idx * 5 + 3) * c_dim + c],
                            tape.b_vt[(e_idx * 5 + 4) * c_dim + c],
                        ];
                        let pv = tape.p_v[e_idx * c_dim + c];

                        let a_gu = stf_matvec(gam_t, u);
                        let a_tg = stf_matvec(te, gam_v);
                        let a_gv = stf_matvec(gam_t, ve);

                        dge[c] = gam_s;
                        dge[c_dim + c] = gam_v[0] * u[0] + gam_v[1] * u[1] + gam_v[2] * u[2];
                        dge[2 * c_dim + c] =
                            gam_v[0] * ve[0] + gam_v[1] * ve[1] + gam_v[2] * ve[2];
                        dge[3 * c_dim + c] =
                            gam_s * pt + (gam_v[0] * wtv[0] + gam_v[1] * wtv[1] + gam_v[2] * wtv[2]);
                        dge[4 * c_dim + c] = gam_t[0] * q[0]
                            + gam_t[1] * q[1]
                            + gam_t[2] * q[2]
                            + gam_t[3] * q[3]
                            + gam_t[4] * q[4];
                        dge[5 * c_dim + c] = gam_t[0] * te[0]
                            + gam_t[1] * te[1]
                            + gam_t[2] * te[2]
                            + gam_t[3] * te[3]
                            + gam_t[4] * te[4];
                        dge[6 * c_dim + c] = gam_s * pv
                            + (gam_t[0] * bvt[0]
                                + gam_t[1] * bvt[1]
                                + gam_t[2] * bvt[2]
                                + gam_t[3] * bvt[3]
                                + gam_t[4] * bvt[4]);

                        let stv_gv_u = stv_outer(gam_v, u);
                        for a in 0..3 {
                            d_v_msg[(j * 3 + a) * c_dim + c] +=
                                gv[c] * gam_v[a] + gvt[c] * (gam_s * u[a] + a_gu[a]);
                            d_u_dir[e_idx][a] += gvr[c] * gam_v[a]
                                + gtv[c] * (2.0 * gam_s * wtv[a] + a_tg[a])
                                + gvt[c] * (gam_s * ve[a] + a_gv[a]);
                        }
                        for m in 0..5 {
                            d_t_msg[(j * 5 + m) * c_dim + c] +=
                                gt[c] * gam_t[m] + gtv[c] * (gam_s * q[m] + stv_gv_u[m]);
                            d_q_quad[e_idx][m] += gtr[c] * gam_t[m];
                        }
                    }
                }

                let ev_in = blk.edge_vec_in_w.as_ref().unwrap();
                let et_in = blk.edge_ten_in_w.as_ref().unwrap();
                linear_bwd_add(&d_v_msg, ev_in, n * 3, t, c_dim, &mut d_v);
                linear_bwd_add(&d_t_msg, et_in, n * 5, t, c_dim, &mut d_t);
            }

            // B.7 Edge Gate, Radial Filter & Scalar Message VJP
            let mut d_m_node = vec![0.0f32; n * mc];
            let mut d_w_rad = vec![0.0f32; n_edges * mc];
            let d_env_deriv = |r: f32| smooth_switch_deriv(r, blk.layer_cutoff, sw);

            for e_idx in 0..n_edges {
                let j = sender[e_idx];
                let scale = 0.25 * tape.env_b[e_idx];
                let dge = &d_g_edge[e_idx * mc..(e_idx + 1) * mc];
                let mn = &tape.m_node[j * mc..(j + 1) * mc];
                let wr = &tape.w_rad[e_idx * mc..(e_idx + 1) * mc];
                let dmn = &mut d_m_node[j * mc..(j + 1) * mc];
                let dwr = &mut d_w_rad[e_idx * mc..(e_idx + 1) * mc];

                let mut dot_env = 0.0f32;
                for k in 0..mc {
                    dmn[k] += dge[k] * wr[k] * scale;
                    dwr[k] += dge[k] * mn[k] * scale;
                    dot_env += dge[k] * mn[k] * wr[k];
                }
                d_r[e_idx] += 0.25 * dot_env * d_env_deriv(dist_e[e_idx]);
            }

            // Radial filter MLP backward -> d_phi
            let mut d_rf_act = vec![0.0f32; n_edges * r_h];
            linear_bwd_add(&d_w_rad, &blk.rf2_w, n_edges, r_h, mc, &mut d_rf_act);
            for idx in 0..d_rf_act.len() {
                d_rf_act[idx] *= silu_grad(tape.rf_pre[idx]);
            }
            linear_bwd_add(&d_rf_act, &blk.rf0_w, n_edges, k_rbf, r_h, &mut d_phi);

            // Scalar message MLP & LayerNorm backward -> d_s
            let mut d_sm_act = vec![0.0f32; n * h];
            linear_bwd_add(&d_m_node, &blk.sm2_w, n, h, mc, &mut d_sm_act);
            for idx in 0..d_sm_act.len() {
                d_sm_act[idx] *= silu_grad(tape.sm_pre[idx]);
            }
            let mut d_s_ln = vec![0.0f32; n * h];
            linear_bwd_add(&d_sm_act, &blk.sm0_w, n, h, h, &mut d_s_ln);
            layernorm_fp64_bwd_add(
                &d_s_ln,
                &blk.scalar_norm_w,
                &tape.ln_scalar,
                n,
                h,
                &mut d_s,
            );
        }

        // Step C: Geometry VJP (`d_q_quad, d_u_dir, d_phi, d_r -> d_vec -> F`)
        let mut forces_ev_a = vec![[0.0f64; 3]; n];
        for e_idx in 0..n_edges {
            let u = u_dir[e_idx];
            let r = dist_e[e_idx];

            // 1. Quadrupole q_e(u_e) -> d_u_dir
            let t_dq_u = stf_matvec(d_q_quad[e_idx], u);
            let du = [
                d_u_dir[e_idx][0] + 2.0 * t_dq_u[0],
                d_u_dir[e_idx][1] + 2.0 * t_dq_u[1],
                d_u_dir[e_idx][2] + 2.0 * t_dq_u[2],
            ];

            // 2. RadialBasis phi_e(r_e) -> d_r
            let mut dr = d_r[e_idx];
            if r < rc {
                let cos_env = 0.5 * ((std::f32::consts::PI * r / rc).cos() + 1.0);
                let d_cos_env = -0.5 * (std::f32::consts::PI / rc) * (std::f32::consts::PI * r / rc).sin();
                for k in 0..k_rbf {
                    let delta = r - w.rbf_centers[k];
                    let g_ek = (-w.rbf_gamma * delta * delta).exp();
                    let dphi_dr = (-2.0 * w.rbf_gamma * delta * cos_env + d_cos_env) * g_ek;
                    dr += d_phi[e_idx * k_rbf + k] * dphi_dr;
                }
            }

            // 3. Unit direction & clamped norm -> d_raw_vec
            let du_dot_u = du[0] * u[0] + du[1] * u[1] + du[2] * u[2];
            let dr_eff = if raw_norm_gt_eps[e_idx] { dr } else { 0.0 };
            let inv_r = 1.0 / r;
            let dd = [
                (dr_eff * u[0] + (du[0] - du_dot_u * u[0]) * inv_r) as f64,
                (dr_eff * u[1] + (du[1] - du_dot_u * u[1]) * inv_r) as f64,
                (dr_eff * u[2] + (du[2] - du_dot_u * u[2]) * inv_r) as f64,
            ];

            // 4. d_e = R[sender] - R[receiver] + shift => F[sender] -= dd, F[receiver] += dd
            let j_src = sender[e_idx];
            let i_dst = receiver[e_idx];
            forces_ev_a[j_src][0] -= dd[0];
            forces_ev_a[j_src][1] -= dd[1];
            forces_ev_a[j_src][2] -= dd[2];
            forces_ev_a[i_dst][0] += dd[0];
            forces_ev_a[i_dst][1] += dd[1];
            forces_ev_a[i_dst][2] += dd[2];
        }

        Ok(PotentialEval {
            energy_ev: total_energy_ev,
            forces_ev_a,
            descriptor: final_descriptor,
            desc_dim: h,
        })
}
