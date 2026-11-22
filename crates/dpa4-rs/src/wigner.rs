//! Smooth dual-chart quaternion frame construction (`build_edge_quaternion`) and
//! packed real spherical Wigner-D rotation matrices (`l = 0..4`) + exact analytical VJP.

use crate::model::Dpa4Weights;

#[derive(Clone, Copy, Default)]
pub struct SavedEdgeQuat {
    pub v: [f32; 3],
    pub rq: f32,
    pub u: [f32; 3],
    pub q_pos_raw_r: f32,
    pub q_pos: [f32; 4],
    pub q_neg_raw_r: f32,
    pub q_neg: [f32; 4],
    pub flip_sign: f32,
    pub t: f32,
    pub beta: f32,
    pub dbeta_dt: f32,
    pub q_blend_r: f32,
    pub q_edge: [f32; 4],
    pub q_wigner_r: f32,
    pub q_wigner: [f32; 4],
}

#[inline]
fn safe_norm4_r(q: [f32; 4], eps: f32) -> f32 {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] + eps * eps;
    1.0 / n2.sqrt()
}

#[inline]
fn norm4_bwd(dq_out: [f32; 4], q_out: [f32; 4], inv_norm: f32) -> [f32; 4] {
    // q_out = q_in * inv_norm, where inv_norm = (||q_in||^2 + eps^2)^(-1/2)
    // d(q_out)/d(q_in) = inv_norm * I - inv_norm^3 * q_in * q_in^T
    // Notice q_in * inv_norm == q_out, EXCEPT eps^2 means ||q_out||^2 = ||q_in||^2 / (||q_in||^2 + eps^2).
    // Exact Jacobian: dq_in = inv_norm * (dq_out - (dq_out . q_out) * q_out ... wait:
    // d(inv_norm)/dq_in = -inv_norm^3 * q_in = -inv_norm^2 * q_out!
    // So <dq_out, d(q_in * inv_norm)> = inv_norm * dq_out + <dq_out, q_in> * (-inv_norm^3 * q_in)
    // Since q_in = q_out / inv_norm, <dq_out, q_in> * inv_norm^3 * q_in = <dq_out, q_out> * inv_norm * q_out!
    // Therefore `inv_norm * (dq_out - (dq_out . q_out) * q_out)` IS 100% EXACT even for finite eps > 0!!
    let dot = dq_out[0] * q_out[0]
        + dq_out[1] * q_out[1]
        + dq_out[2] * q_out[2]
        + dq_out[3] * q_out[3];
    [
        inv_norm * (dq_out[0] - dot * q_out[0]),
        inv_norm * (dq_out[1] - dot * q_out[1]),
        inv_norm * (dq_out[2] - dot * q_out[2]),
        inv_norm * (dq_out[3] - dot * q_out[3]),
    ]
}

#[inline]
fn ipow(base: f32, exp: u8) -> f32 {
    match exp {
        0 => 1.0,
        1 => base,
        2 => base * base,
        3 => base * base * base,
        4 => {
            let b2 = base * base;
            b2 * b2
        }
        5 => {
            let b2 = base * base;
            b2 * b2 * base
        }
        6 => {
            let b3 = base * base * base;
            b3 * b3
        }
        7 => {
            let b3 = base * base * base;
            b3 * b3 * base
        }
        8 => {
            let b2 = base * base;
            let b4 = b2 * b2;
            b4 * b4
        }
        _ => base.powi(exp as i32),
    }
}

/// Forward evaluation of `build_edge_quaternion` + `WignerDCalculator` (`D_full` `(D, D)` row-major).
pub fn build_edge_wigner_fwd(
    v: [f32; 3],
    r_safe: f32,
    eps: f32,
    w: &Dpa4Weights,
    d_full_out: &mut [f32],
) -> SavedEdgeQuat {
    // 1. r_q = sqrt(r_safe^2 + eps^2)
    let rq = (r_safe * r_safe + eps * eps).sqrt();
    let inv_rq = 1.0 / rq;
    let u = [v[0] * inv_rq, v[1] * inv_rq, v[2] * inv_rq];

    // 2. +Z and -Z quaternion charts
    let qp_raw = [1.0 + u[2], u[1], -u[0], 0.0];
    let qp_r = safe_norm4_r(qp_raw, eps);
    let q_pos = [
        qp_raw[0] * qp_r,
        qp_raw[1] * qp_r,
        qp_raw[2] * qp_r,
        0.0,
    ];

    let qn_raw = [-u[0], 0.0, 1.0 - u[2], u[1]];
    let qn_r = safe_norm4_r(qn_raw, eps);
    let q_neg = [
        qn_raw[0] * qn_r,
        0.0,
        qn_raw[2] * qn_r,
        qn_raw[3] * qn_r,
    ];

    // 3. Smooth C^inf step on t = clamp(0.5 * (u_z + 1), 0, 1)
    let t = (0.5 * (u[2] + 1.0)).clamp(0.0, 1.0);
    let f32_eps = f32::EPSILON;
    let (beta, dbeta_dt) = if t <= 0.0 {
        (0.0, 0.0)
    } else if t >= 1.0 {
        (1.0, 0.0)
    } else {
        let tc = t.max(f32_eps);
        let omc = (1.0 - t).max(f32_eps);
        let left = (-1.0 / tc).exp();
        let right = (-1.0 / omc).exp();
        let denom = left + right;
        if denom == 0.0 {
            (if t < 0.5 { 0.0 } else { 1.0 }, 0.0)
        } else {
            let b = left / denom;
            let prod = b * (1.0 - b);
            let db = if prod > 0.0 && left > 0.0 && right > 0.0 {
                prod * (1.0 / (tc * tc) + 1.0 / (omc * omc))
            } else {
                0.0
            };
            (b, db)
        }
    };

    // 4. Shortest-arc NLERP: dot = q_neg . q_pos
    let dot = q_neg[0] * q_pos[0] + q_neg[1] * q_pos[1] + q_neg[2] * q_pos[2] + q_neg[3] * q_pos[3];
    let flip_sign = if dot < 0.0 { -1.0f32 } else { 1.0f32 };
    let omb = 1.0 - beta;
    let bf = beta * flip_sign;
    let q_blend = [
        omb * q_neg[0] + bf * q_pos[0],
        omb * q_neg[1] + bf * q_pos[1],
        omb * q_neg[2] + bf * q_pos[2],
        omb * q_neg[3] + bf * q_pos[3],
    ];
    let q_blend_r = safe_norm4_r(q_blend, eps);
    let q_edge = [
        q_blend[0] * q_blend_r,
        q_blend[1] * q_blend_r,
        q_blend[2] * q_blend_r,
        q_blend[3] * q_blend_r,
    ];

    // 5. WignerDCalculator re-normalizes q_edge
    let q_wigner_r = safe_norm4_r(q_edge, eps);
    let q = [
        q_edge[0] * q_wigner_r,
        q_edge[1] * q_wigner_r,
        q_edge[2] * q_wigner_r,
        q_edge[3] * q_wigner_r,
    ];

    let dim = w.dim_d;
    d_full_out.fill(0.0);
    d_full_out[0] = 1.0;

    let lmax = w.cfg.lmax;
    if lmax >= 1 {
        let (qw, qx, qy, qz) = (q[0], q[1], q[2], q[3]);
        let x2 = qx * qx;
        let y2 = qy * qy;
        let z2 = qz * qz;
        let xy = qx * qy;
        let xz = qx * qz;
        let yz = qy * qz;
        let wx = qw * qx;
        let wy = qw * qy;
        let wz = qw * qz;

        let rot = [
            [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)],
        ];
        let perm = [1usize, 2, 0];
        let sign = [-1.0f32, -1.0, 1.0];
        for a in 0..3 {
            for b in 0..3 {
                d_full_out[(1 + a) * dim + (1 + b)] = sign[a] * sign[b] * rot[perm[a]][perm[b]];
            }
        }
    }

    if lmax >= 2 {
        // _l2_monomial_coeff: (35, 25), exp_l2: (35, 4)
        let exps = w.wigner_exp_l2.as_ref().unwrap();
        let coeff = w.wigner_l2_coeff.as_ref().unwrap();
        let mut blk2 = [0.0f32; 25];
        for (m, &e) in exps.iter().enumerate() {
            let mono = ipow(q[0], e[0]) * ipow(q[1], e[1]) * ipow(q[2], e[2]) * ipow(q[3], e[3]);
            let crow = &coeff[m * 25..(m + 1) * 25];
            for k in 0..25 {
                blk2[k] += mono * crow[k];
            }
        }
        for r in 0..5 {
            for c in 0..5 {
                d_full_out[(4 + r) * dim + (4 + c)] = blk2[r * 5 + c];
            }
        }
    }

    if lmax == 3 {
        // C_l3: (49, 84), exp_l3: (84, 4)
        let exps = w.wigner_exp_l3.as_ref().unwrap();
        let c_l3 = w.wigner_c_l3.as_ref().unwrap();
        let mut monos = [0.0f32; 84];
        for (m, &e) in exps.iter().enumerate() {
            monos[m] = ipow(q[0], e[0]) * ipow(q[1], e[1]) * ipow(q[2], e[2]) * ipow(q[3], e[3]);
        }
        for k in 0..49 {
            let crow = &c_l3[k * 84..(k + 1) * 84];
            let mut sum = 0.0f32;
            for m in 0..84 {
                sum += monos[m] * crow[m];
            }
            d_full_out[(9 + k / 7) * dim + (9 + k % 7)] = sum;
        }
    } else if lmax >= 4 {
        // C_combined_l3l4: (130, 165), exp_l4: (165, 4)
        let exps = w.wigner_exp_l4.as_ref().unwrap();
        let c_34 = w.wigner_c_l3l4.as_ref().unwrap();
        let mut monos = [0.0f32; 165];
        for (m, &e) in exps.iter().enumerate() {
            monos[m] = ipow(q[0], e[0]) * ipow(q[1], e[1]) * ipow(q[2], e[2]) * ipow(q[3], e[3]);
        }
        for k in 0..49 {
            let crow = &c_34[k * 165..(k + 1) * 165];
            let mut sum = 0.0f32;
            for m in 0..165 {
                sum += monos[m] * crow[m];
            }
            d_full_out[(9 + k / 7) * dim + (9 + k % 7)] = sum;
        }
        for k in 0..81 {
            let crow = &c_34[(49 + k) * 165..(49 + k + 1) * 165];
            let mut sum = 0.0f32;
            for m in 0..165 {
                sum += monos[m] * crow[m];
            }
            d_full_out[(16 + k / 9) * dim + (16 + k % 9)] = sum;
        }
    }

    SavedEdgeQuat {
        v,
        rq,
        u,
        q_pos_raw_r: qp_r,
        q_pos,
        q_neg_raw_r: qn_r,
        q_neg,
        flip_sign,
        t,
        beta,
        dbeta_dt,
        q_blend_r,
        q_edge,
        q_wigner_r,
        q_wigner: q,
    }
}

#[inline]
fn accum_monomial_grad(q: [f32; 4], e: [u8; 4], d_mono: f32, dq: &mut [f32; 4]) {
    if d_mono == 0.0 {
        return;
    }
    let p0 = ipow(q[0], e[0]);
    let p1 = ipow(q[1], e[1]);
    let p2 = ipow(q[2], e[2]);
    let p3 = ipow(q[3], e[3]);
    if e[0] > 0 {
        dq[0] += d_mono * (e[0] as f32) * ipow(q[0], e[0] - 1) * p1 * p2 * p3;
    }
    if e[1] > 0 {
        dq[1] += d_mono * (e[1] as f32) * p0 * ipow(q[1], e[1] - 1) * p2 * p3;
    }
    if e[2] > 0 {
        dq[2] += d_mono * (e[2] as f32) * p0 * p1 * ipow(q[2], e[2] - 1) * p3;
    }
    if e[3] > 0 {
        dq[3] += d_mono * (e[3] as f32) * p0 * p1 * p2 * ipow(q[3], e[3] - 1);
    }
}

/// Analytical VJP from `d_D_full` `(D, D)` (already including `d_Dt_full^T`) back to `d_edge_vec` `[f32; 3]`.
pub fn build_edge_wigner_bwd(
    d_dfull: &[f32],
    saved: &SavedEdgeQuat,
    w: &Dpa4Weights,
) -> [f32; 3] {
    let dim = w.dim_d;
    let lmax = w.cfg.lmax;
    let q = saved.q_wigner;
    let mut dq = [0.0f32; 4];

    if lmax >= 1 {
        let perm = [1usize, 2, 0];
        let sign = [-1.0f32, -1.0, 1.0];
        let mut drot = [[0.0f32; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                drot[perm[a]][perm[b]] += sign[a] * sign[b] * d_dfull[(1 + a) * dim + (1 + b)];
            }
        }
        let (qw, qx, qy, qz) = (q[0], q[1], q[2], q[3]);
        // rot[0] = [1 - 2(y2+z2), 2(xy - wz), 2(xz + wy)]
        // rot[1] = [2(xy + wz), 1 - 2(x2+z2), 2(yz - wx)]
        // rot[2] = [2(xz - wy), 2(yz + wx), 1 - 2(x2+y2)]
        dq[0] += 2.0 * (-qz * drot[0][1] + qy * drot[0][2] + qz * drot[1][0] - qx * drot[1][2] - qy * drot[2][0] + qx * drot[2][1]);
        dq[1] += 2.0 * (qy * (drot[0][1] + drot[1][0]) + qz * (drot[0][2] + drot[2][0]) - qw * drot[1][2] + qw * drot[2][1] - 2.0 * qx * (drot[1][1] + drot[2][2]));
        dq[2] += 2.0 * (-2.0 * qy * (drot[0][0] + drot[2][2]) + qx * (drot[0][1] + drot[1][0]) + qw * drot[0][2] + qz * (drot[1][2] + drot[2][1]) - qw * drot[2][0]);
        dq[3] += 2.0 * (-2.0 * qz * (drot[0][0] + drot[1][1]) - qw * drot[0][1] + qx * (drot[0][2] + drot[2][0]) + qw * drot[1][0] + qy * (drot[1][2] + drot[2][1]));
    }

    if lmax >= 2 {
        let exps = w.wigner_exp_l2.as_ref().unwrap();
        let coeff = w.wigner_l2_coeff.as_ref().unwrap();
        let mut g2 = [0.0f32; 25];
        for r in 0..5 {
            for c in 0..5 {
                g2[r * 5 + c] = d_dfull[(4 + r) * dim + (4 + c)];
            }
        }
        for (m, &e) in exps.iter().enumerate() {
            let crow = &coeff[m * 25..(m + 1) * 25];
            let mut dm = 0.0f32;
            for k in 0..25 {
                dm += g2[k] * crow[k];
            }
            accum_monomial_grad(q, e, dm, &mut dq);
        }
    }

    if lmax == 3 {
        let exps = w.wigner_exp_l3.as_ref().unwrap();
        let c_l3 = w.wigner_c_l3.as_ref().unwrap();
        let mut d_monos = [0.0f32; 84];
        for k in 0..49 {
            let g = d_dfull[(9 + k / 7) * dim + (9 + k % 7)];
            if g == 0.0 {
                continue;
            }
            let crow = &c_l3[k * 84..(k + 1) * 84];
            for m in 0..84 {
                d_monos[m] += g * crow[m];
            }
        }
        for (m, &e) in exps.iter().enumerate() {
            accum_monomial_grad(q, e, d_monos[m], &mut dq);
        }
    } else if lmax >= 4 {
        let exps = w.wigner_exp_l4.as_ref().unwrap();
        let c_34 = w.wigner_c_l3l4.as_ref().unwrap();
        let mut d_monos = [0.0f32; 165];
        for k in 0..49 {
            let g = d_dfull[(9 + k / 7) * dim + (9 + k % 7)];
            if g != 0.0 {
                let crow = &c_34[k * 165..(k + 1) * 165];
                for m in 0..165 {
                    d_monos[m] += g * crow[m];
                }
            }
        }
        for k in 0..81 {
            let g = d_dfull[(16 + k / 9) * dim + (16 + k % 9)];
            if g != 0.0 {
                let crow = &c_34[(49 + k) * 165..(49 + k + 1) * 165];
                for m in 0..165 {
                    d_monos[m] += g * crow[m];
                }
            }
        }
        for (m, &e) in exps.iter().enumerate() {
            accum_monomial_grad(q, e, d_monos[m], &mut dq);
        }
    }

    // Backprop through WignerDCalculator input norm -> q_edge
    let dq_edge = norm4_bwd(dq, saved.q_wigner, saved.q_wigner_r);
    // Backprop through NLERP norm -> q_blend
    let dq_blend = norm4_bwd(dq_edge, saved.q_edge, saved.q_blend_r);

    // q_blend = (1 - beta) * q_neg + beta * flip_sign * q_pos
    let omb = 1.0 - saved.beta;
    let bf = saved.beta * saved.flip_sign;
    let dq_neg = [
        omb * dq_blend[0],
        omb * dq_blend[1],
        omb * dq_blend[2],
        omb * dq_blend[3],
    ];
    let dq_pos = [
        bf * dq_blend[0],
        bf * dq_blend[1],
        bf * dq_blend[2],
        bf * dq_blend[3],
    ];
    let mut d_beta = 0.0f32;
    for k in 0..4 {
        d_beta += dq_blend[k] * (saved.flip_sign * saved.q_pos[k] - saved.q_neg[k]);
    }

    let dq_pos_raw = norm4_bwd(dq_pos, saved.q_pos, saved.q_pos_raw_r);
    let dq_neg_raw = norm4_bwd(dq_neg, saved.q_neg, saved.q_neg_raw_r);

    // qp_raw = [1 + uz, uy, -ux, 0], qn_raw = [-ux, 0, 1 - uz, uy]
    // t = clamp(0.5 * (uz + 1), 0, 1) => dt/duz = 0.5 inside (0, 1)
    let dt_duz = if saved.t > 0.0 && saved.t < 1.0 { 0.5 } else { 0.0 };
    let du = [
        -dq_pos_raw[2] - dq_neg_raw[0],
        dq_pos_raw[1] + dq_neg_raw[3],
        dq_pos_raw[0] - dq_neg_raw[2] + d_beta * saved.dbeta_dt * dt_duz,
    ];

    // u = v / rq, where rq = sqrt(||v||^2 + 2 * eps^2)
    let inv_rq = 1.0 / saved.rq;
    let du_dot_u = du[0] * saved.u[0] + du[1] * saved.u[1] + du[2] * saved.u[2];
    [
        inv_rq * (du[0] - du_dot_u * saved.u[0]),
        inv_rq * (du[1] - du_dot_u * saved.u[1]),
        inv_rq * (du[2] - du_dot_u * saved.u[2]),
    ]
}
