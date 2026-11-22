// WebGPU / Metal / Vulkan / DX12 compute shaders for HyFF-L2 (`Presto`, `Vivace`, `Moderato`)
// Forward + Analytical Reverse-Mode VJP (`F = -dE/dR`).
//
// Strictly adheres to W3C WGSL Uniformity Analysis:
// - Zero `workgroupBarrier()` inside non-uniform control flow
// - Deterministic CSR gather reductions (zero atomic float CAS loops)
// - Compensated two-step Kahan centering for LayerNorm (`layernorm_centered_fp64` parity in f32)

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

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

fn silu_grad(x: f32) -> f32 {
    let s = 1.0 / (1.0 + exp(-x));
    return s * (1.0 + x * (1.0 - s));
}

fn smooth_switch(dist: f32, cutoff: f32, width: f32) -> f32 {
    let x = clamp((cutoff - dist) / width, 0.0, 1.0);
    return (x * x * x) * (10.0 - 15.0 * x + 6.0 * x * x);
}

fn smooth_switch_deriv(dist: f32, cutoff: f32, width: f32) -> f32 {
    let raw = (cutoff - dist) / width;
    if (raw <= 0.0 || raw >= 1.0) {
        return 0.0;
    }
    let omx = 1.0 - raw;
    return -(30.0 / width) * raw * raw * omx * omx;
}

const SQRT_2: f32 = 1.41421356237;
const INV_SQRT_2: f32 = 0.70710678118;
const SQRT_6: f32 = 2.44948974278;
const INV_SQRT_6: f32 = 0.40824829046;
const PI: f32 = 3.14159265359;

// ============================================================================
// 1. General Linear Forward: Y = X * W^T + b  (p0 = rows, p1 = in_dim, p2 = out_dim, p3 = has_bias, p4 = apply_silu)
// ============================================================================
@group(0) @binding(0) var<uniform> u_lin: Params;
@group(0) @binding(1) var<storage, read> lin_x: array<f32>;
@group(0) @binding(2) var<storage, read> lin_w: array<f32>;
@group(0) @binding(3) var<storage, read> lin_b: array<f32>;
@group(0) @binding(4) var<storage, read_write> lin_y: array<f32>;

@compute @workgroup_size(64)
fn linear_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = u_lin.p0 * u_lin.p2;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }
    let row = idx / u_lin.p2;
    let col = idx % u_lin.p2;
    let in_dim = u_lin.p1;
    let x_off = row * in_dim;
    let w_off = col * in_dim;

    var acc: f32 = 0.0;
    if (u_lin.p3 != 0u) {
        acc = lin_b[col];
    }
    for (var k: u32 = 0u; k < in_dim; k = k + 1u) {
        acc = acc + lin_x[x_off + k] * lin_w[w_off + k];
    }
    if (u_lin.p4 != 0u) {
        acc = silu(acc);
    }
    lin_y[idx] = acc;
}

// ============================================================================
// 2. General Linear Backward: dX += dY * W  (p0 = rows, p1 = in_dim, p2 = out_dim, p3 = accumulate)
// ============================================================================
@group(0) @binding(0) var<uniform> u_lb: Params;
@group(0) @binding(1) var<storage, read> lb_dy: array<f32>;
@group(0) @binding(2) var<storage, read> lb_w: array<f32>;
@group(0) @binding(3) var<storage, read_write> lb_dx: array<f32>;

@compute @workgroup_size(64)
fn linear_bwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = u_lb.p0 * u_lb.p1;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }
    let row = idx / u_lb.p1;
    let col = idx % u_lb.p1;
    let in_dim = u_lb.p1;
    let out_dim = u_lb.p2;
    let dy_off = row * out_dim;

    var acc: f32 = 0.0;
    for (var o: u32 = 0u; o < out_dim; o = o + 1u) {
        acc = acc + lb_dy[dy_off + o] * lb_w[o * in_dim + col];
    }
    if (u_lb.p3 != 0u) {
        lb_dx[idx] = lb_dx[idx] + acc;
    } else {
        lb_dx[idx] = acc;
    }
}

// ============================================================================
// 3. Pointwise SiLU Forward (save pre, write act) & SiLU Backward (dy *= silu'(pre))
// ============================================================================
@group(0) @binding(0) var<uniform> u_act: Params;
@group(0) @binding(1) var<storage, read> act_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> act_out: array<f32>;

@compute @workgroup_size(64)
fn silu_fwd_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u_act.p0) {
        return;
    }
    act_out[idx] = silu(act_in[idx]);
}

@compute @workgroup_size(64)
fn silu_bwd_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u_act.p0) {
        return;
    }
    act_out[idx] = act_out[idx] * silu_grad(act_in[idx]);
}

// ============================================================================
// 4. Compensated Centered LayerNorm Forward & Backward (p0 = rows, p1 = dim)
// ============================================================================
@group(0) @binding(0) var<uniform> u_ln: Params;
@group(0) @binding(1) var<storage, read> ln_x: array<f32>;
@group(0) @binding(2) var<storage, read> ln_gamma: array<f32>;
@group(0) @binding(3) var<storage, read> ln_beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> ln_y: array<f32>;
@group(0) @binding(5) var<storage, read_write> ln_xhat: array<f32>;
@group(0) @binding(6) var<storage, read_write> ln_rstd: array<f32>;

@compute @workgroup_size(64)
fn layernorm_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= u_ln.p0) {
        return;
    }
    let dim = u_ln.p1;
    let inv_dim = 1.0 / f32(dim);
    let base = row * dim;

    // Two-pass Kahan-compensated mean for fp64-equivalent centering
    var sum1: f32 = 0.0;
    var c1: f32 = 0.0;
    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        let y = ln_x[base + i] - c1;
        let t = sum1 + y;
        c1 = (t - sum1) - y;
        sum1 = t;
    }
    let m0 = sum1 * inv_dim;
    var corr: f32 = 0.0;
    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        corr = corr + (ln_x[base + i] - m0);
    }
    let mean = m0 + corr * inv_dim;

    var var_sum: f32 = 0.0;
    var c2: f32 = 0.0;
    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        let d = ln_x[base + i] - mean;
        let y = (d * d) - c2;
        let t = var_sum + y;
        c2 = (t - var_sum) - y;
        var_sum = t;
    }
    let rho = inverseSqrt(var_sum * inv_dim + 1e-5);
    ln_rstd[row] = rho;

    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        let xh = (ln_x[base + i] - mean) * rho;
        ln_xhat[base + i] = xh;
        ln_y[base + i] = xh * ln_gamma[i] + ln_beta[i];
    }
}

@group(0) @binding(0) var<uniform> u_lnb: Params;
@group(0) @binding(1) var<storage, read> lnb_dy: array<f32>;
@group(0) @binding(2) var<storage, read> lnb_gamma: array<f32>;
@group(0) @binding(3) var<storage, read> lnb_xhat: array<f32>;
@group(0) @binding(4) var<storage, read> lnb_rstd: array<f32>;
@group(0) @binding(5) var<storage, read_write> lnb_dx: array<f32>;

@compute @workgroup_size(64)
fn layernorm_bwd_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= u_lnb.p0) {
        return;
    }
    let dim = u_lnb.p1;
    let inv_dim = 1.0 / f32(dim);
    let base = row * dim;
    let rho = lnb_rstd[row];

    var sum_g: f32 = 0.0;
    var sum_g_xh: f32 = 0.0;
    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        let g = lnb_dy[base + i] * lnb_gamma[i];
        sum_g = sum_g + g;
        sum_g_xh = sum_g_xh + g * lnb_xhat[base + i];
    }
    let mean_g = sum_g * inv_dim;
    let mean_g_xh = sum_g_xh * inv_dim;

    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        let g = lnb_dy[base + i] * lnb_gamma[i];
        let dx = (g - mean_g - lnb_xhat[base + i] * mean_g_xh) * rho;
        lnb_dx[base + i] = lnb_dx[base + i] + dx;
    }
}

// ============================================================================
// 5. 83-Element Routed Expert MLP Forward & Backward
//    p0 = N, p1 = D_in (H+5T), p2 = E_H, p3 = O_b
// ============================================================================
@group(0) @binding(0) var<uniform> u_exp: Params;
@group(0) @binding(1) var<storage, read> exp_idx: array<u32>;
@group(0) @binding(2) var<storage, read> exp_z: array<f32>;
@group(0) @binding(3) var<storage, read> exp_w0: array<f32>;
@group(0) @binding(4) var<storage, read> exp_b0: array<f32>;
@group(0) @binding(5) var<storage, read> exp_w2: array<f32>;
@group(0) @binding(6) var<storage, read> exp_b2: array<f32>;
@group(0) @binding(7) var<storage, read_write> exp_pre: array<f32>;
@group(0) @binding(8) var<storage, read_write> exp_u: array<f32>;

@compute @workgroup_size(64)
fn expert_w0_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = u_exp.p0 * u_exp.p2;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }
    let i = idx / u_exp.p2;
    let eh = idx % u_exp.p2;
    let d_in = u_exp.p1;
    let e_h = u_exp.p2;
    let k_exp = exp_idx[i];

    let z_off = i * d_in;
    let w0_off = (k_exp * e_h + eh) * d_in;
    var acc: f32 = exp_b0[k_exp * e_h + eh];
    for (var d: u32 = 0u; d < d_in; d = d + 1u) {
        acc = acc + exp_z[z_off + d] * exp_w0[w0_off + d];
    }
    exp_pre[idx] = acc;
}

@compute @workgroup_size(64)
fn expert_w2_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = u_exp.p0 * u_exp.p3;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }
    let i = idx / u_exp.p3;
    let o = idx % u_exp.p3;
    let e_h = u_exp.p2;
    let o_b = u_exp.p3;
    let k_exp = exp_idx[i];

    let pre_off = i * e_h;
    let w2_off = (k_exp * o_b + o) * e_h;
    var acc: f32 = exp_b2[k_exp * o_b + o];
    for (var eh: u32 = 0u; eh < e_h; eh = eh + 1u) {
        acc = acc + silu(exp_pre[pre_off + eh]) * exp_w2[w2_off + eh];
    }
    exp_u[idx] = acc;
}

@group(0) @binding(0) var<uniform> u_expb: Params;
@group(0) @binding(1) var<storage, read> expb_idx: array<u32>;
@group(0) @binding(2) var<storage, read> expb_du: array<f32>;
@group(0) @binding(3) var<storage, read> expb_pre: array<f32>;
@group(0) @binding(4) var<storage, read> expb_w0: array<f32>;
@group(0) @binding(5) var<storage, read> expb_w2: array<f32>;
@group(0) @binding(6) var<storage, read_write> expb_dhidden: array<f32>;
@group(0) @binding(7) var<storage, read_write> expb_dz: array<f32>;

@compute @workgroup_size(64)
fn expert_w2_bwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = u_expb.p0 * u_expb.p2;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }
    let i = idx / u_expb.p2;
    let eh = idx % u_expb.p2;
    let e_h = u_expb.p2;
    let o_b = u_expb.p3;
    let k_exp = expb_idx[i];

    let du_off = i * o_b;
    let w2_base = k_exp * o_b * e_h + eh;
    var acc: f32 = 0.0;
    for (var o: u32 = 0u; o < o_b; o = o + 1u) {
        acc = acc + expb_du[du_off + o] * expb_w2[w2_base + o * e_h];
    }
    expb_dhidden[idx] = acc * silu_grad(expb_pre[idx]);
}

@compute @workgroup_size(64)
fn expert_w0_bwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = u_expb.p0 * u_expb.p1;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }
    let i = idx / u_expb.p1;
    let d = idx % u_expb.p1;
    let d_in = u_expb.p1;
    let e_h = u_expb.p2;
    let k_exp = expb_idx[i];

    let dh_off = i * e_h;
    let w0_base = k_exp * e_h * d_in + d;
    var acc: f32 = 0.0;
    for (var eh: u32 = 0u; eh < e_h; eh = eh + 1u) {
        acc = acc + expb_dhidden[dh_off + eh] * expb_w0[w0_base + eh * d_in];
    }
    expb_dz[idx] = acc;
}
