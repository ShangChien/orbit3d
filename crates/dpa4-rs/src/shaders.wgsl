// WebGPU / Metal / Vulkan / DX12 compute shaders for DeePMD DPA-4 (`SeZM`: `Nano`, `Mini`, `Neo`, `Air`, `Plus`)
// Forward + Analytical Reverse-Mode VJP (`F = -dE/dR`).
//
// Strictly adheres to W3C WGSL Uniformity Analysis:
// - Zero `workgroupBarrier()` inside non-uniform control flow
// - Batched focus-major GEMM for `SO2Linear` (`m=0` and `|m|=1` edge blocks, forward & transpose VJP)
// - Batched `SO3GridProjector` (`to_grid_mat` `(G, 3*D)` & `from_grid_mat` `(3*D, G)`)

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
// 1. Batched Matrix Multiply: C[b, r, c] = (+=) sum_k A[b, r, k] * B[b, k, c]
//    (or B^T when p4 != 0: B[b, c, k])
//    p0 = batch (F), p1 = rows (E), p2 = k_dim, p3 = cols, p4 = trans_b, p5 = accumulate
// ============================================================================
@group(0) @binding(0) var<uniform> u_mm: Params;
@group(0) @binding(1) var<storage, read> mm_a: array<f32>;
@group(0) @binding(2) var<storage, read> mm_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> mm_c: array<f32>;

@compute @workgroup_size(64)
fn batched_matmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b_cnt = u_mm.p0;
    let rows = u_mm.p1;
    let k_dim = u_mm.p2;
    let cols = u_mm.p3;
    let total = b_cnt * rows * cols;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let rc_stride = rows * cols;
    let b = idx / rc_stride;
    let rem = idx % rc_stride;
    let r = rem / cols;
    let c = rem % cols;

    let a_off = (b * rows + r) * k_dim;
    var acc: f32 = 0.0;
    if (u_mm.p4 == 0u) {
        let b_off = b * k_dim * cols + c;
        for (var k: u32 = 0u; k < k_dim; k = k + 1u) {
            acc = acc + mm_a[a_off + k] * mm_b[b_off + k * cols];
        }
    } else {
        let b_off = (b * cols + c) * k_dim;
        for (var k: u32 = 0u; k < k_dim; k = k + 1u) {
            acc = acc + mm_a[a_off + k] * mm_b[b_off + k];
        }
    }

    if (u_mm.p5 != 0u) {
        mm_c[idx] = mm_c[idx] + acc;
    } else {
        mm_c[idx] = acc;
    }
}

// ============================================================================
// 2. SO(3) Grid Quadratic Product (`(T_to * L) * (T_to * R)` -> `T_from * prod`)
//    for F=1 tensors shaped (N, 3*D, C):
//    Step A (`to_grid_prod`): L, R: (N, 3*D, C), T_to: (G, 3*D) -> V_g: (N, G, C), L_g, R_g
//    p0 = N, p1 = 3*D, p2 = G, p3 = C
// ============================================================================
@group(0) @binding(0) var<uniform> u_gp: Params;
@group(0) @binding(1) var<storage, read> gp_l: array<f32>;
@group(0) @binding(2) var<storage, read> gp_r: array<f32>;
@group(0) @binding(3) var<storage, read> gp_to: array<f32>;
@group(0) @binding(4) var<storage, read_write> gp_lg: array<f32>;
@group(0) @binding(5) var<storage, read_write> gp_rg: array<f32>;
@group(0) @binding(6) var<storage, read_write> gp_vg: array<f32>;

@compute @workgroup_size(64)
fn so3_to_grid_prod(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_atoms = u_gp.p0;
    let kd = u_gp.p1;
    let g_pts = u_gp.p2;
    let ch = u_gp.p3;
    let total = n_atoms * g_pts * ch;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let gc = g_pts * ch;
    let n = idx / gc;
    let rem = idx % gc;
    let g = rem / ch;
    let c = rem % ch;

    let to_off = g * kd;
    let lr_base = n * kd * ch + c;
    var acc_l: f32 = 0.0;
    var acc_r: f32 = 0.0;
    for (var k: u32 = 0u; k < kd; k = k + 1u) {
        let t_val = gp_to[to_off + k];
        let off = lr_base + k * ch;
        acc_l = acc_l + t_val * gp_l[off];
        acc_r = acc_r + t_val * gp_r[off];
    }
    gp_lg[idx] = acc_l;
    gp_rg[idx] = acc_r;
    gp_vg[idx] = acc_l * acc_r;
}

@group(0) @binding(0) var<uniform> u_fg: Params;
@group(0) @binding(1) var<storage, read> fg_vg: array<f32>;
@group(0) @binding(2) var<storage, read> fg_from: array<f32>;
@group(0) @binding(3) var<storage, read_write> fg_out: array<f32>;

@compute @workgroup_size(64)
fn so3_from_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_atoms = u_fg.p0;
    let kd = u_fg.p1;
    let g_pts = u_fg.p2;
    let ch = u_fg.p3;
    let total = n_atoms * kd * ch;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let kdc = kd * ch;
    let n = idx / kdc;
    let rem = idx % kdc;
    let k = rem / ch;
    let c = rem % ch;

    let from_off = k * g_pts;
    let vg_base = n * g_pts * ch + c;
    var acc: f32 = 0.0;
    for (var g: u32 = 0u; g < g_pts; g = g + 1u) {
        acc = acc + fg_from[from_off + g] * fg_vg[vg_base + g * ch];
    }
    fg_out[idx] = acc;
}
