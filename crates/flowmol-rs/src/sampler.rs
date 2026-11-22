use crate::chem::{inspect_sampled_molecule, SampledMolecule};
use crate::config::{FlowMolConfig, ForwardOutput, PrecomputedModulations, SelfCondState};
use crate::cpu_backend::{forward_cpu, precompute_condition_modulations, remove_com_in_place, GraphTopology};
use crate::format::ModelBundle;
use crate::gpu_backend::WgpuBackend;
use serde::{Deserialize, Serialize};

/// Deterministic, cross-platform (WASM == Native) SplitMix64 + Box-Muller PRNG.
#[derive(Clone, Debug)]
pub struct DetRng {
    state: u64,
    spare_normal: Option<f32>,
}

impl DetRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
            spare_normal: None,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Uniform in [0, 1) with 24-bit mantissa precision
        ((self.next_u64() >> 40) as f32) * (1.0 / ((1u32 << 24) as f32))
    }

    pub fn next_standard_normal(&mut self) -> f32 {
        if let Some(sp) = self.spare_normal.take() {
            return sp;
        }
        let u1 = self.next_f32().clamp(1e-7, 1.0 - 1e-7);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        self.spare_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    pub fn sample_categorical(&mut self, probs: &[f32]) -> usize {
        let u = self.next_f32();
        let mut cdf = 0.0f32;
        let mut last_pos = probs.len().saturating_sub(1);
        for (i, &p) in probs.iter().enumerate() {
            if p > 0.0 {
                last_pos = i;
                cdf += p;
                if u <= cdf {
                    return i;
                }
            }
        }
        last_pos
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleOptions {
    #[serde(default)]
    pub n_atoms: Option<usize>,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default = "default_steps")]
    pub n_steps: usize,
    #[serde(default = "default_solver")]
    pub solver: String,
    #[serde(default = "default_grid")]
    pub grid_type: String,
    #[serde(default = "default_stoch")]
    pub stochasticity: f32,
    #[serde(default = "default_temp")]
    pub cat_temperature: f32,
    #[serde(default)]
    pub allowed_elements: Option<Vec<String>>,
    #[serde(default)]
    pub allowed_charges: Option<Vec<i32>>,
    /// Zero-retrain element pinning: forces specific elements (e.g. `["Rh", "P", "P"]`) onto initial nodes!
    #[serde(default)]
    pub required_elements: Vec<String>,
    #[serde(default)]
    pub cond_values: Option<Vec<f32>>,
    #[serde(default)]
    pub cond_active: Option<Vec<bool>>,
    #[serde(default = "default_one")]
    pub cfg_scale: f32,
    #[serde(default = "default_one")]
    pub xyz_cfg_scale: f32,
    /// Initial Gaussian coordinate prior radius multiplier (`< 1.0`, e.g. `0.85`, keeps initial cloud compact).
    #[serde(default = "default_one")]
    pub prior_scale: f32,
    /// SE(3)-equivariant MST rigid subcluster cohesion pull (`0.0` = off, `0.35` = recommended single-molecule pull).
    #[serde(default)]
    pub cohesion_strength: f32,
    /// Single-molecule connectivity guarantee: `"none"` | `"cohesion"` | `"largest_fragment"` | `"strict_retry"`.
    #[serde(default = "default_single_mode")]
    pub single_molecule_mode: String,
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub force_cpu: bool,
}

fn default_steps() -> usize {
    25
}
fn default_solver() -> String {
    "heun".to_string()
}
fn default_grid() -> String {
    "quadratic_end".to_string()
}
fn default_stoch() -> f32 {
    10.0
}
fn default_temp() -> f32 {
    0.05
}
fn default_one() -> f32 {
    1.0
}
fn default_single_mode() -> String {
    "none".to_string()
}
fn default_max_retries() -> usize {
    4
}
fn default_seed() -> u64 {
    42
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            n_atoms: None,
            domain: None,
            n_steps: 25,
            solver: "heun".to_string(),
            grid_type: "quadratic_end".to_string(),
            stochasticity: 10.0,
            cat_temperature: 0.05,
            allowed_elements: None,
            allowed_charges: None,
            required_elements: Vec::new(),
            cond_values: None,
            cond_active: None,
            cfg_scale: 1.0,
            xyz_cfg_scale: 1.0,
            prior_scale: 1.0,
            cohesion_strength: 0.0,
            single_molecule_mode: "none".to_string(),
            max_retries: 4,
            seed: 42,
            force_cpu: false,
        }
    }
}

pub fn build_inference_tspan(n_steps: usize, grid_type: &str) -> Vec<f32> {
    let ns = n_steps.max(1);
    let mut grid = Vec::with_capacity(ns + 1);
    for i in 0..=ns {
        let s = (i as f32) / (ns as f32);
        let g = match grid_type {
            "uniform" => s,
            "log_end" => 1.0 - (1.0 - s).powi(2),
            _ => 0.35 * s + 0.65 * (1.0 - (1.0 - s).powf(1.5)), // quadratic_end
        };
        grid.push(g);
    }
    grid[0] = 0.0;
    grid[ns] = 1.0;
    grid
}

pub fn eval_alpha_t(cfg: &FlowMolConfig, t: f32) -> [f32; 4] {
    let tf = t.clamp(1e-6, 1.0 - 1e-6);
    let nus = [cfg.nu_x, cfg.nu_a, cfg.nu_c, cfg.nu_e];
    let mut out = [0.0f32; 4];
    for i in 0..4 {
        out[i] = match cfg.schedule_type.as_str() {
            "linear" => tf,
            "power" => tf.powf(nus[i]),
            _ => (0.5 * std::f32::consts::PI * tf.powf(nus[i])).sin().powi(2),
        };
    }
    out
}

fn ctmc_transition_rates(alpha_t: f32, alpha_s: f32, dt: f32, stochasticity: f32, last_step: bool) -> (f32, f32) {
    if last_step {
        return (1.0, 0.0);
    }
    let d_alpha = (alpha_s - alpha_t).max(0.0);
    let rem_s = (1.0 - alpha_s).clamp(0.0, 1.0);
    let denom = (1.0 - alpha_t).max(1e-5);
    let churn = dt * stochasticity * rem_s;
    let unmask = ((d_alpha + churn * alpha_t) / denom).clamp(0.0, 1.0);
    let remask = churn.clamp(0.0, 1.0);
    (unmask, remask)
}

fn softmax_rows(logits: &[f32], rows: usize, cols: usize, mask: Option<&[bool]>) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let lr = &logits[r * cols..(r + 1) * cols];
        let mut max_v = f32::NEG_INFINITY;
        for c in 0..cols {
            if mask.map_or(true, |m| m[c]) && lr[c] > max_v {
                max_v = lr[c];
            }
        }
        let mut sum = 0.0f32;
        let orow = &mut out[r * cols..(r + 1) * cols];
        for c in 0..cols {
            if mask.map_or(true, |m| m[c]) {
                let e = (lr[c] - max_v).exp();
                orow[c] = e;
                sum += e;
            } else {
                orow[c] = 0.0;
            }
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for v in orow.iter_mut() {
            *v *= inv;
        }
    }
    out
}

fn campbell_ctmc_step(
    probs: &[f32],
    xt: &[usize],
    cols: usize,
    alpha_t: f32,
    alpha_s: f32,
    dt: f32,
    mask_idx: usize,
    stochasticity: f32,
    temperature: f32,
    last_step: bool,
    pinned: Option<&[Option<usize>]>,
    rng: &mut DetRng,
) -> Vec<usize> {
    let rows = xt.len();
    let (unmask_p, remask_p) = ctmc_transition_rates(alpha_t, alpha_s, dt, stochasticity, last_step);
    let mut out = xt.to_vec();
    let mut sharp = vec![0.0f32; cols];

    for r in 0..rows {
        if let Some(pin_arr) = pinned {
            if let Some(forced_idx) = pin_arr[r] {
                out[r] = forced_idx;
                continue;
            }
        }
        let pr = &probs[r * cols..(r + 1) * cols];
        if temperature > 0.0 && (temperature - 1.0).abs() > 1e-6 {
            let mut max_log = f32::NEG_INFINITY;
            for c in 0..cols {
                let lv = pr[c].max(1e-9).ln() / temperature;
                sharp[c] = lv;
                if lv > max_log {
                    max_log = lv;
                }
            }
            let mut s = 0.0f32;
            for c in 0..cols {
                sharp[c] = (sharp[c] - max_log).exp();
                s += sharp[c];
            }
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            for c in 0..cols {
                sharp[c] *= inv;
            }
        } else {
            let s: f32 = pr.iter().map(|&v| v.max(1e-9)).sum();
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            for c in 0..cols {
                sharp[c] = pr[c].max(1e-9) * inv;
            }
        }

        if last_step {
            let draw = if temperature <= 0.1 {
                sharp
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            } else {
                rng.sample_categorical(&sharp)
            };
            if xt[r] == mask_idx {
                out[r] = draw;
            }
        } else {
            let draw = rng.sample_categorical(&sharp);
            let u1 = rng.next_f32();
            let u2 = rng.next_f32();
            if xt[r] == mask_idx && u1 < unmask_p {
                out[r] = draw;
            } else if xt[r] != mask_idx && u2 < remask_p {
                out[r] = mask_idx;
            }
        }
    }
    out
}

pub fn sample_empirical_size(bundle: &ModelBundle, domain: Option<&str>, rng: &mut DetRng) -> usize {
    let hist = domain
        .and_then(|d| if d == "all" { None } else { bundle.header.per_source_n_atoms_hist.get(d) })
        .filter(|h| !h.is_empty())
        .unwrap_or(&bundle.header.n_atoms_hist);
    if hist.is_empty() {
        return 32;
    }
    let mut items: Vec<(usize, f64)> = hist
        .iter()
        .filter_map(|(k, &v)| k.parse::<usize>().ok().map(|n| (n, v as f64)))
        .collect();
    items.sort_by_key(|&(n, _)| n);
    let total: f64 = items.iter().map(|&(_, w)| w).sum();
    if total <= 0.0 {
        return 32;
    }
    let target = (rng.next_f32() as f64) * total;
    let mut acc = 0.0;
    for &(n, w) in &items {
        acc += w;
        if target <= acc {
            return n.max(2);
        }
    }
    items.last().map(|&(n, _)| n.max(2)).unwrap_or(32)
}

async fn run_single_predict(
    bundle: &ModelBundle,
    mut gpu: Option<&mut WgpuBackend>,
    topo: &GraphTopology,
    x_in: &[[f32; 3]],
    a_in: &[usize],
    c_in: &[usize],
    e_in: &[usize],
    t_val: f32,
    sc_in: Option<&SelfCondState>,
    cond_mods: Option<&PrecomputedModulations>,
    null_mods: Option<&PrecomputedModulations>,
    use_cfg: bool,
    cfg_scale: f32,
    xyz_cfg_scale: f32,
    elem_mask: Option<&[bool]>,
    chg_mask: Option<&[bool]>,
) -> Result<SelfCondState, String> {
    let out_c: ForwardOutput = if let Some(g) = gpu.as_deref_mut() {
        g.forward_async(bundle, topo, x_in, a_in, c_in, e_in, t_val, sc_in, cond_mods, None)
            .await?
    } else {
        forward_cpu(bundle, topo, x_in, a_in, c_in, e_in, t_val, sc_in, cond_mods, None)?
    };

    let (x_hat, a_log, c_log, e_log) = if use_cfg {
        let out_u = if let Some(g) = gpu {
            g.forward_async(bundle, topo, x_in, a_in, c_in, e_in, t_val, sc_in, null_mods, None)
                .await?
        } else {
            forward_cpu(bundle, topo, x_in, a_in, c_in, e_in, t_val, sc_in, null_mods, None)?
        };
        let mut x_m = vec![[0.0f32; 3]; topo.n_nodes];
        for i in 0..topo.n_nodes {
            x_m[i] = [
                out_u.x[i][0] + xyz_cfg_scale * (out_c.x[i][0] - out_u.x[i][0]),
                out_u.x[i][1] + xyz_cfg_scale * (out_c.x[i][1] - out_u.x[i][1]),
                out_u.x[i][2] + xyz_cfg_scale * (out_c.x[i][2] - out_u.x[i][2]),
            ];
        }
        remove_com_in_place(&mut x_m);
        let a_m: Vec<f32> = out_u
            .a_logits
            .iter()
            .zip(&out_c.a_logits)
            .map(|(&u, &c)| u + cfg_scale * (c - u))
            .collect();
        let c_m: Vec<f32> = out_u
            .c_logits
            .iter()
            .zip(&out_c.c_logits)
            .map(|(&u, &c)| u + cfg_scale * (c - u))
            .collect();
        let e_m: Vec<f32> = out_u
            .e_logits
            .iter()
            .zip(&out_c.e_logits)
            .map(|(&u, &c)| u + cfg_scale * (c - u))
            .collect();
        (x_m, a_m, c_m, e_m)
    } else {
        (out_c.x, out_c.a_logits, out_c.c_logits, out_c.e_logits)
    };

    let cfg = &bundle.header.config;
    let a_probs = softmax_rows(&a_log, topo.n_nodes, cfg.n_atom_types(), elem_mask);
    let c_probs = softmax_rows(&c_log, topo.n_nodes, cfg.n_charges, chg_mask);
    let e_probs = softmax_rows(&e_log, topo.n_up, cfg.n_bond_types, None);

    Ok(SelfCondState {
        x: x_hat,
        a_probs,
        c_probs,
        e_probs,
    })
}

/// SE(3)-equivariant Euclidean Minimum-Spanning-Tree (MST) rigid subcluster cohesion pull.
/// Closes inter-fragment spatial gaps (`> gap_thresh`, e.g. `2.15 A`) rigidly without distorting intra-fragment geometry!
fn apply_mst_cohesion_pull(x: &mut [[f32; 3]], gap_thresh: f32, target_bridge: f32, step_gain: f32) {
    let n = x.len();
    if n < 2 || step_gain <= 1e-6 {
        return;
    }
    let mut in_tree = vec![false; n];
    let mut min_d2 = vec![f32::INFINITY; n];
    let mut parent = vec![0usize; n];
    in_tree[0] = true;
    for v in 1..n {
        let dx = x[v][0] - x[0][0];
        let dy = x[v][1] - x[0][1];
        let dz = x[v][2] - x[0][2];
        min_d2[v] = dx * dx + dy * dy + dz * dz;
        parent[v] = 0;
    }

    let mut mst_edges: Vec<(usize, usize, f32)> = Vec::with_capacity(n - 1);
    for _ in 0..(n - 1) {
        let mut best_v = None;
        let mut best_d2 = f32::INFINITY;
        for v in 0..n {
            if !in_tree[v] && min_d2[v] < best_d2 {
                best_d2 = min_d2[v];
                best_v = Some(v);
            }
        }
        let Some(v) = best_v else { break };
        in_tree[v] = true;
        mst_edges.push((parent[v], v, best_d2.sqrt()));
        for w in 0..n {
            if !in_tree[w] {
                let dx = x[w][0] - x[v][0];
                let dy = x[w][1] - x[v][1];
                let dz = x[w][2] - x[v][2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < min_d2[w] {
                    min_d2[w] = d2;
                    parent[w] = v;
                }
            }
        }
    }

    if !mst_edges.iter().any(|&(_, _, d)| d > gap_thresh) {
        return;
    }

    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (ei, &(u, v, _)) in mst_edges.iter().enumerate() {
        adj[u].push((v, ei));
        adj[v].push((u, ei));
    }

    let mut in_u = vec![false; n];
    let mut stack = Vec::with_capacity(n);
    for (ei, &(u, v, _)) in mst_edges.iter().enumerate() {
        let dx = x[v][0] - x[u][0];
        let dy = x[v][1] - x[u][1];
        let dz = x[v][2] - x[u][2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist <= gap_thresh {
            continue;
        }
        in_u.fill(false);
        stack.clear();
        in_u[u] = true;
        stack.push(u);
        let mut cnt_u = 1usize;
        while let Some(cur) = stack.pop() {
            for &(nb, edge_id) in &adj[cur] {
                if edge_id != ei && !in_u[nb] {
                    in_u[nb] = true;
                    cnt_u += 1;
                    stack.push(nb);
                }
            }
        }
        let cnt_v = n - cnt_u;
        let gap = (dist - target_bridge).max(0.0);
        let pull = (step_gain * gap).min(gap);
        let inv_d = 1.0 / dist.max(1e-6);
        let ux = dx * inv_d;
        let uy = dy * inv_d;
        let uz = dz * inv_d;
        let shift_u = ((cnt_v as f32) / (n as f32)) * pull;
        let shift_v = -((cnt_u as f32) / (n as f32)) * pull;
        for i in 0..n {
            let s = if in_u[i] { shift_u } else { shift_v };
            x[i][0] += s * ux;
            x[i][1] += s * uy;
            x[i][2] += s * uz;
        }
    }
    remove_com_in_place(x);
}

/// Extract the largest connected molecular component (preferring components containing pinned atoms if any).
fn extract_largest_fragment_raw(
    x: &[[f32; 3]],
    a: &[usize],
    c: &[usize],
    e_upper: &[usize],
    pinned: &[Option<usize>],
) -> (Vec<[f32; 3]>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = x.len();
    if n < 2 {
        return (x.to_vec(), a.to_vec(), c.to_vec(), e_upper.to_vec());
    }
    let mut adj = vec![Vec::new(); n];
    let mut edge_mat = vec![0usize; n * n];
    let mut k = 0usize;
    for u in 0..n {
        for v in (u + 1)..n {
            let bt = e_upper[k];
            k += 1;
            edge_mat[u * n + v] = bt;
            edge_mat[v * n + u] = bt;
            if bt > 0 && bt < 6 {
                adj[u].push(v);
                adj[v].push(u);
            }
        }
    }

    let mut visited = vec![false; n];
    let mut comps: Vec<Vec<usize>> = Vec::new();
    let mut stack = Vec::new();
    for i in 0..n {
        if !visited[i] {
            visited[i] = true;
            stack.push(i);
            let mut comp = Vec::new();
            while let Some(u) = stack.pop() {
                comp.push(u);
                for &v in &adj[u] {
                    if !visited[v] {
                        visited[v] = true;
                        stack.push(v);
                    }
                }
            }
            comp.sort_unstable();
            comps.push(comp);
        }
    }
    if comps.len() <= 1 {
        return (x.to_vec(), a.to_vec(), c.to_vec(), e_upper.to_vec());
    }

    // Score by (pinned_count, atom_count, heavy_atom_count)
    let best = comps
        .into_iter()
        .max_by_key(|comp| {
            let pin_cnt = comp.iter().filter(|&&i| i < pinned.len() && pinned[i].is_some()).count();
            let heavy_cnt = comp.iter().filter(|&&i| a[i] != 0).count();
            (pin_cnt, comp.len(), heavy_cnt)
        })
        .unwrap();

    if best.len() < 2 {
        return (x.to_vec(), a.to_vec(), c.to_vec(), e_upper.to_vec());
    }
    let m = best.len();
    let mut x_sub: Vec<[f32; 3]> = best.iter().map(|&i| x[i]).collect();
    remove_com_in_place(&mut x_sub);
    let a_sub: Vec<usize> = best.iter().map(|&i| a[i]).collect();
    let c_sub: Vec<usize> = best.iter().map(|&i| c[i]).collect();
    let mut e_sub = Vec::with_capacity(m * (m - 1) / 2);
    for i in 0..m {
        for j in (i + 1)..m {
            e_sub.push(edge_mat[best[i] * n + best[j]]);
        }
    }
    (x_sub, a_sub, c_sub, e_sub)
}

async fn sample_single_trajectory_async<F>(
    bundle: &ModelBundle,
    mut gpu: Option<&mut WgpuBackend>,
    opts: &SampleOptions,
    seed: u64,
    eff_prior_scale: f32,
    eff_cohesion: f32,
    on_step: &mut F,
) -> Result<(SampledMolecule, Vec<[f32; 3]>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<Option<usize>>), String>
where
    F: FnMut(usize, usize, &[[f32; 3]], &[usize]),
{
    let cfg = &bundle.header.config;
    let mut rng = DetRng::new(seed);
    let n_atoms = opts
        .n_atoms
        .unwrap_or_else(|| sample_empirical_size(bundle, opts.domain.as_deref(), &mut rng))
        .max(opts.required_elements.len())
        .max(2);

    let topo = GraphTopology::complete(n_atoms);
    let a_num = cfg.n_atom_types();
    let c_num = cfg.n_charges;
    let b_num = cfg.n_bond_types;

    // Build element & charge masks
    let mut effective_allow: Option<Vec<String>> = opts.allowed_elements.clone().or_else(|| {
        opts.domain.as_ref().and_then(|dom| {
            if dom == "all" || dom == "open118" {
                None
            } else {
                bundle.header.per_source_active_elements.get(dom).cloned()
            }
        })
    });
    if let Some(ref mut allow_list) = effective_allow {
        for req in &opts.required_elements {
            if !allow_list.contains(req) {
                allow_list.push(req.clone());
            }
        }
    }

    let elem_mask: Option<Vec<bool>> = effective_allow.as_ref().map(|allow| {
        cfg.atom_types.iter().map(|sym| allow.contains(sym)).collect()
    });
    if let Some(ref m) = elem_mask {
        if !m.iter().any(|&b| b) {
            return Err("No allowed elements match model vocabulary".to_string());
        }
    }

    let chg_mask: Option<Vec<bool>> = opts.allowed_charges.as_ref().map(|allow| {
        cfg.formal_charges.iter().map(|q| allow.contains(q)).collect()
    });
    if let Some(ref m) = chg_mask {
        if !m.iter().any(|&b| b) {
            return Err("No allowed formal charges match model vocabulary".to_string());
        }
    }

    // Resolve zero-retrain pinned elements
    let mut pinned_atoms: Vec<Option<usize>> = vec![None; n_atoms];
    for (idx, req_sym) in opts.required_elements.iter().enumerate() {
        let pos = cfg
            .atom_types
            .iter()
            .position(|s| s == req_sym)
            .ok_or_else(|| format!("Unknown required element symbol '{req_sym}'"))?;
        pinned_atoms[idx] = Some(pos);
    }
    let has_pins = !opts.required_elements.is_empty();

    // Size-adaptive zero-COM Gaussian prior `sigma(N) = prior_std * prior_scale * (N / 25)^(1/3)`
    let p_scale = eff_prior_scale.clamp(0.4, 2.0);
    let std_n = (if cfg.size_scale_prior {
        cfg.prior_std * ((n_atoms.max(2) as f32) / 25.0).powf(1.0 / 3.0)
    } else {
        cfg.prior_std
    }) * p_scale;
    let max_rad = 2.6 * std_n;
    let mut x_t = vec![[0.0f32; 3]; n_atoms];
    for p in &mut x_t {
        let mut px = rng.next_standard_normal() * std_n;
        let mut py = rng.next_standard_normal() * std_n;
        let mut pz = rng.next_standard_normal() * std_n;
        // When compact single-molecule prior is active, softly pull extreme >2.6*sigma radial outliers onto shell
        if p_scale < 0.999 {
            let r = (px * px + py * py + pz * pz).sqrt();
            if r > max_rad {
                let s = max_rad / r;
                px *= s;
                py *= s;
                pz *= s;
            }
        }
        *p = [px, py, pz];
    }
    remove_com_in_place(&mut x_t);

    let mut a_t = vec![a_num; n_atoms];
    for i in 0..n_atoms {
        if let Some(p_idx) = pinned_atoms[i] {
            a_t[i] = p_idx;
        }
    }
    let mut c_t = vec![c_num; n_atoms];
    let mut e_t = vec![b_num; topo.n_up];

    // Optional property conditioner precomputation (once per trajectory!)
    let (cond_mods, null_mods, use_cfg) = if let (Some(cv), Some(ca)) = (&opts.cond_values, &opts.cond_active) {
        if bundle.header.conditioner.is_some() && ca.iter().any(|&b| b) {
            let cm = precompute_condition_modulations(bundle, cv, ca)?;
            let do_cfg = (opts.cfg_scale - 1.0).abs() > 1e-5 || (opts.xyz_cfg_scale - 1.0).abs() > 1e-5;
            let nm = if do_cfg {
                let null_mask = vec![false; ca.len()];
                precompute_condition_modulations(bundle, cv, &null_mask)?
            } else {
                None
            };
            (cm, nm, do_cfg)
        } else {
            (None, None, false)
        }
    } else {
        (None, None, false)
    };

    let n_steps = opts.n_steps.max(1);
    let tspan = build_inference_tspan(n_steps, &opts.grid_type);
    let alphas: Vec<[f32; 4]> = tspan.iter().map(|&t| eval_alpha_t(cfg, t)).collect();
    let mut sc_state: Option<SelfCondState> = None;

    for step_idx in 0..n_steps {
        let t_i = tspan[step_idx];
        let s_i = tspan[step_idx + 1];
        let dt = s_i - t_i;
        let last_step = step_idx + 1 == n_steps;

        let mut pred1 = run_single_predict(
            bundle,
            gpu.as_deref_mut(),
            &topo,
            &x_t,
            &a_t,
            &c_t,
            &e_t,
            t_i,
            sc_state.as_ref(),
            cond_mods.as_ref(),
            null_mods.as_ref(),
            use_cfg,
            opts.cfg_scale,
            opts.xyz_cfg_scale,
            elem_mask.as_deref(),
            chg_mask.as_deref(),
        )
        .await?;

        if eff_cohesion > 0.0 && (0.25..=0.85).contains(&t_i) {
            let gain = eff_cohesion * (1.0 - t_i);
            apply_mst_cohesion_pull(&mut pred1.x, 2.80, 2.20, gain);
        }

        let a_next = campbell_ctmc_step(
            &pred1.a_probs,
            &a_t,
            a_num,
            alphas[step_idx][1],
            alphas[step_idx + 1][1],
            dt,
            a_num,
            opts.stochasticity,
            opts.cat_temperature,
            last_step,
            if has_pins { Some(&pinned_atoms) } else { None },
            &mut rng,
        );
        let c_next = campbell_ctmc_step(
            &pred1.c_probs,
            &c_t,
            c_num,
            alphas[step_idx][2],
            alphas[step_idx + 1][2],
            dt,
            c_num,
            opts.stochasticity,
            opts.cat_temperature,
            last_step,
            None,
            &mut rng,
        );
        let e_next = campbell_ctmc_step(
            &pred1.e_probs,
            &e_t,
            b_num,
            alphas[step_idx][3],
            alphas[step_idx + 1][3],
            dt,
            b_num,
            opts.stochasticity,
            opts.cat_temperature,
            last_step,
            None,
            &mut rng,
        );

        let (x_next, sc_next) = if last_step {
            let mut xn = pred1.x.clone();
            remove_com_in_place(&mut xn);
            (xn, pred1)
        } else {
            let ax_t = alphas[step_idx][0];
            let ax_s = alphas[step_idx + 1][0];
            let wx = ((ax_s - ax_t) / (1.0 - ax_t).max(1e-5)).clamp(0.0, 1.0);
            let mut x_euler = vec![[0.0f32; 3]; n_atoms];
            for i in 0..n_atoms {
                x_euler[i] = [
                    x_t[i][0] + wx * (pred1.x[i][0] - x_t[i][0]),
                    x_t[i][1] + wx * (pred1.x[i][1] - x_t[i][1]),
                    x_t[i][2] + wx * (pred1.x[i][2] - x_t[i][2]),
                ];
            }
            remove_com_in_place(&mut x_euler);

            if opts.solver == "heun" {
                let sc_pass2 = if cfg.self_conditioning { Some(&pred1) } else { None };
                let mut pred2 = run_single_predict(
                    bundle,
                    gpu.as_deref_mut(),
                    &topo,
                    &x_euler,
                    &a_next,
                    &c_next,
                    &e_next,
                    s_i,
                    sc_pass2,
                    cond_mods.as_ref(),
                    null_mods.as_ref(),
                    use_cfg,
                    opts.cfg_scale,
                    opts.xyz_cfg_scale,
                    elem_mask.as_deref(),
                    chg_mask.as_deref(),
                )
                .await?;
                if eff_cohesion > 0.0 && (0.25..=0.85).contains(&s_i) {
                    let gain = eff_cohesion * (1.0 - s_i);
                    apply_mst_cohesion_pull(&mut pred2.x, 2.80, 2.20, gain);
                }
                let mut x_heun = vec![[0.0f32; 3]; n_atoms];
                for i in 0..n_atoms {
                    let xbar = [
                        0.5 * (pred1.x[i][0] + pred2.x[i][0]),
                        0.5 * (pred1.x[i][1] + pred2.x[i][1]),
                        0.5 * (pred1.x[i][2] + pred2.x[i][2]),
                    ];
                    x_heun[i] = [
                        x_t[i][0] + wx * (xbar[0] - x_t[i][0]),
                        x_t[i][1] + wx * (xbar[1] - x_t[i][1]),
                        x_t[i][2] + wx * (xbar[2] - x_t[i][2]),
                    ];
                }
                remove_com_in_place(&mut x_heun);
                (x_heun, pred2)
            } else {
                (x_euler, pred1)
            }
        };

        x_t = x_next;
        a_t = a_next;
        c_t = c_next;
        e_t = e_next;
        sc_state = if cfg.self_conditioning { Some(sc_next) } else { None };

        on_step(step_idx + 1, n_steps, &x_t, &a_t);
    }

    let mol = inspect_sampled_molecule(
        &cfg.atom_types,
        &cfg.formal_charges,
        &x_t,
        &a_t,
        &c_t,
        &e_t,
    );
    Ok((mol, x_t, a_t, c_t, e_t, pinned_atoms))
}

pub async fn sample_molecule_async<F>(
    bundle: &ModelBundle,
    mut gpu: Option<&mut WgpuBackend>,
    opts: &SampleOptions,
    mut on_step: F,
) -> Result<SampledMolecule, String>
where
    F: FnMut(usize, usize, &[[f32; 3]], &[usize]),
{
    let mode = opts.single_molecule_mode.as_str();
    let eff_prior_scale = opts.prior_scale;
    let eff_cohesion = if opts.cohesion_strength > 0.0 {
        opts.cohesion_strength
    } else if matches!(mode, "cohesion" | "largest_fragment" | "strict_retry" | "valid_connected") {
        0.22
    } else {
        0.0
    };

    let attempts = if matches!(mode, "strict_retry" | "valid_connected") {
        opts.max_retries.max(1)
    } else {
        1
    };

    let mut best_pack: Option<(
        SampledMolecule,
        Vec<[f32; 3]>,
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
        Vec<Option<usize>>,
    )> = None;
    for attempt in 0..attempts {
        let cur_seed = opts.seed.wrapping_add((attempt as u64).wrapping_mul(9973));
        let pack = sample_single_trajectory_async(
            bundle,
            gpu.as_deref_mut(),
            opts,
            cur_seed,
            eff_prior_scale,
            eff_cohesion,
            &mut on_step,
        )
        .await?;
        let accept = if mode == "valid_connected" {
            pack.0.connected && pack.0.valence_valid
        } else {
            pack.0.connected
        };
        if accept {
            return Ok(pack.0);
        }
        let replace = match &best_pack {
            None => true,
            Some((prev_m, _, _, _, _, _)) => {
                (pack.0.connected as u8, pack.0.valence_valid as u8, usize::MAX - pack.0.n_fragments)
                    > (prev_m.connected as u8, prev_m.valence_valid as u8, usize::MAX - prev_m.n_fragments)
            }
        };
        if replace {
            best_pack = Some(pack);
        }
    }

    let (mol, x_t, a_t, c_t, e_t, pinned_atoms) = best_pack.unwrap();
    if matches!(mode, "largest_fragment" | "strict_retry" | "valid_connected") && !mol.connected {
        let cfg = &bundle.header.config;
        let (xs, as_, cs, es) = extract_largest_fragment_raw(&x_t, &a_t, &c_t, &e_t, &pinned_atoms);
        return Ok(inspect_sampled_molecule(
            &cfg.atom_types,
            &cfg.formal_charges,
            &xs,
            &as_,
            &cs,
            &es,
        ));
    }
    Ok(mol)
}
