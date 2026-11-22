use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};

pub const ELEMENTS_118: [&str; 118] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

pub fn elem_id_to_symbol(id: usize) -> &'static str {
    ELEMENTS_118.get(id).copied().unwrap_or("C")
}

pub fn symbol_to_elem_id(sym: &str) -> Option<usize> {
    ELEMENTS_118.iter().position(|&s| s.eq_ignore_ascii_case(sym))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampledBond {
    pub u: usize,
    pub v: usize,
    pub bond_type_id: usize,
    pub bond_type: String,
}

/// Unified molecule output matching `flowmol-rs`'s `SampledMolecule` schema + `Uni-3DAR-Mol` telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampledUni3darMolecule {
    pub n_atoms: usize,
    pub n_heavy: usize,
    pub formula: String,
    pub atom_symbols: Vec<String>,
    pub atomic_numbers: Vec<usize>,
    pub formal_charges: Vec<i32>,
    pub positions: Vec<[f32; 3]>,
    pub bonds: Vec<SampledBond>,
    pub n_fragments: usize,
    pub connected: bool,
    pub valence_valid: bool,
    pub has_metal: bool,
    pub has_coordinate_bond: bool,
    pub min_dist: f32,
    pub elements: Vec<String>,
    pub metals: Vec<serde_json::Value>,
    // Uni-3DAR-Mol specific fields:
    pub reconcile_status: String,
    pub was_reconciled: bool,
    pub heavy_rmsd_a: f32,
    pub delta_h_count: i32,
    pub stage1_utility: Option<f32>,
    pub stage2_utility: Option<f32>,
    pub predicted_props: BTreeMap<String, f32>,
    pub funnel_spec: [usize; 3],
    pub decode_steps: usize,
}

impl SampledUni3darMolecule {
    pub fn to_sdf(&self, title: &str) -> String {
        let mut s = String::new();
        s.push_str(&format!("{title} | {}\n", self.formula));
        s.push_str("  uni3dar           3D\n\n");
        s.push_str(&format!(
            "{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000\n",
            self.n_atoms,
            self.bonds.len()
        ));
        for i in 0..self.n_atoms {
            let p = self.positions[i];
            s.push_str(&format!(
                "{:>10.4}{:>10.4}{:>10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n",
                p[0], p[1], p[2], self.atom_symbols[i]
            ));
        }
        for b in &self.bonds {
            let sdf_bt = match b.bond_type_id {
                1 => 1,
                2 => 2,
                3 => 3,
                4 => 4,
                _ => 1,
            };
            s.push_str(&format!("{:>3}{:>3}{:>3}  0  0  0  0\n", b.u + 1, b.v + 1, sdf_bt));
        }
        let charged: Vec<(usize, i32)> = self
            .formal_charges
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, q)| q != 0)
            .collect();
        for chunk in charged.chunks(8) {
            s.push_str(&format!("M  CHG{:>3}", chunk.len()));
            for &(idx, q) in chunk {
                s.push_str(&format!(" {:>3} {:>3}", idx + 1, q));
            }
            s.push('\n');
        }
        s.push_str("M  END\n$$$$\n");
        s
    }

    pub fn to_xyz(&self, comment: &str) -> String {
        let mut s = format!("{}\n{} | {}\n", self.n_atoms, self.formula, comment);
        for i in 0..self.n_atoms {
            let p = self.positions[i];
            s.push_str(&format!(
                "{:<3} {:>12.6} {:>12.6} {:>12.6}\n",
                self.atom_symbols[i], p[0], p[1], p[2]
            ));
        }
        s
    }
}

/// Standard covalent radii `r_cov` (Å) matching `uni3dar_slot._ELEMENT_RCOV_A` & RDKit `GetRcovalent`.
pub fn rcov_angstrom(sym: &str) -> f32 {
    match sym {
        "H" => 0.31,
        "B" => 0.84,
        "C" => 0.76,
        "N" => 0.71,
        "O" => 0.66,
        "F" => 0.57,
        "Si" => 1.11,
        "P" => 1.07,
        "S" => 1.05,
        "Cl" => 1.02,
        "As" => 1.19,
        "Se" => 1.20,
        "Br" => 1.20,
        "I" => 1.39,
        _ => 0.80,
    }
}

pub fn max_coord_element(sym: &str) -> i32 {
    match sym {
        "H" | "F" | "Cl" | "Br" | "I" => 1,
        "O" => 2,
        "B" | "N" => 3,
        "C" | "Si" | "Se" => 4,
        "P" | "As" => 5,
        "S" => 6,
        _ => 8,
    }
}

/// Pyykko `(r1, r2, r3)` single / double / triple covalent radii (Å) from `valence_reconciler.py`.
fn pyykko_radii(sym: &str) -> [f64; 3] {
    match sym {
        "H" => [0.32, 0.32, 0.32],
        "B" => [0.85, 0.78, 0.73],
        "C" => [0.75, 0.67, 0.60],
        "N" => [0.71, 0.60, 0.54],
        "O" => [0.63, 0.57, 0.53],
        "F" => [0.64, 0.59, 0.53],
        "Si" => [1.16, 1.07, 1.02],
        "P" => [1.11, 1.02, 0.94],
        "S" => [1.03, 0.94, 0.95],
        "Cl" => [0.99, 0.95, 0.93],
        "As" => [1.21, 1.14, 1.06],
        "Se" => [1.16, 1.07, 1.07],
        "Br" => [1.14, 1.09, 1.10],
        "I" => [1.33, 1.29, 1.25],
        _ => [0.80, 0.72, 0.66],
    }
}

#[inline]
fn ideal_bond_length(sa: &str, sb: &str, order: usize) -> f64 {
    let idx = order.clamp(1, 3) - 1;
    pyykko_radii(sa)[idx] + pyykko_radii(sb)[idx]
}

fn max_clean_heavy_degree(sym: &str) -> Option<usize> {
    match sym {
        "F" | "Cl" | "Br" | "I" => Some(1),
        "O" => Some(2),
        "B" | "C" | "Si" | "N" => Some(4), // B- and N+ allowed in ALLOWED_DIPOLAR_PAIRS
        "P" | "As" => Some(5),
        "S" | "Se" => Some(6),
        _ => None,
    }
}

fn equilibrium_xh_length(sym: &str) -> f64 {
    match sym {
        "C" => 1.09,
        "N" => 1.01,
        "O" => 0.97,
        "B" => 1.19,
        "Si" => 1.48,
        "P" => 1.42,
        "S" => 1.34,
        "As" => 1.52,
        "Se" => 1.46,
        "F" => 0.92,
        "Cl" => 1.27,
        "Br" => 1.41,
        "I" => 1.61,
        _ => 1.09,
    }
}

/// Allowed `(formal_charge, total_valence, state_penalty)` matching `_allowed_clean_valence_states`.
fn allowed_clean_valence_states(sym: &str, nb_syms: &[&str]) -> Vec<(i32, i32, f64)> {
    let has_o_nb = nb_syms.contains(&"O");
    let has_polar_nb = nb_syms
        .iter()
        .any(|&s| matches!(s, "N" | "S" | "P" | "As" | "Se"));

    match sym {
        "C" | "Si" => vec![(0, 4, 0.0)],
        "B" => vec![(0, 3, 0.0), (-1, 4, 0.36)],
        "F" | "Cl" | "Br" | "I" => vec![(0, 1, 0.0)],
        "O" => {
            let mut st = vec![(0, 2, 0.0)];
            if nb_syms.len() == 1 {
                st.push((-1, 1, if has_polar_nb { 0.22 } else { 0.38 }));
            }
            st
        }
        "N" => {
            let mut st = vec![(0, 3, 0.0)];
            st.push((1, 4, if has_o_nb { 0.22 } else { 0.38 }));
            st
        }
        "S" => {
            let mut st = vec![(0, 2, 0.0), (0, 4, 0.12), (0, 6, 0.18)];
            if has_o_nb {
                st.push((1, 3, 0.24));
            }
            st
        }
        "Se" => {
            let mut st = vec![(0, 2, 0.0), (0, 4, 0.15), (0, 6, 0.22)];
            if has_o_nb {
                st.push((1, 3, 0.28));
            }
            st
        }
        "P" | "As" => {
            let mut st = vec![(0, 3, 0.0), (0, 5, 0.10)];
            if has_o_nb {
                st.push((1, 4, 0.24));
            }
            st
        }
        _ => vec![(0, 4, 0.5)],
    }
}

fn geometry_hybridization_hint(coords: &[[f64; 3]], idx: usize, nb_indices: &[usize]) -> Option<i32> {
    let deg = nb_indices.len();
    if deg < 2 {
        return None;
    }
    let p0 = coords[idx];
    let mut vecs = Vec::with_capacity(deg);
    for &j in nb_indices {
        let v = sub3(coords[j], p0);
        let nrm = norm3(v);
        if nrm > 1e-5 {
            vecs.push(scale3(v, 1.0 / nrm));
        }
    }
    if vecs.len() < 2 {
        return None;
    }
    let mut sum_ang = 0.0f64;
    let mut cnt = 0usize;
    for a in 0..vecs.len() {
        for b in (a + 1)..vecs.len() {
            let c = dot3(vecs[a], vecs[b]).clamp(-1.0, 1.0);
            sum_ang += c.acos().to_degrees();
            cnt += 1;
        }
    }
    let mean_ang = sum_ang / (cnt.max(1) as f64);
    if deg == 2 {
        if mean_ang >= 152.0 {
            return Some(3);
        }
        if mean_ang >= 115.5 {
            return Some(2);
        }
        return Some(1);
    }
    if deg == 3 {
        if mean_ang >= 116.5 {
            return Some(2);
        }
        return Some(1);
    }
    Some(1)
}

#[derive(Debug, Clone)]
pub struct ReconciledGraph {
    pub symbols: Vec<String>,
    pub positions: Vec<[f32; 3]>,
    pub formal_charges: Vec<i32>,
    pub bonds: Vec<SampledBond>,
    pub status: String,
    pub was_reconciled: bool,
    pub heavy_rmsd_a: f32,
    pub delta_h_count: i32,
    pub valence_valid: bool,
    pub n_fragments: usize,
}

/// Pure-Rust port of `ValenceReconciler` (`valence_reconciler.py`) with strictly frozen heavy-atom coordinates (`heavy_rmsd_A == 0.0`).
pub fn reconcile_point_cloud(
    elem_ids: &[usize],
    coords: &[[f32; 3]],
    bond_order_probs: Option<&[f32]>, // (K * K * 5) from PostTreeGraphRefiner
    max_csp_nodes: usize,
    allow_largest_heavy_fragment: bool,
) -> Option<ReconciledGraph> {
    let k_tot = elem_ids.len();
    if k_tot == 0 || coords.len() != k_tot {
        return None;
    }
    if coords.iter().any(|p| !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite()) {
        return None;
    }

    let syms: Vec<&'static str> = elem_ids.iter().map(|&id| elem_id_to_symbol(id)).collect();
    let pos_f64: Vec<[f64; 3]> = coords
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();

    let all_heavy_idx: Vec<usize> = (0..k_tot).filter(|&i| syms[i] != "H").collect();
    let all_h_idx: Vec<usize> = (0..k_tot).filter(|&i| syms[i] == "H").collect();

    // Pure H2 special case
    if all_heavy_idx.is_empty() {
        if all_h_idx.len() == 2 {
            let d = dist3(pos_f64[0], pos_f64[1]);
            if (0.55..=1.10).contains(&d) {
                return Some(ReconciledGraph {
                    symbols: vec!["H".to_string(), "H".to_string()],
                    positions: coords.to_vec(),
                    formal_charges: vec![0, 0],
                    bonds: vec![SampledBond {
                        u: 0,
                        v: 1,
                        bond_type_id: 1,
                        bond_type: "SINGLE".to_string(),
                    }],
                    status: "clean_raw".to_string(),
                    was_reconciled: false,
                    heavy_rmsd_a: 0.0,
                    delta_h_count: 0,
                    valence_valid: true,
                    n_fragments: 1,
                });
            }
        }
        return None;
    }

    if all_heavy_idx.iter().any(|&i| max_clean_heavy_degree(syms[i]).is_none()) {
        return None;
    }

    // Build initial heavy covalent adjacency across all heavy atoms (`1.28 * (rcov_u + rcov_v)` + refiner prior `>= 0.55`)
    let nh_all = all_heavy_idx.len();
    let mut d_hh_all = vec![0.0f64; nh_all * nh_all];
    for u in 0..nh_all {
        for v in (u + 1)..nh_all {
            let d = dist3(pos_f64[all_heavy_idx[u]], pos_f64[all_heavy_idx[v]]);
            if d < 0.72 {
                return None; // Heavy-atom steric collision
            }
            d_hh_all[u * nh_all + v] = d;
            d_hh_all[v * nh_all + u] = d;
        }
    }

    let mut raw_adj_all: Vec<Vec<usize>> = vec![Vec::new(); nh_all];
    for u in 0..nh_all {
        for v in (u + 1)..nh_all {
            let su = syms[all_heavy_idx[u]];
            let sv = syms[all_heavy_idx[v]];
            let d = d_hh_all[u * nh_all + v];
            let r_cov_rdkit = (rcov_angstrom(su) + rcov_angstrom(sv)) as f64;
            let r_pyykko = ideal_bond_length(su, sv, 1);
            let mut connected = d <= 1.28 * r_cov_rdkit && (0.58 * r_pyykko..=1.38 * r_pyykko).contains(&d);
            if !connected {
                if let Some(bop) = bond_order_probs {
                    let gu = all_heavy_idx[u];
                    let gv = all_heavy_idx[v];
                    let p_bonded = 1.0 - (bop[(gu * k_tot + gv) * 5] as f64);
                    if p_bonded >= 0.55 && (0.58 * r_pyykko..=1.38 * r_pyykko).contains(&d) {
                        connected = true;
                    }
                }
            }
            if connected {
                raw_adj_all[u].push(v);
                raw_adj_all[v].push(u);
            }
        }
    }

    // Connected components on heavy skeleton
    let comps = connected_components(nh_all, &raw_adj_all);
    let chosen_heavy_local: Vec<usize> = if comps.len() == 1 {
        (0..nh_all).collect()
    } else if allow_largest_heavy_fragment {
        let largest = comps.into_iter().max_by_key(|c| c.len())?;
        // Require the main scaffold to hold at least 70% of heavy atoms (never silently return a tiny 2-atom splinter of a 20-atom cloud)
        if largest.len() * 10 < nh_all * 7 {
            return None;
        }
        largest
    } else {
        return None;
    };

    let dropped_heavy_fragment = chosen_heavy_local.len() < nh_all;
    let heavy_indices: Vec<usize> = chosen_heavy_local.iter().map(|&lu| all_heavy_idx[lu]).collect();
    let n_heavy = heavy_indices.len();
    let heavy_syms: Vec<&'static str> = heavy_indices.iter().map(|&gi| syms[gi]).collect();
    let heavy_pos: Vec<[f64; 3]> = heavy_indices.iter().map(|&gi| pos_f64[gi]).collect();

    // Map raw H atoms to nearest chosen heavy atom within `[0.62, 1.48] Å`
    let mut raw_h_counts = vec![0i32; n_heavy];
    let mut attached_raw_hs: Vec<Vec<usize>> = vec![Vec::new(); n_heavy];
    let mut stray_h_count = 0usize;
    for &hi in &all_h_idx {
        let mut best_u = 0usize;
        let mut best_d = f64::INFINITY;
        for u in 0..n_heavy {
            let d = dist3(heavy_pos[u], pos_f64[hi]);
            if d < best_d {
                best_d = d;
                best_u = u;
            }
        }
        if (0.62..=1.48).contains(&best_d) {
            raw_h_counts[best_u] += 1;
            attached_raw_hs[best_u].push(hi);
        } else {
            stray_h_count += 1;
        }
    }

    // Build subgraph edges & adjacency on `0..n_heavy`
    let mut global_to_sub = HashMap::with_capacity(n_heavy);
    for (sub_i, &lu) in chosen_heavy_local.iter().enumerate() {
        global_to_sub.insert(lu, sub_i);
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_heavy];
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for (u_sub, &lu) in chosen_heavy_local.iter().enumerate() {
        for &lv in &raw_adj_all[lu] {
            if let Some(&v_sub) = global_to_sub.get(&lv) {
                if u_sub < v_sub {
                    let d = d_hh_all[lu * nh_all + lv];
                    adj[u_sub].push(v_sub);
                    adj[v_sub].push(u_sub);
                    edges.push((u_sub, v_sub, d));
                }
            }
        }
    }

    // Verify clean heavy degree caps
    let mut atom_states: Vec<Vec<(i32, i32, f64)>> = Vec::with_capacity(n_heavy);
    let mut max_atom_val = vec![0i32; n_heavy];
    for i in 0..n_heavy {
        let s = heavy_syms[i];
        let nb_syms: Vec<&str> = adj[i].iter().map(|&j| heavy_syms[j]).collect();
        let deg_cap = max_clean_heavy_degree(s)?;
        if adj[i].len() > deg_cap {
            return None;
        }
        let st = allowed_clean_valence_states(s, &nb_syms);
        let mv = st.iter().map(|x| x.1).max().unwrap_or(4);
        atom_states.push(st);
        max_atom_val[i] = mv;
    }

    // Solve clean Kekulé CSP
    let (bond_orders, formal_charges, implicit_hs) = solve_clean_kekule_csp(
        &heavy_syms,
        &heavy_pos,
        &adj,
        &edges,
        &atom_states,
        &max_atom_val,
        &raw_h_counts,
        &heavy_indices,
        k_tot,
        bond_order_probs,
        max_csp_nodes,
    )?;

    // Check if the raw cloud was ALREADY 100% clean (`!dropped_heavy_fragment && stray_h_count == 0 && implicit_hs == raw_h_counts`
    // and no H-H or non-parent X-H RDKit connectivity clash!)
    let exact_raw_h_match = !dropped_heavy_fragment
        && stray_h_count == 0
        && (0..n_heavy).all(|i| implicit_hs[i] == raw_h_counts[i]);

    if exact_raw_h_match && raw_hydrogens_clean_geometry(&heavy_syms, &heavy_pos, &attached_raw_hs, &pos_f64) {
        // Return 100% untouched raw coordinates (`was_reconciled = false`, `status = "clean_raw"`)!
        let mut out_syms = Vec::with_capacity(k_tot);
        let mut out_pos = Vec::with_capacity(k_tot);
        let mut out_q = Vec::with_capacity(k_tot);
        for i in 0..n_heavy {
            out_syms.push(heavy_syms[i].to_string());
            out_pos.push([
                heavy_pos[i][0] as f32,
                heavy_pos[i][1] as f32,
                heavy_pos[i][2] as f32,
            ]);
            out_q.push(formal_charges[i]);
        }
        let mut bonds = build_heavy_sampled_bonds(&edges, &bond_orders);
        for u in 0..n_heavy {
            for &hi in &attached_raw_hs[u] {
                let h_new_idx = out_syms.len();
                out_syms.push("H".to_string());
                out_pos.push(coords[hi]);
                out_q.push(0);
                bonds.push(SampledBond {
                    u,
                    v: h_new_idx,
                    bond_type_id: 1,
                    bond_type: "SINGLE".to_string(),
                });
            }
        }
        return Some(ReconciledGraph {
            symbols: out_syms,
            positions: out_pos,
            formal_charges: out_q,
            bonds,
            status: "clean_raw".to_string(),
            was_reconciled: false,
            heavy_rmsd_a: 0.0,
            delta_h_count: 0,
            valence_valid: true,
            n_fragments: 1,
        });
    }

    // Otherwise synthesize & relax explicit VSEPR hydrogens with strictly frozen `heavy_pos` (`heavy_rmsd_A == 0.0`)!
    let placed_hs = synthesize_and_relax_hydrogens(&heavy_syms, &heavy_pos, &adj, &edges, &bond_orders, &implicit_hs);

    let mut out_syms = Vec::with_capacity(n_heavy + placed_hs.len());
    let mut out_pos = Vec::with_capacity(n_heavy + placed_hs.len());
    let mut out_q = Vec::with_capacity(n_heavy + placed_hs.len());
    for i in 0..n_heavy {
        out_syms.push(heavy_syms[i].to_string());
        out_pos.push([
            heavy_pos[i][0] as f32,
            heavy_pos[i][1] as f32,
            heavy_pos[i][2] as f32,
        ]);
        out_q.push(formal_charges[i]);
    }
    let mut bonds = build_heavy_sampled_bonds(&edges, &bond_orders);
    for (parent_u, h_xyz) in &placed_hs {
        let h_idx = out_syms.len();
        out_syms.push("H".to_string());
        out_pos.push([h_xyz[0] as f32, h_xyz[1] as f32, h_xyz[2] as f32]);
        out_q.push(0);
        bonds.push(SampledBond {
            u: *parent_u,
            v: h_idx,
            bond_type_id: 1,
            bond_type: "SINGLE".to_string(),
        });
    }

    // Verify post-H steric safety (no H-H < 0.85 A, no non-parent X-H false bond)
    if !verify_post_h_geometry(&heavy_syms, &heavy_pos, &placed_hs) {
        return None;
    }

    let delta_h = (placed_hs.len() as i32) - (all_h_idx.len() as i32);
    let status = if delta_h == 0 && !dropped_heavy_fragment {
        "reconciled_zwitterion".to_string()
    } else {
        "reconciled_peripheral_h".to_string()
    };

    Some(ReconciledGraph {
        symbols: out_syms,
        positions: out_pos,
        formal_charges: out_q,
        bonds,
        status,
        was_reconciled: true,
        heavy_rmsd_a: 0.0,
        delta_h_count: delta_h,
        valence_valid: true,
        n_fragments: 1,
    })
}

/// Build a fallback `ReconciledGraph` directly from raw point-cloud covalent distances when reconciliation is disabled or all candidates fail.
pub fn build_raw_point_cloud_graph(elem_ids: &[usize], coords: &[[f32; 3]]) -> ReconciledGraph {
    let n = elem_ids.len();
    let symbols: Vec<String> = elem_ids.iter().map(|&id| elem_id_to_symbol(id).to_string()).collect();
    let mut adj = vec![Vec::new(); n];
    let mut bonds = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            let dx = coords[u][0] - coords[v][0];
            let dy = coords[u][1] - coords[v][1];
            let dz = coords[u][2] - coords[v][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            let r_sum = rcov_angstrom(&symbols[u]) + rcov_angstrom(&symbols[v]);
            if syms_can_bond(&symbols[u], &symbols[v], d, r_sum) {
                let bo_id = if symbols[u] != "H" && symbols[v] != "H" && d <= 0.83 * r_sum {
                    3
                } else if symbols[u] != "H" && symbols[v] != "H" && d <= 0.91 * r_sum {
                    2
                } else {
                    1
                };
                let bname = match bo_id {
                    3 => "TRIPLE",
                    2 => "DOUBLE",
                    _ => "SINGLE",
                };
                adj[u].push(v);
                adj[v].push(u);
                bonds.push(SampledBond {
                    u,
                    v,
                    bond_type_id: bo_id,
                    bond_type: bname.to_string(),
                });
            }
        }
    }
    let comps = connected_components(n, &adj);
    ReconciledGraph {
        symbols,
        positions: coords.to_vec(),
        formal_charges: vec![0; n],
        bonds,
        status: "raw_fallback".to_string(),
        was_reconciled: false,
        heavy_rmsd_a: 0.0,
        delta_h_count: 0,
        valence_valid: false,
        n_fragments: comps.len().max(1),
    }
}

fn syms_can_bond(su: &str, sv: &str, d: f32, r_sum: f32) -> bool {
    if su == "H" && sv == "H" {
        return (0.55..=0.90).contains(&d);
    }
    d >= 0.55 * r_sum && d <= 1.26 * r_sum
}

/// Assemble the final centered `SampledUni3darMolecule` from a `ReconciledGraph` + Critic telemetry.
pub fn finalize_sampled_molecule(
    rg: ReconciledGraph,
    stage1_utility: Option<f32>,
    stage2_utility: Option<f32>,
    predicted_props: BTreeMap<String, f32>,
    funnel_spec: [usize; 3],
    decode_steps: usize,
) -> SampledUni3darMolecule {
    let n = rg.symbols.len();
    let mut positions = rg.positions;
    if n > 0 {
        // Rigidly center at `(0, 0, 0)` (SE(3) translation preserves `heavy_rmsd_a == 0.0` and all internal distances!)
        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        let mut cz = 0.0f64;
        for p in &positions {
            cx += p[0] as f64;
            cy += p[1] as f64;
            cz += p[2] as f64;
        }
        let inv = 1.0 / (n as f64);
        cx *= inv;
        cy *= inv;
        cz *= inv;
        for p in &mut positions {
            p[0] = ((p[0] as f64) - cx) as f32;
            p[1] = ((p[1] as f64) - cy) as f32;
            p[2] = ((p[2] as f64) - cz) as f32;
        }
    }

    let mut min_dist = if n >= 2 { f32::INFINITY } else { 0.0 };
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = positions[i][0] - positions[j][0];
            let dy = positions[i][1] - positions[j][1];
            let dz = positions[i][2] - positions[j][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d < min_dist {
                min_dist = d;
            }
        }
    }

    let atomic_numbers: Vec<usize> = rg
        .symbols
        .iter()
        .map(|s| symbol_to_elem_id(s).map_or(6, |id| id + 1))
        .collect();
    let n_heavy = rg.symbols.iter().filter(|s| s.as_str() != "H").count();

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for s in &rg.symbols {
        *counts.entry(s.clone()).or_insert(0) += 1;
    }
    let elements: Vec<String> = counts.keys().cloned().collect();
    let formula = hill_formula(&counts, rg.formal_charges.iter().sum());

    SampledUni3darMolecule {
        n_atoms: n,
        n_heavy,
        formula,
        atom_symbols: rg.symbols,
        atomic_numbers,
        formal_charges: rg.formal_charges,
        positions,
        bonds: rg.bonds,
        n_fragments: rg.n_fragments,
        connected: rg.n_fragments == 1 && n > 0,
        valence_valid: rg.valence_valid,
        has_metal: false,
        has_coordinate_bond: false,
        min_dist,
        elements,
        metals: Vec::new(),
        reconcile_status: rg.status,
        was_reconciled: rg.was_reconciled,
        heavy_rmsd_a: rg.heavy_rmsd_a,
        delta_h_count: rg.delta_h_count,
        stage1_utility,
        stage2_utility,
        predicted_props,
        funnel_spec,
        decode_steps,
    }
}

fn hill_formula(counts: &BTreeMap<String, usize>, net_q: i32) -> String {
    let mut out = String::new();
    let mut rem = counts.clone();
    if let Some(c_cnt) = rem.remove("C") {
        out.push('C');
        if c_cnt > 1 {
            out.push_str(&c_cnt.to_string());
        }
        if let Some(h_cnt) = rem.remove("H") {
            out.push('H');
            if h_cnt > 1 {
                out.push_str(&h_cnt.to_string());
            }
        }
    }
    for (sym, cnt) in rem {
        out.push_str(&sym);
        if cnt > 1 {
            out.push_str(&cnt.to_string());
        }
    }
    if net_q > 0 {
        out.push_str(&if net_q == 1 { "+".to_string() } else { format!("+{net_q}") });
    } else if net_q < 0 {
        out.push_str(&if net_q == -1 { "-".to_string() } else { format!("{net_q}") });
    }
    out
}

fn build_heavy_sampled_bonds(edges: &[(usize, usize, f64)], bond_orders: &[i32]) -> Vec<SampledBond> {
    edges
        .iter()
        .zip(bond_orders.iter())
        .map(|(&(u, v, _), &bo)| {
            let (bt_id, bt_name) = match bo {
                3 => (3, "TRIPLE"),
                2 => (2, "DOUBLE"),
                _ => (1, "SINGLE"),
            };
            SampledBond {
                u,
                v,
                bond_type_id: bt_id,
                bond_type: bt_name.to_string(),
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn solve_clean_kekule_csp(
    heavy_syms: &[&str],
    heavy_pos: &[[f64; 3]],
    adj: &[Vec<usize>],
    edges: &[(usize, usize, f64)],
    atom_states: &[Vec<(i32, i32, f64)>],
    max_atom_val: &[i32],
    raw_h_counts: &[i32],
    heavy_global_idx: &[usize],
    k_tot: usize,
    bond_order_probs: Option<&[f32]>,
    max_csp_nodes: usize,
) -> Option<(Vec<i32>, Vec<i32>, Vec<i32>)> {
    let n_heavy = heavy_syms.len();
    let n_edges = edges.len();
    if n_edges == 0 {
        let (q0, v0, _) = atom_states[0][0];
        if q0 != 0 {
            return None;
        }
        return Some((Vec::new(), vec![0], vec![v0]));
    }

    let hyb_hints: Vec<Option<i32>> = (0..n_heavy)
        .map(|i| geometry_hybridization_hint(heavy_pos, i, &adj[i]))
        .collect();

    let mut edge_candidates: Vec<Vec<(i32, f64)>> = Vec::with_capacity(n_edges);
    for &(u, v, dist) in edges {
        let su = heavy_syms[u];
        let sv = heavy_syms[v];
        let mut max_bo = max_atom_val[u].min(max_atom_val[v]).min(3);
        if matches!(su, "F" | "Cl" | "Br" | "I") || matches!(sv, "F" | "Cl" | "Br" | "I") {
            max_bo = 1;
        }
        let mu1 = ideal_bond_length(su, sv, 1);

        let p_bo = bond_order_probs.map(|bop| {
            let gu = heavy_global_idx[u];
            let gv = heavy_global_idx[v];
            let s = &bop[(gu * k_tot + gv) * 5..(gu * k_tot + gv + 1) * 5];
            [
                0.0f64,
                (s[1] + 0.5 * s[4]) as f64,
                (s[2] + 0.5 * s[4]) as f64,
                s[3] as f64,
            ]
        });
        let has_multi_prior = p_bo.map_or(false, |p| p[2] > 0.25 || p[3] > 0.25);
        let long_cap = if has_multi_prior { 1.14 * mu1 } else { 1.07 * mu1 };
        if dist > long_cap {
            max_bo = 1;
        }
        let bo2_tol = if p_bo.map_or(false, |p| p[2] > 0.25) { 0.14 } else { 0.06 };
        let bo3_tol = if p_bo.map_or(false, |p| p[3] > 0.25) { 0.10 } else { 0.05 };

        let mut cands: Vec<(i32, f64)> = Vec::new();
        for bo in 1..=max_bo {
            let mu_b = ideal_bond_length(su, sv, bo as usize);
            if bo >= 2 && dist > mu1 + bo2_tol {
                continue;
            }
            if bo == 3 && dist > ideal_bond_length(su, sv, 2) + bo3_tol {
                continue;
            }
            let err = (dist - mu_b) / 0.065;
            let mut cost = err * err;
            if let Some(p) = p_bo {
                cost += -1.2 * p[bo as usize].max(1e-4).ln();
            }
            cands.push((bo, cost));
        }
        if cands.is_empty() {
            let err = (dist - mu1) / 0.065;
            let mut base_c = err * err;
            if let Some(p) = p_bo {
                base_c += -1.2 * p[1].max(1e-4).ln();
            }
            cands.push((1, base_c));
        }
        cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        edge_candidates.push(cands);
    }

    let mut edge_order: Vec<usize> = (0..n_edges).collect();
    edge_order.sort_by(|&a, &b| {
        edge_candidates[a]
            .len()
            .cmp(&edge_candidates[b].len())
            .then_with(|| {
                edge_candidates[a][0]
                    .1
                    .partial_cmp(&edge_candidates[b][0].1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    struct CspCtx<'a> {
        heavy_syms: &'a [&'a str],
        adj: &'a [Vec<usize>],
        edges: &'a [(usize, usize, f64)],
        atom_states: &'a [Vec<(i32, i32, f64)>],
        max_atom_val: &'a [i32],
        raw_h_counts: &'a [i32],
        hyb_hints: &'a [Option<i32>],
        edge_candidates: &'a [Vec<(i32, f64)>],
        edge_order: &'a [usize],
        rem_inc: Vec<i32>,
        cur_sum: Vec<i32>,
        chosen_bo: Vec<i32>,
        best_cost: f64,
        best_sol: Option<(Vec<i32>, Vec<i32>, Vec<i32>)>,
        nodes_visited: usize,
        max_nodes: usize,
    }

    fn score_atoms_and_charges(ctx: &CspCtx<'_>) -> Option<(f64, Vec<i32>, Vec<i32>)> {
        let n_heavy = ctx.heavy_syms.len();
        let mut per_atom_valid: Vec<Vec<(i32, i32, f64)>> = Vec::with_capacity(n_heavy);
        for i in 0..n_heavy {
            let s_used = ctx.cur_sum[i];
            let deg_i = ctx.adj[i].len() as i32;
            let n_pi = s_used - deg_i;
            let sym_i = ctx.heavy_syms[i];
            let hint = ctx.hyb_hints[i];
            if matches!(sym_i, "B" | "C" | "N" | "O") && deg_i >= 2 && n_pi >= 2 && hint != Some(3) {
                return None;
            }
            let mut valid_i = Vec::new();
            for &(q_i, v_tot, st_pen) in &ctx.atom_states[i] {
                let h_imp = v_tot - s_used;
                if !(0..=4).contains(&h_imp) {
                    continue;
                }
                let mut atom_cost = st_pen + 0.45 * ((h_imp - ctx.raw_h_counts[i]).abs() as f64);
                if matches!(sym_i, "B" | "C" | "N" | "O") {
                    if let Some(h) = hint {
                        if h == 2 && n_pi == 0 {
                            atom_cost += 0.40;
                        } else if h == 1 && n_pi >= 1 {
                            atom_cost += 0.45;
                        } else if h == 3 && n_pi < 2 {
                            atom_cost += 0.65;
                        }
                    }
                }
                valid_i.push((q_i, h_imp, atom_cost));
            }
            if valid_i.is_empty() {
                return None;
            }
            per_atom_valid.push(valid_i);
        }

        let mut fixed_cost = 0.0f64;
        let mut fixed_q = vec![0i32; n_heavy];
        let mut fixed_h = vec![0i32; n_heavy];
        let mut flex_atoms = Vec::new();
        for (i, opts) in per_atom_valid.iter().enumerate() {
            if opts.len() == 1 && opts[0].0 == 0 {
                fixed_q[i] = 0;
                fixed_h[i] = opts[0].1;
                fixed_cost += opts[0].2;
            } else {
                flex_atoms.push(i);
            }
        }
        if flex_atoms.is_empty() {
            return Some((fixed_cost, fixed_q, fixed_h));
        }

        // Charge DP over `(net_q, abs_q)` with `abs_q <= 4`
        let mut dp: HashMap<(i32, i32), (f64, Vec<(i32, i32)>)> = HashMap::new();
        dp.insert((0, 0), (fixed_cost, Vec::new()));
        for &a_idx in &flex_atoms {
            let mut nxt: HashMap<(i32, i32), (f64, Vec<(i32, i32)>)> = HashMap::new();
            for (&(net_q, abs_q), (c_sofar, choices)) in &dp {
                for &(q_opt, h_opt, c_opt) in &per_atom_valid[a_idx] {
                    let n_abs = abs_q + q_opt.abs();
                    if n_abs > 4 {
                        continue;
                    }
                    let key = (net_q + q_opt, n_abs);
                    let nc = *c_sofar + c_opt;
                    let replace = nxt.get(&key).map_or(true, |old| nc < old.0);
                    if replace {
                        let mut ch = choices.clone();
                        ch.push((q_opt, h_opt));
                        nxt.insert(key, (nc, ch));
                    }
                }
            }
            if nxt.is_empty() {
                return None;
            }
            dp = nxt;
        }

        let mut best_neutral: Option<(f64, Vec<(i32, i32)>)> = None;
        for ((net_q, _), val) in dp {
            if net_q == 0 && best_neutral.as_ref().map_or(true, |b| val.0 < b.0) {
                best_neutral = Some(val);
            }
        }
        let (tot_atom_c, chosen_flex) = best_neutral?;
        let mut adj_dip_pen = 0.0f64;
        for (pos_i, &a_idx) in flex_atoms.iter().enumerate() {
            fixed_q[a_idx] = chosen_flex[pos_i].0;
            fixed_h[a_idx] = chosen_flex[pos_i].1;
        }
        // Prefer 1-bond adjacent dipoles (`N+-O-`), while still permitting clean conjugated dipoles (`ALLOWED_DIPOLAR_PAIRS`)
        for &a_idx in &flex_atoms {
            let qa = fixed_q[a_idx];
            if qa != 0 && !ctx.adj[a_idx].iter().any(|&nb| fixed_q[nb] * qa < 0) {
                adj_dip_pen += 0.45;
            }
        }
        Some((tot_atom_c + adj_dip_pen, fixed_q, fixed_h))
    }

    fn dfs(ctx: &mut CspCtx<'_>, pos_idx: usize, edge_cost_acc: f64) {
        if edge_cost_acc >= ctx.best_cost || ctx.nodes_visited >= ctx.max_nodes {
            return;
        }
        ctx.nodes_visited += 1;

        if pos_idx == ctx.edges.len() {
            if let Some((a_cost, qs, hs)) = score_atoms_and_charges(ctx) {
                let total_c = edge_cost_acc + a_cost;
                if total_c < ctx.best_cost {
                    ctx.best_cost = total_c;
                    ctx.best_sol = Some((ctx.chosen_bo.clone(), qs, hs));
                }
            }
            return;
        }

        let e_id = ctx.edge_order[pos_idx];
        let (u, v, _) = ctx.edges[e_id];
        ctx.rem_inc[u] -= 1;
        ctx.rem_inc[v] -= 1;

        let cands_len = ctx.edge_candidates[e_id].len();
        for ci in 0..cands_len {
            let (bo, c_edge) = ctx.edge_candidates[e_id][ci];
            let ns_u = ctx.cur_sum[u] + bo;
            let ns_v = ctx.cur_sum[v] + bo;
            let deg_u = ctx.adj[u].len() as i32;
            let deg_v = ctx.adj[v].len() as i32;

            if matches!(ctx.heavy_syms[u], "B" | "C" | "N" | "O")
                && deg_u >= 2
                && ctx.hyb_hints[u] != Some(3)
                && (ns_u + ctx.rem_inc[u] - deg_u) >= 2
            {
                continue;
            }
            if matches!(ctx.heavy_syms[v], "B" | "C" | "N" | "O")
                && deg_v >= 2
                && ctx.hyb_hints[v] != Some(3)
                && (ns_v + ctx.rem_inc[v] - deg_v) >= 2
            {
                continue;
            }
            if ns_u + ctx.rem_inc[u] <= ctx.max_atom_val[u] && ns_v + ctx.rem_inc[v] <= ctx.max_atom_val[v] {
                ctx.cur_sum[u] = ns_u;
                ctx.cur_sum[v] = ns_v;
                ctx.chosen_bo[e_id] = bo;
                dfs(ctx, pos_idx + 1, edge_cost_acc + c_edge);
                ctx.cur_sum[u] -= bo;
                ctx.cur_sum[v] -= bo;
            }
        }

        ctx.rem_inc[u] += 1;
        ctx.rem_inc[v] += 1;
    }

    let mut ctx = CspCtx {
        heavy_syms,
        adj,
        edges,
        atom_states,
        max_atom_val,
        raw_h_counts,
        hyb_hints: &hyb_hints,
        edge_candidates: &edge_candidates,
        edge_order: &edge_order,
        rem_inc: (0..n_heavy).map(|i| adj[i].len() as i32).collect(),
        cur_sum: vec![0; n_heavy],
        chosen_bo: vec![1; n_edges],
        best_cost: f64::INFINITY,
        best_sol: None,
        nodes_visited: 0,
        max_nodes: max_csp_nodes,
    };
    dfs(&mut ctx, 0, 0.0);
    ctx.best_sol
}

fn raw_hydrogens_clean_geometry(
    heavy_syms: &[&str],
    heavy_pos: &[[f64; 3]],
    attached_raw_hs: &[Vec<usize>],
    pos_all: &[[f64; 3]],
) -> bool {
    let n_heavy = heavy_syms.len();
    let mut all_hs: Vec<(usize, [f64; 3])> = Vec::new();
    for u in 0..n_heavy {
        for &hi in &attached_raw_hs[u] {
            let hp = pos_all[hi];
            let d_par = dist3(heavy_pos[u], hp);
            let r_par = (rcov_angstrom(heavy_syms[u]) + 0.31) as f64;
            if d_par < 0.65 || d_par > 1.28 * r_par {
                return false;
            }
            all_hs.push((u, hp));
        }
    }
    verify_post_h_geometry(heavy_syms, heavy_pos, &all_hs)
}

fn verify_post_h_geometry(heavy_syms: &[&str], heavy_pos: &[[f64; 3]], placed_hs: &[(usize, [f64; 3])]) -> bool {
    // 1. No H-H bond/clash (`1.30 * (0.31 + 0.31) = 0.806 A` in RDKit; require `>= 0.90 A`)
    for i in 0..placed_hs.len() {
        for j in (i + 1)..placed_hs.len() {
            if dist3(placed_hs[i].1, placed_hs[j].1) < 0.90 {
                return false;
            }
        }
    }
    // 2. No non-parent heavy atom within RDKit covalent bonding threshold `1.30 * (r_cov(Y) + 0.31)`
    for &(par_u, hp) in placed_hs {
        for y in 0..heavy_syms.len() {
            if y == par_u {
                continue;
            }
            let r_cut = 1.30 * ((rcov_angstrom(heavy_syms[y]) + 0.31) as f64);
            if dist3(heavy_pos[y], hp) <= r_cut + 0.03 {
                return false;
            }
        }
    }
    true
}

/// Deterministic VSEPR explicit hydrogen placement + 36-step tangential steric relaxation (`heavy_rmsd_A === 0.0`).
fn synthesize_and_relax_hydrogens(
    heavy_syms: &[&str],
    heavy_pos: &[[f64; 3]],
    adj: &[Vec<usize>],
    edges: &[(usize, usize, f64)],
    bond_orders: &[i32],
    implicit_hs: &[i32],
) -> Vec<(usize, [f64; 3])> {
    let n_heavy = heavy_syms.len();
    let mut max_bo_atom = vec![1i32; n_heavy];
    for (&(u, v, _), &bo) in edges.iter().zip(bond_orders.iter()) {
        if bo > max_bo_atom[u] {
            max_bo_atom[u] = bo;
        }
        if bo > max_bo_atom[v] {
            max_bo_atom[v] = bo;
        }
    }

    // Per-H state: `(parent_u, r_eq, unit_dir, target_nb_dot)`
    let mut h_states: Vec<(usize, f64, [f64; 3], f64)> = Vec::new();
    let mut nb_unit_vecs: Vec<Vec<[f64; 3]>> = vec![Vec::new(); n_heavy];

    for i in 0..n_heavy {
        for &j in &adj[i] {
            let v = sub3(heavy_pos[j], heavy_pos[i]);
            let nrm = norm3(v);
            if nrm > 1e-6 {
                nb_unit_vecs[i].push(scale3(v, 1.0 / nrm));
            }
        }
        let h_cnt = implicit_hs[i].max(0) as usize;
        if h_cnt == 0 {
            continue;
        }
        let r_eq = equilibrium_xh_length(heavy_syms[i]);
        let mbo = max_bo_atom[i];
        let target_dot = if mbo == 3 {
            -1.0
        } else if mbo == 2 {
            -0.5
        } else {
            -1.0 / 3.0
        };

        let dirs = initial_vsepr_dirs(i, h_cnt, mbo, heavy_pos, adj, &nb_unit_vecs[i]);
        for u_dir in dirs {
            h_states.push((i, r_eq, u_dir, target_dot));
        }
    }

    if h_states.is_empty() {
        return Vec::new();
    }

    // 36 iterations of sphere-tangent VSEPR angle + nonbonded steric relaxation (`pos[H] = heavy_pos[p] + r_eq * u`)
    let n_h = h_states.len();
    for iter in 0..36 {
        let step_sz = 0.14 * (1.0 - 0.55 * (iter as f64 / 36.0));
        let cur_h_pos: Vec<[f64; 3]> = h_states
            .iter()
            .map(|&(p, r, u, _)| add3(heavy_pos[p], scale3(u, r)))
            .collect();

        let mut new_dirs: Vec<[f64; 3]> = Vec::with_capacity(n_h);
        for k in 0..n_h {
            let (p_u, r_eq, u_k, tgt_dot) = h_states[k];
            let hp = cur_h_pos[k];
            let mut force = [0.0f64; 3];

            // 1. Bond-angle potential with parent's bonded heavy neighbors: `-k_ang * (u_k · v_nb - tgt_dot) * v_nb`
            for &v_nb in &nb_unit_vecs[p_u] {
                let d_err = dot3(u_k, v_nb) - tgt_dot;
                force = add3(force, scale3(v_nb, -2.2 * d_err));
            }

            // 2. Geminal H-H angle & non-geminal H-H steric repulsion
            for m in 0..n_h {
                if m == k {
                    continue;
                }
                if h_states[m].0 == p_u {
                    let u_m = h_states[m].2;
                    let d_err = dot3(u_k, u_m) - tgt_dot;
                    force = add3(force, scale3(u_m, -2.4 * d_err));
                } else {
                    let diff = sub3(hp, cur_h_pos[m]);
                    let d = norm3(diff).max(0.25);
                    if d < 2.05 {
                        let mag = 1.8 * (2.05 - d) / (d * d);
                        force = add3(force, scale3(diff, mag * r_eq));
                    }
                }
            }

            // 3. Strong barrier against non-parent heavy atoms (`r_safe = 1.48 * (r_cov(Y) + 0.32)`)
            for y in 0..n_heavy {
                if y == p_u {
                    continue;
                }
                let diff = sub3(hp, heavy_pos[y]);
                let d = norm3(diff).max(0.30);
                let r_safe = 1.48 * ((rcov_angstrom(heavy_syms[y]) + 0.32) as f64);
                if d < r_safe {
                    let mag = 5.5 * (r_safe - d) / (d * d);
                    force = add3(force, scale3(diff, mag * r_eq));
                }
            }

            // Project `force` onto sphere tangent plane perpendicular to `u_k` and re-normalize!
            let f_tan = sub3(force, scale3(u_k, dot3(force, u_k)));
            let f_nrm = norm3(f_tan);
            let max_step = 0.28;
            let eff_tan = if f_nrm * step_sz > max_step {
                scale3(f_tan, max_step / f_nrm.max(1e-8))
            } else {
                scale3(f_tan, step_sz)
            };
            new_dirs.push(normalize3(add3(u_k, eff_tan)));
        }

        for k in 0..n_h {
            h_states[k].2 = new_dirs[k];
        }
    }

    h_states
        .into_iter()
        .map(|(p, r, u, _)| (p, add3(heavy_pos[p], scale3(u, r))))
        .collect()
}

fn initial_vsepr_dirs(
    i: usize,
    h_cnt: usize,
    max_bo: i32,
    heavy_pos: &[[f64; 3]],
    adj: &[Vec<usize>],
    v_nbs: &[[f64; 3]],
) -> Vec<[f64; 3]> {
    let deg = v_nbs.len();
    if deg == 0 {
        let tet = [
            normalize3([1.0, 1.0, 1.0]),
            normalize3([1.0, -1.0, -1.0]),
            normalize3([-1.0, 1.0, -1.0]),
            normalize3([-1.0, -1.0, 1.0]),
        ];
        return tet[..h_cnt.min(4)].to_vec();
    }

    if deg == 1 {
        let a = scale3(v_nbs[0], -1.0); // Points directly away from neighbor `j0`
        // Reference vector from `j0`'s second neighbor if available (for staggered / coplanar pi alignment!)
        let j0 = adj[i][0];
        let mut ref_v = [0.0, 0.0, 1.0];
        for &w in &adj[j0] {
            if w != i {
                let d_jw = sub3(heavy_pos[j0], heavy_pos[w]);
                let proj = sub3(d_jw, scale3(a, dot3(d_jw, a)));
                if norm3(proj) > 1e-4 {
                    ref_v = normalize3(proj);
                    break;
                }
            }
        }
        let (e1, e2) = ortho_basis_with_hint(a, ref_v);
        match h_cnt {
            1 => {
                if max_bo == 3 {
                    return vec![a];
                } else if max_bo == 2 {
                    return vec![normalize3(add3(scale3(a, 0.5), scale3(e1, 0.866_025_403_78)))];
                } else {
                    return vec![normalize3(add3(scale3(a, 1.0 / 3.0), scale3(e1, 0.942_809_041_58)))];
                }
            }
            2 => {
                if max_bo >= 2 {
                    let s60 = 0.866_025_403_78;
                    return vec![
                        normalize3(add3(scale3(a, 0.5), scale3(e1, s60))),
                        normalize3(sub3(scale3(a, 0.5), scale3(e1, s60))),
                    ];
                } else {
                    // Exact regular tetrahedral pair (`109.47°` to `-a` and `109.47°` to each other)
                    let base = add3(scale3(a, 1.0 / 3.0), scale3(e2, (2.0f64).sqrt() / 3.0));
                    let lat = scale3(e1, (2.0f64 / 3.0).sqrt());
                    return vec![normalize3(add3(base, lat)), normalize3(sub3(base, lat))];
                }
            }
            3 => {
                let c_rad = 2.0 * (2.0f64).sqrt() / 3.0;
                let mut out = Vec::with_capacity(3);
                for m in 0..3 {
                    let ang = (m as f64) * 2.0 * std::f64::consts::PI / 3.0;
                    let radial = add3(scale3(e1, ang.cos()), scale3(e2, ang.sin()));
                    out.push(normalize3(add3(scale3(a, 1.0 / 3.0), scale3(radial, c_rad))));
                }
                return out;
            }
            _ => {}
        }
    }

    if deg == 2 {
        let sum_v = add3(v_nbs[0], v_nbs[1]);
        let b = if norm3(sum_v) > 1e-3 {
            normalize3(scale3(sum_v, -1.0))
        } else {
            ortho_basis_with_hint(v_nbs[0], [0.0, 1.0, 0.0]).0
        };
        let cr = cross3(v_nbs[0], v_nbs[1]);
        let n_perp = if norm3(cr) > 1e-4 {
            normalize3(cr)
        } else {
            ortho_basis_with_hint(b, [0.0, 0.0, 1.0]).0
        };
        if h_cnt == 1 {
            return vec![b];
        } else if h_cnt == 2 {
            let ca = (1.0f64 / 3.0).sqrt();
            let cp = (2.0f64 / 3.0).sqrt();
            return vec![
                normalize3(add3(scale3(b, ca), scale3(n_perp, cp))),
                normalize3(sub3(scale3(b, ca), scale3(n_perp, cp))),
            ];
        }
    }

    // General open-hemisphere maximizer for `deg >= 3` (e.g. methine `R3CH`, bridgeheads, quaternary `NH+`)
    let mut placed: Vec<[f64; 3]> = Vec::with_capacity(h_cnt);
    for _ in 0..h_cnt {
        let mut cands: Vec<[f64; 3]> = Vec::with_capacity(32);
        let mut neg_sum = [0.0f64; 3];
        for &v in v_nbs {
            neg_sum = sub3(neg_sum, v);
        }
        for &p in &placed {
            neg_sum = sub3(neg_sum, p);
        }
        if norm3(neg_sum) > 1e-3 {
            cands.push(normalize3(neg_sum));
        }
        if v_nbs.len() >= 3 {
            let n_pl = cross3(sub3(v_nbs[1], v_nbs[0]), sub3(v_nbs[2], v_nbs[0]));
            if norm3(n_pl) > 1e-5 {
                let u_pl = normalize3(n_pl);
                cands.push(u_pl);
                cands.push(scale3(u_pl, -1.0));
            }
        }
        for ix in -1..=1 {
            for iy in -1..=1 {
                for iz in -1..=1 {
                    if ix != 0 || iy != 0 || iz != 0 {
                        cands.push(normalize3([ix as f64, iy as f64, iz as f64]));
                    }
                }
            }
        }
        let mut best_u = cands[0];
        let mut best_min_sep = f64::NEG_INFINITY;
        for c in cands {
            let mut min_sep = f64::INFINITY;
            for &v in v_nbs {
                min_sep = min_sep.min(-dot3(c, v));
            }
            for &p in &placed {
                min_sep = min_sep.min(-dot3(c, p));
            }
            if min_sep > best_min_sep {
                best_min_sep = min_sep;
                best_u = c;
            }
        }
        placed.push(best_u);
    }
    placed
}

fn connected_components(n: usize, adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut vis = vec![false; n];
    let mut comps = Vec::new();
    let mut q = VecDeque::new();
    for s in 0..n {
        if vis[s] {
            continue;
        }
        vis[s] = true;
        q.push_back(s);
        let mut comp = Vec::new();
        while let Some(u) = q.pop_front() {
            comp.push(u);
            for &v in &adj[u] {
                if !vis[v] {
                    vis[v] = true;
                    q.push_back(v);
                }
            }
        }
        comps.push(comp);
    }
    comps
}

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

#[inline]
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm3(sub3(a, b))
}

#[inline]
fn normalize3(a: [f64; 3]) -> [f64; 3] {
    let n = norm3(a);
    if n > 1e-9 {
        scale3(a, 1.0 / n)
    } else {
        [1.0, 0.0, 0.0]
    }
}

fn ortho_basis_with_hint(axis: [f64; 3], hint: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let proj = sub3(hint, scale3(axis, dot3(hint, axis)));
    let e1 = if norm3(proj) > 1e-4 {
        normalize3(proj)
    } else {
        let fb = if axis[0].abs() < 0.85 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        normalize3(sub3(fb, scale3(axis, dot3(fb, axis))))
    };
    let e2 = normalize3(cross3(axis, e1));
    (e1, e2)
}
