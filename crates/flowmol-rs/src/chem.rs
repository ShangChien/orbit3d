use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalNeighbor {
    pub symbol: String,
    pub bond_type: String,
    pub distance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalCenter {
    pub atom_idx: usize,
    pub symbol: String,
    pub formal_charge: i32,
    pub coordination_number: usize,
    pub neighbors: Vec<MetalNeighbor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampledBond {
    pub u: usize,
    pub v: usize,
    pub bond_type_id: usize,
    pub bond_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampledMolecule {
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
    pub metals: Vec<MetalCenter>,
}

const NON_METAL_Z: &[usize] = &[
    1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 33, 34, 35, 36, 52, 53, 54, 85, 86, 117, 118,
];

pub fn is_metal_z(z: usize) -> bool {
    !NON_METAL_Z.contains(&z)
}

/// Orient a `COORDINATE` (`DATIVE`) bond as `(donor_idx, acceptor_idx)` matching `constants._orient_coordinate_pair`.
pub fn orient_coordinate_pair(u: usize, v: usize, z_u: usize, z_v: usize) -> (usize, usize) {
    let m_u = is_metal_z(z_u);
    let m_v = is_metal_z(z_v);
    if m_v && !m_u {
        return (u, v);
    }
    if m_u && !m_v {
        return (v, u);
    }
    let lewis = [5usize, 14, 32, 52];
    let la_u = lewis.contains(&z_u);
    let la_v = lewis.contains(&z_v);
    if la_v && !la_u {
        return (u, v);
    }
    if la_u && !la_v {
        return (v, u);
    }
    if z_u >= z_v {
        (u, v)
    } else {
        (v, u)
    }
}

/// Allowed neutral + charged main-group valence shell capacities (in units of half-bonds, so aromatic = 3, single = 2, double = 4, triple = 6).
fn check_main_group_valence(z: usize, charge: i32, half_bond_sum: usize, has_aromatic: bool) -> bool {
    // Aromatic bonds (1.5 each -> 3 half-bonds): e.g. benzene C has two arom (6) + one single (2) = 8 half-bonds (= 4.0)
    // Pyridine N has two arom (6) = 6 half-bonds (= 3.0); Pyrrole N has two arom (6) + one H (2) = 8? No, pyrrole C-N in RDKit is 1.5+1.5+1 = 4.0? Actually in pyrrole two C-N aromatic + N-H single => half_bond_sum = 8 (or 7 at ring junction)!
    // So for aromatic atoms, allow +-1 half-bond rounding around standard valences!
    let check_val = |allowed_valences: &[i32]| -> bool {
        for &v in allowed_valences {
            if v < 0 {
                continue;
            }
            let target = (v as usize) * 2;
            if half_bond_sum == target || (has_aromatic && half_bond_sum.abs_diff(target) <= 1) {
                return true;
            }
        }
        false
    };

    match z {
        1 => check_val(&[1 - charge.abs()]),                    // H: 1 (or H+/H- = 0)
        2 | 10 | 18 | 36 | 54 | 86 => half_bond_sum == 0,       // Noble gases
        5 => check_val(&[3 - charge]),                          // B: 3 (BH4- -> 4)
        6 => check_val(&[4 - charge.abs()]),                    // C: 4 (C+/C- -> 3)
        7 => check_val(&[3 + charge, 5]),                       // N: 3, N+: 4, N-: 2
        8 => check_val(&[2 + charge]),                          // O: 2, O+: 3, O-: 1
        9 => check_val(&[1 + charge]),                          // F: 1
        14 => check_val(&[4 - charge, 6]),                      // Si: 4 (or hypervalent 5/6)
        15 => check_val(&[3 + charge, 5, 6]),                   // P: 3, 4(P+), 5, 6(PF6-)
        16 => check_val(&[2 + charge, 4, 6]),                   // S: 2, 4, 6, S-: 1, S+: 3
        17 => check_val(&[1 + charge, 3, 5, 7]),                // Cl
        33 => check_val(&[3 + charge, 5]),                      // As
        34 => check_val(&[2 + charge, 4, 6]),                   // Se
        35 => check_val(&[1 + charge, 3, 5]),                   // Br
        52 => check_val(&[2 + charge, 4, 6]),                   // Te
        53 => check_val(&[1 + charge, 3, 5, 7]),                // I
        _ => half_bond_sum <= 32,
    }
}

pub fn inspect_sampled_molecule(
    atom_types_vocab: &[String],
    charge_vocab: &[i32],
    positions: &[[f32; 3]],
    a_idx: &[usize],
    c_idx: &[usize],
    e_upper: &[usize],
) -> SampledMolecule {
    let n = positions.len();
    let mut symbols = Vec::with_capacity(n);
    let mut z_vec = Vec::with_capacity(n);
    let mut charges = Vec::with_capacity(n);
    let mut elem_counts: BTreeMap<String, usize> = BTreeMap::new();

    for i in 0..n {
        let ai = a_idx[i];
        let sym = atom_types_vocab.get(ai).cloned().unwrap_or_else(|| "X".to_string());
        let z = if ai < atom_types_vocab.len() { ai + 1 } else { 0 };
        let chg = charge_vocab.get(c_idx[i]).copied().unwrap_or(0);
        *elem_counts.entry(sym.clone()).or_insert(0) += 1;
        symbols.push(sym);
        z_vec.push(z);
        charges.push(chg);
    }

    // Hill system molecular formula (C first, H second, then alphabetical + net charge)
    let mut formula = String::new();
    let append_el = |f: &mut String, el: &str, cnt: usize| {
        f.push_str(el);
        if cnt > 1 {
            f.push_str(&cnt.to_string());
        }
    };
    if let Some(&c_cnt) = elem_counts.get("C") {
        append_el(&mut formula, "C", c_cnt);
        if let Some(&h_cnt) = elem_counts.get("H") {
            append_el(&mut formula, "H", h_cnt);
        }
        for (el, &cnt) in &elem_counts {
            if el != "C" && el != "H" {
                append_el(&mut formula, el, cnt);
            }
        }
    } else {
        for (el, &cnt) in &elem_counts {
            append_el(&mut formula, el, cnt);
        }
    }
    let net_q: i32 = charges.iter().sum();
    if net_q > 0 {
        if net_q > 1 {
            formula.push_str(&format!("+{net_q}"));
        } else {
            formula.push('+');
        }
    } else if net_q < 0 {
        if net_q < -1 {
            formula.push_str(&format!("{net_q}"));
        } else {
            formula.push('-');
        }
    }

    let mut min_dist = f32::INFINITY;
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

    let bond_names = ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "DATIVE"];
    let mut bonds = Vec::new();
    let mut adj = vec![Vec::new(); n];
    let mut half_bonds_non_metal = vec![0usize; n];
    let mut has_arom = vec![false; n];
    let mut total_deg = vec![0usize; n];
    let mut has_coord = false;

    let mut k = 0;
    for u in 0..n {
        for v in (u + 1)..n {
            let bt = e_upper[k];
            k += 1;
            if bt == 0 || bt >= 6 {
                continue;
            }
            let (bu, bv) = if bt == 5 {
                has_coord = true;
                orient_coordinate_pair(u, v, z_vec[u], z_vec[v])
            } else {
                (u, v)
            };
            bonds.push(SampledBond {
                u: bu,
                v: bv,
                bond_type_id: bt,
                bond_type: bond_names[bt].to_string(),
            });
            adj[u].push((v, bt));
            adj[v].push((u, bt));
            total_deg[u] += 1;
            total_deg[v] += 1;

            // For ligand-subgraph valence check (matching constants._sanitize_organometallic_fallback:
            // detach metal coordination bonds before checking main-group ligand valence!)
            if !is_metal_z(z_vec[u]) && !is_metal_z(z_vec[v]) && bt != 5 {
                let hb = match bt {
                    1 => 2,
                    2 => 4,
                    3 => 6,
                    4 => {
                        has_arom[u] = true;
                        has_arom[v] = true;
                        3
                    }
                    _ => 0,
                };
                half_bonds_non_metal[u] += hb;
                half_bonds_non_metal[v] += hb;
            }
        }
    }

    // Connected components BFS
    let mut visited = vec![false; n];
    let mut n_frags = 0;
    let mut q = VecDeque::new();
    for i in 0..n {
        if !visited[i] {
            n_frags += 1;
            visited[i] = true;
            q.push_back(i);
            while let Some(cur) = q.pop_front() {
                for &(nb, _) in &adj[cur] {
                    if !visited[nb] {
                        visited[nb] = true;
                        q.push_back(nb);
                    }
                }
            }
        }
    }

    let mut valence_ok = n >= 2 && min_dist.is_finite() && !z_vec.contains(&0);
    let mut metals = Vec::new();
    for i in 0..n {
        if is_metal_z(z_vec[i]) {
            if total_deg[i] > 16 {
                valence_ok = false;
            }
            let mut nbrs = Vec::new();
            for &(nb, bt) in &adj[i] {
                let dx = positions[i][0] - positions[nb][0];
                let dy = positions[i][1] - positions[nb][1];
                let dz = positions[i][2] - positions[nb][2];
                nbrs.push(MetalNeighbor {
                    symbol: symbols[nb].clone(),
                    bond_type: bond_names[bt].to_string(),
                    distance: ((dx * dx + dy * dy + dz * dz).sqrt() * 100.0).round() / 100.0,
                });
            }
            metals.push(MetalCenter {
                atom_idx: i,
                symbol: symbols[i].clone(),
                formal_charge: charges[i],
                coordination_number: nbrs.len(),
                neighbors: nbrs,
            });
        } else if !check_main_group_valence(z_vec[i], charges[i], half_bonds_non_metal[i], has_arom[i]) {
            valence_ok = false;
        }
    }

    let n_heavy = symbols.iter().filter(|s| s.as_str() != "H").count();
    let elements: Vec<String> = elem_counts.keys().cloned().collect();

    SampledMolecule {
        n_atoms: n,
        n_heavy,
        formula,
        atom_symbols: symbols,
        atomic_numbers: z_vec,
        formal_charges: charges,
        positions: positions.to_vec(),
        bonds,
        n_fragments: n_frags,
        connected: n_frags == 1,
        valence_valid: valence_ok,
        has_metal: !metals.is_empty(),
        has_coordinate_bond: has_coord,
        min_dist,
        elements,
        metals,
    }
}

impl SampledMolecule {
    /// Export to standard MDL V2000 `.sdf` format (with `M  CHG` lines; coordinate bonds encoded as type `8` = Any/Dative in V2000).
    pub fn to_sdf(&self, title: &str) -> String {
        let mut s = String::new();
        s.push_str(&format!("{title} | {}\n", self.formula));
        s.push_str("  flowmol           3D\n\n");
        s.push_str(&format!("{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000\n", self.n_atoms, self.bonds.len()));
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
                5 => 8, // V2000 query/coordination bond
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
            s.push_str(&format!("{:<3} {:>12.6} {:>12.6} {:>12.6}\n", self.atom_symbols[i], p[0], p[1], p[2]));
        }
        s
    }
}
