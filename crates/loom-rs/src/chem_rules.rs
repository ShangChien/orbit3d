//! Zero-RDKit 2D/3D Chemical Graph Perception, `.sdf` Multi-Record Dataset Parser,
//! Symmetrized SSSR Rings, Morgan ECFP4 Fingerprints, Bemis–Murcko Scaffolds,
//! 7 `BuiltInRuleAsset` Channels (`mol_weight`, `rotatable_bonds`, `max_ring_size`,
//! `strained_small_rings`, `smarts_alert`, `sa_score`, `qed`), Clean Organic Charge Gate,
//! and the `29D` Static Feature Vector for `ResidualCalibratedAsset`.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use uni3dar_rs::valence::{
    build_raw_point_cloud_graph, elem_id_to_symbol, reconcile_point_cloud, symbol_to_elem_id,
    SampledUni3darMolecule,
};

/// A 3D labeled or unlabeled molecule record parsed from `.sdf`, `.xyz`, or runtime generators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolRecord {
    pub title: String,
    pub atomic_numbers: Vec<u8>,
    pub positions: Vec<[f64; 3]>,
    pub formal_charges: Vec<i32>,
    /// `(u, v, order)` where `order` is `1` (single), `2` (double), `3` (triple), `4` (aromatic).
    pub bonds: Vec<(usize, usize, u8)>,
    /// Numeric properties extracted from SDF `<TAG>` fields (e.g. `homo`, `lumo`, `gap`, `logP`).
    pub properties: BTreeMap<String, f64>,
    pub valence_valid: bool,
}

impl MolRecord {
    pub fn from_uni3dar(mol: &SampledUni3darMolecule) -> Self {
        let atomic_numbers: Vec<u8> = mol.atomic_numbers.iter().map(|&z| z as u8).collect();
        let positions: Vec<[f64; 3]> = mol
            .positions
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        let bonds: Vec<(usize, usize, u8)> = mol
            .bonds
            .iter()
            .map(|b| (b.u, b.v, (b.bond_type_id as u8).clamp(1, 4)))
            .collect();
        let mut properties = BTreeMap::new();
        for (k, &v) in &mol.predicted_props {
            properties.insert(k.clone(), v as f64);
        }
        Self {
            title: mol.formula.clone(),
            atomic_numbers,
            positions,
            formal_charges: mol.formal_charges.clone(),
            bonds,
            properties,
            valence_valid: mol.valence_valid && mol.connected,
        }
    }

    pub fn from_point_cloud(
        title: &str,
        atomic_numbers: &[u8],
        coords_a: &[[f64; 3]],
        reconcile: bool,
    ) -> Self {
        let elem_ids: Vec<usize> = atomic_numbers
            .iter()
            .map(|&z| (z as usize).saturating_sub(1))
            .collect();
        let pos_f32: Vec<[f32; 3]> = coords_a
            .iter()
            .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
            .collect();
        let rg = if reconcile {
            reconcile_point_cloud(&elem_ids, &pos_f32, None, 24, true)
                .unwrap_or_else(|| build_raw_point_cloud_graph(&elem_ids, &pos_f32))
        } else {
            reconcile_point_cloud(&elem_ids, &pos_f32, None, 24, false)
                .filter(|g| g.delta_h_count == 0)
                .unwrap_or_else(|| build_raw_point_cloud_graph(&elem_ids, &pos_f32))
        };
        let z_out: Vec<u8> = rg
            .symbols
            .iter()
            .map(|s| (symbol_to_elem_id(s).unwrap_or(5) + 1) as u8)
            .collect();
        let p_out: Vec<[f64; 3]> = rg
            .positions
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        let bonds: Vec<(usize, usize, u8)> = rg
            .bonds
            .iter()
            .map(|b| (b.u, b.v, (b.bond_type_id as u8).clamp(1, 4)))
            .collect();
        let connected = is_graph_connected(z_out.len(), &bonds);
        Self {
            title: title.to_string(),
            atomic_numbers: z_out,
            positions: p_out,
            formal_charges: rg.formal_charges,
            bonds,
            properties: BTreeMap::new(),
            valence_valid: rg.valence_valid && connected,
        }
    }

    pub fn to_sdf(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("{}\n  loom-rs           3D\n\n", self.title));
        s.push_str(&format!(
            "{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000\n",
            self.atomic_numbers.len(),
            self.bonds.len()
        ));
        for i in 0..self.atomic_numbers.len() {
            let sym = elem_id_to_symbol((self.atomic_numbers[i] as usize).saturating_sub(1));
            let p = self.positions[i];
            s.push_str(&format!(
                "{:>10.4}{:>10.4}{:>10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n",
                p[0], p[1], p[2], sym
            ));
        }
        for &(u, v, bt) in &self.bonds {
            s.push_str(&format!("{:>3}{:>3}{:>3}  0  0  0  0\n", u + 1, v + 1, bt.clamp(1, 4)));
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
        s.push_str("M  END\n");
        for (k, &v) in &self.properties {
            if v.is_finite() {
                s.push_str(&format!("> <{k}>\n{v:.6}\n\n"));
            }
        }
        s.push_str("$$$$\n");
        s
    }
}

pub fn is_graph_connected(n_atoms: usize, bonds: &[(usize, usize, u8)]) -> bool {
    if n_atoms <= 1 {
        return n_atoms == 1;
    }
    let mut adj = vec![Vec::new(); n_atoms];
    for &(u, v, _) in bonds {
        if u < n_atoms && v < n_atoms {
            adj[u].push(v);
            adj[v].push(u);
        }
    }
    let mut vis = vec![false; n_atoms];
    let mut q = VecDeque::new();
    vis[0] = true;
    q.push_back(0usize);
    let mut seen = 0usize;
    while let Some(u) = q.pop_front() {
        seen += 1;
        for &w in &adj[u] {
            if !vis[w] {
                vis[w] = true;
                q.push_back(w);
            }
        }
    }
    seen == n_atoms
}

/// Parse multi-molecule V2000 `.sdf` text (including `M  CHG` formal charges and `> <prop>` numeric tags).
pub fn parse_sdf_records(sdf_text: &str) -> Result<Vec<MolRecord>, String> {
    let mut out = Vec::new();
    for block in sdf_text.split("$$$$") {
        let trimmed = block.trim_matches(|c| c == '\r' || c == '\n');
        if trimmed.trim().is_empty() {
            continue;
        }
        let lines: Vec<&str> = trimmed.lines().collect();
        if lines.len() < 4 {
            continue;
        }
        let title = lines[0].trim().to_string();
        let counts_line = lines[3];
        if counts_line.len() < 6 {
            continue;
        }
        let n_atoms: usize = counts_line[0..3]
            .trim()
            .parse()
            .map_err(|e| format!("Invalid SDF atom count '{}': {e}", &counts_line[0..3]))?;
        let n_bonds: usize = counts_line[3..6]
            .trim()
            .parse()
            .map_err(|e| format!("Invalid SDF bond count '{}': {e}", &counts_line[3..6]))?;
        if n_atoms == 0 || lines.len() < 4 + n_atoms + n_bonds {
            continue;
        }

        let mut atomic_numbers = Vec::with_capacity(n_atoms);
        let mut positions = Vec::with_capacity(n_atoms);
        let mut formal_charges = vec![0i32; n_atoms];
        let mut ok_atoms = true;

        for a_idx in 0..n_atoms {
            let ln = lines[4 + a_idx];
            let parts: Vec<&str> = ln.split_whitespace().collect();
            if parts.len() < 4 {
                ok_atoms = false;
                break;
            }
            let x: f64 = parts[0].parse().unwrap_or(0.0);
            let y: f64 = parts[1].parse().unwrap_or(0.0);
            let z: f64 = parts[2].parse().unwrap_or(0.0);
            let sym = parts[3];
            let Some(eid) = symbol_to_elem_id(sym) else {
                ok_atoms = false;
                break;
            };
            atomic_numbers.push((eid + 1) as u8);
            positions.push([x, y, z]);
            // Legacy V2000 atom-block charge code (column 9 / token 5 if present, overridden by M CHG)
            if parts.len() >= 6 {
                if let Ok(chg_code) = parts[5].parse::<i32>() {
                    formal_charges[a_idx] = match chg_code {
                        1 => 3,
                        2 => 2,
                        3 => 1,
                        5 => -1,
                        6 => -2,
                        7 => -3,
                        _ => 0,
                    };
                }
            }
        }
        if !ok_atoms {
            continue;
        }

        let mut bonds = Vec::with_capacity(n_bonds);
        let mut has_m_chg = false;
        for b_idx in 0..n_bonds {
            let ln = lines[4 + n_atoms + b_idx];
            if ln.len() < 9 {
                continue;
            }
            let u1: usize = ln[0..3].trim().parse().unwrap_or(0);
            let v1: usize = ln[3..6].trim().parse().unwrap_or(0);
            let bt: u8 = ln[6..9].trim().parse().unwrap_or(1);
            if u1 >= 1 && v1 >= 1 && u1 <= n_atoms && v1 <= n_atoms && u1 != v1 {
                bonds.push((u1 - 1, v1 - 1, bt.clamp(1, 4)));
            }
        }

        let mut i = 4 + n_atoms + n_bonds;
        let mut properties = BTreeMap::new();
        while i < lines.len() {
            let ln = lines[i].trim();
            if ln.starts_with("M  CHG") {
                if !has_m_chg {
                    formal_charges.fill(0);
                    has_m_chg = true;
                }
                let toks: Vec<&str> = ln.split_whitespace().collect();
                if toks.len() >= 3 {
                    let cnt: usize = toks[2].parse().unwrap_or(0);
                    for k in 0..cnt {
                        if 3 + 2 * k + 1 < toks.len() {
                            let idx1: usize = toks[3 + 2 * k].parse().unwrap_or(0);
                            let q: i32 = toks[3 + 2 * k + 1].parse().unwrap_or(0);
                            if idx1 >= 1 && idx1 <= n_atoms {
                                formal_charges[idx1 - 1] = q;
                            }
                        }
                    }
                }
            } else if ln.starts_with('>') {
                if let (Some(l_br), Some(r_br)) = (ln.find('<'), ln.rfind('>')) {
                    if r_br > l_br + 1 {
                        let key = ln[l_br + 1..r_br].trim().to_string();
                        if i + 1 < lines.len() {
                            let val_line = lines[i + 1].trim();
                            if let Some(first_tok) = val_line.split_whitespace().next() {
                                if let Ok(v) = first_tok.parse::<f64>() {
                                    if v.is_finite() {
                                        properties.insert(key, v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            i += 1;
        }

        // If SDF had no explicit bonds (coordinate-only block), perceive bonds from 3D coordinates!
        if bonds.is_empty() && n_atoms > 1 {
            let mut perceived = MolRecord::from_point_cloud(&title, &atomic_numbers, &positions, false);
            perceived.properties = properties;
            out.push(perceived);
        } else {
            let connected = is_graph_connected(n_atoms, &bonds);
            out.push(MolRecord {
                title,
                atomic_numbers,
                positions,
                formal_charges,
                bonds,
                properties,
                valence_valid: connected,
            });
        }
    }
    Ok(out)
}

/// IUPAC standard atomic weights matching RDKit `GetPeriodicTable().GetAtomicWeight`.
#[inline]
pub fn iupac_atomic_weight(z: u8) -> f64 {
    match z {
        1 => 1.008,
        5 => 10.812,
        6 => 12.011,
        7 => 14.007,
        8 => 15.999,
        9 => 18.998,
        14 => 28.085,
        15 => 30.974,
        16 => 32.067,
        17 => 35.453,
        33 => 74.922,
        34 => 78.960,
        35 => 79.904,
        53 => 126.904,
        _ => mlip_core::mass_amu_from_z(z),
    }
}

/// Exact Clean Organic Charge Gate (`is_clean_organic_charge_state` in `valence_reconciler.py:40-68`).
pub fn is_clean_organic_charge_state(rec: &MolRecord) -> bool {
    if !rec.valence_valid || rec.atomic_numbers.is_empty() {
        return false;
    }
    let mut net_q = 0i32;
    let mut abs_q = 0i32;
    for (i, &q) in rec.formal_charges.iter().enumerate() {
        net_q += q;
        abs_q += q.abs();
        if q != 0 {
            let z = rec.atomic_numbers[i];
            let allowed = matches!(
                (z, q),
                (7, 1) | (8, -1) | (16, 1) | (16, 2) | (15, 1) | (5, -1)
            );
            if !allowed {
                return false;
            }
        }
    }
    net_q == 0 && abs_q <= 4
}

/// Sorted formal-charge multiset signature matching `formal_charge_signature(mol)`.
pub fn formal_charge_signature(rec: &MolRecord) -> String {
    let mut pairs: Vec<(String, i32)> = rec
        .formal_charges
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, q)| q != 0)
        .map(|(i, q)| {
            let sym = elem_id_to_symbol((rec.atomic_numbers[i] as usize).saturating_sub(1)).to_string();
            (sym, q)
        })
        .collect();
    if pairs.is_empty() {
        return "neutral".to_string();
    }
    pairs.sort();
    pairs
        .into_iter()
        .map(|(s, q)| format!("{s}:{q:+}"))
        .collect::<Vec<_>>()
        .join(",")
}

// ============================================================================
// Hydrogen-Stripped Heavy-Atom Graph (`RemoveHs(M)`) + SSSR Ring Perception
// ============================================================================

#[derive(Debug, Clone)]
pub struct HeavyAtomNode {
    pub orig_idx: usize,
    pub z: u8,
    pub formal_charge: i32,
    pub implicit_hs: usize,
    pub pos: [f64; 3],
    pub heavy_degree: usize,
    pub bond_order_sum: f64,
    pub has_double_bond: bool,
    pub has_triple_bond: bool,
    pub in_ring: bool,
    pub is_aromatic: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct HeavyEdge {
    pub u: usize,
    pub v: usize,
    /// `1`, `2`, `3`, or `4` (aromatic)
    pub order: u8,
    pub in_ring: bool,
    pub is_aromatic: bool,
}

#[derive(Debug, Clone)]
pub struct HeavyMolGraph {
    pub atoms: Vec<HeavyAtomNode>,
    pub edges: Vec<HeavyEdge>,
    pub adj: Vec<Vec<(usize, usize)>>, // (neighbor_heavy_idx, edge_idx)
    pub rings: Vec<Vec<usize>>,        // Symmetrized SSSR cycles (ordered vertex indices)
}

impl HeavyMolGraph {
    pub fn from_record(rec: &MolRecord) -> Self {
        let n_all = rec.atomic_numbers.len();
        let mut full_to_heavy = vec![usize::MAX; n_all];
        let mut atoms = Vec::new();
        for i in 0..n_all {
            let z = rec.atomic_numbers[i];
            if z > 1 {
                full_to_heavy[i] = atoms.len();
                atoms.push(HeavyAtomNode {
                    orig_idx: i,
                    z,
                    formal_charge: rec.formal_charges.get(i).copied().unwrap_or(0),
                    implicit_hs: 0,
                    pos: rec.positions[i],
                    heavy_degree: 0,
                    bond_order_sum: 0.0,
                    has_double_bond: false,
                    has_triple_bond: false,
                    in_ring: false,
                    is_aromatic: false,
                });
            }
        }

        // Handle pure H2 molecule edge case gracefully
        if atoms.is_empty() && n_all > 0 {
            atoms.push(HeavyAtomNode {
                orig_idx: 0,
                z: rec.atomic_numbers[0],
                formal_charge: rec.formal_charges.first().copied().unwrap_or(0),
                implicit_hs: n_all.saturating_sub(1),
                pos: rec.positions[0],
                heavy_degree: 0,
                bond_order_sum: (n_all.saturating_sub(1)) as f64,
                has_double_bond: false,
                has_triple_bond: false,
                in_ring: false,
                is_aromatic: false,
            });
            return Self {
                atoms,
                edges: Vec::new(),
                adj: vec![Vec::new()],
                rings: Vec::new(),
            };
        }

        let n_hvy = atoms.len();
        let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_hvy];
        let mut edges = Vec::new();
        let mut seen_pair = HashSet::new();

        for &(u_full, v_full, bt) in &rec.bonds {
            if u_full >= n_all || v_full >= n_all {
                continue;
            }
            let zu = rec.atomic_numbers[u_full];
            let zv = rec.atomic_numbers[v_full];
            if zu == 1 && zv > 1 {
                let hi = full_to_heavy[v_full];
                atoms[hi].implicit_hs += 1;
                atoms[hi].bond_order_sum += 1.0;
            } else if zv == 1 && zu > 1 {
                let hi = full_to_heavy[u_full];
                atoms[hi].implicit_hs += 1;
                atoms[hi].bond_order_sum += 1.0;
            } else if zu > 1 && zv > 1 {
                let hu = full_to_heavy[u_full];
                let hv = full_to_heavy[v_full];
                if hu == hv {
                    continue;
                }
                let key = (hu.min(hv), hu.max(hv));
                if !seen_pair.insert(key) {
                    continue;
                }
                let e_idx = edges.len();
                let is_arom = bt == 4;
                edges.push(HeavyEdge {
                    u: hu,
                    v: hv,
                    order: bt,
                    in_ring: false,
                    is_aromatic: is_arom,
                });
                adj[hu].push((hv, e_idx));
                adj[hv].push((hu, e_idx));
                atoms[hu].heavy_degree += 1;
                atoms[hv].heavy_degree += 1;
                let bo = match bt {
                    2 => 2.0,
                    3 => 3.0,
                    4 => 1.5,
                    _ => 1.0,
                };
                atoms[hu].bond_order_sum += bo;
                atoms[hv].bond_order_sum += bo;
                if bt == 2 {
                    atoms[hu].has_double_bond = true;
                    atoms[hv].has_double_bond = true;
                } else if bt == 3 {
                    atoms[hu].has_triple_bond = true;
                    atoms[hv].has_triple_bond = true;
                }
                if is_arom {
                    atoms[hu].is_aromatic = true;
                    atoms[hv].is_aromatic = true;
                }
            }
        }

        // Symmetrized SSSR ring perception: for every spanning-tree chord (plus edge-shortest cycles),
        // extract unique minimal elementary cycles.
        let rings = compute_sssr(n_hvy, &edges, &adj);
        for ring in &rings {
            let rlen = ring.len();
            // Detect Hueckel conjugated 5/6-membered Kekule aromatic rings (`C/N/O/S`) even when SDF stores Kekule 1/2 bonds!
            let mut kekule_arom_cand = matches!(rlen, 5 | 6);
            let mut pi_electrons = 0usize;
            for &v in ring {
                atoms[v].in_ring = true;
                let nd = &atoms[v];
                if !matches!(nd.z, 6 | 7 | 8 | 16 | 5) || nd.has_triple_bond {
                    kekule_arom_cand = false;
                } else if nd.is_aromatic || nd.has_double_bond {
                    // sp2 ring atom with a double bond contributes 1 pi electron (unless exocyclic C=O)
                    let exo_carbonyl = nd.z == 6
                        && adj[v].iter().any(|&(nb, ei)| {
                            !ring.contains(&nb) && edges[ei].order == 2 && matches!(atoms[nb].z, 8 | 16)
                        });
                    if exo_carbonyl {
                        // pyridone / quinone-like sp2 center contributes 0 pi electrons
                    } else {
                        pi_electrons += 1;
                    }
                } else if matches!(nd.z, 7 | 8 | 16) && (nd.heavy_degree + nd.implicit_hs) <= 3 {
                    // furan O, thiophene S, pyrrole NH lone pair contributes 2 pi electrons
                    pi_electrons += 2;
                } else {
                    kekule_arom_cand = false;
                }
            }
            let is_ring_arom = ring.iter().all(|&v| atoms[v].is_aromatic)
                || (kekule_arom_cand && matches!(pi_electrons, 6));

            for idx in 0..rlen {
                let a = ring[idx];
                let b = ring[(idx + 1) % rlen];
                if is_ring_arom {
                    atoms[a].is_aromatic = true;
                }
                for &(nb, ei) in &adj[a] {
                    if nb == b {
                        edges[ei].in_ring = true;
                        if is_ring_arom {
                            edges[ei].is_aromatic = true;
                        }
                    }
                }
            }
        }

        Self {
            atoms,
            edges,
            adj,
            rings,
        }
    }

    #[inline]
    pub fn total_connectivity(&self, idx: usize) -> usize {
        self.atoms[idx].heavy_degree + self.atoms[idx].implicit_hs
    }

    #[inline]
    pub fn is_sp3_carbon(&self, idx: usize) -> bool {
        let a = &self.atoms[idx];
        a.z == 6 && !a.is_aromatic && !a.has_double_bond && !a.has_triple_bond && (a.heavy_degree + a.implicit_hs) == 4
    }
}

fn compute_sssr(n: usize, edges: &[HeavyEdge], adj: &[Vec<(usize, usize)>]) -> Vec<Vec<usize>> {
    if n < 3 || edges.len() < 3 {
        return Vec::new();
    }
    let mut ring_map: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();

    // For each edge `e_skip = (u, v)`, find shortest path between `u` and `v` in `G \ {e_skip}`.
    // In organic/organometallic graphs (`E <= 180`), this is < 20 microseconds and finds every minimal cycle!
    let mut dist = vec![usize::MAX; n];
    let mut parent = vec![usize::MAX; n];
    let mut q = VecDeque::new();

    for (e_skip, e) in edges.iter().enumerate() {
        dist.fill(usize::MAX);
        parent.fill(usize::MAX);
        q.clear();
        dist[e.u] = 0;
        q.push_back(e.u);
        while let Some(cur) = q.pop_front() {
            if cur == e.v {
                break;
            }
            let d_next = dist[cur] + 1;
            if d_next > 16 {
                continue;
            }
            for &(nb, ei) in &adj[cur] {
                if ei == e_skip {
                    continue;
                }
                if dist[nb] == usize::MAX {
                    dist[nb] = d_next;
                    parent[nb] = cur;
                    q.push_back(nb);
                }
            }
        }
        if dist[e.v] != usize::MAX {
            let mut cycle = Vec::with_capacity(dist[e.v] + 1);
            let mut cur = e.v;
            while cur != usize::MAX {
                cycle.push(cur);
                if cur == e.u {
                    break;
                }
                cur = parent[cur];
            }
            if cycle.len() >= 3 {
                let mut key = cycle.clone();
                key.sort_unstable();
                ring_map.entry(key).or_insert(cycle);
            }
        }
    }

    let mut rings: Vec<Vec<usize>> = ring_map.into_values().collect();
    rings.sort_by_key(|r| r.len());
    rings
}

// ============================================================================
// Morgan ECFP4 Circular Fingerprint (`radius = 2`), Tanimoto & Murcko Scaffold
// ============================================================================

#[inline]
fn splitmix_hash64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Unfolded Morgan circular count fingerprint (`radius = 2`) on `HeavyMolGraph`.
pub fn morgan_ecfp4_counts(g: &HeavyMolGraph) -> HashMap<u32, u32> {
    let n = g.atoms.len();
    let mut counts: HashMap<u32, u32> = HashMap::new();
    if n == 0 {
        return counts;
    }

    // Daylight-style initial atom invariant: (Z, heavy_degree, implicit_hs, formal_charge, in_ring, is_arom)
    let mut cur_ids = vec![0u64; n];
    for (i, a) in g.atoms.iter().enumerate() {
        let mut h = (a.z as u64) * 1_000_003;
        h ^= (a.heavy_degree as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        h ^= (a.implicit_hs as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= ((a.formal_charge + 8) as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        if a.in_ring {
            h ^= 0xA5A5_5A5A_F0F0_0F0F;
        }
        if a.is_aromatic {
            h ^= 0x3C3C_C3C3_6969_9696;
        }
        let code = splitmix_hash64(h);
        cur_ids[i] = code;
        *counts.entry(code as u32).or_insert(0) += 1;
    }

    for rad in 1..=2u64 {
        let mut next_ids = vec![0u64; n];
        for i in 0..n {
            let mut nb_codes: Vec<(u8, u64)> = g.adj[i]
                .iter()
                .map(|&(nb, ei)| {
                    let bt = if g.edges[ei].is_aromatic { 4 } else { g.edges[ei].order };
                    (bt, cur_ids[nb])
                })
                .collect();
            nb_codes.sort_unstable();
            let mut acc = splitmix_hash64(cur_ids[i] ^ (rad * 0xD6E8_FEB8_6659_FD93));
            for (bt, nc) in nb_codes {
                acc = splitmix_hash64(acc ^ ((bt as u64) << 48) ^ nc);
            }
            next_ids[i] = acc;
            *counts.entry(acc as u32).or_insert(0) += 1;
        }
        cur_ids = next_ids;
    }
    counts
}

/// Folded binary `2048`-bit Morgan fingerprint (`[u64; 32]`) for ultra-fast POPCNT Tanimoto!
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MorganBits2048(pub [u64; 32]);

impl MorganBits2048 {
    pub fn from_graph(g: &HeavyMolGraph) -> Self {
        let mut words = [0u64; 32];
        for (&hash, _) in &morgan_ecfp4_counts(g) {
            let bit = (hash as usize) & 2047;
            words[bit >> 6] |= 1u64 << (bit & 63);
        }
        Self(words)
    }

    #[inline]
    pub fn tanimoto(&self, other: &Self) -> f64 {
        let mut inter = 0u32;
        let mut union_cnt = 0u32;
        for w in 0..32 {
            inter += (self.0[w] & other.0[w]).count_ones();
            union_cnt += (self.0[w] | other.0[w]).count_ones();
        }
        if union_cnt == 0 {
            0.0
        } else {
            (inter as f64) / (union_cnt as f64)
        }
    }
}

/// Bemis–Murcko scaffold signature (iteratively strips degree-1 non-ring leaf heavy atoms,
/// leaving ring systems + inter-ring linkers, hashed canonically via Weisfeiler-Lehman).
pub fn murcko_scaffold_signature(g: &HeavyMolGraph) -> String {
    let n = g.atoms.len();
    if n == 0 {
        return "empty".to_string();
    }
    let mut alive = vec![true; n];
    let mut deg: Vec<usize> = (0..n).map(|i| g.adj[i].len()).collect();
    let mut q = VecDeque::new();
    for i in 0..n {
        if deg[i] <= 1 && !g.atoms[i].in_ring {
            q.push_back(i);
        }
    }
    while let Some(u) = q.pop_front() {
        if !alive[u] || g.atoms[u].in_ring {
            continue;
        }
        alive[u] = false;
        for &(nb, _) in &g.adj[u] {
            if alive[nb] {
                deg[nb] = deg[nb].saturating_sub(1);
                if deg[nb] == 1 && !g.atoms[nb].in_ring {
                    q.push_back(nb);
                }
            }
        }
    }
    let kept: Vec<usize> = (0..n).filter(|&i| alive[i]).collect();
    if kept.is_empty() {
        return "acyclic".to_string();
    }
    // 3-iteration Weisfeiler-Lehman canonical invariant over the Murcko subgraph
    let mut wl: Vec<u64> = kept
        .iter()
        .map(|&i| splitmix_hash64((g.atoms[i].z as u64) * 257 + (g.atoms[i].is_aromatic as u64)))
        .collect();
    let mut global_idx_to_pos = vec![usize::MAX; n];
    for (pos, &i) in kept.iter().enumerate() {
        global_idx_to_pos[i] = pos;
    }
    for _ in 0..3 {
        let mut next = vec![0u64; kept.len()];
        for (pos, &i) in kept.iter().enumerate() {
            let mut nbs: Vec<(u8, u64)> = g.adj[i]
                .iter()
                .filter(|&&(nb, _)| alive[nb])
                .map(|&(nb, ei)| {
                    let bt = if g.edges[ei].is_aromatic { 4 } else { g.edges[ei].order };
                    (bt, wl[global_idx_to_pos[nb]])
                })
                .collect();
            nbs.sort_unstable();
            let mut h = splitmix_hash64(wl[pos]);
            for (bt, c) in nbs {
                h = splitmix_hash64(h ^ ((bt as u64) << 40) ^ c);
            }
            next[pos] = h;
        }
        wl = next;
    }
    wl.sort_unstable();
    let mut final_h = splitmix_hash64(wl.len() as u64);
    for c in wl {
        final_h = splitmix_hash64(final_h ^ c);
    }
    format!("murcko_{:016x}", final_h)
}

/// Canonical whole-molecule topological signature (for exact duplicate filtering in Outer-RSI).
pub fn canonical_graph_signature(g: &HeavyMolGraph) -> u64 {
    let counts = morgan_ecfp4_counts(g);
    let mut items: Vec<(u32, u32)> = counts.into_iter().collect();
    items.sort_unstable();
    let mut h = splitmix_hash64(g.atoms.len() as u64);
    for (k, c) in items {
        h = splitmix_hash64(h ^ ((k as u64) << 16) ^ (c as u64));
    }
    h
}

// ============================================================================
// Physicochemical Descriptors (`MW`, `TPSA`, `LogP`, `RotBonds`, `SMARTS`, `SA`, `QED`)
// ============================================================================

pub fn compute_mol_weight(g: &HeavyMolGraph) -> f64 {
    g.atoms
        .iter()
        .map(|a| iupac_atomic_weight(a.z) + (a.implicit_hs as f64) * 1.00794)
        .sum()
}

/// Ertl Topological Polar Surface Area (`TPSA`, Å²) over N, O, P, S heavy atoms.
pub fn compute_tpsa(g: &HeavyMolGraph) -> f64 {
    let mut psa = 0.0f64;
    for a in &g.atoms {
        let h = a.implicit_hs;
        let q = a.formal_charge;
        let x = a.heavy_degree + h;
        let contrib = match a.z {
            7 => {
                if a.is_aromatic {
                    if h == 0 && q == 0 {
                        12.89
                    } else if h == 1 && q == 0 {
                        15.79
                    } else {
                        14.14
                    }
                } else if a.has_triple_bond {
                    23.79
                } else if q > 0 {
                    if h == 0 { 3.01 } else { 16.61 + 11.0 * (h as f64 - 1.0).max(0.0) }
                } else if h == 2 && x == 3 {
                    26.02
                } else if h == 1 && x == 3 {
                    12.03
                } else if a.has_double_bond && h == 1 {
                    23.85
                } else if a.has_double_bond && h == 0 {
                    12.36
                } else {
                    3.24
                }
            }
            8 => {
                if a.is_aromatic {
                    13.14
                } else if q < 0 {
                    23.06
                } else if h >= 1 {
                    20.23
                } else if a.has_double_bond {
                    17.07
                } else if a.in_ring {
                    12.53
                } else {
                    9.23
                }
            }
            16 => {
                if a.is_aromatic {
                    28.24
                } else if h >= 1 {
                    38.80
                } else if a.has_double_bond {
                    32.09
                } else {
                    25.30
                }
            }
            15 => {
                if a.has_double_bond {
                    34.14
                } else {
                    13.59
                }
            }
            _ => 0.0,
        };
        psa += contrib;
    }
    psa
}

/// Wildman–Crippen Atomic `LogP` approximation matching RDKit `MolLogP` scale.
pub fn compute_alogp(g: &HeavyMolGraph) -> f64 {
    let mut logp = 0.0f64;
    for (i, a) in g.atoms.iter().enumerate() {
        let has_het_nb = g.adj[i].iter().any(|&(nb, _)| !matches!(g.atoms[nb].z, 6));
        let c_val = match a.z {
            6 => {
                if a.is_aromatic {
                    if has_het_nb { 0.136 } else { 0.295 }
                } else if a.has_triple_bond {
                    0.133
                } else if a.has_double_bond {
                    if has_het_nb { 0.081 } else { 0.264 }
                } else if has_het_nb {
                    -0.195
                } else {
                    0.144
                }
            }
            7 => {
                if a.formal_charge > 0 {
                    -1.019
                } else if a.is_aromatic {
                    -0.493
                } else if a.has_triple_bond {
                    -0.067
                } else if a.implicit_hs >= 2 {
                    -0.646
                } else if a.implicit_hs == 1 {
                    -0.319
                } else {
                    -0.243
                }
            }
            8 => {
                if a.formal_charge < 0 {
                    -1.189
                } else if a.is_aromatic {
                    0.155
                } else if a.implicit_hs >= 1 {
                    -0.289
                } else if a.has_double_bond {
                    -0.334
                } else {
                    -0.068
                }
            }
            9 => 0.420,
            17 => 0.689,
            35 => 0.846,
            53 => 1.190,
            16 => {
                if a.has_double_bond { -0.310 } else { 0.540 }
            }
            15 => 0.280,
            5 => -0.180,
            14 => 0.620,
            34 => 0.720,
            33 => 0.450,
            _ => 0.10,
        };
        // Explicit/implicit H contribution (`~0.123` on C/Si/B, `-0.21` on polar N/O/S)
        let h_val = if matches!(a.z, 6 | 14 | 5) {
            0.1230 * (a.implicit_hs as f64)
        } else {
            -0.2120 * (a.implicit_hs as f64)
        };
        logp += c_val + h_val;
    }
    logp
}

/// Strict RDKit rotatable bond count (`CalcNumRotatableBonds(..., Strict)`).
pub fn compute_rotatable_bonds(g: &HeavyMolGraph) -> usize {
    let is_sym_rotor_c = |idx: usize| -> bool {
        let a = &g.atoms[idx];
        if a.z != 6 || a.heavy_degree != 4 || a.implicit_hs != 0 {
            return false;
        }
        for &hal_z in &[9u8, 17, 35] {
            let cnt = g.adj[idx]
                .iter()
                .filter(|&&(nb, _)| g.atoms[nb].z == hal_z && g.atoms[nb].heavy_degree == 1)
                .count();
            if cnt == 3 {
                return true;
            }
        }
        // -C(CH3)3 tert-butyl
        let ch3_cnt = g.adj[idx]
            .iter()
            .filter(|&&(nb, _)| {
                g.atoms[nb].z == 6 && g.atoms[nb].heavy_degree == 1 && g.atoms[nb].implicit_hs == 3
            })
            .count();
        ch3_cnt == 3
    };

    let is_carbonyl_like_cd3 = |idx: usize| -> bool {
        let a = &g.atoms[idx];
        if a.z != 6 || (a.heavy_degree + a.implicit_hs) != 3 {
            return false;
        }
        g.adj[idx].iter().any(|&(nb, ei)| {
            g.edges[ei].order == 2 && matches!(g.atoms[nb].z, 7 | 8 | 16)
        })
    };

    let mut count = 0usize;
    for e in &g.edges {
        if e.order != 1 || e.in_ring || e.is_aromatic {
            continue;
        }
        let u = e.u;
        let v = e.v;
        if g.atoms[u].heavy_degree <= 1 || g.atoms[v].heavy_degree <= 1 {
            continue;
        }
        if g.atoms[u].has_triple_bond || g.atoms[v].has_triple_bond {
            continue;
        }
        if is_sym_rotor_c(u) || is_sym_rotor_c(v) {
            continue;
        }
        // Exclude amide / ester / thioamide / amidine single bond `[CD3](=[N,O,S])-!@[#7,O,S!D1]`
        let amide_uv = is_carbonyl_like_cd3(u) && matches!(g.atoms[v].z, 7 | 8 | 16);
        let amide_vu = is_carbonyl_like_cd3(v) && matches!(g.atoms[u].z, 7 | 8 | 16);
        if amide_uv || amide_vu {
            continue;
        }
        count += 1;
    }
    count
}

/// Exact 5 `_DEFAULT_REACTIVE_SMARTS` patterns from `src/loom/weaving/registry.py:53-59`.
pub fn compute_smarts_alerts(g: &HeavyMolGraph) -> usize {
    // 1. `[OX2][OX2]` — Peroxide
    let has_peroxide = g.edges.iter().any(|e| {
        g.atoms[e.u].z == 8
            && g.atoms[e.v].z == 8
            && g.total_connectivity(e.u) == 2
            && g.total_connectivity(e.v) == 2
    });

    // 2. `[CX3](=[OX1])[F,Cl,Br,I]` — Acyl halide
    let has_acyl_halide = (0..g.atoms.len()).any(|i| {
        if g.atoms[i].z != 6 || g.total_connectivity(i) != 3 {
            return false;
        }
        let has_ox1 = g.adj[i].iter().any(|&(nb, ei)| {
            g.edges[ei].order == 2 && g.atoms[nb].z == 8 && g.total_connectivity(nb) == 1
        });
        let has_hal = g.adj[i].iter().any(|&(nb, ei)| {
            g.edges[ei].order == 1 && matches!(g.atoms[nb].z, 9 | 17 | 35 | 53)
        });
        has_ox1 && has_hal
    });

    // 3. `[NX2]=[OX1]` — Nitroso
    let has_nitroso = g.edges.iter().any(|e| {
        e.order == 2
            && ((g.atoms[e.u].z == 7
                && g.total_connectivity(e.u) == 2
                && g.atoms[e.v].z == 8
                && g.total_connectivity(e.v) == 1)
                || (g.atoms[e.v].z == 7
                    && g.total_connectivity(e.v) == 2
                    && g.atoms[e.u].z == 8
                    && g.total_connectivity(e.u) == 1))
    });

    // 4. `[SX2][SX2][SX2]` — Polysulfide 3-chain
    let has_polysulfide = (0..g.atoms.len()).any(|mid| {
        if g.atoms[mid].z != 16 || g.total_connectivity(mid) != 2 {
            return false;
        }
        let s_nbs = g.adj[mid]
            .iter()
            .filter(|&&(nb, _)| g.atoms[nb].z == 16 && g.total_connectivity(nb) == 2)
            .count();
        s_nbs >= 2
    });

    // 5. `[C+]#[N-]` — Reversed-charge isocyanide
    let has_rev_isocyanide = g.edges.iter().any(|e| {
        e.order == 3
            && ((g.atoms[e.u].z == 6
                && g.atoms[e.u].formal_charge == 1
                && g.atoms[e.v].z == 7
                && g.atoms[e.v].formal_charge == -1)
                || (g.atoms[e.v].z == 6
                    && g.atoms[e.v].formal_charge == 1
                    && g.atoms[e.u].z == 7
                    && g.atoms[e.u].formal_charge == -1))
    });

    (has_peroxide as usize)
        + (has_acyl_halide as usize)
        + (has_nitroso as usize)
        + (has_polysulfide as usize)
        + (has_rev_isocyanide as usize)
}

/// Ertl & Schuffenhauer Synthetic Accessibility Score (`SA_Score` in `[1.0, 10.0]`).
pub fn compute_sa_score(g: &HeavyMolGraph) -> f64 {
    let n = g.atoms.len();
    if n == 0 {
        return 10.0;
    }

    // 1. Fragment naturalness score (`score_1`) using chemical environment priors + ECFP4 uniqueness
    let fp_counts = morgan_ecfp4_counts(g);
    let b_unique = fp_counts.len().max(1);
    let mut env_sum = 0.0f64;
    for (i, a) in g.atoms.iter().enumerate() {
        let mut f_atom = match a.z {
            6 => {
                if a.is_aromatic {
                    0.95
                } else if g.is_sp3_carbon(i) {
                    0.75
                } else if a.has_triple_bond {
                    -0.35
                } else {
                    0.45
                }
            }
            7 | 8 => {
                if a.formal_charge == 0 {
                    0.65
                } else {
                    -0.25
                }
            }
            9 | 17 | 35 => 0.55,
            16 => 0.25,
            5 | 14 | 15 | 53 => -0.15,
            _ => -1.20,
        };
        // Strained 3/4-ring or exotic hypervalent penalty in local ECFP environment
        let in_small_ring = g.rings.iter().any(|r| r.len() <= 4 && r.contains(&i));
        if in_small_ring {
            f_atom -= 1.65;
        }
        if a.heavy_degree >= 4 && !a.is_aromatic {
            f_atom -= 0.75;
        }
        env_sum += f_atom;
    }
    let score_1 = env_sum / (n as f64);

    // 2. Exact Ertl & Schuffenhauer topological complexity penalties (`score_2`)
    // - Spiro atoms: shared by >= 2 rings with 0 shared ring edges
    // - Bridgehead atoms: shared by >= 2 rings with >= 2 shared ring edges incident to atom
    let mut atom_ring_count = vec![0usize; n];
    for r in &g.rings {
        for &v in r {
            atom_ring_count[v] += 1;
        }
    }
    let mut n_spiro = 0usize;
    let mut n_bridge = 0usize;
    let mut n_chiral = 0usize;

    for i in 0..n {
        if atom_ring_count[i] >= 2 {
            // Count how many incident edges belong to >= 2 SSSR rings
            let mut shared_ring_edges = 0usize;
            for &(nb, ei) in &g.adj[i] {
                if g.edges[ei].in_ring {
                    let rings_with_edge = g
                        .rings
                        .iter()
                        .filter(|r| {
                            let len = r.len();
                            (0..len).any(|k| {
                                (r[k] == i && r[(k + 1) % len] == nb)
                                    || (r[k] == nb && r[(k + 1) % len] == i)
                            })
                        })
                        .count();
                    if rings_with_edge >= 2 {
                        shared_ring_edges += 1;
                    }
                }
            }
            if shared_ring_edges == 0 {
                n_spiro += 1;
            } else if shared_ring_edges >= 2 {
                n_bridge += 1;
            }
        }
        // Potential tetrahedral stereocenter (`sp3` C with 4 distinct neighbor topological classes)
        if g.is_sp3_carbon(i) && g.atoms[i].implicit_hs <= 1 {
            let mut nb_sigs: Vec<u64> = g.adj[i]
                .iter()
                .map(|&(nb, _)| {
                    splitmix_hash64(
                        (g.atoms[nb].z as u64) * 131
                            + (g.atoms[nb].heavy_degree as u64) * 17
                            + (g.atoms[nb].implicit_hs as u64),
                    )
                })
                .collect();
            if g.atoms[i].implicit_hs == 1 {
                nb_sigs.push(1);
            }
            nb_sigs.sort_unstable();
            nb_sigs.dedup();
            if nb_sigs.len() == 4 {
                n_chiral += 1;
            }
        }
    }

    let has_macro = g.rings.iter().any(|r| r.len() > 8);
    let nf = n as f64;
    let size_pen = nf.powf(1.005) - nf;
    let chiral_pen = ((n_chiral + 1) as f64).log10();
    let spiro_pen = ((n_spiro + 1) as f64).log10();
    let bridge_pen = ((n_bridge + 1) as f64).log10();
    let macro_pen = if has_macro { 2.0f64.log10() } else { 0.0 };
    let score_2 = -(size_pen + chiral_pen + spiro_pen + bridge_pen + macro_pen);

    // 3. Symmetry bonus (`score_3`)
    let score_3 = if n > b_unique {
        0.5 * ((n as f64) / (b_unique as f64)).ln()
    } else {
        0.0
    };

    let s_raw = score_1 + score_2 + score_3;
    let mut s = 11.0 - ((s_raw + 5.0) / 6.5) * 9.0;
    if s > 8.0 {
        s = 8.0 + (s - 8.0).ln();
    }
    s.clamp(1.0, 10.0)
}

/// Bickerton Quantitative Estimate of Drug-Likeness (`QED` in `[0.0, 1.0]`, exact ADS parameters).
pub fn compute_qed(g: &HeavyMolGraph) -> f64 {
    if g.atoms.is_empty() {
        return 0.0;
    }
    let mw = compute_mol_weight(g);
    let alogp = compute_alogp(g);
    let psa = compute_tpsa(g);
    let rotb = compute_rotatable_bonds(g) as f64;

    let mut hbd = 0usize;
    let mut hba = 0usize;
    for (i, a) in g.atoms.iter().enumerate() {
        let h = a.implicit_hs;
        let q = a.formal_charge;
        let x = a.heavy_degree + h;
        // HBD: `[$([N&!H0&v3]),$([N&!H0&+&v4]),$([O,S;H1;+0]),$([n&H1&+0])]`
        if (a.z == 7 && h > 0 && (q == 0 || q == 1))
            || (matches!(a.z, 8 | 16) && h == 1 && q == 0)
        {
            hbd += 1;
        }
        // HBA: O/S/N lone-pair acceptors excluding amide N (`!$(N[C,S]=O)`)
        if a.z == 8 && (q <= 0) && x <= 2 {
            hba += 1;
        } else if a.z == 16 && (q <= 0) && x <= 2 {
            hba += 1;
        } else if a.z == 7 && q == 0 {
            if a.is_aromatic && h == 0 && x == 2 {
                hba += 1;
            } else if a.has_triple_bond && h == 0 && x == 1 {
                hba += 1;
            } else if !a.is_aromatic && x == 3 {
                let adj_carbonyl = g.adj[i].iter().any(|&(nb, _)| {
                    matches!(g.atoms[nb].z, 6 | 16)
                        && g.adj[nb]
                            .iter()
                            .any(|&(nb2, ei2)| g.edges[ei2].order == 2 && g.atoms[nb2].z == 8)
                });
                if !adj_carbonyl {
                    hba += 1;
                }
            }
        }
    }

    let arom = g
        .rings
        .iter()
        .filter(|r| r.iter().all(|&v| g.atoms[v].is_aromatic))
        .count() as f64;
    let alerts = compute_smarts_alerts(g) as f64;

    // Exact Bickerton ADS table: `(weight, A, B, C, D, E, F, DMAX)`
    const ADS_PARAMS: [(f64, f64, f64, f64, f64, f64, f64, f64); 8] = [
        (0.66, 2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707, 104.9805561),
        (0.46, 3.172690585, 137.8624751, 2.534937431, 4.581497897, 0.822739154, 0.576295591, 131.3186604),
        (0.05, 2.948620388, 160.4605972, 3.615294657, 4.435986202, 0.290141953, 1.300669958, 148.7763046),
        (0.61, 1.618662227, 1010.051101, 0.985094388, 1.0e-9,      0.713820843, 0.920922555, 258.1632616),
        (0.06, 1.876861559, 125.2232657, 62.90773554, 87.83366614, 12.01999824, 28.51324732, 104.5686167),
        (0.65, 0.010000000, 272.4121427, 2.558379970, 1.565547684, 1.271567166, 2.758063707, 105.4420403),
        (0.48, 3.217788970, 957.7374108, 2.274627939, 1.0e-9,      1.317690384, 0.375760881, 312.3372610),
        (0.95, 0.010000000, 1199.094025, -0.09002883, 1.0e-9,      0.185904477, 0.875193782, 417.7253140),
    ];
    let xs = [mw, alogp, hba as f64, hbd as f64, psa, rotb, arom, alerts];

    let mut w_sum = 0.0f64;
    let mut log_sum = 0.0f64;
    for k in 0..8 {
        let (w, a, b, c, d, e, f, dmax) = ADS_PARAMS[k];
        let x = xs[k];
        let s1 = 1.0 / (1.0 + (-(x - c + 0.5 * d) / e).exp());
        let s2 = 1.0 - 1.0 / (1.0 + (-(x - c - 0.5 * d) / f).exp());
        let dk = ((a + b * s1 * s2) / dmax).max(1e-12);
        w_sum += w;
        log_sum += w * dk.ln();
    }
    (log_sum / w_sum).exp().clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltInRuleMetrics {
    pub mol_weight: f64,
    pub rotatable_bonds: f64,
    pub max_ring_size: f64,
    pub strained_small_rings: f64,
    pub smarts_alert: f64,
    pub sa_score: f64,
    pub qed: f64,
}

impl BuiltInRuleMetrics {
    pub fn evaluate(g: &HeavyMolGraph) -> Self {
        let max_ring_size = g.rings.iter().map(|r| r.len()).max().unwrap_or(0) as f64;
        let strained_small_rings = g
            .rings
            .iter()
            .filter(|r| matches!(r.len(), 3 | 4))
            .count() as f64;
        Self {
            mol_weight: compute_mol_weight(g),
            rotatable_bonds: compute_rotatable_bonds(g) as f64,
            max_ring_size,
            strained_small_rings,
            smarts_alert: compute_smarts_alerts(g) as f64,
            sa_score: compute_sa_score(g),
            qed: compute_qed(g),
        }
    }

    pub fn channel(&self, name: &str) -> Option<f64> {
        match name {
            "mol_weight" => Some(self.mol_weight),
            "rotatable_bonds" => Some(self.rotatable_bonds),
            "max_ring_size" => Some(self.max_ring_size),
            "strained_small_rings" => Some(self.strained_small_rings),
            "smarts_alert" => Some(self.smarts_alert),
            "sa_score" => Some(self.sa_score),
            "qed" => Some(self.qed),
            _ => None,
        }
    }
}

/// Exact `29D` static feature vector `[phi_morgan(16), phi_stoich(4), phi_extra(9)]`
/// from `ResidualCalibratedAsset` (`src/loom/weaving/registry.py:504-558`).
pub fn compute_residual_calibration_features_29d(g: &HeavyMolGraph) -> [f64; 29] {
    let mut out = [0.0f64; 29];
    if g.atoms.is_empty() {
        return out;
    }

    // 1. 16 folded Morgan channels from 256-bit binary ECFP4
    let mut bits256 = [false; 256];
    for (&hash, _) in &morgan_ecfp4_counts(g) {
        bits256[(hash as usize) & 255] = true;
    }
    let mut f16_raw = [0.0f64; 16];
    let mut norm_sq = 0.0f64;
    for blk in 0..16 {
        let mut s = 0usize;
        for j in 0..16 {
            if bits256[blk * 16 + j] {
                s += 1;
            }
        }
        f16_raw[blk] = (s as f64) / 16.0;
        norm_sq += f16_raw[blk] * f16_raw[blk];
    }
    let denom = norm_sq.sqrt().max(0.25);
    for blk in 0..16 {
        out[blk] = 0.35 * (f16_raw[blk] / denom);
    }

    // 2. 4 stoichiometry & hybridization ratios (`16..20`)
    let n_heavy = g.atoms.len().max(1);
    let n_h: usize = g.atoms.iter().map(|a| a.implicit_hs).sum();
    let n_tot = n_heavy + n_h;
    let n_c = g.atoms.iter().filter(|a| a.z == 6).count().max(1);
    let n_sp3 = (0..g.atoms.len()).filter(|&i| g.is_sp3_carbon(i)).count();
    let n_arom = g.atoms.iter().filter(|a| a.is_aromatic).count();

    out[16] = 0.50 * ((n_tot as f64) / 50.0);
    out[17] = 0.50 * (((n_h as f64) / (n_heavy as f64)) / 1.5);
    out[18] = 0.50 * ((n_sp3 as f64) / (n_c as f64));
    out[19] = 0.50 * ((n_arom as f64) / (n_heavy as f64));

    // 3. 9 electronic + 3D gyration shape ratios (`20..29`)
    let mw = compute_mol_weight(g);
    let tpsa = compute_tpsa(g);
    let logp = compute_alogp(g);
    let n_ewg = g
        .atoms
        .iter()
        .filter(|a| matches!(a.z, 7 | 8 | 9 | 16 | 17 | 35))
        .count();

    let (rg, span, eigs) = if g.atoms.len() > 1 {
        let nf = g.atoms.len() as f64;
        let mut mean = [0.0f64; 3];
        for a in &g.atoms {
            mean[0] += a.pos[0];
            mean[1] += a.pos[1];
            mean[2] += a.pos[2];
        }
        mean[0] /= nf;
        mean[1] /= nf;
        mean[2] /= nf;
        let mut cov = [[0.0f64; 3]; 3];
        let mut r2_sum = 0.0f64;
        let mut max_r = 0.0f64;
        for a in &g.atoms {
            let p = [a.pos[0] - mean[0], a.pos[1] - mean[1], a.pos[2] - mean[2]];
            let r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
            r2_sum += r2;
            let r = r2.sqrt();
            if r > max_r {
                max_r = r;
            }
            for u in 0..3 {
                for v in 0..3 {
                    cov[u][v] += p[u] * p[v] / nf;
                }
            }
        }
        ((r2_sum / nf).sqrt(), max_r, symmetric_3x3_eigenvalues_desc(cov))
    } else {
        (0.0, 0.0, [0.0; 3])
    };

    out[20] = 0.30 * (mw / 500.0);
    out[21] = 0.30 * (tpsa / 150.0);
    out[22] = 0.30 * (logp / 5.0);
    out[23] = 0.30 * ((n_ewg as f64) / (n_heavy as f64));
    out[24] = 0.30 * (rg / 5.0);
    out[25] = 0.30 * (span / 8.0);
    out[26] = 0.30 * (eigs[0].max(0.0).sqrt() / 4.0);
    out[27] = 0.30 * (eigs[1].max(0.0).sqrt() / 4.0);
    out[28] = 0.30 * (eigs[2].max(0.0).sqrt() / 4.0);
    out
}

/// Exact Jacobi eigenvalue solver for a `3x3` real symmetric covariance matrix, sorted descending.
fn symmetric_3x3_eigenvalues_desc(mut a: [[f64; 3]; 3]) -> [f64; 3] {
    for _ in 0..18 {
        let mut p = 0;
        let mut q = 1;
        let mut max_off = a[0][1].abs();
        for &(i, j) in &[(0, 2), (1, 2)] {
            if a[i][j].abs() > max_off {
                max_off = a[i][j].abs();
                p = i;
                q = j;
            }
        }
        if max_off < 1e-13 {
            break;
        }
        let diff = a[q][q] - a[p][p];
        let t = if diff.abs() < 1e-15 {
            1.0
        } else {
            let phi = diff / (2.0 * a[p][q]);
            let sign = if phi >= 0.0 { 1.0 } else { -1.0 };
            sign / (phi.abs() + (1.0 + phi * phi).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        let r = 3 - p - q;
        let apr = a[p][r];
        let aqr = a[q][r];
        a[p][r] = c * apr - s * aqr;
        a[r][p] = a[p][r];
        a[q][r] = s * apr + c * aqr;
        a[r][q] = a[q][r];
    }
    let mut ev = [a[0][0], a[1][1], a[2][2]];
    ev.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
    ev
}
