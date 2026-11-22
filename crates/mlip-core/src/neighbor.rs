//! Spatial neighbor-list construction and Verlet skin caching (`r_list = r_cut + r_skin`)
//! for isolated 3D molecules and 3D periodic unit cells.

use std::collections::HashMap;

/// Directed interatomic edge `(src -> dst)` within cutoff `r_cut`.
///
/// Convention: `vec_a = (r[dst] + shift * cell) - r[src]` pointing from `src` to `dst`.
/// Callers using the opposite convention simply negate `vec_a` or swap `(src, dst)`.
#[derive(Clone, Copy, Debug)]
pub struct DirectedEdge {
    pub src: u32,
    pub dst: u32,
    pub shift: [i32; 3],
    pub vec_a: [f64; 3],
    pub dist_a: f64,
}

/// Compact SOA / AOS directed neighbor graph within a specified cutoff radius.
#[derive(Clone, Debug, Default)]
pub struct NeighborList {
    pub n_atoms: usize,
    pub cutoff_a: f64,
    pub edges: Vec<DirectedEdge>,
}

impl NeighborList {
    /// Build exact directed neighbor list (`0 < dist < cutoff_a`, deterministic `(src, dst, shift)` order).
    pub fn build(
        coords_a: &[[f64; 3]],
        pbc_cell: Option<&[[f64; 3]; 3]>,
        cutoff_a: f64,
    ) -> Self {
        let n = coords_a.len();
        if n == 0 || cutoff_a <= 0.0 {
            return Self {
                n_atoms: n,
                cutoff_a,
                edges: Vec::new(),
            };
        }

        let r2_max = cutoff_a * cutoff_a;
        let mut edges = Vec::new();

        if let Some(cell) = pbc_cell {
            // General triclinic periodic cell search: determine integer image bounds along a, b, c.
            let inv_cell = invert_3x3(cell);
            let max_shifts = if let Some(inv) = inv_cell {
                // Perpendicular heights h_k = 1 / ||row_k(inv_cell^T)|| = 1 / ||col_k(inv_cell)||
                let mut s = [1i32; 3];
                for k in 0..3 {
                    let col_norm = (inv[0][k] * inv[0][k]
                        + inv[1][k] * inv[1][k]
                        + inv[2][k] * inv[2][k])
                        .sqrt();
                    s[k] = (cutoff_a * col_norm).ceil() as i32;
                }
                s
            } else {
                [1, 1, 1]
            };

            for i in 0..n {
                let ri = coords_a[i];
                for j in 0..n {
                    let rj = coords_a[j];
                    for sa in -max_shifts[0]..=max_shifts[0] {
                        for sb in -max_shifts[1]..=max_shifts[1] {
                            for sc in -max_shifts[2]..=max_shifts[2] {
                                if i == j && sa == 0 && sb == 0 && sc == 0 {
                                    continue;
                                }
                                let shift_x = (sa as f64) * cell[0][0]
                                    + (sb as f64) * cell[1][0]
                                    + (sc as f64) * cell[2][0];
                                let shift_y = (sa as f64) * cell[0][1]
                                    + (sb as f64) * cell[1][1]
                                    + (sc as f64) * cell[2][1];
                                let shift_z = (sa as f64) * cell[0][2]
                                    + (sb as f64) * cell[1][2]
                                    + (sc as f64) * cell[2][2];
                                let dx = rj[0] + shift_x - ri[0];
                                let dy = rj[1] + shift_y - ri[1];
                                let dz = rj[2] + shift_z - ri[2];
                                let r2 = dx * dx + dy * dy + dz * dz;
                                if r2 < r2_max && r2 > 0.0 {
                                    edges.push(DirectedEdge {
                                        src: i as u32,
                                        dst: j as u32,
                                        shift: [sa, sb, sc],
                                        vec_a: [dx, dy, dz],
                                        dist_a: r2.sqrt(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        } else if n <= 128 {
            // Fast cache-friendly symmetric upper-triangle sweep for molecular systems (N <= 128),
            // emitting both (i -> j) and (j -> i) then sorting by (src, dst) for deterministic order.
            edges.reserve(n * n.min(48));
            for i in 0..n {
                let ri = coords_a[i];
                for j in (i + 1)..n {
                    let rj = coords_a[j];
                    let dx = rj[0] - ri[0];
                    let dy = rj[1] - ri[1];
                    let dz = rj[2] - ri[2];
                    let r2 = dx * dx + dy * dy + dz * dz;
                    if r2 < r2_max && r2 > 0.0 {
                        let d = r2.sqrt();
                        edges.push(DirectedEdge {
                            src: i as u32,
                            dst: j as u32,
                            shift: [0, 0, 0],
                            vec_a: [dx, dy, dz],
                            dist_a: d,
                        });
                        edges.push(DirectedEdge {
                            src: j as u32,
                            dst: i as u32,
                            shift: [0, 0, 0],
                            vec_a: [-dx, -dy, -dz],
                            dist_a: d,
                        });
                    }
                }
            }
            edges.sort_unstable_by_key(|e| (e.src, e.dst));
        } else {
            // O(N) 3D spatial cell-list hashing for large non-periodic molecular complexes / proteins.
            let inv_cell_size = 1.0 / cutoff_a;
            let mut buckets: HashMap<(i32, i32, i32), Vec<usize>> =
                HashMap::with_capacity(n / 2 + 1);
            for (i, c) in coords_a.iter().enumerate() {
                let key = (
                    (c[0] * inv_cell_size).floor() as i32,
                    (c[1] * inv_cell_size).floor() as i32,
                    (c[2] * inv_cell_size).floor() as i32,
                );
                buckets.entry(key).or_default().push(i);
            }
            edges.reserve(n * 48);
            for (i, ri) in coords_a.iter().enumerate() {
                let cx = (ri[0] * inv_cell_size).floor() as i32;
                let cy = (ri[1] * inv_cell_size).floor() as i32;
                let cz = (ri[2] * inv_cell_size).floor() as i32;
                let mut local_neighbors = Vec::new();
                for ox in -1..=1 {
                    for oy in -1..=1 {
                        for oz in -1..=1 {
                            if let Some(atom_ids) = buckets.get(&(cx + ox, cy + oy, cz + oz)) {
                                for &j in atom_ids {
                                    if i == j {
                                        continue;
                                    }
                                    let rj = coords_a[j];
                                    let dx = rj[0] - ri[0];
                                    let dy = rj[1] - ri[1];
                                    let dz = rj[2] - ri[2];
                                    let r2 = dx * dx + dy * dy + dz * dz;
                                    if r2 < r2_max && r2 > 0.0 {
                                        local_neighbors.push(DirectedEdge {
                                            src: i as u32,
                                            dst: j as u32,
                                            shift: [0, 0, 0],
                                            vec_a: [dx, dy, dz],
                                            dist_a: r2.sqrt(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                local_neighbors.sort_unstable_by_key(|e| e.dst);
                edges.extend(local_neighbors);
            }
        }

        Self {
            n_atoms: n,
            cutoff_a,
            edges,
        }
    }
}

/// Verlet neighbor-list cache with skin buffer (`r_list = r_cut + r_skin`).
///
/// During geometry relaxation and molecular dynamics, atomic displacements per step are small
/// (`0.005 - 0.10 Å`). By caching candidate pairs within `r_cut + r_skin`, we skip spatial
/// graph reconstruction whenever `max_i ||r_i - r_ref_i|| < 0.5 * r_skin` and only refresh
/// exact `vec_a` / `dist_a` over the cached pair list!
#[derive(Clone, Debug)]
pub struct NeighborCache {
    cutoff_a: f64,
    skin_a: f64,
    ref_coords_a: Vec<[f64; 3]>,
    ref_pbc_cell: Option<[[f64; 3]; 3]>,
    skin_candidates: Vec<(u32, u32, [i32; 3])>,
    pub rebuild_count: usize,
    pub reuse_count: usize,
}

impl NeighborCache {
    pub fn new(cutoff_a: f64, skin_a: f64) -> Self {
        Self {
            cutoff_a,
            skin_a: skin_a.max(0.0),
            ref_coords_a: Vec::new(),
            ref_pbc_cell: None,
            skin_candidates: Vec::new(),
            rebuild_count: 0,
            reuse_count: 0,
        }
    }

    pub fn invalidate(&mut self) {
        self.ref_coords_a.clear();
        self.skin_candidates.clear();
    }

    pub fn update(
        &mut self,
        coords_a: &[[f64; 3]],
        pbc_cell: Option<&[[f64; 3]; 3]>,
    ) -> NeighborList {
        let can_reuse = self.skin_a > 0.0
            && self.ref_coords_a.len() == coords_a.len()
            && self.ref_pbc_cell.as_ref() == pbc_cell
            && {
                let half_skin_sq = 0.25 * self.skin_a * self.skin_a;
                self.ref_coords_a
                    .iter()
                    .zip(coords_a.iter())
                    .all(|(r0, r1)| {
                        let dx = r1[0] - r0[0];
                        let dy = r1[1] - r0[1];
                        let dz = r1[2] - r0[2];
                        (dx * dx + dy * dy + dz * dz) < half_skin_sq
                    })
            };

        if !can_reuse {
            let full_list =
                NeighborList::build(coords_a, pbc_cell, self.cutoff_a + self.skin_a);
            self.skin_candidates = full_list
                .edges
                .iter()
                .map(|e| (e.src, e.dst, e.shift))
                .collect();
            self.ref_coords_a = coords_a.to_vec();
            self.ref_pbc_cell = pbc_cell.copied();
            self.rebuild_count += 1;

            let r2_max = self.cutoff_a * self.cutoff_a;
            let edges = full_list
                .edges
                .into_iter()
                .filter(|e| e.dist_a * e.dist_a < r2_max)
                .collect();
            return NeighborList {
                n_atoms: coords_a.len(),
                cutoff_a: self.cutoff_a,
                edges,
            };
        }

        self.reuse_count += 1;
        let r2_max = self.cutoff_a * self.cutoff_a;
        let mut edges = Vec::with_capacity(self.skin_candidates.len());
        for &(src, dst, shift) in &self.skin_candidates {
            let ri = coords_a[src as usize];
            let rj = coords_a[dst as usize];
            let (sx, sy, sz) = if let Some(cell) = pbc_cell {
                (
                    (shift[0] as f64) * cell[0][0]
                        + (shift[1] as f64) * cell[1][0]
                        + (shift[2] as f64) * cell[2][0],
                    (shift[0] as f64) * cell[0][1]
                        + (shift[1] as f64) * cell[1][1]
                        + (shift[2] as f64) * cell[2][1],
                    (shift[0] as f64) * cell[0][2]
                        + (shift[1] as f64) * cell[1][2]
                        + (shift[2] as f64) * cell[2][2],
                )
            } else {
                (0.0, 0.0, 0.0)
            };
            let dx = rj[0] + sx - ri[0];
            let dy = rj[1] + sy - ri[1];
            let dz = rj[2] + sz - ri[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < r2_max && r2 > 0.0 {
                edges.push(DirectedEdge {
                    src,
                    dst,
                    shift,
                    vec_a: [dx, dy, dz],
                    dist_a: r2.sqrt(),
                });
            }
        }

        NeighborList {
            n_atoms: coords_a.len(),
            cutoff_a: self.cutoff_a,
            edges,
        }
    }
}

fn invert_3x3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);
    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    Some([
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ])
}
