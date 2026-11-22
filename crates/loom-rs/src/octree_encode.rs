//! 3D Molecule -> `Uni-3DAR` Octree 2-Token-Per-Cell Sequence Encoder (`octree_encode.rs`).
//!
//! Implements exact Morton Z-order sorting, top-down octree expansion (`subcell_orders`),
//! `SO(3)` rotation + half-voxel shift + sub-voxel `0.01 Å` discretization, `HeavyAtomValenceHead`
//! auxiliary target generation, and trajectory-to-sequence packing for `TopoGRPO`!

use crate::chem_rules::MolRecord;
use serde::{Deserialize, Serialize};
use uni3dar_rs::config::BackboneConfig;
use uni3dar_rs::sampler::DetRng;
use uni3dar_rs::valence::{elem_id_to_symbol, rcov_angstrom};

pub const SUBCELL_ORDERS: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

#[inline]
pub fn interleave_bits_10(mut v: u32) -> u32 {
    v &= 0x03FF;
    v = (v | (v << 16)) & 0x0300_00FF;
    v = (v | (v << 8)) & 0x0300_F00F;
    v = (v | (v << 4)) & 0x030C_30C3;
    v = (v | (v << 2)) & 0x0924_9249;
    v
}

#[inline]
pub fn z_order_3d(x: u32, y: u32, z: u32) -> u32 {
    interleave_bits_10(x) | (interleave_bits_10(y) << 1) | (interleave_bits_10(z) << 2)
}

#[inline]
fn sample_standard_normal(rng: &mut DetRng) -> f64 {
    let u1 = (rng.next_f32() as f64).clamp(1e-7, 1.0 - 1e-7);
    let u2 = rng.next_f32() as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn sample_so3_rotation(rng: &mut DetRng) -> [[f64; 3]; 3] {
    let mut q = [
        sample_standard_normal(rng),
        sample_standard_normal(rng),
        sample_standard_normal(rng),
        sample_standard_normal(rng),
    ];
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        .sqrt()
        .max(1e-12);
    for v in &mut q {
        *v /= norm;
    }
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

#[inline]
fn round_ties_even_i32(x: f64) -> i32 {
    let fl = x.floor();
    let diff = x - fl;
    if diff < 0.5 {
        fl as i32
    } else if diff > 0.5 {
        (fl as i32) + 1
    } else {
        let i = fl as i32;
        if (i & 1) == 0 {
            i
        } else {
            i + 1
        }
    }
}

/// One token in the 2-token-per-cell interleaved `Uni-3DAR` causal sequence.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SeqToken {
    pub typ: usize,
    pub xyz_bins: [usize; 3],
    pub level: i32,
    pub rem_atoms: i32,
    pub phy_pos: [f32; 3],
    pub is_root: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TreeDecisionTarget {
    /// Token position inside `EncodedOctreeSeq::tokens` (always even, `typ == atom_mask_id`).
    pub token_pos: usize,
    pub level: i32,
    /// `0..254` (`bitmask - 1` for `bitmask in 1..=255`).
    pub label_255: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AtomDecisionTarget {
    /// Token position inside `EncodedOctreeSeq::tokens` (always even, `level == 0`).
    pub token_pos: usize,
    /// `0..117` for real element `Z - 1`, or `atom_null_id (345)` for empty `[NULL]` leaf voxel in rollouts.
    pub elem_id: usize,
    /// Sub-voxel `0..23` bins (`[12,12,12]` when `elem_id == atom_null_id`).
    pub xyz_bins: [usize; 3],
    /// Heavy-atom auxiliary valence target: `Some((num_attached_h in 0..=4, charge_cls in 0..=2))` when `0 < elem_id < 342`.
    pub heavy_valence: Option<(usize, usize)>,
    /// True if this is the final leaf cell of the trajectory (used for closed-shell `electron_parity` masking in `TopoGRPO`).
    pub is_terminal_leaf: bool,
    /// Cumulative `sum(Z) % 2` of all prior real atoms in the same trajectory (before this leaf).
    pub prior_z_parity: usize,
    /// Coarse voxel center `[f32; 3]` (Å) of this leaf cell (for geometric guard in `TopoGRPO`).
    pub voxel_center: [f32; 3],
}

/// Encoded octree sequence for a single 3D molecule or sampled rollout trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedOctreeSeq {
    pub n_real_atoms: usize,
    pub tokens: Vec<SeqToken>,
    pub tree_targets: Vec<TreeDecisionTarget>,
    pub atom_targets: Vec<AtomDecisionTarget>,
    /// Optional property target vector aligned with `target_cols`.
    pub targets_raw: Vec<f32>,
    pub active_mask: Vec<bool>,
}

/// Encode a 3D `MolRecord` into an exact `Uni-3DAR` 2-token-per-cell sequence.
pub fn encode_molecule_to_octree(
    rec: &MolRecord,
    bb: &BackboneConfig,
    target_cols: &[String],
    training: bool,
    random_rotation_prob: f32,
    rng: &mut DetRng,
) -> Result<EncodedOctreeSeq, String> {
    let n = rec.atomic_numbers.len();
    if n == 0 {
        return Err("Cannot encode empty molecule".into());
    }

    // 1. Allowed-element gate
    if !bb.allowed_elements.is_empty() {
        for &z in &rec.atomic_numbers {
            let sym = elem_id_to_symbol((z as usize).saturating_sub(1));
            if !bb.allowed_elements.iter().any(|s| s.eq_ignore_ascii_case(sym)) {
                return Err(format!("Unsupported element '{sym}' for Uni-3DAR backbone"));
            }
        }
    }

    let grid_len = bb.grid_len as f64;
    let xyz_res = bb.xyz_resolution as f64;
    let g_box = 1i32 << bb.merge_level;
    let num_xyz = bb.num_xyz as i32;

    // Retry up to 8 times if training augmentation causes 2 covalently close atoms to collide in one 0.24 Å voxel
    let max_tries = if training { 8 } else { 3 };
    let mut chosen_grid: Option<(Vec<[i32; 3]>, Vec<[usize; 3]>)> = None;

    for attempt in 0..max_tries {
        // Center coordinates
        let mut mean = [0.0f64; 3];
        for p in &rec.positions {
            mean[0] += p[0];
            mean[1] += p[1];
            mean[2] += p[2];
        }
        mean[0] /= n as f64;
        mean[1] /= n as f64;
        mean[2] /= n as f64;

        let mut pos: Vec<[f64; 3]> = rec
            .positions
            .iter()
            .map(|p| [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]])
            .collect();

        // Optional SO(3) rotation
        let do_rot = (training && rng.next_f32() < random_rotation_prob) || (!training && attempt > 0);
        if do_rot {
            let rot = sample_so3_rotation(rng);
            for p in &mut pos {
                let x = p[0] * rot[0][0] + p[1] * rot[0][1] + p[2] * rot[0][2];
                let y = p[0] * rot[1][0] + p[1] * rot[1][1] + p[2] * rot[1][2];
                let z = p[0] * rot[2][0] + p[1] * rot[2][1] + p[2] * rot[2][2];
                *p = [x, y, z];
            }
        }

        // Half-voxel shift + optional continuous translation jitter
        let shift_jitter = if training {
            [
                0.5 * grid_len + sample_standard_normal(rng) * (0.5 * grid_len),
                0.5 * grid_len + sample_standard_normal(rng) * (0.5 * grid_len),
                0.5 * grid_len + sample_standard_normal(rng) * (0.5 * grid_len),
            ]
        } else {
            [0.5 * grid_len; 3]
        };
        for p in &mut pos {
            p[0] += shift_jitter[0];
            p[1] += shift_jitter[1];
            p[2] += shift_jitter[2];
        }

        // Voxel & sub-voxel discretization (`rem_euclid` semantics!)
        let mut g_pos = vec![[0i32; 3]; n];
        let mut u_bins = vec![[0usize; 3]; n];
        for i in 0..n {
            for d in 0..3 {
                let g_raw = (pos[i][d] / grid_len).floor();
                let rem = pos[i][d] - g_raw * grid_len;
                let mut g_i = g_raw as i32;
                let mut u_i = round_ties_even_i32(rem / xyz_res);
                if u_i >= num_xyz {
                    g_i += 1;
                    u_i = 0;
                }
                g_pos[i][d] = g_i;
                u_bins[i][d] = u_i.clamp(0, num_xyz - 1) as usize;
            }
        }

        // Zero-min shift -> box-center shift -> optional {-1,0,+1} shift -> per-axis clamp-shift
        for d in 0..3 {
            let min_d = g_pos.iter().map(|g| g[d]).min().unwrap_or(0);
            for g in &mut g_pos {
                g[d] -= min_d;
            }
            let mean_f32: f32 = g_pos.iter().map(|g| g[d] as f32).sum::<f32>() / (n as f32);
            let center_i = mean_f32.trunc() as i32;
            let rand_s = if training {
                ((rng.next_u64() % 3) as i32) - 1
            } else {
                0
            };
            for g in &mut g_pos {
                g[d] = g[d] - center_i + (g_box / 2) + rand_s;
            }
            let min_after = g_pos.iter().map(|g| g[d]).min().unwrap_or(0);
            if min_after < 0 {
                for g in &mut g_pos {
                    g[d] -= min_after;
                }
            }
            let max_after = g_pos.iter().map(|g| g[d]).max().unwrap_or(0);
            if max_after >= g_box {
                let delta = max_after - g_box + 1;
                for g in &mut g_pos {
                    g[d] -= delta;
                }
            }
        }

        let in_box = g_pos
            .iter()
            .all(|g| (0..3).all(|d| g[d] >= 0 && g[d] < g_box));
        if !in_box {
            continue;
        }
        let mut uniq = std::collections::HashSet::with_capacity(n);
        let all_distinct = g_pos.iter().all(|&g| uniq.insert(g));
        if all_distinct {
            chosen_grid = Some((g_pos, u_bins));
            break;
        }
    }

    let (g_pos, u_bins) = chosen_grid
        .ok_or_else(|| "Molecule exceeded octree box or had persistent sub-0.24Å voxel collision".to_string())?;

    // 2. Sort all N atoms in strictly increasing 3D Morton Z-order!
    let mut order: Vec<(u32, usize)> = (0..n)
        .map(|i| {
            let code = z_order_3d(g_pos[i][0] as u32, g_pos[i][1] as u32, g_pos[i][2] as u32);
            (code, i)
        })
        .collect();
    order.sort_unstable_by_key(|&(code, _)| code);

    let sorted_z_codes: Vec<u32> = order.iter().map(|&(c, _)| c).collect();
    let sorted_elem_id: Vec<usize> = order
        .iter()
        .map(|&(_, idx)| (rec.atomic_numbers[idx] as usize).saturating_sub(1))
        .collect();
    let sorted_g: Vec<[i32; 3]> = order.iter().map(|&(_, idx)| g_pos[idx]).collect();
    let sorted_u: Vec<[usize; 3]> = order.iter().map(|&(_, idx)| u_bins[idx]).collect();

    // Reconstructed fine physical coordinates of sorted leaf atoms (for `HeavyAtomValenceHead` targets)
    let fine_pos: Vec<[f32; 3]> = (0..n)
        .map(|m| {
            [
                (sorted_g[m][0] as f32) * bb.grid_len + (sorted_u[m][0] as f32) * bb.xyz_resolution,
                (sorted_g[m][1] as f32) * bb.grid_len + (sorted_u[m][1] as f32) * bb.xyz_resolution,
                (sorted_g[m][2] as f32) * bb.grid_len + (sorted_u[m][2] as f32) * bb.xyz_resolution,
            ]
        })
        .collect();
    let heavy_val_targets = compute_heavy_valence_targets(&sorted_elem_id, &fine_pos);

    // 3. Top-down octree expansion (`level = merge_level .. 1`)
    let ml = bb.merge_level;
    let mut cur_cells: Vec<([i32; 3], usize)> = vec![([0, 0, 0], n)];
    let mut tokens = Vec::with_capacity(2 * (4 * n + 16));
    let mut tree_targets = Vec::new();

    let non_trapping = bb.non_trapping_budget && bb.has_budget_emb;

    for lvl in (1..=ml).rev() {
        let child_step = 1i32 << (lvl - 1);
        let morton_shift = 3 * (lvl - 1);
        let mut next_cells: Vec<([i32; 3], usize)> = Vec::new();
        let mut cum_atoms_before = 0usize;

        for (cell_idx, &(origin, cell_cnt)) in cur_cells.iter().enumerate() {
            let mut bitmask = 0usize;
            for k in 0..8 {
                let sub_o = [
                    origin[0] + child_step * SUBCELL_ORDERS[k][0],
                    origin[1] + child_step * SUBCELL_ORDERS[k][1],
                    origin[2] + child_step * SUBCELL_ORDERS[k][2],
                ];
                let sub_prefix = z_order_3d(sub_o[0] as u32, sub_o[1] as u32, sub_o[2] as u32) >> morton_shift;
                let sub_cnt = sorted_z_codes
                    .iter()
                    .filter(|&&zc| (zc >> morton_shift) == sub_prefix)
                    .count();
                if sub_cnt > 0 {
                    bitmask |= 1 << k;
                    next_cells.push((sub_o, sub_cnt));
                }
            }
            debug_assert!((1..=255).contains(&bitmask));

            let half_edge = 0.5f32 * ((1usize << lvl) as f32) * bb.grid_len;
            let phy_c = [
                (origin[0] as f32) * bb.grid_len + half_edge,
                (origin[1] as f32) * bb.grid_len + half_edge,
                (origin[2] as f32) * bb.grid_len + half_edge,
            ];
            let rem_a = if non_trapping {
                n as i32
            } else if lvl == ml {
                0
            } else {
                (n.saturating_sub(cum_atoms_before)) as i32
            };
            let is_root = lvl == ml && cell_idx == 0;

            let q_pos = tokens.len();
            tokens.push(SeqToken {
                typ: bb.atom_mask_id,
                xyz_bins: [bb.xyz_null_id; 3],
                level: lvl as i32,
                rem_atoms: rem_a,
                phy_pos: phy_c,
                is_root,
            });
            tree_targets.push(TreeDecisionTarget {
                token_pos: q_pos,
                level: lvl as i32,
                label_255: bitmask - 1,
            });

            tokens.push(SeqToken {
                typ: bb.tree0_split_id + bitmask,
                xyz_bins: [bb.xyz_null_id; 3],
                level: lvl as i32,
                rem_atoms: rem_a,
                phy_pos: phy_c,
                is_root: false,
            });

            cum_atoms_before += cell_cnt;
        }
        cur_cells = next_cells;
    }

    // 4. Leaf atom level (`level = 0`, `m = 0..N-1`)
    let mut atom_targets = Vec::with_capacity(n);
    let mut running_z_parity = 0usize;
    for m in 0..n {
        let vox_c = [
            ((sorted_g[m][0] as f32) + 0.5) * bb.grid_len,
            ((sorted_g[m][1] as f32) + 0.5) * bb.grid_len,
            ((sorted_g[m][2] as f32) + 0.5) * bb.grid_len,
        ];
        let rem_a = if non_trapping { n as i32 } else { (n - m) as i32 };
        let q_pos = tokens.len();
        tokens.push(SeqToken {
            typ: bb.atom_mask_id,
            xyz_bins: [bb.xyz_null_id; 3],
            level: 0,
            rem_atoms: rem_a,
            phy_pos: vox_c,
            is_root: false,
        });
        atom_targets.push(AtomDecisionTarget {
            token_pos: q_pos,
            elem_id: sorted_elem_id[m],
            xyz_bins: sorted_u[m],
            heavy_valence: heavy_val_targets[m],
            is_terminal_leaf: m + 1 == n,
            prior_z_parity: running_z_parity,
            voxel_center: vox_c,
        });
        running_z_parity = (running_z_parity + sorted_elem_id[m] + 1) & 1;

        tokens.push(SeqToken {
            typ: sorted_elem_id[m],
            xyz_bins: sorted_u[m],
            level: 0,
            rem_atoms: rem_a,
            phy_pos: fine_pos[m],
            is_root: false,
        });
    }

    // 5. Trailing `[MASK]` + `[CLS]` pair (`level = -1`)
    let cls_p = bb.cls_phy_pos();
    tokens.push(SeqToken {
        typ: bb.atom_mask_id,
        xyz_bins: [bb.xyz_null_id; 3],
        level: -1,
        rem_atoms: 0,
        phy_pos: [cls_p, cls_p, cls_p],
        is_root: false,
    });
    tokens.push(SeqToken {
        typ: bb.atom_cls_id,
        xyz_bins: [bb.xyz_null_id; 3],
        level: -1,
        rem_atoms: 0,
        phy_pos: [cls_p, cls_p, cls_p],
        is_root: false,
    });

    // Extract target property values aligned with `target_cols`
    let mut targets_raw = vec![0.0f32; target_cols.len()];
    let mut active_mask = vec![false; target_cols.len()];
    for (k, col) in target_cols.iter().enumerate() {
        if let Some(&val) = rec.properties.get(col) {
            if val.is_finite() {
                targets_raw[k] = val as f32;
                active_mask[k] = true;
            }
        }
    }

    Ok(EncodedOctreeSeq {
        n_real_atoms: n,
        tokens,
        tree_targets,
        atom_targets,
        targets_raw,
        active_mask,
    })
}

/// Exact `HeavyAtomValenceHead` supervision targets `(num_h in 0..=4, charge_cls in 0..=2)`
/// matching `_compute_valence_aux_loss` in `src/loom/weaving/uni3dar_slot.py`!
pub fn compute_heavy_valence_targets(
    elem_ids: &[usize],
    pos: &[[f32; 3]],
) -> Vec<Option<(usize, usize)>> {
    let n = elem_ids.len();
    let mut num_h = vec![0usize; n];
    let mut heavy_deg = vec![0usize; n];
    let mut has_o_nb = vec![false; n];
    let mut heavy_adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    let rcov: Vec<f32> = elem_ids
        .iter()
        .map(|&id| rcov_angstrom(elem_id_to_symbol(id)))
        .collect();

    for a in 0..n {
        if elem_ids[a] == 0 || elem_ids[a] >= 342 {
            continue;
        }
        let mut h_cnt = 0usize;
        for b in 0..n {
            if a == b || elem_ids[b] >= 342 {
                continue;
            }
            let dx = pos[a][0] - pos[b][0];
            let dy = pos[a][1] - pos[b][1];
            let dz = pos[a][2] - pos[b][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if elem_ids[b] == 0 {
                if d <= 1.32 {
                    h_cnt += 1;
                }
            } else if d <= 1.26 * (rcov[a] + rcov[b]) {
                heavy_deg[a] += 1;
                heavy_adj[a].push(b);
                if elem_ids[b] == 7 {
                    has_o_nb[a] = true;
                }
            }
        }
        num_h[a] = h_cnt.min(4);
    }

    let is_n_plus: Vec<bool> = (0..n)
        .map(|a| elem_ids[a] == 6 && heavy_deg[a] >= 3 && has_o_nb[a] && num_h[a] == 0)
        .collect();

    (0..n)
        .map(|a| {
            if elem_ids[a] == 0 || elem_ids[a] >= 342 {
                return None;
            }
            let is_o_minus = elem_ids[a] == 7
                && heavy_deg[a] == 1
                && num_h[a] == 0
                && heavy_adj[a].iter().any(|&b| is_n_plus[b]);
            let chg_cls = if is_o_minus {
                0
            } else if is_n_plus[a] {
                2
            } else {
                1
            };
            Some((num_h[a], chg_cls))
        })
        .collect()
}
