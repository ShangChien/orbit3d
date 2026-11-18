# Uni-Loom: Predictor-Agnostic 3D Molecular Weaving with Dual-Track Valence Purity and Grounded Recursive Self-Improvement

**Shangqian Chen, Guojiang Zhao, Zheng Cheng, Jiankun Wang, Hongshuai Wang, Zhifeng Gao**  
*DP Technology, Beijing, China*  
Correspondence: `{chensq, zhaogj, chengz, wangjk, wanghs, gaozf}@dp.tech`  
*November 2026*

---

## Abstract

De novo three-dimensional (3D) molecular design rarely lacks isolated foundation components: autoregressive 3D point-cloud generators, large molecular property predictors, and universal machine-learning interatomic potentials (MLIPs) have each matured rapidly. Yet composing these heterogeneous models directly into a closed discovery pipeline frequently fails due to conditional response attenuation, cross-scale policy gradient interference, severe point-cloud valence artifacts (`15%–39%` exotic formal charges induced by single-hydrogen placement errors), reward hacking inside predictor blindspots, and tight coupling between task metrics and training code. Here we present **Uni-Loom** (`loom`), a four-layer, predictor-agnostic 3D molecular weaving architecture that composes frozen generative backbones, arbitrary external property assets, hot-swappable MLIPs, and uncertainty-aware calibration loops into a unified closed-loop discovery system. First, we introduce a uniform protocol (`PredictorAsset`) and declarative multi-objective critic bus (`CompositeCriticBus`) that decouple generative adaptation from metric identity while supporting single-oracle applicability-domain gating and differentiable coordinate refinement alongside MLIP forces. Second, we resolve 3D valence pathology through a **Dual-Track Valence Purity** engine combining source-side coordination guards (`HeavyAtomValenceHead` + bond-order saturation masking) with zero-heavy-atom-displacement Lewis reconciliation (`RMSD = 0.0000 Å`), alongside scale-decoupled topological reinforcement learning (`TopoGRPO`) and detached sub-voxel (`0.01 Å`) coordinate distillation. Third, we couple a heteroscedastic `1D/2D` Gaussian Process reachability atlas (`GPAtlas`) with **Grounded Recursive Self-Improvement (RSI)** to separate generator under-exploration from out-of-domain predictor blindspots and chemical prior ceilings. Across `1,792`-proposal `OE62` benchmarks, Uni-Loom improves target hit rates from `20.70%` (`371/1,792`) to `35.21%` (`631/1,792`), cuts cross-foundation blind audit error by `> 0.78 eV` (`1.536 → 0.744 eV`), reduces cross-foundation `Audit - Reward` divergence by `4.6×` (`0.655 → 0.143 eV`) under `SA_Score ≤ 3.5` synthesis constraints while doubling synthesizable clean hits (`118 → 196`), demonstrates causal zero-retrain blindspot unlocking across a two-round Grounded RSI loop that lifts blind `DPA-4` clean constrained hits `4.40×` (`2.90% → 12.78%`) and cuts untouched blind deep-DFT test error by `33.8%` (`0.990 → 0.655 eV`) using only `64` acquired DFT labels, and achieves `17×` lower training compute (`3.0 vs. 52.0 GPU-h`), `2.6–2.9×` faster test-time decoding (`3.54 mol/s` via mid-tree `OctreeProcessCritic 16→4`), `75.2%–76.0%` surrogate and `52.5%–52.9%` blind synthesizable yields under `+0.12 GPU-h` (`56`-step) hot-swapped flow alignment (`Loom-AWR`), and `42`-prompt `PBE0/def2-TZVP` quantum accuracy (`50.0%–54.8%` raw quantum hit yield; `0.270 eV` HOMO MAE under `Loom-AWR` and `50.0%` parity for `3 GPU-h` `Uni-Loom 16→4`) on `OE62`.

---

## 1. Introduction

Inverse molecular discovery requires simultaneously navigating discrete covalent bonding graphs, continuous three-dimensional nuclear coordinates $\mathbf{R} \in \mathbb{R}^{N \times 3}$, electronic frontier properties [1–3], synthetic tractability [7], and chemical stability. While equivariant diffusion models [4, 5], flow-matching generators [6], and coarse-to-fine octree 3D autoregressive transformers such as `Uni-3DAR` [8] can generate realistic 3D atomic point clouds, and large pretrained property foundations (`Uni-Mol2` [10], `Uni-Mol` [11]) alongside universal MLIPs (`DPA-4` [13], `HyFF` [14]) achieve strong accuracy across chemical space, **industrial deployment rarely starts from a monolithic jointly trained model**. Instead, discovery teams accumulate mature property predictors—differentiable 3D neural models, black-box ensembles, and analytical cheminformatics filters—whose weights, units, and modalities cannot be retrained for every new target specification.

Directly piping a 3D point-cloud generator into external predictors and offline optimization exposes four coupled failure modes:

1. **Conditional compression and scale entanglement in 3D autoregressive generators.** When continuous scalar targets are injected via shallow bounded residual gates (`0.08 * tanh`), conditional response slopes stall near `0.41–0.54`, ignoring extreme electronic requests. Conversely, sending global scalar property rewards backward through high-variance `0.01 Å` sub-voxel Cartesian tokens overwhelms credit assignment on coarse topological octree decisions, broadening within-target variance whenever coordinate and discrete graph updates are coupled.
2. **Exotic formal-charge (zwitterionic) artifacts in raw 3D point clouds.** Autoregressive 3D generators emit unbonded coordinates $\{(Z_i, \mathbf{r}_i)\}_{i=1}^N$. Because covalent hydrogen radii are small (`~0.31 Å`) and sub-voxel discretization straddles voxel boundaries, missing or adding a single peripheral hydrogen causes downstream closed-shell bond perception (`RDKit rdDetermineBonds` at `charge = 0` [9]) to neutralize electron counts by imposing spurious formal charges (`[C-]... [O+]`, `[B-]`, haloniums) on `15.2%–38.9%` of geometrically valid molecules. Post-hoc force-field relaxation cannot repair distorted Lewis formal charges once assigned.
3. **Reward hacking against out-of-domain property assets.** Even billion-parameter predictors degrade outside their training support on exotic or strained skeletons. Evaluated on `4,737` leak-free generated `PBE0/def2-TZVP` structures, frozen `Uni-Mol2 1.1B` exhibits `0.720 / 0.778 eV` MAE on `HOMO/LUMO`, whereas a DFT-calibrated `DPA-4` head achieves `0.186 / 0.183 eV`. Optimizing a policy unguardedly against a single reward model without independent auditing, applicability-domain gating, or structural constraints drives the generator toward high-strain, high-`SA_Score` molecular pathologies that exploit predictor blindspots.
4. **Inability to diagnose *why* target zones underperform during self-improvement.** When hit rates drop in an extremal property window, naïve reinforcement learning cannot distinguish an *under-explored generator mode* (fixable via zero-DFT inner curriculum sampling) from an *external predictor blindspot* (where further RL triggers collapse, requiring targeted DFT or experimental acquisition) or a *chemical prior ceiling* (where no stable neutral organic molecule exists within target bounds).

**Contributions.** To overcome these barriers cleanly without hardcoding target identities such as `homo` or `lumo`, we present **Uni-Loom (`loom v2.0`)**:
- **Predictor-Agnostic Asset Hub & Declarative Critic Bus (Layers 1–2):** A unified capability-aware wrapper interface supporting differentiable 3D models (exposing coordinate gradients $\nabla_{\mathbf{R}}\hat{y}$), black-box 3D/2D QSAR models via zero-retrain Bayesian residual calibration (`ResidualCalibratedAsset`), analytical rules (`SA_Score` [7], `QED` [12], SMARTS alerts), and a hot-swappable `PotentialSlot` (`DPA-4` $\leftrightarrow$ `HyFF`). Tasks are specified declaratively in YAML (`TaskWeavingSpec`) with arbitrary objectives (`point`, `interval`, `max`, `min`), constraint inequalities that consume zero conditioner dimensions, and monotonic hierarchical advantages with `cross_oracle` or `single_oracle_ad` auditing.
- **Scale-Decoupled Weaving & Dual-Track Valence Purity (Layer 3):** Unbounded Fourier target conditioning, 24-layer rank-16 `LoRA` [19], tree-depth annealed Classifier-Free Guidance (CFG) [20], and vectorized `BF16` decoding achieve `32.52 mol/s` on a single `A800-80GB` (`7.29×` over unvectorized FP32 loops). Training decouples coarse octree/element `TopoGRPO` (Stage 2.5A) from detached `0.01 Å` equilibrium coordinate distillation (Stage 2.5B). **Track A** suppresses charge pathology at generation time via bond-order saturation guards and `< 30K`-parameter `HeavyAtomValenceHead` supervision; **Track B** reconciles residual hydrogen/Lewis errors at `0.0000 Å` heavy-atom RMSD while preserving dipolar functional groups (nitro, *N*-oxide, sulfoxide, azide). Both *Raw Point-Cloud* and *Valence-Reconciled* metrics are reported side-by-side throughout.
- **Heteroscedastic `GPAtlas` & Grounded RSI (Layer 4):** Decomposing total target uncertainty into generation dispersion ($\sigma_{\text{gen}}^2$) and predictor epistemic variance ($\sigma_{\text{cal}}^2$) partitions arbitrary property axes into `CONVERGED`, `INNER_UNDEREXPLORED`, `OUTER_BLINDSPOT`, and `PRIOR_CEILING` regimes, enabling zero-DFT inner curriculum bootstrapping, Murcko-diverse outer active acquisition [15, 16], and zero-retrain GP residual updates.

---

## 2. System Architecture & Consolidated Quantum Benchmark

![Figure 1: Uni-Loom four-layer predictor-agnostic architecture. Layers 1–2 wrap heterogeneous predictors and hot-swappable MLIPs into a declarative critic bus; Layer 3 executes scale-decoupled 3D weaving and Dual-Track Valence Purity; Layer 4 orchestrates heteroscedastic GPAtlas diagnostics, Inner RSI self-bootstrapping, and Outer RSI zero-retrain calibration.](figures/fig1_uniloom_architecture.png)

### 2.1 Layers 1 & 2: Predictor-Agnostic Asset Hub, Residual Calibration, and Declarative Critic Bus

Given a 3D molecule $\mathcal{M} = \{(\mathbf{Z}, \mathbf{R})\}$ with heavy/explicit atom numbers $\mathbf{Z} \in \mathbb{N}^N$ and Cartesian positions $\mathbf{R} \in \mathbb{R}^{N \times 3}$, and a user request specifying $K$ target property channels $\mathbf{t} = (t_1, \dots, t_K) \in (\mathbb{R} \cup \{\varnothing\})^K$, Uni-Loom solves the constrained inverse weaving objective:

$$\max_{\theta} \; \mathbb{E}_{\mathbf{t} \sim p_{\text{curr}},\, \mathcal{M} \sim p_\theta(\cdot \mid \mathbf{t})} \!\Big[\, \mathcal{R}_{\text{critic}}(\mathcal{M}, \mathbf{t}) \,\Big], \qquad \text{s.t.} \quad \text{Clean}_{3\text{D}}(\mathcal{M}) = 1, \quad c_r(\mathcal{M}) \le 0 \;\;\forall r \in \{1,\dots,m\}, \quad \sigma_{\text{cal}}(\mathbf{t}) \le \tau_{\text{cal}} \tag{1}$$

Every external evaluator registers under a uniform `PredictorAsset` interface exposing `AssetCapabilities`: input modality (`3d_coords`, `2d_graph`, `rule_analytical`), output names/units, native epistemic uncertainty flag, and whether coordinate gradients $\nabla_{\mathbf{R}}\hat{y}$ are differentiable. Supported adapters include 3D differentiable neural predictors (`UniMol3DAsset`), black-box 3D/2D assets (`SlotOracleAsset`, `FingerprintQSARAsset`, `CallablePredictorAsset`), and deterministic analytical chemistry operators (`SA_Score`, `QED`, molecular weight, rotatable bonds, strained small rings, maximum ring size, `SMARTS` alerts). Alongside predictors, `PotentialSlot` provides hot-swappable MLIP services (`DPA-4` or `HyFF vivace`), returning zero-step maximum atomic force $F_{\max} = \max_a \|\mathbf{F}_a\|_2$, multi-step `FIRE` geometry relaxation graph preservation, and rigid-body Kabsch deformation `RMSD`.

When an industrial black-box model lacks calibrated uncertainty or errs outside its historical domain, `ResidualCalibratedAsset` wraps it **without altering base parameters** over a compact 20-dimensional molecular descriptor $\phi(\mathcal{M})$ (`16` folded Morgan bit-density channels plus molecular weight, heavy-atom fraction, aromatic-carbon fraction, and heteroatom ratio) by combining a global linear Ridge drift $f_{\text{ridge}}$ with a fixed prior-amplitude Matérn-$3/2$ + `WhiteKernel` Gaussian process residual [15]:

$$\hat{y}_{\text{cal}}(\mathcal{M}) = \hat{y}_{\text{base}}(\mathcal{M}) + f_{\text{ridge}}(\phi(\mathcal{M})) + \delta_{\text{GP}}(\phi(\mathcal{M})), \qquad \sigma_{\text{cal}}(\mathcal{M}) = \sqrt{\max\!\big(\sigma_{\text{GP}}^2(\phi(\mathcal{M})) - \sigma_{\text{noise}}^2,\, 0\big)} \tag{2}$$

Subtracting observation aleatoric noise $\sigma_{\text{noise}}^2$ yields a pure epistemic uncertainty $\sigma_{\text{cal}}(\mathcal{M})$ that drops below the trust threshold $\tau_{\text{cal}}$ selectively in chemical sub-regions covered by new reference measurements. When differentiable 3D assets are bound, Layer 2 also provides `PhysicsConstrainedCoordRefiner`, performing joint gradient descent on equilibrium geometry and target alignment:

$$\min_{\mathbf{R}} \; E_{\text{MLIP}}(\mathbf{Z}, \mathbf{R}) + \lambda \sum_{j=1}^{K} \mathcal{L}_{\text{obj}, j}\!\big(\hat{y}_j(\mathbf{Z}, \mathbf{R}), t_j\big) \tag{3}$$

To prevent reward hacking while allowing arbitrary metric binding, `CompositeCriticBus` reads a declarative `TaskWeavingSpec` YAML schema distinguishing: (i) `objectives` (`point` within $\pm\tau_j$, `interval`, `max`, or `min`), which allocate generator Fourier condition dimensions $K$; (ii) inequality `constraints` (e.g., `SA_Score <= 3.5`, `smarts_alert == 0`), which gate feasibility without consuming condition dimensions; and (iii) `auditors`, supporting both `cross_oracle` disagreement $|\hat{y}_{\text{aud}} - \hat{y}_{\text{rew}}|$ when an independent pretrained oracle is available and `single_oracle_ad` (nearest-neighbor Morgan Tanimoto applicability-domain distance + ensemble variance + $F_{\max}$ physics gate) when only one predictive model exists.

### 2.2 Layer 3: Scale-Decoupled 3D Weaving and Dual-Track Valence Purity

`GeneratorSlot` adapts a 24-layer `Uni-3DAR` backbone [8] via: (i) unbounded random Fourier feature embeddings paired with missingness masks for arbitrary subsets of the $K$ target channels, replacing the saturated `0.08 * tanh` gate; (ii) rank-16 `LoRA` adapters [19] across all 24 transformer blocks and output heads; and (iii) tree-depth annealed Classifier-Free Guidance (CFG) [20] over autoregressive octree/element tokens:

$$\tilde{\ell}_t = \ell_t^{\text{uncond}} + s(d_t)\,\big(\ell_t^{\text{cond}} - \ell_t^{\text{uncond}}\big), \qquad s(d_t) = 1 + (s_0 - 1)\exp(-\gamma_{\text{tree}} d_t) \tag{4}$$

with unguided sub-voxel Cartesian refinement (`s_xyz = 1.0`, $\gamma_{\text{tree}} = 0.3$) so spatial tokens preserve chemical equilibrium bond lengths rather than over-sharpening toward extreme scalar logits.

Online reinforcement learning proceeds through a strict monotonic outcome hierarchy:
$$\text{Tier 4 (Clean, ConstraintsMet, TargetHit)} > \text{Tier 3 (Clean, ConstraintsMet, Miss)} > \text{Tier 2 (Clean, ConstraintViolated)} > \text{Tier 1 (Legal, Exotic Formal Charge)} > \text{Tier 0 (Illegal 3D Point Cloud)} \tag{5}$$
normalized via Group Relative Policy Optimization (GRPO, group size $G=8 \to 16$) [21] with dual-finite KL masking (`KL = NaN` completely eliminated). Critically, Uni-Loom decouples discrete topology policy learning from sub-voxel coordinate regression (`Stage 2.5A + Stage 2.5B`):

$$\nabla_\theta \mathcal{J}_{\text{TopoGRPO}} = \mathbb{E}\!\left[\sum_{a \in \mathcal{T}_{\text{coarse}}} A_i \nabla_\theta \log \pi_\theta(a \mid c_{<a}, \mathbf{t})\right], \qquad \mathcal{L}_{\text{XYZ}}(\phi) = -\mathbb{E}_{\mathcal{M}^*}\!\left[\sum_{a \in \mathcal{T}_{\text{XYZ}}} \log \pi_\phi\!\big(a \mid \operatorname{sg}[\mathbf{h}_a]\big)\right] \tag{6}$$

where Stage 2.5A updates only coarse octree splits and element tokens ($\mathcal{T}_{\text{coarse}}$), whereas Stage 2.5B freezes the transformer trunk (`sg[h_a]`) and trains a dedicated `111,744`-parameter rank-8 XYZ `LoRA` adapter on 40-step MLIP-relaxed equilibrium geometries $\mathcal{M}^*$. Because fine leaf-atom and `0.01 Å` coordinate decoding account for `65%–80%` of octree generation latency, `OctreeProcessCritic` (`octree_process_critic.py`) intercepts KV-cached coarse octree prefixes at `level = 1 → 0` (pooling `h_root`, `h_coarse_mean`, `h_l1_mean`, `h_l1_last` alongside `14` `SO(3)`-invariant coarse voxel envelope descriptors) to predict $(\hat{\mu}_{\text{proc}}, \log\sigma_{\text{proc}}^2, \ell_{\text{legal}}, \ell_{\text{constr}})$ and prune `M_init → M_keep` branches (e.g., `16 → 4` or zero-surrogate `8 → 1`) before leaf decoding. Moreover, `GeneratorSlot` is strictly paradigm-agnostic: `LoomFlowMolGenerator` (`flowmol_slot.py`) hot-swaps continuous-time Markov chain (CTMC) + `SE(3)` flow matching (`PropMolFlow` [6]) behind the identical critic bus and supports online Advantage-Weighted Regression (`Loom-AWR`, `awr_sft`) combining strict native-legality gating, online 20-step MLIP coordinate relaxation with post-relaxation critic re-verification, and curated equilibrium conformer replay.

To eliminate zwitterionic formal-charge artifacts both upstream and downstream without mutating 3D scaffolds, we pair two orthogonal tracks:
- **Track A (Source-side generation suppression):** During autoregressive decoding, an incremental bond-order coordination guard masks halogen/boron/oxygen over-coordination steps in real time (e.g., capping carbonyl oxygen $< 1.28\text{ Å}$ or two-neighbor oxygen, singly bonded halogens, and tricoordinate boron), reinforced by a lightweight `< 30K`-parameter `HeavyAtomValenceHead` attached to `atom_heads[0]` that supervises per-heavy-atom attached hydrogen count and formal charge.
- **Track B (`ValenceReconciler`, zero-heavy-atom-displacement Lewis repair):** Given a decoded 3D point cloud, heavy-atom coordinates $\mathbf{R}_{\text{heavy}}$ are locked verbatim (`RMSD = 0.0000 Å`), benign functional dipoles (nitro $-\text{NO}_2$, *N*-oxide, sulfoxide, azide) are explicitly whitelisted, peripheral hydrogens and integer Lewis bond orders/formal charges are reconciled to reach neutral closed-shell valence, and missing hydrogens are rebuilt in standard geometry followed by batched MLIP hydrogen-only local relaxation.

### 2.3 Layer 4: Generalized Heteroscedastic `GPAtlas` and Grounded RSI

Over arbitrary property coordinates $\mathbf{t}$, Layer 4 tracks a generalized heteroscedastic `GPAtlas` decomposing total conditional uncertainty into generator spread and predictor epistemic uncertainty, estimating calibrated target hit probability within tolerance $\tau$:

$$\sigma_{\text{total}}^2(\mathbf{t}) = \sigma_{\text{gen}}^2(\mathbf{t}) + \sigma_{\text{cal}}^2(\mathbf{t}), \qquad P_{\text{hit}}(\mathbf{t}) \approx \int_{\mathbf{t}-\tau}^{\mathbf{t}+\tau} \mathcal{N}\!\big(\mathbf{y};\, \hat{\mu}_{\text{gen}}(\mathbf{t}),\, \operatorname{diag}(\sigma_{\text{total}}^2(\mathbf{t}))\big)\,\mathrm{d}\mathbf{y} \tag{7}$$

Before generation starts, a **Pre-Generation Gatekeeper** evaluates $P_{\text{hit}}(\mathbf{t})$ and $\sigma_{\text{cal}}(\mathbf{t})$: untrusted queries trigger `rerouted_warning` with a nearest reachable alternative target rather than silently fabricating molecules in blindspots. During RSI cycles, `RSIDiagnoser` assigns every grid bin to one of four regimes: (i) `PRIOR_CEILING` (low prior density + large generative pull $|\hat{\mu}_{\text{gen}}(\mathbf{t}) - \mathbf{t}|$; RL weight = `0%`); (ii) `OUTER_BLINDSPOT` ($\sigma_{\text{cal}}(\mathbf{t}) > \tau_{\text{cal}}$; inner RL weight = `0%`, triggers outer active acquisition); (iii) `INNER_UNDEREXPLORED` ($\sigma_{\text{cal}}(\mathbf{t}) \le \tau_{\text{cal}}$, hit rate unsaturated; `80%` deficit-weighted inner RL budget); and (iv) `CONVERGED` (`20%` anti-forgetting anchor replay). `InnerRSIRunner` executes zero-DFT curriculum GRPO gated by a strict non-regression acceptance check. In parallel, `OuterRSIAcquisition` exports scaffold-diverse high-uncertainty `.sdf` batches maximizing:

$$u(\mathcal{M}) = \sigma_{\text{cal}}(\mathcal{M}) \cdot \mathbb{I}\!\big[\text{Clean}(\mathcal{M}) \land \text{ConstraintsMet}(\mathcal{M})\big] \cdot d_{\text{Murcko}}\!\big(\mathcal{M}, \mathcal{S}_{\text{selected}}\big) \tag{8}$$

Upon receiving external DFT or wet-lab values [17, 18], `IncrementalCalibrator` updates the `ResidualCalibratedAsset` GP posterior without modifying generator or base-predictor weights, shrinking $\sigma_{\text{cal}}(\mathbf{t})$ below $\tau_{\text{cal}}$ and causally unlocking the formerly frozen blindspot for the next RSI round.

### 2.4 Consolidated Quantum Benchmark and Inference Throughput

We consolidated historical `PBE0/def2-TZVP/DF` quantum calculations under strict canonical-skeleton anti-leakage deduplication against all validation and test splits, assembling `4,737` leak-free generated 3D DFT molecules (`3,949` train / `398` val / `390` test) merged with `43,858` scaffold-disjoint `OE62` training structures [1] (`47,807` total conformers; up to `80` atoms across `11` heavy elements plus hydrogen: `H, B, C, N, O, F, Si, P, S, Cl, Br, I`). On the `390` held-out generated DFT test structures, frozen `Uni-Mol2 1.1B` records `HOMO/LUMO MAE = 0.720 / 0.778 eV` (`RMSE = 1.106 / 1.253 eV`), whereas our DFT-calibrated `DPA-4 AuditOracle` achieves `MAE = 0.186 / 0.183 eV` (`RMSE = 0.268 / 0.268 eV`), validating strict physical asymmetry between reward and blind audit models. For batched MLIP evaluation, `float64` spatial packing cuts batch-32 force drift from `7.09 × 10^-3` down to `7.89 × 10^-6 eV/Å`.

**Table 1: Single-GPU inference and training throughput (`NVIDIA A800-SXM4-80GB`, `256` molecules/arm).**

| Pipeline Configuration | Mode | Precision | Batch | Peak VRAM (GB) | Throughput (mol/s) | Speedup vs Legacy Loop |
|---|---|---|---:|---:|---:|---:|
| Legacy Variable-Length Python Loop | Sampling | FP32 | 32 | — | 4.46 | 1.00× |
| Vectorized Row-Stream Baseline (`CFG=1.0`) | Sampling | FP32 | 32 | — | 6.41 | 1.44× |
| Uni-Loom `BF16` (`CFG=1.0`) | Sampling | BF16 | 32 | 1.67 | 6.94 | 1.56× |
| Uni-Loom `BF16` (`CFG=1.0`) | Sampling | BF16 | 64 | 1.92 | 11.75 | 2.63× |
| Uni-Loom `BF16` (`CFG=1.0`) | Sampling | BF16 | 128 | 2.39 | 20.19 | 4.53× |
| Uni-Loom `BF16` (`CFG=1.0`) | Sampling | BF16 | 256 | 3.23 | **32.52** | **7.29×** |
| Uni-Loom `BF16` Dual-Branch `CFG=1.75–2.0` + 3D Guard | Sampling | BF16 | 128–256 | 3.36 | 22.25–29.51 | 4.99–6.62× |
| Legacy SFT (No Contrastive Pass) | Train Step | FP32 | 32 | — | 72.85 | 1.00× |
| Uni-Loom `LoRA-16` + 2× Contrastive Pass | Train Step | BF16 | 32 | — | **80.92** | **1.11×** |

---

## 3. `OE62` Orbital Weaving, Scale Decoupling, and Anti-Reward-Hacking

### 3.1 Stagewise Evolution on Held-Out `OE62 HOMO`

All `OE62` models are evaluated under identical held-out target quantiles (`P10, P25, P35, P50, P65, P75, P90`; seeds `[24411, 24412]`; `7 × 256 = 1,792` 3D proposals; target hit window `±0.25 eV`). **Table 2** and **Figure 2** trace every pipeline stage from legacy frozen conditioners through Stage 2.5A+2.5B.

**Table 2: End-to-end `OE62 HOMO` benchmark (`1,792` proposals, `±0.25 eV` window).**

| Stage / Method | Hits | Hit Rate | Clean+Hit (Count, %) | 3D Legality | Slope | Cond. Std (eV) | Cond. MAE (eV) | Relaxed Audit MAE (eV) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy `Frozen SFT` | 371 | 20.70% | — | 72.1% | 0.409 | 0.570 | 0.521 | — |
| Legacy `Margin + GRPO-1024` | 441 | 24.61% | — | 70.0% | 0.535 | 0.542 | 0.451 | — |
| `Stage 0` (`LoRA-16 + CFG=1.75`) | 470 | 26.23% | 184 (10.3%) | 59.6% | 0.934 | 0.420 | 0.347 | 1.536 |
| `Stage 1` (`DPA-4 Hindsight`) | 528 | 29.46% | 286 (16.0%) | 63.6% | 0.933 | **0.418** | **0.339** | 1.137 |
| `v1.0 Stage 2` (`GRPO-1024, CFG=2.0`) | 620 | 34.60% | 449 (25.1%) | 80.5% | 1.045 | 0.439 | 0.364 | 0.757 |
| `v1.1` (`GRPO-1024 + Guard + XYZ Distill`) | 625 | 34.88% | 450 (25.1%) | 80.1% | 1.037 | 0.427 | 0.353 | 0.825 |
| `v1.1 Stage 2.5A` (`CFG=1.75`) | 604 | 33.71% | 436 (24.3%) | 84.6% | **0.986** | 0.480 | 0.403 | **0.744** |
| `v1.1 Stage 2.5A` (`CFG=2.0`) | **631** | **35.21%** | **451 (25.2%)** | 83.5% | 1.100 | 0.471 | 0.393 | 0.755 |
| `v1.1 Stage 2.5A+2.5B` (`CFG=1.75`) | 615 | 34.32% | 447 (24.9%) | **85.4%** | **0.985** | 0.475 | 0.400 | 0.903 |

Removing the bounded `0.08 * tanh` bottleneck immediately lifts conditional calibration slope from `0.409–0.535` to `0.934` in `Stage 0`. Subsequent `DPA-4` hindsight relabeling (`Stage 1`) increases clean hits from `184` to `286` while cutting relaxed `Audit MAE` from `1.536` to `1.137 eV`. Online `GRPO` (`v1.0 Stage 2`) elevates hits to `620 (34.60%)`, clean hits to `449 (25.1%)`, and `Audit MAE` down to `0.757 eV`. Adding real-time 3D guards and `TopoGRPO` (`v1.1 Stage 2.5A`) reaches **`631/1,792` (`35.21%`)**, **`451` clean hits (`621` novel unique graphs)**, **`84.6%` raw legality** (`85.4%` with Stage 2.5B), and the lowest relaxed independent `Audit MAE` (`0.744 eV`, `-0.792 eV` vs Stage 0).

![Figure 2: Stagewise progression on OE62 HOMO (`1,792` proposals) across conditional calibration slope, hit rate, clean non-zwitterionic hits, 3D legality, and independent audit MAE.](figures/fig2_stage_progression.png)

### 3.2 Anti-Reward-Hacking Verification and Zero-Code Potential Hot-Swap

Across `0 → 1,184` online `GRPO` groups (**Figure 3**), every structural and blind electronic metric improves monotonically alongside reward-model hit yield: zero-step median maximum force drops from `7.41` to `5.09 eV/Å`, MLIP relaxation graph preservation rises from `75.8%` to `90.6%`, and independent relaxed `DPA-4 Audit MAE` drops strictly downward `1.137 → 1.010 → 1.000 → 0.876 → 0.757 → 0.755 eV` (`0.744 eV` at `CFG=1.75`). Replacing `DPA-4` with `HyFF vivace` at the identical `512`-group online budget requires zero model-code edits and matches hit efficiency (`32.09%`, `575/1,792`) while delivering lower initial atomic forces (`5.27 vs 5.80 eV/Å`), higher relaxation graph preservation (`88.3% vs 81.6%`), and lower independent audit MAE (`0.939 vs 1.000 eV`).

![Figure 3: Anti-reward-hacking audit curve across 0–1,184 online GRPO groups and zero-code PotentialSlot hot-swap comparison (`DPA-4` vs `HyFF vivace` at 512 groups).](figures/fig3_audit_and_hotswap.png)

### 3.3 Scale-Decoupling Mechanistic Ablation

**Table 3** demonstrates why coarse topological/element policy learning and `0.01 Å` coordinate learning must remain decoupled. When scalar property advantages propagate directly back through `3×` more numerous XYZ coordinate tokens alongside teacher-forced coordinate relaxation (`Control: Coupled RL`), raw legality rises to `88.62%` (`F_max = 4.19 eV/Å`, `RMSD = 0.0495 Å`), but high-variance credit assignment across spatial tokens degrades within-target dispersion `σ_in` from `0.439` to `0.534 eV` and cuts target hits from `620` down to `526`. Decoupling (`Stage 2.5A + 2.5B`) avoids trunk corruption, uniting `631 (451)` hits / `0.744 eV` Audit MAE (`Stage 2.5A`) with `85.38%` legality / `0.0533 Å` Kabsch RMSD (`Stage 2.5A+2.5B`).

**Table 3: Scale-decoupling ablation on `OE62 HOMO` (`1,792` proposals).**

| Ablation Variant | Hits (Clean) | 3D Legality | Exotic `Q` Rate | Cond. Std (eV) | $F_{\max}$ (eV/Å) | Kabsch RMSD (Å) | Relaxed Audit MAE (eV) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v1.0` Baseline (No Guard, Frozen `XYZ`) | 620 (449) | 80.47% | 25.7% | 0.439 | 5.09 | 0.0613 | 0.757 |
| `+ Zero-Shot 3D Guard + Tree CFG` | 622 (444) | 81.75% | 25.5% | 0.436 | 5.12 | 0.0586 | 0.811 |
| `+ Detached 0.01 Å XYZ Distill` | 625 (450) | 81.81% | 25.8% | **0.427** | 5.19 | 0.0581 | 0.825 |
| `Stage 2.5A` (`TopoGRPO`) | **631 (451)** | 84.60% | 25.3% | 0.471 | 5.10 | 0.0559 | **0.744** |
| `Stage 2.5A + 2.5B` (Detached `XYZ` LoRA) | 615 (447) | 85.38% | 26.1% | 0.475 | 5.18 | 0.0533 | 0.903 |
| `Control: Coupled RL` (`XYZ` inside RL) | 526 (402) | **88.62%** | **17.2%** | 0.534 | **4.19** | **0.0495** | 0.786 |

### 3.4 Joint `[HOMO, LUMO]` Weaving and Orthogonalized `1D/2D GPAtlas`

Within a single multi-property `v1.1` conditional model, marginal `LUMO` generation improves substantially over `v1.0`: 3D legality jumps from `75.3%` to `87.33%`, exotic charge rate falls from `27.1%` to `16.2%`, median zero-step force decreases from `5.04` to `3.79 eV/Å`, MLIP graph preservation reaches `94.1%`, Kabsch RMSD contracts to `0.0469 Å`, and calibration slope reaches `1.002`. Evaluated over a `7 × 7 = 49`-cell joint `(HOMO, LUMO)` grid (`1,764` clean relaxed molecules), the orthogonalized `GPAtlas` (**Figures 4–5**) maps: (i) in `1D HOMO` (**Figure 4**), an equal-scale square reachability calibration plot over `[-9.5, -4.5] eV` (**Panel A**) paired with an independent viability-vs-prior plot contrasting the amber trusted reachable interval `[-8.85, -5.77] eV` against the indigo `OE62` source prior KDE (**Panel B**); (ii) `2D` joint clean hit yields peaking at `22.0%` alongside drift vectors $\hat{\mu}(\mathbf{t}) - \mathbf{t}$; and (iii) an empirical cross-foundation predictor divergence surface $\sigma_{\text{cal,2D}}(t_{\text{H}}, t_{\text{L}})$ ranging smoothly from `0.83 eV` in well-populated wide-gap regions to `1.14 eV` in narrow-gap donor–acceptor corners.

![Figure 4: Univariate 1D HOMO heteroscedastic GPAtlas split into two independent panels—Panel A: 1:1 equal-scale square calibration across [-9.5, -4.5] eV; Panel B: standalone target-HOMO axis contrasting amber trusted reachable bounds [-8.85, -5.77] eV against indigo OE62 source prior density.](figures/fig4_gp_atlas_1d_homo.png)

![Figure 5: Orthogonalized 2D (HOMO, LUMO) navigation atlas showing empirical clean joint yield (`22.0%` peak) with drift vectors (left) and measured cross-oracle disagreement surface (`0.83–1.14 eV`) over 49 grid cells (`1,764` molecules, right).](figures/fig5_gp_atlas_2d_joint.png)

### 3.4 Stratified 2D GP-Atlas Live `PBE0/def2-TZVP` Quantum Anchor

To test whether cross-oracle disagreement hotspots in **Figure 5** flag genuine quantum-mechanical breakdown rather than surrogate noise, we evaluated a stratified sample of `80` unique DPA-4-relaxed clean molecules (`5` per cell across the `8` lowest-divergence cells, `cal_rmse_2d = 0.620–0.764 eV`, and the `8` highest-divergence cells, `1.239–1.434 eV`) via live `GPU4PySCF` single-point `PBE0/def2-TZVP/DF` (`def2-universal-jkfit`, grid level 3, SCF threshold `1e-8 Ha`) calculations alongside full `PBE-D3BJ/def2-TZVP/DF` `geomeTRIC` geometry optimizations on `16` cell representatives (`1` per cell; **Table 4**) [10,13,17,18]. All `80/80` single points (`100.0%`) converged via DIIS with zero Newton fallbacks. Against true `PBE0`, calibrated `DPA-4` [13] tracks quantum orbital energies tightly (`HOMO MAE = 0.1893 eV`, `r = 0.9822`, `ρ = 0.9767`; `LUMO MAE = 0.2879 eV`, `r = 0.9755`, `ρ = 0.9712`; joint 2D `MAE = 0.2386 eV`), whereas frozen `Uni-Mol2` [10] exhibits `3–4×` higher error (`HOMO MAE = 0.7235 eV`, `r = 0.7110`; `LUMO MAE = 0.8677 eV`, `r = 0.7449`; joint 2D `MAE = 0.7956 eV`). Crucially, moving from Low-Divergence (`n=40`) to High-Divergence (`n=40`) cells pushes empirical `|Uni-Mol2 - DPA-4|` HOMO discrepancy from `0.5180` to `0.9512 eV`, triggering a `+78.0%` jump in `Uni-Mol2` `PBE0` HOMO MAE (`0.5205 → 0.9265 eV`, RMSE `0.7209 → 1.2624 eV`) and a drop in Spearman rank correlation from `0.7964` to `0.4914` (`r: 0.7569 → 0.5573`), while LUMO error tracks per-channel empirical disagreement across extreme-LUMO Low-Divergence cells (`0.8366` vs. `0.6094 eV`). Moreover, `16/16` (`100.0%`) full DFT relaxations converge, preserving molecular bond graphs in `15/16` cases (`93.75%`) with tight conformational agreement (`0.0540 Å` median / `0.1578 Å` mean heavy-atom Kabsch RMSD; `0.1582 / 0.2698 Å` all-atom) and modest adiabatic `PBE0` orbital shifts (`0.3056 eV` HOMO, `0.1785 eV` LUMO).

**Table 4: Stratified 2D `(HOMO, LUMO)` GP-Atlas live quantum chemical anchor verification (`PBE0/def2-TZVP/DF`).**

| Cohort / Stratum | Oracle vs. `PBE0` | Joint 2D MAE (eV) | HOMO MAE / RMSE (eV) | HOMO `r / ρ` | LUMO MAE / RMSE (eV) | LUMO `r / ρ` |
|---|---|---:|---:|---:|---:|---:|
| Overall (`n=80`, SCF `100%`, emp diff `0.8133 eV`) | **DPA-4** | **0.2386** | **0.1893 / 0.2497** | **0.9822 / 0.9767** | **0.2879 / 0.4005** | **0.9755 / 0.9712** |
| Overall (`n=80`, emp diff `0.8133 eV`) | Uni-Mol2 | 0.7956 | 0.7235 / 1.0279 | 0.7110 / 0.7292 | 0.8677 / 1.1631 | 0.7449 / 0.8120 |
| Low-Divergence (`n=40`, emp diff `0.7624 eV`) | **DPA-4** | **0.2011** | **0.1498 / 0.2038** | **0.9798 / 0.9702** | **0.2523 / 0.3542** | **0.9787 / 0.9341** |
| Low-Divergence (`n=40`, emp diff `0.7624 eV`) | Uni-Mol2 | 0.7639 | 0.5205 / 0.7209 | 0.7569 / 0.7964 | 1.0073 / 1.3140 | 0.6578 / 0.8139 |
| High-Divergence (`n=40`, emp diff `0.8641 eV`) | **DPA-4** | **0.2761** | **0.2288 / 0.2883** | **0.9778 / 0.9728** | **0.3235 / 0.4420** | **0.9661 / 0.9722** |
| High-Divergence (`n=40`, emp diff `0.8641 eV`) | Uni-Mol2 | 0.8273 | 0.9265 / 1.2624 | 0.5573 / 0.4914 | 0.7281 / 0.9895 | 0.7751 / 0.7587 |

*Full `PBE-D3BJ/def2-TZVP/DF` DFT Relaxation (`n=16`, `1/cell`):* `16/16 (100.0%)` converged; `15/16 (93.75%)` bond graphs preserved; heavy-atom Kabsch RMSD `0.0540 Å` median (`0.1578 Å` mean), all-atom RMSD `0.1582 Å` median (`0.2698 Å` mean); mean adiabatic `PBE0` HOMO/LUMO shifts `0.3056 / 0.1785 eV`.

---

## 4. Dual-Track Valence Purity and Multi-Asset Synthesizable Weaving (`v2.0`)

### 4.1 Experimental Protocol and Dual-Track Honest Reporting

To evaluate arbitrary predictor-agnostic constraints without altering foundation property weights, we configured `TaskWeavingSpec` (`configs/tasks/oe62_orbital_synthesizable.yaml`) combining frozen external `ext_unimol2` (`HOMO ±0.25 eV`) with three analytical chemistry inequality constraints consuming zero conditioner inputs (`SA_Score <= 3.5`, `smarts_alert == 0`, `MolWt <= 550 Da`) and a strictly blinded `audit_dpa4` examiner. **Table 5** reports identically sampled `1,792`-molecule cohorts side-by-side under both **Raw Point-Cloud** (zero post-processing) and **Valence-Reconciled** (`0.0000 Å` heavy-atom freeze) evaluation.

**Table 5: Dual-Track valence purity and declarative synthesizable weaving (`SA_Score ≤ 3.5 & SMARTS = 0`, `1,792` proposals).**

| Model & Evaluation Track | 3D Legality (Count) | Exotic `Q` Rate | Median `SA_Score` | Constraints Met (%) | $F_{\max}$ (eV/Å) | Synth Clean Hits (%) | Blind Audit Hits (%) | `Audit - Reward` Gap (eV) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pre-Align `v1.1` (**Raw Point-Cloud**) | 81.92% (1,468) | 25.75% | 4.87 | 21.05% | 5.07 | 118 (6.58%) | 250 (13.95%) | 0.655 |
| Pre-Align `v1.1` (**Valence-Reconciled**) | 89.23% (1,599) | **0.00%** | 4.70 | 20.95% | 5.89 | 127 (7.09%) | 292 (16.29%) | 0.555 |
| `Uni-Loom v2.0` (**Raw Point-Cloud**) | **95.20% (1,706)** | **8.26%** | **3.04** | **58.32%** | **1.79** | **184 (10.27%)** | **338 (18.86%)** | **0.190** |
| `Uni-Loom v2.0` (**Valence-Reconciled**) | **98.49% (1,765)** | **0.00%** | **3.04** | **58.87%** | 1.91 | **196 (10.94%)** | **360 (20.09%)** | **0.143** |

### 4.2 Causal Findings and Reward-Gap Collapse

1. **Source-side Track A dominates raw improvements.** Without any post-hoc reconciliation, Track A raises raw point-cloud 3D legality `+13.28 pp` (`81.92% → 95.20%`), cuts broken skeletons `110 → 18` and over-coordinated atoms `66 → 5`, lifts natively clean raw molecules from `1,090` to `1,565`, and cuts raw exotic charges `3.1×` (`25.75% → 8.26%`, later reaching `3.61%` in Round 2 RSI).
2. **Track B eliminates residual zwitterions with zero scaffold distortion.** Across the remaining `200` border cases (`60` peripheral H-count mismatches and `140` Lewis resonance reassignments), `ValenceReconciler` freezes heavy atoms identically (`max heavy-atom RMSD = 0.0000 Å`), pushing deliverable legality to `98.49% (1,765/1,792)` and `0.00%` exotic charges.
3. **Synthesizability constraints collapse cross-foundation reward hacking.** Enforcing `SA_Score <= 3.5` admits `25.5%` (`614/2,404`) of replay buffers into training, drives generated scaffolds toward flat, low-strain conjugated motifs (`SA_Score 4.87 → 3.04`; constrained legality `21.05% → 58.87%`; median $F_{\max}$ `5.07 → 1.79 eV/Å`), expands synthesizable clean hits `+66.1%` (`118 → 196`, encompassing `137` unique and `120` novel hit graphs; calibration slope `1.028`; `-0.43 eV` trade-off shift against unconstrained yields `664 → 378`), increases strictly blind `DPA-4 Audit` hits `+44.0%` (`250 → 360`), and compresses the cross-foundation `Audit - Reward MAE` gap by **`4.6×` (`0.655 → 0.143 eV`)**.

![Figure 6: Dual-Track elimination of exotic formal charges and collapse of the Audit–Reward MAE divergence gap (`0.655 → 0.143 eV`) under declarative `SA_Score ≤ 3.5` and SMARTS constraints.](figures/fig6_zwitterion_and_synthesizability.png)

---

## 5. Two-Round Grounded Recursive Self-Improvement Closed Loop (`v2.0`)

### 5.1 Four-Way Leakage-Free Industrial Blindspot Protocol

We constructed a strictly segregated industrial simulation (`1,792` proposals/round): (1) a restricted historical 5-fold enterprise QSAR asset wrapped in `ResidualCalibratedAsset` trained exclusively on `3,086` molecules with `HOMO > -6.92 eV` (leaving deep acceptors `HOMO ≤ -6.92 eV` as an explicit OOD blindspot), monitored only by `single_oracle_ad` and `SA_Score <= 3.8 & SMARTS = 0` constraints (`τ_cal = 0.35 eV`); (2) an isolated external `PBE0/def2-TZVP` pool (`73` molecules) supplying true DFT labels exclusively upon active query selection without editing any base predictor weights; (3) an untouched blind deep-DFT test set (`73` molecules); and (4) an independent blind `DPA-4 AuditOracle` scoring every generated molecule across mid-range under-explored (`P35–P90`) and deep blindspot (`P10–P25`) targets.

**Table 6: Two-round Grounded RSI closed-loop metrics (`1,792` proposals/round, `τ_cal = 0.35 eV`).**

| Cycle Stage | Zones (`C/I/B/P`) | `P10` $\sigma_{\text{cal}}$ (eV) | Legality (`Raw / Rec`) | Raw Exotic `Q` | Constraints Met | Mid (`P35–P90`) Audit Hit | Deep (`P10–P25`) Audit Hit | Blind `DPA-4` Clean Hits |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Round 0** (Initial Diagnosis) | `0 / 6 / 1 / 1` | 0.490 | 66.0% / 76.1% | 38.88% | 13.27% | 2.11% | 4.88% | 52 (2.90%) |
| **Round 1A** (Inner RSI, `0` DFT) | `3 / 3 / 1 / 1` | 0.483 | 94.9% / 98.4% | 7.35% | 57.06% | 8.59% (**4.07×**) | 5.47% *(protected)* | 138 (7.70%, **2.65×**) |
| **Round 1B** (Outer `64` DFT, `12` Scaffolds) | **`2 / 5 / 0 / 1`** | **0.291** | *0 base weight edits* | *Blind test DFT (`n=73`) MAE:* | *0.990 → 0.655 eV* | *(-33.8%; uncal 1.187 eV)* | — | **Unlocks `P10–P25`** |
| **Round 2** (Unlocked Inner RSI) | `3 / 4 / 0 / 1` | **0.285** | **97.4% / 99.3%** | **3.61%** | **75.04%** | **14.69% (6.96×)** | **8.01% (+64.1%)** | **229 (12.78%, 4.40×)** |

### 5.2 Round-by-Round Causal Verification

1. **Round 0 (Diagnosis & Pre-Generation Rerouting):** `RSIDiagnoser` classifies the extreme `-8.55 eV` query as `PRIOR_CEILING` (prior density $\rho_{\text{src}} = 0.013$, pull bias `+2.06 eV`) and `P10 = -7.45 eV` as `OUTER_BLINDSPOT` ($\sigma_{\text{cal}} = 0.490 > 0.35\text{ eV}$), locking both to `0%` RL budget while routing `recommend_target(-7.446)` to `rerouted_warning` (`P_hit = 5.6%`).
2. **Round 1A (Zero-DFT Inner RSI):** Concentrating `100%` of curriculum updates on the `6` trusted `INNER_UNDEREXPLORED` bins (passing the non-regression gate with surrogate constrained yield `2.57% → 19.03%`) lifts mid-range blind `DPA-4` clean hits `4.07×` (`2.11% → 8.59%`, promoting `3` bins to `CONVERGED`) and total blind clean hits `2.65×` (`52 → 138`) while leaving the frozen blindspot completely uncorrupted (`4.88% → 5.47%`).
3. **Round 1B (Outer Active Acquisition & Zero-Retrain Calibration):** `OuterRSIAcquisition` exports `64` Murcko-diverse candidates (`12` scaffolds, $\bar{\sigma}_{\text{cal}} = 0.453\text{ eV}$; uncalibrated `MAE = 1.187 eV`). With zero base-predictor weight updates, `IncrementalCalibrator` reduces error on the untouched `n = 73` blind deep-DFT test set by **`33.8%` (`0.990 → 0.655 eV`)**, compresses `P10` $\sigma_{\text{cal}}$ from `0.483` to `0.291 eV` (`< 0.35 eV`), and reduces `OUTER_BLINDSPOT` bins from `1` to **`0`**.
4. **Round 2 (Unlocked Inner RSI):** With deep acceptors unlocked, Round 2 boosts deep-blindspot (`P10–P25`) blind `DPA-4` clean hits `+64.1%` over Round 0 (`+46.4%` over Round 1A) to `8.01%` (deep Audit MAE `1.030 → 0.955 eV`; `P10/P25` calibrated hit rates rising `0% → 78.1% / 58.2%`, while shallow bound `-5.85` settles near `-6.86 eV`), drives mid-range clean hits to `14.69%` (`6.96×`), and achieves `97.38%` raw / `99.27%` reconciled legality, `3.61%` raw exotic charge, `0.93 eV/Å` median atomic force, `75.04%` constraint satisfaction, and `229/1,792 (12.78%, 4.40×)` overall blind clean hits. Re-querying `recommend_target(-7.446)` transitions to `trusted` (`P_hit = 44.85%`, $\sigma_{\text{cal}} = 0.285\text{ eV}$).

![Figure 7: Two-round Grounded RSI causal flywheel demonstrating outer-loop DFT calibration (`-33.8%` blind DFT MAE), blindspot unlocking (`σ_cal: 0.483 → 0.285 eV`), and monotonic blind DPA-4 clean hit growth (`4.40×`).](figures/fig7_grounded_rsi_closed_loop.png)

---

## 6. Public QM9 Multi-Property Conditional Generation and Pareto Scaling

To assess portability outside our internal OE62/DFT pipeline, isolate conditioning covariates, and test generator-agnostic critic scaling, we benchmarked **Uni-Loom** initialized from unconditional `Uni-3DAR` (`qm9.pt`) [8] alongside official `PropMolFlow` [6] checkpoints on the leakage-free `50k/50k` disjoint QM9 split [6,22] (`44,453` training and `1,500` development conformers matched from `train_rQM_v0`, verified at `0` training and `0` holdout overlap with the `50,000`-molecule `gvp_rQM_v0` oracle split). All hindsight relabeling, RL rewards, blind dev sweeps (`N = 250`), and test-time `Best-of-M` (`M ∈ {1, 2, 4, 8, 16}`) selections use exclusively an internal `Uni-Mol` [11] surrogate trained strictly on `train_rQM_v0` (`18` epochs in `214.25 s`, dev `std_mae = 0.0432`), scoring only the final selected Top-1 molecule per target (`N = 1,000`/property) once on frozen `PropMolFlow` `GVPRegressor` oracles on a single NVIDIA A800-SXM4-80GB GPU (**Table 7**).

Four **Uni-Loom** training regimes probe task interference and covariate parity under matched decoding (`cfg_tree_decay = 0.25`, Track A `geometric_guard = True`): (i) one unified scalar-only `K=6` SFT checkpoint (`31.60M / 120.74M` trainable params, `6,000` steps at `220.27 mol/s`, `0.968 GPU-h` total); (ii) six dedicated scalar-only `K=1` SFT checkpoints (`22.47M / 111.61M` params/model, `3.076 GPU-h` total at `194.18 mol/s`); (iii) six scalar-only `K=1` Full checkpoints adding Hindsight and validity-first Stage 2.5A `TopoGRPO` (`4.500 GPU-h` total; cold `bs = 128 / 250` raw decoding `46.16 / 75.70 mol/s`); and (iv) **Scheme 2**, six covariate-matched `(t, N_atoms)` checkpoints (`27.34M / 116.48M` params/model, rank-32 `LoRA`, `4.626 GPU-h` SFT at `207.52 mol/s`, `6.309 GPU-h` Full; cold `bs = 128 / 250` decoding `42.49 / 69.84 mol/s`) matching `EDM`/`GeoLDM`/`PropMolFlow`'s paired `(t, N_atoms)` conditioning in a canonical `SO(3)` polarity-aligned inertial PCA frame with fixed-`N_atoms` contrastive margin, size-anchored CFG (`uncond_active_mask`), and tight point-regression `TopoGRPO`.

Holdout sanity checks (`N = 1,000`, `973` evaluable) separate the `GVP` Oracle Lower-Bound (`α = 0.5544 Bohr^3`, `Δε = 190.08 meV`, `ε_HOMO = 83.78 meV`, `ε_LUMO = 135.89 meV`, `μ = 0.1422 D`, `C_v = 0.1726 cal/(mol·K)`) from the Random Baseline (`9.3116`, `1,423.78`, `653.09`, `1,481.53`, `1.5177`, `4.5040`). Direct A800 auditing of all six official `PropMolFlow` checkpoints (`7.09M` params/model, `4,370,124` steps at `996.25 mol/s`, `155.97 GPU-h` total) confirms `28.73–28.88 mol/s` single-shot (`M = 1`) throughput (`100` ODE steps, max `bs = 128`), `95.0%–98.9%` native sanitization (`0.00%–1.11%` exotic `Q`), and `85.7%–93.0%` raw `→ 96.1%–98.5%` coordinate-reconciled legality (`0.00%` exotic `Q`).

Across single-shot (`M = 1`) and critic-guided (**Scheme 1** / **Scheme 2**, `M = 1 → 16`) regimes, three causal findings emerge:

1. **Covariate Parity and Canonical `SO(3)` Polarity Alignment (`M = 1`):** Moving from `1 × K=6` SFT to `6 × K=1` SFT lowers single-shot raw MAE on `5/6` tasks (`α: 5.4282 → 4.3330`), while **Scheme 2** shrinks atom-count error `3×` (`|ΔN|: 3.12 → 0.98` raw, `0.59` at `M = 16`), cuts extensive `C_v` MAE by `-23.2%` (`3.1136 → 2.3902` reconciled, `2.0636` raw), and lifts single-shot dipole `μ` Pearson `r` `3.9×` (`0.076 → 0.298`) by eliminating `4π` rotational frame smearing. Nevertheless, single-pass discrete octree sampling (`80.3–92.6 mol/s` amortized effective throughput) retains higher intrinsic single-sample dispersion than `100`-step continuous ODE integration.
2. **High-Throughput Critic-Guided Weaving (Uni-Loom `M = 1 → 16`):** Activating **Uni-Loom**'s native zero-leakage `CriticBus` + `ValenceReconciler` collapses **Uni-Loom** error by `55%–70%` while guaranteeing `100.0%` closed-shell legality and `0.00%` exotic charges across all six properties at `M ≥ 8`. With zero retraining on the `0.97 GPU-h` unified `1 × K=6` checkpoint (**Scheme 1**), `M = 4` (`21.44 mol/s`, matching `PropMolFlow` `M = 1` wall-clock speed) already surpasses published `EDM` and `GeoLDM` on `Δε` (`602.05 meV`), `ε_HOMO` (`289.92 meV`), `ε_LUMO` (`543.32 meV`), and `μ` (`0.7218 D`); at `M = 16` (`5.36 mol/s`, `161×` less training compute than `PropMolFlow`), `1 × K=6` reaches `Δε = 374.63 meV`, `ε_HOMO = 199.19 meV`, `ε_LUMO = 341.31 meV`, and `μ = 0.4740 D` (`α = 2.9810 Bohr^3` and `C_v = 1.8332` under `(t, N_atoms)` selection), outperforming single-shot `PropMolFlow` on `Δε`, `ε_HOMO`, and `μ`. **Scheme 2** (`6.31 GPU-h`, `5.31 mol/s` at `M = 16`) further sharpens `Δε` to `358.16 meV` (`348.85` raw) and cuts extensive `C_v` MAE by `-31.2%` to `1.2619 cal/(mol·K)` (`1.1496` raw; `α = 2.7457` raw / `3.0671` reconciled).
3. **Generator-Agnostic Synergy (`PropMolFlow` + Uni-Loom `CriticBus`):** Mounting the exact same zero-leakage `train_rQM_v0` `Uni-Mol` critic and `ValenceReconciler` onto `PropMolFlow` slashes its own errors by `59%–75%` across `M = 1 → 16` (`1.80 mol/s` native, `1.75 mol/s` coordinate-reconciled), driving every property to the `GVP` holdout noise floor (`α = 0.4566 Bohr^3`, `Δε = 145.92 meV`, `ε_HOMO = 79.42 meV`, `ε_LUMO = 113.40 meV`, `μ = 0.1587 D`, `C_v = 0.1771`; `r = 0.968–0.998`) with `100.0%` legality and `0.00%` exotic charges—confirming that generator backbone choice and **Uni-Loom**'s weaving layer are strictly orthogonal and complementary.

**Table 7: Public QM9 conditional generation and zero-leakage `Best-of-M` Pareto benchmark on the `50k/50k` disjoint split (`N = 1,000` targets/property, single NVIDIA A800-SXM4-80GB).**

| Method / Weaving Protocol | Models | `M` | `α` (`Bohr^3`) | `Δε` (meV) | `ε_HOMO` (meV) | `ε_LUMO` (meV) | `μ` (D) | `C_v` (`cal/mol·K`) | Legal (%) / Exotic `Q` (%) | Train (`mol/s \| GPU-h`) | Eff. Out (`mol/s`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GVP Oracle Lower-Bound (`n=973`) | — | — | 0.5544 | 190.08 | 83.78 | 135.89 | 0.1422 | 0.1726 | 94.9 → 97.3 / — | — | — |
| Random Target Baseline | — | — | 9.3116 | 1423.78 | 653.09 | 1481.53 | 1.5177 | 4.5040 | — | — | — |
| EDM (published) [4] | 6 | 1 | 2.76 | 655.0 | 356.0 | 584.0 | 1.111 | 1.101 | —* | — | — |
| GeoLDM (published) [5] | 6 | 1 | 2.37 | 587.0 | 340.0 | 522.0 | 1.108 | 1.025 | —* | — | — |
| PropMolFlow (published) [6] | 6 | 1 | 1.3115 | 390.98 | 253.99 | 314.52 | 0.6202 | 0.6259 | —* | — | — |
| PropMolFlow (A800 Native audit) | 6 | 1 | 1.2688 | 367.22 | 256.43 | 322.82 | 0.6189 | 0.6004 | 95.0–98.9 / ≤1.11 | 996.25 \| 155.97 | 28.73 |
| PropMolFlow (Coord Reconciled) | 6 | 1 | 1.4414 | 410.57 | 262.17 | 336.61 | 0.6526 | 0.6259 | 96.1–98.5 / **0.00** | 996.25 \| 155.97 | 28.11 |
| PropMolFlow + Uni-Loom Critic (Native) | 6 | 4 | 0.6511 | 188.43 | 114.86 | 143.28 | 0.2498 | 0.2713 | **100.0** / ≤0.10 | 996.25 \| 155.97 | 7.22 |
| PropMolFlow + Uni-Loom Critic (Native) | 6 | 8 | 0.5148 | 162.75 | 91.08 | 121.70 | 0.1855 | 0.2092 | **100.0** / **0.00** | 996.25 \| 155.97 | 3.60 |
| PropMolFlow + Uni-Loom Critic (Native) | 6 | 16 | **0.4566** | **145.92** | **79.42** | **113.40** | **0.1587** | **0.1771** | **100.0** / **0.00** | 996.25 \| 155.97 | 1.80 |
| PropMolFlow + Uni-Loom Critic (Coord-Rec) | 6 | 16 | 0.5323 | 163.03 | 81.56 | 122.82 | 0.1630 | 0.2083 | **100.0** / **0.00** | 996.25 \| 155.97 | 1.75 |
| **Uni-Loom (`K=6` SFT, Raw / Rec)** | 1 | 1 | 5.43 / 5.66 | 1089 / 1074 | 571.3 / 532.1 | 1100 / 1054 | 1.719 / 1.590 | 2.933 / 3.114 | 85.7–89.1 / **0.00** | 220.27 \| **0.97** | 84.70 |
| **Uni-Loom (`K=1` SFT, Raw / Rec)** | 6 | 1 | 4.33 / 4.90 | 1147 / 1094 | 522.2 / 509.0 | 1073 / 1063 | 1.644 / 1.523 | 2.460 / 2.838 | 74.0–86.0 / **0.00** | 194.18 \| 3.08 | 60.21 |
| **Uni-Loom (`K=1` Full, Raw / Rec)** | 6 | 1 | 5.28 / 5.50 | 1318 / 1198 | 597.9 / 527.3 | 1105 / 1093 | 1.815 / 1.809 | 2.669 / 2.866 | 80.6–95.8 / **0.00** | 195.79 \| 4.50 | **92.63** |
| **Uni-Loom S2 (`[t,N]` PCA, Raw / Rec)** | 6 | 1 | 4.91 / 5.15 | 1037 / 1085 | 516.7 / 507.1 | 1118 / 1093 | 1.624 / 1.514 | 2.064 / 2.390 | 84.4–91.1 / **0.00** | 207.52 \| 6.31 | 80.33 |
| **Uni-Loom S1 (`K=6` SFT, Rec)** | 1 | 4 | 3.6900 | 602.05 | 289.92 | 543.32 | 0.7218 | 2.1888 | 99.5–100 / **0.00** | 220.27 \| **0.97** | 21.44 |
| **Uni-Loom S1 (`K=6` SFT, Rec)** | 1 | 8 | 3.4664 | 449.25 | 221.65 | 407.04 | 0.5601 | 2.0075 | **100.0** / **0.00** | 220.27 \| **0.97** | 10.71 |
| **Uni-Loom S1 (`K=6` SFT, Rec, `t`-only)** | 1 | 16 | 3.0801 | 374.63 | 199.19 | 341.31 | 0.4740 | 1.8972 | **100.0** / **0.00** | 220.27 \| **0.97** | 5.36 |
| **Uni-Loom S1 (`K=6` SFT, Rec, `t+N`)** | 1 | 16 | 2.9810 | 388.67 | 205.64 | 364.78 | 0.5051 | 1.8332 | **100.0** / **0.00** | 220.27 \| **0.97** | 5.36 |
| **Uni-Loom S2 (`[t,N]` PCA, Rec)** | 6 | 4 | 3.7355 | 574.99 | 285.56 | 573.43 | 0.8027 | 1.5811 | 99.1–100 / **0.00** | 207.52 \| 6.31 | 21.00 |
| **Uni-Loom S2 (`[t,N]` PCA, Rec)** | 6 | 8 | 3.2677 | 431.34 | 229.50 | 442.24 | 0.6303 | 1.3643 | 99.9–100 / **0.00** | 207.52 \| 6.31 | 10.63 |
| **Uni-Loom S2 (`[t,N]` PCA, Raw)** | 6 | 16 | 2.7457 | 348.85 | 216.30 | 389.34 | 0.6216 | 1.1496 | 99.5–100 / ≤0.40 | 207.52 \| 6.31 | 5.31 |
| **Uni-Loom S2 (`[t,N]` PCA, Rec)** | 6 | 16 | 3.0671 | 358.16 | 204.34 | 380.21 | 0.5434 | 1.2619 | **100.0** / **0.00** | 207.52 \| 6.31 | 5.31 |

*Pearson `r` at `M = 16` and audit notes:* **Uni-Loom S1** (`K=6` Rec, `t`-only: `α 0.899`, `Δε 0.916`, `ε_HOMO 0.918`, `ε_LUMO 0.935`, `μ 0.840`, `C_v 0.936`; `t+N`: `0.892, 0.914, 0.911, 0.919, 0.816, 0.933`); **Uni-Loom S2** (Rec: `0.888, 0.925, 0.890, 0.914, 0.806, 0.936`); **PropMolFlow + Critic** (`M = 16` Native: `0.997, 0.988, 0.985, 0.991, 0.968, 0.998`). Single-shot `M = 1` rows displaying `Raw / Rec` list Reconciled legality (`0.00%` exotic `Q`); raw `M = 1` legality is `64.4–69.6%` (`K=6`), `50.2–64.4%` (`K=1` SFT), `56.5–87.8%` (`K=1` Full), and `67.5–78.8%` (S2). Cold single-batch `bs = 128 / 250` raw decoding speeds are `41.96 / 67.26` (`K=6`), `35.99 / 60.21` (`K=1` SFT), `46.16 / 75.70` (`K=1` Full), and `42.49 / 69.84 mol/s` (S2); `M ∈ {1..16}` `Eff. Out` reflects steady-state `bs = 250` (`Uni-Loom`) or max `bs = 128` (`PropMolFlow`) end-to-end throughput (`M = 2` rows archived in JSON). `*`Published EDM/GeoLDM tables omit RDKit sanitization, formal charges, and A800 timings.

### 6.2 Cross-Paradigm `OE62` Compute Scaling, Mid-Tree Process Pruning, Size Anchoring, and Dual-Mode `PBE0` Verification

Whereas `QM9` spans small `≤ 29`-atom molecules across five light elements, `OE62` organic semiconductors reach `80` atoms across `12` elements (`H, B, C, N, O, F, Si, P, S, Cl, Br, I`) under joint electronic (`|y_HOMO - t| ≤ 0.25 eV`) and synthetic accessibility (`SA_Score ≤ 3.5 & SMARTS = 0`) targets. Borrowing compute-optimal scaling diagnostics from large language models, we benchmarked (`N = 1,792` proposals across `7` quantiles `× 256` on `NVIDIA A800-SXM4-80GB`): (i) from-scratch `PropMolFlow-OE62` [6] scaled continuously across `3.0 → 6.0 → 12.0 → 24.0 → 52.0 GPU-h` (`Epoch 1,247`, `val_loss = 1.0372`); (ii) hot-swapped online `PropMolFlow 52h + Loom-AWR` (`56` steps, `+0.12 GPU-h`); (iii) pretrained **Uni-Loom v2.0** (`~3.0 GPU-h` adaptation) under outcome `Best-of-M` (`M = 1 → 8`) versus coarse-to-fine `OctreeProcessCritic` (`M_init → M_keep`); and (iv) covariate-matched **Size-Anchored Uni-Loom** (`[homo, n_atoms]`) on an uncontended `A800` GPU alongside `42`-prompt dual-mode live `GPU4PySCF` `PBE0/def2-TZVP/DF` verification (**Figure 8** and **Table 8**).

![Figure 8: LLM-style cross-paradigm compute scaling, inference-depth scaling, and 50k-molecule campaign cost frontier on OE62.](figures/fig8_oe62_cross_paradigm_scaling.png)

**Table 8: Cross-paradigm `OE62` training/test-time scaling, `OctreeProcessCritic`, hot-swappable `Loom-AWR`, covariate-matched size anchoring, and `42`-prompt live `PBE0/def2-TZVP` verification (`N = 1,792`, `SA_Score ≤ 3.5 & SMARTS = 0`).**

| Paradigm / Weaving Arm | Decode Mode | Train (GPU-h) | Out (mol/s) | 50k Cost (GPU-h) | Atoms / Scaffolds | `SA≤3.5` / Legal (%) | $F_{\max}$ (eV/Å) | Surr. Synth Hits (%) | Blind DPA-4 Hits (%) | Blind MAE (eV) | Unbiased `PBE0` Hit (Raw / Cen) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Uni-Loom v2.0** (Unanchored) | `Out M=1` | 3.00 | **28.84** | 3.48 | — / —†† | 58.7 / 97.8 | 1.81 | 194 (10.8%) | 251 (14.0%) | 0.836 | — |
| **Uni-Loom v2.0** (Unanchored) | `Proc 8→1`† | 3.00 | 9.38 | 4.48 | — / —†† | 68.3 / 98.6 | 2.02 | 501 (28.0%) | 386 (21.5%) | 0.636 | — |
| **Uni-Loom v2.0** (Unanchored) | `Out M=8` | 3.00 | 3.26 | 7.26 | — / —†† | **98.8** / **100** | 1.35 | 1,049 (58.5%) | 742 (41.4%) | 0.424 | — |
| **Uni-Loom v2.0 + ProcessCritic** | `Proc 16→4` | 3.00 | 3.54 | 6.92 | 15.6 / 498 | 97.2 / **100** | **1.28** | 1,187 (66.2%) | 809 (45.1%) | 0.382 | 50.0% / 47.6% |
| **Size-Anchored Uni-Loom** (`[homo,N]`) | `Out M=1` | **1.28** | 11.13 | **2.53** | 33.6 / 1,317 | 18.6 / 82.5 | 9.22 | 130 (7.3%) | 96 (5.4%) | 1.098 | — |
| **Size-Anchored Uni-Loom** (`[homo,N]`) | `Proc 8→1`† | **1.28** | 3.79 | 4.94 | 33.0 / 1,408 | 33.2 / 88.9 | 7.31 | 259 (14.5%) | 227 (12.7%) | 0.891 | — |
| **Size-Anchored Uni-Loom** (`[homo,N]`) | `Proc 16→4` | **1.28** | 1.41 | 11.13 | **33.5** / **1,525** | 59.1 / 99.9 | 5.17 | 706 (39.4%) | 494 (27.6%) | 0.636 | — |
| `PropMolFlow-OE62` (From scratch) | `M=8 Coord` | 3.00 | 1.18 | 14.77 | — / —†† | 79.1 / **100** | 3.72 | 702 (39.2%) | 426 (23.8%) | 0.629 | — |
| `PropMolFlow-OE62` (From scratch) | `M=8 Coord` | 6.00 | 1.25 | 17.11 | 29.7 / 1,153 | 81.4 / **100** | 3.14 | 803 (44.8%) | 567 (31.6%) | 0.495 | — |
| `PropMolFlow-OE62` (`Ep. 311`) | `M=8 Coord` | 12.00 | 1.23 | 23.29 | 30.2 / 1,260 | 86.4 / **100** | 3.07 | 1,026 (57.3%) | 718 (40.1%) | 0.396 | — |
| `PropMolFlow-OE62` (`Ep. 592`) | `M=8 Coord` | 24.00 | 1.21 | 35.48 | 30.6 / 1,099 | 91.4 / **100** | 3.17 | 1,182 (66.0%) | 783 (43.7%) | 0.388 | — |
| `PropMolFlow-OE62` (`Ep. 1247`, Native) | `Out M=1` | 52.00 | 11.80 | 53.18 | 37.5 / 1,194 | 30.0 / 81.6 | 3.40 | 225 (12.6%) | 165 (9.2%) | 0.683 | — |
| `PropMolFlow-OE62` (`Ep. 1247`, Native) | `Out M=8` | 52.00 | 1.38 | 62.06 | 30.0 / 1,098 | 88.7 / **100** | 3.16 | 1,172 (65.4%) | 848 (47.3%) | 0.335 | 47.6% / 52.4% |
| `PropMolFlow-OE62` (`Ep. 1247`, Coord-Rec) | `Out M=8` | 52.00 | 1.22 | 63.38 | 30.0 / 1,117 | 88.6 / **100** | 3.08 | 1,179 (65.8%) | 856 (47.8%) | 0.345 | 42.9% / **54.8%** |
| **PMF 52h + Loom-AWR** (Native, `+56` steps) | `Out M=1` | 52.12 | 11.79 | 53.30 | 37.8 / 1,155 | 41.8 / 82.4 | 3.15 | 283 (15.8%) | 211 (11.8%) | 0.695 | — |
| **PMF 52h + Loom-AWR** (Native, `+56` steps) | `Out M=8` | 52.12 | 1.38 | 62.18 | 30.3 / 1,072 | 96.2 / **100** | 2.84 | 1,347 (75.2%) | **948 (52.9%)** | **0.332** | 50.0% / 47.6% |
| **PMF 52h + Loom-AWR** (Coord-Rec, `+56` steps) | `Out M=8` | 52.12 | 1.23 | 63.41 | 30.3 / 1,084 | 96.5 / **100** | 2.64 | **1,362 (76.0%)** | 941 (52.5%) | 0.334 | **54.8%** / 50.0% |

*Notes:* `†Proc 8→1` runs pure internal `OctreeProcessCritic` pruning with zero external `Uni-Mol2` calls. `††`Full-cohort SDFs were not archived on disk for `Uni-Loom Out M=1 / Proc 8→1 / Out M=8` (whose unique synthesizable hit-graph counts in JSON are `139`, `300`, and `653`, vs. `718` for `Proc 16→4`) or `PMF 3h`. Unbiased `42`-prompt `PBE0/def2-TZVP` HOMO MAE (`raw / centered`): `0.3468 / 0.3141 eV` (**Uni-Loom `16→4`**, `42/42` SCF), `0.3445 / 0.3285 eV` (`PMF 52h Native M=8`, `42/42` SCF), `0.3609 / 0.3472 eV` (`PMF 52h Coord M=8`, `42/42` SCF), `0.2702 / 0.2696 eV` (`PMF 52h + Loom-AWR Native M=8`, `40/42` SCF), and `0.2880 / 0.2849 eV` (`PMF 52h + Loom-AWR Coord-Rec M=8`, `42/42` SCF); on top-hit cohorts (`n = 42`, `100%` SCF), centered `PBE0` hit yields reach `71.4%` (**Uni-Loom `16→4`** & `PMF 52h + Loom-AWR Coord-Rec M=8`, `MAE = 0.2380 / 0.2077 eV`) and `81.0%` (`PMF Native`). Blind `DPA-4` tracks `PBE0` tightly across all arms (`MAE = 0.096–0.158 eV`, `r = 0.933–0.973`).

Four findings anchor this cross-paradigm comparison:
1. **Single-shot MLE saturation vs. logarithmic critic scaling:** Even at `52.0 GPU-h` (`Epoch 1,247`), single-shot `PropMolFlow` (`M = 1`) plateaus at `9.2%–9.8%` blind synthesizable yield (`SA ≤ 3.5` stuck at `27.9%–30.0%` due to `OE62`'s `41.9%` high-SA prior), whereas wrapping it inside **Uni-Loom**'s `CriticBus + ValenceReconciler` (`M = 8`) scales blind `DPA-4` yield logarithmically from `23.8% (3h)` to `47.8% (52h, MAE = 0.345 eV)`.
2. **Mid-tree `OctreeProcessCritic` cuts 50k-campaign compute `9×`:** Pruning `75%` of octree prefixes at `level 1→0` (`Process 16→4`) beats full-decoding `Outcome M=8` in both accuracy (`45.1%` vs. `41.4%` blind yield) and speed (`3.54` vs. `3.26 mol/s`, `2.6–2.9×` faster than `PropMolFlow M=8`), compressing end-to-end `50,000`-molecule campaign compute from `62.1–63.4 GPU-h` to `6.92 GPU-h`.
3. **Hot-swappable `Loom-AWR` breaks the MLE prior ceiling and sharpens `PBE0` quantum accuracy:** Adding just `+0.12 GPU-h` (`56` steps, `443.6 s`) of native-gated, 20-step `DPA-4`-relaxed online `Loom-AWR` onto converged `PropMolFlow` preserves native single-shot legality (`81.6% → 82.4%`), cuts zero-step median strain (`F_max: 3.40 → 3.15 eV/Å` at `M = 1`; `3.16 → 2.84` native / `2.64 eV/Å` coord-reconciled at `M = 8`), lifts `M = 8` `SA ≤ 3.5` compliance to `96.2%–96.5%`, and elevates `M = 8` surrogate and blind `DPA-4` synthesizable yields to `75.2%–76.0%` (`1,347–1,362/1,792`) and `52.9%–52.5%` (`948–941/1,792`; `MAE = 0.332 eV`). Unbiased `42`-prompt `PBE0/def2-TZVP` verification confirms both quantum parity between `3 GPU-h` **Uni-Loom `16→4`** (`50.0%` raw / `47.6%` centered) and `52 GPU-h` base `PropMolFlow M=8` (`47.6% / 42.9%` raw), and a substantial post-training quantum gain under `Loom-AWR`—cutting unbiased `PBE0` HOMO MAE from `0.345–0.361 eV` down to `0.270–0.288 eV` (`r = 0.770–0.781`) and lifting raw `PBE0` synthesizable yield to `50.0%` (native) and `54.8%` (`23/42`, coord-reconciled, `100%` SCF).
4. **Size-anchored weaving eliminates small-molecule RL shortcuts:** Conditioning jointly on `[homo, n_atoms]` restores full `OE62` size (`33.5` mean atoms, `|ΔN| = 3.52–4.56`, `<15`-atom rate cut `54.1% → 1.4%`, `38×`), expands unique SMILES to `99.78%` (`98.66%` novel) across `1,525` Murcko scaffolds (`3.1×`), and surpasses `3 GPU-h` from-scratch `PropMolFlow M=8` in blind yield (`27.6%` vs. `23.8%` at `1.28 GPU-h`, `MAE = 0.636 eV`) while retaining faster `Process 16→4` decoding than `PropMolFlow M=8` (`1.41` vs. `1.22–1.38 mol/s`; `11.13 mol/s` at `M = 1`).

---

## 7. Discussion and Conclusion

Uni-Loom demonstrates that industrial 3D molecular generation does not require training a monolithic predictor–generator supermodel from scratch. By isolating predictor capabilities behind a declarative critic bus, protecting discrete topological reinforcement learning from fine-grained coordinate gradient variance, combining source-side valence supervision with zero-displacement Lewis reconciliation, and gating self-improvement through calibrated epistemic uncertainty, heterogeneous pretrained backbones compose reliably without reward collapse.

Several limitations motivate future extensions. First, although raw point-cloud exotic charge artifacts drop substantially (`38.88% → 3.61%`), zero-charge guarantees still depend on deterministic `ValenceReconciler` cleanup for boundary hydrogens; native hydrogen-shell tokenization in future 3D generators could eliminate post-hoc Lewis reconciliation altogether. Second, `single_oracle_ad` Tanimoto guarding relies on structural scaffold proximity and can be complemented by latent uncertainty representations or differentiable multi-oracle ensembles. Third, replacing asynchronous `.sdf` DFT handoffs with on-the-fly GPU-accelerated quantum kernels (`GPU4PySCF` [18]) would compress outer-loop iteration latency toward continuous autonomous closed-loop optimization over multi-state optoelectronic (`S1`, `T1`, oscillator strength) and catalytic reaction objectives.

---

## Data and Code Availability

The `loom` codebase (`v2.0.0`, test suite `328 passed, 5 skipped`), YAML task configurations (`configs/tasks/oe62_orbital_weaving.yaml`, `configs/tasks/oe62_orbital_synthesizable.yaml`), consolidated DFT split manifests (`consolidation_manifest.json`), evaluation outputs (`speed_benchmark.json`, `oe62_audit_curve.json`, `gp_atlas.json`, `atlas2d_dft_anchor_eval.json`, `v20_synthesizable_homo_eval.json`, `v20_grounded_rsi_2round_eval.json`, `v20_rsi_acquisition_batch_r1.manifest.json`, `qm9_conditional_benchmark.json`, `qm9_conditional_benchmark_v2.json`, `qm9_propmolflow_audit.json`, `qm9_scheme1_best_of_m.json`, `qm9_scheme1_and_scheme2_summary.json`, `qm9_propmolflow_best_of_m.json`, `oe62_octree_process_critic.json`, `oe62_propmolflow_52gpuh.json`, `oe62_propmolflow_52gpuh_loom_awr.json`, `oe62_head2head_dft_anchor_52gpuh_unbiased.json`, `oe62_size_anchored_clean_gpu0_eval.json`), and manuscript build scripts are hosted in the project repository. Underlying benchmark molecules derive from the public `OE62` [1] and `QM9` [22] datasets.

---

## References

[1] A. Stuke, C. Kunkel, D. Golze, M. Todorović, J. T. Margraf, K. Reuter, P. Rinke, and H. Oberhofer. Atomic structures and orbital energies of 61,489 crystal-forming organic molecules. *Scientific Data*, 7:58, 2020. DOI: `10.1038/s41597-020-0385-y`.  
[2] V. Bhat, P. Sornberger, B. S. S. Pokuri, R. Duke, B. Ganapathysubramanian, and C. Risko. Electronic, redox, and optical property prediction of organic π-conjugated molecules through a hierarchy of machine learning approaches. *Chemical Science*, 14:203–213, 2023.  
[3] G. Zhao, Z. Cheng, K. Xu, T. Wan, M. Feng, S. Lu, H. Wang, Z. Gao, Q. Ou, L. Hu, D. Feng, Z. Zhao, and Z. Gao. Fast and flexible 3D molecular design framework for novel organic optoelectronic materials. *JACS Au*, 2026. DOI: `10.1021/jacsau.5c01756`.  
[4] E. Hoogeboom, V. G. Satorras, C. Vignac, and M. Welling. Equivariant diffusion for molecule generation in 3D. *Proceedings of the 39th International Conference on Machine Learning*, 162:8867–8887, PMLR, 2022.  
[5] M. Xu, A. S. Powers, R. O. Dror, S. Ermon, and J. Leskovec. Geometric latent diffusion models for 3D molecule generation. *Proceedings of the 40th International Conference on Machine Learning*, 202:38592–38610, PMLR, 2023.  
[6] C. Zeng, J. Jin, C. Ambrose, G. Karypis, M. Transtrum, E. B. Tadmor, R. G. Hennig, A. Roitberg, S. Martiniani, and M. Liu. PropMolFlow: Property-guided molecule generation with geometry-complete flow matching. *Nature Computational Science*, 6:233–242, 2026. DOI: `10.1038/s43588-025-00946-y`.  
[7] P. Ertl and A. Schuffenhauer. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. *Journal of Cheminformatics*, 1(1):8, 2009.  
[8] S. Lu, H. Lin, L. Yao, Z. Gao, X. Ji, Y. Liang, W. E, L. Zhang, and G. Ke. Unified cross-scale 3D generation and understanding via autoregressive modeling. *arXiv preprint arXiv:2503.16278*, 2025. URL: https://arxiv.org/abs/2503.16278.  
[9] RDKit Contributors. *RDKit rdDetermineBonds API Documentation*, 2026. URL: https://www.rdkit.org/docs/source/rdkit.Chem.rdDetermineBonds.html.  
[10] X. Ji, Z. Wang, Z. Gao, H. Zheng, L. Zhang, G. Ke, and W. E. Uni-Mol2: Exploring molecular pretraining model at scale. *Advances in Neural Information Processing Systems*, 37, 2024.  
[11] G. Zhou, Z. Gao, Q. Ding, H. Zheng, H. Xu, Z. Wei, L. Zhang, and G. Ke. Uni-Mol: A universal 3D molecular representation learning framework. *International Conference on Learning Representations*, 2023. URL: https://github.com/dptech-corp/Uni-Mol.  
[12] G. R. Bickerton, G. V. Paolini, J. Besnard, S. Muresan, and A. L. Hopkins. Quantifying the chemical beauty of drugs. *Nature Chemistry*, 4:90–98, 2012.  
[13] DeePMD-kit Developers. *DPA4/SeZM Model Documentation*, 2026. URL: https://github.com/deepmodeling/deepmd-kit/blob/master/doc/model/dpa4.md.  
[14] Tencent Hunyuan Hyra Team. *HyFF: Molecular Force Fields and Inference Runtime* (v0.2.0), 2026. URL: https://github.com/Tencent-Hunyuan/Hyra-results/tree/main/AI4Science/hyff.  
[15] C. E. Rasmussen and C. K. I. Williams. *Gaussian Processes for Machine Learning*. MIT Press, 2006.  
[16] G. W. Bemis and M. A. Murcko. The properties of known drugs. 1. Molecular frameworks. *Journal of Medicinal Chemistry*, 39(15):2887–2893, 1996.  
[17] Q. Sun et al. Recent developments in the PySCF program package. *The Journal of Chemical Physics*, 153:024109, 2020. DOI: `10.1063/5.0006074`.  
[18] R. Li, Q. Sun, X. Zhang, and G. K.-L. Chan. Introducing GPU-acceleration into the Python-based Simulations of Chemistry Framework. *arXiv preprint arXiv:2407.09700*, 2024.  
[19] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations*, 2022.  
[20] J. Ho and T. Salimans. Classifier-free diffusion guidance. *arXiv preprint arXiv:2207.12598*, 2022.  
[21] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024.  
[22] R. Ramakrishnan, P. O. Dral, M. Rupp, and O. A. von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. *Scientific Data*, 1:140022, 2014. DOI: `10.1038/sdata.2014.22`.
