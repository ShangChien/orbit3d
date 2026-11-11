# ORBIT-3D: Interval-Targeted 3D Molecular Design with Gated Policy Optimization

**Shangqian Chen, Guojiang Zhao, Zheng Cheng, Jiankun Wang, Hongshuai Wang, Zhifeng Gao**\
DP Technology, Beijing, China\
`{chensq, zhaogj, chengz, wangjk, wanghs, gaozf}@dp.tech`\
November 2026

## Abstract

Inverse molecular design often specifies acceptable property ranges, yet a candidate can leave those ranges during geometric relaxation. We present ORBIT-3D, a five-stage framework that separates discrete 3D proposal learning from constrained refinement. An octree autoregressive prior is adapted to property intervals, paired with a self-ranking head, and aligned through group-relative policy optimization using property predictions, uncertainty estimates, and interatomic forces. A terminal relaxer accepts only finite, chemically admissible steps that satisfy explicit property and energy tests. The property evaluator may be differentiable or black-box; the force-field backend is selected according to its supported elements and electronic states. We give a conditional feasibility result for the acceptance rule and distinguish it from empirical molecular accuracy. Scaffold-held-out OCELOT training, three GRPO training seeds, matched refinement controls, and full-pipeline replay establish implementation behavior. Refinement gains depend on the evaluator: a separate fingerprint model favors energy-only relaxation over property-preserving acceptance. Independent model and quantum checks expose remaining failures in conditional generation; these are reported alongside the archived octree/GVP pilot. The resulting formulation makes proposal quality, oracle reliability, and relaxation acceptance separately testable, while identifying the matched-budget experiments and independent quantum-chemical validation required to establish the effectiveness of the unified implementation.

## 1. Introduction

Many molecular design tasks seek a feasible region rather than an extreme property value. An emitter may require an orbital-energy window, while a multi-objective campaign may impose several simultaneous intervals. A useful candidate must also preserve its chemical identity and retain an acceptable geometry after refinement. Optimizing a predictor score alone does not establish any of these conditions.

ORBIT-3D organizes this problem into five stages: learn a 3D prior, condition it on intervals, rank proposals, align the policy with external feedback, and refine the selected structures. Uni-3DAR provides the octree autoregressive starting point [@lu2025uni3dar]. The central design choice is to apply continuous force-based refinement to completed molecular proposals, with an explicit acceptance rule that checks the same target properties used during policy alignment.

Our contributions are a modular formulation of interval-targeted generation, a reward construction that separates positive gated utility from invalidity penalties, and a finite-step acceptance rule for constrained refinement. We treat foundation generators, property predictors, and interatomic potentials as reusable components. The contribution is their coordinated use and evaluation, not a new claim about their underlying architectures.

**Scope of the numerical evidence.** The method sections specify the target architecture. Section 4 separates current unified-package experiments from archived octree/GVP pilot runs. The current results include paired implementation checks and negative conditioning results; neither set establishes general superiority or DFT-level property accuracy.

### 1.1. Relation to existing methods

REINVENT 4 combines molecular sequence models with task-specific scoring and reinforcement learning [@loeffler2024reinvent4]. GenMol uses masked discrete diffusion over SAFE representations for de novo and fragment-constrained generation [@lee2025genmol]. These methods generate molecular strings or graphs; a conformer generator and a geometry evaluator can be attached when a task requires 3D information. Their representation should not be confused with an inability to use 3D scores or encode charged chemistry.

PropMolFlow generates property-conditioned 3D molecules using geometry-complete flow matching [@zeng2026propmolflow]. Gradient-guided extensions offer direct geometric control when a differentiable property model is available. Conflicting guidance directions are a practical optimization issue, but do not establish a fundamental impossibility of continuous generation. O2-GEN provides another 3D design route with global and local generation [@zhao2026o2gen]. A fair comparison must include conformer construction, filtering, property calls, and refinement in the computational budget.

Domain-specific components can serve complementary roles. OCNet predicts molecular properties and characterization signals, and OSCFF supplies a force field for organic semiconductors [@zhao2025ocnet; @zhao2025oscff]. They should be evaluated in those roles rather than placed in a single generator ranking. We make no quantitative comparison with these systems without a matched experimental run.

## 2. Five-stage architecture

![ORBIT-3D five-stage architecture. Stages 1-4 learn and align a proposal policy; the dashed edge updates the interval conditioner during GRPO. At deployment, the learned policy generates and ranks candidates before Stage 5 refinement. Shared property and force evaluators support alignment and acceptance. An output is released only after explicit validation.](figures/fig0_orbit3d_system_architecture.png){#fig:arch width=100%}

### 2.1. Molecular state and interval constraints

We represent a candidate as $\mathcal{M}=(G,\mathbf{z},\mathbf{X},Q,m)$, where $G$ stores connectivity, bond orders, and atomic formal charges; $\mathbf{z}$ contains atomic numbers; $\mathbf{X}\in\mathbb{R}^{N\times3}$ contains coordinates; $Q$ is total charge; and $m=2S+1$ is spin multiplicity. Stereochemical labels and atom identities must remain consistent across tokenization, oracle evaluation, and export.

For an active target set $\mathcal{K}$, let $\mathcal{I}=\{[a_k,b_k]:k\in\mathcal{K}\}$. With positive training-set scales $s_k$ and nonnegative weights $w_k$, define the normalized violation

$$
d_k(\mathcal{M})=\frac{\max(a_k-\hat y_k(\mathcal{M}),0)+\max(\hat y_k(\mathcal{M})-b_k,0)}{s_k},
\qquad L_{\mathrm{int}}=\sum_{k\in\mathcal{K}}w_k d_k^2.
$$

The chemical predicate $C(\mathcal{M})$ checks the task's requirements for connectivity, valence, formal charge, and electronic state. Closed-shell validity is required only for closed-shell tasks; radicals and triplets require their own explicit state checks. An even electron count is not sufficient evidence of a closed shell.

### 2.2. Stage 1: learn the 3D proposal prior

The generator represents 3D structure through a coarse-to-fine octree and predicts the resulting token sequence autoregressively. The spatial range, quantization resolution, element vocabulary, and sequence limits belong to the model specification and are saved with its checkpoint. Inputs outside that specification are rejected or processed by an explicitly trained alternative representation; they are not silently clipped, truncated, or converted to another element.

Octree grammar constrains spatial tokenization. Chemical constraints require additional state, including connectivity and remaining valence, followed by validation of the completed structure. We distinguish these constraints because a syntactically valid spatial sequence need not describe a chemically valid molecule. Decoding must preserve atom identities and recover a graph consistent with the coordinates, rather than assigning every inferred bond as a single bond.

### 2.3. Stage 2: learn interval conditioning

During supervised fine-tuning (SFT), intervals are sampled around observed properties, with independently varied widths and active-target masks. The current condition contains normalized interval bounds and target identities. The tested mainline is restricted to neutral closed-shell molecules; broad charge/spin-conditioned generation requires separate representation and training validation. A condition prefix and layer-wise adaptive normalization modulate the autoregressive policy. Any new input semantics, such as rescaled spatial bins or additional elements, require explicit checkpoint migration and validation.

The token likelihood is trained on the same admissible vocabulary used for sampling. All active property dimensions contribute to conditioning; missing targets are represented by masks, not by placeholder values that can be mistaken for measurements.

### 2.4. Stage 3: learn and use self-ranking

A structure-understanding head predicts properties from the generated structure representation and ranks a pool of candidates before expensive external evaluation. To avoid learning the interval midpoint as a shortcut, its regression training uses an unconditional or condition-masked pass. A joint objective is

$$
L_{\mathrm{gen}}=\overline{\mathrm{CE}}_{\mathrm{tree}}
+\overline{\mathrm{CE}}_{\mathrm{atom}}
+\frac{0.1}{3}\sum_{j\in\{x,y,z\}}\overline{\mathrm{CE}}_j,
\qquad
L_{\mathrm{SFT}}=L_{\mathrm{gen}}+
\lambda_{\mathrm{self}}\,\operatorname{SmoothL1}_{0.1}
(\hat{\mathbf y}_{\mathrm{self}},\mathbf y).
$$

Each cross entropy is averaged over its own decision type; the self loss uses training-normalized targets and $\lambda_{\mathrm{self}}=0.5$ in the reported SFT runs. Self-ranking reduces external oracle calls, but still has a computational cost and can introduce selection bias. The candidate-pool size, ranking criteria, rejected proposals, and retained count must therefore be reported. At deployment, the trained head is used without further SFT.

### 2.5. Stage 4: align with gated group-relative feedback

For each target condition, the policy samples a group of proposals. A property oracle returns predicted means and uncertainty estimates; a force field returns finite energies and unmodified forces. Ensemble disagreement is computed from individual fold predictions and calibrated on held-out structures. A shared backbone with several heads is not automatically a cross-validated ensemble.

The implemented validity gate also requires finite evaluator outputs and membership in the declared Oracle applicability domain. Rejected or out-of-domain proposals receive the same explicit failure reward.

Let $u(\mathcal M)=\exp[-L_{\mathrm{int}}(\mathcal M)]$ be a nonnegative interval utility, and let $G_{\mathrm{unc}},G_{\mathrm{phys}}\in[0,1]$ decrease with uncertainty and physical strain. We use

$$
R(\mathcal M)=C(\mathcal M)\,G_{\mathrm{unc}}(\mathcal M)\,G_{\mathrm{phys}}(\mathcal M)\,u(\mathcal M)
-\lambda_{\mathrm{invalid}}[1-C(\mathcal M)].
$$

In the reported runs, $G_{\mathrm{phys}}=\exp[-\lambda_F\max(0,F_{\mathrm{rms}}-0.5)]$ with forces in eV/Å, and $G_{\mathrm{unc}}=\exp[-\lambda_U\operatorname{mean}_k(\sigma_k/s_k)]$. The 0.5 threshold is an experimental setting, not a universal stability criterion. The default $\lambda_U=0$ reflects weak held-out error ranking. This construction avoids multiplying a negative penalty by a gate that would make a worse proposal less penalized. Group-relative advantages are $A_g=(R_g-\bar R)/\max(\sigma_R,\epsilon)$. Following GRPO [@shao2024grpo], the policy objective combines a clipped likelihood-ratio surrogate with a reference-policy KL term and SFT replay:

$$
\begin{aligned}
L_{\mathrm{policy}}={}&-\frac1G\sum_{g=1}^{G}\frac1{T_g}\sum_{t=1}^{T_g}
\min\!\left(r_{g,t}A_g,\operatorname{clip}(r_{g,t},1-\varepsilon,1+\varepsilon)A_g\right)\\
&+\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})+\lambda_{\mathrm{anchor}}L_{\mathrm{gen}}.
\end{aligned}
$$

Replay updates the conditioner using the generation loss only. The KL is averaged over the same per-molecule decision counts as the policy term. Here $r_{g,t}$ is the new-policy to rollout-policy probability ratio under the same token constraints. The reference is frozen; the current-policy KL contribution must retain gradients. Physics and uncertainty gates are risk controls, not proofs that reward exploitation has been eliminated. Independent property validation remains necessary.

### 2.6. Stage 5: refine with explicit acceptance

For a completed proposal, refinement holds its chemical graph, charge, multiplicity, and any fixed scaffold atoms constant. Let $\mathbf F=-\nabla_{\mathbf X}E$ and $\mathbf g=\nabla_{\mathbf X}L_{\mathrm{geom}}$, restricted to movable coordinates. The current relaxer uses $L_{\mathrm{geom}}=\sum_k v_k^2$, where $v_k=s_kd_k$ is the violation in physical property units; the reported HOMO/LUMO targets both use eV. When gradients are available, a regularized projection damps the force component that increases interval violation:

$$
\mathbf F_{\perp}=\mathbf F-\frac{\max(0,\langle\mathbf F,\mathbf g\rangle)}{\|\mathbf g\|_F^2+\epsilon}\mathbf g,
\qquad \widetilde{\mathbf X}=\mathbf X+\operatorname{clip}_{\delta}\!\left[\eta(\mathbf F_{\perp}-\alpha\mathbf g)\right].
$$

The norm and inner product are taken over the full movable molecular coordinate vector. The operator $\operatorname{clip}_{\delta}$ multiplies the entire proposed displacement by $\min(1,\delta/\max_j\|\Delta\mathbf X_j\|)$, preserving its direction while bounding every atom displacement. For a black-box oracle, the candidate direction can use the force alone. In either case, projection is only a proposal mechanism: a finite step, clipping, or a locally flat hinge loss can still violate an interval.

The candidate is accepted only if evaluations are finite, chemical constraints remain satisfied, and

$$
\begin{aligned}
E(\widetilde{\mathbf X})&\leq E(\mathbf X)+\epsilon_E,\\
v_k(\widetilde{\mathbf X})&\leq v_k(\mathbf X)+\epsilon_k \quad\text{for every active }k.
\end{aligned}
$$

The implementation also anchors energy and each violation to their initial values: $E(\widetilde{\mathbf X})\leq E(\mathbf X_0)+\epsilon_E$ and $v_k(\widetilde{\mathbf X})\leq v_k(\mathbf X_0)+\epsilon_k$. Thus an initially feasible constraint remains within its fixed tolerance band; tolerances do not accumulate at each step. Rejected steps leave the state unchanged and reduce the step size. The accepted coordinates are re-evaluated before export, using the full oracle ensemble, including any non-differentiable stacking model.

**Proposition 1 (conditional feasibility preservation).** Assume fixed deterministic evaluators, exact acceptance tests with $\epsilon_E=\epsilon_k=0$, and an initially chemically valid candidate satisfying all active intervals. Then every accepted or retained iterate remains feasible, and its evaluated potential energy is nonincreasing.

**Proof.** The initial iterate satisfies the conditions by assumption. An accepted iterate satisfies the chemical and per-target tests and cannot increase energy. A rejected step retains the previous iterate. Induction establishes both statements. This result does not imply convergence to a stationary point, discovery of a feasible molecule from an infeasible start, or agreement of surrogate properties and energies with DFT. The implemented initial-state tests bound drift when tolerances are nonzero; they do not guarantee progress from an infeasible start.

## 3. Evaluator interfaces and implementation requirements

A property oracle exposes predictions, uncertainty information, supported targets, and optional coordinate gradients. Uni-Mol [@zhou2023unimol], task-specific heads on frozen representations, and Uni-QSAR ensembles can implement this interface. Every path must return the same physical-unit prediction for the same structure. Replacing nonlinear stacking with a mean is not a valid differentiation strategy.

A force-field backend exposes energy, forces, supported elements and states, and a failure indicator. DPA-4 [@deepmodeling2026dpa4] and HyFF [@hyra2026hyff] are alternative backends; their precise capabilities depend on the selected version and checkpoint. OMol25 provides broad molecular, elemental, charge, and spin coverage [@levine2025omol25], but dataset coverage alone does not certify a generator or a downstream adapter for each regime. The legacy DPA checkpoints used below were called through a geometry-and-element interface; this does not establish a family-wide limitation on charge or spin conditioning.

The two evaluators may share a backbone to reduce memory use, but remain distinct logical interfaces. One scalar per molecule limits the largest atom displacement while preserving the projected direction; physical diagnostics use raw finite forces. Element mappings, bond orders, formal charges, coordinate units, target scales, and electronic-state conventions must be preserved across both interfaces.

The five stages describe a learning-and-refinement workflow. They are not five models retrained for each requested molecule. Deployment uses a saved, aligned policy and ranking head, followed by Stage 5. Specialist online adaptation is an optional separate operation with its own budget and checkpoint.

## 4. Experimental protocol and pilot evidence

### 4.1. Current implementation: OCELOT and QM9s

The executable package uses a vendored pretrained Uni-3DAR octree, an explicit
multi-target conditioner, independent bootstrap property adapters, and DPA4-Mini
physical evaluation. MIM-Flow QM9s [@cheng2026mimdata] supplies structures and its original 240-channel spectra;
HOMO/LUMO predictions on QM9s are clearly marked pseudo-labels. OCELOT [@bhat2023ocelot] supplies
reference HOMO/LUMO labels in eV. Its available official archive has no original
DFT geometries or per-molecule range parameters, so the 3D property model uses
ETKDG/MMFF reconstructions and fixed scaffold-separated splits in the initial experiments. A separate expanded-element experiment explicitly uses ETKDG/UFF for every input.

<!-- BEGIN CURRENT RESULTS -->
| Candidate ranking | Mean joint success (%) | Across-seed SD (percentage points) |
| :--- | ---: | ---: |
| first valid | 11.46 | 1.80 |
| self | 11.46 | 1.80 |
| external | 20.83 | 1.80 |

Table: Three inference seeds and identical proposal pools, before applying the later applicability-domain filter. Joint success uses the optimizing surrogate and includes all 32 attempts in its denominator.

| Conditioner | Even-electron constraint | Chemical validity (%) | Joint success (%) |
| :--- | :--- | ---: | ---: |
| v2 reset | off | 47.40 | 3.12 |
| v2 reset | on | 88.02 | 3.12 |
| v3 smooth | off | 42.19 | 1.56 |
| v3 smooth | on | 82.29 | 2.08 |

Table: Matched conditioner comparison with domain filtering, 64 attempts per inference seed (three seeds). Both variants use the same structural prior and adaptation labels; v3 jointly changes frequency encoding and interval sampling. The validity gain does not establish improved joint success.

| GRPO objective | Valid / 192 | Optimizing-oracle joint hits / 192 | Independent-surrogate joint hits / 192 |
| :--- | ---: | ---: | ---: |
| full | 161 | 8 | 8 |
| no kl | 159 | 2 | 3 |
| no physics | 158 | 3 | 5 |
| no replay | 169 | 7 | 8 |

Table: Fifty rollout iterations (two optimizer passes per group), training seed 42, three inference seeds, and identical float64 energy reporting. The independent evaluator is a separately trained fingerprint surrogate, not DFT truth.

| Policy stage | Attempts | Optimizing joint hits | Independent joint hits |
| :--- | ---: | ---: | ---: |
| Unconditional prior | 192 | 1 | 2 |
| Interval SFT | 192 | 4 | 4 |
| SFT + full GRPO | 576 | 20 | 20 |

Table: Three inference seeds per policy. GRPO additionally uses three training seeds (42-44), all initialized from one SFT checkpoint. The prior-to-SFT comparison changes both conditioning and the self-shortlist.

| Refinement rule | Optimizing joint hits / 192 | Independent joint hits / 192 | Mean RMS force (eV/Å) |
| :--- | ---: | ---: | ---: |
| None | 0 | 0 | 2.5858 |
| Energy only | 7 | 17 | 0.1709 |
| Projection only | 7 | 8 | 1.0572 |
| Full acceptance | 8 | 8 | 1.3537 |

Table: Identical 34 selected starting structures from 192 proposals; 30-step budget and production step size 0.004. Selection is fixed, but backtracking changes evaluator cost. The independent fingerprint surrogate favors energy-only refinement.

| Matched OCELOT test (256 structures) | 4096 training records, MAE (eV) | 8192 training records, MAE (eV) |
| :--- | ---: | ---: |
| HOMO | 0.2995 | 0.2863 |
| LUMO | 0.2379 | 0.2431 |

Table: Exact same test structures and reconstructed geometries; no overlap with either training set. Expanding training improves HOMO and slightly worsens LUMO on this test.

| Shared optical test (512 molecules) | DPA adapter MAE (eV) | Fingerprint MAE (eV) |
| :--- | ---: | ---: |
| HOMO | 0.2895 | 0.3593 |
| LUMO | 0.2326 | 0.3354 |
| Vertical S1 | 0.2314 | 0.2968 |
| Vertical T1 | 0.1381 | 0.2616 |
| Paired vertical S1–T1 | 0.2226 | 0.2546 |

Table: Official OCELOT labels on the same 512 scaffold-held-out molecules. Training budgets are 8192 for the DPA adapter and 20356 for the fingerprint model; this is not a matched architecture ablation. The last row averages per-molecule absolute gap errors.
<!-- END CURRENT RESULTS -->

The three ranking runs use identical proposal tokens for each seed, with a fixed
32-proposal/8-retained budget. External ranking consumes additional property-model
calls, so its gain is a selection result, not an equal-oracle-budget superiority
claim. Self-ranking does not improve the measured joint success over first-valid.
The initial BF16 policy comparison failed behavior/recomputed-probability checks;
GRPO now uses FP32 throughout. In the matched 50-rollout-iteration pilot, removing KL or physical feedback
reduces joint hits; removing replay leaves the independent joint count unchanged.
Each group receives two optimizer passes, updating the interval conditioner only.
For the full objective, three GRPO training seeds give optimizing joint counts
8/7/5 and independent counts 8/5/7, each out of 192 attempts. Both total 3.47%,
with an across-training-seed standard deviation of 0.80 percentage points. The
SFT initialization is shared, and the reward-removal table still contains only
one training seed.

Matched refinement exposes evaluator dependence. Full acceptance preserves all
six initially property-feasible molecules and reaches eight optimizing joint
hits; energy-only and projection-only each lose one and reach seven. Yet the
independent fingerprint surrogate gives energy-only 17 joint hits versus eight
for full acceptance. Its 17 property hits are unchanged across rules because
it is insensitive to geometry. The lowest mean residual force also comes from
energy-only refinement. These results support the acceptance invariant but do
not establish superior physical or independent property quality.

Larger OCELOT adaptation improves domain coverage without establishing conditional
success: the continued 4096-record run has 32 chemically valid proposals, 5 domain-
eligible retained molecules and 0 joint successes from 64 attempts. Independent
fingerprint evaluation gives 3 property hits and 1 joint success among the same 5
structures. This disagreement is retained rather than choosing the favorable
surrogate. Independent calibration on 512 unused validation structures achieves
91.4% joint test coverage, but mean interval widths are 1.94/1.73 eV and ensemble
standard deviation correlates weakly with errors; uncertainty-based reward is
disabled by default. Expanding the adapter training set to 8192 records yields
HOMO/LUMO MAEs of 0.2878/0.2429 eV on a separate 512-structure test. On the original
256 structures, only HOMO improves. A disjoint 512-record calibration set gives
90.82% joint test coverage with mean widths 1.72/1.78 eV; spread/error Spearman
correlations remain only 0.173/0.105. Exact adapter/scaler replay is separated
from repeated native-backbone floating-point variation.

A separate 15-element OCELOT run uses 8192 uniform ETKDG/UFF training conformers,
three bootstrap adapters and 512 test structures, giving HOMO/LUMO MAEs of
0.2985/0.2382 eV. Test subgroups contain B 13, Si seven, Se one, As one and Te four
structures; the latter counts cannot support broad element-specific accuracy
claims. Geometry failures, including unsupported UFF Se types, remain exclusions.
The exact input graphs are preserved in property training. A coordinate-only
audit of 321 rare-element inputs recovers 295 exact identities, adds previously
unspecified stereo in 18, changes charge/bond assignments in seven and rejects
one. The same counts after 0.01 Å rounding implicate graph inference rather than
that rounding step.

The 8188 encodable UFF training structures also adapt the 15-element generator.
Across three inference seeds and 192 proposals per window, the original,
training-median and lower-gap regimes give 110/95/89 valid structures, 9/9/12
domain-eligible proposals and 2/0/0 optimizing joint successes. Independent joint
counts are 1/0/0. Window widths differ, so these are task regimes rather than an
equal-difficulty comparison. None of the 294 valid structures contains B, Si,
Se, As or Te; the supported element vocabulary alone is insufficient.

Three matched exploratory interventions expose the quality tradeoff. Raising
atom temperature from 0.3 to 1.0 introduces all five rare elements into valid
outputs, but lowers validity from 110/192 to 89/192 and optimizing joint hits
from two to zero. A paired-condition hinge objective and its unpaired control
both obtain 3 optimizing and 4 independent joint hits in 384 proposals. Inverse-
square-root element-frequency weighting increases valid B/Si occurrence to two
and six molecules, respectively, but reduces optimizing hits from 7/384 to
3/384 while independent hits remain 6/384. Each training intervention uses one
matched seed; none establishes improved conditional design, and neither the
default temperature nor the default objective is changed.

The four-target property extension adds the official vertical S1 and T1 labels
while reusing the exact UFF conformers and splits. Its paired vertical S1–T1
MAE is 0.2226 eV; the absolute difference of means is 0.0537 eV and would hide
substantial molecule-level errors. These supervised CSV errors neither reproduce
the unavailable original DFT geometries nor establish an adiabatic gap error.

An explicit fragment-attachment CLI separately tests fixed-scaffold refinement.
It preserves two designated attachment sites and world-frame scaffold positions
for benzene, naphthalene, pyridine and a boron-containing input, including large
translations. All 24 proposals are valid, 16 receive external ranking and eight
are refined; none reaches joint success in the eight-step diagnostic. Full replay
retains graphs, placements and coordinates exactly, with ordinary property/force
comparisons using the existing numerical tolerances. This graph-action workflow
is not learned octree scaffold completion.

HyFF-presto uses the same pipeline with its own property adapters. A 512-training-
record, 64-test-record diagnostic gives MAEs 0.3853/0.3208 eV; its 32-proposal
end-to-end run has 26 valid candidates, three domain-eligible structures and two
joint hits under each surrogate. Its sample budget differs from the DPA study,
so these values are not a backend ranking. Whole-cohort replay passes for all
three HyFF exports and eleven DPA exports in separate 32/64-proposal runs.

An explicit depth-six to depth-seven migration retains 0.24 Angstrom leaf cells
and expands the spatial range to 30.72 Angstrom. After backbone adaptation, a
64-proposal diagnostic produces 48 valid structures, including three above the
old 40-atom limit. It does not demonstrate reliable generation throughout the
expanded spatial range. A separate structural-only adaptation adds boron using
512 closed-shell MIM-Flow structures. Of 64 attempts, 33 are valid and ten contain
boron; all ten are novel relative to the adaptation structures. Energy-only
refinement preserves their graphs and lowers mean RMS force from 4.29 to 0.41
eV/Angstrom, with nine below 0.5. This model has no validated boron property
conditioner. The original 450-channel spectra are retained separately and are
not relabeled as orbital energies.

The entire retained cohort of the earlier 1024-record/4000-step adaptation consists
of one molecule. At an explicit IP-tuned LC-wPBE/def2-SVP diagnostic level, its
optimized graph is preserved and the DFT RMS force is 0.00255 eV/Å. Its HOMO/LUMO
are $-8.81566/-0.00179$ eV, versus the optimizing surrogate's $-7.74292/-0.26839$ eV;
the HOMO misses the original $[-8.2,-7.2]$ eV interval. Full spin-adapted TDDFT gives
vertical S1/T1 energies 4.03502/3.08312 eV. Separate state optimizations at the
same S0-tuned range parameter give adiabatic excitations of 3.63105/2.51208 eV
and an S1-T1 difference of 1.11897 eV, without zero-point corrections. Central
differences of all 51 gradient coordinates give 45 positive vibrational modes
in each state (lowest 47.46/23.90 inverse centimeters); both preserve the graph.
These checks use corrected third-order XC evaluation and response-state indexing.
Because reference naphthalene calculations do not reproduce the published OCELOT
CSV, these are internally consistent protocol diagnostics, not dataset MAEs.

### 4.2. What the archived experiments establish

We report a limited pilot from the original octree implementation on QM9-derived molecules [@ramakrishnan2014quantum]. The property oracle is the GVP regressor used by the archived pipeline, not a five-fold Uni-Mol ensemble. The policy was aligned with DPA4-Air feedback; a shared candidate pool was then refined with different force-field scales. DPA4-Plus supplies the reported force evaluation.

Each task requests 48 retained candidates from 576 sampled trajectories (12-fold oversampling). Up to 96 ranked candidates enter refinement. Ranking includes chemical filters, the self-head, and the GVP interval score. Consequently, the table reports performance after selection and refinement; it is not a raw generator acceptance rate, and the GVP hit rate is not independent validation. Wall times include the recorded candidate-pool sampling/ranking and refinement components on an A800, excluding checkpoint loading, force-field warmup, and final metric evaluation. They are not per-candidate latency guarantees for the unified package. The table presents the Mini and Plus backends; the archived JSON includes all recorded variants.

<!-- BEGIN GENERATED PILOT TABLE -->
**Pilot targets (Hartree).** Gap: [0.320, 0.345]; HOMO: [-0.205, -0.190]; LUMO: [0.060, 0.080]. Joint: Gap [0.285, 0.320], HOMO [-0.245, -0.220], LUMO [0.055, 0.085].

| Task | Refinement backend | GVP hit (%) | RMS force (eV/Å) | Wall time (s) |
| :--- | :--- | ---: | ---: | ---: |
| Gap | DPA4-Mini | 93.75 | 0.3655 | 51.45 |
| Gap | DPA4-Plus | 93.75 | 0.3648 | 73.60 |
| HOMO | DPA4-Mini | 89.58 | 0.7602 | 44.96 |
| HOMO | DPA4-Plus | 91.67 | 0.7518 | 61.67 |
| LUMO | DPA4-Mini | 100.00 | 0.2043 | 48.30 |
| LUMO | DPA4-Plus | 100.00 | 0.1982 | 68.49 |
| Joint | DPA4-Mini | 100.00 | 0.3848 | 49.85 |
| Joint | DPA4-Plus | 100.00 | 0.3761 | 70.12 |

Table: Archived octree/GVP pilot, 48 retained candidates per task. Forces are evaluated by DPA4-Plus. Values describe selected outputs under the legacy protocol, not independent validation of the target architecture.
<!-- END GENERATED PILOT TABLE -->

The archived LUMO run combines a 100% retained-pool surrogate hit rate with a mean molecular RMS force of approximately 0.198 eV/Å for the Plus backend. This observation supports further investigation of terminal refinement. It does not prove a local minimum, chemical novelty, independent property accuracy, or a universal benefit of any one stage. The weaker HOMO result also shows why each target regime must be reported separately. No cross-generator superiority claim is made from this table.

### 4.3. Evaluation of the unified method

The primary success metric is a per-sample conjunction over all generated proposals:

$$
S=\frac1{N_{\mathrm{attempted}}}\sum_i
\mathbf1\{C(\mathcal M_i)=1\}\,
\prod_{k\in\mathcal K}\mathbf1\{\hat y_k(\mathcal M_i)\in[a_k,b_k]\}\,
\mathbf1\{F_{\mathrm{rms}}(\mathcal M_i)\leq F_{\max}\}.
$$

The force statistic is $F_{\mathrm{rms}}=\sqrt{N^{-1}\sum_j\|\mathbf F_j\|_2^2}$; its evaluation backend and threshold are reported. Marginal validity and hit rates are not multiplied to estimate the joint metric. We additionally report raw and retained counts, connected validity, graph/coordinate consistency, novelty, diversity, per-target violations, maximum force, and numerical-failure rates. Closed-shell statistics apply only to the corresponding target states.

Matched-budget ablations should compare the prior, interval SFT, self-ranking, gated policy alignment, and refinement while controlling candidate-pool size and oracle calls. Gate ablations independently remove uncertainty, physics, KL, and replay terms. Refinement controls compare pure energy minimization, projection without acceptance, and the complete acceptance rule. Runs require multiple seeds, uncertainty intervals, and matched selection criteria; changing oversampling together with a model component does not isolate that component's effect.

The property evaluator used to optimize and select molecules must be distinguished from the independent evaluator used to assess them. For a representative, predeclared subset, quantum-chemical calculations should verify structures and target properties at a compatible level of theory. For spin-sensitive tasks, vertical excitation-like differences, adiabatic differences, and emission energies require separate definitions and per-molecule comparisons. A difference of ensemble means is not an MAE.

### 4.4. Reproducibility

Each run must identify the code revision, environment, input/split versions, tokenizer specification, full target intervals, model and oracle checkpoint hashes, electronic states, seed policy, sampling budget, and filtering rules. Reload checks compare every molecule's tokens, graph, coordinates, and predictions under declared tolerances. Independent random streams do not by themselves guarantee bitwise agreement across batching strategies or attention kernels.

The pilot table is generated from the accompanying JSON archive; source hashes and protocol notes are included in its provenance manifest. The archive does not contain all weights and raw sample-level records needed to rerun or independently re-score the experiment. Current package runs separately save full source snapshots, runtime versions, checkpoint hashes, raw trajectories and per-attempt states. A 32-candidate/8-export batch-path replay passes at 1e-5 prediction/coordinate tolerance. An earlier serial/batch comparison preserved all tokens and decisions. After correcting physical-energy accumulation to float64, strict zero-tolerance acceptance differs for one of eight structures; an explicit 1e-5 eV/property tolerance restores matched decisions within the declared cross-path numerical thresholds. Both outcomes are retained. Removing identity KV-cache row copies preserves all tokens, coordinates and scores in an eight-trajectory test; three shared-GPU repeats change median latency from 8.96 to 7.94 seconds. This is not a general throughput guarantee. Broader scientific effectiveness remains a separate experimental requirement. The historical dense-grid CLI and the octree pilot must not be labeled as the same trained model.

## 5. Limitations and scope

First, acceptance guarantees are relative to fixed surrogate evaluators. Correlated errors between property models and force fields can survive both uncertainty and physical gates. Multiple MLIP evaluators provide a useful diagnostic, but are not substitutes for quantum-chemical validation.

Second, explicit electronic-state inputs do not establish accuracy for all elements, charges, or multiplicities. The generator vocabulary, training distribution, graph representation, and selected checkpoint must all support the requested chemistry. Open-shell and organometallic design require dedicated validation rather than extrapolation from closed-shell QM9 results.

Third, constrained refinement can stagnate when no step simultaneously improves energy and preserves the required properties. Such a result should be reported as a failed or unresolved candidate. The method does not guarantee thermodynamic stability, synthetic accessibility, excited-state accuracy, or global optimality.

Finally, autoregressive decoding, candidate selection, ensemble evaluation, and refinement can be expensive. KV caching and batching are engineering opportunities, but their speed and output-quality effects require measurement. Scaffold-conditioned generation additionally requires preservation of the input graph, explicit attachment constraints, and projection in the movable coordinate subspace.

## 6. Conclusion

ORBIT-3D formulates interval-targeted molecular design as discrete proposal learning followed by explicitly constrained refinement. Its five stages give distinct roles to the generative prior, interval conditioning, self-ranking, externally scored policy alignment, and terminal relaxation. A common evaluator interface supports differentiable and black-box properties, while an acceptance test makes the conditions for retaining a refinement step explicit.

The archived octree/GVP results motivate this formulation but do not validate every component of the unified system. The corrected implementation now has matched-budget conditioning and reward ablations, independent surrogate checks, and a state-specific quantum diagnostic. Replicating SFT and successful training interventions, improving performance across target regimes, and completing independent quantum cohorts remain necessary for general effectiveness claims. This distinction keeps the architectural claim precise and makes subsequent improvements measurable.

## References

Bibliographic metadata and primary-source links are maintained in [references.bib](references.bib). Citations in this Markdown source use Pandoc keys; the generated LaTeX resolves them to numbered references.
