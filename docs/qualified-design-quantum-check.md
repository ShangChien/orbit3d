# Quantum check of qualified generated designs

Use the completed seed-42 SFT continuation at the predeclared total 20000-step
checkpoint, with all six original/central-window × inference-seed 2026–2028 runs.
Within each nonempty run, select exactly the surrogate-qualified candidate with
the smallest original candidate ID. Do not replace an empty stratum with another
seed or choose using quantum results. The resulting cohort contains at most six
structures. Selection occurs after surrogate evaluation and before any quantum
calculation on these structures; it is a diagnostic subset of delivered designs.

Freeze the source SDFs, candidate ledgers, intervals, property units, model/input
hashes and selected atom graphs. `scripts/select_qualified_quantum_cohort.py`
exports the precise coordinates and an explicit run/candidate-to-quantum-index
map. Retain all 384 original attempts and every unselected or failed case in its
own category. A subset quantum hit count is not an all-proposal success rate.

Use the existing quantum protocol: IP-tuned LC-wPBE/def2-SVP with Jh tuning,
99/590 integration grid, SCF tolerance 1e-9, no density fitting, 100 SCF cycles
and 80 geometry steps per optimization. The geometry/omega loop, graph and stereo
checks are unchanged. These source runs target HOMO/LUMO, so excited-state
calculations are not required for the primary check. No lower-level method,
longer-budget rescue or automatic graph relabeling enters the primary cohort.

Report each source surrogate prediction, source force, quantum result/failure,
original target-window membership and graph-preservation status. The original
OCELOT per-molecule omega/geometries remain unavailable; these calculations use
the same internally defined level as the earlier quantum diagnostics and do not
claim exact reproduction of the released CSV protocol.

The fixed selection contains five structures from eight surrogate-qualified
outputs in the 384-attempt seed-42 evaluation. Original/seed-2028 has no qualified
candidate and contributes no replacement. The source candidates are original
2026/ID58, original 2027/ID5, central 2026/ID15, central 2027/ID5 and central
2028/ID44. All contain only C/H/N/O/F. The new def2 ECP construction therefore
leaves their Hamiltonians unchanged. The exact selection and input geometries
are retained in `experiments/qualified20k_selection_20261112.json` and
`experiments/qualified20k_quantum_input_20261112.sdf`.

## Completed fixed cohort

All five calculations finish with the source graph preserved. Two retain both
original target intervals; the other three miss HOMO. All five LUMO values are in
their respective windows. The quantum RMS forces are 0.00117–0.00176 eV/Å and
the singlet S2 diagnostics are numerically zero. No Hessian calculation is included.

| Quantum case | Window / inference seed | Source DPA HOMO / LUMO (eV) | Quantum HOMO / LUMO (eV) | Both targets |
| --- | --- | ---: | ---: | --- |
| 0 | Original / 2026 | -7.5762 / -0.0745 | -7.8003 / 0.1750 | Hit |
| 1 | Original / 2027 | -7.9748 / -0.3420 | -8.6653 / 0.1991 | HOMO miss |
| 2 | Central / 2026 | -6.9484 / -0.5410 | -7.4414 / -0.4153 | Hit |
| 3 | Central / 2027 | -7.7398 / -0.6621 | -7.8836 / -0.6121 | HOMO miss |
| 4 | Central / 2028 | -7.7032 / -0.8856 | -8.1364 / -0.5140 | HOMO miss |

The original window is HOMO [-8.2, -7.2], LUMO [-0.6, 0.8] eV; the central
window is HOMO [-7.75, -6.75], LUMO [-1.13, -0.13] eV. The complete accounting
remains 384 original proposals, eight surrogate-qualified outputs, five selected
quantum cases, two observed quantum hits and three uncomputed qualified outputs.
This selected-subset result is not an estimated all-proposal quantum success rate.

## Same-geometry disagreement and reference-label overlap

Re-evaluating the frozen wide two-target DPA and fingerprint models on these
five unchanged quantum geometries gives mean absolute HOMO/LUMO disagreements
of 0.4689/0.2846 eV and 0.3906/0.4082 eV, respectively. All five remain inside
the declared graph applicability domain. The source-coordinate DPA disagreements
were 0.3969/0.2676 eV; moving the evaluator to the quantum geometry does not
remove the discrepancy. Neither comparison is an OCELOT held-out MAE.

A post-hoc exact-graph comparison finds one match in the common 8188 labelled
training conformers: case 1 is `csd_RASWEO`. Its released HOMO/LUMO labels are
-7.7901/-0.4019 eV, whereas this quantum calculation gives -8.6653/0.1991 eV.
The differences are -0.8752/+0.6011 eV. This case is not an unseen graph, and
the DPA prediction at the quantum geometry (-7.8780/-0.4530 eV) is much closer
to the CSV labels. The absent original quantum geometry and omega prevent
separating reference-protocol and conformer effects. No offset is fitted to
these five outcomes, and no extra calculation replaces a missed target.

[Primary accounting](experiments/qualified20k_quantum_20261112.json),
[full quantum coordinates and raw forces](experiments/qualified20k_quantum_cohort_20261112.json),
[same-geometry model comparison](experiments/qualified20k_same_geometry_20261112.json)
and [exact training-graph overlap](experiments/qualified20k_label_overlap_20261112.json)
retain the separate definitions and input hashes.
