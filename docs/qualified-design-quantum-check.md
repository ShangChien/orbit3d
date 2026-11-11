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
