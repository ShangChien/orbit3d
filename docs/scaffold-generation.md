# Explicit scaffold graph growth

`orbit3d.generator.scaffold.ScaffoldState` provides an experimental graph-growth
proposal path. It is separate from pretrained octree sampling. Each action names
an open attachment site, fragment anchor and bond order. Specified hydrogens are
replaced explicitly; no distance reconstruction adds bonds to the scaffold.

```python
from rdkit import Chem
from orbit3d.generator.scaffold import ScaffoldState

scaffold = next(iter(Chem.SDMolSupplier("scaffold.sdf", removeHs=False)))
state = ScaffoldState.from_molecule(scaffold, open_atom_indices=[0, 3])
left, right = state.ports
state = state.attach("C", left).attach("C#N", right)
Chem.MolToMolFile(state.molecule, "extended.mol")
```

Existing positive atom maps and the supplied world frame are preserved; unmapped
atoms receive stable unique maps. Input graph bonds and every surviving scaffold
coordinate remain unchanged. Attachment geometry uses the replaced H direction,
a chosen torsion and MMFF optimization of new atoms with the scaffold fixed.
Valence, collisions and bond lengths are checked before returning a new state.
A failed action leaves its parent state unchanged and reports a `ValueError`.
A caller controls its finite proposal/retry budget; there is no success guarantee.

`OpenPort` can declare multiple replaceable hydrogens and allowed single/double/
triple bond orders. Consumed capacity is removed from both endpoints. New fragment
atoms become attachment sites only through `open_new_ports`; existing closed
scaffold sites cannot be opened by that method. `close_ring` is a separate,
explicit two-port bond action, followed by the same geometry and identity checks.
Current unit tests cover benzene, carbonyl and heteroaromatic cores, multiple
ports, existing maps, translated coordinates, double bonds, ring closure and
invalid/dead-end actions.

The real GPU diagnostic in `scripts/validate_scaffold_pipeline.py` uses three
cores (benzene, naphthalene, pyridine), six fixed fragments, seed 2026 and 15
refinement steps. It evaluates all 18 proposals and labels applicability-domain
failures explicitly. All 18 are chemically valid and complete refinement; three
meet the domain, optimizing HOMO/LUMO intervals and global RMS-force threshold.
The original MMFF core is a supplied constraint, not a DFT geometry. DPA runs in
float32: the conversion error to the original world coordinates is below 2e-6 A,
and fixed coordinates are bitwise unchanged during refinement.
[Per-candidate actions and results](experiments/scaffold_graph_growth_20261112.json)
are archived. This finite fragment-menu diagnostic is neither a learned scaffold
completion model nor proof of synthesizability. Native octree scaffold-context
training and joint XYZ/attachment masking remain separate work.


## Joint XYZ support on a local attachment grid

`scaffold_geometry.attachment_grid` constructs at most 1536 quantized points from
512 directions and three covalent-radius bond lengths. It restricts the direction
around the replaced hydrogen, removes anchor collisions, and optionally applies
world-coordinate bounds. X/Y prefix masks retain only bins with a full XYZ
continuation in this finite catalog. This avoids enumerating a dense spatial cube;
it does not prove that the catalog contains every feasible point.

`propose_attachment` uses explicit endpoint/bond-order actions and at most 12
whole-fragment proposals. A feasible anchor does not imply that the remaining
fragment fits: failed placements stay in the attempt ledger, and exhausted budgets
return a failed proposal without mutating the parent scaffold. Attachment preserves
the source world frame and every surviving fixed atom. This remains an explicit
graph proposal backend, not learned octree scaffold completion.

The new local-grid GPU diagnostic produced 18/18 chemically valid structures in
18 anchor attempts, all retained fixed coordinates and completed DPA refinement.
None passed the joint domain/property/force threshold in this small run. The
previous hydrogen-axis run included MMFF preoptimization, while this test preserves
the sampled grid placement before refinement; their success counts are not an
isolated mask ablation. [Per-candidate records](experiments/scaffold_grid_20261112.json)
include the selected cells, conditional support sizes, torsions and final metrics.

## Configured input and generation

`orbit3d --config configs/scaffold_graph_growth.json` selects the explicit
`scaffold_graph_growth` proposal backend. Supply an SDF containing fixed 3D
coordinates and an `OpenAtomIndices` JSON property on every record, for example
`[0, 3]`. These zero-based input atom indices identify allowed single-bond
hydrogen-replacement sites. Every surviving source atom, including its hydrogen
coordinates, remains fixed. SDF atom maps are retained; missing maps are assigned
before any graph edit.

The fragment menu and finite placement-attempt limit are explicit. Multiple
attachments per candidate consume the remaining open ports. A dead end is a
failed whole proposal, retaining the placement ledger. The backend uses uniform
fragment/site choices with independent candidate seeds; it does not use the
pretrained octree as a scaffold-conditioned model. The lower-level graph API
also supports explicitly specified bond orders, new ports and ring closures.

Eligible proposals are evaluated in candidate order up to the external-call
budget, then sorted by squared interval violation in physical units. The selected
structures use the same property/force evaluators and constrained relaxers as the
octree workflow. `batch_refinement: true` requires a shared DPA/HyFF property
adapter; black-box fingerprint evaluation uses explicit serial refinement.
No out-of-domain proposal contributes a qualified design.

Outputs include every attempted candidate, the per-growth placement ledger,
all chemically valid proposals (including OOD structures), retained and qualified
SDFs, source hashes, raw forces and relaxation traces. Exported property values
are rechecked after SDF reload. This route preserves float64 world coordinates
through conversion and refinement, avoiding float32 rounding of distant fixed
scaffolds. The property adapter forms relative coordinates before its float32
network conversion; the existing float32 octree path retains its original
numerical behavior.

Input measured labels and spectra are retained under `ScaffoldSourceProperties`,
not as labels on a newly substituted molecule. The derived candidate receives
its actual charge/multiplicity and separately named predicted properties. This
prevents a generated structure from accidentally becoming a training example
with its source scaffold's measurements.
