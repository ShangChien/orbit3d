# MIM-Flow boron structural adaptation

This experiment expands the trained octree support from H/C/N/O/F to include B.
It uses structural likelihood only. The original 450-channel MR-TADF spectrum
vectors are retained separately and never relabelled as HOMO/LUMO measurements.

The publisher's forward-data archive contains many molecular intermediates:
12,862/15,158 de-novo training records and 1,357/1,371 official-validation records
have radical SMILES. The adapter does not automatically hydrogenate them or reuse
their spectral labels after a graph change. It selects neutral, nonradical,
complete explicit-H structures whose atom order, supplied graph, coordinates and
recovered graph agree. The selected 512 training and 64 development structures
are disjoint, and every official-validation identity is excluded from both.
No independent test subset was selected in this structural pilot. Training
structures have 26-74 atoms and contain B/C/H/N; no broader element coverage is
inferred from the archive filename. See the
[subset manifest](../data/mrtadf_closed_shell512_manifest.json).

The starting checkpoint is the depth-7 OCELOT model after 1000 initial and 4000
continued backbone steps. The periodic-table token IDs stay unchanged; the
new B output category is enabled by explicit migration. Five hundred backbone
steps (batch 4, learning rate 1e-5) use no property conditions, while the conditioner
and property-understanding head remain frozen. Held-out mean batch generation
loss falls from 2.59117 to 1.05536. This is the model's weighted tree/atom/XYZ loss,
not an unweighted token NLL. The saved checkpoint explicitly blocks conditional
inference until further labelled adaptation covers the new element.

With 64 unconditional proposals and the same even-electron constraint, 33 are
chemically valid, including 10 B-containing samples; all 10 are novel against the
adaptation identities. [Per-proposal results](experiments/boron_structural500_20261112.json)
and [all valid generated structures](experiments/boron_structural500_generated_20261112.sdf)
are archived. Novelty here does not establish usefulness or synthesizability.

All 10 B-containing samples, selected without property or force screening,
undergo 50 steps of real DPA energy-only relaxation with graph/stereo checks.
No property Oracle is substituted: predicted properties and interval-hit status
are explicitly absent. Every sample completes without graph changes. Mean raw
RMS force falls from 4.29105 to 0.40984 eV/A; nine finish below 0.5 eV/A, while the
remaining sample is retained in the report at 2.05886 eV/A.
[Complete traces and raw forces](experiments/boron_physics50_20261112.json) are saved.
This is a force-field execution/geometry result, not independent quantum accuracy.
The fixed 10-molecule cohort is now undergoing the same internally specified
IP-tuned LC-wPBE/def2-SVP diagnostic used for earlier generated structures. It is
not a reproduction of the MR-TADF publisher's B3LYP/TDA vibronic-spectra protocol.

The common validator now also rejects a supplied tetrahedral or E/Z graph label
when its coordinates encode the opposite stereochemistry. The reflected-center regression failed before that correction; both reflected-center
and cis/trans checks pass afterward. Explicit
charge/multiplicity is forwarded through energy-only relaxation to state-aware
backends; the present DPA interface rejects unsupported nonneutral/nonsinglet
requests before computation. The octree property pipeline likewise refuses to
silently ignore an unsupported electronic-state configuration.


The quantum cohort has exposed failures that the earlier finite-step DPA check
did not establish. For cohort index 4, the final same-level SCF geometry conflicts
with the input E/Z assignment and its inferred graph differs. Fixed-geometry
re-evaluation from the saved SCF density reproduces the energy within 1.52e-10 Ha
and gives RMS force 0.00137 eV/Å; a low force therefore does not rescue the
source-graph validation failure. [Coordinates, raw forces and the failed graph check](experiments/boron_quantum_failed_geometry4_20261112.json)
are preserved separately, and the original cohort outcome remains failed.
