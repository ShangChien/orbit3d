# ORBIT-3D

Interval-targeted 3D molecular generation with a pretrained octree prior,
property oracles and constrained force-field refinement.

**Authors:** Shangqian Chen, Guojiang Zhao, Zheng Cheng, Jiankun Wang, Hongshuai Wang, Zhifeng Gao.

The executable mainline is **Uni-3DAR octree → interval adaptation → candidate
ranking → DPA4 refinement**. GRPO is available for experiments and disabled in the
recommended configuration because its first matched-budget comparison did not
improve final success. The dense-grid implementation remains a baseline.

![Five-stage architecture](paper/figures/fig0_orbit3d_system_architecture.svg)

## Verified scope

- MIM-Flow QM9s: explicit-H structure adaptation, fixed train/dev/test selection,
  molecular identity checks and original spectrum preservation.
- OCELOT: full compatible-subset fingerprint training and a separate small 3D
  bootstrap-DPA property pilot for HOMO/LUMO, with scaffold-separated evaluation.
- Real pretrained octree generation, multi-target conditioning, chemical checks,
  candidate ranking, real DPA4 forces and finite-step property/energy acceptance.
- Whole-candidate ledgers, precise SDF export and re-evaluated exported predictions.

See the [experiment report](docs/qm9s-ocelot-experiments-2026-11-11.md) for metrics,
failed experiments and evidence. Oracle interval success is not independent
quantum-chemical accuracy. Current generation covers neutral closed-shell
H/C/N/O/F molecules; broad-element and open-shell generation are not established.

## Setup

```bash
uv sync --frozen --extra octree
uv run --frozen --extra octree pytest tests -q
uv build --wheel
python scripts/check_wheel.py
```

DPA4 experiments additionally need a compatible DeepMD runtime and mounted DPA4
weights. Remote validation used PyTorch 2.7.1 and the available DeepMD SeZM runtime;
this does not establish GPU compatibility of every optional package or Docker
image. The wheel includes the pinned Uni-3DAR runtime and its license. Weights and
raw data are obtained separately; see [data notes](data/README.md).

## Data and training

```bash
# Source is the extracted directory containing train.lmdb and valid.lmdb.
uv run --extra octree python scripts/adapt_qm9s.py \
  --source /data/mim-flow/qm9s --output dataset/processed/qm9s_octree

python scripts/prepare_ocelot.py --help
python scripts/train_ocelot_baseline.py --help
python scripts/train_ocelot_3d.py --help
```

QM9s spectral vectors are not HOMO/LUMO labels. In the main configuration,
`label_source: oracle` explicitly predicts adaptation labels with the OCELOT
model. Original OCELOT DFT geometries are unavailable in the supplied CSV;
the 3D property pilot uses ETKDG/MMFF input conformers.

Edit the mounted paths in [`configs/qm9s_ocelot_octree.json`](configs/qm9s_ocelot_octree.json), then run:

```bash
orbit3d --config configs/qm9s_ocelot_octree.json
```

For inference, set `mode` to `predict` and supply `checkpoint_load_path` for the
saved octree checkpoint. The original pretrained file and training SDF can then
be omitted. New checkpoints retain target scaling and training identities.

Each run writes `candidates.csv`, `generated.sdf`, `qualified.sdf`,
`relaxation_traces.json`, `run_config.json` and `summary.json`. `generated.sdf`
contains all retained/refined candidates; `qualified.sdf` contains those passing
both property intervals and the configured raw RMS-force threshold. Failed and
filtered proposals remain in the original candidate denominator.

## Code map

| Module | Responsibility |
| --- | --- |
| `generator/octree_model.py`, `octree_backend.py` | Pretrained hierarchical generator, interval conditioner and structure head |
| `generator/octree_alignment.py` | Complete-trajectory GRPO, differentiable KL and replay |
| `oracles/dpa4_oracle.py`, `fingerprint_oracle.py` | Independently bootstrapped 3D adapters and 2D reference predictor |
| `physics/dpa_forcefield.py`, `conformer_relaxer.py` | Coordinate-connected descriptors, raw forces and accepted refinement |
| `pipeline/octree_runner.py` | Main training/inference workflow |
| `utils/qm9s.py` | Spectrum-preserving structural adapter |
| `vendor/` | Pinned Uni-3DAR source and minimal runtime dependencies |

The older `internal_*.json` configurations and `benchmarks/runners/` belong to
previous baselines or archived experiments. They are not the recommended route
for the new QM9s/OCELOT workflow. Historical hardcoded comparisons cannot be
reported as fresh benchmark results.

[Architecture decision](docs/architecture-decision.md) · [Repair TODO](TODO.md) ·
[Initial regression report](docs/validation-2026-11-11.md) · [Manuscript](paper/README.md)

`dataset/`, `local_dev/`, weights and runtime outputs are ignored by Git. Private
server details are kept only in `local_dev/`.
