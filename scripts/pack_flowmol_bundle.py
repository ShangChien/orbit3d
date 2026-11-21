"""Stage a clean, self-contained Trisol code bundle for FlowMol corpus building & 16xA100 pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import stat

REPO_ROOT = Path(__file__).resolve().parent.parent


RUN_BUILD_CORPUS_SH = r"""#!/usr/bin/env bash
set -euo pipefail

echo "=== [FLOWMOL STAGE 2] CORPUS BUILD START @ $(date -u) on $(hostname) ==="
export PYTHONUNBUFFERED=1
export PYTHONPATH="/trisol/input/model/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

python3 -c "import rdkit" 2>/dev/null || pip install --no-cache-dir -q rdkit

OUT_DIR="${TRISOL_OUTPUT_DIR:-/trisol/output}"
WORK_DIR="${FLOWMOL_WORK_DIR:-/dev/shm/flowmol_corpus_work}"
SOURCES="${FLOWMOL_SOURCES:-qm9,oe62,pcqm4mv2,geom_drugs,tmqm}"
MAX_ATOMS="${FLOWMOL_MAX_ATOMS:-120}"
GEOM_CONFS="${FLOWMOL_GEOM_MAX_CONFS:-5}"
WORKERS="${FLOWMOL_WORKERS:-$(python3 -c 'import os; print(min(44, max(4, (os.cpu_count() or 8) - 2)))')}"

echo "OUT_DIR=${OUT_DIR} WORK_DIR=${WORK_DIR} SOURCES=${SOURCES} MAX_ATOMS=${MAX_ATOMS} WORKERS=${WORKERS}"
df -h / /dev/shm "${OUT_DIR}" || true

REUSE_ARGS=()
if [ -n "${FLOWMOL_REUSE_CORPUS_DIR:-}" ] && [ -f "${FLOWMOL_REUSE_CORPUS_DIR}/meta.json" ]; then
  REUSE_ARGS=(--reuse-corpus-dir "${FLOWMOL_REUSE_CORPUS_DIR}" --reuse-sources "${FLOWMOL_REUSE_SOURCES:-geom_drugs}")
fi

python3 /trisol/input/model/scripts/build_flowmol_corpus.py \
  --output-dir "${OUT_DIR}" \
  --work-dir "${WORK_DIR}" \
  --sources "${SOURCES}" \
  --max-atoms "${MAX_ATOMS}" \
  --geom-max-confs "${GEOM_CONFS}" \
  --workers "${WORKERS}" \
  "${REUSE_ARGS[@]}"

rm -rf "${WORK_DIR}"
ls -lh "${OUT_DIR}"
echo "=== [FLOWMOL STAGE 2] CORPUS BUILD DONE @ $(date -u) ==="
"""


RUN_PRECISION_AUDIT_SH = r"""#!/usr/bin/env bash
set -euo pipefail

echo "=== [FLOWMOL STAGE 3A] 16-GPU PRECISION AUDIT @ $(date -u) on $(hostname) ==="
export PYTHONUNBUFFERED=1
export PYTHONPATH="/trisol/input/model/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

python3 -c "import rdkit" 2>/dev/null || pip install --no-cache-dir -q rdkit

NNODES="${NNODES:-2}"
NPROC="${NPROC_PER_NODE:-8}"
NR="${NODE_RANK:-${JOB_COMPLETION_INDEX:-0}}"
MADDR="${MASTER_ADDR:?MASTER_ADDR must be set}"
MPORT="${MASTER_PORT:-29500}"

if [ -n "${FLOWMOL_CORPUS_DIR:-}" ]; then
  CORPUS_DIR="${FLOWMOL_CORPUS_DIR}"
elif [ -f "/trisol/input/datasets/ds-0/meta.json" ]; then
  CORPUS_DIR="/trisol/input/datasets/ds-0"
else
  CORPUS_DIR="/trisol/input/models/model-0"
fi
OUT_DIR="${TRISOL_OUTPUT_DIR:-/trisol/output}"
PRESET="${FLOWMOL_PRESET:-flagship}"
AUDIT_STEPS="${FLOWMOL_AUDIT_STEPS:-2000}"
MAX_ATOMS="${FLOWMOL_MAX_ATOMS:-120}"
MAX_EDGES="${FLOWMOL_MAX_EDGES:-75000}"
MAX_MOLS="${FLOWMOL_MAX_MOLS:-160}"
LR="${FLOWMOL_LR:-2.5e-4}"
WORKERS="${FLOWMOL_DATALOADER_WORKERS:-8}"
SOURCE_WEIGHTS="${FLOWMOL_SOURCE_WEIGHTS:-{\"oe62\":3.0,\"tmqm\":4.0,\"qm9\":1.5}}"
EXTRA_ARGS="${FLOWMOL_EXTRA_ARGS:-}"

echo "DIST: NNODES=${NNODES} NPROC=${NPROC} NR=${NR} MASTER=${MADDR}:${MPORT} PRESET=${PRESET} STEPS=${AUDIT_STEPS}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC}" \
  --node_rank="${NR}" \
  --master_addr="${MADDR}" \
  --master_port="${MPORT}" \
  /trisol/input/model/scripts/train_flowmol_pretrain.py \
  --corpus-dir "${CORPUS_DIR}" \
  --output-dir "${OUT_DIR}" \
  --preset "${PRESET}" \
  --precision-audit-steps "${AUDIT_STEPS}" \
  --max-atoms "${MAX_ATOMS}" \
  --max-edges "${MAX_EDGES}" \
  --max-mols-per-batch "${MAX_MOLS}" \
  --source-weights "${SOURCE_WEIGHTS}" \
  --lr "${LR}" \
  --warmup-steps 200 \
  --log-every 100 \
  --val-every 1000 \
  --sample-eval-mols 64 \
  --num-workers "${WORKERS}" \
  ${EXTRA_ARGS}

echo "=== [FLOWMOL STAGE 3A] DONE @ $(date -u) on $(hostname) ==="
"""


RUN_PRETRAIN_16GPU_SH = r"""#!/usr/bin/env bash
set -euo pipefail

echo "=== [FLOWMOL STAGE 3B] 16-GPU PRETRAINING START @ $(date -u) on $(hostname) ==="
export PYTHONUNBUFFERED=1
export PYTHONPATH="/trisol/input/model/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

python3 -c "import rdkit" 2>/dev/null || pip install --no-cache-dir -q rdkit

NNODES="${NNODES:-2}"
NPROC="${NPROC_PER_NODE:-8}"
NR="${NODE_RANK:-${JOB_COMPLETION_INDEX:-0}}"
MADDR="${MASTER_ADDR:?MASTER_ADDR must be set}"
MPORT="${MASTER_PORT:-29500}"

if [ -n "${FLOWMOL_CORPUS_DIR:-}" ]; then
  CORPUS_DIR="${FLOWMOL_CORPUS_DIR}"
elif [ -f "/trisol/input/datasets/ds-0/meta.json" ]; then
  CORPUS_DIR="/trisol/input/datasets/ds-0"
else
  CORPUS_DIR="/trisol/input/models/model-0"
fi
OUT_DIR="${TRISOL_OUTPUT_DIR:-/trisol/output}"
PRESET="${FLOWMOL_PRESET:-flagship}"
TARGET_SAMPLES="${FLOWMOL_TARGET_SAMPLES:-180000000}"
MAX_STEPS="${FLOWMOL_MAX_STEPS:-0}"
MAX_ATOMS="${FLOWMOL_MAX_ATOMS:-120}"
MAX_EDGES="${FLOWMOL_MAX_EDGES:-80000}"
MAX_MOLS="${FLOWMOL_MAX_MOLS:-160}"
LR="${FLOWMOL_LR:-2.5e-4}"
WARMUP="${FLOWMOL_WARMUP_STEPS:-2000}"
LOG_EVERY="${FLOWMOL_LOG_EVERY:-100}"
VAL_EVERY="${FLOWMOL_VAL_EVERY:-2500}"
SAVE_EVERY="${FLOWMOL_SAVE_EVERY:-5000}"
SAMPLE_MOLS="${FLOWMOL_SAMPLE_MOLS:-128}"
WORKERS="${FLOWMOL_DATALOADER_WORKERS:-8}"
SOURCE_WEIGHTS="${FLOWMOL_SOURCE_WEIGHTS:-{\"oe62\":3.0,\"tmqm\":4.0,\"qm9\":1.5}}"
GRAD_CKPT_FLAG=""
if [ "${FLOWMOL_GRAD_CKPT:-0}" = "1" ]; then
  GRAD_CKPT_FLAG="--grad-checkpointing"
fi

echo "DIST: NNODES=${NNODES} NPROC=${NPROC} NR=${NR} MASTER=${MADDR}:${MPORT}"
echo "CFG: PRESET=${PRESET} TARGET_SAMPLES=${TARGET_SAMPLES} MAX_STEPS=${MAX_STEPS} MAX_ATOMS=${MAX_ATOMS} MAX_EDGES=${MAX_EDGES} LR=${LR}"

CMD=(
  torchrun
  --nnodes="${NNODES}"
  --nproc_per_node="${NPROC}"
  --node_rank="${NR}"
  --master_addr="${MADDR}"
  --master_port="${MPORT}"
  /trisol/input/model/scripts/train_flowmol_pretrain.py
  --corpus-dir "${CORPUS_DIR}"
  --output-dir "${OUT_DIR}"
  --preset "${PRESET}"
  --target-samples "${TARGET_SAMPLES}"
  --max-steps "${MAX_STEPS}"
  --max-atoms "${MAX_ATOMS}"
  --max-edges "${MAX_EDGES}"
  --max-mols-per-batch "${MAX_MOLS}"
  --lr "${LR}"
  --warmup-steps "${WARMUP}"
  --log-every "${LOG_EVERY}"
  --val-every "${VAL_EVERY}"
  --save-every "${SAVE_EVERY}"
  --sample-eval-mols "${SAMPLE_MOLS}"
  --num-workers "${WORKERS}"
)
if [ -n "${SOURCE_WEIGHTS}" ]; then
  CMD+=(--source-weights "${SOURCE_WEIGHTS}")
fi
if [ -n "${GRAD_CKPT_FLAG}" ]; then
  CMD+=("${GRAD_CKPT_FLAG}")
fi

"${CMD[@]}"
echo "=== [FLOWMOL STAGE 3B] PRETRAINING DONE @ $(date -u) on $(hostname) ==="
"""


RUN_EVAL_CHECKPOINTS_SH = r"""#!/usr/bin/env bash
set -euo pipefail

echo "=== [FLOWMOL MULTI-CHECKPOINT N>=1024 BENCHMARK] START @ $(date -u) on $(hostname) ==="
export PYTHONUNBUFFERED=1
export PYTHONPATH="/trisol/input/model/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4

python3 -c "import rdkit" 2>/dev/null || pip install --no-cache-dir -q rdkit

NPROC="${NPROC_PER_NODE:-$(python3 -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
CAND_DIR="${FLOWMOL_CANDIDATES_DIR:-/trisol/input/models/model-0}"
if [ -n "${FLOWMOL_CORPUS_DIR:-}" ]; then
  CORPUS_DIR="${FLOWMOL_CORPUS_DIR}"
elif [ -f "/trisol/input/datasets/ds-0/meta.json" ]; then
  CORPUS_DIR="/trisol/input/datasets/ds-0"
else
  CORPUS_DIR="/trisol/input/models/model-1"
fi
OUT_DIR="${TRISOL_OUTPUT_DIR:-/trisol/output}"
N_MOLS="${FLOWMOL_EVAL_MOLS:-1024}"
VAL_BATCHES="${FLOWMOL_VAL_BATCHES:-300}"
BATCH_CHUNK="${FLOWMOL_BATCH_CHUNK:-64}"

torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  /trisol/input/model/scripts/eval_flowmol_checkpoints.py \
  --candidates-dir "${CAND_DIR}" \
  --corpus-dir "${CORPUS_DIR}" \
  --output-dir "${OUT_DIR}" \
  --n-mols "${N_MOLS}" \
  --val-batches "${VAL_BATCHES}" \
  --batch-chunk "${BATCH_CHUNK}"

echo "=== [FLOWMOL MULTI-CHECKPOINT BENCHMARK] DONE @ $(date -u) ==="
"""


def stage_bundle(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Copy src/loom excluding caches
    shutil.copytree(
        REPO_ROOT / "src" / "loom",
        dest / "src" / "loom",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store"),
    )

    scripts_out = dest / "scripts"
    scripts_out.mkdir(parents=True, exist_ok=True)
    for name in ("build_flowmol_corpus.py", "train_flowmol_pretrain.py", "eval_flowmol_checkpoints.py"):
        shutil.copy2(REPO_ROOT / "scripts" / name, scripts_out / name)

    for fname, body in (
        ("run_build_corpus.sh", RUN_BUILD_CORPUS_SH),
        ("run_precision_audit.sh", RUN_PRECISION_AUDIT_SH),
        ("run_pretrain_16gpu.sh", RUN_PRETRAIN_16GPU_SH),
        ("run_eval_checkpoints.sh", RUN_EVAL_CHECKPOINTS_SH),
    ):
        fpath = dest / fname
        fpath.write_text(body)
        fpath.chmod(fpath.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    total_bytes = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    n_files = sum(1 for p in dest.rglob("*") if p.is_file())
    print(f"Staged FlowMol Trisol bundle -> {dest} ({n_files} files, {total_bytes / 1024:.1f} KiB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, required=True)
    args = p.parse_args()
    stage_bundle(args.dest)


if __name__ == "__main__":
    main()
