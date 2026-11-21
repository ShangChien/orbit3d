#!/usr/bin/env bash
# Idempotent 16xA100 RDMA orchestrator for FlowMol-118:
#   1. flagship (40.19M, 180M samples) -> pretrained/flowmol/flagship/
#   2. 7-checkpoint cosine-tail large-sample benchmark (N=1024 x 7 suites/ckpt on 8xA100)
#      comparing step-152k (best_ema), step-155k (near 156k), step-160k, step-165k,
#      step-170k, step-175k, and final_ema (~177.2k) -> pretrained/flowmol/flagship/eval_1024/
#   3. standard (12.74M, 180M samples) -> pretrained/flowmol/standard/
#   4. small    ( 4.23M, 180M samples) -> pretrained/flowmol/small/

set -euo pipefail

TRISOL="${TRISOL_BIN:-/Users/dp/.local/bin/trisol}"
PYTHON="/Users/dp/project/mol_gen/.venv/bin/python"
OUT_ROOT="/Users/dp/project/mol_gen/pretrained/flowmol"
IMAGE="registry.dp.tech/infra/trisol/ms-swift:3.9.1-cu128-torch2.8-rdma"
DATASET="flowmol-corpus-118:1"
CODEBASE="flowmol-codebase:9"

mkdir -p "${OUT_ROOT}/flagship/candidates" "${OUT_ROOT}/flagship/eval_1024" "${OUT_ROOT}/standard" "${OUT_ROOT}/small"

wait_job_resilient() {
  local job_ref="$1"
  echo "[pipeline] waiting on job ${job_ref} @ $(date -u)..."
  while true; do
    if "${TRISOL}" train wait "${job_ref}" --poll-interval 25s --timeout 0; then
      break
    fi
    local st
    st=$("${TRISOL}" train get "${job_ref}" -o yaml 2>/dev/null | awk '/^status:/{print $2}' || true)
    case "${st}" in
      succeeded|failed|canceled|billing_failed) break ;;
    esac
    sleep 15
  done
  local final_st
  final_st=$("${TRISOL}" train get "${job_ref}" -o yaml | awk '/^status:/{print $2}')
  echo "[pipeline] job ${job_ref} reached terminal status=${final_st} @ $(date -u)"
  if [ "${final_st}" != "succeeded" ]; then
    echo "[pipeline] ERROR: job ${job_ref} ended with status=${final_st}" >&2
    "${TRISOL}" train logs "${job_ref}" --history --tail 80 >&2 || true
    exit 1
  fi
}

run_or_attach_stage() {
  local preset="$1"
  local job_name="$2"
  local out_model="$3"
  local max_edges="$4"
  local max_mols="$5"
  local dest_dir="${OUT_ROOT}/${preset}"

  if [ -f "${dest_dir}/best_ema.pt" ] && [ -f "${dest_dir}/train_summary.json" ]; then
    echo "[pipeline] stage '${preset}' already downloaded at ${dest_dir}, skipping."
    return 0
  fi

  local existing_st=""
  existing_st=$("${TRISOL}" train get "${job_name}" -o yaml 2>/dev/null | awk '/^status:/{print $2}' || true)

  if [ -z "${existing_st}" ]; then
    echo "[pipeline] submitting ${job_name} (preset=${preset}, max_edges=${max_edges}) @ $(date -u)..."
    "${TRISOL}" train submit "${job_name}" \
      --cluster w1 \
      --framework custom \
      --mode full \
      --image-ref "${IMAGE}" \
      --base-model "${CODEBASE}" \
      --dataset "${DATASET}" \
      --output-model "${out_model}" \
      --create-output-model \
      --checkpoint-prefix step- \
      --checkpoint-marker COMMITTED \
      --checkpoint-keep-last 6 \
      --nodes 2 \
      --gpus-per-node 8 \
      --transport rdma \
      --rdma-custom-image-ack \
      --command bash \
      --command /trisol/input/model/run_pretrain_16gpu.sh \
      --env "FLOWMOL_PRESET=${preset}" \
      --env "FLOWMOL_TARGET_SAMPLES=180000000" \
      --env "FLOWMOL_MAX_ATOMS=120" \
      --env "FLOWMOL_MAX_EDGES=${max_edges}" \
      --env "FLOWMOL_MAX_MOLS=${max_mols}" \
      --env "FLOWMOL_LR=2.5e-4" \
      --env "FLOWMOL_WARMUP_STEPS=2500" \
      --env "FLOWMOL_LOG_EVERY=100" \
      --env "FLOWMOL_VAL_EVERY=4000" \
      --env "FLOWMOL_SAVE_EVERY=5000" \
      --env "FLOWMOL_SAMPLE_MOLS=128" \
      --env "FLOWMOL_DATALOADER_WORKERS=8" \
      --description "Standalone 118-element FlowMol ${preset} 16xA100 RDMA Scaling Law pretraining (180M samples) on Wenyon flowmol-corpus-118:1"
  else
    echo "[pipeline] found existing job ${job_name} (status=${existing_st}), attaching..."
  fi

  wait_job_resilient "${job_name}"

  echo "[pipeline] downloading ${out_model}:latest -> ${dest_dir}..."
  mkdir -p "${dest_dir}"
  "${TRISOL}" model download "${out_model}:latest" -o "${dest_dir}" --quiet
  ls -lh "${dest_dir}"
}

run_flagship_tail_benchmark() {
  local eval_dir="${OUT_ROOT}/flagship/eval_1024"
  local cand_dir="${OUT_ROOT}/flagship/candidates"
  if [ -f "${eval_dir}/checkpoint_eval_summary.json" ]; then
    echo "[pipeline] flagship 7-checkpoint N=1024 benchmark already complete at ${eval_dir}, skipping."
    return 0
  fi

  cp -f "${OUT_ROOT}/flagship/best_ema.pt" "${cand_dir}/step152k_best_ema.pt"
  cp -f "${OUT_ROOT}/flagship/final_ema.pt" "${cand_dir}/final_ema.pt"
  ls -lh "${cand_dir}"

  if ! "${TRISOL}" model get flowmol-flagship-candidates >/dev/null 2>&1; then
    "${TRISOL}" model create flowmol-flagship-candidates --description "7 cosine-tail EMA checkpoints of FlowMol-118 flagship for N=1024 evaluation" >/dev/null
  fi
  local cand_ver="v1"
  if ! "${TRISOL}" model get flowmol-flagship-candidates | grep -q '"version_code": 1'; then
    echo "[pipeline] uploading 7 candidate EMA checkpoints -> flowmol-flagship-candidates:1..."
    "${TRISOL}" model upload flowmol-flagship-candidates "${cand_dir}" --version "${cand_ver}" --progress=none -y
  fi

  local eval_job="flowmol-flagship-ckpt-eval-v1"
  local existing_st=""
  existing_st=$("${TRISOL}" train get "${eval_job}" -o yaml 2>/dev/null | awk '/^status:/{print $2}' || true)
  if [ -z "${existing_st}" ]; then
    echo "[pipeline] submitting 8xA100 N=1024 7-checkpoint benchmark ${eval_job} @ $(date -u)..."
    "${TRISOL}" train submit "${eval_job}" \
      --cluster w1 \
      --framework custom \
      --mode full \
      --image-ref "${IMAGE}" \
      --base-model "${CODEBASE}" \
      --model "flowmol-flagship-candidates:1" \
      --dataset "${DATASET}" \
      --output-model "flowmol-flagship-eval-out" \
      --create-output-model \
      --checkpoint-disable \
      --gpu-count 8 \
      --command bash \
      --command /trisol/input/model/run_eval_checkpoints.sh \
      --env "FLOWMOL_CANDIDATES_DIR=/trisol/input/models/model-0" \
      --env "FLOWMOL_EVAL_MOLS=1024" \
      --env "FLOWMOL_VAL_BATCHES=300" \
      --env "FLOWMOL_BATCH_CHUNK=64" \
      --description "Parallel 8xA100 N=1024 x 7-suite generation + 300-batch val benchmark across 7 cosine-tail flagship checkpoints (152k..final)"
  else
    echo "[pipeline] attaching to existing ${eval_job} (status=${existing_st})..."
  fi

  wait_job_resilient "${eval_job}"
  "${TRISOL}" model download "flowmol-flagship-eval-out:latest" -o "${eval_dir}" --quiet

  # Promote the N=1024 composite generation winner to OUT_ROOT/selected_best_ema.pt & OUT_ROOT/best_ema.pt
  "${PYTHON}" - "${eval_dir}/checkpoint_eval_summary.json" "${cand_dir}" "${OUT_ROOT}" <<'PY'
import json, shutil, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
cand_dir, out_root = Path(sys.argv[2]), Path(sys.argv[3])
winner = s.get("best_by_composite_generation") or "step152k_best_ema.pt"
src = cand_dir / winner
if src.exists():
    shutil.copy2(src, out_root / "selected_best_ema.pt")
    shutil.copy2(src, out_root / "best_ema.pt")
    print(f"[pipeline] promoted N=1024 benchmark winner '{winner}' -> {out_root / 'best_ema.pt'}")
PY
}

# 1. Flagship (40.19M)
run_or_attach_stage "flagship" "flowmol-pretrain-flagship-16gpu-v1" "flowmol-pretrained-v1" "95000" "192"
cp -f "${OUT_ROOT}/flagship/best_ema.pt" "${OUT_ROOT}/best_ema.pt"
cp -f "${OUT_ROOT}/flagship/final_ema.pt" "${OUT_ROOT}/final_ema.pt"
cp -f "${OUT_ROOT}/flagship/config.json" "${OUT_ROOT}/config.json"
cp -f "${OUT_ROOT}/flagship/train_summary.json" "${OUT_ROOT}/train_summary.json"

# 2. Large-sample (N=1024 x 7 suites) evaluation of all 7 cosine-tail checkpoints (152k..final)
run_flagship_tail_benchmark

# 3. Standard (12.74M) Scaling Law tier
run_or_attach_stage "standard" "flowmol-pretrain-standard-16gpu-v1" "flowmol-pretrained-standard-v1" "125000" "224"

# 4. Small (4.23M) Scaling Law tier
run_or_attach_stage "small" "flowmol-pretrain-small-16gpu-v1" "flowmol-pretrained-small-v1" "160000" "256"

# 5. Unified N=1024 x 7-suite benchmark for standard & small (apple-to-apple with flagship seed=202611)
SCALE_EVAL_DIR="${OUT_ROOT}/scaling_eval_1024"
if [ ! -f "${SCALE_EVAL_DIR}/checkpoint_eval_summary.json" ]; then
  STAGE_CANDS="/tmp/flowmol_scaling_cands"
  rm -rf "${STAGE_CANDS}" && mkdir -p "${STAGE_CANDS}" "${SCALE_EVAL_DIR}"
  cp -f "${OUT_ROOT}/small/best_ema.pt" "${STAGE_CANDS}/small_best_ema.pt"
  cp -f "${OUT_ROOT}/small/final_ema.pt" "${STAGE_CANDS}/small_final_ema.pt"
  cp -f "${OUT_ROOT}/standard/best_ema.pt" "${STAGE_CANDS}/standard_best_ema.pt"
  cp -f "${OUT_ROOT}/standard/final_ema.pt" "${STAGE_CANDS}/standard_final_ema.pt"

  if ! "${TRISOL}" model get flowmol-scaling-candidates >/dev/null 2>&1; then
    "${TRISOL}" model create flowmol-scaling-candidates --description "Standard (12.74M) and Small (4.23M) EMA weights for N=1024 Scaling Law eval" >/dev/null
  fi
  if ! "${TRISOL}" model get flowmol-scaling-candidates | grep -q '"version_code": 1'; then
    "${TRISOL}" model upload flowmol-scaling-candidates "${STAGE_CANDS}" --version v1 --progress=none -y
  fi
  rm -rf "${STAGE_CANDS}"

  SCALE_EVAL_JOB="flowmol-scaling-ckpt-eval-v1"
  existing_se=$("${TRISOL}" train get "${SCALE_EVAL_JOB}" -o yaml 2>/dev/null | awk '/^status:/{print $2}' || true)
  if [ -z "${existing_se}" ]; then
    "${TRISOL}" train submit "${SCALE_EVAL_JOB}" \
      --cluster w1 \
      --framework custom \
      --mode full \
      --image-ref "${IMAGE}" \
      --base-model "${CODEBASE}" \
      --model "flowmol-scaling-candidates:1" \
      --dataset "${DATASET}" \
      --output-model "flowmol-scaling-eval-out" \
      --create-output-model \
      --checkpoint-disable \
      --gpu-count 4 \
      --command bash \
      --command /trisol/input/model/run_eval_checkpoints.sh \
      --env "FLOWMOL_CANDIDATES_DIR=/trisol/input/models/model-0" \
      --env "FLOWMOL_EVAL_MOLS=1024" \
      --env "FLOWMOL_VAL_BATCHES=300" \
      --env "FLOWMOL_BATCH_CHUNK=64" \
      --description "4xA100 N=1024 x 7-suite Scaling Law benchmark for standard (12.74M) and small (4.23M)"
  fi
  wait_job_resilient "${SCALE_EVAL_JOB}"
  "${TRISOL}" model download "flowmol-scaling-eval-out:latest" -o "${SCALE_EVAL_DIR}" --quiet
fi

echo "=== ALL FLOWMOL-118 BENCHMARKS & SCALING LAW TIERS COMPLETE @ $(date -u) ==="
