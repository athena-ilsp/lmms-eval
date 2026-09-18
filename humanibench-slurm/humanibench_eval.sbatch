#!/bin/bash
#SBATCH --job-name=humanibench-eval
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Two-node HumaniBench eval on Vector HPC.
#   NODE 0 -> vLLM judge (Gemma-4-31B-it, TP=4, all 4 A40s)
#   NODE 1 -> lmms_eval; talks to the judge by hostname:port
#
# 3d covers any single task: the longest (t4_closed, 6865 rows) is ~38h, about
# 12h generating + 26h judging. One job for the whole `humanibench` group runs
# all 10 back to back - pass --time 7-00:00:00 for that.
# 08:00:00 / a40_b2 is only enough for --limit smoke tests.
#
# Partition / account are cluster-specific. Pass them on the sbatch line:
#   ./run.sh --partition <name> --account <name>
# or: sbatch --partition <name> --account <name> humanibench_eval.sbatch
#
# Required env (set in CONFIG below, or export before ./run.sh):
#   JUDGE_CONDA_ENV  EVAL_CONDA_ENV  JUDGE_MODEL_PATH
set -euo pipefail

# ===================== CONFIG (override via environment) =====================
HERE="${HERE:-$(pwd)}"
REPO="${REPO:-$(cd "${HERE}/.." && pwd)}"

# HuggingFace cache (Leonardo: set HF_HOME to the work filesystem). Optional HF_TOKEN.
HF_HOME_DIR="${HF_HOME:-${HF_HOME_DIR:-$HOME/.cache/huggingface}}"

# Python prefixes with bin/activate (venv, uv, or conda env prefix). Paste both paths
# (same path twice is fine on Vector; Leonardo uses two envs).
JUDGE_CONDA_ENV="${JUDGE_CONDA_ENV:-}"
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-}"

# Vector: module load cuda before vLLM. Leonardo: CUDA_MODULE=none to skip.
CUDA_MODULE="${CUDA_MODULE:-cuda}"

# --- model under test ---
MODEL_NAME="${MODEL_NAME:-qwen2_5_vl}"
if [ -z "${MODEL_ARGS:-}" ]; then
  MODEL_ARGS="pretrained=Qwen/Qwen2.5-VL-3B-Instruct"
fi

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
LIMIT="${LIMIT:-0}"    # 0 or empty = full run. T7 / group humanibench require 0.

# --- judge (vLLM OpenAI-compatible server on NODE 0) ---
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-gemma4-31b-judge}"
JUDGE_TP="${JUDGE_TP:-4}"   # 31B bf16 ~62GB; TP=4 uses the node's 4x A40 (48GB). TP=2 also fits.
JUDGE_PORT="${JUDGE_PORT:-$(( 20000 + RANDOM % 20000 ))}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-16384}"
JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.85}"
JUDGE_MAX_WAIT="${JUDGE_MAX_WAIT:-3600}"

TASKS="${TASKS:-humanibench}"   # all 10 leaves, including humanibench_t7. Do not pass --limit.

# Per-task array: SLURM_ARRAY_TASK_ID indexes HB_TASKS (comma-separated).
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  IFS=',' read -ra _HB_TASKS <<< "${HB_TASKS:-humanibench_t1_plain,humanibench_t1_cot,humanibench_t2,humanibench_t3,humanibench_t4_closed,humanibench_t4_open,humanibench_t5,humanibench_t6_factual,humanibench_t6_empathic,humanibench_t7}"
  TASKS="${_HB_TASKS[$SLURM_ARRAY_TASK_ID]}"
  echo "[array] ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} -> ${TASKS}"
fi
# =============================================================================

missing=()
[ -n "$JUDGE_CONDA_ENV" ] || missing+=(JUDGE_CONDA_ENV)
[ -n "$EVAL_CONDA_ENV" ] || missing+=(EVAL_CONDA_ENV)
[ -n "$JUDGE_MODEL_PATH" ] || missing+=(JUDGE_MODEL_PATH)
if [ "${#missing[@]}" -gt 0 ]; then
  echo "[config] ERROR: set these before submitting: ${missing[*]}"
  echo "         They must reach the job:  sbatch --export=ALL humanibench_eval.sbatch"
  echo "         (plain sbatch does not pass your shell exports.)"
  echo "         Example: export JUDGE_CONDA_ENV=/path/to/judge-env"
  echo "                  export EVAL_CONDA_ENV=/path/to/eval-env"
  echo "                  export JUDGE_MODEL_PATH=/path/or/hf-id"
  echo "         Use the env prefix that has bin/activate (echo \$VIRTUAL_ENV or \$CONDA_PREFIX)."
  exit 1
fi

mkdir -p "${HERE}/logs" "${HERE}/results"

# T7 retention needs clean + attacked rows in the same run. The group `humanibench` includes T7.
if [ -n "${LIMIT:-}" ] && [ "${LIMIT}" != "0" ]; then
  IFS=',' read -ra _TASK_LIST <<< "$TASKS"
  for _t in "${_TASK_LIST[@]}"; do
    _t="${_t// /}"
    if [ "$_t" = "humanibench" ] || [ "$_t" = "humanibench_t7" ]; then
      echo "[config] ERROR: --limit is invalid for T7 (retention needs clean + attacked rows)."
      echo "         Use LIMIT=0, or drop humanibench_t7 / the humanibench group."
      exit 1
    fi
  done
fi

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
JUDGE_NODE="${NODES[0]}"
EVAL_NODE="${NODES[1]}"
JUDGE_URL="http://${JUDGE_NODE}:${JUDGE_PORT}/v1"
SERVER_LOG="${HERE}/logs/judge_server_${SLURM_JOB_ID}.log"
MODEL_TAG=$(echo "${MODEL_ARGS}" | sed -nE 's/.*(pretrained|model)=([^,]+).*/\2/p' | sed 's#.*/##; s#[^A-Za-z0-9._-]#_#g')
MODEL_TAG="${MODEL_TAG:-model}"
TASK_TAG=$(echo "${TASKS}" | tr ',' '+' | sed 's#[[:space:]]##g; s#[^A-Za-z0-9._+-]#_#g')
TASK_TAG="${TASK_TAG:-tasks}"

# Fully-qualified id for this job. In an array element SLURM_JOB_ID is the
# element's own id, EXCEPT for the element that inherits the array base id --
# and `scancel <base id>` means the WHOLE array. Always signal JOB_LABEL.
JOB_LABEL="${SLURM_JOB_ID}"
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
  JOB_LABEL="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi

# Judge identity comes from the weights, not the served alias (the alias is
# free-form; the weights are what actually move the scores). Override: JUDGE_TAG.
JUDGE_TAG="${JUDGE_TAG:-$(printf '%s' "${JUDGE_MODEL_PATH}" | sed 's#/*$##; s#.*/##; s#[^A-Za-z0-9._-]#_#g')}"
JUDGE_TAG="${JUDGE_TAG:-${JUDGE_SERVED_NAME}}"

#   results/<judge>/<model>/<task>/           <- this run
#   results/<judge>/<model>/SUMMARY.txt       <- overall, written by collect_results.py
# No job ids in the path; the id lives in RUN.json and in the log filename.
MODEL_ROOT="${HERE}/results/${JUDGE_TAG}/${MODEL_TAG}"
RESULTS_DIR="${RESULTS_DIR:-${MODEL_ROOT}/${TASK_TAG}}"

# A re-run of the same task would land on the same path: park whatever is there
# under _superseded/ rather than overwrite a good result with a half-finished one.
if [ -d "${RESULTS_DIR}" ] && [ -n "$(ls -A "${RESULTS_DIR}" 2>/dev/null)" ]; then
  ARCHIVE="${MODEL_ROOT}/_superseded/${TASK_TAG}_$(date +%Y%m%d-%H%M%S)_${SLURM_JOB_ID}"
  mkdir -p "$(dirname "${ARCHIVE}")"
  mv "${RESULTS_DIR}" "${ARCHIVE}"
  echo "[results] previous ${TASK_TAG} run parked at ${ARCHIVE}"
fi
mkdir -p "${RESULTS_DIR}"

if [ -n "${LIMIT:-}" ] && [ "${LIMIT}" != "0" ]; then
  LIMIT_FLAG="--limit ${LIMIT}"
  echo "[config] capped run: --limit ${LIMIT} per task"
else
  LIMIT_FLAG=""
  echo "[config] full run (no --limit)"
fi

echo "=========================================="
echo "Repo       : $REPO"
echo "Judge env  : $JUDGE_CONDA_ENV"
echo "Eval env   : $EVAL_CONDA_ENV"
echo "HF_HOME    : $HF_HOME_DIR"
echo "CUDA module: ${CUDA_MODULE}  (none = skip module load)"
echo "Judge node : $JUDGE_NODE  (${JUDGE_SERVED_NAME}, TP=${JUDGE_TP})"
echo "Eval node  : $EVAL_NODE   (${MODEL_NAME})"
echo "Judge URL  : $JUDGE_URL"
echo "Tasks      : $TASKS"
echo "Results    : $RESULTS_DIR"
echo "Server log : $SERVER_LOG"
echo "=========================================="

# RUN.json: one machine-readable breadcrumb per run, written before the work
# starts so even a killed run says what it was. $1=status $2=exit code.
write_manifest() {
  RUN_STATUS="$1" RUN_EXIT="${2:-}" \
  RUN_DIR="$RESULTS_DIR" RUN_MODEL="$MODEL_NAME" RUN_MODEL_ARGS="$MODEL_ARGS" \
  RUN_MODEL_TAG="$MODEL_TAG" RUN_TASKS="$TASKS" RUN_LIMIT="${LIMIT:-0}" \
  RUN_JOB_LABEL="$JOB_LABEL" RUN_JOB_ID="${SLURM_JOB_ID}" \
  RUN_ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-}" RUN_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}" \
  RUN_JUDGE_MODEL="$JUDGE_MODEL_PATH" RUN_JUDGE_NAME="$JUDGE_SERVED_NAME" \
  RUN_JUDGE_TAG="$JUDGE_TAG" \
  RUN_JUDGE_TP="$JUDGE_TP" RUN_JUDGE_URL="$JUDGE_URL" \
  RUN_JUDGE_NODE="$JUDGE_NODE" RUN_EVAL_NODE="$EVAL_NODE" \
  RUN_STARTED="$RUN_STARTED" RUN_LOG_OUT="${HERE}/logs/humanibench-eval-${SLURM_JOB_ID}.out" \
  RUN_JUDGE_LOG="$SERVER_LOG" \
  python3 - <<'PY' || echo "[manifest] WARNING: could not write RUN.json"
import json, os, time
env = os.environ.get
tasks = [t.strip() for t in env("RUN_TASKS", "").split(",") if t.strip()]
try:
    limit = float(env("RUN_LIMIT") or 0)
    limit = int(limit) if limit == int(limit) else limit
except ValueError:
    limit = env("RUN_LIMIT")
exit_code = env("RUN_EXIT") or ""
status = env("RUN_STATUS")
manifest = {
    "schema": 1,
    "status": status,
    "exit_code": int(exit_code) if exit_code.strip().lstrip("-").isdigit() else None,
    "model": env("RUN_MODEL"),
    "model_args": env("RUN_MODEL_ARGS"),
    "model_tag": env("RUN_MODEL_TAG"),
    "tasks": tasks,
    "limit": limit or None,
    "job_label": env("RUN_JOB_LABEL"),
    "job_id": env("RUN_JOB_ID"),
    "array_job_id": env("RUN_ARRAY_JOB_ID") or None,
    "array_task_id": env("RUN_ARRAY_TASK_ID") or None,
    "judge": {
        "tag": env("RUN_JUDGE_TAG"),
        "model": env("RUN_JUDGE_MODEL"),
        "served_name": env("RUN_JUDGE_NAME"),
        "tensor_parallel": int(env("RUN_JUDGE_TP") or 0) or None,
        "url": env("RUN_JUDGE_URL"),
    },
    "nodes": {"judge": env("RUN_JUDGE_NODE"), "eval": env("RUN_EVAL_NODE")},
    "started": env("RUN_STARTED") or None,
    "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z") if status != "running" else None,
    "logs": {"job": env("RUN_LOG_OUT"), "judge": env("RUN_JUDGE_LOG")},
}
path = os.path.join(env("RUN_DIR"), "RUN.json")
tmp = path + ".tmp"
with open(tmp, "w") as fh:
    json.dump(manifest, fh, indent=2)
    fh.write("\n")
os.replace(tmp, path)
PY
}

activate_cmd() {
  # $1 = env prefix with bin/activate. set +u: venv activate trips on unset vars.
  printf '%s\n' \
    "if [ ! -x \"$1/bin/python\" ]; then echo \"ERROR: $1/bin/python not found\"; exit 127; fi" \
    "set +u" \
    "if [ -f \"$1/bin/activate\" ]; then source \"$1/bin/activate\"; fi" \
    "export PATH=\"$1/bin:\$PATH\""
}

load_cuda_cmd() {
  if [ "$CUDA_MODULE" = "none" ] || [ -z "$CUDA_MODULE" ]; then
    printf '%s\n' "echo '[cuda] skip module load (CUDA_MODULE=${CUDA_MODULE:-empty})'"
    return
  fi
  printf '%s\n' \
    "set +u" \
    "source /etc/profile >/dev/null 2>&1 || true" \
    "if type module >/dev/null 2>&1; then module load ${CUDA_MODULE}; echo \"[cuda] loaded ${CUDA_MODULE} on \$(hostname)\"; else echo '[cuda] WARNING: module command not found, continuing'; fi"
}

RUN_STARTED="$(date +%Y-%m-%dT%H:%M:%S%z)"
write_manifest running
echo "[manifest] ${RESULTS_DIR}/RUN.json"

# ---- 1. start the judge server on NODE 0 (background srun step) ----
srun --nodes=1 --ntasks=1 --nodelist="$JUDGE_NODE" --gres=gpu:a40:4 \
  bash -c "
    set -e
    cd \"\$HOME\"
    $(load_cuda_cmd)
    $(activate_cmd "$JUDGE_CONDA_ENV")
    export HF_HOME='${HF_HOME_DIR}'
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONNOUSERSITE=1
    echo \"[judge] launching on \$(hostname) via ${JUDGE_CONDA_ENV}/bin/vllm\"
    exec '${JUDGE_CONDA_ENV}/bin/vllm' serve '${JUDGE_MODEL_PATH}' \
      --served-model-name '${JUDGE_SERVED_NAME}' \
      --tensor-parallel-size ${JUDGE_TP} \
      --max-model-len ${JUDGE_MAX_LEN} \
      --gpu-memory-utilization ${JUDGE_GPU_UTIL} \
      --api-key '${JUDGE_API_KEY}' \
      --host 0.0.0.0 --port ${JUDGE_PORT} \
      --trust-remote-code
  " > "$SERVER_LOG" 2>&1 &
SERVER_STEP_PID=$!

cleanup() {
  echo "[cleanup] stopping judge server step (pid $SERVER_STEP_PID)"
  kill "$SERVER_STEP_PID" 2>/dev/null || true
  # Killed mid-run (scancel, time limit, node failure): leave that in RUN.json
  # instead of a stale "running".
  if [ -f "${RESULTS_DIR}/RUN.json" ] && grep -q '"status": "running"' "${RESULTS_DIR}/RUN.json"; then
    write_manifest interrupted
  fi
  # JOB_LABEL, never SLURM_JOB_ID: the array element holding the base id would
  # otherwise SIGTERM every sibling still running.
  scancel --signal=TERM "${JOB_LABEL}" 2>/dev/null || true
}
trap cleanup EXIT

# ---- 2. wait until the judge is up (served-model-name in /v1/models) ----
echo "[wait] polling ${JUDGE_URL}/models for '${JUDGE_SERVED_NAME}' (up to $((JUDGE_MAX_WAIT/60)) min)..."
START=$(date +%s)
READY=0
while [ $(($(date +%s) - START)) -lt "$JUDGE_MAX_WAIT" ]; do
  if ! kill -0 "$SERVER_STEP_PID" 2>/dev/null; then
    echo "[wait] ERROR: judge server step exited early. Last log:"; tail -50 "$SERVER_LOG"; exit 1
  fi
  if srun --nodes=1 --ntasks=1 --nodelist="$EVAL_NODE" --overlap --gres=none \
        curl -sf "http://${JUDGE_NODE}:${JUDGE_PORT}/v1/models" \
        -H "Authorization: Bearer ${JUDGE_API_KEY}" 2>/dev/null \
        | grep -q "\"${JUDGE_SERVED_NAME}\""; then
    READY=1
    echo "[wait] judge READY after $(($(date +%s) - START))s"
    break
  fi
  sleep 10
done
if [ "$READY" -ne 1 ]; then
  echo "[wait] ERROR: judge not ready in time. Last log:"; tail -50 "$SERVER_LOG"; exit 1
fi

# ---- 3. run lmms_eval on NODE 1 ----
# set +e so a failing eval still reaches the summary + manifest below.
set +e
srun --nodes=1 --ntasks=1 --nodelist="$EVAL_NODE" --gres=gpu:a40:4 --overlap \
  bash -c "
    set -e
    $(load_cuda_cmd)
    $(activate_cmd "$EVAL_CONDA_ENV")
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export HF_HOME='${HF_HOME_DIR}'
    export HF_HUB_OFFLINE=1
    export HF_TOKEN='${HF_TOKEN:-}'
    export LMMS_EVAL_DATASETS_CACHE=\"\$HF_HOME/datasets\"
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONNOUSERSITE=1

    export API_TYPE='openai'
    export OPENAI_API_KEY='${JUDGE_API_KEY}'
    export OPENAI_API_URL='${JUDGE_URL}'
    export MODEL_VERSION='${JUDGE_SERVED_NAME}'

    cd '${REPO}'
    echo \"[eval] python=${EVAL_CONDA_ENV}/bin/python host=\$(hostname)\"
    '${EVAL_CONDA_ENV}/bin/python' -B -m lmms_eval \
      --model '${MODEL_NAME}' \
      --model_args '${MODEL_ARGS}' \
      --tasks '${TASKS}' \
      --batch_size ${EVAL_BATCH_SIZE} ${LIMIT_FLAG} \
      --output_path '${RESULTS_DIR}' \
      --log_samples
  "
EVAL_RC=$?
set -e

# ---- 4. SUMMARY.txt + per-split JSON + final manifest ----
python3 "${HERE}/summarize_results.py" \
  "$RESULTS_DIR" "$JOB_LABEL" "$MODEL_ARGS" "$TASKS" "$LIMIT" || true
python3 "${HERE}/group_samples.py" "$RESULTS_DIR" || true

if [ "$EVAL_RC" -eq 0 ]; then write_manifest ok "$EVAL_RC"; else write_manifest failed "$EVAL_RC"; fi

echo "[done] eval exit code: $EVAL_RC"
echo "[done] results dir  : $RESULTS_DIR   (see SUMMARY.txt and humanibench_splits.json)"
exit $EVAL_RC
