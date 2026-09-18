#!/bin/bash
# Easy launcher for humanibench_eval.sbatch (Vector HPC, two nodes).
# Set only what you want to change; CONFIG in the sbatch / exported env fills the rest.
#
# Required once (venv prefixes + judge weights). Paste both env paths
# (same path twice is fine). Optional: CUDA_MODULE=cuda/12.2 or none (Leonardo skip).
#   export JUDGE_CONDA_ENV=/path/to/judge-env
#   export EVAL_CONDA_ENV=/path/to/eval-env
#   export JUDGE_MODEL_PATH=/path/or/hf-id
#
# Examples
#   ./run.sh --partition <name> --account <name>                          # full group, LIMIT=0
#   ./run.sh --partition <name> --tasks humanibench_t3 --limit 5
#   ./run.sh --partition <name> --tasks humanibench_t1_plain --limit 5
#   ./run.sh --partition <name> --tasks humanibench --limit 0
#   ./run.sh --partition <name> --model Qwen/Qwen2.5-VL-3B-Instruct
#   ./run.sh --cuda-module cuda/12.2                 # pin CUDA module (default: cuda)
#
# After it finishes:
#   results/<judge>/<model>/<task>/SUMMARY.txt
#   results/<judge>/<model>/<task>/humanibench_splits.json
#   logs/humanibench-eval-<jobid>.{out,err}
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
SBATCH="$HERE/humanibench_eval.sbatch"
mkdir -p "$HERE/logs" "$HERE/results"

# Optional repo .env (HF_HOME, HF_TOKEN, JUDGE_*). Already-exported vars win if you set them first.
if [ -f "$REPO/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO/.env"
  set +a
fi
export HERE REPO

MODEL=""; TASKS=""; LIMIT=""; MODEL_ARGS=""
JUDGE_MODEL=""; JUDGE_ENV=""; JUDGE_NAME=""; JUDGE_TP=""; EVAL_ENV=""; TIME=""
PARTITION=""; ACCOUNT=""; CUDA_MOD=""

while [ $# -gt 0 ]; do
  case "$1" in
    --model)             MODEL="$2"; shift 2 ;;
    --model-args)        MODEL_ARGS="$2"; shift 2 ;;
    --tasks)             TASKS="$2"; shift 2 ;;
    --limit)             LIMIT="$2"; shift 2 ;;
    --judge-model)       JUDGE_MODEL="$2"; shift 2 ;;
    --judge-served-name) JUDGE_NAME="$2"; shift 2 ;;
    --judge-tp)          JUDGE_TP="$2"; shift 2 ;;
    --judge-env)         JUDGE_ENV="$2"; shift 2 ;;
    --eval-env)          EVAL_ENV="$2"; shift 2 ;;
    --cuda-module)       CUDA_MOD="$2"; shift 2 ;;
    --partition)         PARTITION="$2"; shift 2 ;;
    --account)           ACCOUNT="$2"; shift 2 ;;
    --time)              TIME="$2"; shift 2 ;;
    -h|--help)           sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

if [ -n "$MODEL" ] && [ -z "$MODEL_ARGS" ]; then
  MODEL_ARGS="pretrained=${MODEL}"
fi

if [ -n "$MODEL_ARGS" ]; then export MODEL_ARGS; fi
if [ -n "$TASKS" ];      then export TASKS; fi
if [ -n "$LIMIT" ];      then export LIMIT; fi
if [ -n "$JUDGE_MODEL" ];then export JUDGE_MODEL_PATH="$JUDGE_MODEL"; fi
if [ -n "$JUDGE_NAME" ]; then export JUDGE_SERVED_NAME="$JUDGE_NAME"; fi
if [ -n "$JUDGE_TP" ];   then export JUDGE_TP; fi
if [ -n "$JUDGE_ENV" ];  then export JUDGE_CONDA_ENV="$JUDGE_ENV"; fi
if [ -n "$EVAL_ENV" ];   then export EVAL_CONDA_ENV="$EVAL_ENV"; fi
if [ -n "$CUDA_MOD" ];   then export CUDA_MODULE="$CUDA_MOD"; fi

SB_ARGS=(--export=ALL)
if [ -n "$PARTITION" ]; then SB_ARGS+=(--partition="$PARTITION"); fi
if [ -n "$ACCOUNT" ]; then SB_ARGS+=(--account="$ACCOUNT"); fi
if [ -n "$TIME" ]; then SB_ARGS+=(--time="$TIME"); fi

echo "submitting from $HERE"
echo "overrides:"
for v in MODEL_ARGS TASKS LIMIT JUDGE_MODEL_PATH JUDGE_SERVED_NAME JUDGE_TP JUDGE_CONDA_ENV EVAL_CONDA_ENV CUDA_MODULE; do
  if [ -n "${!v:-}" ]; then echo "  $v=${!v}"; fi
done
sbatch "${SB_ARGS[@]}" "$SBATCH"
