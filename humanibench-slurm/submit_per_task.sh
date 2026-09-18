#!/bin/bash
# Submit one SLURM array element per HumaniBench leaf. At most --max jobs run
# at once; the next starts when one finishes. Each task gets its own 2-node
# job (judge + eval) and writes results/<judge>/<model>/<task>/.
#
#   export JUDGE_CONDA_ENV=... EVAL_CONDA_ENV=...
#   ./submit_per_task.sh --partition a40_b5 --account aieng --max 3 \
#     --model-args 'pretrained=Qwen/Qwen2.5-VL-7B-Instruct,device_map=cuda:0' \
#     --judge-model google/gemma-4-31b-it --judge-served-name gemma4-31b-judge --judge-tp 4
#
# A small CPU-only collect job is queued behind the array
# (--dependency=afterany) and writes results/<judge>/<model>/SUMMARY.txt once
# the last element leaves the queue. --no-collect skips it.
#
# Skip a task already running (e.g. t6_factual):
#   --tasks humanibench_t1_plain,humanibench_t1_cot,humanibench_t2,humanibench_t3,humanibench_t4_closed,humanibench_t4_open,humanibench_t5,humanibench_t6_empathic,humanibench_t7
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MAX=3
TASKS="humanibench_t1_plain,humanibench_t1_cot,humanibench_t2,humanibench_t3,humanibench_t4_closed,humanibench_t4_open,humanibench_t5,humanibench_t6_factual,humanibench_t6_empathic,humanibench_t7"
MODEL=""; MODEL_ARGS=""; JUDGE_MODEL=""; JUDGE_NAME=""; JUDGE_TP=""
PARTITION=""; ACCOUNT=""; TIME=""
COLLECT=1; COLLECT_PARTITION=""; COLLECT_TIME=""

while [ $# -gt 0 ]; do
  case "$1" in
    --max)               MAX="$2"; shift 2 ;;
    --tasks)             TASKS="$2"; shift 2 ;;
    --model)             MODEL="$2"; shift 2 ;;
    --model-args)        MODEL_ARGS="$2"; shift 2 ;;
    --judge-model)       JUDGE_MODEL="$2"; shift 2 ;;
    --judge-served-name) JUDGE_NAME="$2"; shift 2 ;;
    --judge-tp)          JUDGE_TP="$2"; shift 2 ;;
    --partition)         PARTITION="$2"; shift 2 ;;
    --account)           ACCOUNT="$2"; shift 2 ;;
    --time)              TIME="$2"; shift 2 ;;
    --no-collect)        COLLECT=0; shift ;;
    --collect-partition) COLLECT_PARTITION="$2"; shift 2 ;;
    --collect-time)      COLLECT_TIME="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

if [ -n "$MODEL" ] && [ -z "$MODEL_ARGS" ]; then
  MODEL_ARGS="pretrained=${MODEL}"
fi
if [ -n "$MODEL_ARGS" ]; then export MODEL_ARGS; fi
if [ -n "$JUDGE_MODEL" ]; then export JUDGE_MODEL_PATH="$JUDGE_MODEL"; fi
if [ -n "$JUDGE_NAME" ]; then export JUDGE_SERVED_NAME="$JUDGE_NAME"; fi
if [ -n "$JUDGE_TP" ]; then export JUDGE_TP; fi

export HERE REPO="$(cd "$HERE/.." && pwd)"
export HB_TASKS="$TASKS"
IFS=',' read -ra _T <<< "$TASKS"
LAST=$((${#_T[@]} - 1))

SB_ARGS=(--export=ALL --array="0-${LAST}%${MAX}")
if [ -n "$PARTITION" ]; then SB_ARGS+=(--partition="$PARTITION"); fi
if [ -n "$ACCOUNT" ]; then SB_ARGS+=(--account="$ACCOUNT"); fi
if [ -n "$TIME" ]; then SB_ARGS+=(--time="$TIME"); fi

echo "array 0-${LAST}%${MAX}  (${#_T[@]} tasks, ${MAX} at a time, 2 nodes each)"
i=0
for t in "${_T[@]}"; do echo "  [$i] $t"; i=$((i + 1)); done

SUBMIT_OUT="$(sbatch "${SB_ARGS[@]}" "$HERE/humanibench_eval.sbatch")"
echo "$SUBMIT_OUT"
ARRAY_ID="$(printf '%s' "$SUBMIT_OUT" | grep -oE '[0-9]+$' || true)"

# Collect once the array is done. afterany: runs even if some element fails, so
# a partial suite still gets a report (missing tasks are listed as missing).
if [ "$COLLECT" = "1" ] && [ -n "$ARRAY_ID" ]; then
  CB_ARGS=(--export=ALL --dependency="afterany:${ARRAY_ID}" --kill-on-invalid-dep=yes)
  COLLECT_PART="${COLLECT_PARTITION:-$PARTITION}"
  if [ -n "$COLLECT_PART" ]; then CB_ARGS+=(--partition="$COLLECT_PART"); fi
  if [ -n "$ACCOUNT" ]; then CB_ARGS+=(--account="$ACCOUNT"); fi
  if [ -n "$COLLECT_TIME" ]; then CB_ARGS+=(--time="$COLLECT_TIME"); fi
  if [ -n "${MODEL_ARGS:-}" ]; then
    COLLECT_MODEL="$(printf '%s' "$MODEL_ARGS" | sed -nE 's/.*(pretrained|model)=([^,]+).*/\2/p' | sed 's#.*/##')"
    export COLLECT_MODEL
  fi
  echo "collect: queued behind ${ARRAY_ID} (afterany)"
  sbatch "${CB_ARGS[@]}" "$HERE/collect.sbatch"
else
  if [ "$COLLECT" != "1" ]; then
    echo "collect: skipped (--no-collect); run 'python3 collect_results.py' yourself"
  else
    echo "collect: SKIPPED - could not read the array job id from: $SUBMIT_OUT" >&2
  fi
fi
