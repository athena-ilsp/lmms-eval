#!/bin/bash
#SBATCH --job-name=humanibench-collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Merge the per-task runs into results/<judge>/<model>/SUMMARY.txt.
# submit_per_task.sh queues this with --dependency=afterany:<array job>, so it
# runs once, after the last array element leaves the queue (however it left).
# CPU only; no GPU, no judge.
#
# By hand:
#   sbatch --dependency=afterany:<arrayjobid> collect.sbatch
#   COLLECT_MODEL=Qwen2.5-VL-7B-Instruct sbatch collect.sbatch
set -euo pipefail

HERE="${HERE:-$(pwd)}"
cd "$HERE"

PY="${COLLECT_PYTHON:-python3}"
ARGS=()
if [ -n "${COLLECT_MODEL:-}" ]; then ARGS+=(--model "${COLLECT_MODEL}"); fi
if [ -n "${COLLECT_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  ARGS+=(${COLLECT_ARGS})
fi

echo "[collect] $(date -Is) on $(hostname)"
echo "[collect] ${PY} collect_results.py ${ARGS[*]:-}"
"$PY" collect_results.py "${ARGS[@]}"
