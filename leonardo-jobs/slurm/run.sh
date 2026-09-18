#!/bin/bash
# Easy launcher for all_benchmarks_eval_judge_enhanced.sbatch.
# Set only what you want to change; everything else uses the sbatch defaults.
#
# Examples
#   ./run.sh                                        # full default run (16 tasks, LIMIT=500)
#   ./run.sh --tasks ai2d,mmmu_val --limit 50       # a couple of tasks, 50 samples each
#   ./run.sh --model llava-hf/llava-1.5-7b-hf        # different evaluated VLM
#   ./run.sh --tasks vibe_eval --limit 0             # one task, full set
#   ./run.sh --judge-model /path/to/model --judge-served-name my-judge --judge-tp 4
#   ./run.sh --judge-env /path/to/env                # different judge conda env
#
# After it finishes, scores are in:
#   <repo>/slurm/results/<jobid>/SUMMARY.txt         # human-readable
#   plus the job logs: <repo>/slurm/logs/lmms-eval-judge-<jobid>.{out,err}
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SBATCH="$HERE/all_benchmarks_eval_judge_enhanced.sbatch"

# ---- defaults come from the sbatch; only pass overrides the user set ----
declare -A OV                 # name -> sbatch var
MODEL=""; TASKS=""; LIMIT=""; MODEL_ARGS=""
JUDGE_MODEL=""; JUDGE_ENV=""; JUDGE_NAME=""; JUDGE_TP=""; EVAL_ENV=""; TIME=""
USE_JUDGE=""; ACCOUNT=""; MODEL_BACKEND=""

while [ $# -gt 0 ]; do
  case "$1" in
    --model)             MODEL="$2"; shift 2 ;;       # evaluated VLM hf id/path
    --model-args)        MODEL_ARGS="$2"; shift 2 ;;  # full MODEL_ARGS override (advanced)
    --model-backend)     MODEL_BACKEND="$2"; shift 2 ;; # lmms-eval --model (default vllm)
    --tasks)             TASKS="$2"; shift 2 ;;
    --limit)             LIMIT="$2"; shift 2 ;;
    --judge-model)       JUDGE_MODEL="$2"; shift 2 ;; # judge model hf id/path
    --judge-served-name) JUDGE_NAME="$2"; shift 2 ;;  # name clients ask for (default gemma4-31b-judge)
    --judge-tp)          JUDGE_TP="$2"; shift 2 ;;    # judge tensor-parallel GPUs (default 2)
    --judge-env)         JUDGE_ENV="$2"; shift 2 ;;   # judge conda env
    --no-judge)          USE_JUDGE="0"; shift ;;      # force judge off (1-node job)
    --judge)             USE_JUDGE="1"; shift ;;      # force judge on (2-node job)
    --account)           ACCOUNT="$2"; shift 2 ;;     # SLURM account override
    --eval-env)          EVAL_ENV="$2"; shift 2 ;;
    --time)              TIME="$2"; shift 2 ;;        # walltime, e.g. 02:00:00
    -h|--help)           sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

# Decide the judge here too: the node count is a #SBATCH directive, so it has to be
# settled at submit time. Keep this list in sync with JUDGE_TASKS in the sbatch.
JUDGE_TASKS="vibe_eval,vibe_eval_greek,mmmu_val,mmmu_pro_standard,hallusion_bench_image,mmbench_en_dev_lite,geometry3k_en"
if [ -z "$USE_JUDGE" ]; then
  if [ -n "$TASKS" ]; then
    USE_JUDGE=0
    for t in ${TASKS//,/ }; do
      case ",${JUDGE_TASKS}," in *",${t},"*) USE_JUDGE=1; break ;; esac
    done
  else
    USE_JUDGE=1   # sbatch default TASKS include judge tasks
  fi
fi

# --model is a convenience: expand it into MODEL_ARGS unless a full --model-args was given
if [ -n "$MODEL" ] && [ -z "$MODEL_ARGS" ]; then
  MODEL_ARGS="model=$MODEL,tensor_parallel_size=1,gpu_memory_utilization=0.70,max_model_len=32768,trust_remote_code=True,limit_mm_per_prompt={\"image\":8,\"video\":1}"
fi

# Export overrides into the ENVIRONMENT, then use `sbatch --export=ALL`. This avoids
# SLURM's inline `--export=VAR=a,b` comma-splitting, which would mangle MODEL_ARGS (it
# contains commas). Only non-empty values are exported; the rest fall back to sbatch defaults.
if [ -n "$MODEL_ARGS" ]; then export MODEL_ARGS; fi
if [ -n "$TASKS" ];      then export TASKS; fi
if [ -n "$LIMIT" ];      then export LIMIT; fi
if [ -n "$MODEL_BACKEND" ]; then export MODEL_NAME="$MODEL_BACKEND"; fi
export USE_JUDGE
if [ -n "$JUDGE_MODEL" ];then export JUDGE_MODEL_PATH="$JUDGE_MODEL"; fi
if [ -n "$JUDGE_NAME" ]; then export JUDGE_SERVED_NAME="$JUDGE_NAME"; fi
if [ -n "$JUDGE_TP" ];   then export JUDGE_TP; fi
if [ -n "$JUDGE_ENV" ];  then export JUDGE_CONDA_ENV="$JUDGE_ENV"; fi
if [ -n "$EVAL_ENV" ];   then export EVAL_CONDA_ENV="$EVAL_ENV"; fi

SB_ARGS=(--export=ALL)
if [ -n "$TIME" ]; then SB_ARGS+=(--time="$TIME"); fi
if [ -n "$ACCOUNT" ]; then SB_ARGS+=(--account="$ACCOUNT"); fi

# No judge -> no judge node. The sbatch asks for 2; override it down to 1 so we
# don't hold an idle GPU node (and queue faster).
if [ "$USE_JUDGE" = "1" ]; then
  SB_ARGS+=(--nodes=2)
else
  SB_ARGS+=(--nodes=1)
fi

echo "submitting with overrides:"
echo "  USE_JUDGE=$USE_JUDGE ($([ "$USE_JUDGE" = 1 ] && echo '2 nodes: judge + eval' || echo '1 node: eval only'))"
for v in MODEL_NAME MODEL_ARGS TASKS LIMIT JUDGE_MODEL_PATH JUDGE_SERVED_NAME JUDGE_TP JUDGE_CONDA_ENV EVAL_CONDA_ENV; do
  if [ -n "${!v:-}" ]; then echo "  $v=${!v}"; fi
done
sbatch "${SB_ARGS[@]}" "$SBATCH"