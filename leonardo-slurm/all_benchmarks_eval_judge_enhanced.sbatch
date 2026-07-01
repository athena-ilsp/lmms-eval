#!/bin/bash
#SBATCH --job-name=lmms-eval-judge
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D26_056
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=06:00:00
#SBATCH --output=/leonardo_work/EUHPC_D26_056/nxiros/VLM-evaluation/lmms-eval/leonardo-slurm/logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_D26_056/nxiros/VLM-evaluation/lmms-eval/leonardo-slurm/logs/%x-%j.err
#
# Two-node job for LLM-as-judge VLM benchmarks (vibe_eval, vibe_eval_greek, mmmu, hallusion):
#   NODE 0 -> vLLM judge server (Gemma-4-31B-it, TP=2)   [conda gemma4_env]
#   NODE 1 -> lmms_eval VLM benchmarks (Qwen2.5-VL-3B, TP=1) [conda vlm_eval_vllm]
# The eval node reaches the judge over Leonardo's internal network by hostname.
#
# Usage: prefer ./run.sh (see llm-judge-setup/README.md), or submit directly:
#   sbatch all_benchmarks_eval_judge_enhanced.sbatch
#   sbatch --export=ALL,TASKS=mmmu_val,LIMIT=0 all_benchmarks_eval_judge_enhanced.sbatch
set -euo pipefail

# ===================== CONFIG =====================
REPO="/leonardo_work/EUHPC_D26_056/nxiros/VLM-evaluation/lmms-eval"
JUDGE_BASE="/leonardo_work/EUHPC_D26_056/nxiros/llm-judge"
CACHE_ROOT="/leonardo_work/EUHPC_D26_056/nxiros"
HF_HOME_DIR="${CACHE_ROOT}/hf_cache"

# Judge server env: needs a vLLM build that knows the Gemma-4 (gemma4) architecture.
# See llm-judge-setup/README.md for why this must be a cu12 build on Leonardo.
JUDGE_CONDA_ENV="${JUDGE_CONDA_ENV:-/leonardo_work/EUHPC_D33_215/solvability_faithfulness/envs/gemma4_env}"

# --- model under test (VLM) ---
MODEL_NAME="${MODEL_NAME:-vllm}"
# tensor_parallel_size=1: avoids a multi-GPU NCCL init hang seen on these nodes.
# limit_mm_per_prompt: several tasks send multi-image prompts (vLLM defaults to 1 image).
# gpu_memory_utilization=0.70: leaves headroom for the larger max_model_len / image limit.
MODEL_ARGS="${MODEL_ARGS:-model=Qwen/Qwen2.5-VL-3B-Instruct,tensor_parallel_size=1,gpu_memory_utilization=0.70,max_model_len=32768,trust_remote_code=True,limit_mm_per_prompt={\"image\":8,\"video\":1}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
LIMIT="${LIMIT:-500}"    # per-task sample cap; 0 or empty = full run
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-/leonardo_work/EUHPC_D26_056/nxiros/envs/vlm_eval_vllm}"

# --- judge model (served via vLLM) ---
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/leonardo_work/EUHPC_D33_216/mzoumpou/model/gemma-4-31B-it}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-gemma4-31b-judge}"
JUDGE_TP="${JUDGE_TP:-2}"
JUDGE_PORT="${JUDGE_PORT:-$(( 20000 + RANDOM % 20000 ))}"   # random: compute nodes are shared
JUDGE_API_KEY="${JUDGE_API_KEY:-token-abc123}"
JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-16384}"   # room for image + text tokens (image-aware grading)
JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.85}"
JUDGE_MAX_WAIT="${JUDGE_MAX_WAIT:-3600}"  # seconds to wait for the judge to come up
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

# vibe_eval / vibe_eval_greek / mmmu_val / mmmu_pro_standard / hallusion_bench_image use the judge.
TASKS="${TASKS:-ai2d,chartqa,docvqa_val,docvqa_test,hallusion_bench_image,mmbench_en_dev_lite,mme,mmmu_val,scienceqa_img,seedbench,textvqa_val,mmmu_pro_standard,vibe_eval,vmcbench,livexiv_tqa_v6,vibe_eval_greek}"
# ==================================================

mkdir -p "${REPO}/leonardo-slurm/logs"

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
JUDGE_NODE="${NODES[0]}"
EVAL_NODE="${NODES[1]}"
JUDGE_URL="http://${JUDGE_NODE}:${JUDGE_PORT}/v1"
SERVER_LOG="${REPO}/leonardo-slurm/logs/judge_server_${SLURM_JOB_ID}.log"
RESULTS_DIR="${RESULTS_DIR:-${REPO}/leonardo-slurm/results/${SLURM_JOB_ID}}"
mkdir -p "${RESULTS_DIR}"

if [ -n "${LIMIT:-}" ] && [ "${LIMIT}" != "0" ]; then
  LIMIT_FLAG="--limit ${LIMIT}"
  echo "[config] capped run: --limit ${LIMIT} per task"
else
  LIMIT_FLAG=""
  echo "[config] full run (no --limit)"
fi

echo "=========================================="
echo "Judge node : $JUDGE_NODE  (vLLM ${JUDGE_SERVED_NAME}, TP=${JUDGE_TP})"
echo "Eval node  : $EVAL_NODE   (lmms_eval ${MODEL_NAME})"
echo "Judge URL  : $JUDGE_URL"
echo "Server log : $SERVER_LOG"
echo "=========================================="

# ---- 1. start the judge server on NODE 0 (background srun step) ----
srun --nodes=1 --ntasks=1 --nodelist="$JUDGE_NODE" --gres=gpu:4 \
  bash -c "
    set -e
    cd ${JUDGE_BASE}   # neutral CWD; avoids a numpy import-shadowing crash from the repo tree
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate '${JUDGE_CONDA_ENV}'
    # Native PyTorch sampler: FlashInfer's JIT sampler fails to compile in this env.
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export CUDA_HOME=\"\$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13\"
    export PATH=\"\$CUDA_HOME/bin:\$PATH\"
    export FLASHINFER_WORKSPACE_BASE=${CACHE_ROOT}/.flashinfer_cache
    mkdir -p \"\$FLASHINFER_WORKSPACE_BASE\"
    # Keep vLLM/torch/triton caches off \$HOME (small quota).
    export VLLM_CACHE_ROOT=${CACHE_ROOT}/.vllm_cache
    export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/.torchinductor_cache
    export TRITON_CACHE_DIR=${CACHE_ROOT}/.triton_cache
    mkdir -p \"\$VLLM_CACHE_ROOT\" \"\$TORCHINDUCTOR_CACHE_DIR\" \"\$TRITON_CACHE_DIR\"
    export HF_HOME=${HF_HOME_DIR}
    export NCCL_IB_HCA=mlx5
    export NCCL_P2P_DISABLE=1
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export VLLM_USE_V1=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONNOUSERSITE=1
    echo \"[judge] launching on \$(hostname)\"
    exec vllm serve '${JUDGE_MODEL_PATH}' \
      --served-model-name '${JUDGE_SERVED_NAME}' \
      --tensor-parallel-size ${JUDGE_TP} \
      --max-model-len ${JUDGE_MAX_LEN} \
      --gpu-memory-utilization ${JUDGE_GPU_UTIL} \
      --max-num-seqs ${MAX_NUM_SEQS} \
      --enforce-eager \
      --disable-custom-all-reduce \
      --api-key '${JUDGE_API_KEY}' \
      --host 0.0.0.0 --port ${JUDGE_PORT} \
      --trust-remote-code
  " > "$SERVER_LOG" 2>&1 &
SERVER_STEP_PID=$!

cleanup() {
  echo "[cleanup] stopping judge server step (pid $SERVER_STEP_PID)"
  kill "$SERVER_STEP_PID" 2>/dev/null || true
  scancel --signal=TERM "${SLURM_JOB_ID}" 2>/dev/null || true
}
trap cleanup EXIT

# ---- 2. wait until the judge is up (verify served-model-name in /v1/models) ----
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

# ---- 3. run lmms_eval on NODE 1, pointing the judge env vars at NODE 0 ----
srun --nodes=1 --ntasks=1 --nodelist="$EVAL_NODE" --gres=gpu:4 --overlap \
  bash -c "
    set -euo pipefail
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate '${EVAL_CONDA_ENV}'

    export NCCL_IB_HCA=mlx5
    export NCCL_P2P_DISABLE=1
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    export HF_HOME=${HF_HOME_DIR}
    export HF_HUB_OFFLINE=1
    export LMMS_EVAL_DATASETS_CACHE=\"\$HF_HOME/datasets\"
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONNOUSERSITE=1
    export VLLM_CACHE_ROOT=${CACHE_ROOT}/.vllm_cache
    export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/.torchinductor_cache
    export TRITON_CACHE_DIR=${CACHE_ROOT}/.triton_cache
    mkdir -p \"\$VLLM_CACHE_ROOT\" \"\$TORCHINDUCTOR_CACHE_DIR\" \"\$TRITON_CACHE_DIR\"

    # Judge endpoint for vibe_eval_greek.
    export VIBE_EVAL_GREEK_JUDGE_BASE_URL='${JUDGE_URL}'
    export VIBE_EVAL_GREEK_JUDGE_API_KEY='${JUDGE_API_KEY}'
    export VIBE_EVAL_GREEK_JUDGE_MODEL='${JUDGE_SERVED_NAME}'

    # Judge endpoint for MMMU / MMMU-Pro / HallusionBench / vibe_eval (OpenAI-compatible client).
    export API_TYPE='openai'
    export OPENAI_API_KEY='${JUDGE_API_KEY}'
    export OPENAI_API_URL='${JUDGE_URL}'
    export MODEL_VERSION='${JUDGE_SERVED_NAME}'

    cd \"\$HOME\"   # avoid a numpy import-shadowing crash from the repo tree

    python -B -m lmms_eval \
      --model '${MODEL_NAME}' \
      --model_args '${MODEL_ARGS}' \
      --tasks '${TASKS}' \
      --batch_size ${EVAL_BATCH_SIZE} ${LIMIT_FLAG} \
      --output_path '${RESULTS_DIR}' \
      --log_samples
  "
EVAL_RC=$?

# ---- 4. distil lmms_eval's results.json into a human-readable SUMMARY.txt ----
python3 - "$RESULTS_DIR" "$SLURM_JOB_ID" "$MODEL_ARGS" "$TASKS" "$LIMIT" <<'PYEOF' || true
import json, sys, glob, os
results_dir, jobid, model_args, tasks, limit = sys.argv[1:6]
files = sorted(glob.glob(os.path.join(results_dir, "**", "*_results.json"), recursive=True),
               key=os.path.getmtime)
if not files:
    print("[summary] no *_results.json found; skipping SUMMARY.txt"); sys.exit(0)
data = json.load(open(files[-1]))
res  = data.get("results", {})
model = data.get("model_name") or model_args.split(",")[0]

lines = []
lines.append(f"# lmms-eval summary  (job {jobid})")
lines.append(f"model   : {model}")
lines.append(f"limit   : {limit} per task")
lines.append(f"date    : {data.get('date','?')}")
lines.append(f"results : {files[-1]}")
lines.append("")
lines.append(f"{'task':<26} {'metric':<26} {'value':>10}")
lines.append("-" * 64)
for task in sorted(res):
    row = res[task]
    for key, val in row.items():
        if key == "alias" or "stderr" in key: continue
        if not isinstance(val, (int, float)): continue
        metric = key.split(",")[0]
        lines.append(f"{task:<26} {metric:<26} {val:>10.4f}")
out = os.path.join(results_dir, "SUMMARY.txt")
open(out, "w").write("\n".join(lines) + "\n")
print("[summary] wrote", out)
print("\n".join(lines))
PYEOF

echo "[done] eval exit code: $EVAL_RC"
echo "[done] results dir  : $RESULTS_DIR   (see SUMMARY.txt)"
exit $EVAL_RC