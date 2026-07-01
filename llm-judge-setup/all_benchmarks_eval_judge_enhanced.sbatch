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
#SBATCH --output=/leonardo_work/EUHPC_D26_056/nxiros/VLM-evaluation/lmms-eval/slurm/logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_D26_056/nxiros/VLM-evaluation/lmms-eval/slurm/logs/%x-%j.err
#
# lmms-eval VLM benchmarks with a local LLM-as-judge (two nodes):
#   NODE 0  vLLM judge server  (Gemma-4-31B-it, TP=2)   [conda: gemma4_env]
#   NODE 1  lmms_eval VLM run  (Qwen2.5-VL-3B,  TP=1)    [conda: vlm_eval_vllm]
# Node 1 reaches the judge by hostname over Leonardo's internal network.
#
# Submit:   sbatch all_benchmarks_eval_judge_enhanced.sbatch
# Override: sbatch --export=ALL,TASKS=mmmu_val,LIMIT=0 all_benchmarks_eval_judge_enhanced.sbatch
# Details & troubleshooting: see README.md / TESTING.md in llm-judge-bench-setup/.
set -euo pipefail

# ===================== CONFIG (all overridable via --export) =====================
REPO="/leonardo_work/EUHPC_D26_056/nxiros/VLM-evaluation/lmms-eval"
JUDGE_BASE="/leonardo_work/EUHPC_D26_056/nxiros/llm-judge"   # neutral CWD for the judge step
HF_CACHE="/leonardo_work/EUHPC_D26_056/nxiros/hf_cache"
CACHE_ROOT="/leonardo_work/EUHPC_D26_056/nxiros"            # vLLM/torch/triton caches live here

# --- evaluated model (NODE 1) ---
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-/leonardo_work/EUHPC_D26_056/nxiros/envs/vlm_eval_vllm}"
MODEL_NAME="${MODEL_NAME:-vllm}"
# tensor_parallel_size=1: 3B fits one A100-64GB and avoids multi-GPU NCCL hangs on these nodes.
# limit_mm_per_prompt: multi-image tasks (mmmu/seedbench/...) need >1 image or vLLM hard-raises.
# gpu_memory_utilization=0.70: leaves headroom given the large max_model_len + image limits.
MODEL_ARGS="${MODEL_ARGS:-model=Qwen/Qwen2.5-VL-3B-Instruct,tensor_parallel_size=1,gpu_memory_utilization=0.70,max_model_len=32768,trust_remote_code=True,limit_mm_per_prompt={\"image\":8,\"video\":1}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
LIMIT="${LIMIT:-500}"   # per-task sample cap; 0/empty = full run

# --- judge model (NODE 0) ---
JUDGE_CONDA_ENV="${JUDGE_CONDA_ENV:-/leonardo_work/EUHPC_D33_215/solvability_faithfulness/envs/gemma4_env}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/leonardo_work/EUHPC_D33_216/mzoumpou/model/gemma-4-31B-it}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-gemma4-31b-judge}"
JUDGE_TP="${JUDGE_TP:-2}"                                   # ~62GB model on 2x A100-64GB
JUDGE_PORT="${JUDGE_PORT:-$(( 20000 + RANDOM % 20000 ))}"   # random: nodes are shared
JUDGE_API_KEY="${JUDGE_API_KEY:-token-abc123}"
JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-16384}"                     # room for an image + text (image-aware grading)
JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.85}"
JUDGE_MAX_WAIT="${JUDGE_MAX_WAIT:-3600}"                    # seconds to wait for model load
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

# vibe_eval / vibe_eval_greek / mmmu* / hallusion_bench_image use the judge; the rest don't.
TASKS="${TASKS:-ai2d,chartqa,docvqa_val,docvqa_test,hallusion_bench_image,mmbench_en_dev_lite,mme,mmmu_val,scienceqa_img,seedbench,textvqa_val,mmmu_pro_standard,vibe_eval,vmcbench,livexiv_tqa_v6,vibe_eval_greek}"
# =================================================================================

mkdir -p "${REPO}/slurm/logs"

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
JUDGE_NODE="${NODES[0]}"
EVAL_NODE="${NODES[1]}"
JUDGE_URL="http://${JUDGE_NODE}:${JUDGE_PORT}/v1"
SERVER_LOG="${REPO}/slurm/logs/judge_server_${SLURM_JOB_ID}.log"
RESULTS_DIR="${RESULTS_DIR:-${REPO}/slurm/results/${SLURM_JOB_ID}}"   # required by hallusion_bench
mkdir -p "${RESULTS_DIR}"

if [ -n "${LIMIT:-}" ] && [ "${LIMIT}" != "0" ]; then
  LIMIT_FLAG="--limit ${LIMIT}"; echo "[config] capped run: --limit ${LIMIT} per task"
else
  LIMIT_FLAG="";                 echo "[config] full run (no --limit)"
fi

echo "=========================================="
echo "Judge node : $JUDGE_NODE  (${JUDGE_SERVED_NAME}, TP=${JUDGE_TP})"
echo "Eval node  : $EVAL_NODE   (${MODEL_NAME})"
echo "Judge URL  : $JUDGE_URL"
echo "Server log : $SERVER_LOG"
echo "Results    : $RESULTS_DIR"
echo "=========================================="

# ---- 1. judge server on NODE 0 (background srun step) ----
# gemma4_env bundles its own CUDA via pip wheels -> do NOT module-load cuda here.
# VLLM_USE_FLASHINFER_SAMPLER=0: this build's FlashInfer sampler JIT is broken; use native sampler.
# Caches -> /leonardo_work to avoid filling the small $HOME quota.
srun --nodes=1 --ntasks=1 --nodelist="$JUDGE_NODE" --gres=gpu:4 \
  bash -c "
    set -e
    cd ${JUDGE_BASE}                                  # neutral CWD (avoids numpy import shadowing)
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate '${JUDGE_CONDA_ENV}'
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export CUDA_HOME=\"\$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13\"
    export PATH=\"\$CUDA_HOME/bin:\$PATH\"
    export HF_HOME=${HF_CACHE}
    export FLASHINFER_WORKSPACE_BASE=${CACHE_ROOT}/.flashinfer_cache
    export VLLM_CACHE_ROOT=${CACHE_ROOT}/.vllm_cache
    export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/.torchinductor_cache
    export TRITON_CACHE_DIR=${CACHE_ROOT}/.triton_cache
    mkdir -p \"\$FLASHINFER_WORKSPACE_BASE\" \"\$VLLM_CACHE_ROOT\" \"\$TORCHINDUCTOR_CACHE_DIR\" \"\$TRITON_CACHE_DIR\"
    export NCCL_IB_HCA=mlx5
    export NCCL_P2P_DISABLE=1                          # multi-GPU NCCL deadlocks otherwise
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export VLLM_USE_V1=1
    export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
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


# ----
# Cleanup
cleanup() {
  echo "[cleanup] stopping judge server (pid $SERVER_STEP_PID)"
  kill "$SERVER_STEP_PID" 2>/dev/null || true
  scancel --signal=TERM "${SLURM_JOB_ID}" 2>/dev/null || true
}
trap cleanup EXIT
# ----


# ---- 2. wait for OUR judge: verify served-model-name in /v1/models (probe from NODE 1) ----
echo "[wait] polling ${JUDGE_URL}/models for '${JUDGE_SERVED_NAME}' (up to $((JUDGE_MAX_WAIT/60)) min)..."
START=$(date +%s); READY=0
while [ $(($(date +%s) - START)) -lt "$JUDGE_MAX_WAIT" ]; do
  if ! kill -0 "$SERVER_STEP_PID" 2>/dev/null; then
    echo "[wait] ERROR: judge exited early. Last log:"; tail -50 "$SERVER_LOG"; exit 1
  fi
  if srun --nodes=1 --ntasks=1 --nodelist="$EVAL_NODE" --overlap --gres=none \
        curl -sf "http://${JUDGE_NODE}:${JUDGE_PORT}/v1/models" \
        -H "Authorization: Bearer ${JUDGE_API_KEY}" 2>/dev/null \
        | grep -q "\"${JUDGE_SERVED_NAME}\""; then
    READY=1; echo "[wait] judge READY after $(($(date +%s) - START))s"; break
  fi
  sleep 10
done
[ "$READY" -eq 1 ] || { echo "[wait] ERROR: judge not ready in time. Last log:"; tail -50 "$SERVER_LOG"; exit 1; }

# ---- 3. lmms_eval on NODE 1, judge endpoint via env vars ----
# Run from $HOME (not the repo) to avoid a numpy circular-import on startup.
srun --nodes=1 --ntasks=1 --nodelist="$EVAL_NODE" --gres=gpu:4 --overlap \
  bash -c "
    set -euo pipefail
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate '${EVAL_CONDA_ENV}'

    export NCCL_IB_HCA=mlx5
    export NCCL_P2P_DISABLE=1
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export HF_HOME=${HF_CACHE}
    export HF_HUB_OFFLINE=1
    export LMMS_EVAL_DATASETS_CACHE=\"\$HF_HOME/datasets\"
    export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
    export VLLM_CACHE_ROOT=${CACHE_ROOT}/.vllm_cache
    export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/.torchinductor_cache
    export TRITON_CACHE_DIR=${CACHE_ROOT}/.triton_cache
    mkdir -p \"\$VLLM_CACHE_ROOT\" \"\$TORCHINDUCTOR_CACHE_DIR\" \"\$TRITON_CACHE_DIR\"

    # judge endpoint for the judge-using tasks (vibe_eval_greek uses its own vars; the rest
    # read the OPENAI_*/MODEL_VERSION set below)
    export VIBE_EVAL_GREEK_JUDGE_BASE_URL='${JUDGE_URL}'
    export VIBE_EVAL_GREEK_JUDGE_API_KEY='${JUDGE_API_KEY}'
    export VIBE_EVAL_GREEK_JUDGE_MODEL='${JUDGE_SERVED_NAME}'
    export API_TYPE='openai'
    export OPENAI_API_KEY='${JUDGE_API_KEY}'
    export OPENAI_API_URL='${JUDGE_URL}'
    export MODEL_VERSION='${JUDGE_SERVED_NAME}'

    cd \"\$HOME\"
    python -B -m lmms_eval \
      --model '${MODEL_NAME}' \
      --model_args '${MODEL_ARGS}' \
      --tasks '${TASKS}' \
      --batch_size ${EVAL_BATCH_SIZE} ${LIMIT_FLAG} \
      --output_path '${RESULTS_DIR}' \
      --log_samples
  "
EVAL_RC=$?

echo "[done] eval exit code: $EVAL_RC"
exit $EVAL_RC