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
# Two-node job for LLM-as-judge VLM benchmarks (vibe_eval, vibe_eval_greek, mmmu, hallusion):
#   NODE 0  -> vLLM judge server (Gemma-4-31B-it, TP=2) on a random port    [conda gemma4_env]
#   NODE 1  -> lmms_eval VLM benchmarks (Qwen2.5-VL-3B, TP=1)          [conda vlm_eval_vllm]
# The eval reaches the judge over Leonardo's internal network by hostname.
#
# Submit:  sbatch all_benchmarks_eval_judge_enhanced.sbatch
set -euo pipefail

# ===================== CONFIG =====================
REPO="/leonardo_work/EUHPC_D26_056/nxiros/VLM-evaluation/lmms-eval"
JUDGE_BASE="/leonardo_work/EUHPC_D26_056/nxiros/llm-judge"
# Conda env for the JUDGE server. The default llm-judge venv (vLLM 0.11 / transformers
# 4.57) does NOT know the gemma4 arch; this conda env has vLLM 0.23.x-dev + transformers
# 5.12 which serves Gemma4ForConditionalGeneration. It bundles its own CUDA (pip nvidia
# wheels) so the judge step must NOT load a system cuda module.
JUDGE_CONDA_ENV="${JUDGE_CONDA_ENV:-/leonardo_work/EUHPC_D33_215/solvability_faithfulness/envs/gemma4_env}"

# --- model under test (VLM) ---
# vLLM backend: continuous batching + paged attention across the eval node's 4 GPUs.
# MUCH faster than the HF 'qwen2_5_vl' backend (which runs ~1 req at a time on 1 GPU).
# NOTE: the vLLM wrapper uses key 'model=' (not 'pretrained=').
MODEL_NAME="${MODEL_NAME:-vllm}"
# tensor_parallel_size=1: the 3B VLM fits easily on one A100-64GB, and this is the config
# validated end-to-end by the smoke test. TP>1 deadlocked at an NCCL barrier on these
# shared/old-kernel nodes (job 47817311 hung with rank0 spinning, workers idle).
# max_model_len=32768 (Qwen2.5-VL-3B's native max): 16384 was too small -- some samples'
# text+image tokens exceed it and vLLM hard-raises (ai2d had a 16465-token prompt). Plenty
# of KV headroom at TP=1 on a 64GB A100 with a 3B model.
# limit_mm_per_prompt: some tasks (mmmu_val, seedbench, mmbench...) send multi-image
# prompts; vLLM defaults to 1 image/prompt and hard-raises ("At most 1 image(s)...").
# image=8 covers MMMU's max (~7 images). The wrapper json.loads the {..} value and
# passes it through to vLLM's LLM(); the model_args parser is brace-aware so the inner
# commas are safe.
# gpu_memory_utilization=0.70: raising max_model_len->32768 + limit_mm_per_prompt image=8
# enlarged the KV-cache worst-case reservation; at 0.85 a multi-image batch OOM'd (432MiB
# free). 0.70 leaves headroom for activations; a 3B model still has ample KV room.
MODEL_ARGS="${MODEL_ARGS:-model=Qwen/Qwen2.5-VL-3B-Instruct,tensor_parallel_size=1,gpu_memory_utilization=0.70,max_model_len=32768,trust_remote_code=True,limit_mm_per_prompt={\"image\":8,\"video\":1}}"
# requests fed to the vLLM scheduler per step (vLLM batches internally; bigger = better util)
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
# LIMIT>0 caps samples PER TASK (fast smoke test); empty/0 = full run.
# 500/task: full no-limit run can't fit in walltime (ai2d alone is 57k samples ~3h);
# this caps each of the 16 tasks at 500 for full COVERAGE within 6h. Drop to 0 for a
# true full-test run on selected tasks later.
LIMIT="${LIMIT:-500}"
# Conda env for the EVAL step. The vLLM backend needs the cloned env that has vllm
# installed (vlm_eval itself has no vllm). Set to 'vlm_eval' to use the HF backend.
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-/leonardo_work/EUHPC_D26_056/nxiros/envs/vlm_eval_vllm}"

# --- judge model (text LLM, served via vLLM) ---
# Reusing an already-downloaded Gemma-4-31B-it instruct model under another (readable)
# account. Gemma-4-31B bf16 ~62 GB, comfortable across 2xA100-64GB with TP=2.
# NOTE: this is a multimodal Gemma4ForConditionalGeneration checkpoint (model_type
# 'gemma4', requires a recent transformers/vLLM that knows the gemma4 arch). Because it's
# multimodal, the judge can ALSO grade image-aware (vibe_eval sends the image inline).
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/leonardo_work/EUHPC_D33_216/mzoumpou/model/gemma-4-31B-it}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-gemma4-31b-judge}"
# TP=2 is the validated config (62GB model fits comfortably on 2x A100-64GB, ~22GB KV free).
# Paired with --disable-custom-all-reduce in the serve command below.
JUDGE_TP="${JUDGE_TP:-2}"
# Compute nodes are SHARED — port 8000 is often taken by another user's vLLM, and a
# bare /health check could pass against THEIR server (eval would grade against the wrong
# model). Use a random high port and verify our served-model-name below.
JUDGE_PORT="${JUDGE_PORT:-$(( 20000 + RANDOM % 20000 ))}"
JUDGE_API_KEY="${JUDGE_API_KEY:-token-abc123}"
# 16384: vibe_eval now grades image-aware, so the judge receives an image (many tokens) plus
# the prompt/reference/generation text. 8192 was fine for text-only judging but too tight
# once an image is included.
JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-16384}"
JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.85}"
JUDGE_MAX_WAIT="${JUDGE_MAX_WAIT:-3600}"   # seconds to wait for model load
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

# Tasks: full list PLUS vibe_eval_greek (needs the judge)
# 'vibe_eval' (English) now grades via the local image-aware Gemma judge instead of Reka's
# hosted API (its evaluate() was rewritten). 'vibe_eval_greek' is the Greek variant (text
# judge). Both use the same local judge server.
TASKS="${TASKS:-ai2d,chartqa,docvqa_val,docvqa_test,hallusion_bench_image,mmbench_en_dev_lite,mme,mmmu_val,scienceqa_img,seedbench,textvqa_val,mmmu_pro_standard,vibe_eval,vmcbench,livexiv_tqa_v6,vibe_eval_greek}"
# ==================================================

mkdir -p "${REPO}/slurm/logs"

# ---- figure out which node is which ----
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
JUDGE_NODE="${NODES[0]}"
EVAL_NODE="${NODES[1]}"
JUDGE_URL="http://${JUDGE_NODE}:${JUDGE_PORT}/v1"
SERVER_LOG="${REPO}/slurm/logs/judge_server_${SLURM_JOB_ID}.log"
# Per-job results dir. Required by hallusion_bench (its aggregation writes gpt_response
# files to --output_path; with no path it crashes: os.path.join(None,...)). Also makes
# lmms_eval persist per-task results+samples to disk, so a late crash no longer loses
# everything finished so far.
RESULTS_DIR="${RESULTS_DIR:-${REPO}/slurm/results/${SLURM_JOB_ID}}"
mkdir -p "${RESULTS_DIR}"

# Optional per-task sample cap (smoke test). LIMIT empty or 0 -> full run.
if [ -n "${LIMIT:-}" ] && [ "${LIMIT}" != "0" ]; then
  LIMIT_FLAG="--limit ${LIMIT}"
  echo "[config] SMOKE RUN: --limit ${LIMIT} per task"
else
  LIMIT_FLAG=""
  echo "[config] FULL RUN (no --limit)"
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
    # cd to a neutral dir so nothing in the submit CWD (the lmms-eval repo tree) can
    # shadow stdlib/numpy and trigger the intermittent numpy circular-import crash.
    cd ${JUDGE_BASE}
    # gemma4 judge env (conda). Brings its own CUDA via pip nvidia wheels, so do NOT
    # load a system cuda module here.
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate '${JUDGE_CONDA_ENV}'
    # vLLM's FlashInfer sampler JIT-compiles at startup; that build is broken in this env
    # (cu13 nvcc vs 12.9 runtime headers -> CCCL incompat; binutils 2.30 too old for CUDA-13
    # asm). The judge does short/greedy sampling, so use vLLM's native PyTorch sampler -> no
    # JIT, no toolchain dependency. CUDA_HOME still points at the bundled cu13 toolkit, and
    # the FlashInfer JIT cache is moved off the tight \$HOME quota onto /leonardo_work.
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export CUDA_HOME=\"\$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13\"
    export PATH=\"\$CUDA_HOME/bin:\$PATH\"
    export FLASHINFER_WORKSPACE_BASE=/leonardo_work/EUHPC_D26_056/nxiros/.flashinfer_cache
    mkdir -p \"\$FLASHINFER_WORKSPACE_BASE\"
    # \$HOME has a tight 50G quota (was overflowing -> 'Disk quota exceeded' during warmup).
    # Keep vLLM/torch/triton compile caches on /leonardo_work instead of ~/.cache.
    export VLLM_CACHE_ROOT=/leonardo_work/EUHPC_D26_056/nxiros/.vllm_cache
    export TORCHINDUCTOR_CACHE_DIR=/leonardo_work/EUHPC_D26_056/nxiros/.torchinductor_cache
    export TRITON_CACHE_DIR=/leonardo_work/EUHPC_D26_056/nxiros/.triton_cache
    mkdir -p \"\$VLLM_CACHE_ROOT\" \"\$TORCHINDUCTOR_CACHE_DIR\" \"\$TRITON_CACHE_DIR\"
    export HF_HOME=/leonardo_work/EUHPC_D26_056/nxiros/hf_cache
    export NCCL_IB_HCA=mlx5
    export NCCL_P2P_DISABLE=1
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export VLLM_USE_V1=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONNOUSERSITE=1
    echo \"[judge] launching on \$(hostname) from \$(pwd)\"
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

# Make sure we always tear the server down, however we exit
cleanup() {
  echo "[cleanup] stopping judge server step (pid $SERVER_STEP_PID)"
  kill "$SERVER_STEP_PID" 2>/dev/null || true
  # also signal the srun step itself
  scancel --signal=TERM "${SLURM_JOB_ID}" 2>/dev/null || true
}
trap cleanup EXIT

# ---- 2. wait until OUR judge is up: verify served-model-name in /v1/models (probe from NODE 1) ----
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

    # The VLM now runs via vLLM with TP=4 on this node. On these shared, old-kernel
    # nodes the default NCCL P2P transport deadlocks at the init barrier (job hangs
    # right after engine init, before any generation). Disable P2P + pin the IB HCA,
    # exactly as the judge step does.
    export NCCL_IB_HCA=mlx5
    export NCCL_P2P_DISABLE=1
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    # reduce CUDA allocator fragmentation (multi-image batches vary in size); vLLM
    # explicitly recommends this when KV/activation memory is tight.
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    export HF_HOME=/leonardo_work/EUHPC_D26_056/nxiros/hf_cache
    export HF_HUB_OFFLINE=1
    export LMMS_EVAL_DATASETS_CACHE=\"\$HF_HOME/datasets\"
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONNOUSERSITE=1   # don't let a user-site numpy leak in (circular-import crash)
    # keep vLLM/torch compile caches off the tight \$HOME quota (this node runs vLLM too)
    export VLLM_CACHE_ROOT=/leonardo_work/EUHPC_D26_056/nxiros/.vllm_cache
    export TORCHINDUCTOR_CACHE_DIR=/leonardo_work/EUHPC_D26_056/nxiros/.torchinductor_cache
    export TRITON_CACHE_DIR=/leonardo_work/EUHPC_D26_056/nxiros/.triton_cache
    mkdir -p \"\$VLLM_CACHE_ROOT\" \"\$TORCHINDUCTOR_CACHE_DIR\" \"\$TRITON_CACHE_DIR\"

    # ---- judge endpoint for vibe_eval_greek ----
    export VIBE_EVAL_GREEK_JUDGE_BASE_URL='${JUDGE_URL}'
    export VIBE_EVAL_GREEK_JUDGE_API_KEY='${JUDGE_API_KEY}'
    export VIBE_EVAL_GREEK_JUDGE_MODEL='${JUDGE_SERVED_NAME}'

    # ---- judge endpoint for MMMU / MMMU-Pro / HallusionBench / vibe_eval ----
    # These tasks build an OpenAI-compatible judge client (some at import time); point them
    # at the SAME local Gemma judge server instead of api.openai.com. HallusionBench reads
    # MODEL_VERSION for the model name and OPENAI_API_URL for the endpoint (see its utils.py).
    export API_TYPE='openai'
    export OPENAI_API_KEY='${JUDGE_API_KEY}'
    export OPENAI_API_URL='${JUDGE_URL}'
    export MODEL_VERSION='${JUDGE_SERVED_NAME}'

    # Run from a NEUTRAL dir, not the repo root. 'python -m lmms_eval' from inside the repo
    # prepends the repo tree to sys.path and intermittently triggers a numpy circular-import
    # crash during interpreter startup. lmms_eval is pip-installed (editable) in the env, so
    # it resolves fine from anywhere.
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

# ---- 4. write a human-readable SUMMARY.txt from lmms_eval's results.json ----
# lmms_eval already saves a machine-readable *_results.json under RESULTS_DIR; distil the
# scores into RESULTS_DIR/SUMMARY.txt so you don't have to dig through logs.
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
        # keep the real metric values, skip aliases/stderr/clustered noise
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
