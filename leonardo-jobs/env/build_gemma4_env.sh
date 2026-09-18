#!/bin/bash
# Build the JUDGE conda env (Gemma-4-31B on vLLM) from scratch, on a LOGIN node.
#
# WHY this is not a plain `pip install vllm`:
#   Leonardo's GPU driver is CUDA 12.2, so torch/vLLM MUST be cu12 builds (cu13 -> GPU init
#   fails: "NVIDIA driver ... too old (12020)"). The default PyPI vllm is cu13. The cu12 (cu129)
#   builds live on vLLM's dedicated nightly index below; torch's cu129 build on PyTorch's index.
#
# Usage:  bash build_gemma4_env.sh [DEST_ENV_PATH]
set -euo pipefail

DEST="${1:-/leonardo_work/EUHPC_D26_056/nxiros/envs/gemma4_env}"
CACHE_ROOT="/leonardo_work/EUHPC_D26_056/nxiros"

# keep conda/pip caches off the small $HOME quota
export CONDA_PKGS_DIRS="$CACHE_ROOT/.conda_pkgs"
export PIP_CACHE_DIR="$CACHE_ROOT/.pip_cache"
export TMPDIR="$CACHE_ROOT/.tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"

source ~/miniconda3/etc/profile.d/conda.sh

echo "[1/3] create env: $DEST"
conda create -p "$DEST" python=3.11 -y --solver classic     # libmamba isn't configured here

echo "[2/3] cu12.9 torch (matches the 12.2 driver)"
"$DEST/bin/python" -m pip install --no-user \
  --index-url https://download.pytorch.org/whl/cu129 \
  torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129

echo "[3/3] cu12.9 vLLM (nightly cu129 index) + judge deps, WITHOUT clobbering torch"
# --extra-index-url adds vLLM's cu129 wheels; the torch constraint pins the cu129 build so
# vLLM's `torch==2.11.0` requirement resolves to our +cu129, not a fresh cu13 torch.
# CRITICAL: pin the EXACT cu129 wheel (…+g<hash>.cu129). Without the version pin, pip prefers
# the newer PyPI vllm (cu13 -> links libcudart.so.13 -> fails on the 12.2 driver). The cu129
# nightly index keeps only the latest build, so discover its version at run time.
VLLM_CU129=$(curl -s "https://wheels.vllm.ai/nightly/cu129/vllm/" \
  | grep -oE 'vllm-[0-9][^"<>]*\.cu129' | sort -u | tail -1 | sed 's/^vllm-//')
[ -n "$VLLM_CU129" ] || { echo "ERROR: no cu129 vllm wheel found on the index"; exit 1; }
echo "    pinning vllm==$VLLM_CU129"
printf 'torch==2.11.0+cu129\ntorchvision==0.26.0+cu129\ntorchaudio==2.11.0+cu129\n' > "$TMPDIR/torch_constraint.txt"
"$DEST/bin/python" -m pip install --no-user \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --constraint "$TMPDIR/torch_constraint.txt" \
  "vllm==$VLLM_CU129" transformers openai

# vLLM lazily imports these but doesn't always declare them; the JOB runs PYTHONNOUSERSITE=1,
# so a copy in ~/.local is INVISIBLE at runtime. Force them INTO the env (--ignore-installed
# bypasses any masking ~/.local copy). numpy pinned <2.4 (mistral_common/vLLM constraint).
"$DEST/bin/python" -m pip install --no-user --ignore-installed \
  regex pyyaml pandas pyarrow msgpack "numpy<2.4"

echo
echo "=== verify: cu12 stack + gemma4 arch (catches a cu13 regression at build time) ==="
# the vLLM native lib MUST link libcudart.so.12 — .so.13 means a cu13 wheel slipped in and
# it will die on the compute node's 12.2 driver.
SO=$(ls "$DEST"/lib/python3.11/site-packages/vllm/_C_stable_libtorch*.so 2>/dev/null | head -1)
if [ -n "$SO" ]; then
  CUDART=$(strings "$SO" | grep -oE 'libcudart\.so\.[0-9]+' | sort -u | tr '\n' ' ')
  echo "    vllm links: $CUDART"
  echo "$CUDART" | grep -q "libcudart.so.12" || { echo "FAIL: vllm links cu13, not cu12"; exit 1; }
fi
"$DEST/bin/python" - <<'EOF'
import torch, vllm
print("torch:", torch.__version__, "| CUDA:", torch.version.cuda)   # expect 2.11.0+cu129 / 12.9
print("vllm :", vllm.__version__)                                    # expect 0.23.x…cu129
from transformers import AutoConfig
c = AutoConfig.from_pretrained("/leonardo_work/EUHPC_D33_216/mzoumpou/model/gemma-4-31B-it",
                               trust_remote_code=True)
print("arch :", c.architectures)                                     # ['Gemma4ForConditionalGeneration']
assert "cu129" in torch.__version__, "torch is NOT cu129 -> will fail on the 12.2 driver"
print("OK: cu129 stack + gemma4 recognized")
EOF
echo "done. point JUDGE_CONDA_ENV at: $DEST"