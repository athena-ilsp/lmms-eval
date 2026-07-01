# Self-test guide — recreate the envs + run a small end-to-end check

Goal: prove, from scratch, that (a) you can rebuild both conda envs and (b) the full
judge+eval pipeline runs end-to-end. Work through the phases in order; each has a
**PASS check** so you know whether to continue.

Paths assume the account layout `/leonardo_work/EUHPC_D26_056/nxiros/...`. Adjust if yours
differs. Run **Phase 0–2 on a LOGIN node** (internet); Phase 3 submits to compute.

---

## Phase 0 — preflight (login node, 1 min)

```bash
WORK=/leonardo_work/EUHPC_D26_056/nxiros
REPO=$WORK/VLM-evaluation/lmms-eval

# conda present?
ls ~/miniconda3/etc/profile.d/conda.sh && echo "conda OK"

# judge model on disk (~59 GB)?
du -sh /leonardo_work/EUHPC_D33_216/mzoumpou/model/gemma-4-31B-it

# eval model + key datasets cached offline?
ls -d $WORK/hf_cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct && echo "eval model cached"
ls -d $WORK/hf_cache/datasets/*vibeeval_greek* $WORK/hf_cache/datasets/*RekaAI* && echo "judge datasets cached"

# neutral cd dir the judge step uses (must exist, else 'cd' fails):
mkdir -p $WORK/llm-judge
```
**PASS:** all five lines print without "No such file".

---

## Phase 1 — rebuild the EVAL env (login node, ~10–20 min)

Build it at a **test path** so you don't disturb the working one.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -p $WORK/envs/TEST_vlm_eval_vllm python=3.12 -y
conda activate $WORK/envs/TEST_vlm_eval_vllm

pip install vllm==0.7.3
cd $REPO
pip install -e .                 # editable lmms_eval
```
**PASS checks** (run from `$HOME`, not the repo, to mimic the job):
```bash
cd $HOME
python -c "import vllm, lmms_eval, numpy; print('eval env OK', vllm.__version__)"
python -c "from vllm import LLM, SamplingParams; print('vllm API OK')"
```
Expect `eval env OK 0.7.3` and `vllm API OK`. If `numpy` errors with a circular import,
you launched from inside the repo — `cd $HOME` and retry (this is the known gotcha the
sbatch handles with `cd $HOME`).

---

## Phase 2 — get the JUDGE env (login node)

**CLONE the known-good env — do NOT rebuild from pip.** The judge env is a bespoke
**CUDA-12.9** stack (`torch 2.11.0+cu129` + a vLLM `…cu129` wheel that supports Gemma-4).
It is NOT reproducible via `pip install vllm` today, because:
- Leonardo's GPU driver is **CUDA 12.2**, so torch/vLLM must be **cu12** builds — a cu13
  build fails at GPU init with `RuntimeError: The NVIDIA driver ... is too old (12020)`.
- The matching **cu12 vLLM wheel** (a nightly, `0.23.1rc1.dev193+…cu129`) has been **deleted
  from the index**, and the current stable `vllm==0.23.0` is **cu13-only**.

So the reproducible artifact is the clone:

```bash
# known-good source env (already validated)
SRC=/leonardo_work/EUHPC_D26_056/nxiros/envs/gemma4_env       # your own clone, or...
# SRC=/leonardo_work/EUHPC_D33_215/solvability_faithfulness/envs/gemma4_env  # original

# clone into your account (16 GB, a few min). classic solver: libmamba isn't configured here.
conda create --clone $SRC -p $WORK/envs/gemma4_env -y --solver classic
```

**PASS check** — cloned env has the cu12.9 stack and the model arch is recognized:
```bash
$WORK/envs/gemma4_env/bin/python - <<'EOF'
import torch; print("torch:", torch.__version__)          # expect 2.11.0+cu129  (cu12!)
import vllm;  print("vllm :", vllm.__version__)            # expect 0.23.1rc1.dev*…cu129
from transformers import AutoConfig
c = AutoConfig.from_pretrained("/leonardo_work/EUHPC_D33_216/mzoumpou/model/gemma-4-31B-it",
                               trust_remote_code=True)
print("arch :", c.architectures)                          # ['Gemma4ForConditionalGeneration']
EOF
```
If `torch` shows a `+cu13`/no-suffix build, you rebuilt instead of cloned — it will fail on
the compute node's 12.2 driver. Re-clone.

> A different judge model that a *standard* `pip install vllm` (cu13) supports could be built
> normally — but on this cluster the cu13 driver mismatch makes cloning the safe default.

> Reference exact validated versions: `requirements/judge_env_gemma4.txt` (judge) and
> `requirements/eval_env_vlm.txt` (eval). They're records, not clean `pip install -r`
> targets — see the headers in those files.

---

## Phase 3 — small end-to-end run (submit to compute, ~20–30 min)

Run the **real sbatch** but override it to a tiny, fast scope that still exercises BOTH the
non-judge path and the judge path (text + image-aware). Point it at your TEST envs.

```bash
cd $REPO/slurm

sbatch --export=ALL,\
EVAL_CONDA_ENV=$WORK/envs/TEST_vlm_eval_vllm,\
JUDGE_CONDA_ENV=$WORK/envs/gemma4_env,\
TASKS=ai2d,vibe_eval,vibe_eval_greek,\
LIMIT=4 \
  all_benchmarks_eval_judge_enhanced.sbatch
```
(`JUDGE_CONDA_ENV` = the cloned env from Phase 2; `EVAL_CONDA_ENV` = your rebuilt eval env,
or drop it to use the default.)

Why these three tasks:
- `ai2d` — pure VLM generation, no judge (proves the eval side works).
- `vibe_eval` — exercises the **image-aware** judge (sends an image to the judge).
- `vibe_eval_greek` — exercises the **text** judge.

`LIMIT=4` = 4 samples/task → minutes of compute once the models load.

### Watch it
```bash
JID=<jobid>           # from the sbatch output / squeue -u $USER
L=$REPO/slurm/logs
tail -f $L/lmms-eval-judge-${JID}.out     # readiness, INFO, final table
# also useful:
#   $L/lmms-eval-judge-${JID}.err          (per-task progress + tracebacks)
#   $L/judge_server_${JID}.log             (judge model load + requests)
```

### PASS checks (in order, in the .out)
1. `[wait] judge READY after <N>s` — judge server came up.
2. `Selected Tasks: ['ai2d', 'vibe_eval', 'vibe_eval_greek']`
3. Each task shows `Model Responding: 100%` then `Postprocessing: 100%`.
4. A results table prints with all three tasks and **non-N/A** values, e.g.:
   ```
   |ai2d           |exact_match|...|
   |vibe_eval      |all        |...|   <- judge graded (image-aware)
   |vibe_eval_greek|all        |...|   <- judge graded (text)
   ```
5. `[done] eval exit code: 0`

Confirm the judge actually served requests:
```bash
grep -c 'chat/completions HTTP/1.1" 200' $REPO/slurm/logs/judge_server_${JID}.log
# > 0 means the judge graded; vibe_eval also sends image_url payloads
```

**If all five pass + judge requests > 0, the pipeline is reproducible end-to-end.**

---

## Phase 4 — cleanup (optional)

```bash
conda env remove -p $WORK/envs/TEST_vlm_eval_vllm
conda env remove -p $WORK/envs/TEST_gemma4_env
rm -rf $REPO/slurm/results/$JID                 # the small test's results
rm -rf $WORK/.triton_cache_*                     # per-PID JIT scratch from the run
```

---

## Common failures (fast triage)

| Symptom | Cause / fix |
|---|---|
| `cd: .../llm-judge: No such file` in judge step | `mkdir -p $WORK/llm-judge` (Phase 0). |
| numpy circular-import crash | launched from the repo dir; the job uses `cd $HOME` — don't change that. |
| judge never READY, server log shows FlashInfer/JIT error | judge env: keep `VLLM_USE_FLASHINFER_SAMPLER=0` (set in the sbatch); don't `module load cuda`. |
| `unknown model type gemma4` | judge env's vLLM/transformers too old — upgrade (Phase 2). |
| `ModuleNotFoundError: No module named 'regex'` (or similar) at `vllm serve` | a vLLM runtime dep isn't in the env; `pip install regex` **into the env**. A copy in `~/.local` doesn't count (`PYTHONNOUSERSITE=1`). |
| eval hangs after engine init (no generation) | multi-GPU NCCL hang — keep eval `tensor_parallel_size=1`. |
| `At most 1 image(s)...` | a multi-image task without `limit_mm_per_prompt` — it's already in MODEL_ARGS; don't strip it. |
| CUDA OOM | lower `gpu_memory_utilization` (already 0.70) or `max_model_len`. |
| `Disk quota exceeded` during warmup | caches must point to `/leonardo_work` (the `*_CACHE_*` exports) — don't let them fall back to `$HOME`. |
| HallusionBench scores all 0 | judge model/url misrouted — needs `MODEL_VERSION` + `OPENAI_API_URL` (set in sbatch). |