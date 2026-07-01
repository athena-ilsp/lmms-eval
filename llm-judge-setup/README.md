# VLM benchmarks with a local LLM-as-judge (lmms-eval on Leonardo)

Run the lmms-eval VLM benchmark suite where some tasks need an LLM judge
(`vibe_eval`, `vibe_eval_greek`, `mmmu_val`, `mmmu_pro_standard`,
`hallusion_bench_image`), grading against a **local judge model** served with vLLM —
no OpenAI / Reka API.


## Architecture

```
            sbatch (2 nodes)
 ┌─────────────────────────┐        ┌──────────────────────────────┐
 │ NODE 0: JUDGE server     │  HTTP  │ NODE 1: lmms_eval VLM run     │
 │ vLLM OpenAI API          │◄───────│ judge tasks call the endpoint │
 │ judge model (e.g. Gemma) │ :PORT  │ model under test (e.g. Qwen)  │
 │ env: gemma4_env          │        │ env: vlm_eval_vllm            │
 └─────────────────────────┘        └──────────────────────────────┘
   eval node reaches the judge by hostname on Leonardo's internal network
```

1. NODE 0 starts the judge as a background `srun` step and serves an OpenAI-compatible API
   on a random high port.
2. The script waits until `/v1/models` reports the served model name (not just `/health` —
   nodes are shared, a neighbor's vLLM could answer otherwise).
3. NODE 1 runs `lmms_eval`; judge-using tasks read the endpoint from env vars and POST to it.
4. On exit, a trap tears the judge down.

## Environments

**Two separate conda envs**, one per node (paths set by `JUDGE_CONDA_ENV` / `EVAL_CONDA_ENV`).
They're independent (different nodes); the judge needs a much newer vLLM than the eval stack.

### Judge env — `…/envs/gemma4_env`

```bash
bash build_gemma4_env.sh <dest>/gemma4_env
```

Must be a **cu12** build: Leonardo's driver is CUDA 12.2, so cu13 wheels fail at GPU init.
The script installs cu12.9 torch + the cu12 vLLM nightly (from `wheels.vllm.ai/nightly/cu129`)
and verifies the stack. Then point `JUDGE_CONDA_ENV` at `<dest>/gemma4_env`.

### Eval env — `…/envs/vlm_eval_vllm` → pip-buildable

```bash
conda create -p <dest>/vlm_eval_vllm python=3.12 -y
conda activate <dest>/vlm_eval_vllm
pip install vllm==0.7.3                               # lmms-eval's tested vLLM
cd /leonardo_work/EUHPC_D26_056/nxiros/VLM-evaluation/lmms-eval && pip install -e .
```

> Frozen package lists are in `requirements/` (reference only — see their headers).
> Install runtime deps **into the env**, not `~/.local`: the job runs `PYTHONNOUSERSITE=1`,
> so anything in user-site is invisible at runtime.


## How to use

Submit with the `run.sh` launcher (in `../leonardo-slurm/`) — set only what you want to change,
everything else falls back to the sbatch defaults:

```bash
cd /leonardo_work/EUHPC_D26_056/nxiros/VLM-evaluation/lmms-eval/leonardo-slurm

./run.sh                                       # full default run (16 tasks, LIMIT=500)
./run.sh --tasks ai2d,mmmu_val --limit 50      # specific tasks, 50 samples each
./run.sh --model llava-hf/llava-1.5-7b-hf       # different evaluated VLM
./run.sh --tasks vibe_eval --limit 0            # one task, full set (no cap)
./run.sh --time 02:00:00                        # override walltime

# different judge model:
./run.sh --judge-model /path/or/hf-id --judge-served-name my-judge --judge-tp 4 \
         --judge-env /path/to/env               # --judge-env only if it needs a different env
```

Flags: `--model`, `--model-args`, `--tasks`, `--limit`, `--time`, and for the judge
`--judge-model`, `--judge-served-name`, `--judge-tp`, `--judge-env`, `--eval-env`
(`--help` lists them). Or submit the sbatch directly with `--export`.

Swapping the judge: its **conda env** must have a vLLM that supports the judge model's arch
and is a **cu12** build (Leonardo's 12.2 driver rejects cu13 — see Environments). Image-aware
grading (`vibe_eval`, `hallusion_bench_image`) needs a **VLM** judge; a text-only judge still
grades the text tasks (`vibe_eval_greek`, MMMU) but scores image tasks on text alone.

### Results

Watch a running job live:

```bash
JID=<jobid>
tail -f leonardo-slurm/logs/lmms-eval-judge-${JID}.err     # per-task progress
tail -f leonardo-slurm/logs/lmms-eval-judge-${JID}.out     # INFO + final results table
tail -f leonardo-slurm/logs/judge_server_${JID}.log        # judge loading / requests
```

When the job finishes, scores are written to a human-readable summary:

```
leonardo-slurm/results/<jobid>/SUMMARY.txt          # task | metric | value table
leonardo-slurm/results/<jobid>/.../*_results.json   # full machine-readable results
leonardo-slurm/logs/lmms-eval-judge-<jobid>.out     # run log (also prints the table at the end)
```

`SUMMARY.txt` looks like:

```
task                       metric                          value
----------------------------------------------------------------
ai2d                       exact_match                    0.8340
mmmu_val                   mmmu_acc                       0.4360
vibe_eval                  all                           42.4721
```



### Before you run: datasets must be pre-downloaded

Compute nodes are **offline**, so every task's dataset (and the model) must already be in the
HF cache (`$HF_HOME = /leonardo_work/EUHPC_D26_056/nxiros/hf_cache`). Download on a **login
node** first:

```bash
hf download <dataset-repo> --repo-type dataset   # into $HF_HOME
hf download <model-repo>                          # the evaluated model, if not cached
```

The full list of task names you can pass to `--tasks` (and the dataset each needs) is in:
`docs/advanced/current_tasks.md`.

## Leonardo notes
- **GPU driver is CUDA 12.2** → all torch/vLLM must be **cu12** builds (cu13 fails at GPU init).
- Login nodes have internet (build envs, download models/datasets); compute nodes are offline
  (`HF_HUB_OFFLINE=1`, read from `HF_HOME` cache).
- Compute nodes are shared → random ports + served-model-name checks.
- Keep large caches (conda/pip/compile) off `$HOME` (small quota) → `/leonardo_work`.



## Key settings (in the sbatch, all commented inline)

- **Eval `tensor_parallel_size=1`** — single-GPU avoids multi-GPU NCCL hangs on these nodes;
  raise only if the model doesn't fit on one GPU.
- **`max_model_len` / `limit_mm_per_prompt`** — size to the biggest prompt+image; multi-image
  tasks need `limit_mm_per_prompt={"image":N}` or vLLM hard-raises.
- **`gpu_memory_utilization`** — lower (~0.7) when raising the above, or batches OOM.
- **`--output_path` / `--log_samples`** — required by HallusionBench; also persists per-task
  results so a crash doesn't lose finished work.





## Adaptation of lmms-eval repo

Changes made on top of upstream lmms-eval to support this local-judge, Leonardo-hosted setup.

### Greek benchmark
- Added **`tasks/vibe_eval_greek/`** — a new task: VibeEval with Greek prompts/references.
  Since the Greek dataset has no image column, images are looked up offline from the local
  `RekaAI/VibeEval` cache by `example_id`. Grades via the local judge endpoint.

### vLLM judge integration
- **`tasks/vibe_eval/utils.py`** — rewired grading off Reka's hosted API onto the local judge
  endpoint, and made it **image-aware** (the image is sent inline as base64, not a URL).
- **`tasks/vibe_eval/vibe_eval.yaml`** — evaluator switched to `reka-core` (image-aware mode).
- **`tasks/hallusion_bench/utils.py`** — reads `MODEL_VERSION` for the judge model (was
  hardcoded `gpt-4`, which silently failed against a local judge); fixed the API URL path
  (appends `/chat/completions`).
- **`models/chat/vllm.py`, `vllm_generate.py`, `simple/vllm.py`** — compatibility guards for
  the local vLLM server API (e.g. `get_format_metrics` when the client lacks `get_metrics()`).
- Both vibe_eval tasks got an empty-category guard so a `--limit` run with zero samples in a
  category doesn't crash aggregation.

### Leonardo SLURM + envs
- **`leonardo-slurm/`** — `all_benchmarks_eval_judge_enhanced.sbatch` (two-node judge+eval job) and
  `run.sh` (CLI launcher: `--tasks`, `--limit`, `--model`, `--judge-model`, ... plus a
  `SUMMARY.txt` written per run).
- **`llm-judge-setup/`** — this README, `TESTING.md`, `requirements/*.txt`, and
  `build_gemma4_env.sh` (builds the judge's conda env from scratch, working around
  Leonardo's CUDA 12.2 driver by pinning cu12.9 torch + the matching cu129 vLLM nightly).





## Example Results — Qwen2.5-VL-3B, LIMIT=500 (validation pass)

> `LIMIT=500` takes the **first** 500 samples (not random). MME (a sum) and order-sensitive ratios are
> not directly comparable. For comparable scores rerun selected tasks with `LIMIT=0`.

| Task | Metric | Value |
|------|--------|------:|
| ai2d | exact_match | 0.834 |
| chartqa | relaxed_overall | 0.706 |
| docvqa_val | anls | 0.932 |
| hallusion_bench_image | aAcc / fAcc / qAcc | 59.0 / 31.8 / 31.9 |
| livexiv_tqa_v6 | livexiv_tqa | 0.596 |
| mmbench_en_dev_lite | gpt_eval_score | 84.09 |
| mme | cognition / perception | 162.5 / 283.5 (sum, capped) |
| mmmu_pro_standard | mmmu_acc | 0.320 |
| mmmu_val | mmmu_acc | 0.436 |
| scienceqa_img | exact_match | 0.818 |
| seedbench | seed_all | 0.754 |
| textvqa_val | exact_match | 0.819 |
| vibe_eval (image-aware judge) | all / hard / normal | 42.5 / 24.3 / 53.3 |
| vibe_eval_greek (text judge) | all / hard / normal | 15.4 / 11.0 / 18.0 |
| vmcbench | average | 0.628 |
