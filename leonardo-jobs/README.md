# VLM benchmarks with a local LLM-as-judge (lmms-eval on Leonardo)

This runs the lmms-eval VLM benchmark suite. Some tasks need an LLM judge:
`vibe_eval`, `vibe_eval_greek`, `mmmu_val`, `mmmu_pro_standard`, `hallusion_bench_image`.
Grading uses a **local judge model** served with vLLM. No OpenAI or Reka API is used.


## Layout

```
leonardo-jobs/
├── README.md
├── slurm/
│   ├── all_benchmarks_eval_judge_enhanced.sbatch   # the SLURM job
│   └── run.sh                                      # CLI launcher
├── env/
│   ├── build_gemma4_env.sh                         # builds the judge conda env
│   └── requirements/                               # frozen package lists (reference only)
├── summarize_results.py                            # writes SUMMARY.txt per run
├── logs/                                           # per-job logs
└── results/                                        # per-job results, one dir per <model>_<jobid>
```


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

1. NODE 0 starts the judge as a background `srun` step. It serves an OpenAI-compatible
   API on a random high port.
2. The script waits until `/v1/models` reports the served model name. It does not just
   check `/health` — nodes are shared, so a neighbor's vLLM could answer instead.
3. NODE 1 runs `lmms_eval`. Judge-using tasks read the endpoint from env vars and POST to it.
4. On exit, a trap tears the judge down.


## Environments

There are **two separate conda envs**, one per node. Paths are set by `JUDGE_CONDA_ENV`
and `EVAL_CONDA_ENV`. They run on different nodes, so they're independent. The judge
needs a much newer vLLM than the eval stack.

### Judge env — `…/envs/gemma4_env`

```bash
bash env/build_gemma4_env.sh <dest>/gemma4_env
```

This must be a **cu12** build. Leonardo's driver is CUDA 12.2, so cu13 wheels fail at
GPU init. The script installs cu12.9 torch plus the cu12 vLLM nightly (from
`wheels.vllm.ai/nightly/cu129`) and verifies the stack. Then point `JUDGE_CONDA_ENV`
at `<dest>/gemma4_env`.

### Eval env — `…/envs/vlm_eval_vllm` → pip-buildable

```bash
conda create -p <dest>/vlm_eval_vllm python=3.12 -y
conda activate <dest>/vlm_eval_vllm
pip install vllm==0.7.3                               # lmms-eval's tested vLLM
cd /leonardo_work/EUHPC_D26_056/nxiros/VLM-evaluation/lmms-eval && pip install -e .
```

> Frozen package lists are in `env/requirements/` (reference only — see their headers).
> Install runtime deps **into the env**, not `~/.local`. The job runs with
> `PYTHONNOUSERSITE=1`, so anything in user-site is invisible at runtime.


## How to use

Submit with the `slurm/run.sh`.

```bash
cd /leonardo_work/EUHPC_D26_056/nxiros/VLM-evaluation/lmms-eval/leonardo-jobs/slurm

./run.sh                                       # default run (9 tasks, LIMIT=500/task)
./run.sh --tasks ai2d,mmmu_val --limit 50      # specific tasks, 50 samples each
./run.sh --model llava-hf/llava-1.5-7b-hf       # different evaluated VLM
./run.sh --tasks vibe_eval --limit 0            # one task, full set (no cap)
./run.sh --time 02:00:00                        # override walltime

# different judge model:
./run.sh --judge-model /path/or/hf-id --judge-served-name my-judge --judge-tp 4 \
         --judge-env /path/to/env               # --judge-env only if it needs a different env
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/`<br>`Qwen2.5-VL-3B-Instruct` | Evaluated VLM, as an HF id or local path. Expands into `MODEL_ARGS` unless `--model-args` is given. |
| `--model-args` | built from `--model` | Full `MODEL_ARGS` override. Sets `tensor_parallel_size`, `gpu_memory_utilization`, `max_model_len`, `limit_mm_per_prompt`, etc. directly. |
| `--tasks` | `hallusion_bench_image`<br>`mmmu_val`<br>`mmmu_pro_standard`<br>`mmbench_en_dev_lite`<br>`vibe_eval`<br>`vibe_eval_greek`<br>`vmcbench`<br>`greek_geometry_3k`<br>`geometry3k_en` | Comma-separated task list to run. |
| `--limit` | `500` | Per-task sample cap. `0` or empty means a full run with no cap. |
| `--time` | `06:00:00` | SLURM walltime override, e.g. `02:00:00`. |
| `--judge-model` | `/leonardo_work/`<br>`EUHPC_D33_216/mzoumpou/`<br>`model/gemma-4-31B-it` | Judge model, as an HF id or local path. |
| `--judge-served-name` | `gemma4-31b-judge` | Name the judge is served under. Clients must ask for this exact name. |
| `--judge-tp` | `2` | Tensor-parallel GPU count for the judge. |
| `--judge-env` | `/leonardo_work/`<br>`EUHPC_D33_215/`<br>`solvability_faithfulness/`<br>`envs/gemma4_env` | Conda env used to serve the judge. Only needed if the judge model needs a different env. |
| `--eval-env` | `/leonardo_work/`<br>`EUHPC_D26_056/nxiros/`<br>`envs/vlm_eval_vllm` | Conda env used to run the evaluated model. Point this at a newer env when the model needs a vLLM version `vlm_eval_vllm` doesn't have. |
| `--no-judge` / `--judge` | auto | Force the judge off/on. By default it's inferred from `--tasks`: the judge starts only if a judge-using task is present. No judge means a **1-node** job. |
| `--account` | `EUHPC_D26_056` | SLURM account override. |
| `--model-backend` | `vllm` | lmms-eval `--model` backend. Use e.g. `krikri_vlm` for a plain HF-transformers model that vLLM can't serve. |


### The judge is started only when a task needs it

`vibe_eval`, `vibe_eval_greek`, `mmmu_val`, `mmmu_pro_standard`, `hallusion_bench_image`,
`mmbench_en_dev_lite` and `geometry3k_en` grade through the judge endpoint. For anything
else the judge is pure overhead — a 31B model takes ~20 min to load — so it's skipped and
the job drops to a single node:

```bash
./run.sh --tasks ai2d --limit 4                  # 1 node, no judge (auto)
./run.sh --tasks vibe_eval_greek --limit 50      # 2 nodes, judge started (auto)
./run.sh --tasks ai2d --judge                    # force the judge on anyway
./run.sh --tasks vibe_eval --no-judge            # force it off (grading will fail)
```

Keep the list in `JUDGE_TASKS` in sync between `run.sh` and the sbatch — `run.sh` needs it
to pick the node count at submit time, since `--nodes` is a `#SBATCH` directive.


### Swapping the judge

Its **conda env** must have a vLLM that supports the judge model's architecture, and it
must be a **cu12** build. Leonardo's 12.2 driver rejects cu13 (see Environments above).

Image-aware grading (`vibe_eval`, `hallusion_bench_image`) needs a **VLM** judge.

### Swapping the evaluated model

`vlm_eval_vllm` (vLLM 0.7.3) is older version.

`Qwen/Qwen3-VL-4B-Instruct` run by pointing `--eval-env` at `gemma4_env` instead — that env has a
newer vLLM:

```bash
./run.sh --model Qwen/Qwen3-VL-4B-Instruct --eval-env <path>/gemma4_env
```


## Results

The results dir is named `<model>_<jobid>`.

`SUMMARY.txt` looks like this:

```
task                       metric                          value
----------------------------------------------------------------
ai2d                       exact_match                    0.8340
mmmu_val                   mmmu_acc                       0.4360
vibe_eval                  all                           42.4721
```

## Before you run: datasets must be pre-downloaded

Compute nodes are **offline**. Every task's dataset, and the model, must already be in
the HF cache (`$HF_HOME = /leonardo_work/EUHPC_D26_056/nxiros/hf_cache`). Download on a
**login node** first:

```bash
hf download <dataset-repo> --repo-type dataset   # into $HF_HOME
hf download <model-repo>                          # the evaluated model, if not cached
```

The full list of task names for `--tasks`, and the dataset each needs, is in:
`docs/advanced/current_tasks.md`.


## Leonardo notes

- **GPU driver is CUDA 12.2.** All torch/vLLM must be **cu12** builds. cu13 fails at GPU init.
- Login nodes have internet, so build envs and download models/datasets there. Compute
  nodes are offline (`HF_HUB_OFFLINE=1`), and read from the `HF_HOME` cache.
- Compute nodes are shared. That's why the setup uses random ports and served-model-name checks.
- Keep large caches (conda/pip/compile) off `$HOME` — its quota is small. Use `/leonardo_work`.


## Key settings (in the sbatch, all commented inline)

- **Eval `tensor_parallel_size=1`.** Single-GPU avoids multi-GPU NCCL hangs on these
  nodes. Raise it only if the model doesn't fit on one GPU.
- **`max_model_len` / `limit_mm_per_prompt`.** Size these to the biggest prompt+image.
  Multi-image tasks need `limit_mm_per_prompt={"image":N}`, or vLLM hard-raises.
- **`gpu_memory_utilization`.** Lower it (~0.7) when raising the settings above, or
  batches OOM.
- **`EVAL_BATCH_SIZE=256`.** Raised from 64. At 64, GPU KV-cache usage sat around ~5%
  — heavily underutilized on a 3-4B model on a 64GB A100. Raise further if KV usage
  stays low.
- **`--output_path` / `--log_samples`.** Required by HallusionBench. Also persists
  per-task results, so a crash doesn't lose finished work.


## Adaptation of lmms-eval repo

These are the changes made on top of upstream lmms-eval to support this local-judge,
Leonardo-hosted setup.

### Greek benchmark
- Added **`tasks/vibe_eval_greek/`**, a new task: VibeEval with Greek prompts/references.
  The Greek dataset has no image column, so images are looked up offline from the local
  `RekaAI/VibeEval` cache by `example_id`. It grades via the local judge endpoint.

### vLLM judge integration
- **`tasks/vibe_eval/utils.py`** — rewired grading off Reka's hosted API onto the local
  judge endpoint. Made it **image-aware**: the image is sent inline as base64, not a URL.
- **`tasks/vibe_eval/vibe_eval.yaml`** — evaluator switched to `reka-core` (image-aware mode).
- **`tasks/hallusion_bench/utils.py`** — reads `MODEL_VERSION` for the judge model. It
  was hardcoded to `gpt-4` before, which silently failed against a local judge. Also
  fixed the API URL path (now appends `/chat/completions`).
- **`models/chat/vllm.py`, `vllm_generate.py`, `simple/vllm.py`** — compatibility guards
  for the local vLLM server API, e.g. `get_format_metrics` when the client lacks
  `get_metrics()`.
- Both vibe_eval tasks got an empty-category guard, so a `--limit` run with zero
  samples in a category doesn't crash aggregation.

### Leonardo SLURM + envs
- **`slurm/`** — `all_benchmarks_eval_judge_enhanced.sbatch` (two-node judge+eval job)
  and `run.sh` (CLI launcher).
- **`env/`** — `build_gemma4_env.sh` (builds the judge's conda env from scratch,
  working around Leonardo's CUDA 12.2 driver by pinning cu12.9 torch plus the matching
  cu129 vLLM nightly) and `requirements/*.txt` (frozen package lists, reference only).
- **`summarize_results.py`** — writes `SUMMARY.txt` per run.
