# HumaniBench eval (2 nodes)

Node 0 runs a vLLM judge. Node 1 runs `lmms_eval` and sends judge calls to it.
Dataset: `vector-institute/HumaniBench`. Task code: `lmms_eval/tasks/humanibench/`.

## Tasks

`--tasks` takes these names (comma-separated). Default is all 10.

| Task | What it is | Scores |
| --- | --- | --- |
| `humanibench_t1_plain` | Scene understanding, plain question | accuracy, bias↓, hallucination↓, faithfulness, relevance, coherence |
| `humanibench_t1_cot` | Same images, chain-of-thought version | same |
| `humanibench_t2` | Instance identity (who / what in the image) | same |
| `humanibench_t3` | Multiple-choice VQA | `accuracy_statistical` (letter match) + judge scores |
| `humanibench_t4_closed` | Multilingual MCQ | same as t3, plus a language split |
| `humanibench_t4_open` | Multilingual open-ended | judge scores, plus a language split |
| `humanibench_t5` | Visual grounding (bounding box) | mean_iou, map50, map75, missing↓ — no judge |
| `humanibench_t6_factual` | Factual image caption | accuracy, bias↓, hallucination↓, faithfulness |
| `humanibench_t6_empathic` | Empathic caption | empathy, joy, anxiety↓, sadness↓ |
| `humanibench_t7` | Image resilience (clean vs attacked) | same as t1 + `retention` (attacked / clean) |

↓ = lower is better. Everything else is higher is better.
`--limit` is rejected for `humanibench_t7` and the group `humanibench` (retention needs clean + attacked rows).

## 1. Set up once

```bash
export JUDGE_CONDA_ENV=/path/to/env    # prefix with bin/activate
export EVAL_CONDA_ENV=/path/to/env     # same path twice is fine
export HF_HOME=/path/to/hf_cache       # optional; default $HOME/.cache/huggingface
```

If the model under test is **not** Qwen2.5-VL, `export MODEL_NAME=<backend>`
first (`llava`, `internvl2`, `internvl3`, ... — keys in
`lmms_eval/models/__init__.py`). Default is `qwen2_5_vl`.

Optional: `HF_TOKEN` if the dataset 401s. `CUDA_MODULE=none` to skip
`module load cuda` (Leonardo).

## 2. Flags

All of these are optional. `run` = `run.sh`, `submit` = `submit_per_task.sh`.

| Flag | What it does | Default | Where |
| --- | --- | --- | --- |
| `--model` | weights, shorthand for `--model-args pretrained=...` | Qwen2.5-VL-3B | both |
| `--model-args` | full lmms_eval model args string | `pretrained=Qwen/Qwen2.5-VL-3B-Instruct` | both |
| `--tasks` | comma-separated task list | all 10 | both |
| `--judge-model` | judge weights (path or hf id) | `$JUDGE_MODEL_PATH` | both |
| `--judge-served-name` | name the judge answers to | `gemma4-31b-judge` | both |
| `--judge-tp` | judge tensor-parallel GPUs | `4` | both |
| `--time` | walltime override | 3 days | both |
| `--partition` | cluster partition | unset | both |
| `--account` | cluster account | unset | both |
| `--max N` | how many tasks run **at the same time** | `3` | submit |
| `--no-collect` | skip the auto merge job at the end | collect runs | submit |
| `--collect-partition` | partition for the merge job | same as `--partition` | submit |
| `--collect-time` | walltime for the merge job | 20 min | submit |
| `--limit N` | only N samples per task (smoke test) | `0` = full run | run |
| `--cuda-module` | module to load for vLLM | `cuda` | run |
| `--judge-env` | override `$JUDGE_CONDA_ENV` | env var | run |
| `--eval-env` | override `$EVAL_CONDA_ENV` | env var | run |

`--partition` / `--account`: cluster-specific, skip if your site has defaults.
`--max`: concurrency cap (default 3), not a sequential switch. You can omit it.

## 3. Full suite

```bash
cd humanibench-slurm

./submit_per_task.sh --max 3 \
  --model-args 'pretrained=<vlm>,device_map=cuda:0' \
  --judge-model <judge> \
  --judge-served-name <alias> \
  --judge-tp 4
```

`<vlm>` / `<judge>` = HuggingFace id or local path. `<alias>` = any short name
the eval job will call the judge by.

`--max 3` = at most 3 tasks running at once (that is also the default).
`device_map=cuda:0` pins the inference model to one GPU; use `device_map=auto`
if it needs more. GPU count per node is `#SBATCH --gres` in
`humanibench_eval.sbatch` — change it for your cluster.

Filled in:

```bash
./submit_per_task.sh --max 3 \
  --model-args 'pretrained=Qwen/Qwen2.5-VL-7B-Instruct,device_map=cuda:0' \
  --judge-model google/gemma-4-31b-it \
  --judge-served-name gemma4-31b-judge \
  --judge-tp 4
```

When the last task ends, a small CPU job merges everything into one report.
`--no-collect` skips that.

## 4. Smoke test / one job

`--limit` cannot include t7. Drop t7 (and do not pass the `humanibench` group).

```bash
cd humanibench-slurm

./run.sh --time 02:00:00 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --judge-model Qwen/Qwen2.5-VL-3B-Instruct \
  --judge-served-name qwen2.5-vl-3b-judge \
  --judge-tp 1 \
  --limit 8 \
  --tasks humanibench_t1_plain,humanibench_t1_cot,humanibench_t2,humanibench_t3,humanibench_t4_closed,humanibench_t4_open,humanibench_t5,humanibench_t6_factual,humanibench_t6_empathic
```

Single task, full size: `./run.sh --tasks humanibench_t3`
All 10 back to back (no `--limit`): `./run.sh`

The judge takes ~9 min to load before the eval starts. That is normal.

## 5. Where results go

```
results/<judge>/<model>/
    SUMMARY.txt                all tasks (written by collect)
    SOURCES.txt                which run each number came from
    humanibench_splits.txt     breakdown by attribute / language / attack
    <task>/                    one folder per task
        RUN.json               task, model, judge, job id, status
        SUMMARY.txt            this task only
        <model__name>/*_results.json, *_samples_*.jsonl
```

Re-running a task moves the old folder to `_superseded/` instead of overwriting it.

Logs are flat, by job id: `logs/humanibench-eval-<jobid>.{out,err}` and
`logs/judge_server_<jobid>.log`.

If every judge metric is `0.0`, the eval never reached the judge - check that
`OPENAI_API_URL` was set in the job and read the judge server log.

`SUMMARY.txt` after a full collect looks like this (numbers from one Qwen2.5-VL-7B
run, yours will differ):

```
# lmms-eval summary  (collected from per-task jobs)
model   : Qwen/Qwen2.5-VL-7B-Instruct
judge   : gemma-4-31b-it
missing : (none)

task                       metric                          value
----------------------------------------------------------------
humanibench_t1_plain       accuracy                      60.6160
humanibench_t1_plain       bias                           1.9878
humanibench_t1_plain       hallucination                 18.0281
...
humanibench_t5             mean_iou                      89.6475
humanibench_t5             map50                         96.4912
humanibench_t6_empathic    empathy                       45.1324
humanibench_t6_empathic    anxiety                       23.1667
humanibench_t7             accuracy                      68.1000
humanibench_t7             retention                     89.5442
```

`humanibench_splits.txt` further splits those by Age / Ethnicity / Gender /
Occupation / Sport, by language on t4, and by attack type on t7.

## 6. Merge reports by hand

Safe to re-run any time, never moves the originals:

```bash
python3 collect_results.py --model <model-tag>
python3 collect_results.py --dry-run
```

`--limit` runs are excluded unless you pass `--include-limited`. A task with
fewer rows than the full test set gets flagged `[SHORT]` in `SOURCES.txt`.
