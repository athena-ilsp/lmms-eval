#!/usr/bin/env python3
"""Merge per-task HumaniBench result folders into one report per model.

Current layout, one folder per task, no job ids in the path:

    results/<judge>/<model>/<task>/           <- a single eval run
    results/<judge>/<model>/SUMMARY.txt       <- the report this script writes

Each run folder carries a RUN.json written by the sbatch (task, model, judge,
job id, status), so nothing here has to guess. Older flat <model>_<jobid>
folders have no manifest and are still read the old way: newest
*_results.json, then samples / SUMMARY / job log, to see which task ran.
Either way the complete full-size run per task wins.

Smoke / --limit jobs are not chosen unless you pass --include-limited:
  1. config.limit is set (e.g. 8)  -> limited
  2. n-samples.effective is well below the known full test size -> not full

Usage:
    python3 collect_results.py
    python3 collect_results.py --model Qwen2.5-VL-7B-Instruct
    python3 collect_results.py --dry-run

Writes {SUMMARY.txt,SOURCES.txt,sources.json,humanibench_splits.json,
humanibench_splits.txt} next to the task folders, under
results/<judge>/<model>/ (or results/collected/<model>/ when no run reports a
judge). Originals are never moved.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from group_samples import render_txt  # noqa: E402

SUITE = [
    "humanibench_t1_plain",
    "humanibench_t1_cot",
    "humanibench_t2",
    "humanibench_t3",
    "humanibench_t4_closed",
    "humanibench_t4_open",
    "humanibench_t5",
    "humanibench_t6_factual",
    "humanibench_t6_empathic",
    "humanibench_t7",
]
# Full test sizes (T1 split by version_type; T4 after drop_null_fields).
EXPECTED_N = {
    "humanibench_t1_plain": 6834,
    "humanibench_t1_cot": 6834,
    "humanibench_t2": 1343,
    "humanibench_t3": 1844,
    "humanibench_t4_closed": 6865,
    "humanibench_t4_open": 6850,
    "humanibench_t5": 285,
    "humanibench_t6_factual": 204,
    "humanibench_t6_empathic": 204,
    "humanibench_t7": 1500,
}
TASK_RE = re.compile(r"humanibench_[a-z0-9_]+")
SAMPLES_RE = re.compile(r"samples_(humanibench_[a-z0-9_]+)\.jsonl$")
JOBID_RE = re.compile(r"_(\d+(?:_\d+)?)$")
SUMMARY_TASK_RE = re.compile(r"^(humanibench_[a-z0-9_]+)\s")
LOG_TASKS_RE = re.compile(r"Tasks\s*:\s*(.+)$")
LOG_ARRAY_RE = re.compile(r"\[array\].*->\s*(humanibench_[a-z0-9_]+)")


def model_tag(model_name):
    tag = (model_name or "model").rstrip("/").split("/")[-1]
    return JOBID_RE.sub("", tag)


def model_from_dir(path):
    """Model tag from the folder name.

    Old layout: <model>_<jobid>. New layout: <judge>/<model>/<task>, where the
    leaf is a task name and the model is the parent.
    """
    path = path.rstrip(os.sep)
    name = JOBID_RE.sub("", os.path.basename(path))
    if name.startswith("humanibench"):
        return os.path.basename(os.path.dirname(path))
    return name


def jobid_from_dir(path):
    match = JOBID_RE.search(os.path.basename(path.rstrip(os.sep)))
    return match.group(1) if match else "?"


def read_manifest(run_dir):
    """RUN.json written by humanibench_eval.sbatch, or {} for older runs."""
    try:
        with open(os.path.join(run_dir, "RUN.json")) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def newest_results_json(run_dir):
    files = glob.glob(os.path.join(run_dir, "**", "*_results.json"), recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def newest_samples(run_dir):
    by_task = {}
    for path in glob.glob(os.path.join(run_dir, "**", "*_samples_humanibench_*.jsonl"), recursive=True):
        match = SAMPLES_RE.search(os.path.basename(path))
        if not match:
            continue
        task = match.group(1)
        prev = by_task.get(task)
        if prev is None or os.path.getmtime(path) >= os.path.getmtime(prev):
            by_task[task] = path
    return by_task


def parse_tasks_csv(text):
    found = []
    for part in text.split(","):
        part = part.strip()
        if part.startswith("humanibench_") and part != "humanibench":
            found.append(part)
        elif part == "humanibench":
            found.extend(SUITE)
    return found


def tasks_from_summary(path):
    tasks = []
    try:
        with open(path) as handle:
            for line in handle:
                match = SUMMARY_TASK_RE.match(line)
                if match and match.group(1) not in tasks:
                    tasks.append(match.group(1))
    except OSError:
        pass
    return tasks


def tasks_from_log(log_path):
    tasks = []
    try:
        with open(log_path) as handle:
            for line in handle:
                match = LOG_ARRAY_RE.search(line)
                if match:
                    return [match.group(1)]
                match = LOG_TASKS_RE.search(line)
                if match:
                    parsed = parse_tasks_csv(match.group(1).strip())
                    if parsed:
                        tasks = parsed
    except OSError:
        pass
    return tasks


def summary_limit(path):
    try:
        with open(path) as handle:
            for line in handle:
                if line.startswith("limit"):
                    rest = line.split(":", 1)[-1].strip()
                    token = rest.split()[0] if rest else ""
                    if token in ("?", "0", "null", "None"):
                        return None
                    try:
                        return float(token)
                    except ValueError:
                        return token
    except OSError:
        pass
    return None


def is_limited(limit):
    if limit is None or limit == "" or limit is False:
        return False
    try:
        return float(limit) > 0
    except (TypeError, ValueError):
        return True


def is_full_n(task, n_effective):
    expected = EXPECTED_N.get(task)
    if expected is None or n_effective is None:
        return False
    return n_effective >= max(1, int(0.95 * expected))


def metric_rows(results_obj):
    rows = []
    for task in sorted(results_obj):
        for key, val in results_obj[task].items():
            if key == "alias" or "stderr" in key:
                continue
            if not isinstance(val, (int, float)):
                continue
            rows.append((task, key.split(",")[0], float(val)))
    return rows


def inspect_run(run_dir, logs_dir):
    manifest = read_manifest(run_dir)
    results_path = newest_results_json(run_dir)
    summary_path = os.path.join(run_dir, "SUMMARY.txt")
    has_summary = os.path.isfile(summary_path)
    samples = newest_samples(run_dir)
    jobid = manifest.get("job_label") or jobid_from_dir(run_dir)

    data = {}
    if results_path:
        with open(results_path) as handle:
            data = json.load(handle)

    model = data.get("model_name") or manifest.get("model_tag") or model_from_dir(run_dir)
    date = data.get("date") or manifest.get("ended") or manifest.get("started") or ""
    cfg = data.get("config") or {}
    limit = cfg.get("limit")
    if limit is None and has_summary:
        limit = summary_limit(summary_path)
    if limit is None:
        limit = manifest.get("limit")

    judge = manifest.get("judge") or {}
    judge_tag = judge.get("tag") or judge.get("served_name") or ""
    status = manifest.get("status") or ""

    cli_tasks = ""
    resolved = cfg.get("resolved_cli_args") or {}
    if isinstance(resolved, dict):
        cli_tasks = resolved.get("tasks") or ""

    log_path = (manifest.get("logs") or {}).get("job") or os.path.join(
        logs_dir, f"humanibench-eval-{jobid}.out"
    )

    tasks = []
    for source in (
        parse_tasks_csv(",".join(manifest.get("tasks") or [])),
        list((data.get("results") or {}).keys()),
        list((data.get("n-samples") or {}).keys()),
        parse_tasks_csv(str(cli_tasks)),
        list(samples),
        tasks_from_summary(summary_path) if has_summary else [],
        tasks_from_log(log_path),
    ):
        for task in source:
            if task in SUITE and task not in tasks:
                tasks.append(task)

    n_samples = data.get("n-samples") or {}
    rows = metric_rows(data.get("results") or {})
    complete = bool(results_path) and has_summary
    splits_path = os.path.join(run_dir, "humanibench_splits.json")

    records = []
    for task in tasks:
        ns = n_samples.get(task) or {}
        n_eff = ns.get("effective")
        n_orig = ns.get("original")
        records.append(
            {
                "task": task,
                "dir": os.path.abspath(run_dir),
                "jobid": jobid,
                "model": model or model_tag(os.path.basename(run_dir)),
                "judge_tag": judge_tag,
                "status": status,
                "date": date,
                "limit": limit,
                "limited": is_limited(limit),
                "n_effective": n_eff,
                "n_original": n_orig,
                "full": is_full_n(task, n_eff),
                "complete": complete,
                "results_json": results_path,
                "splits_json": splits_path if os.path.isfile(splits_path) else None,
                "metrics": [(t, m, v) for t, m, v in rows if t == task],
            }
        )
    return records


def model_matches(record, needle):
    if not needle:
        return True
    needle = needle.lower()
    tag = model_tag(record["model"]).lower()
    name = (record["model"] or "").lower()
    folder = os.path.basename(record["dir"]).lower()
    return needle in tag or needle in name or folder.startswith(needle.lower())


def pick_reason(winner, loser):
    if loser["limited"] and not winner["limited"]:
        return f"limit={loser['limit']} (smoke/capped; winner is full)"
    if winner["complete"] and not loser["complete"]:
        return "incomplete (no SUMMARY + results.json)"
    if winner["full"] and not loser["full"]:
        return f"n={loser['n_effective']} below full test (expected {EXPECTED_N.get(loser['task'])})"
    if (winner["n_effective"] or 0) > (loser["n_effective"] or 0):
        return f"smaller n={loser['n_effective']} vs {winner['n_effective']}"
    if (winner["date"] or "") > (loser["date"] or ""):
        return f"older date {loser['date'] or '?'} vs {winner['date']}"
    return f"not chosen; kept {winner['jobid']}"


def incomplete_reason(rec):
    if rec.get("status") == "running":
        return "still running (RUN.json says running)"
    if rec.get("status") == "failed":
        return "died before scoring finished (RUN.json says failed)"
    if rec.get("status") == "interrupted":
        return "killed mid-run (RUN.json says interrupted)"
    return "no SUMMARY + results.json yet (still running or crashed)"


def sort_key(rec):
    # Prefer: not limited, complete, full n, larger n, newer date.
    return (
        0 if rec["limited"] else 1,
        1 if rec["complete"] else 0,
        1 if rec["full"] else 0,
        rec["n_effective"] if rec["n_effective"] is not None else -1,
        rec["date"] or "",
        rec["jobid"],
    )


def choose(records, include_limited):
    by_task = {}
    skipped = []
    for rec in records:
        by_task.setdefault(rec["task"], []).append(rec)

    chosen = {}
    for task, group in by_task.items():
        eligible = group if include_limited else [r for r in group if not r["limited"]]
        if not eligible:
            for rec in group:
                skipped.append({**rec, "reason": f"limit={rec['limit']} (pass --include-limited to keep smoke runs)"})
            continue
        finished = [r for r in eligible if r["complete"]]
        if not finished:
            for rec in eligible:
                skipped.append({**rec, "reason": incomplete_reason(rec)})
            if not include_limited:
                for rec in group:
                    if rec["limited"]:
                        skipped.append({**rec, "reason": f"limit={rec['limit']} (smoke/capped)"})
            continue
        ranked = sorted(finished, key=sort_key, reverse=True)
        winner = ranked[0]
        chosen[task] = winner
        for loser in ranked[1:]:
            skipped.append({**loser, "reason": pick_reason(winner, loser)})
        for rec in eligible:
            if rec is winner or rec in ranked[1:]:
                continue
            skipped.append({**rec, "reason": incomplete_reason(rec)})
        if not include_limited:
            for rec in group:
                if rec["limited"]:
                    skipped.append({**rec, "reason": f"limit={rec['limit']} (smoke/capped)"})
    return chosen, skipped


def record_public(rec, extra=None):
    out = {
        "task": rec["task"],
        "dir": rec["dir"],
        "jobid": rec["jobid"],
        "judge": rec.get("judge_tag") or None,
        "status": rec.get("status") or None,
        "date": rec["date"] or None,
        "n": rec["n_effective"],
        "limit": rec["limit"],
        "complete": rec["complete"],
        "full": rec["full"],
        "limited": rec["limited"],
    }
    if extra:
        out.update(extra)
    return out


def write_sources_txt(chosen, skipped, missing, model):
    judges = judges_of(chosen)
    lines = [f"# collected sources  ({model})", f"# judge: {', '.join(judges) if judges else '?'}", ""]
    lines.append("chosen:")
    if not chosen:
        lines.append("  (none)")
    for task in SUITE:
        rec = chosen.get(task)
        if not rec:
            continue
        n = rec["n_effective"]
        exp = EXPECTED_N.get(task)
        warn = ""
        if rec["limited"]:
            warn = "  [LIMITED]"
        elif not rec["full"]:
            warn = f"  [SHORT n={n} expected {exp}]"
        lines.append(
            f"  {task:<26} job={rec['jobid']:<12} n={str(n):<6} date={rec['date'] or '?'}  {rec['dir']}{warn}"
        )
    if missing:
        lines.append("")
        lines.append("missing:")
        for task in missing:
            exp = EXPECTED_N.get(task, "?")
            lines.append(f"  {task:<26} (no full complete run yet; full n≈{exp})")
    if skipped:
        lines.append("")
        lines.append("skipped:")
        for rec in skipped:
            lines.append(
                f"  {rec['task']:<26} job={rec['jobid']:<12} {rec.get('reason', '')}  {rec['dir']}"
            )
    lines.append("")
    return "\n".join(lines)


def judges_of(chosen):
    """Judge tags across the chosen runs, most common first."""
    tags = [rec["judge_tag"] for rec in chosen.values() if rec.get("judge_tag")]
    return sorted(set(tags), key=tags.count, reverse=True)


def write_summary(chosen, missing, model):
    jobs = ", ".join(f"{task.split('humanibench_', 1)[-1]}={chosen[task]['jobid']}" for task in SUITE if task in chosen)
    dates = sorted({chosen[t]["date"] for t in chosen if chosen[t]["date"]})
    judges = judges_of(chosen)
    unknown = [t for t in chosen if not chosen[t].get("judge_tag")]
    judge_line = ", ".join(judges) if judges else "?"
    if unknown:
        judge_line += f"  (+{len(unknown)} run(s) with no RUN.json)"
    lines = [
        "# lmms-eval summary  (collected from per-task jobs)",
        f"model   : {model}",
        f"judge   : {judge_line}",
        f"jobs    : {jobs or '(none)'}",
        f"dates   : {', '.join(dates) if dates else '?'}",
        f"missing : {', '.join(missing) if missing else '(none)'}",
    ]
    if len(judges) > 1:
        lines.append(f"WARNING : mixed judges across tasks ({', '.join(judges)}); scores are not comparable")
    lines += [
        "",
        f"{'task':<26} {'metric':<26} {'value':>10}",
        "-" * 64,
    ]
    for task in SUITE:
        rec = chosen.get(task)
        if not rec:
            continue
        for t, metric, val in rec["metrics"]:
            lines.append(f"{t:<26} {metric:<26} {val:>10.4f}")
    lines.append("")
    return "\n".join(lines)


def merge_splits(chosen):
    payload = {}
    for task in SUITE:
        rec = chosen.get(task)
        if not rec or not rec["splits_json"]:
            continue
        with open(rec["splits_json"]) as handle:
            data = json.load(handle)
        if task in data:
            payload[task] = data[task]
        elif len(data) == 1:
            payload[task] = next(iter(data.values()))
        else:
            # multi-task folder: take this task only
            if task in data:
                payload[task] = data[task]
    return payload


def is_report_dir(path):
    """A folder this script wrote (model root or results/collected/<model>)."""
    return os.path.isfile(os.path.join(path, "sources.json"))


def is_run_dir(path):
    if os.path.isfile(os.path.join(path, "RUN.json")):
        return True
    if is_report_dir(path):
        return False
    # lmms_eval writes <run>/<model__name>/*_results.json.
    return bool(
        glob.glob(os.path.join(path, "*_results.json"))
        or glob.glob(os.path.join(path, "*", "*_results.json"))
    )


def report_out_dir(results_dir, chosen, tag):
    """results/<judge>/<model>/ - next to the task folders it summarises.

    Falls back to results/collected/<model>/ when no chosen run names a judge
    (old flat folders, which predate RUN.json).
    """
    judges = judges_of(chosen)
    if judges:
        return os.path.join(results_dir, judges[0], tag)
    return os.path.join(results_dir, "collected", tag)


def iter_run_dirs(results_dir, depth=0, max_depth=3):
    """Yield run folders in either layout.

    Old: results/<model>_<jobid>.  New: results/<judge>/<model>/<task>.
    Report folders are walked through, not yielded; _superseded/ is skipped.
    """
    for name in sorted(os.listdir(results_dir)):
        if name.startswith(".") or name == "_superseded":
            continue
        path = os.path.join(results_dir, name)
        if not os.path.isdir(path):
            continue
        if is_run_dir(path):
            yield path
        elif depth < max_depth:
            yield from iter_run_dirs(path, depth + 1, max_depth)


def collect(results_dir, logs_dir, model_filter, include_limited):
    records = []
    for run_dir in iter_run_dirs(results_dir):
        records.extend(inspect_run(run_dir, logs_dir))
    if model_filter:
        records = [r for r in records if model_matches(r, model_filter)]

    by_model = {}
    for rec in records:
        by_model.setdefault(model_tag(rec["model"]), []).append(rec)

    reports = []
    for tag, group in sorted(by_model.items()):
        display = next((r["model"] for r in group if "/" in (r["model"] or "")), group[0]["model"])
        chosen, skipped = choose(group, include_limited)
        if not include_limited and not any(r["complete"] and not r["limited"] for r in chosen.values()):
            if not model_filter:
                continue
        missing = [t for t in SUITE if t not in chosen]
        reports.append((display, chosen, skipped, missing, group))
    return reports


def dump_report(out_dir, model, chosen, skipped, missing, dry_run):
    sources = {
        "model": model,
        "chosen": {task: record_public(rec) for task, rec in chosen.items()},
        "skipped": [record_public(rec, {"reason": rec.get("reason")}) for rec in skipped],
        "missing": missing,
    }
    summary = write_summary(chosen, missing, model)
    sources_txt = write_sources_txt(chosen, skipped, missing, model)
    splits = merge_splits(chosen)
    splits_txt = render_txt(splits) if splits else "# HumaniBench splits (from sample JSONL)\n\n(none)\n"

    print(sources_txt)
    print(summary)
    if dry_run:
        print(f"[dry-run] would write under {out_dir}")
        return out_dir

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sources.json"), "w") as handle:
        json.dump(sources, handle, indent=2)
        handle.write("\n")
    with open(os.path.join(out_dir, "SOURCES.txt"), "w") as handle:
        handle.write(sources_txt if sources_txt.endswith("\n") else sources_txt + "\n")
    with open(os.path.join(out_dir, "SUMMARY.txt"), "w") as handle:
        handle.write(summary if summary.endswith("\n") else summary + "\n")
    with open(os.path.join(out_dir, "humanibench_splits.json"), "w") as handle:
        json.dump(splits, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with open(os.path.join(out_dir, "humanibench_splits.txt"), "w") as handle:
        handle.write(splits_txt if splits_txt.endswith("\n") else splits_txt + "\n")
    print("[collect] wrote", out_dir)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    parser.add_argument("--logs-dir", default=os.path.join(HERE, "logs"))
    parser.add_argument(
        "--out-dir",
        default=None,
        help="default: <results-dir>/<judge>/<model-tag> (collected/<model-tag> if no judge is known)",
    )
    parser.add_argument("--model", default="", help="substring match on model name / folder prefix")
    parser.add_argument("--include-limited", action="store_true", help="also keep --limit / smoke runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        print(f"[collect] no results dir: {results_dir}", file=sys.stderr)
        return 1

    reports = collect(results_dir, os.path.abspath(args.logs_dir), args.model, args.include_limited)
    if not reports:
        print("[collect] no matching runs (full complete jobs, or pass --include-limited / --model)")
        return 1

    for model, chosen, skipped, missing, _group in reports:
        tag = model_tag(model)
        out_dir = args.out_dir or report_out_dir(results_dir, chosen, tag)
        if args.out_dir and len(reports) > 1:
            out_dir = os.path.join(args.out_dir, tag)
        dump_report(out_dir, model, chosen, skipped, missing, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
