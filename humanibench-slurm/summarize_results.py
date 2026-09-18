#!/usr/bin/env python3
"""Distil lmms_eval's *_results.json into a human-readable SUMMARY.txt.

Usage:
    python3 summarize_results.py <results_dir> [job_id] [model_args] [tasks] [limit]

Writes <results_dir>/SUMMARY.txt and prints it. Standalone (stdlib only).
"""
import glob
import json
import os
import sys


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    jobid = sys.argv[2] if len(sys.argv) > 2 else "?"
    model_args = sys.argv[3] if len(sys.argv) > 3 else ""
    limit = sys.argv[5] if len(sys.argv) > 5 else "?"

    files = sorted(
        glob.glob(os.path.join(results_dir, "**", "*_results.json"), recursive=True),
        key=os.path.getmtime,
    )
    if not files:
        print(f"[summary] no *_results.json under {results_dir}; skipping SUMMARY.txt")
        return 0

    data = json.load(open(files[-1]))
    res = data.get("results", {})
    model = data.get("model_name") or (model_args.split(",")[0] if model_args else "?")

    lines = [
        f"# lmms-eval summary  (job {jobid})",
        f"model   : {model}",
        f"limit   : {limit} per task",
        f"date    : {data.get('date', '?')}",
        f"results : {files[-1]}",
        "",
        f"{'task':<26} {'metric':<26} {'value':>10}",
        "-" * 64,
    ]
    for task in sorted(res):
        for key, val in res[task].items():
            if key == "alias" or "stderr" in key:
                continue
            if not isinstance(val, (int, float)):
                continue
            lines.append(f"{task:<26} {key.split(',')[0]:<26} {val:>10.4f}")

    out = os.path.join(results_dir, "SUMMARY.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[summary] wrote", out)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
