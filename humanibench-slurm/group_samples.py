#!/usr/bin/env python3
"""Post-aggregate HumaniBench sample JSONL by attribute / language / attack_type.

Usage:
    python3 group_samples.py <results_dir>

Reads the newest *_samples_humanibench_*.jsonl per task under results_dir
(works on a SLURM results/<model>_<jobid>/ tree or on outputs/<run>/).

Writes:
    <results_dir>/humanibench_splits.json
    <results_dir>/humanibench_splits.txt

Standalone (stdlib only). T6 social_attribute combos are kept as stored (no comma split).
T5 has no attribute column — overall only. T7 retention = 100 * attacked / clean.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

SKIP_KEYS = {"doc_id", "target", "filtered_resps", "input", "token_counts", "doc_hash"}
GROUP_KEYS = ("attribute", "language", "social_attribute", "attack_type")
TASK_FROM_NAME = re.compile(r"samples_(humanibench_[a-z0-9_]+)\.jsonl$")
METRIC_ALIASES = {"map_50": "map50", "map_75": "map75", "missing_pct": "missing"}
MCQ_TASKS = {"humanibench_t3", "humanibench_t4_closed"}

# T4 high vs low resource. High-resource = Latin/CJK with large VLM pretraining coverage.
HIGH_RESOURCE = {"English", "French", "Spanish", "Portuguese", "Mandarin", "Korean"}
LOW_RESOURCE = {"Bengali", "Persian", "Punjabi", "Tamil", "Urdu"}


def mean_n(scores):
    if not scores:
        return {"mean": 0.0, "n": 0}
    return {"mean": sum(scores) / len(scores), "n": len(scores)}


def task_name(path):
    match = TASK_FROM_NAME.search(os.path.basename(path))
    return match.group(1) if match else os.path.basename(path)


def newest_jsonl_per_task(results_dir):
    by_task = {}
    for path in glob.glob(os.path.join(results_dir, "**", "*_samples_humanibench_*.jsonl"), recursive=True):
        name = task_name(path)
        prev = by_task.get(name)
        if prev is None or os.path.getmtime(path) >= os.path.getmtime(prev):
            by_task[name] = path
    return by_task


def collect_metrics(path):
    """metric -> list of {score, attribute?, language?, social_attribute?, attack_type?}."""
    buckets = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key, val in row.items():
                if key in SKIP_KEYS or not isinstance(val, dict) or "score" not in val:
                    continue
                try:
                    score = float(val["score"])
                except (TypeError, ValueError):
                    continue
                item = {"score": score}
                for group_key in GROUP_KEYS:
                    if val.get(group_key) is not None:
                        item[group_key] = val[group_key]
                buckets[key].append(item)
    return buckets


def normalize_metrics(task, metrics):
    out = dict(metrics)
    for old, new in METRIC_ALIASES.items():
        if old not in out:
            continue
        if new not in out:
            out[new] = out[old]
        del out[old]
    if task in MCQ_TASKS and "accuracy" in out and "accuracy_statistical" not in out:
        out["accuracy_statistical"] = out.pop("accuracy")
    return out


def group_axis(items, key):
    buckets = defaultdict(list)
    for item in items:
        if key not in item:
            continue
        buckets[str(item[key])].append(item["score"])
    if not buckets:
        return {}
    return {name: mean_n(scores) for name, scores in sorted(buckets.items())}


def language_gap(items):
    high, low = [], []
    for item in items:
        lang = item.get("language")
        if lang in HIGH_RESOURCE:
            high.append(item["score"])
        elif lang in LOW_RESOURCE:
            low.append(item["score"])
    if not high and not low:
        return {}
    high_m = mean_n(high)
    low_m = mean_n(low)
    gap = None
    if high_m["n"] and low_m["n"]:
        gap = high_m["mean"] - low_m["mean"]
    return {
        "high_resource": {**high_m, "languages": sorted(HIGH_RESOURCE)},
        "low_resource": {**low_m, "languages": sorted(LOW_RESOURCE)},
        "high_minus_low": gap,
    }


def retention_block(items):
    clean = [i["score"] for i in items if i.get("attack_type") == "clean"]
    attacked = [i["score"] for i in items if i.get("attack_type") not in (None, "clean")]
    clean_m = mean_n(clean)
    attacked_m = mean_n(attacked)
    overall = 0.0 if clean_m["mean"] <= 0 else 100.0 * attacked_m["mean"] / clean_m["mean"]
    by_attack = {}
    for name, stats in group_axis(items, "attack_type").items():
        entry = dict(stats)
        if name != "clean" and clean_m["mean"] > 0:
            entry["retention"] = 100.0 * stats["mean"] / clean_m["mean"]
        by_attack[name] = entry
    by_attr = {}
    attr_names = sorted({str(i["attribute"]) for i in items if "attribute" in i})
    for attr in attr_names:
        subset = [i for i in items if str(i.get("attribute")) == attr]
        c = mean_n([i["score"] for i in subset if i.get("attack_type") == "clean"])
        a = mean_n([i["score"] for i in subset if i.get("attack_type") not in (None, "clean")])
        ret = 0.0 if c["mean"] <= 0 else 100.0 * a["mean"] / c["mean"]
        by_attr[attr] = {"retention": ret, "clean": c, "attacked": a}
    return {
        "overall": {
            "mean": overall,
            "n": len(items),
            "clean": clean_m,
            "attacked": attacked_m,
        },
        "by_attack_type": by_attack,
        "by_attribute": by_attr,
    }


def summarize_task(metrics):
    out = {}
    for metric, items in sorted(metrics.items()):
        scores = [i["score"] for i in items]
        block = {"overall": mean_n(scores)}
        for key in GROUP_KEYS:
            grouped = group_axis(items, key)
            if grouped:
                block[f"by_{key}"] = grouped
        gap = language_gap(items)
        if gap:
            block["high_low_resource"] = gap
        if metric == "retention" and any("attack_type" in i for i in items):
            block = retention_block(items)
            # keep grouped means of the raw score as well
            attr = group_axis(items, "attribute")
            if attr:
                block.setdefault("by_attribute_score", attr)
        out[metric] = block
    return out


def fmt_mean(stats):
    return f"{stats['mean']:.4f} (n={stats['n']})"


def render_txt(payload):
    lines = ["# HumaniBench splits (from sample JSONL)", ""]
    for task, metrics in sorted(payload.items()):
        lines.append(f"## {task}")
        for metric, block in metrics.items():
            overall = block.get("overall", {})
            if "clean" in overall:
                lines.append(
                    f"  {metric}: retention={overall.get('mean', 0):.4f}  "
                    f"clean={fmt_mean(overall['clean'])}  attacked={fmt_mean(overall['attacked'])}"
                )
            else:
                lines.append(f"  {metric}: {fmt_mean(overall)}")
            for axis in ("by_attribute", "by_language", "by_social_attribute", "by_attack_type"):
                grouped = block.get(axis)
                if not grouped:
                    continue
                parts = []
                for name, stats in grouped.items():
                    if "mean" in stats:
                        extra = ""
                        if axis == "by_attack_type" and "retention" in stats:
                            extra = f" ret={stats['retention']:.2f}"
                        parts.append(f"{name}={stats['mean']:.3f}(n={stats['n']}{extra})")
                    elif "retention" in stats:
                        parts.append(f"{name}={stats['retention']:.2f}")
                if parts:
                    lines.append(f"    {axis}: " + ", ".join(parts))
            gap = block.get("high_low_resource")
            if gap:
                hi, lo = gap["high_resource"], gap["low_resource"]
                gap_v = gap.get("high_minus_low")
                gap_s = "n/a" if gap_v is None else f"{gap_v:.4f}"
                lines.append(
                    f"    high/low resource: high={fmt_mean(hi)}  low={fmt_mean(lo)}  gap={gap_s}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    files = newest_jsonl_per_task(results_dir)
    if not files:
        print(f"[splits] no *_samples_humanibench_*.jsonl under {results_dir}; skipping")
        return 0

    payload = {}
    for task, path in sorted(files.items()):
        payload[task] = summarize_task(normalize_metrics(task, collect_metrics(path)))

    json_path = os.path.join(results_dir, "humanibench_splits.json")
    txt_path = os.path.join(results_dir, "humanibench_splits.txt")
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    text = render_txt(payload)
    with open(txt_path, "w") as handle:
        handle.write(text)
    print("[splits] wrote", json_path)
    print("[splits] wrote", txt_path)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
