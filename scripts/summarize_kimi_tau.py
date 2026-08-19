#!/usr/bin/env python3
"""Summarize Kimi OPMD tau diagnostics without ranking by short-run accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def values(records: list[dict[str, Any]], key: str) -> list[float]:
    return [float(record[key]) for record in records if key in record]


def mean_or_nan(items: list[float]) -> float:
    return statistics.fmean(items) if items else math.nan


def max_or_nan(items: list[float]) -> float:
    return max(items) if items else math.nan


def median_or_nan(items: list[float]) -> float:
    return statistics.median(items) if items else math.nan


def final_or_nan(items: list[float]) -> float:
    return items[-1] if items else math.nan


def summarize(run_dir: Path) -> dict[str, Any]:
    records = load_jsonl(run_dir / "metrics.jsonl")
    outer = [record for record in records if record.get("split") == "train"]
    events = [record for record in records if record.get("split") == "kimi_outer"]
    inner1 = [
        record
        for record in records
        if record.get("split") == "kimi_inner" and record.get("kimi/inner_step") == 1
    ]
    inner2 = [
        record
        for record in records
        if record.get("split") == "kimi_inner" and record.get("kimi/inner_step") == 2
    ]
    if not outer or not inner1 or not inner2:
        raise ValueError(f"Incomplete Kimi metrics in {run_dir}.")

    ratio_abs = values(inner2, "kimi/seq_log_ratio_abs_mean")
    mirror_ratio = values(inner2, "kimi/mirror_to_abs_pg_loss_ratio")
    magnitude_ratio = values(inner2, "kimi/mirror_to_pg_magnitude_ratio")
    nonfinite = values(inner1 + inner2, "kimi/nonfinite")
    return {
        "run_dir": str(run_dir),
        "tau": float(inner2[0]["kimi/tau"]),
        "outer_steps": len(outer),
        "inner_epochs": int(events[0]["kimi/inner_epochs"]),
        "optimizer_resets": int(sum(values(events, "kimi/optimizer_reset"))),
        "reference_refreshes": int(sum(values(events, "kimi/reference_refreshed"))),
        "inner1_mirror_loss_max": max_or_nan(values(inner1, "kimi/mirror_loss")),
        "inner2_seq_log_ratio_abs_mean_avg": mean_or_nan(ratio_abs),
        "inner2_seq_log_ratio_abs_mean_median": median_or_nan(ratio_abs),
        "inner2_seq_log_ratio_abs_mean_max": max_or_nan(ratio_abs),
        "inner2_seq_log_ratio_abs_mean_final": final_or_nan(ratio_abs),
        "inner2_mirror_loss_avg": mean_or_nan(values(inner2, "kimi/mirror_loss")),
        "inner2_mirror_to_abs_pg_ratio_avg": mean_or_nan(mirror_ratio),
        "inner2_mirror_to_abs_pg_ratio_max": max_or_nan(mirror_ratio),
        "inner2_mirror_to_pg_magnitude_ratio_avg": mean_or_nan(magnitude_ratio),
        "inner2_grad_norm_avg": mean_or_nan(values(inner2, "grad_norm")),
        "inner2_grad_norm_max": max_or_nan(values(inner2, "grad_norm")),
        "train_reward_avg": mean_or_nan(values(outer, "reward_mean")),
        "train_reward_final": final_or_nan(values(outer, "reward_mean")),
        "response_length_avg": mean_or_nan(values(inner2, "kimi/response_length_mean")),
        "nonfinite_count": int(sum(nonfinite)),
        "exact_zero_movement_steps": sum(value <= 1e-8 for value in ratio_abs),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    summaries = [summarize(path) for path in args.run_dirs]
    summaries.sort(key=lambda item: item["tau"])
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))
    print(f"Wrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
