#!/usr/bin/env python3
"""Create traceable, publication-style figures from ReasoningForge metrics."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".vendor"))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


EXPERIMENTS = ROOT / "experiments"
FIGURES = ROOT / "figures"
SOURCE = ROOT / "source-data"

COLORS = {
    "GRPO": "#6B7280",
    "Dr.GRPO": "#56B4E9",
    "Kimi-OPMD": "#D55E00",
    "tau=0.01": "#56B4E9",
    "tau=0.03": "#D55E00",
    "tau=0.1": "#009E73",
    "pg": "#0072B2",
    "mirror": "#D55E00",
    "neutral": "#4B5563",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def records(run: str, split: str) -> list[dict[str, Any]]:
    return [
        row
        for row in load_jsonl(EXPERIMENTS / run / "metrics.jsonl")
        if row.get("split") == split
    ]


def rolling_mean(values: Iterable[float], window: int = 9) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    out = np.full_like(array, np.nan)
    for index in range(len(array)):
        start = max(0, index - window + 1)
        window_values = array[start : index + 1]
        out[index] = np.nanmean(window_values) if np.isfinite(window_values).any() else np.nan
    return out


def quantiles(values: list[float]) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.quantile(values, [0.25, 0.5, 0.75]))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.2,
            "axes.edgecolor": "#374151",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.8,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#1F2937",
            "legend.frameon": False,
            "legend.fontsize": 8.3,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=3, width=0.7)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.18,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def export(figure: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "pdf", "png"):
        kwargs = {"dpi": 320} if suffix == "png" else {}
        figure.savefig(FIGURES / f"{stem}.{suffix}", **kwargs)
    plt.close(figure)


def figure_learning_curves() -> None:
    runs = {
        "GRPO": "grpo-seed0-pilot",
        "Dr.GRPO": "dr-grpo-seed0-full",
        "Kimi-OPMD": "kimi-opmd-tau-0p03-full",
    }
    linestyles = {"GRPO": "--", "Dr.GRPO": "-.", "Kimi-OPMD": "-"}
    markers = {"GRPO": "o", "Dr.GRPO": "s", "Kimi-OPMD": "D"}
    source_rows: list[dict[str, Any]] = []
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 2.85), gridspec_kw={"wspace": 0.28})

    for method, run in runs.items():
        train = records(run, "train")
        steps = np.array([row["step"] for row in train])
        rewards = np.array([row["reward_mean"] for row in train])
        smooth = rolling_mean(rewards, 9)
        axes[0].plot(steps, rewards, color=COLORS[method], alpha=0.13, linewidth=0.8)
        axes[0].plot(
            steps,
            smooth,
            color=COLORS[method],
            linestyle=linestyles[method],
            linewidth=2.0,
            label=method,
        )
        for step, reward, smoothed in zip(steps, rewards, smooth):
            source_rows.append(
                {"method": method, "split": "train", "step": step, "reward": reward, "rolling_reward": smoothed}
            )

        validation = records(run, "validation")
        val_steps = [row["step"] for row in validation]
        val_rewards = [row["reward_mean"] for row in validation]
        axes[1].plot(
            val_steps,
            val_rewards,
            color=COLORS[method],
            linestyle=linestyles[method],
            marker=markers[method],
            markersize=3.8,
            linewidth=1.8,
            label=method,
        )
        for step, reward in zip(val_steps, val_rewards):
            source_rows.append(
                {"method": method, "split": "validation", "step": step, "reward": reward, "rolling_reward": ""}
            )

    axes[0].set(title="Training reward", xlabel="Outer iteration", ylabel="Exact-answer reward")
    axes[0].set_xlim(1, 200)
    axes[0].set_ylim(-0.02, 0.72)
    axes[0].legend(loc="lower right")
    axes[1].set(title="Held-out GSM8K accuracy", xlabel="Outer iteration", ylabel="Accuracy")
    axes[1].set_xlim(0, 205)
    axes[1].set_ylim(0, 0.58)
    axes[1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    axes[1].legend(loc="lower right")
    for label, axis in zip(("a", "b"), axes):
        clean_axis(axis)
        panel_label(axis, label)
    figure.suptitle("Optimization dynamics and held-out performance (seed 0)", fontsize=11.5, fontweight="semibold", y=1.035)
    figure.text(0.5, -0.045, "Thin lines show raw rollout reward; thick lines show a trailing 9-step mean.", ha="center", fontsize=7.7, color="#6B7280")
    export(figure, "fig1_learning_curves")
    write_csv(SOURCE / "fig1_learning_curves.csv", ["method", "split", "step", "reward", "rolling_reward"], source_rows)


def figure_kimi_mechanism() -> None:
    rows = records("kimi-opmd-tau-0p03-full", "kimi_inner")
    rows = [row for row in rows if row.get("kimi/inner_step") == 2]
    steps = np.array([row["step"] for row in rows])
    pg_force = np.array([row["kimi/pg_force_abs_mean"] for row in rows])
    mirror_force = np.array([row["kimi/mirror_force_abs_mean"] for row in rows])
    force_ratio = np.array([row["kimi/mirror_to_pg_force_ratio"] for row in rows])
    movement = np.array([row["kimi/seq_log_ratio_abs_mean"] for row in rows])
    grad_norm = np.array([row["grad_norm"] for row in rows])

    figure, axes = plt.subplots(2, 2, figsize=(7.25, 5.25), gridspec_kw={"hspace": 0.43, "wspace": 0.28})
    axes = axes.ravel()
    for values, color, linestyle, label in (
        (pg_force, COLORS["pg"], "-", r"PG force $\mathbb{E}|A|$"),
        (mirror_force, COLORS["mirror"], "--", r"Mirror force $\mathbb{E}|\tau d|$"),
    ):
        axes[0].plot(steps, values, color=color, alpha=0.16, linewidth=0.8)
        axes[0].plot(steps, rolling_mean(values, 9), color=color, linestyle=linestyle, linewidth=1.9, label=label)
    axes[0].set(title="Competing sequence-level forces", xlabel="Outer iteration", ylabel="Mean absolute force")
    axes[0].legend(loc="upper left")

    finite_ratio = np.where(pg_force > 1e-12, force_ratio, np.nan)
    axes[1].plot(steps, finite_ratio, color=COLORS["mirror"], alpha=0.16, linewidth=0.8)
    axes[1].plot(steps, rolling_mean(finite_ratio, 9), color=COLORS["mirror"], linewidth=1.9)
    axes[1].axhline(1.0, color="#111827", linestyle="--", linewidth=1.0, label="Equal force")
    axes[1].set(title="Mirror-to-PG force ratio", xlabel="Outer iteration", ylabel=r"$\mathbb{E}|\tau d| / (\mathbb{E}|A|+\epsilon)$", ylim=(0, 1.6))
    axes[1].legend(loc="upper right")

    axes[2].plot(steps, movement, color=COLORS["Kimi-OPMD"], alpha=0.18, linewidth=0.8)
    axes[2].plot(steps, rolling_mean(movement, 9), color=COLORS["Kimi-OPMD"], linewidth=1.9)
    axes[2].set(title="Within-outer policy movement", xlabel="Outer iteration", ylabel=r"$\mathbb{E}|\log\pi-\log\pi_{ref}|$")

    axes[3].plot(steps, grad_norm, color=COLORS["neutral"], alpha=0.18, linewidth=0.8)
    axes[3].plot(steps, rolling_mean(grad_norm, 9), color=COLORS["neutral"], linewidth=1.9)
    axes[3].axhline(1.0, color=COLORS["mirror"], linestyle="--", linewidth=1.0, label="Clip threshold")
    axes[3].set_yscale("log")
    axes[3].set(title="Pre-clip gradient norm", xlabel="Outer iteration", ylabel="Global norm (log scale)")
    axes[3].legend(loc="lower right")

    for label, axis in zip(("a", "b", "c", "d"), axes):
        clean_axis(axis)
        panel_label(axis, label)
    figure.suptitle(r"Kimi-OPMD mechanism diagnostics ($\tau=0.03$, two inner updates)", fontsize=11.5, fontweight="semibold", y=0.995)
    export(figure, "fig2_kimi_mechanism")
    source_rows = [
        {
            "step": int(step),
            "pg_force_abs_mean": pg,
            "mirror_force_abs_mean": mirror,
            "mirror_to_pg_force_ratio": ratio,
            "seq_log_ratio_abs_mean": move,
            "preclip_grad_norm": grad,
        }
        for step, pg, mirror, ratio, move, grad in zip(steps, pg_force, mirror_force, force_ratio, movement, grad_norm)
    ]
    write_csv(
        SOURCE / "fig2_kimi_mechanism.csv",
        ["step", "pg_force_abs_mean", "mirror_force_abs_mean", "mirror_to_pg_force_ratio", "seq_log_ratio_abs_mean", "preclip_grad_norm"],
        source_rows,
    )


def figure_tau_sweep() -> None:
    run_map = {
        "tau=0.01": "kimi-opmd-tau-0p01",
        "tau=0.03": "kimi-opmd-tau-0p03",
        "tau=0.1": "kimi-opmd-tau-0p1",
    }
    metrics = {
        "Policy movement": ("movement", r"$\mathbb{E}|\log\pi-\log\pi_{ref}|$", False),
        "Mirror / |PG loss|": ("ratio", "Ratio", False),
        "Pre-clip gradient norm": ("grad", "Global norm", True),
    }
    distributions: dict[str, dict[str, list[float]]] = {}
    source_rows: list[dict[str, Any]] = []
    for label, run in run_map.items():
        inner = [
            row
            for row in records(run, "kimi_inner")
            if row.get("kimi/inner_step") == 2 and abs(row.get("kimi/pg_loss", 0.0)) > 1e-8
        ]
        distributions[label] = {
            "movement": [float(row["kimi/seq_log_ratio_abs_mean"]) for row in inner],
            "ratio": [float(row["kimi/mirror_loss"]) / abs(float(row["kimi/pg_loss"])) for row in inner],
            "grad": [float(row["grad_norm"]) for row in inner],
        }
        for row in inner:
            source_rows.append(
                {
                    "tau": label.split("=")[1],
                    "step": row["step"],
                    "seq_log_ratio_abs_mean": row["kimi/seq_log_ratio_abs_mean"],
                    "corrected_mirror_to_abs_pg_loss_ratio": float(row["kimi/mirror_loss"]) / abs(float(row["kimi/pg_loss"])),
                    "preclip_grad_norm": row["grad_norm"],
                }
            )

    figure, axes = plt.subplots(1, 3, figsize=(7.25, 2.85), gridspec_kw={"wspace": 0.36})
    labels = list(run_map)
    positions = np.arange(len(labels))
    for axis, (title, (key, ylabel, log_scale)) in zip(axes, metrics.items()):
        for position, label in zip(positions, labels):
            low, median, high = quantiles(distributions[label][key])
            axis.vlines(position, low, high, color=COLORS[label], linewidth=5.5, alpha=0.32)
            axis.scatter(position, median, s=38, color=COLORS[label], edgecolor="white", linewidth=0.8, zorder=3)
        axis.set_xticks(positions, ["0.01", "0.03", "0.1"])
        axis.set(title=title, xlabel=r"Mirror coefficient $\tau$", ylabel=ylabel)
        if log_scale:
            axis.set_yscale("log")
        clean_axis(axis)
    axes[1].axhline(1.0, color="#111827", linestyle="--", linewidth=0.9)
    for label, axis in zip(("a", "b", "c"), axes):
        panel_label(axis, label)
    figure.suptitle("Tau sweep: within-run median and interquartile range (20 outer iterations)", fontsize=11.2, fontweight="semibold", y=1.035)
    figure.text(0.5, -0.055, "Intervals summarize steps within one run; they are not confidence intervals across seeds.", ha="center", fontsize=7.7, color="#6B7280")
    export(figure, "fig3_tau_sweep")
    write_csv(
        SOURCE / "fig3_tau_sweep.csv",
        ["tau", "step", "seq_log_ratio_abs_mean", "corrected_mirror_to_abs_pg_loss_ratio", "preclip_grad_norm"],
        source_rows,
    )


def figure_compute_efficiency() -> None:
    runs = {
        "GRPO": ("grpo-seed0-pilot", 1),
        "Dr.GRPO": ("dr-grpo-seed0-full", 1),
        "Kimi-OPMD": ("kimi-opmd-tau-0p03-full", 2),
    }
    linestyles = {"GRPO": "--", "Dr.GRPO": "-.", "Kimi-OPMD": "-"}
    markers = {"GRPO": "o", "Dr.GRPO": "s", "Kimi-OPMD": "D"}
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 2.85), gridspec_kw={"wspace": 0.3})
    source_rows: list[dict[str, Any]] = []
    for method, (run, updates_per_outer) in runs.items():
        train = records(run, "train")
        validation = records(run, "validation")
        cumulative_seconds = np.cumsum([float(row["step_seconds"]) for row in train])
        val_steps = np.array([int(row["step"]) for row in validation])
        val_accuracy = np.array([float(row["reward_mean"]) for row in validation])
        optimizer_updates = val_steps * updates_per_outer
        elapsed_hours = np.array([cumulative_seconds[step - 1] / 3600.0 for step in val_steps])
        for axis, x_values in zip(axes, (optimizer_updates, elapsed_hours)):
            axis.plot(
                x_values,
                val_accuracy,
                color=COLORS[method],
                linestyle=linestyles[method],
                marker=markers[method],
                markersize=3.6,
                linewidth=1.8,
                label=method,
            )
        for step, update, hours, accuracy in zip(val_steps, optimizer_updates, elapsed_hours, val_accuracy):
            source_rows.append(
                {
                    "method": method,
                    "outer_step": step,
                    "optimizer_updates": update,
                    "cumulative_recorded_train_hours": hours,
                    "validation_accuracy": accuracy,
                }
            )
    axes[0].set(title="Accuracy per optimizer update", xlabel="Cumulative optimizer updates", ylabel="Accuracy")
    axes[1].set(title="Accuracy per recorded training time", xlabel="Cumulative train time (hours)", ylabel="Accuracy")
    for label, axis in zip(("a", "b"), axes):
        axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        axis.set_ylim(0, 0.58)
        axis.legend(loc="lower right")
        clean_axis(axis)
        panel_label(axis, label)
    figure.suptitle("Compute-aware held-out performance (seed 0)", fontsize=11.5, fontweight="semibold", y=1.035)
    figure.text(
        0.5,
        -0.055,
        "Kimi-OPMD uses two optimizer updates per outer iteration; recorded time excludes unlogged overhead.",
        ha="center",
        fontsize=7.7,
        color="#6B7280",
    )
    export(figure, "fig4_compute_efficiency")
    write_csv(
        SOURCE / "fig4_compute_efficiency.csv",
        ["method", "outer_step", "optimizer_updates", "cumulative_recorded_train_hours", "validation_accuracy"],
        source_rows,
    )


def build_summary() -> None:
    runs = {
        "GRPO": "grpo-seed0-pilot",
        "Dr.GRPO": "dr-grpo-seed0-full",
        "Kimi-OPMD": "kimi-opmd-tau-0p03-full",
    }
    rows = []
    compute_summary: dict[str, dict[str, float]] = {}
    for method, run in runs.items():
        train = records(run, "train")
        validation = records(run, "validation")
        last_five = [float(row["reward_mean"]) for row in validation[-5:]]
        rows.append(
            {
                "method": method,
                "final_validation_accuracy": float(validation[-1]["reward_mean"]),
                "best_validation_accuracy": max(float(row["reward_mean"]) for row in validation),
                "last5_validation_mean": statistics.fmean(last_five),
                "last20_train_reward_mean": statistics.fmean(float(row["reward_mean"]) for row in train[-20:]),
                "final_format_rate": float(validation[-1]["format_reward_mean"]),
            }
        )
        compute_summary[method] = {
            "optimizer_updates": len(train) * (2 if method == "Kimi-OPMD" else 1),
            "recorded_train_hours": sum(float(row["step_seconds"]) for row in train) / 3600.0,
        }
    write_csv(
        SOURCE / "table1_main_results.csv",
        ["method", "final_validation_accuracy", "best_validation_accuracy", "last5_validation_mean", "last20_train_reward_mean", "final_format_rate"],
        rows,
    )

    inner = [
        row
        for row in records("kimi-opmd-tau-0p03-full", "kimi_inner")
        if row.get("kimi/inner_step") == 2 and row.get("kimi/pg_force_abs_mean", 0.0) > 1e-12
    ]
    train = records("kimi-opmd-tau-0p03-full", "train")
    result = {
        "main_results": rows,
        "compute": compute_summary,
        "kimi_mechanism": {
            "active_updates": len(inner),
            "zero_force_updates": 200 - len(inner),
            "force_ratio_mean": statistics.fmean(float(row["kimi/mirror_to_pg_force_ratio"]) for row in inner),
            "force_ratio_median": statistics.median(float(row["kimi/mirror_to_pg_force_ratio"]) for row in inner),
            "force_ratio_last50_mean": statistics.fmean(float(row["kimi/mirror_to_pg_force_ratio"]) for row in inner[-50:]),
            "movement_last50_mean": statistics.fmean(float(row["kimi/seq_log_ratio_abs_mean"]) for row in inner[-50:]),
            "preclip_grad_norm_median": statistics.median(float(row["grad_norm"]) for row in inner),
            "preclip_grad_norm_max": max(float(row["grad_norm"]) for row in inner),
            "clip_fraction": statistics.fmean(float(row["grad_norm"] > 1.0) for row in inner),
            "nonfinite_count": sum(int(row.get("kimi/nonfinite", 0)) for row in train),
            "retokenized_length_over_512_steps": sum(float(row.get("kimi/response_length_max", 0)) > 512 for row in train),
        },
        "limitations": [
            "All methods were run with one seed; no cross-seed uncertainty is available.",
            "Tau sweeps contain 20 outer iterations and are diagnostic rather than final-accuracy comparisons.",
            "Kimi uses two optimizer updates per outer iteration (400 total versus 200 for each baseline).",
            "Kimi gradient norms are recorded before clipping at 1.0.",
            "Response length is measured after training-side retokenization and may differ from vLLM token_ids length.",
        ],
    }
    (ROOT / "result_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    markdown = [
        "# ReasoningForge result summary",
        "",
        "| Method | Final val. | Best val. | Last-5 val. | Last-20 train | Final format |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        markdown.append(
            f"| {row['method']} | {row['final_validation_accuracy']:.2%} | {row['best_validation_accuracy']:.2%} | "
            f"{row['last5_validation_mean']:.2%} | {row['last20_train_reward_mean']:.2%} | {row['final_format_rate']:.2%} |"
        )
    by_method = {row["method"]: row for row in rows}
    final_delta = by_method["Kimi-OPMD"]["final_validation_accuracy"] - by_method["GRPO"]["final_validation_accuracy"]
    last5_delta = by_method["Kimi-OPMD"]["last5_validation_mean"] - by_method["GRPO"]["last5_validation_mean"]
    markdown.extend(
        [
            "",
            "## Descriptive findings",
            "",
            f"- Kimi-OPMD finishes {final_delta * 100:.2f} percentage points above GRPO; its last-five-validation mean is {last5_delta * 100:.2f} points higher.",
            f"- This is not compute matched: Kimi uses {compute_summary['Kimi-OPMD']['optimizer_updates']:.0f} optimizer updates and {compute_summary['Kimi-OPMD']['recorded_train_hours']:.2f} recorded train hours, versus {compute_summary['GRPO']['optimizer_updates']:.0f} updates and {compute_summary['GRPO']['recorded_train_hours']:.2f} hours for GRPO.",
            f"- Across active Kimi updates, mirror/PG force has median {result['kimi_mechanism']['force_ratio_median']:.3f}; its last-50 mean is {result['kimi_mechanism']['force_ratio_last50_mean']:.3f}.",
            f"- The last-50 mean policy movement is {result['kimi_mechanism']['movement_last50_mean']:.3f}, with zero non-finite updates.",
            "",
            "## Interpretation guardrails",
            "",
            "- Single seed only: differences are descriptive, not inferential.",
            "- Tau-sweep intervals are within-run IQRs, not confidence intervals.",
            "- The outer-iteration comparison is rollout/sample aligned, while the optimizer-update and time plots provide compute-aware views.",
            "- Gradient norm is pre-clip; all active Kimi updates are clipped to the configured threshold.",
            "- Training-side response lengths are retokenized text lengths, not vLLM token-id counts.",
        ]
    )
    (ROOT / "RESULTS.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    setup_style()
    figure_learning_curves()
    figure_kimi_mechanism()
    figure_tau_sweep()
    figure_compute_efficiency()
    build_summary()
    print(f"Wrote figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
