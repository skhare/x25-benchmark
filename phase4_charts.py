"""
Phase 4 — Charts and summary.

Reads strategy_results.json and emits:
    radar.png   — 5-axis radar (quality, cost-eff, speed, audit, consistency)
    results.png — cost-vs-quality scatter, one labeled point per strategy
    heatmap.png — per-task quality heatmap (5 tasks × 4 strategies)
    summary.md  — headline numbers
"""

from __future__ import annotations

import json
import os
import sys
from typing import Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

DATA_DIR  = os.path.join(HERE, "data")
IN_PATH   = os.path.join(DATA_DIR, "strategy_results.json")
RADAR     = os.path.join(DATA_DIR, "radar.png")
RESULTS   = os.path.join(DATA_DIR, "results.png")
HEATMAP   = os.path.join(DATA_DIR, "heatmap.png")
SUMMARY   = os.path.join(DATA_DIR, "summary.md")

STRAT_LABELS = {
    "always_frontier": "Always Frontier",
    "always_cheap":    "Always Cheap",
    "random":          "Random",
    "x25":             "X25",
}

STRAT_COLORS = {
    "always_frontier": "#d4634a",
    "always_cheap":    "#6ba368",
    "random":          "#9b8cb6",
    "x25":             "#1f3a5f",
}


def _norm(x: float, x_min: float, x_max: float) -> float:
    if x_max - x_min < 1e-9:
        return 0.5
    return (x - x_min) / (x_max - x_min)


def _consistency(per_task_quality: dict[str, float]) -> float:
    vals = list(per_task_quality.values())
    if not vals:
        return 0.0
    std = float(np.std(vals))
    return max(0.0, 1.0 - std)


def make_radar(summary: dict, out_path: str = RADAR) -> str:
    axes = ["Quality", "Cost-eff.", "Speed", "Auditability", "Consistency"]
    n_axes = len(axes)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    max_cost    = max(s["total_cost"]  for s in summary.values()) or 1.0
    max_latency = max(s["avg_latency"] for s in summary.values()) or 1.0

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    for name, s in summary.items():
        values = [
            s["avg_quality"],
            1.0 - _norm(s["total_cost"],  0, max_cost),
            1.0 - _norm(s["avg_latency"], 0, max_latency),
            1.0 if s["has_audit_log"] else 0.0,
            _consistency(s["per_task_quality"]),
        ]
        values += values[:1]
        color = STRAT_COLORS.get(name, "#888")
        label = STRAT_LABELS.get(name, name)
        ax.plot(angles,  values, linewidth=2,   color=color, label=label)
        ax.fill(angles,  values, alpha=0.15,    color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes, fontsize=11)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#888")
    ax.set_ylim(0, 1)
    ax.set_title("Routing Strategy — 5-Axis Comparison", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_scatter(summary: dict, out_path: str = RESULTS) -> str:
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, s in summary.items():
        color = STRAT_COLORS.get(name, "#888")
        ax.scatter(s["total_cost"], s["avg_quality"],
                   s=260, color=color, edgecolors="white", linewidths=2,
                   zorder=3, label=STRAT_LABELS.get(name, name))
        ax.annotate(STRAT_LABELS.get(name, name),
                    (s["total_cost"], s["avg_quality"]),
                    xytext=(8, 8), textcoords="offset points", fontsize=10)

    ax.set_xlabel("Total cost across benchmark (USD)")
    ax.set_ylabel("Average quality (0–1)")
    ax.set_title("Cost vs. Quality — lower-right is worst, upper-left is best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_heatmap(summary: dict, out_path: str = HEATMAP) -> str:
    tasks: list[str] = []
    for s in summary.values():
        for t in s["per_task_quality"].keys():
            if t not in tasks:
                tasks.append(t)
    strategy_names = list(summary.keys())

    data = np.zeros((len(strategy_names), len(tasks)))
    for i, sname in enumerate(strategy_names):
        per = summary[sname]["per_task_quality"]
        for j, t in enumerate(tasks):
            data[i, j] = per.get(t, 0.0)

    fig, ax = plt.subplots(figsize=(2 + 1.5 * len(tasks), 2 + 0.6 * len(strategy_names)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.set_yticks(range(len(strategy_names)))
    ax.set_yticklabels([STRAT_LABELS.get(n, n) for n in strategy_names])
    for i in range(len(strategy_names)):
        for j in range(len(tasks)):
            ax.text(j, i, f"{data[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax, label="quality")
    ax.set_title("Per-task quality by strategy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_summary_md(summary: dict, threshold: float, out_path: str = SUMMARY) -> str:
    rows: list[str] = []
    rows.append("# X25 Benchmark — Headline Numbers\n")
    rows.append(f"_Quality threshold for X25 escalation: **{threshold}**_\n")
    rows.append("")
    rows.append("| Strategy | Avg quality | Total cost (USD) | Avg latency (ms) | Audit log |")
    rows.append("|---|---:|---:|---:|:---:|")
    for name, s in summary.items():
        rows.append(
            f"| {STRAT_LABELS.get(name, name)} | "
            f"{s['avg_quality']:.3f} | "
            f"${s['total_cost']:.4f} | "
            f"{s['avg_latency']:.0f} | "
            f"{'yes' if s['has_audit_log'] else 'no'} |"
        )

    if "x25" in summary:
        x25 = summary["x25"]
        for name in ("always_frontier", "always_cheap"):
            if name in summary:
                base = summary[name]
                base_cost = max(base["total_cost"], 1e-9)
                cost_delta_pct = (x25["total_cost"] - base["total_cost"]) / base_cost * 100
                qual_delta_pts = (x25["avg_quality"] - base["avg_quality"]) * 100
                cost_word = "cheaper" if cost_delta_pct < 0 else "more expensive"
                qual_word = "higher quality" if qual_delta_pts >= 0 else "lower quality"
                rows.append("")
                rows.append(
                    f"**X25 vs {STRAT_LABELS.get(name, name)}** — "
                    f"{abs(cost_delta_pct):.1f}% {cost_word}, "
                    f"{abs(qual_delta_pts):.1f} pts {qual_word}"
                )

    rows.append("")
    rows.append("## Per-task quality")
    rows.append("")
    tasks: list[str] = []
    for s in summary.values():
        for t in s["per_task_quality"].keys():
            if t not in tasks:
                tasks.append(t)
    header = "| Strategy | " + " | ".join(tasks) + " |"
    align  = "|---|" + "---:|" * len(tasks)
    rows.append(header)
    rows.append(align)
    for name, s in summary.items():
        cells = [STRAT_LABELS.get(name, name)]
        for t in tasks:
            cells.append(f"{s['per_task_quality'].get(t, 0.0):.2f}")
        rows.append("| " + " | ".join(cells) + " |")

    text = "\n".join(rows) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return out_path


def make_all() -> dict[str, str]:
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"{IN_PATH} not found — run phase3_strategies.py first.")
    with open(IN_PATH, "r") as f:
        data = json.load(f)
    summary   = data["summary"]
    threshold = data.get("quality_threshold", 0.6)

    paths = {
        "radar":   make_radar(summary),
        "scatter": make_scatter(summary),
        "heatmap": make_heatmap(summary),
        "summary": make_summary_md(summary, threshold),
    }
    print("[phase4] wrote:")
    for k, v in paths.items():
        print(f"  {k:8s} {v}")
    return paths


if __name__ == "__main__":
    make_all()
