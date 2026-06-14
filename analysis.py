"""Load a run's JSON output and generate a matplotlib chart."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def plot_run(run_dir: str) -> None:
    p = Path(run_dir)
    with open(p / "run.json", encoding="utf-8") as f:
        data = json.load(f)

    rounds_data = [e for e in data["events"] if e["type"] == "round"]
    if not rounds_data:
        print("No round data to plot.")
        return

    scenario = data["scenario"]
    outcome = data.get("outcome", "unknown")
    state_keys = list(rounds_data[0]["shared_state"].keys())
    x = [r["round"] for r in rounds_data]

    n_plots = len(state_keys) + 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3.5 * n_plots))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(f"{scenario} — outcome: {outcome}", fontsize=13, fontweight="bold")

    colors = ["#1d3a5f", "#b91c1c", "#166534", "#92600a", "#5b21b6"]
    for i, key in enumerate(state_keys):
        y = [r["shared_state"][key] for r in rounds_data]
        axes[i].plot(x, y, color=colors[i % len(colors)], linewidth=2, marker="o", markersize=5)
        axes[i].set_title(key, fontsize=10)
        axes[i].set_xlabel("Round")
        axes[i].set_ylabel(key)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xticks(x)

    # Choice distribution
    choice_counts: dict[str, int] = {}
    for r in rounds_data:
        for choice in r["choices"].values():
            choice_counts[choice] = choice_counts.get(choice, 0) + 1

    ax = axes[-1]
    ax.bar(choice_counts.keys(), choice_counts.values(), color="#1d3a5f", alpha=0.8)
    ax.set_title("Choice distribution — all agents, all rounds", fontsize=10)
    ax.set_ylabel("Count")

    plt.tight_layout()
    out = p / "plot.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Plot saved: {out}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analysis.py <run_output_dir>")
        sys.exit(1)
    plot_run(sys.argv[1])
