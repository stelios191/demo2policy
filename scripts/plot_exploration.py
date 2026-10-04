"""Plot exploratory results: success and aiming error against training length, and loss curves.

Reads whatever has been recorded so far under artifacts/2_diagnosis/ and runs/, so it can
be run while experiments are still in progress.

    python -m scripts.plot_exploration
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.common import ARTIFACTS, DIAGNOSIS, OUT, relative_path
from scripts.report import GRID, SERIES, SURFACE, TEXT, TEXT_SECONDARY, extract_training_curves, style_axes

EXPLORATION = ARTIFACTS / DIAGNOSIS
# fixed order, so each experiment keeps its colour
EXPERIMENTS = [
    ("explore_mlp256_200k", "256-unit network"),
    ("explore_mlp1024_200k", "1024-unit network"),
    ("explore_gmm1024_200k", "1024-unit network, GMM output"),
]
TICKS = [10000, 25000, 50000, 100000, 200000]


def recorded_points(experiment):
    """Evaluated checkpoints of one experiment as (update, successes, completed, aiming error)."""
    points = []
    for results in sorted((EXPLORATION / experiment).glob("update_*/results.json")):
        summary_path = EXPLORATION / experiment / "summary.json"
        data = json.loads(results.read_text(encoding="utf-8"))
        offsets = {}
        if summary_path.exists():
            offsets = {
                row["update"]: row["median_offset_mm"] for row in json.loads(summary_path.read_text(encoding="utf-8"))
            }
        update = int(results.parent.name.split("_")[1])
        points.append((update, data["summary"]["successes"], data["summary"]["completed"], offsets.get(update)))
    return sorted(points)


def spread_end_labels(series, ylim, gap=0.055):
    """Positions for the end-of-line value labels, nudged apart where two lines end close together."""
    span = (ylim[1] - ylim[0]) if ylim else 1.0
    ends = sorted((ys[-1], xs[-1]) for _, xs, ys in series)
    placed = []
    for y, x in ends:
        label_y = y if not placed else max(y, placed[-1][2] + gap * span)
        placed.append((x, y, label_y))
    return placed


def line_chart(series, ylabel, title, subtitle, path, ylim=None, reference=None):
    """Line chart against training length on a log axis, with each series labelled at its end."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    style_axes(ax)
    for (label, xs, ys), colour in zip(series, SERIES, strict=False):
        ax.plot(xs, ys, color=colour, linewidth=2, label=label, solid_capstyle="round")
        ax.plot(xs, ys, "o", color=colour, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    for x, y, label_y in spread_end_labels(series, ylim):
        ax.annotate(
            f"{y:g}", (x, label_y), xytext=(8, 0), textcoords="offset points", va="center", color=TEXT, fontsize=9
        )
    if reference:
        value, text = reference
        ax.axhline(value, color=TEXT_SECONDARY, linewidth=1)
        ax.annotate(
            text, (TICKS[0], value), xytext=(0, 5), textcoords="offset points", color=TEXT_SECONDARY, fontsize=9
        )
    ax.set_xscale("log")
    ax.set_xticks(TICKS, [f"{t:,}" for t in TICKS])
    ax.minorticks_off()
    ax.set_xlim(TICKS[0] * 0.85, TICKS[-1] * 1.35)
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_xlabel("optimizer updates (log scale)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11, pad=22)
    ax.text(0, 1.04, subtitle, transform=ax.transAxes, color=TEXT_SECONDARY, fontsize=9)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Plot whatever exploratory results have been recorded so far."""
    out = EXPLORATION / "figures"
    out.mkdir(parents=True, exist_ok=True)
    success, offset, status = [], [], []
    for experiment, label in EXPERIMENTS:
        points = recorded_points(experiment)
        status.append(f"{label}: {len(points)} of {len(TICKS)} checkpoints evaluated")
        if not points:
            continue
        success.append((label, [p[0] for p in points], [p[1] for p in points]))
        with_offset = [p for p in points if p[3] is not None]
        if with_offset:
            offset.append((label, [p[0] for p in with_offset], [p[3] for p in with_offset]))

    subtitle = "100 exploration starting positions, one training run each (diagnostic, not a final result)"
    line_chart(
        success,
        "cube lifted, out of 100 trials",
        "Task success against training length",
        subtitle,
        out / "success_vs_updates.png",
        ylim=(75, 101),
    )
    if offset:
        line_chart(
            offset,
            "sideways distance from cube when closing (mm)",
            "Aiming error against training length",
            subtitle,
            out / "aiming_error_vs_updates.png",
            ylim=(0, 15),
            reference=(6.0, "human demonstrations: 6 mm"),
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    style_axes(ax)
    for (experiment, label), colour in zip(EXPERIMENTS[:2], SERIES, strict=False):
        logs = sorted((OUT / "runs" / experiment).glob("*/logs/log.txt"))
        if not logs:
            continue
        curves = extract_training_curves(logs[-1], 100)
        ax.plot(curves["updates"], curves["train_loss"], color=colour, linewidth=2, label=f"{label}: training")
        ax.plot(
            curves["updates"],
            curves["validation_loss_sampled"],
            color=colour,
            linewidth=1,
            alpha=0.55,
            label=f"{label}: validation (sampled)",
        )
    ax.set_yscale("log")
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_xlabel("optimizer updates", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("mean squared action error (log scale)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title("Training and validation loss", loc="left", color=TEXT, fontsize=11, pad=22)
    ax.text(
        0,
        1.04,
        "thick line: training episodes; thin line: validation episodes",
        transform=ax.transAxes,
        color=TEXT_SECONDARY,
        fontsize=9,
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    print("\n".join(status))
    print(f"figures written to {relative_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
