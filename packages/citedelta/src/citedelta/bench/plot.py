"""The ANN recall-vs-QPS plot."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display on a VPS; must precede pyplot
import matplotlib.pyplot as plt

MARKERS = {
    "brute-force": "*",
    "ivf-flat": "o",
    "hnsw": "s",
    "pgvector-hnsw": "^",
    "pgvector-ivf": "v",
}


def plot_results(result_files: list[Path], out: Path, *, title: str = "") -> None:
    """Recall on x, QPS on y (log). The convention ANN-Benchmarks uses.

    Log-scaled QPS because the interesting differences are multiplicative:
    'four times the throughput at the same recall' is the claim, and a linear
    axis flattens exactly the region where the indexes separate.

    Up and to the right is better. A point that is below AND left of another
    is dominated — strictly worse on both axes, no tradeoff to argue about.
    """
    datasets: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in result_files:
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for row in payload if isinstance(payload, list) else []:
            if "dataset" not in row or "index" not in row:
                continue
            datasets[row["dataset"]][row["index"]].append((row["recall"], row["qps"]))

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for ax, (dataset, by_index) in zip(axes[0], sorted(datasets.items()), strict=True):
        for index_name, points in sorted(by_index.items()):
            points.sort()
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(
                xs,
                ys,
                marker=MARKERS.get(index_name, "^"),
                label=index_name,
                linewidth=1.6,
                markersize=7,
            )

        ax.set_yscale("log")
        ax.set_xlabel("recall@10 (tie-aware, vs. brute-force oracle)")
        ax.set_ylabel("queries/sec (single-threaded)")
        ax.set_title(dataset)
        ax.grid(visible=True, which="both", alpha=0.25)
        ax.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_selectivity_sweep(points: list[dict[str, Any]], out: Path, *, index: str = "hnsw") -> None:
    """Two panels: recall and throughput, both against selectivity.

    Selectivity on a LOG x-axis and descending left-to-right, so "the filter
    gets tighter" reads rightward, matching how the story is told. The
    interesting region is 0.005-0.05, which a linear axis would compress into
    the left margin.
    """
    rows = [p for p in points if p["index"] == index and p["filter_kind"] == "synthetic"]
    temporal = [p for p in points if p["index"] == index and p["filter_kind"] == "temporal"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    styles = {
        "post-filter": ("o", "tab:red"),
        "post-filter+overfetch": ("s", "tab:orange"),
        "in-index": ("D", "tab:green"),
    }

    for strategy, (marker, colour) in styles.items():
        series = sorted(
            (p for p in rows if p["strategy"] == strategy),
            key=lambda p: -float(p["selectivity"]),
        )
        if not series:
            continue
        xs = [p["selectivity"] for p in series]
        axes[0].plot(xs, [p["recall"] for p in series], marker=marker, color=colour, label=strategy)
        axes[1].plot(xs, [p["qps"] for p in series], marker=marker, color=colour, label=strategy)

        for point in temporal:
            if point["strategy"] == strategy:
                axes[0].scatter(
                    [point["selectivity"]],
                    [point["recall"]],
                    marker="*",
                    s=260,
                    color=colour,
                    zorder=5,
                    edgecolors="black",
                    linewidths=0.6,
                )

    for ax, ylabel, title in (
        (axes[0], "recall@10 vs filtered oracle", f"{index}: accuracy under a filter"),
        (axes[1], "queries/sec", f"{index}: throughput under a filter"),
    ):
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("selectivity (fraction of corpus admissible) — tighter to the right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(visible=True, which="both", alpha=0.25)
        ax.legend()
    axes[1].set_yscale("log")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].axhline(0.95, linestyle=":", color="grey", linewidth=1)

    fig.suptitle("★ = the real temporal filter · dotted line = 0.95 recall target")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
