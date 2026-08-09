"""The ANN recall-vs-QPS plot."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a VPS; must precede pyplot
import matplotlib.pyplot as plt

MARKERS = {"brute-force": "*", "ivf-flat": "o", "hnsw": "s"}


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
        for row in json.loads(path.read_text()):
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
