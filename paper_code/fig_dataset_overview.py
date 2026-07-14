"""Render paper Figure 1 from the shared AnimalLex analysis bundle."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from paper_code.bundle_render import load_analysis_bundle, ROOT
except ModuleNotFoundError:
    from bundle_render import load_analysis_bundle, ROOT


PALETTE = [
    "#15616d",
    "#ff7d00",
    "#6a4c93",
    "#2a9d8f",
    "#bc4749",
    "#577590",
    "#8c6d31",
    "#e56b6f",
    "#3a86a8",
    "#7f8f3a",
    "#9c6644",
    "#5f6caf",
    "#6b705c",
]


def _keyword_panel(ax, rows, title, limit=16):
    rows = rows[:limit][::-1]
    groups = list(dict.fromkeys(row["group"] for row in rows))
    colors = {group: PALETTE[index % len(PALETTE)] for index, group in enumerate(groups)}
    ax.barh(
        [row["keyword"].replace("_", " ") for row in rows],
        [row["percent_calls"] for row in rows],
        color=[colors[row["group"]] for row in rows],
    )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Calls with keyword (%)")
    ax.legend(
        handles=[Patch(facecolor=colors[group], label=group.title()) for group in groups],
        title="Keyword group",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        frameon=False,
        fontsize=7,
        title_fontsize=7,
    )
    ax.spines[["top", "right"]].set_visible(False)


def render(bundle_path: Path | str | None = None, output: Path | str | None = None) -> Path:
    bundle = load_analysis_bundle(bundle_path) if bundle_path else load_analysis_bundle()
    overview = bundle.analysis["overview"]
    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(2, 2, hspace=0.72, wspace=0.35)
    _keyword_panel(fig.add_subplot(grid[0, 0]), overview["semantic_keywords"], "(A) Semantic functions")
    _keyword_panel(fig.add_subplot(grid[0, 1]), overview["acoustic_keywords"], "(B) Acoustic features")
    ax = fig.add_subplot(grid[1, :])
    rows = overview["species_counts"]
    groups = list(dict.fromkeys(row["taxonomic_group"] for row in rows))
    colors = {group: PALETTE[index % len(PALETTE)] for index, group in enumerate(groups)}
    ax.bar(
        range(len(rows)),
        [row["n_calls"] for row in rows],
        color=[colors[row["taxonomic_group"]] for row in rows],
    )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(rows[index]["common_name"] for index in range(len(rows)))
    ax.tick_params(axis="x", rotation=90, labelsize=5.5)
    ax.set_ylabel("Calls in database")
    ax.set_title("(C) Calls per species", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[Patch(facecolor=colors[group], label=group) for group in groups],
        title="Taxonomic group",
        loc="upper left",
        bbox_to_anchor=(1.005, 1),
        frameon=False,
        fontsize=7,
        title_fontsize=7,
    )
    start = 0
    while start < len(rows):
        end = start + 1
        while end < len(rows) and rows[end]["taxonomic_group"] == rows[start]["taxonomic_group"]:
            end += 1
        if start:
            ax.axvline(start - 0.5, color="#c9d2d4", linewidth=0.8)
        start = end
    output_path = Path(output) if output else ROOT / "plots" / "fig1_dataset_overview.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    print(f"Saved {render()}")


if __name__ == "__main__":
    main()
