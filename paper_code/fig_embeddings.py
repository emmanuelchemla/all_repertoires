"""Render the Figure 2 form to meaning panel from the shared bundle."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from paper_code.bundle_render import load_analysis_bundle, ROOT
except ModuleNotFoundError:
    from bundle_render import load_analysis_bundle, ROOT


COLORS = {
    "within species": "#2a9d8f",
    "same family": "#ff7d00",
    "different families": "#6a4c93",
}


def draw_mantel_result(ax, result):
    for group, values in result["groups"].items():
        if not values["sample"]:
            continue
        x = np.array([row["acoustic_distance"] for row in values["sample"]])
        y = np.array([row["semantic_distance"] for row in values["sample"]])
        p_text = "p < 0.001" if values["p"] < 0.001 else f"p = {values['p']:.3f}"
        ax.scatter(x, y, color=COLORS[group], s=5, alpha=0.28, linewidths=0)
        ax.plot(
            [0, 1],
            [values["intercept"], values["intercept"] + values["slope"]],
            color=COLORS[group],
            linewidth=1.8,
            label=f"{group} (r={values['r']:.2f}, {p_text}, n={values['n_pairs']:,})",
        )
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Acoustic distance", ylabel="Semantic distance")
    ax.set_title("(C) Form to meaning correlation", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, loc="upper left")


def render(bundle_path: Path | str | None = None, output: Path | str | None = None) -> Path:
    bundle = load_analysis_bundle(bundle_path) if bundle_path else load_analysis_bundle()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    draw_mantel_result(ax, bundle.analysis["form_meaning"])
    fig.tight_layout()
    output_path = Path(output) if output else ROOT / "plots" / "fig2C_mantel.pdf"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    print(f"Saved {render()}")


if __name__ == "__main__":
    main()
