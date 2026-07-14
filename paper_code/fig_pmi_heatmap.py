"""Render the acoustic and semantic PMI heatmap from the shared bundle."""

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


def render(bundle_path: Path | str | None = None, output: Path | str | None = None) -> Path:
    bundle = load_analysis_bundle(bundle_path) if bundle_path else load_analysis_bundle()
    result = bundle.analysis["pmi"]
    matrix = np.asarray(result["matrix"], dtype=float)
    limit = max(float(np.nanmax(np.abs(matrix))), 1.0)
    fig, ax = plt.subplots(figsize=(16, 8))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(result["semantic_keywords"])))
    ax.set_yticks(range(len(result["acoustic_keywords"])))
    ax.set_xticklabels(
        [value.replace("_", " ") for value in result["semantic_keywords"]],
        rotation=55,
        ha="right",
        fontsize=7,
    )
    ax.set_yticklabels(
        [value.replace("_", " ") for value in result["acoustic_keywords"]],
        fontsize=7,
    )
    ax.set_title("Acoustic and semantic keyword association", fontweight="bold")
    fig.colorbar(image, ax=ax, label="PMI (bits)", fraction=0.025, pad=0.02)
    fig.tight_layout()
    output_path = Path(output) if output else ROOT / "plots" / "pmi_heatmap_paper.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    print(f"Saved {render()}")


if __name__ == "__main__":
    main()
