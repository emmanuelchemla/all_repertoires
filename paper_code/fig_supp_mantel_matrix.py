"""Render the species pair correlation matrix from the shared bundle."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repertoire_explorer.animallex_analysis import compute_pairwise_r as shared_compute_pairwise_r

try:
    from paper_code.bundle_render import load_analysis_bundle
except ModuleNotFoundError:
    from bundle_render import load_analysis_bundle


MIN_PAIRS = 6


def compute_pairwise_r(calls, Sa, Ss):
    """Compatibility wrapper for older paper code."""
    return shared_compute_pairwise_r(calls, Sa, Ss, minimum_pairs=MIN_PAIRS)


def render(bundle_path: Path | str | None = None, output: Path | str | None = None) -> Path:
    bundle = load_analysis_bundle(bundle_path) if bundle_path else load_analysis_bundle()
    result = bundle.analysis["species_matrix"]
    common_names = {
        row["species"]: row.get("common_name") or row["species"]
        for row in bundle.analysis["overview"]["species_counts"]
    }
    labels = [common_names.get(species, species) for species in result["species"]]
    matrix = np.asarray(result["matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(15, 13))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(result["species"])))
    ax.set_yticks(range(len(result["species"])))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_title("Species pair acoustic and semantic correlation", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Pearson r", fraction=0.035, pad=0.02)
    fig.tight_layout()
    output_path = Path(output) if output else ROOT / "plots" / "fig_supp_mantel_matrix.pdf"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    print(f"Saved {render()}")


if __name__ == "__main__":
    main()
