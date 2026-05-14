# `analysis/` — main-paper results from `repertoires/`

Single-entry pipeline that turns the curated `repertoires/species/*.yaml`
files into the figures, tables, and macros used by `paper/main.tex`.

```bash
# Full run (≈1–2 min; Mantel permutations dominate)
python -m analysis.run

# Quick run, skip Mantel permutation p-values
python -m analysis.run --skip-perm
```

Outputs:

| File                                   | What                                                |
| -------------------------------------- | --------------------------------------------------- |
| `plots/fig1_dataset_overview.png`      | Semantic & acoustic keyword frequency; calls/species |
| `plots/fig2A_semantic_umap.pdf`        | UMAP of semantic-description embeddings              |
| `plots/fig2B_acoustic_umap.pdf`        | UMAP of acoustic-description embeddings              |
| `plots/fig2C_mantel.pdf`               | Mantel scatter with within / same-family / cross-family colouring |
| `plots/fig_supp_mantel_matrix.pdf`     | Species-pairwise acoustic–semantic correlation       |
| `plots/pmi_heatmap_paper.{png,pdf}`    | PMI between acoustic & semantic keywords             |
| `plots/fig4_landmark_h.pdf`            | Mutual top-1 acoustic & semantic nearest-neighbour groups |
| `paper/paper_macros.tex`               | `\nspecies`, `\ncalls`, `\nmammals`, … auto-updated  |
| `analysis/results.json`                | All numbers (Mantel r/p, prediction Table 1, counts) |

The pipeline reads YAML directly and uses each call's controlled
`acoustic_keywords` / `semantic_keywords` for PMI — no free-text keyword
matching, no dependency on `database.json`.

To refresh the paper after adding new species:

```bash
python -m analysis.run
cd paper && xelatex main && biber main && xelatex main && xelatex main
```

`paper/main.tex` `\InputIfFileExists{paper_macros.tex}` so headline counts
update on the next compile without manual edits.
