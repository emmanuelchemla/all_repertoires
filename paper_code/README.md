# Paper pipeline

All scripts in this directory consume `database.json` at the repository
root, which is **generated from the curated YAML repertoires** under
`repertoires/species/`. Do not hand-edit `database.json`.

## Refresh everything after editing repertoires

```sh
python paper_code/build_database.py        # YAMLs  → database.json + ac_emb.npy / se_emb.npy
python paper_code/mantel.py                # Mantel tests           → mantel_results.json
python paper_code/predict.py               # Prediction table       → prediction_results.json
python paper_code/fig_dataset_overview.py  # Fig. 1 (dataset overview)
python paper_code/fig_embeddings.py        # Fig. 2 (UMAPs + Mantel scatter)
python paper_code/fig_pmi_heatmap.py       # Fig. 3 (PMI heatmap)
python paper_code/fig_landmark_v2.py       # Fig. 4 (landmark cliques)
python paper_code/fig_landmark_calls.py    # Fig. 5 (landmark rosettes)
python paper_code/fig_supp_mantel_matrix.py
python paper_code/fig_supp_sensitivity.py
python paper_code/fig_supp_cliques.py      # also writes supp_cliques_table.tex
python paper_code/fig_human_bursts.py      # Human-bursts supplementary figures
```

Outputs land in `../plots/` (figures) and in this directory (JSON / TeX
tables consumed by `paper/main.tex`). Re-compile `paper/main.tex` to
pick up the refreshed figures and tables.

## Layout

- `build_database.py` — converts `repertoires/species/*.yaml` into the
  legacy `database.json` schema and refreshes the sentence-transformer
  embedding caches (`ac_emb.npy`, `se_emb.npy`) so every figure script
  sees a consistent set of calls.
- `embed_and_analyze.py` — older standalone pipeline kept for backwards
  compatibility; `mantel.py` / `predict.py` are the up-to-date entry
  points used by the paper.
- `fig_*.py` — one Python script per (main or supplementary) figure.
- `results.json`, `mantel_results.json`, `prediction_results.json`,
  `supp_cliques_table.tex` — generated artefacts read back by
  `paper/main.tex`.

## Adding new species

1. Add a new YAML under `repertoires/species/` (see
   `repertoires/SKILL.md` and `repertoires/schema.json`).
2. Run `python repertoires/validate.py repertoires/species/<file>.yaml`.
3. Run the commands above to regenerate `database.json`, embeddings,
   figures, and tables.
4. Update the head-counts in `paper/main.tex` if needed (`\nspecies`,
   `\ncalls`, taxonomic breakdown).
