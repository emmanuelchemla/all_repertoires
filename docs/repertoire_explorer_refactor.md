# Repertoire Explorer Refactor

## Goal

The apes comparison and all-repertoires projects share the same core shape:

- load a database of species calls;
- compute or load pairwise similarity measures;
- run analyses over calls, species, families, and modalities;
- expose the database and analyses in an exploratory web app;
- export figures/tables for papers.

The useful abstraction is therefore not "apes" versus "all repertoires", but:

1. canonical call data;
2. similarity measures;
3. analysis payloads;
4. UI tabs and paper exports that consume those payloads.

## Current First Step

`src/repertoire_explorer/` now contains reusable primitives:

- `datasets.py`: loads `apes_comparison` CSV and `all_repertoires` JSON into one canonical call table.
- `similarity.py`: computes generic similarity matrices from TF-IDF text, keyword Jaccard, or precomputed call-pair similarities.
- `summaries.py`: computes dataset summaries and within/across species similarity summaries.

`analyses/repertoire_analysis.py` is a generic CLI entry point:

```bash
python analyses/repertoire_analysis.py --dataset apes
python analyses/repertoire_analysis.py --dataset all_repertoires --family Hominidae
```

It already accepts precomputed pairwise similarities with the columns:

- `call_id_1`
- `call_id_2`
- `similarity`

This is the hook for LLM-based judgments when they are available.

## Intended Unified App Shape

The app should be driven by a dataset selector:

- `apes_comparison`;
- `all_repertoires`;
- `all_repertoires` subset by family, order, class, or manually selected species.

Tabs should consume analysis payloads rather than project-specific globals:

- `Description`: call counts, taxonomy, keyword usage, within/across similarity distributions.
- `Explore repertoires`: compact per-species repertoire table and keyword distributions.
- `Similarity measures`: compare available acoustic/semantic measures.
- `One modality`: best-match coverage, similarity matrices, thresholded summaries.
- `Acoustic/Semantic correlations`: continuous and binary form-meaning relationships.
- `Keyword correspondences`: semantic-acoustic keyword cooccurrence when both keyword modalities exist.
- `Call correspondences`: thresholded cross-species correspondence table.
- `Family-level analyses`: only enabled when the dataset has multiple families.
- `Embedding/3D views`: UMAP/3D views from all_repertoires, enabled when embeddings are computable.
- `Exports`: LaTeX-ready tables and publication figures generated from the same payloads.

Tabs should declare their required capabilities. If a dataset lacks a capability, the tab should show a clear unavailable state instead of failing.

Examples:

- `apes_comparison` has semantic and acoustic keywords plus LLM matrices.
- `all_repertoires` has broad taxonomy and ontology keywords, but not necessarily acoustic keywords or LLM judgments.
- family-level comparison is meaningful for `all_repertoires`, but not for the four-ape paper subset.

## Next Refactor Steps

1. Move reusable permutation tests, best-match summaries, and correlation analyses out of `app/analysis.py`.
2. Make `app/analysis.py` a thin orchestrator over canonical datasets and analysis modules.
3. Add FastAPI endpoints per analysis tab instead of one large `run_analysis` endpoint.
4. Add the dataset/family selector to the webapp and route unavailable analyses through capability checks.
5. Move Plotly/HTML rendering into tab-specific JS modules or server-side payload endpoints.
6. Add export functions that turn analysis payloads into `.tex`, `.csv`, `.png`, and `.pdf` outputs.
