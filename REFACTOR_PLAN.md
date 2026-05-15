# Refactor: data-source-agnostic figure pipeline

## Context

The repo currently mixes two parallel datasets — the legacy "old" repertoire (`database.json` + `database_humans.json` at repo root, with semantic-only keywords) and the "new" repertoire (`repertoires/species/*.yaml` with separate acoustic + semantic keywords). Branch `codex/old-keyword-plot-folder` started the split: `paper_code/data_sources.py` already supports `load_calls("old"|"new")` and the paper switches figure folders via `\plotdir`. But only **4 of ~15 paper scripts** actually accept `--data-source`; the rest hard-code `database.json`, write to a shared `plots/` directory that gets overwritten, and some read stale legacy embeddings (`ac_emb.npy`/`se_emb.npy` without suffix). Goal: clean split of data locations, fully data-source-agnostic figure code, one driver script, audited for cross-source contamination.

## Target layout

```
old_repertoire_db/
    database.json           # moved from repo root
    database_humans.json    # moved from repo root
new_repertoire_db/
    species/*.yaml          # moved from repertoires/species/
    species_index.yaml      # moved from repertoires/
    schema.json
    validate.py
    explorer.py
    README.md, SKILL.md
paper_code/
    artifacts/              # cached embeddings + mantel json, dataset-keyed
        ac_emb_{old,new}.npy
        se_emb_{old,new}.npy
        mantel_results_{old,new}.json
        .embedding_cache.json     # safe to share (content-hashed)
    ...figure scripts...
    regenerate_figures.py   # NEW: centralized driver
paper/figures/
    old_database_keyword_inference/<fig>.{pdf,png}
    new_database_keyword_inference/<fig>.{pdf,png}
```

`paper/main.tex` already toggles via `\newcommand{\plotdir}{figures/<source>_database_keyword_inference}` ([main.tex:36](paper/main.tex:36)) — no further LaTeX work needed beyond pointing to the right folder.

## Changes

### 1. Move data files (no code change yet, just `git mv`)
- `database.json`, `database_humans.json` → `old_repertoire_db/`
- `repertoires/` → `new_repertoire_db/` (rename folder)
- Update `paper_code/data_sources.py` paths only (`ROOT/"old_repertoire_db/database.json"`, `ROOT/"new_repertoire_db/species/*.yaml"`).
- `main.py:31` still references `database.json` for the live app — leave the default arg, but pass `old_repertoire_db/database.json` from launcher / update default. Confirm with user (see Q1).

### 1b. main.py — update default

Change [main.py:31](main.py:31) default `json_path: Path | str = "database.json"` → `"old_repertoire_db/database.json"`. Verify the app launches.

### 2. Make every figure/analysis script `--data-source`-aware

For each script in the table below: add an `--data-source {old,new}` arg (default `old` for backward-compat), replace hardcoded `database.json` loads with `data_sources.load_calls(args.data_source)`, replace bare embedding loads (`np.load("ac_emb.npy")`) with `data_sources.load_embedding_artifacts(args.data_source, n_calls)`, and add an `--output-dir` defaulting to `paper/figures/{source}_database_keyword_inference/`.

| Script | Current state | Fix |
|---|---|---|
| [mantel.py](paper_code/mantel.py) | ✓ already done | — |
| [fig_embeddings.py](paper_code/fig_embeddings.py) | has `--data-source` but writes to `plots/` | route output to `paper/figures/<src>_..` |
| [fig_pmi_heatmap.py](paper_code/fig_pmi_heatmap.py) | same | same |
| [fig_supp_mantel_matrix.py](paper_code/fig_supp_mantel_matrix.py) | same | same |
| [fig_landmark_v2.py:687](paper_code/fig_landmark_v2.py) | **bug**: hardcoded `database.json` + tries `artifact_path("ac_emb","old",..)` with legacy fallback — mixes sources | full conversion |
| [fig_dataset_overview.py](paper_code/fig_dataset_overview.py) | hardcoded old | full conversion |
| [fig_landmark_calls.py](paper_code/fig_landmark_calls.py) | hardcoded old | full conversion |
| [fig_supp_cliques.py](paper_code/fig_supp_cliques.py) | hardcoded old | full conversion |
| [fig_supp_sensitivity.py](paper_code/fig_supp_sensitivity.py) | hardcoded old | full conversion |
| [fig_merged.py](paper_code/fig_merged.py) | hardcoded old + reads legacy `ac_emb.npy` | **delete** (per user) |
| [predict.py:35](paper_code/predict.py) | broken import `app.utils.load_calls` | switch to `data_sources` + add `--data-source` |
| [phylo_alignment.py](paper_code/phylo_alignment.py) | hardcoded old | full conversion |
| [fetch_timetree_divergence.py](paper_code/fetch_timetree_divergence.py) | hardcoded old | full conversion |
| [translate.py](paper_code/translate.py) | hardcoded old + legacy `embeddings.npz` | full conversion |
| [embed_and_analyze.py](paper_code/embed_and_analyze.py) | legacy pipeline, superseded by `mantel.py` | **delete** (per user) |
| [fig_human_bursts.py](paper_code/fig_human_bursts.py) | hardcoded to old + `database_humans.json` | leave old-only (human data only exists for old), but make embeddings path resolve via `data_sources` |

### 3. Delete legacy untagged artifacts

Currently in `paper_code/`: `ac_emb.npy`, `se_emb.npy`, `mantel_results.json`, `embeddings.npz`, `human_ac_emb.npy`, `human_se_emb.npy`. The first three are ambiguous (no source tag) and read by scripts as fallbacks — those fallbacks are the contamination risk. After step 2, remove the untagged files and the fallback branches. Keep `human_*` (human-only, unambiguous).

### 4. New centralized driver `paper_code/regenerate_figures.py`

```
python paper_code/regenerate_figures.py \
    --data-source {old,new} \
    [--recompute-keywords]      # re-run keyword inference (currently only relevant to new source)
    [--recompute-embeddings]    # delete ac_emb/se_emb/mantel_results for this source, force mantel.py recompute
    [--figures fig1,fig2,...]   # subset; default = all
```

Pipeline (sequential):
1. (optional) re-run keyword inference for the new source — needs clarification, see Q3.
2. Run `mantel.py --data-source <src>` (skips work if artifacts exist and `--recompute-embeddings` not set; the embedding cache `.embedding_cache.json` makes re-runs cheap).
3. Run each figure script with `--data-source <src> --output-dir paper/figures/<src>_database_keyword_inference/`.
4. Print a summary of what was (re)generated and which figures are tied to a specific source (e.g. `fig_human_*` only runs for `old`).

The driver is a thin subprocess/dispatch script — no business logic, so figure scripts remain individually runnable.

### 5. Audit checks added to the driver

- Assert `len(calls) == len(ac_emb) == len(se_emb)` (already in `load_embedding_artifacts`) but also assert the call-text hash matches what was cached (currently nothing prevents loading old embeddings against new calls if filenames are right but contents changed) — add a small `calls_hash_{old,new}.txt` written by `mantel.py`.
- Assert no script reads `ac_emb.npy` / `se_emb.npy` / `mantel_results.json` (untagged). A simple grep test in `regenerate_figures.py --check`.
- Confirm `.embedding_cache.json` is content-hashed (it is — keyed by text hash, so safe across sources).

## Critical files to modify

- [paper_code/data_sources.py](paper_code/data_sources.py) — update `ROOT` paths to new folders.
- All 14 figure/analysis scripts in [paper_code/](paper_code/) — add `--data-source`, redirect outputs.
- [paper_code/regenerate_figures.py](paper_code/regenerate_figures.py) — **new file**.
- [paper/main.tex:36](paper/main.tex:36) — confirm both folders work (already supports it).
- [main.py:31](main.py:31) — update default `json_path` to new location.
- `.gitignore` — keep `plots/` ignored; `paper/figures/` is committed (already is).

## Verification

1. `python paper_code/regenerate_figures.py --data-source old --recompute-embeddings` regenerates all of `paper/figures/old_database_keyword_inference/` from scratch. Diff PDFs against current committed versions — should be byte-identical or visually identical.
2. Same with `--data-source new`. Compare against current empty `paper/figures/new_database_keyword_inference/`.
3. `cd paper && latexmk -pdf main.tex` succeeds with both values of `\plotdir`.
4. Live app: `python main.py` still launches (data path update working).
5. Grep-test: `grep -rn 'ac_emb\.npy\|se_emb\.npy\|mantel_results\.json' paper_code/` returns no untagged hits.

## Resolved decisions

- `main.py` is updated to read from `old_repertoire_db/database.json`.
- `embed_and_analyze.py` and `fig_merged.py` are deleted.
- `--recompute-keywords`: user wants the same extraction process used for old-source `ontology_keywords` to be applied to the new source on demand.

## Outstanding gap: keyword extraction pipeline

**I could not locate a keyword-inference script in this repo.** `ontology_keywords` in `database.json` and `database_humans.json` appear to be statically authored — no LLM call, no extractor script in `paper_code/`, `app/`, or `analysis/` produces them. Likewise, `semantic_keywords`/`acoustic_keywords` in `new_repertoire_db/species/*.yaml` are already populated.

To wire `--recompute-keywords` to "whatever was used for the old data", I need to know where that extractor lives (likely external — another repo, a notebook, or a manual process). Two paths:

- **(a) If there is an external extractor**: I'll add a thin wrapper in `paper_code/extract_keywords.py` that the user points at the extractor (e.g. via env var / import), and `regenerate_figures.py --recompute-keywords` will (i) read each call's `acoustic_description` + `semantic_description`, (ii) run the extractor, (iii) overwrite the YAML/JSON `*_keywords` fields in place. The driver will then proceed to mantel + figures with the new keywords.
- **(b) If keyword authoring was manual**: `--recompute-keywords` is genuinely a no-op until such an extractor exists; the driver will print a clear message and exit if the flag is passed.

I'll proceed with the rest of the refactor assuming (b), and stub `--recompute-keywords` to raise `NotImplementedError("Point me to the extractor — see plan §Outstanding gap")` until the user clarifies. The refactor is otherwise independent.
