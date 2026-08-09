# All repertoires in the world

`repertoires/compiled/database.json` contains an LLM-generated list of species with their repertoires.
`repertoires/compiled/database_humans.json` contains the compiled human vocalization comparison database.

First prompt was:

```text
give me a json with all the calls of a species. The json should contain
- species name
- call name
- acoustic description
- semantic description (context of use)
- scientific references (as many as possible, ideally urls)
- subjective reliability
- comments
- users (only females, only young individuals, only some groups/populations, etc)

Start with chimps
```

Install dependencies:

```bash
uv sync
```

This project pins Python 3.11 for `uv` via `.python-version`.

Optional dependencies for the legacy Streamlit repertoire explorer:

```bash
uv sync --extra repertoires
```

Run the webapp:

```bash
uv run python analyses/build_animallex_release.py
uv run python -m webapp.main
```

The build command creates the shared AnimalLex analysis bundle used by both the
Dash app and the paper figure renderers. It uses 999 permutations for fast
iteration. Use `--full` for publication results with 9,999 permutations. Validate
an existing bundle with the same analysis settings by adding `--validate-only`
(and `--full` when validating a full bundle).

```bash
uv run python analyses/build_animallex_release.py --full
```

Regenerate the paper's static figures and LaTeX result macros from that same
bundle with:

```bash
uv run python paper_code/bundle_render.py
```

The paper renderer consumes the bundle's `include` species-fit view, matching
the web app's default species-fit setting, with all confidence levels retained.
The app can additionally show `include + caution` or all indexed species. Do not
edit `paper/generated_results.tex` or the files under
`paper/figures/generated/` by hand.

Run the generic analysis entry point:

```bash
uv run python analyses/repertoire_analysis.py --dataset all_repertoires
```

The legacy `requirements.txt` is kept as an export/compatibility file, but `pyproject.toml` is the canonical dependency list.

The dashboard is served at `http://127.0.0.1:8050` by default.

## Structure

- `webapp/`: Dash app, display helpers, static assets, app text, and data-loading wrappers for UI use.
- `analyses/`: runnable analysis entry points and analysis-specific helpers.
- `src/repertoire_explorer/`: reusable data/similarity/summary code shared by analyses and the webapp.
- `paper/` and `paper_code/`: paper assets and figure/table scripts.
- `repertoires/`: source repertoire material.
