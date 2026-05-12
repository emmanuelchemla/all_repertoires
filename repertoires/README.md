The repertoire database is organized as follows:

- One YAML file per species: `species/<scientific-name-kebab-case>.yaml` (e.g. `pan-paniscus.yaml`).
- LLM agents grow the database using `SKILL.md`, which defines the curation rules.
- The schema lives in `schema.json`. Before committing, an agent always validates with `python validate.py species/<file>.yaml`.
- We additionally have a `explorer.py` interactive web app which allows us to explore the database.
