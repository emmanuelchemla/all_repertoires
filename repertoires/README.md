The repertoire database is organized as follows:

- One YAML file per species: `species/<scientific-name-kebab-case>.yaml` (e.g. `pan-paniscus.yaml`).
- `species_index.yaml` lists candidate species grouped by family. A species is "done" iff its file exists in `species/`.
- LLM agents grow the database using `SKILL.md`, which defines the curation rules.
- The schema lives in `schema.json`. Before committing, an agent always validates with `python validate.py species/<file>.yaml`.
- We additionally have an `explorer.py` interactive web app which allows us to explore the database.
- To add new species, prompt `Codex GPT-5.5 Extra High` (or `Claude Code Opus 4.7 Extra High`) with something like:

```
Invoke the species-repertoire skill to add 10 new species, one subagent per species, on a new branch with one commit per species and a single PR. Completely ignore database.json for this request.
```

