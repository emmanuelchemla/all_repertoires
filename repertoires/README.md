The repertoire databases are organized as follows:

- Literature-extraction DB: `llm_literature_extraction/species/<scientific-name-kebab-case>.yaml` (e.g. `pan-paniscus.yaml`).
- LLM-knowledge-plus-search DB: `llm_knowledge+search/species/<scientific-name-kebab-case>.yaml`.
- `species_index.yaml` lists candidate species grouped by family. A species is "done" iff its file exists in `llm_literature_extraction/species/`.
- LLM agents grow the literature-extraction database using `.agents/skills/llm-literature-extraction-repertoire/SKILL.md`, which defines the curation rules.
- The schema and validator live with that skill. Before committing, an agent always validates with `python .agents/skills/llm-literature-extraction-repertoire/validate.py repertoires/llm_literature_extraction/species/<file>.yaml` from the repository root.
- We additionally have an `explorer.py` interactive web app which allows us to explore the database.
- To add new species, prompt `Codex GPT-5.5 Extra High` (or `Claude Code Opus 4.7 Extra High`) with something like:

```
Follow `.agents/skills/llm-literature-extraction-repertoire/SKILL.md` to add 10 new species.
```
