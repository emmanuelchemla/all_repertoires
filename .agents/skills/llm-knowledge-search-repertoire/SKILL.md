---
name: llm-knowledge-search-repertoire
description: "Use this skill when creating, editing, or reviewing exploratory species vocal repertoire YAML files for the LLM knowledge + online search database under repertoires/llm_knowledge+search/species/. This workflow is faster and looser than the literature-extraction repertoire skill: references are encouraged but optional, claims may be synthesized from model knowledge plus web search, and every call must honestly mark confidence."
---

# LLM knowledge + search repertoire skill

The goal of this skill is to create a wild vocal repertoire for a species using LLM knowledge plus online search.

## File layout

- One YAML file per species at `repertoires/llm_knowledge+search/species/<scientific-name-kebab-case>.yaml`.
- `repertoires/species_index.yaml` is the shared candidate queue. A species is unfinished for this database iff it appears in that index and there is no matching YAML file under `repertoires/llm_knowledge+search/species/`.
- One file = one species = atomic unit of work.
- Schema lives with this skill at `.agents/skills/llm-knowledge-search-repertoire/schema.json`.
- Validator lives with this skill at `.agents/skills/llm-knowledge-search-repertoire/validate.py`.
- Validate with:

```bash
python .agents/skills/llm-knowledge-search-repertoire/validate.py repertoires/llm_knowledge+search/species/<file>.yaml
```

## Source policy

- Use online search when it can improve accuracy, especially for uncertain species, call names, newer literature, taxonomy, or audio samples.
- References are encouraged, but not required. It is acceptable to include a call based on LLM knowledge and search synthesis without a stable citation.
- Never overstate evidence. Every call must include `confidence`: `low`, `medium`, or `high`.

## Top-level fields

- `scientific_name`: binomial.
- `common_name`: widely used English common name.
- `taxonomy`: object with `kingdom`, `phylum`, `class`, `order`, `family`, `genus`, and `species`. `species` is the epithet only.
- `calls`: list of call-type objects.
- `references`: optional bibliography map. Use stable IDs in `lastname_year` or `lastname_etal_year` form when available.
- `provenance`: append-only list of generation/update events.

## Per-call fields

- `name`: concise canonical call name.
- `alternative_names`: optional list of synonyms or variant labels.
- `scope`: optional but preferred object:
  - `life_stages`: use `infant`, `juvenile`, `adult`, or `unknown`.
  - `sexes`: use `female`, `male`, or `unknown`.
  - `population_specific`: boolean.
  - `note`: short note; required when `population_specific` is true.
- `acoustic_keywords`: controlled acoustic tags. Use only:
  - `high_frequency`, `low_frequency`, `frequency_modulated`
  - `tonal`, `broadband`, `noisy`, `harmonic`
  - `short`, `long`, `abrupt`, `repetitive`, `pulsed`, `multi_component`
  - `loud`, `quiet`, `graded`
- `acoustic_description`: what the call sounds like, describe spectro-temporal structure such as pitch, tonality/noisiness, duration, contour, rhythm/repetition, component structure, amplitude when relevant. Avoid function/context here.
- `semantic_keywords`: controlled function/context tags. Use only:
  - `contact`, `group_coordination`, `affiliation`
  - `threat`, `aggression`, `submission`
  - `alarm`, `predator`
  - `distress`, `begging`, `caregiving`
  - `courtship`, `mating`
  - `food`, `recruitment`
  - `territorial`, `spacing`
  - `identity`, `attention`
  - `play`, `display`
  - `combinatorial`
- `semantic_description`: when or why the call is produced, context of production. Avoid acoustic terms here.
- `confidence`: `low`, `medium`, or `high`.
- `references`: optional list of reference IDs from the top-level `references` map.

## Inclusion guidance

- Aim for a useful comprehensive and complete repertoire, but not exhaustive literature reconciliation.
- Include common, named call types and well-known variants when they are useful for comparative analysis.
- Merge obvious synonyms into `alternative_names`.
- Keep broad graded categories broad unless a split is widely recognized or helpful.
- If a call boundary is uncertain, include it with `confidence: low` or `medium` and explain the uncertainty.

## Workflow

1. Pick species from `repertoires/species_index.yaml` for which `repertoires/llm_knowledge+search/species/<scientific-name-kebab-case>.yaml` does not exist yet.
2. Search the web for current taxonomy and obvious repertoire/call-type sources when useful.
3. Draft a lean YAML file with broad call coverage and honest evidence labels.
4. Validate with `.agents/skills/llm-knowledge-search-repertoire/validate.py`.
5. Fix validation errors, then do a quick readability pass for acoustic/semantic separation.