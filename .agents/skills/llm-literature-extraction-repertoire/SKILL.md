---
name: llm-literature-extraction-repertoire
description: Use this skill whenever creating, editing, or reviewing a literature-extraction species vocal repertoire YAML file under repertoires/llm_literature_extraction/species/. This workflow is based on LLM extraction from primary literature, requires cited acoustic and semantic evidence, and provides the schema, conventions, agreement rubrics, and validation steps that keep multi-agent edits consistent.
---

# LLM literature extraction repertoire skill

The goal of this skill is to create a file with a comprehensive wild vocal repertoire for a species.

## File layout

- One YAML file per species at `repertoires/llm_literature_extraction/species/<scientific-name-kebab-case>.yaml` (e.g. `repertoires/llm_literature_extraction/species/corvus-corax.yaml`).
- One file = one species = atomic unit of parallel work. Never split a species across files.
- Schema lives with this skill at `.agents/skills/llm-literature-extraction-repertoire/schema.json`.
- Validator lives with this skill at `.agents/skills/llm-literature-extraction-repertoire/validate.py`. Run `python .agents/skills/llm-literature-extraction-repertoire/validate.py repertoires/llm_literature_extraction/species/<file>.yaml` before committing.

## Field conventions

### Top-level

- `scientific_name`: binomial, italicized intent only (plain text in YAML).
- `common_name`: most widely used English common name.
- `taxonomy`: required object with `kingdom`, `phylum`, `class`, `order`, `family`, `genus`, and `species`. Use canonical Latin taxon names; `species` is the species epithet only, so `genus` + `species` should match `scientific_name`.
- `primary_inventory`: the single paper used as the spine for the call list.
  - Pick the most comprehensive paper that explicitly enumerates the species' vocal repertoire and is widely cited as such. Set `id` to its reference ID; explain the pick in `rationale`.
  - If no such paper exists, omit `id` and use `rationale` to explain why. Every call must then be marked `in_primary_inventory: false`.
- `calls`: list of call-type objects (see below). See inclusion rules below.
- `provenance`: append-only list of agent events. Add one entry each time you generate or materially update the file:
  - `timestamp`: ISO 8601 UTC, e.g. `2026-05-14T10:23:00Z`.
  - `model`: full model ID, e.g. `claude-opus-4-7`. If you are not able to introspect `model` and `thinking` values with absolute certainty, and the user hasn't reported it, ask the user directly. Do not guess this.
  - `thinking`: one of `low` | `medium` | `high` | `extra_high` | `max`.
  - `action`: `generated` (first entry only) or `updated`.
  Never remove or modify existing entries.
- `references`: bibliography for the whole file. Cite with **stable string IDs** in `lastname_year` form (`smith_2019`, `marler_etal_1967`).

### Inclusion rules

- **From the primary inventory**: include every call the inventory paper lists.
- **Outside the inventory**: include a call if ≥1 peer-reviewed source describes it as a *distinct* call type in this species, with enough detail to fill both descriptions and cite ≥1 acoustic + ≥1 semantic reference. Err on the side of inclusion: a `weak` call with an honest explanation is more useful than silent omission.
- **Renamings** of an inventory call go into `alternative_names` on the existing entry, not as a new call.
- **Subtypes/variants** of an existing call belong in that call's agreement explanations, not as new entries.
- **Splits** (one inventory call separated into two by a credible source) are added as new calls; explain the split in agreement explanations of both.
- Do not include calls known only from captivity when there is no wild evidence.

### Per call type

- `name`: canonical name used in the literature. Prefer the most common term.
- `alternative_names`: list of synonyms encountered in other papers.
- `in_primary_inventory`: `true` if the call appears in the species' primary inventory paper; `false` for additions sourced from other literature (or for every call when no inventory exists).
- `scope`: object describing which animals and populations the call type is supported for:
  - `life_stages`: list using only `infant`, `juvenile`, `adult`, or `unknown`.
  - `sexes`: list using only `female`, `male`, or `unknown`.
  - `population_specific`: `true` when the call is known to be population-specific or absent from some studied populations; otherwise `false`.
  - `note`: if `population_specific` is `true`, short note explaining why and citing references. Otherwise, leave it empty.
- `acoustic_keywords`: concise controlled tags for spectro-temporal form. Use only:
  - frequency: `high_frequency`, `low_frequency`, `frequency_modulated`
  - spectral quality: `tonal`, `broadband`, `noisy`, `harmonic`
  - temporal structure: `short`, `long`, `abrupt`, `repetitive`, `pulsed`, `multi_component`
  - amplitude: `loud`, `quiet`
  - variation: `graded`
- `acoustic_description`: prose. What the call sounds like / its spectro-temporal structure. Avoid functional terms here.
- `semantic_keywords`: concise controlled tags for communicative function/context. Use only:
  - social cohesion: `contact`, `group_coordination`, `affiliation`
  - agonistic: `threat`, `aggression`, `submission`
  - danger: `alarm`, `predator`
  - distress and care: `distress`, `begging`, `caregiving`
  - reproduction: `courtship`, `mating`
  - resources: `food`, `recruitment`
  - territorial spacing: `territorial`, `spacing`
  - identity and attention: `identity`, `attention`
  - metacommunicative: `play`, `display`
  - combinatorial: `combinatorial`
  Do not use semantic keywords for caller class (`infant` belongs in `scope.life_stages`), transmission range, or evidence type.
- `semantic_description`: prose. Context of use, i.e. when it is made. Avoid acoustic terms here.
- `acoustic_references` / `semantic_references`: lists of `{id, url}` objects. **At least one of each is required** — they provide the basis for validating the corresponding description. `id` must exist in top-level `references`.
- `audio_samples`: optional list of URLs (Macaulay Library, xeno-canto, etc.).
- `playback_references`: list of `{id, url}` citations to playback studies on this call. Empty or omitted = no playback experiments. Non-empty = yes, with these citations.

### Three agreement labels (with explanations)

Each call has three orthogonal agreement labels, each on the weak/medium/strong rubric below, and each paired with a required `_explanation` that justifies the label by citing relevant sources.

- `call_type_existence_agreement`: do studies agree that this exists as a discrete call type in the repertoire, or do some merge / split / rename it?
- `acoustic_description_agreement`: do sources agree on the acoustic structure, and are there enough of them?
- `semantic_description_agreement`: do sources agree on the context of use / function?

Each is paired with:

- `call_type_existence_explanation`
- `acoustic_description_explanation`
- `semantic_description_explanation`

The explanation is **always required**, regardless of label. A `strong` label should name the independent sources that earn it; a `weak` label should name the evidence gap, disagreement, or speculative boundary. Without an explanation, the label is unverifiable.

## Agreement rubric

Applies to all three agreement labels.

- **strong** — Multiple independent sources agree; little substantive disagreement.
- **medium** — Some support, but sources are limited, partially disagree, or the category is broad/graded.
- **weak** — Sparse, speculative, contested, or based mainly on one source.

## Acoustic vs semantic — keep them separate

A common drift is to let function leak into acoustic descriptions (e.g. "alarm-like trill"). Acoustic = what it sounds like to a spectrogram. Semantic = when/why it is produced and to what effect. If a description mixes both, split it.

## Workflow

1. Checkout to main and pull.
2. Pick unfinished species from `repertoires/species_index.yaml` (no file yet at `repertoires/llm_literature_extraction/species/<kebab-name>.yaml`) and generate YAML repertoire files for all.
3. Run `python .agents/skills/llm-literature-extraction-repertoire/validate.py repertoires/llm_literature_extraction/species/<file>.yaml` and commit directly to main and push.
4. Never touch `schema.json`, `validate.py`, or this skill as a side-effect.
5. Do **not** use `database.json` as a source. Build every repertoire from scratch using primary literature only.
