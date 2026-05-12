---
name: species-repertoire
description: Use this skill whenever creating, editing, or reviewing a species repertoire YAML file under repertoires/species/. Triggers on any work involving call types, acoustic/semantic descriptions, or repertoire bibliography. Provides the schema, conventions, agreement rubrics, and validation steps that keep multi-agent edits consistent.
---

# Species repertoire skill

## File layout

- One YAML file per species at `species/<scientific-name-kebab-case>.yaml` (e.g. `corvus-corax.yaml`).
- One file = one species = atomic unit of parallel work. Never split a species across files.
- Schema lives at `schema.json`. Run `python validate.py species/<file>.yaml` before committing.

## Field conventions

### Top-level

- `scientific_name`: binomial, italicized intent only (plain text in YAML).
- `common_name`: most widely used English common name.
- `taxonomy`: required object with `kingdom`, `phylum`, `class`, `order`, `family`, `genus`, and `species`. Use canonical Latin taxon names; `species` is the species epithet only, so `genus` + `species` should match `scientific_name`.
- `calls`: list of call-type objects (see below).
- `references`: bibliography for the whole file. Cite with **stable string IDs** in `lastname_year` form (`smith_2019`, `marler_etal_1967`).

Do not include calls known only from captivity when there is no wild evidence for that call type.

### Per call type

- `name`: canonical name used in the literature. Prefer the most common term.
- `alternative_names`: list of synonyms encountered in other papers.
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

## Multi-agent etiquette

- One agent per species file per session. The git layer is your conflict detector.
- Do not edit `schema.json` or this skill as a side-effect of species work. Propose schema changes separately.
- After any edit: run `validate.py`. If it fails, fix before reporting done.
