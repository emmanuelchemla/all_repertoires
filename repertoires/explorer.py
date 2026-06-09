"""Streamlit explorer for the species repertoire YAML files.

Run:
    pip install streamlit pyyaml
    streamlit run explorer.py
"""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).parent


DB_DIRS = {
    "llm_literature_extraction": ROOT / "llm_literature_extraction" / "species",
    "llm_knowledge+search": ROOT / "llm_knowledge+search" / "species",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--db",
        default="llm_literature_extraction",
        help="Database to load: llm_literature_extraction, llm_knowledge+search, or a path to a species YAML directory.",
    )
    args, _ = parser.parse_known_args()
    return args


def resolve_species_dir(db: str) -> Path:
    return DB_DIRS.get(db, Path(db)).expanduser()


def db_options(initial_db: str) -> list[str]:
    options = list(DB_DIRS)
    if initial_db not in options:
        options.append(initial_db)
    return options


def species_cache_key(species_dir: Path) -> tuple[tuple[str, int], ...]:
    return tuple(
        (path.name, path.stat().st_mtime_ns)
        for path in sorted(species_dir.glob("*.yaml"))
    )


@st.cache_data
def load_species(species_dir: Path, _cache_key: tuple[tuple[str, int], ...]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(species_dir.glob("*.yaml")):
        out[path.stem] = yaml.safe_load(path.read_text())
    return out


AGREEMENT_COLORS = {
    "weak": "#b00020",
    "medium": "#a16207",
    "strong": "#166534",
}

LIFE_STAGE_LABELS = {
    "infant": "infant",
    "juvenile": "juvenile",
    "adult": "adult",
    "unknown": "?",
}

LIFE_STAGE_COLORS = {
    "infant": "#7c3aed",
    "juvenile": "#2563eb",
    "adult": "#166534",
    "unknown": "#555",
}

SEX_LABELS = {
    "female": "♀",
    "male": "♂",
    "unknown": "?",
}

SEX_COLORS = {
    "female": "#be185d",
    "male": "#1d4ed8",
    "unknown": "#555",
}

ACOUSTIC_KEYWORD_GROUPS = {
    "frequency": {
        "color": "#d96bc2",
        "keywords": ("high_frequency", "low_frequency", "frequency_modulated"),
    },
    "spectral": {
        "color": "#b8bf18",
        "keywords": ("tonal", "broadband", "noisy", "harmonic"),
    },
    "temporal": {
        "color": "#22b8c7",
        "keywords": ("short", "long", "abrupt", "repetitive", "pulsed", "multi_component"),
    },
    "amplitude": {
        "color": "#96594e",
        "keywords": ("loud", "quiet"),
    },
    "variation": {
        "color": "#64748b",
        "keywords": ("graded",),
    },
}

ACOUSTIC_KEYWORD_TO_GROUP = {
    keyword: group
    for group, spec in ACOUSTIC_KEYWORD_GROUPS.items()
    for keyword in spec["keywords"]
}

SEMANTIC_KEYWORD_GROUPS = {
    "social cohesion": {
        "color": "#1f77b4",
        "keywords": ("contact", "group_coordination", "affiliation"),
    },
    "agonistic": {
        "color": "#9467bd",
        "keywords": ("threat", "aggression", "submission"),
    },
    "danger": {
        "color": "#d62728",
        "keywords": ("alarm", "predator"),
    },
    "distress and care": {
        "color": "#ff7f0e",
        "keywords": ("distress", "begging", "caregiving"),
    },
    "reproduction": {
        "color": "#c026d3",
        "keywords": ("courtship", "mating"),
    },
    "resources": {
        "color": "#2ca02c",
        "keywords": ("food", "recruitment"),
    },
    "territorial spacing": {
        "color": "#8c564b",
        "keywords": ("territorial", "spacing"),
    },
    "identity and attention": {
        "color": "#7f7f7f",
        "keywords": ("identity", "attention"),
    },
    "metacommunicative": {
        "color": "#17becf",
        "keywords": ("play", "display"),
    },
    "combinatorial": {
        "color": "#bcbd22",
        "keywords": ("combinatorial",),
    },
}

SEMANTIC_KEYWORD_TO_GROUP = {
    keyword: group
    for group, spec in SEMANTIC_KEYWORD_GROUPS.items()
    for keyword in spec["keywords"]
}


def normalize_agreement(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"
    label = value.strip().lower()
    return label if label in AGREEMENT_COLORS else "unknown"


def agreement_badge(value: object) -> str:
    label = normalize_agreement(value)
    color = AGREEMENT_COLORS.get(label, "#444")
    return (
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:999px;font-weight:600;font-size:0.85em'>{label}</span>"
    )


def agreement_summary_badges(calls: list[dict], key: str) -> str:
    counts = {
        label: sum(
            1 for call in calls if normalize_agreement(call.get(key)) == label
        )
        for label in ("strong", "medium", "weak")
    }
    bits = " ".join(
        f"{agreement_badge(label)} <span style='font-weight:700'>{counts[label]}</span>"
        for label in ("strong", "medium", "weak")
    )
    return f"<div style='line-height:2'>{bits}</div>"


def render_citation(c: dict | str, refs: dict) -> str:
    if isinstance(c, str):
        rid = c
        url = (refs.get(rid) or {}).get("url") or (refs.get(rid) or {}).get("doi")
    else:
        rid = c.get("id", "?")
        url = c.get("url") or (refs.get(rid) or {}).get("url") or (refs.get(rid) or {}).get("doi")
    if url and not url.startswith("http"):
        url = f"https://doi.org/{url}"
    label = f"[{rid}]"
    return f"[{label}]({url})" if url else label


def taxonomy_line(data: dict) -> str:
    taxonomy = data.get("taxonomy") or {}
    ranks = (
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    )
    return " > ".join(taxonomy[rank] for rank in ranks if taxonomy.get(rank))


def chip(label: str, color: str, title: str) -> str:
    return (
        f"<span title='{escape(title)}' style='display:inline-block;"
        f"background:{color};color:white;padding:2px 8px;margin:0 4px 4px 0;"
        f"border-radius:999px;font-weight:700;font-size:0.8em'>{escape(label)}</span>"
    )


def render_scope(data: dict) -> None:
    scope = data.get("scope") or {}
    bits = []
    for stage in scope.get("life_stages") or []:
        bits.append(
            chip(
                LIFE_STAGE_LABELS.get(stage, stage),
                LIFE_STAGE_COLORS.get(stage, "#555"),
                f"Life stage: {stage}",
            )
        )
    for sex in scope.get("sexes") or []:
        bits.append(
            chip(
                SEX_LABELS.get(sex, sex),
                SEX_COLORS.get(sex, "#555"),
                f"Sex: {sex}",
            )
        )
    if scope.get("population_specific"):
        note = scope.get("note") or "Population-specific call"
        bits.append(chip("!", "#b45309", f"Population specific: {note}"))
    if bits:
        st.markdown("".join(bits), unsafe_allow_html=True)


def render_keyword_groups(
    keywords: list[str],
    groups: dict[str, dict],
    keyword_to_group: dict[str, str],
) -> None:
    if not keywords:
        return

    rendered_groups = [
        "<div style='display:flex;flex-wrap:wrap;gap:4px 12px;"
        "align-items:center;margin:2px 0 8px 0'>"
    ]
    for group, spec in groups.items():
        group_keywords = [kw for kw in spec["keywords"] if kw in keywords]
        if not group_keywords:
            continue
        chips = "".join(
            chip(kw.replace("_", " "), spec["color"], f"{group}: {kw}")
            for kw in group_keywords
        )
        rendered_groups.append(
            f"<span style='display:inline-flex;align-items:baseline;"
            f"white-space:nowrap'>{chips}</span>"
        )

    unknown_keywords = [
        kw for kw in keywords if kw not in keyword_to_group
    ]
    if unknown_keywords:
        chips = "".join(chip(kw.replace("_", " "), "#555", f"unknown group: {kw}") for kw in unknown_keywords)
        rendered_groups.append(
            f"<span style='display:inline-flex;align-items:baseline;"
            f"white-space:nowrap'>{chips}</span>"
        )

    rendered_groups.append("</div>")
    st.markdown("".join(rendered_groups), unsafe_allow_html=True)


def render_acoustic_keywords(call: dict) -> None:
    render_keyword_groups(
        call.get("acoustic_keywords") or [],
        ACOUSTIC_KEYWORD_GROUPS,
        ACOUSTIC_KEYWORD_TO_GROUP,
    )


def render_semantic_keywords(call: dict) -> None:
    render_keyword_groups(
        call.get("semantic_keywords") or [],
        SEMANTIC_KEYWORD_GROUPS,
        SEMANTIC_KEYWORD_TO_GROUP,
    )


def render_sources(label: str, citations: list, refs: dict) -> None:
    if citations:
        st.caption(
            f"{label}: " + ", ".join(render_citation(c, refs) for c in citations)
        )


def has_agreement_fields(call: dict) -> bool:
    return any(
        key in call
        for key in (
            "call_type_existence_agreement",
            "acoustic_description_agreement",
            "semantic_description_agreement",
        )
    )


def render_call(call: dict, refs: dict) -> None:
    addition = call.get("in_primary_inventory") is False
    badge = (
        " &nbsp; <span style='background:#b45309;color:white;padding:2px 8px;"
        "border-radius:999px;font-weight:600;font-size:0.7em;vertical-align:middle'>addition</span>"
        if addition
        else ""
    )
    st.markdown(f"### {escape(call['name'])}{badge}", unsafe_allow_html=True)
    if call.get("alternative_names"):
        st.caption("Also known as: " + ", ".join(call["alternative_names"]))
    render_scope(call)
    render_acoustic_keywords(call)
    render_semantic_keywords(call)

    if call.get("confidence"):
        st.markdown(f"**Confidence:** `{escape(str(call['confidence']))}`")

    if has_agreement_fields(call):
        cols = st.columns(3)
        triples = [
            ("Call type existence", "call_type_existence_agreement", "call_type_existence_explanation"),
            ("Acoustic description", "acoustic_description_agreement", "acoustic_description_explanation"),
            ("Semantic description", "semantic_description_agreement", "semantic_description_explanation"),
        ]
        for col, (label, agreement_key, expl_key) in zip(cols, triples):
            with col:
                agreement = call.get(agreement_key, "unknown")
                st.markdown(
                    f"**{label}** &nbsp; {agreement_badge(agreement)}",
                    unsafe_allow_html=True,
                )
                if call.get(expl_key):
                    with st.expander("why?"):
                        st.write(call[expl_key])

    st.markdown("**Acoustic description**")
    st.write(call.get("acoustic_description", ""))
    render_sources("Sources", call.get("acoustic_references") or call.get("references") or [], refs)

    st.markdown("**Semantic description**")
    st.write(call.get("semantic_description", ""))
    render_sources("Sources", call.get("semantic_references") or call.get("references") or [], refs)

    pb = call.get("playback_references") or []
    audio = call.get("audio_samples") or []
    bits = []
    if pb:
        bits.append(
            "🔁 Playback studies: " + ", ".join(render_citation(c, refs) for c in pb)
        )
    else:
        bits.append("🔁 No playback experiments on record")
    if audio:
        bits.append("🔊 Audio: " + " · ".join(f"[sample {i + 1}]({u})" for i, u in enumerate(audio)))
    st.markdown("  \n".join(bits))

    st.divider()


def main() -> None:
    args = parse_args()

    st.set_page_config(page_title="Species Repertoires", layout="wide")
    st.title("🐾 Species Repertoires")

    options = db_options(args.db)
    with st.sidebar:
        st.header("Database")
        db_choice = st.selectbox(
            "Select database",
            options,
            index=options.index(args.db),
            format_func=lambda db: db.replace("_", " "),
        )
        species_dir = resolve_species_dir(db_choice)
        st.caption(str(species_dir))
        st.divider()

    species = load_species(species_dir, species_cache_key(species_dir))
    if not species:
        st.info(f"No species files found in `{species_dir}`. Add a YAML file to get started.")
        return

    with st.sidebar:
        st.header("Species")
        labels = {
            k: f"{v.get('common_name', k)} — *{v.get('scientific_name', '')}*"
            for k, v in species.items()
        }
        choice = st.radio(
            "Select species", list(species.keys()), format_func=lambda k: labels[k]
        )
        st.divider()
        all_loaded_calls = [call for data in species.values() for call in data.get("calls", [])]
        has_agreements = any(has_agreement_fields(call) for call in all_loaded_calls)
        has_inventory_flags = any("in_primary_inventory" in call for call in all_loaded_calls)
        existence_filter = ["strong", "medium", "weak"]
        inventory_filter = "all"
        if has_agreements or has_inventory_flags:
            st.header("Filters")
        if has_agreements:
            existence_filter = st.multiselect(
                "Call type existence",
                ["strong", "medium", "weak"],
                default=["strong", "medium", "weak"],
            )
        if has_inventory_flags:
            inventory_filter = st.radio(
                "Primary inventory",
                ["all", "only inventory", "only additions"],
                index=0,
            )
        st.divider()
        st.caption(f"{len(species)} species, {sum(len(v.get('calls', [])) for v in species.values())} calls total")

    data = species[choice]
    refs = data.get("references", {}) or {}
    st.header(f"{data['common_name']}")
    st.markdown(f"*{data['scientific_name']}*")
    if line := taxonomy_line(data):
        st.caption(line)

    inventory = data.get("primary_inventory")
    inv_id = (inventory or {}).get("id")
    rationale = (inventory or {}).get("rationale", "")
    if inventory:
        if inv_id:
            citation = render_citation({"id": inv_id}, refs)
            st.markdown(f"📖 **Primary inventory:** {citation}")
        else:
            st.markdown("📖 **Primary inventory:** *none — every call is an addition*")
        if rationale:
            with st.expander("inventory rationale"):
                st.write(rationale)

    all_calls = data.get("calls", [])
    if not all_calls:
        st.warning("No calls described yet.")
        return

    def keep(call: dict) -> bool:
        if has_agreements and normalize_agreement(call.get("call_type_existence_agreement")) not in existence_filter:
            return False
        if inventory_filter == "only inventory" and not call.get("in_primary_inventory"):
            return False
        if inventory_filter == "only additions" and call.get("in_primary_inventory"):
            return False
        return True

    calls = [c for c in all_calls if keep(c)]
    if not calls:
        st.info(f"No calls match the current filters (hiding {len(all_calls)}).")
        return

    n_from_inv = sum(1 for c in calls if c.get("in_primary_inventory"))
    filtered = len(calls) != len(all_calls)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Calls shown", len(calls))
        if filtered:
            st.caption(f"filtered from {len(all_calls)}")
        elif inv_id:
            st.caption(f"{n_from_inv} from inventory · {len(calls) - n_from_inv} additions")
        elif has_inventory_flags:
            st.caption(f"{len(calls)} additions")
    if has_agreements:
        for metric, label, key in (
            (m2, "Existence", "call_type_existence_agreement"),
            (m3, "Acoustic", "acoustic_description_agreement"),
            (m4, "Semantic", "semantic_description_agreement"),
        ):
            with metric:
                st.caption(label)
                st.markdown(agreement_summary_badges(calls, key), unsafe_allow_html=True)
    else:
        for metric, label in ((m2, "High"), (m3, "Medium"), (m4, "Low")):
            with metric:
                st.metric(f"{label} confidence", sum(1 for c in calls if c.get("confidence") == label.lower()))
    st.divider()

    for call in calls:
        render_call(call, refs)

    with st.expander(f"📚 Bibliography ({len(refs)} entries)"):
        for rid, ref in sorted(refs.items()):
            url = ref.get("url") or (f"https://doi.org/{ref['doi']}" if ref.get("doi") else None)
            header = f"**{rid}**" + (f" — [link]({url})" if url else "")
            st.markdown(header)
            st.caption(ref.get("citation", ""))


if __name__ == "__main__":
    main()
