"""Streamlit explorer for the species repertoire YAML files.

Run:
    pip install streamlit pyyaml
    streamlit run explorer.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).parent
SPECIES_DIR = ROOT / "species"


def species_cache_key() -> tuple[tuple[str, int], ...]:
    return tuple(
        (path.name, path.stat().st_mtime_ns)
        for path in sorted(SPECIES_DIR.glob("*.yaml"))
    )


@st.cache_data
def load_species(_cache_key: tuple[tuple[str, int], ...]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(SPECIES_DIR.glob("*.yaml")):
        out[path.stem] = yaml.safe_load(path.read_text())
    return out


AGREEMENT_COLORS = {
    "weak": "#b00020",
    "medium": "#a16207",
    "strong": "#166534",
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


def render_citation(c: dict, refs: dict) -> str:
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


def render_scope(data: dict, refs: dict) -> None:
    scope = data.get("scope") or {}
    life_stages = ", ".join(scope.get("life_stages") or [])
    sexes = ", ".join(scope.get("sexes") or [])
    population = "population-specific" if scope.get("population_specific") else "not population-specific"
    bits = [b for b in (life_stages, sexes, population) if b]
    if bits:
        st.caption("Scope: " + " | ".join(bits))
    if scope.get("note"):
        citations = ", ".join(
            render_citation(c, refs) for c in scope.get("scope_references") or []
        )
        suffix = f" Sources: {citations}" if citations else ""
        st.caption(scope["note"] + suffix)


def render_call(call: dict, refs: dict) -> None:
    st.markdown(f"### {call['name']}")
    if call.get("alternative_names"):
        st.caption("Also known as: " + ", ".join(call["alternative_names"]))
    render_scope(call, refs)

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
            with st.expander("why?"):
                st.write(call.get(expl_key, "—"))

    st.markdown("**Acoustic description**")
    st.write(call["acoustic_description"])
    st.caption(
        "Sources: " + ", ".join(render_citation(c, refs) for c in call["acoustic_references"])
    )

    st.markdown("**Semantic description**")
    st.write(call["semantic_description"])
    st.caption(
        "Sources: " + ", ".join(render_citation(c, refs) for c in call["semantic_references"])
    )

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
    st.set_page_config(page_title="Species Repertoires", layout="wide")
    st.title("🐾 Species Repertoires")

    species = load_species(species_cache_key())
    if not species:
        st.info("No species files found in `species/`. Add a YAML file to get started.")
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
        st.caption(f"{len(species)} species, {sum(len(v.get('calls', [])) for v in species.values())} calls total")

    data = species[choice]
    refs = data.get("references", {}) or {}
    st.header(f"{data['common_name']}")
    st.markdown(f"*{data['scientific_name']}*")
    if line := taxonomy_line(data):
        st.caption(line)

    calls = data.get("calls", [])
    if not calls:
        st.warning("No calls described yet.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Calls described", len(calls))
    for metric, label, key in (
        (m2, "Existence", "call_type_existence_agreement"),
        (m3, "Acoustic", "acoustic_description_agreement"),
        (m4, "Semantic", "semantic_description_agreement"),
    ):
        with metric:
            st.caption(label)
            st.markdown(agreement_summary_badges(calls, key), unsafe_allow_html=True)
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
