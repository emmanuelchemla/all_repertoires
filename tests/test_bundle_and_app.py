from __future__ import annotations

from pathlib import Path

import pytest

from repertoire_explorer import load_bundle, validate_bundle
from webapp.charts import coverage_chart, form_meaning_chart, pmi_chart, species_matrix_chart
from webapp.main import create_app
from webapp.pages import (
    call_display_labels,
    species_display_labels,
    species_options,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "animallex" / "latest"
SOURCE = ROOT / "repertoires" / "llm_knowledge+search" / "species"


def test_generated_bundle_is_current_and_complete() -> None:
    validate_bundle(BUNDLE, SOURCE)
    bundle = load_bundle(BUNDLE)

    assert bundle.manifest["n_species"] == 128
    assert bundle.manifest["n_calls"] == 1507
    assert bundle.analysis["overview"]["n_calls"] == 1507
    assert bundle.analysis["coverage"]["default_group"] == "all"
    assert bundle.acoustic_similarity.shape == (1507, 1507)
    assert bundle.semantic_similarity.shape == (1507, 1507)

    with pytest.raises(ValueError, match="settings changed"):
        validate_bundle(BUNDLE, SOURCE, expected_config={"analysis_version": "different"})


def test_dash_app_starts_from_generated_bundle() -> None:
    app = create_app(BUNDLE)
    client = app.server.test_client()

    assert client.get("/overview").status_code == 200
    assert client.get("/explore?species=Pan%20paniscus").status_code == 200
    assert client.get("/analysis").status_code == 200
    assert "page-content" in client.get("/_dash-layout").get_data(as_text=True)


def test_explore_species_search_uses_common_and_scientific_names() -> None:
    bundle = load_bundle(BUNDLE)

    assert species_options(bundle, "bonobo") == [
        {"label": "Bonobo (Pan paniscus)", "value": "Pan paniscus"}
    ]
    assert species_options(bundle, "pan paniscus") == [
        {"label": "Bonobo (Pan paniscus)", "value": "Pan paniscus"}
    ]


def test_form_meaning_tooltips_use_readable_call_labels() -> None:
    bundle = load_bundle(BUNDLE)
    labels = call_display_labels(bundle)
    figure = form_meaning_chart(bundle.analysis["form_meaning"], labels)
    marker_trace = next(trace for trace in figure.data if trace.mode == "markers")

    assert marker_trace.customdata
    assert all("|||" not in label for pair in marker_trace.customdata for label in pair)
    assert all("(" in label and label.endswith(")") for pair in marker_trace.customdata for label in pair)


def test_pmi_chart_exposes_significance_values() -> None:
    bundle = load_bundle(BUNDLE)
    figure = pmi_chart(bundle.analysis["pmi"])
    trace = figure.data[0]

    assert len(trace.customdata[0][0]) == 4
    assert trace.text is None
    assert "permutation p" in trace.hovertemplate
    assert "FDR q" in trace.hovertemplate


def test_species_matrix_uses_common_names_and_matching_color_direction() -> None:
    bundle = load_bundle(BUNDLE)
    figure = species_matrix_chart(
        bundle.analysis["species_matrix"], species_display_labels(bundle)
    )
    trace = figure.data[0]

    assert "Bonobo" in trace.x
    assert "Pan paniscus" not in trace.x
    assert trace.colorscale[0][1] == "rgb(5,48,97)"
    assert trace.colorscale[-1][1] == "rgb(103,0,31)"


def test_coverage_chart_uses_species_percent_on_x_axis() -> None:
    bundle = load_bundle(BUNDLE)
    result = bundle.analysis["coverage"]
    group = result["groups"][result["default_group"]]

    figure = coverage_chart(
        group, "semantic", result["default_threshold"], result["thresholds"]
    )
    trace = figure.data[0]

    assert list(trace.x) == [25, 50, 75, 100]
    assert figure.layout.xaxis.title.text == "% of species represented"
    assert list(figure.layout.xaxis.ticktext) == ["25%", "50%", "75%", "100%"]
    assert figure.layout.yaxis.ticksuffix == "%"


def test_dash_app_has_a_clear_missing_bundle_state(tmp_path: Path) -> None:
    app = create_app(tmp_path / "missing")

    assert app.server.test_client().get("/overview").status_code == 200
