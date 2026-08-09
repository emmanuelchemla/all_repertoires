from __future__ import annotations

from pathlib import Path

import pytest

from paper_code.bundle_render import render_results_tex
from repertoire_explorer import load_bundle, validate_bundle
from webapp.charts import (
    call_count_distribution_chart,
    coverage_chart,
    form_meaning_chart,
    pmi_chart,
    prediction_chart,
    species_matrix_chart,
)
from webapp.data import bundle_for_confidence, bundle_for_filters
from webapp.main import create_app
from webapp.pages import (
    analysis_page,
    call_display_labels,
    species_display_labels,
    species_options,
    translations_page,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "animallex" / "latest"
SOURCE = ROOT / "repertoires" / "llm_knowledge+search" / "species"


def test_call_count_histogram_limits_y_axis_tick_density() -> None:
    figure = call_count_distribution_chart(
        [
            {"n_calls": n_calls, "n_species": n_species}
            for n_calls, n_species in [(1, 11), (2, 21), (3, 27)]
        ]
    )

    assert figure.layout.yaxis.nticks == 7
    assert figure.layout.yaxis.dtick is None
    assert figure.layout.yaxis.tickformat == ",d"


def test_generated_bundle_is_current_and_complete() -> None:
    validate_bundle(BUNDLE, SOURCE)
    bundle = load_bundle(BUNDLE)

    assert bundle.manifest["n_species"] == 128
    assert bundle.manifest["n_calls"] == 1477
    assert bundle.analysis["overview"]["n_calls"] == 1477
    assert bundle.analysis["species_fit_views"]["include"]["overview"]["n_calls"] == 1311
    assert bundle.analysis["species_fit_views"]["include_caution"]["overview"]["n_calls"] == 1413
    assert bundle.analysis["coverage"]["default_group"] == "all"
    assert bundle.manifest["config"]["n_permutations"] == 999
    assert bundle.manifest["build_mode"] == "iteration"
    assert bundle.manifest["confidence_counts"] == {
        "high": 548,
        "medium": 733,
        "low": 196,
    }
    assert bundle.manifest["species_fit_counts"] == {
        "include": 1311,
        "caution": 102,
        "exclude": 64,
    }
    assert bundle.analysis["confidence_views"]["medium_plus"]["overview"]["n_calls"] == 1281
    assert bundle.analysis["confidence_views"]["high"]["overview"]["n_calls"] == 548
    assert bundle.analysis["form_meaning"]["permutation_unit"] == (
        "semantic call identities shuffled within species"
    )
    assert bundle.analysis["form_meaning"]["similarity_method"] == (
        "cosine similarity of description embeddings"
    )
    assert bundle.analysis["form_meaning_keywords"]["similarity_method"] == (
        "Jaccard similarity of keyword sets"
    )
    assert len(bundle.analysis["prediction"]["conditions"]) == 3
    assert len(bundle.analysis["confidence_views"]["medium_plus"]["prediction"]["conditions"]) == 3
    assert len(bundle.analysis["confidence_views"]["high"]["prediction"]["conditions"]) == 3
    assert bundle.analysis["motifs"]["n_motifs"] > 0
    assert bundle.analysis["species_fit_views"]["include"]["motifs"]["n_motifs"] > 0
    assert bundle.acoustic_similarity.shape == (1477, 1477)
    assert bundle.semantic_similarity.shape == (1477, 1477)

    with pytest.raises(ValueError, match="settings changed"):
        validate_bundle(BUNDLE, SOURCE, expected_config={"analysis_version": "different"})


def test_paper_assets_use_the_include_only_webapp_view(tmp_path: Path) -> None:
    bundle = load_bundle(BUNDLE)
    output = tmp_path / "generated_results.tex"

    render_results_tex(bundle, output)

    generated = output.read_text()
    manuscript = (ROOT / "paper" / "main.tex").read_text().split(
        r"\end{document}", maxsplit=1
    )[0]
    include = bundle_for_filters(bundle, "include", "all")
    assert include.analysis["form_meaning"]["similarity_method"].startswith(
        "cosine similarity"
    )
    assert r"\newcommand{\ncalls}{1{,}311\xspace}" in generated
    assert r"\newcommand{\nspecies}{111\xspace}" in generated
    assert rf"\newcommand{{\nmotifs}}{{{include.analysis['motifs']['n_motifs']}\xspace}}" in generated
    assert r"\input{generated_results.tex}" in manuscript
    assert "figures/generated/form_meaning.pdf" in manuscript
    assert "figures/generated/prediction.pdf" in manuscript
    assert "TODO: Select curated examples from web app" in manuscript
    assert "UMAP" not in manuscript


def test_dash_app_starts_from_generated_bundle() -> None:
    app = create_app(BUNDLE)
    client = app.server.test_client()

    assert client.get("/overview").status_code == 200
    assert client.get("/explore?species=Pan%20paniscus").status_code == 200
    assert client.get("/analysis").status_code == 200
    assert client.get("/translations").status_code == 200
    assert "page-content" in client.get("/_dash-layout").get_data(as_text=True)
    assert "confidence-select" in client.get("/_dash-layout").get_data(as_text=True)
    assert "species-fit-select" in client.get("/_dash-layout").get_data(as_text=True)


def test_form_meaning_keywords_callback_returns_keyword_chart() -> None:
    app = create_app(BUNDLE)
    callback = app.callback_map["form-meaning-chart.figure"]["callback"].__wrapped__

    figure = callback("keywords", "all", "include")

    marker_trace = next(trace for trace in figure.data if trace.mode == "markers")
    assert marker_trace.x
    assert marker_trace.y


def test_confidence_changes_preserve_the_current_section() -> None:
    script = (ROOT / "webapp" / "assets" / "preserve-scroll.js").read_text()

    assert '#confidence-select' in script
    assert '#species-fit-select' in script
    assert 'querySelectorAll("main section[data-analysis-section]")' in script
    assert '"click"' in script
    assert "MutationObserver" in script
    assert "window.scrollTo" in script


def test_confidence_view_filters_calls_and_analysis_together() -> None:
    bundle = load_bundle(BUNDLE)

    medium = bundle_for_confidence(bundle, "medium_plus")
    high = bundle_for_confidence(bundle, "high")

    assert len(medium.calls) == medium.analysis["overview"]["n_calls"] == 1281
    assert {call["confidence"] for call in medium.calls} == {"medium", "high"}
    assert len(high.calls) == high.analysis["overview"]["n_calls"] == 548
    assert {call["confidence"] for call in high.calls} == {"high"}
    assert medium.acoustic_similarity.shape == (1281, 1281)
    assert high.acoustic_similarity.shape == (548, 548)
    assert high.semantic_similarity.shape == (548, 548)
    assert medium.analysis["prediction"]["n_calls"] == 1281
    assert high.analysis["prediction"]["n_calls"] == 548
    high_family_ridge = high.analysis["prediction"]["results"]["held_out_families"][
        "ridge"
    ]["cosine"]["mean"]
    all_family_ridge = bundle.analysis["prediction"]["results"][
        "held_out_families"
    ]["ridge"]["cosine"]["mean"]
    assert high_family_ridge != all_family_ridge


def test_species_fit_view_filters_calls_and_analysis_together() -> None:
    bundle = load_bundle(BUNDLE)

    include = bundle_for_filters(bundle, "include", "all")
    caution = bundle_for_filters(bundle, "include_caution", "all")

    assert len(include.calls) == include.analysis["overview"]["n_calls"] == 1311
    assert {call["species_fit"] for call in include.calls} == {"include"}
    assert len(caution.calls) == caution.analysis["overview"]["n_calls"] == 1413
    assert {call["species_fit"] for call in caution.calls} == {"include", "caution"}
    assert include.acoustic_similarity.shape == (1311, 1311)


def test_analysis_page_has_prediction_chart_and_section_navigation() -> None:
    bundle = load_bundle(BUNDLE)
    page = analysis_page(bundle, "public")
    toc, content = page.children

    assert page.className == "analysis-page"
    assert toc.className == "analysis-toc"
    assert [link.href for link in toc.children[1].children] == [
        "#form-meaning",
        "#prediction",
        "#cross-species-overlap",
        "#species-matrix",
        "#keyword-association",
    ]
    sections = [child for child in content.children if getattr(child, "id", None)]
    assert [section.id for section in sections] == [
        "form-meaning",
        "prediction",
        "cross-species-overlap",
        "species-matrix",
        "keyword-association",
    ]
    form_heading = sections[0].children[0]
    method_toggle = form_heading.children[1]
    assert method_toggle.id == "form-meaning-basis"
    assert method_toggle.value == "descriptions"
    assert [option["label"] for option in method_toggle.options] == [
        "Descriptions",
        "Keywords",
    ]


def test_translations_page_has_compact_ranked_motif_carousel() -> None:
    bundle = load_bundle(BUNDLE)
    result = bundle.analysis["motifs"]
    page = translations_page(bundle, "public")
    controls = next(
        child for child in page.children if getattr(child, "className", None) == "motif-controls"
    )
    stage = next(child for child in page.children if getattr(child, "id", None) == "motif-carousel-stage")

    assert page.className == "page translations-page"
    assert [control.children[1].id for control in controls.children] == [
        "motif-acoustic-threshold",
        "motif-semantic-threshold",
        "motif-minimum-species",
    ]
    assert all(control.children[1].allow_direct_input is False for control in controls.children)
    assert stage.children.className == "motif-card motif-carousel-card"
    assert result["n_motifs"] == 73
    assert result["criteria"] == {
        "minimum_species": 4,
        "minimum_families": 1,
        "minimum_orders": 1,
        "minimum_acoustic_similarity": 0.65,
        "minimum_semantic_similarity": 0.65,
        "excluded_semantic_keywords": ["attention"],
        "exclude_low_confidence": False,
        "overlap_jaccard": 1.1,
        "maximum_per_semantic_signature": 10_000,
        "maximum_results": 10_000,
    }
    assert all(motif["n_species"] >= 4 for motif in result["motifs"])
    assert all(
        motif["minimum_acoustic_similarity"] >= 0.65
        and motif["minimum_semantic_similarity"] >= 0.65
        for motif in result["motifs"]
    )
    bottlenecks = [
        min(motif["minimum_acoustic_similarity"], motif["minimum_semantic_similarity"])
        for motif in result["motifs"]
    ]
    assert bottlenecks == sorted(bottlenecks, reverse=True)
    assert all(
        "acoustic_keywords" in member and "semantic_keywords" in member
        for motif in result["motifs"]
        for member in motif["members"]
    )


def test_prediction_chart_shows_models_splits_and_uncertainty() -> None:
    bundle = load_bundle(BUNDLE)
    result = bundle.analysis["prediction"]
    figure = prediction_chart(result)

    assert len(figure.data) == 6
    assert {trace.name for trace in figure.data} == set(result["models"].values())
    assert all(trace.error_y.array is not None for trace in figure.data)
    labels = [condition["label"] for condition in result["conditions"]]
    assert all(list(trace.x) == labels for trace in figure.data)
    assert [annotation.text for annotation in figure.layout.annotations] == [
        "Cosine similarity ↑",
        "Mean reciprocal rank ↑",
    ]
    assert 0 < figure.layout.yaxis.range[1] < 1
    assert 0 < figure.layout.yaxis2.range[1] < 1
    assert figure.layout.yaxis.range[1] > max(
        result["results"][condition["key"]][model]["cosine"]["ci_high"]
        for condition in result["conditions"]
        for model in result["models"]
    )
    assert figure.layout.yaxis2.range[1] > max(
        result["results"][condition["key"]][model]["mrr"]["ci_high"]
        for condition in result["conditions"]
        for model in result["models"]
    )


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
    first_group = next(iter(bundle.analysis["form_meaning"]["groups"].values()))

    assert marker_trace.customdata
    assert marker_trace.name.endswith(f"n = {first_group['n_pairs']:,}")
    assert "p ≤ 0.001" in marker_trace.name
    assert all("|||" not in label for pair in marker_trace.customdata for label in pair)
    assert all("(" in label and label.endswith(")") for pair in marker_trace.customdata for label in pair)
    assert figure.layout.xaxis.title.text == "Acoustic similarity"
    assert figure.layout.yaxis.title.text == "Semantic similarity"
    assert 0 < figure.layout.legend.x < 0.05
    assert 0.95 < figure.layout.legend.y < 1
    assert figure.layout.legend.xanchor == "left"
    assert figure.layout.legend.yanchor == "top"
    assert figure.layout.legend.bgcolor == "rgba(255, 255, 252, 0.78)"
    assert 0 < figure.layout.legend.borderwidth < 1
    assert figure.layout.legend.title.text is None
    assert figure.layout.margin.t < 50
    assert "Acoustic similarity" in marker_trace.hovertemplate
    assert "Semantic similarity" in marker_trace.hovertemplate
    assert marker_trace.x[0] == pytest.approx(
        1 - first_group["sample"][0]["acoustic_distance"]
    )
    assert marker_trace.y[0] == pytest.approx(
        1 - first_group["sample"][0]["semantic_distance"]
    )


def test_pmi_chart_exposes_significance_values() -> None:
    bundle = load_bundle(BUNDLE)
    figure = pmi_chart(bundle.analysis["pmi"])
    trace = figure.data[0]

    assert len(trace.customdata[0][0]) == 4
    assert trace.text is None
    assert "permutation p" in trace.hovertemplate
    assert "FDR q" in trace.hovertemplate
    assert "PMI =" in trace.hovertemplate
    assert "expected after global shuffling" in trace.hovertemplate
    assert trace.colorbar.title.text == "PMI (bits)"
    assert figure.layout.margin.l <= 100
    assert figure.layout.margin.r <= 100


def test_pmi_chart_groups_keywords_without_changing_cells() -> None:
    bundle = load_bundle(BUNDLE)
    result = bundle.analysis["pmi"]
    figure = pmi_chart(result)
    trace = figure.data[0]

    assert list(trace.x[:3]) == ["affiliation", "contact", "group coordination"]
    assert list(trace.y[:3]) == [
        "frequency modulated",
        "high frequency",
        "low frequency",
    ]
    assert {annotation.text for annotation in figure.layout.annotations} >= {
        "Social Cohesion",
        "Frequency",
    }
    assert len(figure.layout.shapes) == 15
    assert figure.layout.margin.l == 60
    assert min(shape.x0 for shape in figure.layout.shapes) > 0
    assert figure.layout.xaxis.domain[0] == pytest.approx(0.22)

    acoustic_index = result["acoustic_keywords"].index("frequency_modulated")
    semantic_index = result["semantic_keywords"].index("affiliation")
    assert trace.z[0][0] == result["matrix"][acoustic_index][semantic_index]


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
    assert len(figure.layout.shapes) > 0

    classes = bundle.analysis["species_matrix"]["classes"]
    assert len(figure.layout.shapes) == 9
    assert {annotation.text for annotation in figure.layout.annotations} == {
        "Amphibia",
        "Aves",
        "Mammalia",
    }
    for class_name in set(classes):
        positions = [index for index, value in enumerate(classes) if value == class_name]
        assert positions == list(range(min(positions), max(positions) + 1))


def test_coverage_chart_uses_species_percent_on_x_axis() -> None:
    bundle = load_bundle(BUNDLE)
    result = bundle.analysis["coverage"]
    group = result["groups"][result["default_group"]]

    figure = coverage_chart(
        group, "semantic", result["default_threshold"], result["thresholds"]
    )
    trace = figure.data[0]

    assert list(trace.x) == [25, 50, 75, 100]
    assert figure.layout.xaxis.title.text == "% of eligible target species represented"
    assert list(figure.layout.xaxis.ticktext) == ["25%", "50%", "75%", "100%"]
    assert figure.layout.yaxis.ticksuffix == "%"
    assert group["n_target_species"] == group["n_species"] - 1


def test_dash_app_has_a_clear_missing_bundle_state(tmp_path: Path) -> None:
    app = create_app(tmp_path / "missing")

    assert app.server.test_client().get("/overview").status_code == 200
