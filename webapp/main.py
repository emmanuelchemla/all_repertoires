from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qs, urlencode

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
import pandas as pd

from .charts import coverage_chart
from .data import BUNDLE_PATH, bundle_for_confidence, load_animallex_bundle

from repertoire_explorer import (
    AnalysisConfig,
    CanonicalDataset,
    compute_cross_species_motifs,
)
from .pages import (
    analysis_page,
    explore_page,
    missing_bundle_page,
    motif_carousel_card,
    overview_page,
    repertoire_table,
    species_options,
    translations_page,
)


def _query(search: str | None) -> dict[str, list[str]]:
    return parse_qs((search or "").lstrip("?"))


def _motif_dataset(view) -> CanonicalDataset:
    common_names = {
        row["species"]: {"common_name": row.get("common_name") or row["species"]}
        for row in view.analysis["overview"]["species_counts"]
    }
    return CanonicalDataset(
        name=view.manifest["dataset"],
        calls=pd.DataFrame(view.calls),
        source_path=Path(view.manifest["source"]),
        species_metadata=common_names,
    )


def create_app(bundle_path: Path | str = BUNDLE_PATH) -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True, title="AnimalLex")
    try:
        bundle = load_animallex_bundle(bundle_path)
    except FileNotFoundError:
        bundle = None

    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            html.Header(
                [
                    dcc.Link("AnimalLex", href="/overview", className="brand"),
                    html.Nav(
                        [
                            dcc.Link("Overview", href="/overview"),
                            dcc.Link("Explore", href="/explore"),
                            dcc.Link("Analysis", href="/analysis"),
                            dcc.Link("Translations", href="/translations"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Confidence", className="control-label"),
                                    dcc.RadioItems(
                                        id="confidence-select",
                                        options=[
                                            {"label": "All", "value": "all"},
                                            {"label": "Medium+", "value": "medium_plus"},
                                            {"label": "High", "value": "high"},
                                        ],
                                        value="all",
                                        inline=True,
                                        persistence=True,
                                        persistence_type="local",
                                        className="confidence-select",
                                    ),
                                ],
                                className="confidence-control",
                            ),
                            dcc.RadioItems(
                                id="mode-select",
                                options=[
                                    {"label": "Public", "value": "public"},
                                    {"label": "Research", "value": "research"},
                                ],
                                value="public",
                                inline=True,
                                persistence=True,
                                persistence_type="local",
                                className="mode-select",
                            ),
                        ],
                        className="header-controls",
                    ),
                ],
                className="site-header",
            ),
            html.Div(id="page-content"),
        ],
        className="site",
    )

    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
        Input("mode-select", "value"),
        Input("confidence-select", "value"),
        State("url", "search"),
    )
    def render_page(
        pathname: str | None,
        mode: str,
        confidence_filter: str,
        search: str | None,
    ):
        if bundle is None:
            return missing_bundle_page()
        view = bundle_for_confidence(bundle, confidence_filter)
        path = pathname or "/overview"
        if path == "/explore":
            selected = (_query(search).get("species") or [None])[0]
            return explore_page(view, selected)
        if path == "/analysis":
            return analysis_page(view, mode or "public")
        if path == "/translations":
            return translations_page(view, mode or "public")
        return overview_page(view, mode or "public")

    @app.callback(
        Output("explore-table", "children"),
        Output("explore-summary", "children"),
        Output("url", "search"),
        Input("species-select", "value", allow_optional=True),
        Input("call-filter", "value", allow_optional=True),
        State("confidence-select", "value"),
        State("url", "search"),
        prevent_initial_call=True,
    )
    def update_explore(
        species: str,
        filter_value: str | None,
        confidence_filter: str,
        current_search: str | None,
    ):
        if bundle is None or not species:
            return no_update, no_update, no_update
        view = bundle_for_confidence(bundle, confidence_filter)
        calls = [call for call in view.calls if call["species"] == species]
        query = (filter_value or "").strip().lower()
        if query:
            calls = [
                call
                for call in calls
                if query in call["call_name"].lower()
                or any(query in keyword.lower() for keyword in call["acoustic_keywords"])
                or any(query in keyword.lower() for keyword in call["semantic_keywords"])
            ]
        overview_row = next(
            (
                row
                for row in view.analysis["overview"]["species_counts"]
                if row["species"] == species
            ),
            {"common_name": species},
        )
        common_name = overview_row.get("common_name") or species
        summary = f"{common_name} ({species}). {len(calls):,} calls shown."
        target_search = "?" + urlencode({"species": species})
        return repertoire_table(calls), summary, (
            no_update if current_search == target_search else target_search
        )

    @app.callback(
        Output("species-select", "options"),
        Output("species-search-count", "children"),
        Input("species-search", "value", allow_optional=True),
        State("confidence-select", "value"),
        State("species-select", "value", allow_optional=True),
        prevent_initial_call=True,
    )
    def filter_species_list(
        search: str | None,
        confidence_filter: str,
        selected_species: str | None,
    ):
        if bundle is None:
            return no_update, no_update
        view = bundle_for_confidence(bundle, confidence_filter)
        options = species_options(view, search, selected_species)
        total = len(view.analysis["overview"]["species_counts"])
        return options, f"{len(options):,} of {total:,} species"

    @app.callback(
        Output("semantic-coverage-chart", "figure"),
        Output("acoustic-coverage-chart", "figure"),
        Input("coverage-group", "value", allow_optional=True),
        Input("coverage-threshold", "value", allow_optional=True),
        State("confidence-select", "value"),
        prevent_initial_call=True,
    )
    def update_coverage(
        group_key: str, threshold: float, confidence_filter: str
    ):
        if bundle is None or not group_key or threshold is None:
            return no_update, no_update
        view = bundle_for_confidence(bundle, confidence_filter)
        group = view.analysis["coverage"]["groups"].get(group_key)
        if group is None:
            group = view.analysis["coverage"]["groups"]["all"]
        thresholds = view.analysis["coverage"]["thresholds"]
        selected_threshold = round(float(threshold), 2)
        return (
            coverage_chart(group, "semantic", selected_threshold, thresholds),
            coverage_chart(group, "acoustic", selected_threshold, thresholds),
        )

    @app.callback(
        Output("motif-results", "data"),
        Input("motif-acoustic-threshold", "value", allow_optional=True),
        Input("motif-semantic-threshold", "value", allow_optional=True),
        Input("motif-minimum-species", "value", allow_optional=True),
        Input("confidence-select", "value"),
        prevent_initial_call=True,
    )
    def update_motif_results(
        acoustic_threshold: float | None,
        semantic_threshold: float | None,
        minimum_species: int | None,
        confidence_filter: str,
    ):
        if (
            bundle is None
            or acoustic_threshold is None
            or semantic_threshold is None
            or minimum_species is None
        ):
            return no_update
        view = bundle_for_confidence(bundle, confidence_filter)
        config = replace(
            AnalysisConfig(**bundle.manifest["config"]),
            motif_acoustic_similarity=float(acoustic_threshold),
            motif_semantic_similarity=float(semantic_threshold),
            motif_minimum_species=int(minimum_species),
            motif_minimum_families=1,
            motif_minimum_orders=1,
            motif_overlap_jaccard=1.1,
            motif_max_per_signature=10_000,
            motif_max_results=10_000,
            motif_exclude_low_confidence=False,
        )
        return compute_cross_species_motifs(
            _motif_dataset(view),
            view.acoustic_similarity,
            view.semantic_similarity,
            config,
        )

    @app.callback(
        Output("motif-index", "data"),
        Input("motif-results", "data", allow_optional=True),
        Input("motif-previous", "n_clicks", allow_optional=True),
        Input("motif-next", "n_clicks", allow_optional=True),
        State("motif-index", "data", allow_optional=True),
        prevent_initial_call=True,
    )
    def update_motif_index(results, _previous, _next, current_index):
        motifs = (results or {}).get("motifs", [])
        if not motifs or ctx.triggered_id == "motif-results":
            return 0
        index = int(current_index or 0)
        if ctx.triggered_id == "motif-previous":
            return (index - 1) % len(motifs)
        if ctx.triggered_id == "motif-next":
            return (index + 1) % len(motifs)
        return min(index, len(motifs) - 1)

    @app.callback(
        Output("motif-carousel-stage", "children"),
        Output("motif-position", "children"),
        Output("motif-result-summary", "children"),
        Output("motif-previous", "disabled"),
        Output("motif-next", "disabled"),
        Input("motif-results", "data", allow_optional=True),
        Input("motif-index", "data", allow_optional=True),
        prevent_initial_call=True,
    )
    def render_motif_carousel(results, current_index):
        motifs = (results or {}).get("motifs", [])
        if not motifs:
            return (
                html.Div(
                    "No motifs meet these parameters for the selected confidence level.",
                    className="motif-empty",
                ),
                "0 / 0",
                "0 motifs",
                True,
                True,
            )
        index = min(int(current_index or 0), len(motifs) - 1)
        return (
            motif_carousel_card(motifs[index]),
            f"{index + 1:,} / {len(motifs):,}",
            f"{len(motifs):,} motifs · strongest minimum similarity first",
            len(motifs) == 1,
            len(motifs) == 1,
        )

    @app.callback(
        Output("species-matrix-detail", "children"),
        Input("species-matrix-chart", "clickData", allow_optional=True),
        prevent_initial_call=True,
    )
    def matrix_detail(click_data):
        if not click_data:
            return no_update
        point = click_data["points"][0]
        if point.get("z") is None:
            return "There are not enough call pairs for this cell."
        return (
            f"{point['y']} and {point['x']}: r = {point['z']:.3f} "
            f"from {int(point['customdata']):,} call pairs."
        )

    @app.callback(
        Output("pmi-detail", "children"),
        Input("pmi-chart", "clickData", allow_optional=True),
        prevent_initial_call=True,
    )
    def pmi_detail(click_data):
        if not click_data:
            return no_update
        point = click_data["points"][0]
        joint_count, expected_count, p_value, q_value = point["customdata"]
        return (
            f"{point['y']} and {point['x']}: PMI = {point['z']:.2f} bits, "
            f"from {int(joint_count):,} matching calls. "
            f"The global shuffle expected {expected_count:.2f} calls. "
            f"Permutation p = {p_value:.3g}, FDR q = {q_value:.3g}."
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)
