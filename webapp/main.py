from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlencode

from dash import Dash, Input, Output, State, dcc, html, no_update

from .charts import coverage_chart
from .data import BUNDLE_PATH, load_animallex_bundle
from .pages import (
    analysis_page,
    explore_page,
    missing_bundle_page,
    overview_page,
    repertoire_table,
    species_options,
)


def _query(search: str | None) -> dict[str, list[str]]:
    return parse_qs((search or "").lstrip("?"))


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
                        ]
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
        State("url", "search"),
    )
    def render_page(pathname: str | None, mode: str, search: str | None):
        if bundle is None:
            return missing_bundle_page()
        path = pathname or "/overview"
        if path == "/explore":
            selected = (_query(search).get("species") or [None])[0]
            return explore_page(bundle, selected)
        if path == "/analysis":
            return analysis_page(bundle, mode or "public")
        return overview_page(bundle, mode or "public")

    @app.callback(
        Output("explore-table", "children"),
        Output("explore-summary", "children"),
        Output("url", "search"),
        Input("species-select", "value"),
        Input("call-filter", "value"),
        State("url", "search"),
    )
    def update_explore(species: str, filter_value: str | None, current_search: str | None):
        if bundle is None or not species:
            return no_update, no_update, no_update
        calls = [call for call in bundle.calls if call["species"] == species]
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
            row
            for row in bundle.analysis["overview"]["species_counts"]
            if row["species"] == species
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
        Input("species-search", "value"),
        State("species-select", "value"),
    )
    def filter_species_list(search: str | None, selected_species: str | None):
        if bundle is None:
            return no_update, no_update
        options = species_options(bundle, search, selected_species)
        total = len(bundle.analysis["overview"]["species_counts"])
        return options, f"{len(options):,} of {total:,} species"

    @app.callback(
        Output("semantic-coverage-chart", "figure"),
        Output("acoustic-coverage-chart", "figure"),
        Input("coverage-group", "value"),
        Input("coverage-threshold", "value"),
    )
    def update_coverage(group_key: str, threshold: float):
        if bundle is None or not group_key or threshold is None:
            return no_update, no_update
        group = bundle.analysis["coverage"]["groups"][group_key]
        thresholds = bundle.analysis["coverage"]["thresholds"]
        selected_threshold = round(float(threshold), 2)
        return (
            coverage_chart(group, "semantic", selected_threshold, thresholds),
            coverage_chart(group, "acoustic", selected_threshold, thresholds),
        )

    @app.callback(
        Output("species-matrix-detail", "children"),
        Input("species-matrix-chart", "clickData"),
        prevent_initial_call=True,
    )
    def matrix_detail(click_data):
        point = click_data["points"][0]
        if point.get("z") is None:
            return "There are not enough call pairs for this cell."
        return (
            f"{point['y']} and {point['x']}: r = {point['z']:.3f} "
            f"from {int(point['customdata']):,} call pairs."
        )

    @app.callback(
        Output("pmi-detail", "children"),
        Input("pmi-chart", "clickData"),
        prevent_initial_call=True,
    )
    def pmi_detail(click_data):
        point = click_data["points"][0]
        joint_count, expected_count, p_value, q_value = point["customdata"]
        return (
            f"{point['y']} and {point['x']}: within-species log2(O/E) = "
            f"{point['z']:.2f}. "
            f"from {int(joint_count):,} matching calls. "
            f"The within-species shuffle expected {expected_count:.2f} calls. "
            f"Permutation p = {p_value:.3g}, FDR q = {q_value:.3g}."
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
