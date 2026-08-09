from __future__ import annotations

from typing import Any

from dash import dash_table, dcc, html

from repertoire_explorer import AnimalLexBundle

from .charts import (
    call_count_distribution_chart,
    coverage_chart,
    form_meaning_chart,
    keyword_frequency_chart,
    pmi_chart,
    species_counts_chart,
    species_matrix_chart,
)


GRAPH_CONFIG = {"displaylogo": False, "responsive": True}


def _research_details(bundle: AnimalLexBundle, mode: str) -> Any:
    if mode != "research":
        return None
    config = bundle.manifest["config"]
    return html.Details(
        [
            html.Summary("Methods and release details"),
            html.Dl(
                [
                    html.Dt("Build mode"),
                    html.Dd(bundle.manifest.get("build_mode", "unknown")),
                    html.Dt("Embedding model"),
                    html.Dd(config["embedding_model"]),
                    html.Dt("Permutation count"),
                    html.Dd(f"{config['n_permutations']:,}"),
                    html.Dt("Source commit"),
                    html.Dd(bundle.manifest["source_commit"]),
                    html.Dt("Excluded calls"),
                    html.Dd(f"{bundle.manifest['excluded_calls']:,}"),
                ]
            ),
        ],
        className="methods",
    )


def overview_page(bundle: AnimalLexBundle, mode: str) -> html.Main:
    result = bundle.analysis["overview"]
    return html.Main(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [html.Span("Species"), html.Strong(f"{result['n_species']:,}")],
                                className="stat",
                            ),
                            html.Div(
                                [html.Span("Calls"), html.Strong(f"{result['n_calls']:,}")],
                                className="stat",
                            ),
                        ],
                        className="stats",
                    ),
                    dcc.Graph(
                        figure=call_count_distribution_chart(
                            result["call_count_distribution"]
                        ),
                        config=GRAPH_CONFIG,
                        className="summary-histogram",
                    ),
                ],
                className="overview-summary",
            ),
            _research_details(bundle, mode),
            html.Section(
                [
                    html.H2("Keyword frequencies"),
                    html.Div(
                        [
                            dcc.Graph(
                                figure=keyword_frequency_chart(
                                    result["semantic_keywords"],
                                    "Semantic functions",
                                    limit=18,
                                ),
                                config=GRAPH_CONFIG,
                            ),
                            dcc.Graph(
                                figure=keyword_frequency_chart(
                                    result["acoustic_keywords"],
                                    "Acoustic features",
                                    limit=18,
                                ),
                                config=GRAPH_CONFIG,
                            ),
                        ],
                        className="chart-grid",
                    ),
                ]
            ),
            html.Section(
                [
                    html.H2("Calls per species"),
                    dcc.Graph(
                        id="species-counts-chart",
                        figure=species_counts_chart(result["species_counts"]),
                        config=GRAPH_CONFIG,
                    ),
                ]
            ),
        ],
        className="page",
    )


def explore_page(bundle: AnimalLexBundle, selected_species: str | None) -> html.Main:
    all_options = species_options(bundle)
    species = [option["value"] for option in all_options]
    selected = selected_species if selected_species in species else species[0]
    options = species_options(bundle, selected_species=selected)
    return html.Main(
        [
            html.Aside(
                [
                    html.H2("Species"),
                    dcc.Input(
                        id="species-search",
                        type="search",
                        placeholder="Search species",
                        debounce=False,
                        className="species-search",
                    ),
                    html.Div(
                        f"{len(options):,} species",
                        id="species-search-count",
                        className="species-search-count",
                    ),
                    dcc.RadioItems(
                        id="species-select",
                        options=options,
                        value=selected,
                        className="species-list",
                    ),
                ],
                className="species-sidebar",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.Label("Filter calls", htmlFor="call-filter"),
                            dcc.Input(
                                id="call-filter",
                                type="search",
                                placeholder="Call name or keyword",
                                debounce=True,
                            ),
                        ],
                        className="field",
                    ),
                    html.Div(id="explore-summary", className="lede explore-summary"),
                    html.Div(id="explore-table"),
                ],
                className="explore-content",
            ),
        ],
        className="explore-page",
    )


def species_options(
    bundle: AnimalLexBundle,
    query: str | None = None,
    selected_species: str | None = None,
) -> list[dict[str, str]]:
    rows = bundle.analysis["overview"]["species_counts"]
    species = [
        {
            "common_name": str(row.get("common_name") or row["species"]),
            "scientific_name": str(row["species"]),
        }
        for row in rows
    ]
    search = (query or "").strip().casefold()
    if search:
        species = [
            row
            for row in species
            if search in row["common_name"].casefold()
            or search in row["scientific_name"].casefold()
        ]
    species.sort(key=lambda row: (row["common_name"].casefold(), row["scientific_name"]))
    if not search and selected_species:
        species.sort(key=lambda row: row["scientific_name"] != selected_species)
    return [
        {
            "label": f"{row['common_name']} ({row['scientific_name']})",
            "value": row["scientific_name"],
        }
        for row in species
    ]


def repertoire_table(calls: list[dict[str, Any]]) -> dash_table.DataTable:
    rows = [
        {
            "call_name": call["call_name"],
            "acoustic": call["acoustic_description"],
            "acoustic_keywords": ", ".join(call["acoustic_keywords"]),
            "semantic": call["semantic_description"],
            "semantic_keywords": ", ".join(call["semantic_keywords"]),
        }
        for call in calls
    ]
    return dash_table.DataTable(
        data=rows,
        columns=[
            {"name": "Call", "id": "call_name"},
            {"name": "Acoustic description", "id": "acoustic"},
            {"name": "Acoustic keywords", "id": "acoustic_keywords"},
            {"name": "Semantic description", "id": "semantic"},
            {"name": "Semantic keywords", "id": "semantic_keywords"},
        ],
        page_size=25,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left",
            "whiteSpace": "normal",
            "height": "auto",
            "minWidth": "140px",
            "maxWidth": "360px",
            "padding": "12px",
            "fontFamily": "Inter, system-ui, sans-serif",
        },
        style_header={"fontWeight": "600", "backgroundColor": "#eaf2f2"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8fbfb"}
        ],
    )


def call_display_labels(bundle: AnimalLexBundle) -> dict[str, str]:
    common_names = {
        row["species"]: row.get("common_name") or row["species"]
        for row in bundle.analysis["overview"]["species_counts"]
    }
    return {
        call["call_id"]: f"{call['call_name']} ({common_names[call['species']]})"
        for call in bundle.calls
    }


def species_display_labels(bundle: AnimalLexBundle) -> dict[str, str]:
    return {
        row["species"]: row.get("common_name") or row["species"]
        for row in bundle.analysis["overview"]["species_counts"]
    }


def analysis_page(bundle: AnimalLexBundle, mode: str) -> html.Main:
    coverage = bundle.analysis["coverage"]
    default_group = coverage["default_group"]
    default_threshold = coverage["default_threshold"]
    coverage_groups = coverage["groups"]
    return html.Main(
        [
            _research_details(bundle, mode),
            html.Section(
                [
                    html.H2("Form to meaning correlation"),
                    dcc.Graph(
                        id="form-meaning-chart",
                        figure=form_meaning_chart(
                            bundle.analysis["form_meaning"], call_display_labels(bundle)
                        ),
                        config=GRAPH_CONFIG,
                    )
                ]
            ),
            html.Section(
                [
                    html.H2("Cross species semantic and acoustic overlap"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        "Taxonomic group", htmlFor="coverage-group"
                                    ),
                                    dcc.Dropdown(
                                        id="coverage-group",
                                        options=[
                                            {
                                                "label": (
                                                    f"{group['label']} "
                                                    f"({group['n_species']} species)"
                                                ),
                                                "value": key,
                                            }
                                            for key, group in coverage_groups.items()
                                        ],
                                        value=default_group,
                                        clearable=False,
                                    ),
                                ],
                                className="field",
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Embedding similarity threshold",
                                        htmlFor="coverage-threshold",
                                    ),
                                    dcc.Slider(
                                        id="coverage-threshold",
                                        min=0,
                                        max=1,
                                        step=0.01,
                                        value=default_threshold,
                                        marks={
                                            value: f"{value:.1f}"
                                            for value in [0, 0.2, 0.4, 0.6, 0.8, 1]
                                        },
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                    ),
                                ],
                                className="field coverage-threshold-field",
                            ),
                        ],
                        className="coverage-controls",
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id="semantic-coverage-chart",
                                figure=coverage_chart(
                                    coverage_groups[default_group],
                                    "semantic",
                                    default_threshold,
                                    coverage["thresholds"],
                                ),
                                config=GRAPH_CONFIG,
                            ),
                            dcc.Graph(
                                id="acoustic-coverage-chart",
                                figure=coverage_chart(
                                    coverage_groups[default_group],
                                    "acoustic",
                                    default_threshold,
                                    coverage["thresholds"],
                                ),
                                config=GRAPH_CONFIG,
                            ),
                        ],
                        className="chart-grid coverage-grid",
                    ),
                ]
            ),
            html.Section(
                [
                    html.H2("Species pair acoustic and semantic correlation"),
                    dcc.Graph(
                        id="species-matrix-chart",
                        figure=species_matrix_chart(
                            bundle.analysis["species_matrix"], species_display_labels(bundle)
                        ),
                        config=GRAPH_CONFIG,
                    ),
                    html.P(
                        "Colored axis bands and diagonal outlines identify taxonomic "
                        "classes; species are clustered within each class.",
                        className="lede",
                    ),
                    html.Div(
                        "Select a matrix cell for details.",
                        id="species-matrix-detail",
                        className="selection-detail",
                    ),
                ]
            ),
            html.Section(
                [
                    html.H2("Within-species acoustic and semantic keyword association"),
                    dcc.Graph(
                        id="pmi-chart",
                        figure=pmi_chart(bundle.analysis["pmi"]),
                        config=GRAPH_CONFIG,
                    ),
                    html.Div("Select a heatmap cell for details.", id="pmi-detail", className="selection-detail"),
                ]
            ),
        ],
        className="page",
    )


def missing_bundle_page() -> html.Main:
    return html.Main(
        [
            html.H2("Build the AnimalLex analysis bundle"),
            html.P("The app does not calculate analyses when it starts."),
            html.Pre("uv run python analyses/build_animallex_release.py"),
        ],
        className="page empty-state",
    )
