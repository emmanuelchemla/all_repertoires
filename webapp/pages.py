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
    prediction_chart,
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
                    html.Dt("Prediction folds"),
                    html.Dd(f"{config.get('prediction_folds', 10):,}"),
                    html.Dt("Prediction bootstrap samples"),
                    html.Dd(f"{config.get('prediction_bootstrap_samples', 0):,}"),
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
            "confidence": str(call.get("confidence", "")).title(),
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
            {"name": "Confidence", "id": "confidence"},
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
        style_cell_conditional=[
            {
                "if": {"column_id": "call_name"},
                "minWidth": "108px",
                "width": "108px",
                "maxWidth": "160px",
            },
            {
                "if": {"column_id": "confidence"},
                "minWidth": "104px",
                "width": "104px",
                "maxWidth": "104px",
                "textAlign": "center",
                "paddingLeft": "8px",
                "paddingRight": "8px",
                "fontSize": "13px",
            },
        ],
        style_header={"fontWeight": "600", "backgroundColor": "#eaf2f2"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8fbfb"},
            {
                "if": {"filter_query": '{confidence} = "Low"', "column_id": "confidence"},
                "backgroundColor": "#fdebea",
                "color": "#8a3834",
                "fontWeight": "600",
            },
            {
                "if": {"filter_query": '{confidence} = "Medium"', "column_id": "confidence"},
                "backgroundColor": "#fff4d6",
                "color": "#765812",
                "fontWeight": "600",
            },
            {
                "if": {"filter_query": '{confidence} = "High"', "column_id": "confidence"},
                "backgroundColor": "#e8f4ec",
                "color": "#2f6b43",
                "fontWeight": "600",
            },
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


def _motif_title(keywords: list[str]) -> str:
    signature = frozenset(keywords)
    preferred = {
        frozenset({"alarm", "predator"}): "Predator alarm calls",
        frozenset({"alarm", "threat"}): "Alarm and threat calls",
        frozenset({"aggression", "threat"}): "Aggressive threat calls",
        frozenset({"begging", "caregiving"}): "Begging and caregiving calls",
        frozenset({"affiliation", "contact", "group_coordination"}): (
            "Contact and coordination calls"
        ),
    }
    if signature in preferred:
        return preferred[signature]
    if not keywords:
        return "Acoustic–semantic motif"
    concise = " · ".join(
        keyword.replace("_", " ").title() for keyword in keywords[:2]
    )
    return f"{concise} motif" if len(keywords) > 2 else concise


def _species_emoji(member: dict[str, Any]) -> str:
    order = member.get("order", "")
    if member.get("class") == "Amphibia":
        return "🐸"
    if member.get("class") == "Aves":
        return "🐦"
    return {
        "Primates": "🐒",
        "Carnivora": "🐾",
        "Rodentia": "🐭",
        "Chiroptera": "🦇",
        "Cetacea": "🐋",
        "Artiodactyla": "🦌",
        "Perissodactyla": "🐴",
    }.get(order, "🐾")


def motif_carousel_card(motif: dict[str, Any]) -> html.Article:
    title = _motif_title(motif["shared_semantic_keywords"])
    if motif["n_classes"] > 1 and motif["shared_semantic_keywords"] == ["distress"]:
        title = "Cross-class distress calls"
    taxonomic_summary = (
        f"{motif['n_species']} species · {motif['n_families']} families · "
        f"{motif['n_orders']} orders"
    )
    if motif["n_classes"] > 1:
        taxonomic_summary += f" · {motif['n_classes']} classes"
    return html.Article(
        [
            html.Header(
                [
                    html.Div(
                        [
                            html.H2(title),
                            html.P(taxonomic_summary),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span(
                                f"Acoustic ≥ {motif['minimum_acoustic_similarity']:.2f}"
                            ),
                            html.Span(
                                f"Semantic ≥ {motif['minimum_semantic_similarity']:.2f}"
                            ),
                        ],
                        className="motif-similarities",
                    ),
                ],
                className="motif-header",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Meaning", className="motif-keyword-label"),
                            *[
                                html.Span(keyword.replace("_", " "))
                                for keyword in motif["shared_semantic_keywords"]
                            ],
                        ],
                        className="motif-keyword-row semantic-keywords",
                    ),
                    html.Div(
                        [
                            html.Span("Form", className="motif-keyword-label"),
                            *[
                                html.Span(keyword.replace("_", " "))
                                for keyword in motif["shared_acoustic_keywords"]
                            ],
                        ],
                        className="motif-keyword-row acoustic-keywords",
                    ),
                ],
                className="motif-shared-keywords",
            ),
            html.Ul(
                [
                    html.Li(
                        [
                            html.Span(
                                _species_emoji(member),
                                className="motif-species-emoji",
                                **{"aria-hidden": "true"},
                            ),
                            html.Div(
                                [
                                    html.Strong(member["common_name"]),
                                    html.Em(member["species"]),
                                ],
                                className="motif-species",
                            ),
                            html.Div(member["call_name"], className="motif-call-name"),
                            html.Div(
                                [
                                    *[
                                        html.Span(
                                            keyword.replace("_", " "),
                                            className="semantic-chip",
                                        )
                                        for keyword in member["semantic_keywords"][:3]
                                        if keyword != "attention"
                                    ],
                                    *[
                                        html.Span(
                                            keyword.replace("_", " "),
                                            className="acoustic-chip",
                                        )
                                        for keyword in member["acoustic_keywords"][:3]
                                    ],
                                ],
                                className="motif-member-keywords",
                            ),
                        ],
                        title=(
                            f"Meaning: {member['semantic_description']}\n\n"
                            f"Form: {member['acoustic_description']}"
                        ),
                        tabIndex=0,
                    )
                    for member in motif["members"]
                ],
                className="motif-members",
            ),
        ],
        className="motif-card motif-carousel-card",
        id=motif["motif_id"],
    )


def translations_page(bundle: AnimalLexBundle, mode: str) -> html.Main:
    result = bundle.analysis["motifs"]
    criteria = result["criteria"]
    motifs = result["motifs"]
    initial_card: Any = (
        motif_carousel_card(motifs[0])
        if motifs
        else html.Div(
            "No motifs meet these parameters for the selected confidence level.",
            className="motif-empty",
        )
    )
    return html.Main(
        [
            dcc.Store(id="motif-results", data=result),
            dcc.Store(id="motif-index", data=0),
            html.Header(
                [
                    html.Div("Translations", className="page-kicker"),
                    html.H1("Cross-species motifs"),
                    html.P(
                        "Calls from different species that repeatedly align in both "
                        "described acoustic form and documented communicative function.",
                        className="lede",
                    ),
                ],
                className="translations-intro",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.Label("Acoustic threshold", htmlFor="motif-acoustic-threshold"),
                            dcc.Slider(
                                id="motif-acoustic-threshold",
                                min=0.5,
                                max=0.9,
                                step=0.01,
                                value=criteria["minimum_acoustic_similarity"],
                                marks={value: f"{value:.1f}" for value in [0.5, 0.6, 0.7, 0.8, 0.9]},
                                tooltip={"placement": "bottom", "always_visible": True},
                                updatemode="mouseup",
                                allow_direct_input=False,
                            ),
                        ],
                        className="motif-control",
                    ),
                    html.Div(
                        [
                            html.Label("Semantic threshold", htmlFor="motif-semantic-threshold"),
                            dcc.Slider(
                                id="motif-semantic-threshold",
                                min=0.5,
                                max=0.9,
                                step=0.01,
                                value=criteria["minimum_semantic_similarity"],
                                marks={value: f"{value:.1f}" for value in [0.5, 0.6, 0.7, 0.8, 0.9]},
                                tooltip={"placement": "bottom", "always_visible": True},
                                updatemode="mouseup",
                                allow_direct_input=False,
                            ),
                        ],
                        className="motif-control",
                    ),
                    html.Div(
                        [
                            html.Label("Minimum species", htmlFor="motif-minimum-species"),
                            dcc.Slider(
                                id="motif-minimum-species",
                                min=3,
                                max=10,
                                step=1,
                                value=criteria["minimum_species"],
                                marks={value: str(value) for value in range(3, 11)},
                                tooltip={"placement": "bottom", "always_visible": True},
                                updatemode="mouseup",
                                allow_direct_input=False,
                            ),
                        ],
                        className="motif-control",
                    ),
                ],
                className="motif-controls",
            ),
            _research_details(bundle, mode),
            html.Div(
                [
                    html.Div(
                        f"{result['n_motifs']:,} motifs · strongest minimum similarity first",
                        id="motif-result-summary",
                    ),
                    html.Div(
                        [
                            html.Button("←", id="motif-previous", n_clicks=0, **{"aria-label": "Previous motif"}),
                            html.Span(
                                f"1 / {result['n_motifs']:,}" if motifs else "0 / 0",
                                id="motif-position",
                                **{"aria-live": "polite"},
                            ),
                            html.Button("→", id="motif-next", n_clicks=0, **{"aria-label": "Next motif"}),
                        ],
                        className="motif-carousel-navigation",
                    ),
                ],
                className="motif-carousel-toolbar",
            ),
            html.Div(initial_card, id="motif-carousel-stage"),
            html.P(
                "These are correspondences among documented descriptions, not proven "
                "translations, evolutionary homologies, or evidence of identical animal "
                "experience. They are intended as focused hypotheses for comparison.",
                className="motif-caveat",
            ),
        ],
        className="page translations-page",
    )


def analysis_page(bundle: AnimalLexBundle, mode: str) -> html.Main:
    coverage = bundle.analysis["coverage"]
    default_group = coverage["default_group"]
    default_threshold = coverage["default_threshold"]
    coverage_groups = coverage["groups"]
    sections = [
        html.Section(
            [
                html.Div(
                    [
                        html.H2("Form to meaning correlation"),
                        dcc.RadioItems(
                            id="form-meaning-basis",
                            options=[
                                {"label": "Descriptions", "value": "descriptions"},
                                {"label": "Keywords", "value": "keywords"},
                            ],
                            value="descriptions",
                            inline=True,
                            className="filter-select analysis-method-select",
                        ),
                    ],
                    className="analysis-section-heading",
                ),
                html.P(
                    "Descriptions use cosine similarity of text embeddings; keywords "
                    "use Jaccard set similarity.",
                    className="analysis-method-note",
                ),
                dcc.Graph(
                    id="form-meaning-chart",
                    figure=form_meaning_chart(
                        bundle.analysis["form_meaning"], call_display_labels(bundle)
                    ),
                    config=GRAPH_CONFIG,
                ),
            ],
            id="form-meaning",
            **{"data-analysis-section": "true"},
        ),
        html.Section(
            [
                html.H2("Acoustic-to-semantic prediction across held-out taxa"),
                html.P(
                    "Models use each call's acoustic text embedding to predict the "
                    "semantic text embedding of that same call.",
                    className="prediction-note",
                ),
                dcc.Graph(
                    id="prediction-chart",
                    figure=prediction_chart(bundle.analysis["prediction"]),
                    config=GRAPH_CONFIG,
                ),
            ],
            id="prediction",
            **{"data-analysis-section": "true"},
        ),
        html.Section(
            [
                html.H2("Cross species semantic and acoustic overlap"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Taxonomic group", htmlFor="coverage-group"),
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
            ],
            id="cross-species-overlap",
            **{"data-analysis-section": "true"},
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
                html.Div(
                    id="species-matrix-detail",
                    className="selection-detail",
                ),
            ],
            id="species-matrix",
            **{"data-analysis-section": "true"},
        ),
        html.Section(
            [
                html.H2("Acoustic and semantic keyword association"),
                dcc.Graph(
                    id="pmi-chart",
                    figure=pmi_chart(bundle.analysis["pmi"]),
                    config=GRAPH_CONFIG,
                ),
                html.Div(id="pmi-detail", className="selection-detail"),
            ],
            id="keyword-association",
            **{"data-analysis-section": "true"},
        ),
    ]
    contents = [
        ("Form–meaning correlation", "#form-meaning"),
        ("Prediction across taxa", "#prediction"),
        ("Cross-species overlap", "#cross-species-overlap"),
        ("Species-pair matrix", "#species-matrix"),
        ("Keyword association", "#keyword-association"),
    ]
    return html.Main(
        [
            html.Aside(
                [
                    html.Div("On this page", className="analysis-toc-title"),
                    html.Nav(
                        [html.A(label, href=href) for label, href in contents],
                        **{"aria-label": "Analysis sections"},
                    ),
                ],
                className="analysis-toc",
            ),
            html.Div([_research_details(bundle, mode), *sections], className="analysis-content"),
        ],
        className="analysis-page",
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
