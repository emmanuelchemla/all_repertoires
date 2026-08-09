from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = [
    "#15616d",
    "#ff7d00",
    "#6a4c93",
    "#2a9d8f",
    "#bc4749",
    "#577590",
    "#8c6d31",
    "#e56b6f",
    "#3a86a8",
    "#7f8f3a",
    "#9c6644",
    "#5f6caf",
    "#6b705c",
]

CLASS_COLORS = {
    "Amphibia": "#2a9d8f",
    "Aves": "#e88d14",
    "Mammalia": "#6a4c93",
}

KEYWORD_GROUP_COLORS = {
    "frequency": "#287f8b",
    "spectral": "#725a9a",
    "temporal": "#d47b29",
    "amplitude": "#b65358",
    "variation": "#607d8b",
    "social cohesion": "#267f75",
    "agonistic": "#b94f55",
    "danger": "#dc7735",
    "distress and care": "#9a6280",
    "reproduction": "#c0658a",
    "resources": "#9a7626",
    "territorial spacing": "#628247",
    "identity and attention": "#477a9f",
    "metacommunicative": "#76629b",
    "combinatorial": "#66747c",
}

# Keep these display groupings local to the chart so the web app can start before
# the analysis package is added to its import path.
ACOUSTIC_GROUPS = {
    "frequency": {"high_frequency", "low_frequency", "frequency_modulated"},
    "spectral": {"tonal", "broadband", "noisy", "harmonic"},
    "temporal": {"short", "long", "abrupt", "repetitive", "pulsed", "multi_component"},
    "amplitude": {"loud", "quiet"},
    "variation": {"graded"},
}
SEMANTIC_GROUPS = {
    "social cohesion": {"contact", "group_coordination", "affiliation"},
    "agonistic": {"threat", "aggression", "submission"},
    "danger": {"alarm", "predator"},
    "distress and care": {"distress", "begging", "caregiving"},
    "reproduction": {"courtship", "mating"},
    "resources": {"food", "recruitment"},
    "territorial spacing": {"territorial", "spacing"},
    "identity and attention": {"identity", "attention"},
    "metacommunicative": {"play", "display"},
    "combinatorial": {"combinatorial"},
}


def _base_layout(fig: go.Figure, *, height: int = 440) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=30, r=20, t=55, b=45),
        font=dict(family="Inter, system-ui, sans-serif", color="#18323a"),
        hoverlabel=dict(namelength=-1),
    )
    return fig


def keyword_frequency_chart(
    rows: list[dict[str, Any]], title: str, *, limit: int
) -> go.Figure:
    selected = rows[:limit][::-1]
    groups = list(dict.fromkeys(row["group"] for row in selected))
    color_map = {group: COLORS[index % len(COLORS)] for index, group in enumerate(groups)}
    labels = [row["keyword"].replace("_", " ") for row in selected]
    fig = go.Figure()
    for group in groups:
        group_rows = [row for row in selected if row["group"] == group]
        fig.add_trace(
            go.Bar(
                x=[row["percent_calls"] for row in group_rows],
                y=[row["keyword"].replace("_", " ") for row in group_rows],
                orientation="h",
                marker_color=color_map[group],
                customdata=[row["count"] for row in group_rows],
                name=group.title(),
                hovertemplate=(
                    "%{y}<br>%{x:.1f}% of calls<br>%{customdata:,} calls"
                    f"<br>{group}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        barmode="overlay",
        legend=dict(title="Keyword group", orientation="h", yanchor="top", y=-0.2),
    )
    fig.update_xaxes(
        title="Calls with keyword (%)", rangemode="tozero", ticksuffix="%", gridcolor="#e5ecee"
    )
    fig.update_yaxes(title=None, categoryorder="array", categoryarray=labels)
    fig = _base_layout(fig, height=500)
    fig.update_layout(margin=dict(b=120))
    return fig


def call_count_distribution_chart(rows: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[row["n_calls"] for row in rows],
            y=[row["n_species"] for row in rows],
            width=0.9,
            marker_color=COLORS[0],
            hovertemplate=(
                "%{x} calls per species<br>%{y} species<extra></extra>"
            ),
        )
    )
    fig.update_layout(title="Distribution of repertoire sizes", showlegend=False, bargap=0.04)
    fig.update_xaxes(title="Calls per species", rangemode="tozero", gridcolor="#e5ecee")
    fig.update_yaxes(title="Species", rangemode="tozero", dtick=1, gridcolor="#e5ecee")
    return _base_layout(fig, height=250)


def species_counts_chart(rows: list[dict[str, Any]]) -> go.Figure:
    selected = list(reversed(rows))
    groups = list(dict.fromkeys(row["taxonomic_group"] for row in rows))
    color_map = {group: COLORS[index % len(COLORS)] for index, group in enumerate(groups)}
    labels = [row["common_name"] for row in selected]
    fig = go.Figure()
    for group in groups:
        group_rows = [row for row in selected if row["taxonomic_group"] == group]
        fig.add_trace(
            go.Bar(
                x=[row["n_calls"] for row in group_rows],
                y=[row["common_name"] for row in group_rows],
                orientation="h",
                marker_color=color_map[group],
                name=group,
                customdata=[
                    [row["species"], row["class"], row["order"], row["family"]]
                    for row in group_rows
                ],
                hovertemplate=(
                    "%{y} (<i>%{customdata[0]}</i>)<br>%{x:,} calls<br>"
                    "%{customdata[1]} · %{customdata[2]} · %{customdata[3]}"
                    f"<br>{group}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        barmode="overlay",
        legend=dict(title="Taxonomic group", orientation="h", yanchor="top", y=-0.03),
    )
    fig.update_xaxes(title="Calls", gridcolor="#e5ecee")
    fig.update_yaxes(
        title=None, tickfont=dict(size=10), categoryorder="array", categoryarray=labels
    )
    return _base_layout(fig, height=max(720, 18 * len(selected) + 150))


def form_meaning_chart(
    result: dict[str, Any], call_labels: dict[str, str]
) -> go.Figure:
    fig = go.Figure()
    symbols = ["circle", "diamond", "square"]
    for index, (group, values) in enumerate(result["groups"].items()):
        sample = values["sample"]
        if not sample:
            continue
        p_value = values["p"]
        p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
        fig.add_trace(
            go.Scattergl(
                x=[1 - row["acoustic_distance"] for row in sample],
                y=[1 - row["semantic_distance"] for row in sample],
                mode="markers",
                name=(
                    f"{group}, r = {values['r']:.2f}, {p_text}, "
                    f"n = {values['n_pairs']:,}"
                ),
                marker=dict(color=COLORS[index], size=6, opacity=0.34, symbol=symbols[index]),
                customdata=[
                    [
                        call_labels.get(row["call_id_1"], row["call_id_1"]),
                        call_labels.get(row["call_id_2"], row["call_id_2"]),
                    ]
                    for row in sample
                ],
                hovertemplate=(
                    "Acoustic similarity %{x:.3f}<br>Semantic similarity %{y:.3f}"
                    "<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[
                    1 - values["intercept"] - values["slope"],
                    1 - values["intercept"],
                ],
                mode="lines",
                line=dict(color=COLORS[index], width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.update_layout(legend_title="Relationship")
    fig.update_xaxes(title="Acoustic similarity", range=[0, 1], gridcolor="#e5ecee")
    fig.update_yaxes(title="Semantic similarity", range=[0, 1], gridcolor="#e5ecee")
    return _base_layout(fig, height=620)


def prediction_chart(result: dict[str, Any]) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cosine similarity ↑", "Mean reciprocal rank ↑"),
        horizontal_spacing=0.12,
    )
    conditions = result["conditions"]
    condition_keys = [condition["key"] for condition in conditions]
    condition_labels = [condition["label"] for condition in conditions]
    model_colors = {
        "random": "#9aa9ad",
        "retrieval": "#e88d14",
        "ridge": "#15616d",
    }
    for column, metric in enumerate(("cosine", "mrr"), start=1):
        metric_highs: list[float] = []
        for model, model_label in result["models"].items():
            rows = [result["results"][key][model][metric] for key in condition_keys]
            means = [row["mean"] for row in rows]
            lows = [row["ci_low"] for row in rows]
            highs = [row["ci_high"] for row in rows]
            metric_highs.extend(highs)
            fig.add_trace(
                go.Bar(
                    x=condition_labels,
                    y=means,
                    name=model_label,
                    legendgroup=model,
                    showlegend=column == 1,
                    marker_color=model_colors[model],
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=[high - mean for high, mean in zip(highs, means)],
                        arrayminus=[mean - low for mean, low in zip(means, lows)],
                        color="#334e55",
                        thickness=1.2,
                        width=4,
                    ),
                    customdata=np.column_stack(
                        [lows, highs, [condition["n_folds"] for condition in conditions]]
                    ),
                    hovertemplate=(
                        "%{x}<br>" + model_label + f"<br>{result['metrics'][metric]} = "
                        "%{y:.3f}<br>95% CI %{customdata[0]:.3f}–%{customdata[1]:.3f}"
                        "<br>%{customdata[2]:.0f} outer folds<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        padded_max = max(metric_highs, default=0.0) * 1.12
        axis_max = min(1.0, max(0.1, np.ceil(padded_max / 0.05) * 0.05))
        fig.update_yaxes(
            range=[0, axis_max], gridcolor="#e5ecee", row=1, col=column
        )
        fig.update_xaxes(tickangle=0, row=1, col=column)
    fig.update_layout(
        barmode="group",
        bargap=0.2,
        bargroupgap=0.06,
        legend=dict(orientation="h", yanchor="top", y=-0.17, xanchor="center", x=0.5),
    )
    fig = _base_layout(fig, height=500)
    fig.update_layout(margin=dict(l=55, r=25, t=70, b=115))
    return fig


def coverage_chart(
    group: dict[str, Any], dimension: str, threshold: float, thresholds: list[float]
) -> go.Figure:
    threshold_index = thresholds.index(round(float(threshold), 2))
    percentages = group[dimension]["percent_calls"][threshold_index]
    counts = group[dimension]["n_calls"][threshold_index]
    species_percentages = [25, 50, 75, 100]
    color = COLORS[1] if dimension == "semantic" else COLORS[0]
    fig = go.Figure(
        go.Bar(
            x=species_percentages,
            y=percentages,
            marker_color=color,
            customdata=[
                [
                    group["minimum_species"][index],
                    counts[index],
                    group["n_calls"],
                    group["n_target_species"],
                ]
                for index in range(len(species_percentages))
            ],
            text=[f"{value:.1f}%" for value in percentages],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y:.1f}% of calls (%{customdata[1]:,} of %{customdata[2]:,})"
                "<br>Represented in at least %{x}% of species"
                "<br>%{customdata[0]} of %{customdata[3]} eligible target species"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{dimension.title()} overlap",
        showlegend=False,
    )
    fig.update_xaxes(
        title="% of eligible target species represented",
        tickvals=[25, 50, 75, 100],
        ticktext=["25%", "50%", "75%", "100%"],
        range=[12.5, 112.5],
        gridcolor="#e5ecee",
    )
    fig.update_yaxes(
        title="Calls",
        ticksuffix="%",
        range=[0, 108],
        gridcolor="#e5ecee",
    )
    return _base_layout(fig, height=440)


def species_matrix_chart(
    result: dict[str, Any], species_labels: dict[str, str]
) -> go.Figure:
    matrix = np.asarray(result["matrix"], dtype=float)
    counts = np.asarray(result["pair_counts"], dtype=int)
    labels = [species_labels.get(species, species) for species in result["species"]]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            customdata=counts,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="r"),
            hovertemplate=(
                "%{y} × %{x}<br>r = %{z:.3f}<br>%{customdata:,} call pairs<extra></extra>"
            ),
        )
    )
    fig = _base_layout(fig, height=980)
    fig.update_layout(margin=dict(l=32, r=24, t=94, b=170))
    fig.update_xaxes(
        domain=[0.22, 1],
        tickangle=55,
        tickfont=dict(size=8),
        automargin=False,
    )
    fig.update_yaxes(
        domain=[0, 0.94],
        tickfont=dict(size=8),
        autorange="reversed",
        automargin=False,
    )
    classes = result.get("classes", [])
    n_species = len(classes)
    start = 0
    while start < len(classes):
        end = start + 1
        while end < len(classes) and classes[end] == classes[start]:
            end += 1
        if classes[start] and end - start >= 2:
            class_name = classes[start]
            color = CLASS_COLORS.get(class_name, COLORS[0])
            fig.add_shape(
                type="rect",
                x0=start - 0.5,
                x1=end - 0.5,
                y0=start - 0.5,
                y1=end - 0.5,
                line=dict(color=color, width=2.4),
                fillcolor="rgba(0, 0, 0, 0)",
                layer="above",
            )
            x0 = 0.22 + 0.78 * start / n_species
            x1 = 0.22 + 0.78 * end / n_species
            y0 = 0.94 * (1 - end / n_species)
            y1 = 0.94 * (1 - start / n_species)
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="paper",
                x0=x0,
                x1=x1,
                y0=0.955,
                y1=0.969,
                line=dict(width=0),
                fillcolor=color,
                layer="above",
            )
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.045,
                x1=0.057,
                y0=y0,
                y1=y1,
                line=dict(width=0),
                fillcolor=color,
                layer="above",
            )
            fig.add_annotation(
                x=(x0 + x1) / 2,
                y=0.982,
                xref="paper",
                yref="paper",
                text=class_name,
                showarrow=False,
                font=dict(color=color, size=11),
                xanchor="center",
                yanchor="bottom",
            )
            fig.add_annotation(
                x=0.037,
                y=(y0 + y1) / 2,
                xref="paper",
                yref="paper",
                text=class_name,
                showarrow=False,
                font=dict(color=color, size=11),
                xanchor="right",
                yanchor="middle",
            )
        start = end
    return fig


def pmi_chart(result: dict[str, Any]) -> go.Figure:
    acoustic_keywords = result["acoustic_keywords"]
    semantic_keywords = result["semantic_keywords"]
    acoustic_group_by_keyword = {
        keyword: group for group, keywords in ACOUSTIC_GROUPS.items() for keyword in keywords
    }
    semantic_group_by_keyword = {
        keyword: group for group, keywords in SEMANTIC_GROUPS.items() for keyword in keywords
    }
    acoustic_order = sorted(
        range(len(acoustic_keywords)),
        key=lambda index: (
            list(ACOUSTIC_GROUPS).index(acoustic_group_by_keyword[acoustic_keywords[index]]),
            acoustic_keywords[index],
        ),
    )
    semantic_order = sorted(
        range(len(semantic_keywords)),
        key=lambda index: (
            list(SEMANTIC_GROUPS).index(semantic_group_by_keyword[semantic_keywords[index]]),
            semantic_keywords[index],
        ),
    )
    ordered_acoustic = [acoustic_keywords[index] for index in acoustic_order]
    ordered_semantic = [semantic_keywords[index] for index in semantic_order]
    customdata = [
        [
            [
                result["joint_counts"][i][j],
                result["expected_counts"][i][j],
                result["p_values"][i][j],
                result["q_values"][i][j],
            ]
            for j in semantic_order
        ]
        for i in acoustic_order
    ]
    fig = go.Figure(
        go.Heatmap(
            z=np.asarray(result["matrix"])[np.ix_(acoustic_order, semantic_order)],
            x=[value.replace("_", " ") for value in ordered_semantic],
            y=[value.replace("_", " ") for value in ordered_acoustic],
            customdata=customdata,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="PMI (bits)"),
            hovertemplate=(
                "%{y} × %{x}<br>PMI = %{z:.2f} bits"
                "<br>%{customdata[0]:,} matching calls"
                "<br>%{customdata[1]:.2f} expected after global shuffling"
                "<br>permutation p = %{customdata[2]:.3g}"
                "<br>FDR q = %{customdata[3]:.3g}"
                "<extra></extra>"
            ),
        )
    )
    for axis, keywords, group_by_keyword in (
        ("x", ordered_semantic, semantic_group_by_keyword),
        ("y", ordered_acoustic, acoustic_group_by_keyword),
    ):
        start = 0
        while start < len(keywords):
            group = group_by_keyword[keywords[start]]
            end = start + 1
            while end < len(keywords) and group_by_keyword[keywords[end]] == group:
                end += 1
            color = KEYWORD_GROUP_COLORS[group]
            if axis == "x":
                fig.add_shape(
                    type="rect",
                    x0=start / len(keywords),
                    x1=end / len(keywords),
                    y0=1.012,
                    y1=1.028,
                    xref="paper",
                    yref="paper",
                    fillcolor=color,
                    line_width=0,
                )
                fig.add_annotation(
                    x=(start + end) / (2 * len(keywords)),
                    y=1.045,
                    xref="paper",
                    yref="paper",
                    text=group.title(),
                    textangle=-90,
                    showarrow=False,
                    font=dict(color=color, size=10),
                    xanchor="center",
                    yanchor="bottom",
                )
            else:
                y0 = 1 - end / len(keywords)
                y1 = 1 - start / len(keywords)
                fig.add_shape(
                    type="rect",
                    x0=-0.225,
                    x1=-0.212,
                    y0=y0,
                    y1=y1,
                    xref="paper",
                    yref="paper",
                    fillcolor=color,
                    line_width=0,
                )
                fig.add_annotation(
                    x=-0.242,
                    y=(y0 + y1) / 2,
                    xref="paper",
                    yref="paper",
                    text=group.title(),
                    showarrow=False,
                    font=dict(color=color, size=10),
                    xanchor="right",
                    yanchor="middle",
                )
            start = end
    fig.update_xaxes(tickangle=45, title="Semantic keywords")
    fig.update_yaxes(autorange="reversed")
    fig = _base_layout(fig, height=760)
    fig.update_layout(margin=dict(l=320, r=90, t=170, b=145))
    return fig
