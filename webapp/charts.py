from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go


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
        title="Calls per species",
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
                x=[row["acoustic_distance"] for row in sample],
                y=[row["semantic_distance"] for row in sample],
                mode="markers",
                name=f"{group}, r = {values['r']:.2f}, {p_text}",
                marker=dict(color=COLORS[index], size=6, opacity=0.34, symbol=symbols[index]),
                customdata=[
                    [
                        call_labels.get(row["call_id_1"], row["call_id_1"]),
                        call_labels.get(row["call_id_2"], row["call_id_2"]),
                    ]
                    for row in sample
                ],
                hovertemplate=(
                    "Acoustic distance %{x:.3f}<br>Semantic distance %{y:.3f}"
                    "<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[values["intercept"], values["intercept"] + values["slope"]],
                mode="lines",
                line=dict(color=COLORS[index], width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.update_layout(title="Form to meaning correlation", legend_title="Relationship")
    fig.update_xaxes(title="Acoustic distance", range=[0, 1], gridcolor="#e5ecee")
    fig.update_yaxes(title="Semantic distance", range=[0, 1], gridcolor="#e5ecee")
    return _base_layout(fig, height=620)


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
    fig.update_layout(title="Species pair acoustic and semantic correlation")
    fig.update_xaxes(tickangle=55, tickfont=dict(size=8))
    fig.update_yaxes(tickfont=dict(size=8), autorange="reversed")
    return _base_layout(fig, height=900)


def pmi_chart(result: dict[str, Any]) -> go.Figure:
    customdata = [
        [
            [
                result["joint_counts"][i][j],
                result["p_values"][i][j],
                result["q_values"][i][j],
            ]
            for j in range(len(result["semantic_keywords"]))
        ]
        for i in range(len(result["acoustic_keywords"]))
    ]
    fig = go.Figure(
        go.Heatmap(
            z=result["matrix"],
            x=[value.replace("_", " ") for value in result["semantic_keywords"]],
            y=[value.replace("_", " ") for value in result["acoustic_keywords"]],
            customdata=customdata,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="PMI bits"),
            hovertemplate=(
                "%{y} × %{x}<br>PMI = %{z:.2f} bits"
                "<br>%{customdata[0]:,} matching calls"
                "<br>p = %{customdata[1]:.3g}<br>FDR q = %{customdata[2]:.3g}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(title="Acoustic and semantic keyword association")
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(autorange="reversed")
    return _base_layout(fig, height=650)
