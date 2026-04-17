"""
charts.py — Plotly chart helpers for Financial DNA visualizations.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils.ratio_engine import RATIO_LABELS, RATIO_NAMES


def radar_chart(company_ratios: dict, industry_avg: dict,
                company_name: str = "Company", industry_name: str = "Industry Avg",
                title: str = "Financial DNA Profile"):
    """
    Create a radar/spider chart comparing a company's ratios to industry average.
    """
    categories = RATIO_LABELS

    company_vals = [company_ratios.get(r, 0) for r in RATIO_NAMES]
    industry_vals = [industry_avg.get(r, 0) for r in RATIO_NAMES]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=company_vals + [company_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=company_name,
        line=dict(color='#636EFA', width=2),
        fillcolor='rgba(99, 110, 250, 0.15)',
    ))

    fig.add_trace(go.Scatterpolar(
        r=industry_vals + [industry_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=industry_name,
        line=dict(color='#EF553B', width=2, dash='dash'),
        fillcolor='rgba(239, 85, 59, 0.1)',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-0.5, 1.2]),
        ),
        showlegend=True,
        title=dict(text=title, x=0.5),
        height=500,
        margin=dict(t=80, b=40),
    )

    return fig


def dual_radar_chart(ratios_a: dict, ratios_b: dict,
                     name_a: str = "Company A", name_b: str = "Company B",
                     title: str = "Head-to-Head Comparison"):
    """
    Dual radar chart comparing two companies side by side.
    """
    categories = RATIO_LABELS

    vals_a = [ratios_a.get(r, 0) for r in RATIO_NAMES]
    vals_b = [ratios_b.get(r, 0) for r in RATIO_NAMES]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=vals_a + [vals_a[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=name_a,
        line=dict(color='#636EFA', width=2.5),
        fillcolor='rgba(99, 110, 250, 0.15)',
    ))

    fig.add_trace(go.Scatterpolar(
        r=vals_b + [vals_b[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=name_b,
        line=dict(color='#00CC96', width=2.5),
        fillcolor='rgba(0, 204, 150, 0.15)',
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-0.5, 1.2])),
        showlegend=True,
        title=dict(text=title, x=0.5),
        height=500,
        margin=dict(t=80, b=40),
    )

    return fig


def confusion_matrix_heatmap(cm, labels, title="Confusion Matrix"):
    """
    Create an interactive confusion matrix heatmap.
    """
    # Normalize by row (true labels)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Rate: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=600,
        width=700,
        yaxis=dict(autorange="reversed"),
    )

    return fig


def feature_importance_chart(importances: list, title="Feature Importance"):
    """
    Horizontal bar chart of feature importances.
    """
    df = pd.DataFrame({
        "Feature": RATIO_LABELS,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation='h',
        marker_color='#636EFA',
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Importance",
        height=400,
        margin=dict(l=150, t=60, b=40),
    )

    return fig


def peer_scatter_pca(df_all, target_ticker, peer_tickers, pca_model=None):
    """
    2D PCA scatter plot showing all companies, with the target and its peers highlighted.
    df_all must have columns: ticker, sector, PC1, PC2.
    """
    df = df_all.copy()
    df["group"] = "Other"
    df.loc[df["ticker"].isin(peer_tickers), "group"] = "Peer"
    df.loc[df["ticker"] == target_ticker, "group"] = "Target"

    color_map = {"Other": "#D3D3D3", "Peer": "#00CC96", "Target": "#EF553B"}
    size_map = {"Other": 5, "Peer": 10, "Target": 14}

    df["size"] = df["group"].map(size_map)

    fig = go.Figure()

    for group in ["Other", "Peer", "Target"]:
        subset = df[df["group"] == group]
        fig.add_trace(go.Scatter(
            x=subset["PC1"],
            y=subset["PC2"],
            mode='markers+text' if group != "Other" else 'markers',
            marker=dict(
                size=subset["size"],
                color=color_map[group],
                line=dict(width=1, color='white') if group != "Other" else dict(width=0),
            ),
            text=subset["ticker"] if group != "Other" else None,
            textposition="top center",
            name=group,
            hovertemplate="%{text}<extra></extra>" if group != "Other" else "%{customdata[0]}<extra></extra>",
            customdata=subset[["ticker"]].values if group == "Other" else None,
        ))

    fig.update_layout(
        title=dict(text=f"Financial DNA Map — {target_ticker} & Peers", x=0.5),
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        height=500,
        showlegend=True,
    )

    return fig
