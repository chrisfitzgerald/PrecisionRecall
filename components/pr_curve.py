import numpy as np
import plotly.graph_objects as go


BASELINE = 150 / 750  # 0.20 — random guessing precision at actual prevalence


def build_score_distribution_figure(
    responsive_scores: np.ndarray,
    non_responsive_scores: np.ndarray,
    cutoff: float,
) -> go.Figure:
    """Overlapping histogram of prediction scores coloured by ground truth.

    Shows how well-separated the two populations are — flat/overlapping in
    early reviews, bimodal in mature reviews.
    """
    fig = go.Figure()

    # Not relevant (gray, behind)
    fig.add_trace(
        go.Histogram(
            x=non_responsive_scores,
            name="Coded Not Relevant",
            nbinsx=25,
            marker_color="rgba(156, 163, 175, 0.55)",
            marker_line_color="rgba(107,114,128,0.7)",
            marker_line_width=0.5,
            hovertemplate="Score: %{x:.2f}<br>Count: %{y}<extra>Not Relevant</extra>",
        )
    )

    # Responsive (green, in front)
    fig.add_trace(
        go.Histogram(
            x=responsive_scores,
            name="Responsive",
            nbinsx=25,
            marker_color="rgba(22, 163, 74, 0.70)",
            marker_line_color="rgba(21,128,61,0.8)",
            marker_line_width=0.5,
            hovertemplate="Score: %{x:.2f}<br>Count: %{y}<extra>Responsive</extra>",
        )
    )

    # Positive Cutoff line
    fig.add_vline(
        x=cutoff,
        line_dash="dash",
        line_color="#f97316",
        line_width=2,
        annotation_text=f"Positive Cutoff = {cutoff:.2f}",
        annotation_position="top right",
        annotation_font={"size": 11, "color": "#f97316"},
    )

    # Shaded zones: left = predicted not relevant, right = predicted relevant
    fig.add_vrect(
        x0=0, x1=cutoff,
        fillcolor="rgba(249,115,22,0.04)",
        layer="below",
        line_width=0,
    )
    fig.add_vrect(
        x0=cutoff, x1=1,
        fillcolor="rgba(22,163,74,0.04)",
        layer="below",
        line_width=0,
    )

    fig.update_layout(
        barmode="overlay",
        title={
            "text": "Rank Distribution",
            "font": {"size": 14, "color": "#1a202c"},
            "x": 0.5,
            "xanchor": "center",
        },
        height=300,
        margin={"l": 50, "r": 20, "t": 50, "b": 60},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        xaxis={
            "title": "Prediction Score (0 = Not Relevant → 1 = Responsive)",
            "range": [0, 1],
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        yaxis={"title": "Document Count", "gridcolor": "#e5e7eb"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.35,
            "xanchor": "center",
            "x": 0.5,
        },
    )

    return fig


def build_bucket_bar_figure(b1: int, b2: int, b3: int, b4: int) -> go.Figure:
    """Horizontal stacked bar showing all 750 docs split across the 4 Review Center buckets."""
    total = b1 + b2 + b3 + b4

    fig = go.Figure()

    traces = [
        ("Coded Not Relevant", b1, "#e5e7eb", "#6b7280"),
        ("Coded Relevant", b2, "#16a34a", "#15803d"),
        ("Uncoded: Predicted Not Relevant", b3, "#fef3c7", "#d97706"),
        ("Uncoded: In PRQ", b4, "#1a56db", "#1e40af"),
    ]

    for label, val, fill, border in traces:
        fig.add_trace(
            go.Bar(
                name=label,
                y=["Documents"],
                x=[val],
                orientation="h",
                marker={"color": fill, "line": {"color": border, "width": 1}},
                text=[f"{val}<br>({val/total:.0%})"],
                textposition="inside",
                insidetextanchor="middle",
                textfont={"size": 11, "color": "#1a202c"},
                hovertemplate=f"<b>{label}</b><br>{val} documents ({val/total:.1%})<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        title={
            "text": "Document Buckets",
            "font": {"size": 14, "color": "#1a202c"},
            "x": 0.5,
            "xanchor": "center",
        },
        height=200,
        margin={"l": 20, "r": 20, "t": 45, "b": 100},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        xaxis={"title": "Document Count", "range": [0, total], "gridcolor": "#e5e7eb"},
        yaxis={"visible": False},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.35,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 10},
        },
        showlegend=True,
    )

    return fig



def build_pr_curve_figure(
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds: np.ndarray,
    current_threshold: float,
    current_precision: float,
    current_recall: float,
    optimal_threshold: float,
    optimal_precision: float,
    optimal_recall: float,
) -> go.Figure:
    fig = go.Figure()

    # 1. PR curve line with fill
    fig.add_trace(
        go.Scatter(
            x=recalls,
            y=precisions,
            mode="lines",
            name="PR Curve",
            line={"color": "#1a56db", "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(26, 86, 219, 0.08)",
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        )
    )

    # 2. Random baseline
    fig.add_hline(
        y=BASELINE,
        line_dash="dash",
        line_color="#9ca3af",
        line_width=1.5,
        annotation_text="Random guessing baseline",
        annotation_position="bottom right",
        annotation_font={"size": 11, "color": "#6b7280"},
    )

    # 3. F1-optimal point
    fig.add_trace(
        go.Scatter(
            x=[optimal_recall],
            y=[optimal_precision],
            mode="markers+text",
            name="Best F1",
            marker={
                "symbol": "star",
                "size": 14,
                "color": "#0d9488",
                "line": {"width": 1, "color": "#ffffff"},
            },
            text=["Best F1"],
            textposition="top right",
            textfont={"size": 11, "color": "#0d9488"},
            hovertemplate=(
                f"F1-Optimal Threshold: {optimal_threshold:.2f}<br>"
                "Precision: %{y:.3f}<br>Recall: %{x:.3f}<extra></extra>"
            ),
        )
    )

    # 4. Current threshold marker (orange dot — updates live)
    fig.add_trace(
        go.Scatter(
            x=[current_recall],
            y=[current_precision],
            mode="markers",
            name=f"Threshold = {current_threshold:.2f}",
            marker={
                "symbol": "circle",
                "size": 14,
                "color": "#f97316",
                "line": {"width": 2, "color": "#ffffff"},
            },
            hovertemplate=(
                f"Threshold: {current_threshold:.2f}<br>"
                "Precision: %{y:.3f}<br>Recall: %{x:.3f}<extra></extra>"
            ),
        )
    )

    # Zone annotations
    fig.add_annotation(
        x=0.12, y=0.93,
        text="<b>High Precision Zone</b><br>Fewer false alarms<br>but miss more docs",
        showarrow=False,
        font={"size": 10, "color": "#374151"},
        bgcolor="rgba(240,244,248,0.85)",
        bordercolor="#d1d5db",
        borderwidth=1,
        borderpad=5,
        align="center",
    )
    fig.add_annotation(
        x=0.88, y=0.12,
        text="<b>High Recall Zone</b><br>Find more responsive docs<br>but review more junk",
        showarrow=False,
        font={"size": 10, "color": "#374151"},
        bgcolor="rgba(240,244,248,0.85)",
        bordercolor="#d1d5db",
        borderwidth=1,
        borderpad=5,
        align="center",
    )

    fig.update_layout(
        title={
            "text": "Precision-Recall Curve",
            "font": {"size": 16, "color": "#1a202c"},
            "x": 0.5,
            "xanchor": "center",
        },
        height=420,
        margin={"l": 60, "r": 20, "t": 60, "b": 60},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        xaxis={
            "title": "Recall (% of Responsive Docs Found)",
            "range": [0, 1],
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        yaxis={
            "title": "Precision (% of Retrieved Docs That Are Responsive)",
            "range": [0, 1.05],
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.25,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 11},
        },
        showlegend=True,
    )

    return fig


def build_f1_curve_figure(
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds: np.ndarray,
    optimal_threshold: float,
) -> go.Figure:
    f1_scores = []
    for p, r in zip(precisions, recalls):
        if p + r == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * p * r / (p + r))
    f1_scores = np.array(f1_scores)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=f1_scores,
            mode="lines",
            name="F1 Score",
            line={"color": "#7c3aed", "width": 2.5},
            hovertemplate="Threshold: %{x:.2f}<br>F1: %{y:.3f}<extra></extra>",
        )
    )

    # Mark the peak
    peak_idx = int(np.argmax(f1_scores))
    fig.add_vline(
        x=optimal_threshold,
        line_dash="dash",
        line_color="#f97316",
        line_width=1.5,
        annotation_text=f"Optimal threshold: {optimal_threshold:.2f}",
        annotation_position="top",
        annotation_font={"size": 11, "color": "#f97316"},
    )
    fig.add_trace(
        go.Scatter(
            x=[thresholds[peak_idx]],
            y=[f1_scores[peak_idx]],
            mode="markers",
            name=f"Peak F1 = {f1_scores[peak_idx]:.3f}",
            marker={"symbol": "diamond", "size": 12, "color": "#f97316"},
            hovertemplate=f"Peak F1: {f1_scores[peak_idx]:.3f}<br>Threshold: {thresholds[peak_idx]:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "F1 Score at Each Threshold",
            "font": {"size": 16, "color": "#1a202c"},
            "x": 0.5,
            "xanchor": "center",
        },
        height=300,
        margin={"l": 60, "r": 20, "t": 60, "b": 60},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        xaxis={
            "title": "Threshold",
            "range": [0, 1],
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        yaxis={
            "title": "F1 Score",
            "range": [0, max(f1_scores) * 1.15],
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        legend={"font": {"size": 11}},
    )

    return fig
