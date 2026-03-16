import numpy as np
import plotly.graph_objects as go


BASELINE = 600 / 3000  # 0.20 — random guessing precision at actual prevalence


def _gauss_smooth(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply a Gaussian kernel to a 1-D histogram array for smooth curves."""
    radius = int(3.0 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return np.maximum(0.0, np.convolve(arr.astype(float), kernel, mode="same"))


def build_score_distribution_figure(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    coded_mask: np.ndarray,
    cutoff: float,
    doc_scale: float = 1.0,
) -> go.Figure:
    """Stacked bar chart of rank scores — emulates the Relativity Review Center
    Rank Distribution chart.

    Coded counts are distributed proportionally across each class's score density
    curve (smooth expected-value approach), so yellow always appears at low ranks
    and blue at high ranks regardless of how many documents have been coded.
    """
    n_bins = 100
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 * 100  # 0–100 space
    bar_width = 100 / n_bins * 0.95

    nr_mask   = ground_truth == 0
    resp_mask = ground_truth == 1
    n_coded_nr   = int((coded_mask & nr_mask).sum())
    n_coded_resp = int((coded_mask & resp_mask).sum())

    # Score density of each ground-truth class under the current model — smoothed
    nr_density,   _ = np.histogram(scores[nr_mask],   bins=bin_edges)
    resp_density, _ = np.histogram(scores[resp_mask], bins=bin_edges)
    nr_density   = _gauss_smooth(nr_density)
    resp_density = _gauss_smooth(resp_density)
    total_density   = nr_density + resp_density

    # Expected coded counts per bin: scale each class's density by its coding fraction
    coded_nr   = nr_density   * (n_coded_nr   / max(nr_mask.sum(),   1))
    coded_resp = resp_density * (n_coded_resp / max(resp_mask.sum(), 1))
    remaining  = np.maximum(0.0, total_density - coded_nr - coded_resp)

    categories = [
        ("Docs Remaining", remaining,  "#a8c4de", "#7aa0c0"),
        ("Not Relevant",   coded_nr,   "#fbbf24", "#d97706"),
        ("Responsive",     coded_resp, "#1a56db", "#1e40af"),
    ]

    fig = go.Figure()
    for label, counts_f, fill, border in categories:
        counts = np.maximum(0, np.round(counts_f * doc_scale)).astype(int)
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=counts,
            name=label,
            marker_color=fill,
            marker_line_color=border,
            marker_line_width=0.5,
            width=bar_width,
            hovertemplate=f"<b>{label}</b><br>Rank: %{{x:.0f}}<br>Docs: %{{y:,}}<extra></extra>",
        ))

    # Positive Cutoff line (scaled to 0–100)
    fig.add_vline(
        x=cutoff * 100,
        line_dash="dash",
        line_color="#f97316",
        line_width=2,
        annotation_text=f"Cutoff = {cutoff:.0%}",
        annotation_position="top right",
        annotation_font={"size": 11, "color": "#f97316"},
    )

    fig.update_layout(
        barmode="stack",
        title={
            "text": "Rank Distribution",
            "font": {"size": 14, "color": "#1a202c"},
            "x": 0.5,
            "xanchor": "center",
        },
        height=420,
        margin={"l": 70, "r": 20, "t": 50, "b": 70},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        xaxis={
            "title": "Relevance Rank",
            "range": [0, 100],
            "gridcolor": "#e5e7eb",
            "zeroline": False,
            "tickvals": list(range(0, 101, 10)),
        },
        yaxis={
            "title": "Documents",
            "gridcolor": "#e5e7eb",
            "tickformat": ",",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.45,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 10},
        },
    )

    return fig


def build_bucket_bar_figure(b1: int, b2: int, b3: int, b4: int) -> go.Figure:
    """Vertical bar chart showing the 4 Review Center document buckets."""
    total = b1 + b2 + b3 + b4

    x_labels = [
        "Not<br>Relevant",
        "Responsive",
        "Uncoded Pred.<br>Not Relevant",
        "Uncoded<br>In PRQ",
    ]
    full_labels = [
        "Not Relevant",
        "Responsive",
        "Uncoded: Predicted Not Relevant",
        "Uncoded: In PRQ",
    ]
    values = [b1, b2, b3, b4]
    fills  = ["#fbbf24", "#1a56db", "#a8c4de", "#fed7aa"]
    borders = ["#d97706", "#1e40af", "#7aa0c0", "#f97316"]

    fig = go.Figure()

    for x, full, val, fill, border in zip(x_labels, full_labels, values, fills, borders):
        fig.add_trace(
            go.Bar(
                name=full,
                x=[x],
                y=[val],
                marker={"color": fill, "line": {"color": border, "width": 1}},
                text=[f"{val:,}<br>({val/total:.0%})"],
                textposition="outside",
                textfont={"size": 11, "color": "#1a202c"},
                hovertemplate=f"<b>{full}</b><br>{val:,} documents ({val/total:.1%})<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        title={
            "text": "Document Buckets",
            "font": {"size": 14, "color": "#1a202c"},
            "x": 0.5,
            "xanchor": "center",
        },
        height=420,
        margin={"l": 20, "r": 20, "t": 50, "b": 60},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        xaxis={"tickfont": {"size": 11}},
        yaxis={"title": "Documents", "gridcolor": "#e5e7eb", "tickformat": ","},
        showlegend=False,
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
