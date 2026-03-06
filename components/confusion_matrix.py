import plotly.graph_objects as go


def build_confusion_matrix_figure(tp: int, fp: int, fn: int, tn: int) -> go.Figure:
    # z values control cell colors via the colorscale — use 4 distinct levels
    # Row 0 = Actually Responsive, Row 1 = Actually Non-Responsive
    # Col 0 = Predicted Responsive, Col 1 = Predicted Non-Responsive
    z = [[3, 0], [1, 2]]  # 3=TP green, 0=FN red, 1=FP orange, 2=TN gray

    cell_text = [
        [f"<b>FOUND IT</b><br>True Positive<br>n = {tp}", f"<b>MISSED</b><br>False Negative<br>n = {fn}"],
        [f"<b>FALSE ALARM</b><br>False Positive<br>n = {fp}", f"<b>CORRECTLY SKIPPED</b><br>True Negative<br>n = {tn}"],
    ]

    colorscale = [
        [0.0, "#dc2626"],   # 0 → red (FN)
        [0.33, "#dc2626"],
        [0.34, "#f97316"],  # 1 → orange (FP)
        [0.66, "#f97316"],
        [0.67, "#e5e7eb"],  # 2 → light gray (TN)
        [0.99, "#e5e7eb"],
        [1.0, "#16a34a"],   # 3 → green (TP)
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            text=cell_text,
            texttemplate="%{text}",
            textfont={"size": 13},
            colorscale=colorscale,
            showscale=False,
            xgap=4,
            ygap=4,
            zmin=0,
            zmax=3,
        )
    )

    fig.update_layout(
        title={
            "text": "What the Model Predicted vs. Reality",
            "font": {"size": 16, "color": "#1a202c"},
            "x": 0.5,
            "xanchor": "center",
        },
        height=380,
        margin={"l": 120, "r": 20, "t": 60, "b": 80},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis={
            "tickvals": [0, 1],
            "ticktext": ["<b>Predicted<br>Responsive</b>", "<b>Predicted<br>Non-Responsive</b>"],
            "tickfont": {"size": 12},
            "side": "bottom",
        },
        yaxis={
            "tickvals": [0, 1],
            "ticktext": ["<b>Actually<br>Responsive</b>", "<b>Actually<br>Non-Responsive</b>"],
            "tickfont": {"size": 12},
            "autorange": "reversed",
        },
    )

    return fig
