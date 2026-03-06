import streamlit as st


def _color(value: float) -> str:
    if value >= 0.80:
        return "#16a34a"
    elif value >= 0.50:
        return "#d97706"
    else:
        return "#dc2626"


def render_metrics_row(
    precision: float,
    recall: float,
    f1: float,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    threshold: float,
) -> None:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Positive Cutoff</div>
                <div class="metric-value" style="color:#1a56db">{threshold:.2f}</div>
                <div class="metric-caption">Prediction score cutoff for the PRQ</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        total_flagged = tp + fp
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value" style="color:{_color(precision)}">{precision:.3f}</div>
                <div class="metric-caption">Of {total_flagged} docs added to the PRQ, {tp} were coded relevant</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        total_responsive = tp + fn
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value" style="color:{_color(recall)}">{recall:.3f}</div>
                <div class="metric-caption">Found {tp} of {total_responsive} truly responsive docs — {fn} eluded review</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value" style="color:{_color(f1)}">{f1:.3f}</div>
                <div class="metric-caption">Harmonic mean of precision and recall</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
