import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from utils.calculations import (
    apply_threshold,
    compute_confusion_values,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_pr_curve,
    find_f1_optimal_threshold,
)
from components.confusion_matrix import build_confusion_matrix_figure
from components.pr_curve import (
    build_pr_curve_figure,
    build_f1_curve_figure,
    build_score_distribution_figure,
    build_bucket_bar_figure,
)
from components.metrics_cards import render_metrics_row

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Precision & Recall in Review Center | RelativityOne",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(Path(__file__).parent / "data" / "documents.csv")


@st.cache_data
def get_pr_curve_data(df_hash: int) -> dict:
    df = load_data()
    precisions, recalls, thresholds = compute_pr_curve(df["ground_truth"], df["confidence_score"])
    optimal_threshold = find_f1_optimal_threshold(df["ground_truth"], df["confidence_score"])
    # Find metrics at the optimal threshold
    opt_preds = apply_threshold(df["confidence_score"], optimal_threshold)
    opt_cv = compute_confusion_values(df["ground_truth"], opt_preds)
    optimal_precision = compute_precision(opt_cv["tp"], opt_cv["fp"])
    optimal_recall = compute_recall(opt_cv["tp"], opt_cv["fn"])
    return {
        "precisions": precisions,
        "recalls": recalls,
        "thresholds": thresholds,
        "optimal_threshold": optimal_threshold,
        "optimal_precision": optimal_precision,
        "optimal_recall": optimal_recall,
    }


df = load_data()
curve = get_pr_curve_data(0)

N_TOTAL = len(df)
N_RESPONSIVE = int(df["ground_truth"].sum())
N_NON_RESPONSIVE = N_TOTAL - N_RESPONSIVE

DISPLAY_TOTAL = 100_000
DISPLAY_SCALE = DISPLAY_TOTAL / N_TOTAL  # ≈ 133.33


def d(n: int) -> int:
    """Scale a raw simulation count up to the 100k display size."""
    return int(round(n * DISPLAY_SCALE))


# ── Helper: compute metrics for a given threshold ──────────────────────────────
def metrics_at(thr: float) -> dict:
    preds = apply_threshold(df["confidence_score"], thr)
    cv = compute_confusion_values(df["ground_truth"], preds)
    p = compute_precision(cv["tp"], cv["fp"])
    r = compute_recall(cv["tp"], cv["fn"])
    f = compute_f1(p, r)
    return {**cv, "precision": p, "recall": r, "f1": f}


@st.cache_data
def compute_training_stages() -> list:
    """Simulate three maturity stages of a Relativity Learning TAR workflow.

    Each stage blends the dataset's real prediction scores with uniform noise to
    model how the score distribution evolves as more documents are coded.
    Reviewers code high-scoring documents first (simulating PRQ-driven review).
    """
    df_local = load_data()
    ground_truth = df_local["ground_truth"].values
    real_scores = df_local["confidence_score"].values
    n = len(df_local)
    cutoff = 0.50

    # (stage name, n_coded, noise_weight, real_weight, seed)
    configs = [
        ("Early Review",   50,  0.80, 0.20, 101),
        ("Mid-Review",    180,  0.38, 0.62, 102),
        ("End of Review", 310,  0.07, 0.93, 103),
    ]

    stages = []
    for name, n_coded, noise_w, real_w, seed in configs:
        rng = np.random.default_rng(seed)
        noise = rng.uniform(0, 1, n)
        scores = np.clip(noise_w * noise + real_w * real_scores, 0.0, 1.0)

        # Simulate PRQ-driven coding: reviewers work highest-score docs first
        sorted_idx = np.argsort(scores)[::-1]
        coded_mask = np.zeros(n, dtype=bool)
        coded_mask[sorted_idx[:n_coded]] = True

        predicted_relevant = scores >= cutoff

        # Relativity's four document buckets
        b1 = int(( coded_mask & (ground_truth == 0)).sum())   # Coded Not Relevant
        b2 = int(( coded_mask & (ground_truth == 1)).sum())   # Coded Relevant
        b3 = int((~coded_mask & ~predicted_relevant).sum())   # Uncoded Predicted Not Relevant
        b4 = int((~coded_mask &  predicted_relevant).sum())   # Uncoded in PRQ

        # Elusion rate: of uncoded-predicted-not-relevant docs, how many are actually responsive?
        b3_eluded = int((~coded_mask & ~predicted_relevant & (ground_truth == 1)).sum())
        elusion_rate = b3_eluded / max(b3, 1)

        # Model-level precision and recall across the whole collection
        tp_all = int((predicted_relevant & (ground_truth == 1)).sum())
        fp_all = int((predicted_relevant & (ground_truth == 0)).sum())
        fn_all = int((~predicted_relevant & (ground_truth == 1)).sum())
        model_precision = tp_all / max(tp_all + fp_all, 1)
        model_recall = tp_all / max(tp_all + fn_all, 1)

        stages.append({
            "name": name,
            "n_coded": n_coded,
            "cutoff": cutoff,
            "scores": scores,
            "b1": b1, "b2": b2, "b3": b3, "b4": b4,
            "b3_eluded": b3_eluded,
            "elusion_rate": elusion_rate,
            "model_precision": model_precision,
            "model_recall": model_recall,
            "prq_size": b4,
        })

    return stages


@st.cache_data
def get_base_arrays() -> tuple:
    """Cached fixed noise + real score arrays for the live simulation slider."""
    df_local = load_data()
    noise = np.random.default_rng(42).uniform(0, 1, len(df_local))
    return (
        df_local["confidence_score"].values.copy(),
        df_local["ground_truth"].values.copy(),
        noise,
    )


def simulate_live_stage(n_coded: int, cutoff: float) -> dict:
    """Compute queue metrics for any n_coded / cutoff pair.

    Uses the same fixed noise array every call so the Rank Distribution
    evolves smoothly as the slider moves.
    """
    real_scores, ground_truth, noise = get_base_arrays()
    n = len(real_scores)

    noise_w = max(0.05, (1.0 - n_coded / n) ** 0.5) if n_coded > 0 else 1.0
    real_w = 1.0 - noise_w
    scores = np.clip(noise_w * noise + real_w * real_scores, 0.0, 1.0)

    sorted_idx = np.argsort(scores)[::-1]
    coded_mask = np.zeros(n, dtype=bool)
    if n_coded > 0:
        coded_mask[sorted_idx[:n_coded]] = True

    predicted_relevant = scores >= cutoff

    b1 = int(( coded_mask & (ground_truth == 0)).sum())
    b2 = int(( coded_mask & (ground_truth == 1)).sum())
    b3 = int((~coded_mask & ~predicted_relevant).sum())
    b4 = int((~coded_mask &  predicted_relevant).sum())
    b3_eluded = int((~coded_mask & ~predicted_relevant & (ground_truth == 1)).sum())

    tp = int((predicted_relevant & (ground_truth == 1)).sum())
    fp = int((predicted_relevant & (ground_truth == 0)).sum())
    fn = int((~predicted_relevant & (ground_truth == 1)).sum())

    return {
        "scores": scores,
        "b1": b1, "b2": b2, "b3": b3, "b4": b4,
        "b3_eluded": b3_eluded,
        "elusion_rate": b3_eluded / max(b3, 1),
        "relevance_rate": b2 / max(b1 + b2, 1),
        "prq_size": b4,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def queue_status_badge(elusion_rate: float, n_coded: int) -> tuple:
    """Return (label, colour) matching Review Center queue status values."""
    if n_coded == 0:
        return "Not Started", "#6b7280"
    elif elusion_rate > 0.10:
        return "Active", "#1a56db"
    elif elusion_rate > 0.05:
        return "Active — approaching Ready for Validation", "#d97706"
    else:
        return "Ready for Validation", "#16a34a"



# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="hero-section">
        <h1>Measuring What Matters in Review Center</h1>
        <p class="hero-sub">
            How Relativity Learning surfaces responsive documents in a TAR workflow —
            and how Review Validation Statistics tell you if it's working
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div class="info-card">
            <div class="card-icon">📁</div>
            <div class="card-title">The Challenge</div>
            <div class="card-body">A typical litigation may involve millions of documents.
            Human reviewers can only code a fraction. <strong>Relativity Learning</strong>
            uses machine learning to predict which documents are likely
            <em>responsive</em> to the case, surfacing them in the
            <strong>Prioritized Review Queue</strong>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div class="info-card">
            <div class="card-icon">🎯</div>
            <div class="card-title">The Goal</div>
            <div class="card-body">Find <em>every</em> document that matters to the case
            without forcing reviewers to wade through irrelevant material.
            Relativity Learning labels documents as <strong>Responsive</strong> or
            <strong>Not Relevant</strong> based on reviewer coding decisions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """
        <div class="info-card">
            <div class="card-icon">📊</div>
            <div class="card-title">The Measurement</div>
            <div class="card-body"><strong>Precision</strong>, <strong>Recall</strong>,
            <strong>Richness</strong>, and <strong>Elusion Rate</strong> are the
            Review Validation Statistics that tell you how well the TAR workflow
            is performing. This app lets you explore them interactively.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — THE DATASET
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## A Simulated Document Collection")
st.markdown(
    "The following dataset represents a typical e-discovery collection — "
    "100,000 documents from 15 custodians, spanning emails, contracts, memos, and more. "
    "Documents are classified into four buckets by Review Center: "
    "**coded relevant**, **coded not relevant**, **uncoded predicted relevant**, and **uncoded predicted not relevant**."
)

left, right = st.columns([2, 3])

with left:
    st.metric("Total Documents", f"{DISPLAY_TOTAL:,}")
    st.metric("Coded Relevant (Responsive)", f"{d(N_RESPONSIVE):,}", delta=f"{N_RESPONSIVE/N_TOTAL:.0%} of collection")
    st.metric("Coded Not Relevant", f"{d(N_NON_RESPONSIVE):,}", delta=f"{N_NON_RESPONSIVE/N_TOTAL:.0%} of collection")
    st.metric("Richness", f"{N_RESPONSIVE/N_TOTAL:.1%}", help="Richness = the percentage of relevant documents across the entire review.")
    st.caption(
        "**Richness** tells you how dense the collection is. In real matters, richness often falls "
        "between 5–20%. Low richness makes recall especially difficult — every missed document matters."
    )

with right:
    # Stacked bar chart: doc type breakdown by responsive / non-responsive
    type_counts = (
        df.groupby(["doc_type", "ground_truth"])
        .size()
        .reset_index(name="count")
    )
    responsive_counts = type_counts[type_counts["ground_truth"] == 1].set_index("doc_type")["count"]
    non_responsive_counts = type_counts[type_counts["ground_truth"] == 0].set_index("doc_type")["count"]
    all_types = sorted(df["doc_type"].unique())

    bar_fig = go.Figure()
    bar_fig.add_trace(
        go.Bar(
            y=all_types,
            x=[responsive_counts.get(t, 0) for t in all_types],
            name="Responsive",
            orientation="h",
            marker_color="#16a34a",
        )
    )
    bar_fig.add_trace(
        go.Bar(
            y=all_types,
            x=[non_responsive_counts.get(t, 0) for t in all_types],
            name="Non-Responsive",
            orientation="h",
            marker_color="#d1d5db",
        )
    )
    bar_fig.update_layout(
        barmode="stack",
        height=280,
        margin={"l": 20, "r": 20, "t": 10, "b": 40},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.35, "xanchor": "center", "x": 0.5},
        xaxis={"title": "Document Count", "gridcolor": "#e5e7eb"},
        yaxis={"gridcolor": "#e5e7eb"},
    )
    st.plotly_chart(bar_fig, use_container_width=True)

with st.expander("Browse sample documents"):
    sample = df.sample(10, random_state=1)[
        ["doc_id", "doc_type", "custodian", "date", "snippet", "ground_truth", "confidence_score"]
    ].rename(columns={"ground_truth": "Responsive?", "confidence_score": "Prediction Score"})
    st.dataframe(
        sample,
        use_container_width=True,
        column_config={
            "Responsive?": st.column_config.NumberColumn(
                label="Responsive?",
                help="1 = Coded Relevant, 0 = Coded Not Relevant",
                format="%d",
            ),
            "Prediction Score": st.column_config.ProgressColumn(
                label="Prediction Score",
                min_value=0,
                max_value=1,
                format="%.3f",
            ),
        },
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2.5 — TAR WORKFLOW: LEARNING OVER TIME
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## How Relativity Learning Gets Smarter Over Time")
st.markdown(
    "The key advantage of TAR is that **reviewers don't need to code every document**. "
    "Relativity Learning assigns a prediction score to every document based on what reviewers "
    "have already coded. As more documents are coded, the model improves — "
    "scores separate into two distinct groups and the Prioritized Review Queue becomes more targeted. "
    "Below are three stages of a typical TAR2 workflow on this collection."
)

stages = compute_training_stages()
ground_truth_arr = df["ground_truth"].values

stage_icons = ["🌱", "📈", "✅"]
stage_insights = [
    (
        "The model has limited training data. Prediction scores are clustered near the middle — "
        "Relativity Learning can't yet reliably distinguish responsive from not relevant. "
        "The PRQ is large and the elusion rate is high. **Reviewers should continue coding "
        "documents from the PRQ** to improve the model."
    ),
    (
        "The model is improving. Prediction scores are beginning to separate — responsive documents "
        "are trending toward higher scores and not-relevant documents toward lower scores. "
        "The PRQ is shrinking and becoming more precise. "
        "**Continue coding** until the elusion rate drops to an acceptable level."
    ),
    (
        "The model is mature. Scores are well-separated into two distinct groups. "
        "The PRQ is tight and high-precision. "
        "Project Validation of the **Uncoded: Predicted Not Relevant** bucket confirms a low "
        "elusion rate — very few responsive documents are eluding review. "
        "**Review Center can be defensibly closed** without manually coding the remaining documents."
    ),
]

tab_labels = [f"{icon}  {s['name']}" for icon, s in zip(stage_icons, stages)]
tabs = st.tabs(tab_labels)

for tab, stage, insight in zip(tabs, stages, stage_insights):
    with tab:
        # ── Metric row ────────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Documents Coded</div>
                    <div class="metric-value" style="color:#1a56db">{d(stage["n_coded"]):,}</div>
                    <div class="metric-caption">of {DISPLAY_TOTAL:,} total documents</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Uncoded in PRQ</div>
                    <div class="metric-value" style="color:#1a56db">{d(stage["prq_size"]):,}</div>
                    <div class="metric-caption">predicted relevant, awaiting review</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            prec_color = "#16a34a" if stage["model_precision"] >= 0.80 else ("#d97706" if stage["model_precision"] >= 0.50 else "#dc2626")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Model Precision</div>
                    <div class="metric-value" style="color:{prec_color}">{stage["model_precision"]:.1%}</div>
                    <div class="metric-caption">of PRQ docs that are truly responsive</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            elusion_color = "#dc2626" if stage["elusion_rate"] >= 0.15 else ("#d97706" if stage["elusion_rate"] >= 0.05 else "#16a34a")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Elusion Rate</div>
                    <div class="metric-value" style="color:{elusion_color}">{stage["elusion_rate"]:.1%}</div>
                    <div class="metric-caption">responsive docs eluding review (lower is better)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Histogram + insight ───────────────────────────────────────────────
        hist_col, insight_col = st.columns([3, 2])

        with hist_col:
            resp_scores = stage["scores"][ground_truth_arr == 1]
            non_resp_scores = stage["scores"][ground_truth_arr == 0]
            st.plotly_chart(
                build_score_distribution_figure(resp_scores, non_resp_scores, stage["cutoff"]),
                use_container_width=True,
            )

        with insight_col:
            st.markdown("#### What This Stage Means")
            st.markdown(insight)
            uncoded_not_rel = stage["b3"]
            st.markdown(
                f"""
                <div class="callout-box">
                    <strong>{d(uncoded_not_rel):,} documents</strong> are currently in the
                    <em>Uncoded: Predicted Not Relevant</em> bucket.
                    Elusion rate: <strong>{stage["elusion_rate"]:.1%}</strong>
                    ({d(stage["b3_eluded"]):,} responsive docs estimated to have eluded review).
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Bucket bar ────────────────────────────────────────────────────────
        st.plotly_chart(
            build_bucket_bar_figure(d(stage["b1"]), d(stage["b2"]), d(stage["b3"]), d(stage["b4"])),
            use_container_width=True,
        )

# ── Efficiency summary (outside tabs) ────────────────────────────────────────
end_stage = stages[-1]
docs_saved = N_TOTAL - end_stage["n_coded"]
pct_saved = docs_saved / N_TOTAL
st.info(
    f"**TAR Efficiency:** By the End of Review stage, reviewers manually coded "
    f"**{d(end_stage['n_coded']):,} of {DISPLAY_TOTAL:,} documents ({end_stage['n_coded']/N_TOTAL:.0%})**. "
    f"Relativity Learning processed the remaining **{d(docs_saved):,} documents ({pct_saved:.0%})** — "
    f"determining their relevancy through prediction scores rather than manual review. "
    f"Project Validation confirms an elusion rate of **{end_stage['elusion_rate']:.1%}** "
    f"in the not-reviewed bucket."
)

# ── Interactive: Simulate any point in the review ─────────────────────────────
st.markdown("### Try It: Simulate Any Point in Your Review")
st.markdown(
    "Drag the sliders to see exactly how the **Rank Distribution** and document buckets look "
    "at any point during the review — mirroring what you would see on the "
    "Review Center monitoring dashboard."
)

live_slider_col, live_cutoff_col = st.columns([3, 1])
with live_slider_col:
    n_coded_live = st.slider(
        "Docs Coded",
        min_value=0,
        max_value=DISPLAY_TOTAL,
        value=6_500,
        step=500,
        help="The number of documents reviewers have coded from the Prioritized Review Queue.",
        key="n_coded_live",
    )
with live_cutoff_col:
    live_cutoff = st.slider(
        "Positive Cutoff",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Rank score threshold — documents at or above this value enter the PRQ.",
        key="live_cutoff",
    )

# Convert display-scale coded count back to simulation scale (750-doc universe)
n_coded_sim = round(n_coded_live / DISPLAY_SCALE)
live = simulate_live_stage(n_coded_sim, live_cutoff)
status_label, status_color = queue_status_badge(live["elusion_rate"], n_coded_live)

# Status badge + 5-metric row
st.markdown(
    f'<div style="display:inline-block;background:{status_color};color:#fff;'
    f'font-weight:600;font-size:0.85rem;padding:0.3rem 0.9rem;border-radius:999px;'
    f'margin-bottom:0.75rem;">Queue Status: {status_label}</div>',
    unsafe_allow_html=True,
)

lm1, lm2, lm3, lm4, lm5 = st.columns(5)

def _live_card(label, value, caption, color="#1a202c"):
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color}">{value}</div>'
        f'<div class="metric-caption">{caption}</div>'
        f'</div>'
    )

with lm1:
    st.markdown(_live_card("Docs Coded", f"{n_coded_live:,}", f"of {DISPLAY_TOTAL:,} total", "#1a56db"), unsafe_allow_html=True)
with lm2:
    rel_color = "#16a34a" if live["relevance_rate"] >= 0.15 else "#d97706"
    st.markdown(_live_card("Relevance Rate", f"{live['relevance_rate']:.1%}", "of coded docs are responsive", rel_color), unsafe_allow_html=True)
with lm3:
    st.markdown(_live_card("Uncoded in PRQ", f"{d(live['prq_size']):,}", "predicted responsive, awaiting review", "#1a56db"), unsafe_allow_html=True)
with lm4:
    prec_color = "#16a34a" if live["precision"] >= 0.80 else ("#d97706" if live["precision"] >= 0.50 else "#dc2626")
    st.markdown(_live_card("Precision", f"{live['precision']:.1%}", "of PRQ docs truly responsive", prec_color), unsafe_allow_html=True)
with lm5:
    el_color = "#dc2626" if live["elusion_rate"] >= 0.15 else ("#d97706" if live["elusion_rate"] >= 0.05 else "#16a34a")
    st.markdown(_live_card("Elusion Rate", f"{live['elusion_rate']:.1%}", "responsive docs eluding review", el_color), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

live_hist_col, live_bucket_col = st.columns([3, 2])
with live_hist_col:
    resp_scores_live = live["scores"][ground_truth_arr == 1]
    non_resp_scores_live = live["scores"][ground_truth_arr == 0]
    st.plotly_chart(
        build_score_distribution_figure(resp_scores_live, non_resp_scores_live, live_cutoff),
        use_container_width=True,
    )
with live_bucket_col:
    st.plotly_chart(
        build_bucket_bar_figure(d(live["b1"]), d(live["b2"]), d(live["b3"]), d(live["b4"])),
        use_container_width=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — THE INTERACTIVE CORE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## The Positive Cutoff: Where You Draw the Line")
st.markdown(
    "Relativity Learning assigns every document a **prediction score** from 0 "
    "*(predicted not relevant)* to 1 *(predicted relevant)*. "
    "The **Positive Cutoff** is where you draw the line — documents scoring at or above it "
    "are added to the **Prioritized Review Queue (PRQ)**. "
    "Move the slider below to see how that choice affects every outcome."
)

threshold = st.slider(
    label="Positive Cutoff",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
    help="Documents with a prediction score at or above this value are classified as Responsive and added to the Prioritized Review Queue.",
)

m = metrics_at(threshold)
render_metrics_row(
    precision=m["precision"],
    recall=m["recall"],
    f1=m["f1"],
    tp=d(m["tp"]),
    fp=d(m["fp"]),
    fn=d(m["fn"]),
    tn=d(m["tn"]),
    threshold=threshold,
)

st.markdown("<br>", unsafe_allow_html=True)

cm_col, explain_col = st.columns([1, 1])

with cm_col:
    st.plotly_chart(
        build_confusion_matrix_figure(d(m["tp"]), d(m["fp"]), d(m["fn"]), d(m["tn"])),
        use_container_width=True,
    )

with explain_col:
    st.markdown("#### What do these cells mean in Review Center?")
    with st.expander('✅ True Positive — "FOUND IT"', expanded=False):
        st.markdown(
            "Relativity Learning predicted this document as relevant, **and it really is responsive**. "
            "In the Review Center bucket framework, this document is **coded relevant** and was "
            "correctly added to the Prioritized Review Queue. "
            "Maximising true positives means capturing the most critical evidence in your matter."
        )
    with st.expander('🚨 False Negative — "MISSED" (Eluded Documents)', expanded=False):
        st.markdown(
            "Relativity Learning predicted this document as **not relevant**, but it actually **is responsive**. "
            "In Review Center, this document falls into the **uncoded, predicted not relevant** bucket — "
            "it *eluded* the review and was never added to the Prioritized Review Queue. "
            "\n\n"
            "**Elusion Rate** measures how many truly responsive documents eluded review. "
            "This is the most dangerous error in a TAR workflow — missed documents could mean "
            "critical evidence is never reviewed or produced. Recall measures what fraction of "
            "these errors you are making."
        )
    with st.expander('⚠️ False Positive — "FALSE ALARM"', expanded=False):
        st.markdown(
            "Relativity Learning predicted this document as relevant, but it **is not responsive**. "
            "In Review Center, this document is added to the Prioritized Review Queue unnecessarily — "
            "a reviewer must open, read, and code it **not relevant** before moving on. "
            "Precision measures how often the predictions added to the PRQ are actually correct."
        )
    with st.expander('✔️ True Negative — "CORRECTLY SKIPPED"', expanded=False):
        st.markdown(
            "Relativity Learning correctly predicted this document as not relevant, "
            "keeping it out of the Prioritized Review Queue. "
            "In Review Center, this document remains in the **uncoded, predicted not relevant** bucket "
            "and is excluded from manual review. "
            "A large number of true negatives is the efficiency gain that makes TAR worthwhile — "
            "reviewers never have to touch these documents."
        )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — THE TRADEOFF
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## The Precision-Recall Tradeoff")

curve_col, text_col = st.columns([3, 2])

# Find current position on the PR curve (match closest threshold)
closest_idx = int(np.argmin(np.abs(curve["thresholds"] - threshold)))
current_precision_on_curve = float(curve["precisions"][closest_idx])
current_recall_on_curve = float(curve["recalls"][closest_idx])

with curve_col:
    st.plotly_chart(
        build_pr_curve_figure(
            precisions=curve["precisions"],
            recalls=curve["recalls"],
            thresholds=curve["thresholds"],
            current_threshold=threshold,
            current_precision=current_precision_on_curve,
            current_recall=current_recall_on_curve,
            optimal_threshold=curve["optimal_threshold"],
            optimal_precision=curve["optimal_precision"],
            optimal_recall=curve["optimal_recall"],
        ),
        use_container_width=True,
    )

with text_col:
    st.markdown("#### Why You Can't Have Both")
    st.markdown(
        "Pushing **recall higher** (lowering the Positive Cutoff) means adding *more* documents "
        "to the Prioritized Review Queue. "
        "Some of those will be true positives — but more will be false positives. "
        "Precision falls.\n\n"
        "Pushing **precision higher** (raising the Positive Cutoff) means only documents "
        "Relativity Learning is very confident about enter the PRQ. You'll have fewer false alarms, "
        "but more responsive documents will elude review entirely. Recall falls."
    )
    st.markdown(
        """
        <div class="callout-box">
            <strong>TAR1-style reviews</strong> focus on human reviewers coding all positive
            documents — typically no Positive Cutoff is applied and the full queue is reviewed.
            <strong>TAR2-style reviews</strong> trust Relativity Learning to predict some
            positive documents, applying a Positive Cutoff to exclude predicted non-relevant
            documents from review. Most TAR2 workflows target
            <strong>recall &ge; 0.75 or &ge; 0.80</strong>, accepting lower precision to
            minimise the risk of eluded responsive documents.
        </div>
        """,
        unsafe_allow_html=True,
    )
    opt = curve["optimal_threshold"]
    opt_m = metrics_at(opt)
    st.markdown(
        f"The **F1-optimal Positive Cutoff** for this dataset is **{opt:.2f}**, "
        f"yielding precision = {opt_m['precision']:.3f} and recall = {opt_m['recall']:.3f}."
    )

st.plotly_chart(
    build_f1_curve_figure(
        precisions=curve["precisions"],
        recalls=curve["recalls"],
        thresholds=curve["thresholds"],
        optimal_threshold=curve["optimal_threshold"],
    ),
    use_container_width=True,
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PRACTICAL IMPLICATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## What This Means for Your Review")
st.markdown(
    "Different matters call for different Positive Cutoff strategies in Review Center. "
    "Here are three common approaches and what they mean in practice for this dataset. "
    "After setting a Positive Cutoff, **Project Validation** samples uncoded documents from "
    "the predicted not-relevant bucket to confirm the elusion rate is acceptable."
)

scenarios = [
    {
        "title": "Cost-Focused Strategy",
        "icon": "💰",
        "threshold": 0.70,
        "description": (
            "Set a high Positive Cutoff to keep the PRQ tight. Reviewers spend less time on "
            "not-relevant documents, but some responsive documents will elude review. "
            "Best suited for high-volume, lower-stakes matters where review budget is a primary constraint."
        ),
    },
    {
        "title": "Balanced Strategy (F1-Optimal)",
        "icon": "⚖️",
        "threshold": curve["optimal_threshold"],
        "description": (
            "Mathematically optimal balance between recall and PRQ efficiency. "
            "A defensible default starting point for most matters in Review Center — "
            "adjust the Positive Cutoff based on case risk and matter needs."
        ),
    },
    {
        "title": "High-Recall / Defensible TAR2",
        "icon": "🔍",
        "threshold": 0.30,
        "description": (
            "Set a low Positive Cutoff to maximise the chance of catching every responsive document. "
            "Appropriate for high-stakes litigation where eluded documents carry significant legal, "
            "financial, or reputational risk. PRQ size increases substantially — "
            "validate with Project Validation before closing review."
        ),
    },
]

s_cols = st.columns(3)
for col, scenario in zip(s_cols, scenarios):
    sm = metrics_at(scenario["threshold"])
    with col:
        st.markdown(
            f"""
            <div class="scenario-card">
                <div class="scenario-icon">{scenario["icon"]}</div>
                <div class="scenario-title">{scenario["title"]}</div>
                <div class="scenario-threshold">Positive Cutoff: {scenario["threshold"]:.2f}</div>
                <div class="scenario-body">{scenario["description"]}</div>
                <div class="scenario-stats">
                    <div class="stat-row"><span class="stat-label">Precision</span>
                        <span class="stat-value">{sm["precision"]:.3f}</span></div>
                    <div class="stat-row"><span class="stat-label">Recall</span>
                        <span class="stat-value">{sm["recall"]:.3f}</span></div>
                    <div class="stat-row"><span class="stat-label">Added to PRQ</span>
                        <span class="stat-value">{d(sm["tp"] + sm["fp"]):,}</span></div>
                    <div class="stat-row"><span class="stat-label">Coded Relevant Found</span>
                        <span class="stat-value">{d(sm["tp"]):,} / {d(N_RESPONSIVE):,}</span></div>
                    <div class="stat-row"><span class="stat-label">Eluded (Missed)</span>
                        <span class="stat-value">{d(sm["fn"]):,}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
st.success(
    "**Precision, recall, and elusion rate are not just abstract metrics** — they represent real decisions "
    "about reviewer time and case risk. Review Center and Relativity Learning give legal teams the tools "
    "to set a defensible Positive Cutoff, validate it with Project Validation, "
    "and close review with confidence."
)
