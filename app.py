"""
Streamlit UI for the semantic+ModernBERT ABSA pipeline (see README.md).

Lets a user paste in free text (one or more sentences), then runs:
  1. Semantic aspect matching (sentence-transformers, cosine similarity
     against the 42 aspect definitions from semantic_modernbert_demo.py)
  2. Sentiment scoring (tabularisai/ModernFinBERT)
and displays the aspect + sentiment for each sentence.

This is the "robustness pipeline" from Section VI-C of the report, not the
primary keyword+FinBERT pipeline (that pipeline's source isn't in this repo
- only its output, aspect_sentences.csv, is). See README.md for why there
are two pipelines.

Run with:
    streamlit run app.py
"""

import os
import re
from pathlib import Path

# The C: drive on this machine is nearly full, and Hugging Face's default
# cache lives there (~/.cache/huggingface). Point it at a folder on D:
# instead so model downloads don't fail/corrupt. Must run before importing
# transformers/sentence-transformers.
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / ".hf_cache"))

import pandas as pd
import streamlit as st

from semantic_modernbert_demo import ASPECT_DEFINITIONS, semantic_match_aspects

st.set_page_config(page_title="ABSA Studio", page_icon="🧭", layout="wide")

# ---------------------------------------------------------------------------
# Sector tagging (matches the grouping in ASPECT_DEFINITIONS) — purely
# cosmetic, gives each aspect badge a sector icon.
# ---------------------------------------------------------------------------

SECTOR_ICON = {
    "universal": "🌐",
    "banking": "🏦",
    "it_services": "💻",
    "pharma": "💊",
    "metals": "⛏️",
    "fmcg": "🛒",
}

ASPECT_SECTOR = {
    **{a: "universal" for a in ["revenue", "margin", "guidance", "cost", "capex", "demand"]},
    **{
        a: "banking"
        for a in [
            "loan_growth", "deposit_growth", "nim", "credit_cost", "slippages",
            "recoveries", "capital_adequacy", "fee_income",
        ]
    },
    **{
        a: "it_services"
        for a in [
            "attrition", "headcount", "tcv", "large_deals", "pipeline",
            "pricing", "geography_vertical", "ai_adoption",
        ]
    },
    **{
        a: "pharma"
        for a in [
            "approvals", "regulatory", "us_business", "api_business",
            "r_and_d", "chronic_therapy", "domestic_formulations",
        ]
    },
    **{
        a: "metals"
        for a in [
            "capacity_expansion", "capacity_utilization", "steel_prices",
            "coking_coal", "exports_import", "realization", "volumes",
        ]
    },
    **{
        a: "fmcg"
        for a in [
            "rural_urban", "premiumisation", "distribution", "brand_investment",
            "raw_material", "volume_growth",
        ]
    },
}

SENTIMENT_STYLE = {
    "bullish": {"emoji": "📈", "color": "#1F9D55", "bg": "#E9FBF0"},
    "positive": {"emoji": "📈", "color": "#1F9D55", "bg": "#E9FBF0"},
    "bearish": {"emoji": "📉", "color": "#D64545", "bg": "#FDECEC"},
    "negative": {"emoji": "📉", "color": "#D64545", "bg": "#FDECEC"},
    "neutral": {"emoji": "➖", "color": "#8A6D00", "bg": "#FFF7DC"},
}
DEFAULT_SENTIMENT_STYLE = {"emoji": "❔", "color": "#5B5B6B", "bg": "#F0F0F5"}


def sentiment_style(label):
    return SENTIMENT_STYLE.get(label.lower(), DEFAULT_SENTIMENT_STYLE)


def sector_icon(aspect):
    return SECTOR_ICON.get(ASPECT_SECTOR.get(aspect, ""), "❔" if aspect == "none" else "🌐")


# ---------------------------------------------------------------------------
# Model loaders (cached across reruns)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading semantic embedding model...")
def load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading ModernFinBERT sentiment model...")
def load_sentiment_model():
    from transformers import pipeline as hf_pipeline

    return hf_pipeline("text-classification", model="tabularisai/ModernFinBERT", top_k=None)


def split_sentences(text):
    """Split on '.', '!', '?' followed by whitespace; also treat newlines as breaks."""
    chunks = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        chunks.extend(p.strip() for p in parts if p.strip())
    return chunks


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #F5F3FF 0%, #FFFFFF 320px); }

    .absa-hero {
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #6C5CE7 0%, #A29BFE 50%, #74B9FF 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 30px -12px rgba(108, 92, 231, 0.55);
    }
    .absa-hero h1 { margin: 0; font-size: 1.9rem; }
    .absa-hero p { margin: 0.35rem 0 0; opacity: 0.92; font-size: 0.95rem; }

    .absa-card {
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        background: white;
        border: 1px solid #ECEAF9;
        box-shadow: 0 4px 14px -10px rgba(30, 27, 46, 0.35);
    }
    .absa-sentence { font-size: 0.98rem; color: #1E1B2E; margin-bottom: 0.55rem; }

    .absa-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.18rem 0.65rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }

    .absa-bar-track {
        background: #F0EEFA;
        border-radius: 999px;
        height: 6px;
        width: 100%;
        overflow: hidden;
        margin-top: 0.35rem;
    }
    .absa-bar-fill { height: 100%; border-radius: 999px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🧭 About this pipeline")
    st.markdown(
        "**Semantic aspect matching + ModernFinBERT** — the robustness "
        "pipeline from Section VI-C of the report, shown live here.\n\n"
        "The **primary** keyword+FinBERT pipeline (Section 6/7, 76.5% aspect "
        "/ 75.8% sentiment accuracy) produced `aspect_sentences.csv`, but its "
        "extraction code isn't in this repo — only its CSV output is."
    )

    st.markdown("---")
    st.markdown("### 🎚️ Settings")
    threshold = st.slider(
        "Aspect-match confidence threshold",
        min_value=0.0,
        max_value=0.8,
        value=0.30,
        step=0.01,
        help="Below this cosine-similarity score, a sentence is tagged 'none'.",
    )
    celebrate = st.toggle("🎉 Celebrate strongly bullish results", value=True)

    st.markdown("---")
    st.markdown("### 📚 Aspect taxonomy")
    query = st.text_input("Search the 42 aspects", placeholder="e.g. margin, attrition...")
    for name, definition in ASPECT_DEFINITIONS.items():
        if query and query.lower() not in name.lower() and query.lower() not in definition.lower():
            continue
        st.markdown(f"{sector_icon(name)} **{name}** — {definition}")

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="absa-hero">
        <h1>🧭 ABSA Studio</h1>
        <p>Paste in earnings-call-style text — each sentence gets tagged with its
        closest aspect (semantic similarity across 42 finance aspects) and a
        sentiment read from ModernFinBERT.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

default_text = (
    "Net interest margin improved sequentially on the back of lower cost of funds.\n"
    "Attrition came down to 12% this quarter, which helped headcount stability.\n"
    "We remain cautious given rising input costs in the near term."
)

text = st.text_area(
    "Enter one or more sentences (one per line, or a paragraph):",
    value=default_text,
    height=160,
)

analyze = st.button("✨ Analyze", type="primary")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

if analyze:
    sentences = split_sentences(text)
    if not sentences:
        st.warning("Enter at least one sentence.")
        st.stop()

    embedder = load_embedder()
    sentiment_model = load_sentiment_model()

    with st.spinner(f"Analyzing {len(sentences)} sentence(s)..."):
        aspect_results = semantic_match_aspects(sentences, embedder, threshold=threshold)
        sentiment_outputs = sentiment_model(sentences)

    rows = []
    for sentence, (aspect, aspect_score), sentiment_out in zip(
        sentences, aspect_results, sentiment_outputs
    ):
        top = max(sentiment_out, key=lambda x: x["score"])
        rows.append(
            {
                "sentence": sentence,
                "aspect": aspect,
                "aspect definition": ASPECT_DEFINITIONS.get(aspect, "—"),
                "aspect confidence": round(aspect_score, 3),
                "sentiment": top["label"],
                "sentiment confidence": round(top["score"], 3),
            }
        )

    results_df = pd.DataFrame(rows)

    # --- Summary strip -----------------------------------------------------
    dominant_sentiment = results_df["sentiment"].mode().iloc[0]
    dominant_style = sentiment_style(dominant_sentiment)
    dominant_aspect = (
        results_df.loc[results_df["aspect"] != "none", "aspect"].mode()
        if (results_df["aspect"] != "none").any()
        else pd.Series(["none"])
    )
    dominant_aspect = dominant_aspect.iloc[0] if not dominant_aspect.empty else "none"
    avg_conf = results_df["sentiment confidence"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sentences analyzed", len(results_df))
    m2.metric("Dominant sentiment", f"{dominant_style['emoji']} {dominant_sentiment}")
    m3.metric("Most common aspect", f"{sector_icon(dominant_aspect)} {dominant_aspect}")
    m4.metric("Avg. sentiment confidence", f"{avg_conf:.0%}")

    if celebrate and dominant_sentiment.lower() in ("bullish", "positive") and avg_conf > 0.7:
        st.balloons()

    st.markdown("### Results")

    # --- Per-sentence cards --------------------------------------------------
    for _, row in results_df.iterrows():
        s_style = sentiment_style(row["sentiment"])
        icon = sector_icon(row["aspect"])
        st.markdown(
            f"""
            <div class="absa-card">
                <div class="absa-sentence">{row['sentence']}</div>
                <span class="absa-badge" style="background:{s_style['bg']};color:{s_style['color']};">
                    {s_style['emoji']} {row['sentiment']} · {row['sentiment confidence']:.0%}
                </span>
                <span class="absa-badge" style="background:#F0EEFA;color:#4B3FA8;">
                    {icon} {row['aspect']} · {row['aspect confidence']:.0%}
                </span>
                <div class="absa-bar-track">
                    <div class="absa-bar-fill" style="width:{row['aspect confidence']*100:.0f}%;background:{s_style['color']};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("📋 View as table"):
        st.dataframe(results_df, width="stretch", hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Aspect distribution")
        st.bar_chart(results_df["aspect"].value_counts())
    with col2:
        st.subheader("Sentiment distribution")
        st.bar_chart(results_df["sentiment"].value_counts())

    st.download_button(
        "⬇️ Download results as CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="absa_results.csv",
        mime="text/csv",
    )
