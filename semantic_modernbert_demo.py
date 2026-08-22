"""
Semantic-search + ModernBERT ABSA demo pipeline.

This is the "robustness pipeline" from Section VI-C of the paper
(semantic aspect matching + ModernBERT sentiment), shown here as a
standalone, runnable demo — separate from the primary keyword+FinBERT
pipeline that produced aspect_sentiments.csv and that Section 7's
intrinsic evaluation was measured against.

Usage:
    python semantic_modernbert_demo.py --csv aspect_sentiments.csv --n 5

Requires (pip install):
    sentence-transformers
    transformers
    torch
    pandas
"""

import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline as hf_pipeline

# ---------------------------------------------------------------------------
# 1. Aspect taxonomy — same 42 aspects as the primary pipeline, each given a
#    short natural-language definition to use as the semantic-matching anchor.
# ---------------------------------------------------------------------------

ASPECT_DEFINITIONS = {
    # Universal (all sectors)
    "revenue": "discussion of total revenue, sales, or top-line growth",
    "margin": "discussion of profit margin, EBITDA margin, or profitability",
    "guidance": "forward-looking guidance, outlook, or targets for future performance",
    "cost": "discussion of operating costs, expenses, or cost control",
    "capex": "discussion of capital expenditure or investment spending",
    "demand": "discussion of customer demand or market demand conditions",

    # Banking
    "loan_growth": "growth or change in the bank's loan book or credit disbursement",
    "deposit_growth": "growth or change in customer deposits",
    "nim": "net interest margin (NIM) trends",
    "credit_cost": "provisioning or credit cost as a share of advances",
    "slippages": "loans slipping into non-performing asset (NPA) status",
    "recoveries": "recovery or upgrade of previously stressed loans",
    "capital_adequacy": "capital adequacy ratio or regulatory capital buffers",
    "fee_income": "non-interest or fee-based income",

    # IT Services
    "attrition": "employee attrition or voluntary resignation rate",
    "headcount": "total employee headcount or hiring levels",
    "tcv": "total contract value (TCV) of deals signed",
    "large_deals": "large or mega deal wins",
    "pipeline": "sales pipeline strength or deal pipeline",
    "pricing": "pricing pressure or pricing power on client contracts",
    "geography_vertical": "performance by geography or industry vertical",
    "ai_adoption": "adoption of AI or generative AI in service offerings",

    # Pharma
    "approvals": "regulatory product approvals or launches",
    "regulatory": "regulatory inspections, warning letters, or compliance actions",
    "us_business": "performance of the US market business",
    "api_business": "active pharmaceutical ingredient (API) business",
    "r_and_d": "research and development spend or pipeline",
    "chronic_therapy": "chronic therapy or specialty franchise business",
    "domestic_formulations": "domestic formulations business performance",

    # Metals
    "capacity_expansion": "capacity expansion projects or greenfield/brownfield growth",
    "capacity_utilization": "utilization rate of existing production capacity",
    "steel_prices": "steel price trends",
    "coking_coal": "coking coal supply or cost",
    "exports_import": "export or import volumes and duties",
    "realization": "price realization relative to market benchmarks",
    "volumes": "production or sales volumes",

    # FMCG
    "rural_urban": "rural versus urban demand trends",
    "premiumisation": "premiumisation or shift to higher-value products",
    "distribution": "distribution reach or trade network expansion",
    "brand_investment": "brand investment, advertising, or marketing spend",
    "raw_material": "raw material or input commodity cost trends",
    "volume_growth": "volume growth in units sold",
}

ASPECT_NAMES = list(ASPECT_DEFINITIONS.keys())
ASPECT_TEXTS = list(ASPECT_DEFINITIONS.values())


def semantic_match_aspects(sentences, embedder, top_k=1, threshold=0.30):
    """Embed sentences + aspect definitions, return best-matching aspect(s)."""
    aspect_emb = embedder.encode(ASPECT_TEXTS, convert_to_tensor=True)
    sent_emb = embedder.encode(sentences, convert_to_tensor=True)
    sims = util.cos_sim(sent_emb, aspect_emb)  # [n_sentences, n_aspects]

    results = []
    for row in sims:
        best_idx = int(row.argmax())
        best_score = float(row[best_idx])
        if best_score < threshold:
            results.append(("none", best_score))
        else:
            results.append((ASPECT_NAMES[best_idx], best_score))
    return results


def run_demo(csv_path, n_samples, seed=42):
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Pull a small sample of sentences that already have keyword+FinBERT output,
    # so we can show both pipelines side by side on the SAME sentences.
    sample = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    print("Loading semantic embedding model (sentence-transformers)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading ModernBERT financial sentiment model...")
    sentiment_model = hf_pipeline(
        "text-classification",
        model="tabularisai/ModernFinBERT",
        top_k=None,
    )

    sentences = sample["sentence"].tolist()
    semantic_results = semantic_match_aspects(sentences, embedder)

    print("\n" + "=" * 100)
    print(f"{'SENTENCE':<55} | {'KEYWORD+FinBERT (primary)':<28} | {'SEMANTIC+ModernBERT (demo)'}")
    print("=" * 100)

    for i, row in sample.iterrows():
        sem_aspect, sem_score = semantic_results[i]
        modernbert_out = sentiment_model(row["sentence"])[0]
        top_label = max(modernbert_out, key=lambda x: x["score"])

        primary = f"{row['aspect']} / {row['sentiment_score']:+.2f}"
        demo = f"{sem_aspect} ({sem_score:.2f}) / {top_label['label']} ({top_label['score']:.2f})"

        print(f"{row['sentence'][:53]:<55} | {primary:<28} | {demo}")

    print("=" * 100)
    print(
        "\nNote: aspect and sentiment labels are expected to differ between the two\n"
        "pipelines on some sentences — this is the extraction-method sensitivity\n"
        "documented in Section VI-C of the paper, not a bug in either pipeline."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="aspect_sentiments.csv", help="Path to the aspect-tagged CSV")
    parser.add_argument("--n", type=int, default=5, help="Number of sample sentences to run")
    args = parser.parse_args()
    run_demo(args.csv, args.n)
