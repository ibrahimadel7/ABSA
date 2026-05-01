#!/usr/bin/env python3
# =============================================================================
# inference.py — DeepX ABSA: generate submission.json from unlabeled data
#
# Usage:
#   python inference.py
#   python inference.py --unlabeled_path data/DeepX_unlabeled.xlsx \
#                       --weights_dir    weights \
#                       --output         submission.json
#
# Requires trained weights in --weights_dir (run train.py first).
# =============================================================================

import argparse
import json
import os

import joblib
import pandas as pd

from config import ASPECTS, FEATURE_COLS, SENTIMENT_FEATURE_COLS, UNLABELED_PATH, WEIGHTS_DIR, SUBMISSION_PATH
from preprocessing import prepare_dataset
from models import combine_aspect_candidates, keyword_aspect_fallback


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepX ABSA inference")
    parser.add_argument("--unlabeled_path", default=UNLABELED_PATH)
    parser.add_argument("--weights_dir",    default=WEIGHTS_DIR)
    parser.add_argument("--output",         default=SUBMISSION_PATH)
    parser.add_argument("--fallback_mode", choices=["auto", "always", "only_if_empty", "never"], default="auto")
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pickle(path: str):
    """Load a trained joblib artifact."""
    return joblib.load(path)


def load_thresholds(path: str) -> dict:
    """Load tuned aspect thresholds when the file exists."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_aspect_metadata(path: str) -> dict:
    """Load optional metadata saved alongside the aspect model."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_json_value(value):
    """Convert pandas/numpy scalars into JSON-friendly Python values."""
    return value.item() if hasattr(value, "item") else value


def predict_aspects_batch(df, aspect_model, mlb, thresholds, fallback_mode):
    """Predict aspect lists for all rows using one batched decision_function call."""
    scores = aspect_model.decision_function(df[FEATURE_COLS])
    all_aspects = []

    for row_idx, (_, row) in enumerate(df.iterrows()):
        model_aspects = [
            aspect
            for i, aspect in enumerate(mlb.classes_)
            if scores[row_idx, i] >= thresholds.get(aspect, 0.0)
        ]
        fallback_aspects = keyword_aspect_fallback(row["review_text"]) or []
        all_aspects.append(
            combine_aspect_candidates(model_aspects, fallback_aspects, fallback_mode=fallback_mode)
        )

    return all_aspects


def predict_sentiments_batch(df, predicted_aspects, sentiment_model):
    """Predict all per-aspect sentiment rows in one model call."""
    rows = []
    index_pairs = []

    for row_idx, (_, row) in enumerate(df.iterrows()):
        for aspect in predicted_aspects[row_idx]:
            rows.append({
                "review_text_clean": row["review_text_clean"],
                "business_category_grouped": row["business_category_grouped"],
                "platform": row["platform"],
                "star_rating_num": row["star_rating_num"],
                "aspect": aspect,
                **{col: row[col] for col in df.columns if col.startswith("kw_")},
            })
            index_pairs.append((row_idx, aspect))

    sentiment_df = pd.DataFrame(rows, columns=SENTIMENT_FEATURE_COLS)
    predictions = sentiment_model.predict(sentiment_df) if len(sentiment_df) else []

    aspect_sentiments = [{} for _ in range(len(df))]
    for (row_idx, aspect), sentiment in zip(index_pairs, predictions):
        aspect_sentiments[row_idx][aspect] = (
            sentiment if sentiment in ["positive", "negative", "neutral"] else "neutral"
        )

    return aspect_sentiments


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Load weights ───────────────────────────────────────────────────────
    print("\n[1/4] Loading model weights...")
    aspect_model = load_pickle(os.path.join(args.weights_dir, "aspect_model.joblib"))
    sentiment_model = load_pickle(os.path.join(args.weights_dir, "sentiment_model.joblib"))
    mlb = load_pickle(os.path.join(args.weights_dir, "mlb.joblib"))
    thresholds = load_thresholds(os.path.join(args.weights_dir, "aspect_thresholds.json"))
    aspect_metadata = load_aspect_metadata(os.path.join(args.weights_dir, "aspect_model.json"))
    fallback_mode = args.fallback_mode
    if fallback_mode == "auto":
        fallback_mode = aspect_metadata.get("fallback_mode", "only_if_empty")
    print("  weights loaded.")
    print(f"  fallback_mode={fallback_mode}")

    # ── 2. Load and preprocess unlabeled data ─────────────────────────────────
    print("\n[2/4] Loading unlabeled data...")
    df_raw = pd.read_excel(args.unlabeled_path)
    print(f"  rows: {len(df_raw):,}")

    df = prepare_dataset(df_raw, has_labels=False)
    print("  preprocessing done.")

    # ── 3. Run ABSA inference ─────────────────────────────────────────────────
    print("\n[3/4] Running inference...")
    predicted_aspects = predict_aspects_batch(df, aspect_model, mlb, thresholds, fallback_mode)
    predicted_sentiments = predict_sentiments_batch(df, predicted_aspects, sentiment_model)

    records = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        records.append({
            "review_id": to_json_value(row["review_id"]),
            "aspects": list(predicted_sentiments[row_idx].keys()),
            "aspect_sentiments": predicted_sentiments[row_idx],
        })

    print(f"  inference done — {len(records):,} records.")

    # ── 4. Save submission JSON ───────────────────────────────────────────────
    print(f"\n[4/4] Saving submission → {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(args.output) / 1e3
    print(f"  saved. ({size_kb:.1f} KB)")

    # ── Quick sanity check ────────────────────────────────────────────────────
    print("\nSample predictions (first 3):")
    for r in records[:3]:
        print(f"  review_id={r['review_id']}  →  {r['aspect_sentiments']}")


if __name__ == "__main__":
    main()
