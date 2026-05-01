#!/usr/bin/env python3
# =============================================================================
# train.py — DeepX ABSA: end-to-end training and evaluation script
#
# Usage:
#   python train.py
#   python train.py --train_path data/DeepX_train.xlsx \
#                   --val_path   data/DeepX_validation.xlsx
#   python train.py --data_path data/DeepX_train.xlsx --val_size 0.15 --test_size 0.15
#
# Notes:
#   - The script trains the existing model architecture only.
#   - Precision / recall / F1 are reported only when labels are available.
#   - Unlabeled data such as the hidden test set cannot be used for training metrics.
# =============================================================================

import argparse
import json
import os
import random
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from config import (
    TRAIN_PATH,
    VAL_PATH,
    WEIGHTS_DIR,
    ASPECT_MODEL_PATH,
    SENTIMENT_MODEL_PATH,
    MLB_PATH,
    ASPECT_MODEL_JSON_PATH,
    SENTIMENT_MODEL_JSON_PATH,
    MLB_JSON_PATH,
    ASPECT_THRESHOLDS_PATH,
    ASPECTS,
    FEATURE_COLS,
    SENTIMENT_FEATURE_COLS,
)
from preprocessing import (
    prepare_dataset,
    remove_duplicate_reviews,
    build_sentiment_training_df,
    parse_list,
)
from models import build_aspect_model, build_sentiment_model
from models import combine_aspect_candidates, keyword_aspect_fallback


TRAIN_ASPECTS = [aspect for aspect in ASPECTS if aspect != "none"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepX ABSA models")
    parser.add_argument("--train_path", default=TRAIN_PATH)
    parser.add_argument("--val_path", default=VAL_PATH)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--weights_dir", default=WEIGHTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.0)
    parser.add_argument(
        "--threshold_strategy",
        choices=["binary_f1", "micro_f1", "absa_micro_f1", "none"],
        default="absa_micro_f1",
    )
    parser.add_argument("--fallback_mode", choices=["always", "only_if_empty", "never"], default="only_if_empty")
    parser.add_argument("--threshold_min", type=float, default=-1.5)
    parser.add_argument("--threshold_max", type=float, default=1.5)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--threshold_tune_ratio", type=float, default=0.6)
    parser.add_argument("--precision_guardrail", type=float, default=0.0)
    parser.add_argument("--aspect_c", type=float, default=1.0)
    parser.add_argument("--sentiment_c", type=float, default=2.0)
    parser.add_argument("--svm_max_iter", type=int, default=12000)
    parser.add_argument("--word_ngram_max", type=int, default=2)
    parser.add_argument("--char_ngram_max", type=int, default=5)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_df", type=float, default=0.95)
    parser.add_argument("--sublinear_tf", action="store_true", default=True)
    parser.add_argument("--no_sublinear_tf", action="store_false", dest="sublinear_tf")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--experiment_log", default="weights/experiments.jsonl")
    parser.add_argument("--keep_split_overlaps", action="store_true", default=False)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_table(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  saved → {path}  ({size_mb:.1f} MB)")


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  saved → {path}")


def has_label_columns(df: pd.DataFrame) -> bool:
    return {"aspects", "aspect_sentiments"}.issubset(df.columns)


def label_signature(aspects_list):
    if not aspects_list:
        return "__empty__"
    return "|".join(sorted(map(str, aspects_list)))


def safe_split(df: pd.DataFrame, test_size: float, seed: int):
    if test_size <= 0:
        return df, None

    stratify = None
    if "aspects_list" in df.columns:
        signatures = df["aspects_list"].apply(label_signature)
    elif "aspects" in df.columns:
        signatures = df["aspects"].apply(parse_list).apply(label_signature)
    else:
        signatures = None

    if signatures is not None:
        signature_counts = signatures.value_counts()
        if len(signature_counts) > 1 and signature_counts.min() >= 2:
            stratify = signatures

    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_labeled_data(df: pd.DataFrame, val_size: float, test_size: float, seed: int):
    if val_size < 0 or test_size < 0:
        raise ValueError("val_size and test_size must be non-negative")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1")

    work_df = df.copy()

    if test_size > 0:
        train_df, test_df = safe_split(work_df, test_size=test_size, seed=seed)
    else:
        train_df, test_df = work_df.reset_index(drop=True), None

    if val_size > 0:
        if test_df is None:
            effective_val = val_size
        else:
            effective_val = val_size / max(1e-12, 1 - test_size)

        train_df, val_df = safe_split(train_df, test_size=effective_val, seed=seed)
    else:
        val_df = None

    return train_df, val_df, test_df


def split_validation_for_threshold_tuning(val_df: pd.DataFrame, tune_ratio: float, seed: int):
    """Split validation into tuning and untouched evaluation subsets."""
    if val_df is None:
        return None, None
    if len(val_df) < 2:
        return val_df, val_df
    if not (0.0 < tune_ratio < 1.0):
        return val_df, val_df

    tune_df, eval_df = safe_split(
        val_df,
        test_size=(1.0 - tune_ratio),
        seed=seed,
    )
    if tune_df is None or eval_df is None or tune_df.empty or eval_df.empty:
        return val_df, val_df

    return tune_df, eval_df


def overlap_count(df_a: pd.DataFrame | None, df_b: pd.DataFrame | None) -> int:
    if df_a is None or df_b is None:
        return 0
    if "review_text_clean" not in df_a.columns or "review_text_clean" not in df_b.columns:
        return 0
    return len(set(df_a["review_text_clean"]) & set(df_b["review_text_clean"]))


def drop_overlaps(reference_df: pd.DataFrame, candidate_df: pd.DataFrame | None) -> tuple[pd.DataFrame | None, int]:
    if candidate_df is None:
        return None, 0
    if reference_df is None:
        return candidate_df, 0

    reference_texts = set(reference_df["review_text_clean"])
    overlap_mask = candidate_df["review_text_clean"].isin(reference_texts)
    removed = int(overlap_mask.sum())
    if removed == 0:
        return candidate_df, 0

    filtered = candidate_df.loc[~overlap_mask].reset_index(drop=True)
    return filtered, removed


def print_metric_table(title: str, metrics: list[dict]):
    print(f"\n{title}")
    print(pd.DataFrame(metrics).to_string(index=False))


def multilabel_metrics(y_true, y_pred):
    rows = []
    for avg in ["micro", "macro", "weighted"]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=avg,
            zero_division=0,
        )
        rows.append({
            "average": avg,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    rows.append({
        "average": "subset_accuracy",
        "precision": np.nan,
        "recall": np.nan,
        "f1": accuracy_score(y_true, y_pred),
    })
    return rows


def multiclass_metrics(y_true, y_pred):
    rows = []
    for avg in ["macro", "weighted"]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=avg,
            zero_division=0,
        )
        rows.append({
            "average": avg,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    rows.append({
        "average": "accuracy",
        "precision": np.nan,
        "recall": np.nan,
        "f1": accuracy_score(y_true, y_pred),
    })
    return rows


def tune_aspect_thresholds_binary_f1(model, X_val, y_val, aspect_labels, threshold_values):
    scores = model.decision_function(X_val)
    thresholds = {}
    tuned_predictions = np.zeros_like(y_val)

    for i, aspect in enumerate(aspect_labels):
        best_threshold = 0.0
        best_f1 = -1.0

        for threshold in threshold_values:
            preds = (scores[:, i] >= threshold).astype(int)
            score = precision_recall_fscore_support(
                y_val[:, i],
                preds,
                average="binary",
                zero_division=0,
            )[2]
            if score > best_f1:
                best_f1 = score
                best_threshold = float(round(threshold, 2))

        thresholds[aspect] = best_threshold
        tuned_predictions[:, i] = (scores[:, i] >= best_threshold).astype(int)

    return thresholds, tuned_predictions


def _micro_prf(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )
    return precision, recall, f1


def tune_aspect_thresholds_micro_f1(model, X_val, y_val, aspect_labels, threshold_values, passes=2):
    scores = model.decision_function(X_val)
    thresholds = {aspect: 0.0 for aspect in aspect_labels}
    tuned_predictions = (scores >= 0.0).astype(int)

    for _ in range(passes):
        improved = False
        for i, aspect in enumerate(aspect_labels):
            best_threshold = thresholds[aspect]
            best_preds_col = tuned_predictions[:, i].copy()
            best_precision, best_recall, best_f1 = _micro_prf(y_val, tuned_predictions)

            for threshold in threshold_values:
                candidate_preds = tuned_predictions.copy()
                candidate_preds[:, i] = (scores[:, i] >= threshold).astype(int)
                precision, recall, f1 = _micro_prf(y_val, candidate_preds)
                if (f1, precision, recall) > (best_f1, best_precision, best_recall):
                    best_threshold = float(round(threshold, 2))
                    best_preds_col = candidate_preds[:, i]
                    best_precision, best_recall, best_f1 = precision, recall, f1
                    improved = True

            thresholds[aspect] = best_threshold
            tuned_predictions[:, i] = best_preds_col

        if not improved:
            break

    return thresholds, tuned_predictions


def build_aspect_predictions_from_scores(df, aspect_labels, scores, thresholds, fallback_mode):
    predicted_aspects = []

    for row_idx, (_, row) in enumerate(df.iterrows()):
        model_aspects = [
            aspect
            for i, aspect in enumerate(aspect_labels)
            if scores[row_idx, i] >= thresholds.get(aspect, 0.0)
        ]
        fallback_aspects = keyword_aspect_fallback(row["review_text"]) or []
        predicted_aspects.append(
            combine_aspect_candidates(model_aspects, fallback_aspects, fallback_mode=fallback_mode)
        )

    return predicted_aspects


def precompute_sentiment_predictions(df, sentiment_model, aspect_labels):
    sentiment_rows = []
    position = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        for aspect in aspect_labels:
            sentiment_rows.append({
                "review_text_clean": row["review_text_clean"],
                "business_category_grouped": row["business_category_grouped"],
                "platform": row["platform"],
                "star_rating_num": row["star_rating_num"],
                "aspect": aspect,
                **{col: row[col] for col in df.columns if col.startswith("kw_")},
            })
            position.append((row_idx, aspect))

    lookup = [dict() for _ in range(len(df))]
    if sentiment_rows:
        sentiment_df = pd.DataFrame(sentiment_rows, columns=SENTIMENT_FEATURE_COLS)
        pred_sentiments = sentiment_model.predict(sentiment_df)
        for (row_idx, aspect), sentiment in zip(position, pred_sentiments):
            if sentiment not in ["positive", "negative", "neutral"]:
                sentiment = "neutral"
            lookup[row_idx][aspect] = sentiment

    return lookup


def build_absa_predictions_from_scores(df, aspect_labels, scores, sentiment_lookup, thresholds, fallback_mode):
    predicted_aspects = build_aspect_predictions_from_scores(
        df=df,
        aspect_labels=aspect_labels,
        scores=scores,
        thresholds=thresholds,
        fallback_mode=fallback_mode,
    )

    pred_dicts = [dict() for _ in range(len(df))]
    for row_idx, aspects in enumerate(predicted_aspects):
        for aspect in aspects:
            pred_dicts[row_idx][aspect] = sentiment_lookup[row_idx].get(aspect, "neutral")

    return pred_dicts


def build_absa_predictions(df, aspect_model, sentiment_model, mlb, thresholds, fallback_mode):
    scores = aspect_model.decision_function(df[FEATURE_COLS])
    sentiment_lookup = precompute_sentiment_predictions(df, sentiment_model, list(mlb.classes_))
    return build_absa_predictions_from_scores(
        df=df,
        aspect_labels=list(mlb.classes_),
        scores=scores,
        sentiment_lookup=sentiment_lookup,
        thresholds=thresholds,
        fallback_mode=fallback_mode,
    )


def _absa_score_tuple(overall_metrics):
    return (
        overall_metrics["micro_f1"],
        overall_metrics["recall"],
        overall_metrics["precision"],
    )


def tune_absa_thresholds_micro_f1(
    df,
    aspect_labels,
    scores,
    sentiment_lookup,
    threshold_values,
    initial_thresholds=None,
    initial_fallback_mode="only_if_empty",
    precision_guardrail=0.0,
    passes=2,
):
    thresholds = {aspect: 0.0 for aspect in aspect_labels}
    if initial_thresholds:
        thresholds.update(initial_thresholds)

    fallback_mode = initial_fallback_mode
    pred_dicts = build_absa_predictions_from_scores(
        df=df,
        aspect_labels=aspect_labels,
        scores=scores,
        sentiment_lookup=sentiment_lookup,
        thresholds=thresholds,
        fallback_mode=fallback_mode,
    )
    best_overall, _ = evaluate_absa_tuple_metrics(df, pred_dicts)

    for _ in range(passes):
        improved = False

        for candidate_mode in ["only_if_empty", "never", "always"]:
            candidate_pred_dicts = build_absa_predictions_from_scores(
                df=df,
                aspect_labels=aspect_labels,
                scores=scores,
                sentiment_lookup=sentiment_lookup,
                thresholds=thresholds,
                fallback_mode=candidate_mode,
            )
            candidate_overall, _ = evaluate_absa_tuple_metrics(df, candidate_pred_dicts)
            if (
                candidate_overall["precision"] + precision_guardrail >= best_overall["precision"]
                and _absa_score_tuple(candidate_overall) > _absa_score_tuple(best_overall)
            ):
                fallback_mode = candidate_mode
                best_overall = candidate_overall
                improved = True

        for i, aspect in enumerate(aspect_labels):
            best_threshold = thresholds[aspect]

            for threshold in threshold_values:
                candidate_thresholds = dict(thresholds)
                candidate_thresholds[aspect] = float(round(threshold, 2))
                candidate_pred_dicts = build_absa_predictions_from_scores(
                    df=df,
                    aspect_labels=aspect_labels,
                    scores=scores,
                    sentiment_lookup=sentiment_lookup,
                    thresholds=candidate_thresholds,
                    fallback_mode=fallback_mode,
                )
                candidate_overall, _ = evaluate_absa_tuple_metrics(df, candidate_pred_dicts)
                if (
                    candidate_overall["precision"] + precision_guardrail >= best_overall["precision"]
                    and _absa_score_tuple(candidate_overall) > _absa_score_tuple(best_overall)
                ):
                    best_threshold = float(round(threshold, 2))
                    best_overall = candidate_overall
                    improved = True

            thresholds[aspect] = best_threshold

        if not improved:
            break

    tuned_pred_dicts = build_absa_predictions_from_scores(
        df=df,
        aspect_labels=aspect_labels,
        scores=scores,
        sentiment_lookup=sentiment_lookup,
        thresholds=thresholds,
        fallback_mode=fallback_mode,
    )
    tuned_overall, tuned_per_aspect = evaluate_absa_tuple_metrics(df, tuned_pred_dicts)
    return thresholds, fallback_mode, tuned_overall, tuned_per_aspect


def evaluate_absa_tuple_metrics(df, pred_dicts):
    tp = fp = fn = 0
    aspect_stats = {
        aspect: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for aspect in TRAIN_ASPECTS
    }

    for idx, (_, row) in enumerate(df.iterrows()):
        gold = row["aspect_sentiments_dict"] if isinstance(row["aspect_sentiments_dict"], dict) else {}
        pred = pred_dicts[idx] if isinstance(pred_dicts[idx], dict) else {}

        gold_set = {(aspect, sentiment) for aspect, sentiment in gold.items()}
        pred_set = {(aspect, sentiment) for aspect, sentiment in pred.items()}

        tp_set = gold_set & pred_set
        fp_set = pred_set - gold_set
        fn_set = gold_set - pred_set

        tp += len(tp_set)
        fp += len(fp_set)
        fn += len(fn_set)

        for aspect, sentiment in gold_set:
            if aspect in aspect_stats:
                aspect_stats[aspect]["support"] += 1
        for aspect, sentiment in tp_set:
            if aspect in aspect_stats:
                aspect_stats[aspect]["tp"] += 1
        for aspect, sentiment in fp_set:
            if aspect in aspect_stats:
                aspect_stats[aspect]["fp"] += 1
        for aspect, sentiment in fn_set:
            if aspect in aspect_stats:
                aspect_stats[aspect]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    per_aspect_rows = []
    for aspect in TRAIN_ASPECTS:
        a_tp = aspect_stats[aspect]["tp"]
        a_fp = aspect_stats[aspect]["fp"]
        a_fn = aspect_stats[aspect]["fn"]
        a_prec = a_tp / (a_tp + a_fp) if (a_tp + a_fp) else 0.0
        a_rec = a_tp / (a_tp + a_fn) if (a_tp + a_fn) else 0.0
        a_f1 = (2 * a_prec * a_rec / (a_prec + a_rec)) if (a_prec + a_rec) else 0.0
        per_aspect_rows.append({
            "aspect": aspect,
            "support": aspect_stats[aspect]["support"],
            "precision": a_prec,
            "recall": a_rec,
            "f1": a_f1,
            "fp": a_fp,
            "fn": a_fn,
        })

    overall = {"precision": precision, "recall": recall, "micro_f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return overall, per_aspect_rows


def append_experiment_log(path, payload):
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.data_path:
        raw_df = load_table(args.data_path)
        if not has_label_columns(raw_df):
            raise SystemExit(
                f"Input file {args.data_path} does not contain labels. "
                "Precision, recall, and F1 cannot be computed on hidden-test data."
            )
        train_raw = raw_df
        val_raw = None
        test_raw = None
        if args.val_size > 0 or args.test_size > 0:
            train_raw, val_raw, test_raw = split_labeled_data(
                raw_df,
                val_size=args.val_size,
                test_size=args.test_size,
                seed=args.seed,
            )
    else:
        train_raw = load_table(args.train_path)
        val_raw = load_table(args.val_path) if args.val_path else None
        test_raw = None

        if not has_label_columns(train_raw):
            raise SystemExit(
                f"Training file {args.train_path} does not contain labels. "
                "Precision, recall, and F1 cannot be computed."
            )

        if val_raw is None:
            train_raw, val_raw, test_raw = split_labeled_data(
                train_raw,
                val_size=args.val_size,
                test_size=args.test_size,
                seed=args.seed,
            )

    print("\n[1/6] Loading data...")
    print(f"  train rows: {len(train_raw):,}")
    if val_raw is not None:
        print(f"  val rows:   {len(val_raw):,}")
    if test_raw is not None:
        print(f"  test rows:  {len(test_raw):,}")

    print("\n[2/6] Preprocessing...")
    train_df = prepare_dataset(train_raw, has_labels=True)
    train_df = remove_duplicate_reviews(train_df)
    print(f"  train after dedup: {len(train_df):,}")

    val_df = prepare_dataset(val_raw, has_labels=True) if val_raw is not None else None
    test_df = prepare_dataset(test_raw, has_labels=True) if test_raw is not None else None
    if val_df is not None:
        val_before = len(val_df)
        val_df = remove_duplicate_reviews(val_df)
        print(f"  val after dedup:   {len(val_df):,} (removed {val_before - len(val_df):,})")
    if test_df is not None:
        test_before = len(test_df)
        test_df = remove_duplicate_reviews(test_df)
        print(f"  test after dedup:  {len(test_df):,} (removed {test_before - len(test_df):,})")

    if not args.keep_split_overlaps:
        val_df, removed_val_overlap = drop_overlaps(train_df, val_df)
        test_df, removed_test_overlap = drop_overlaps(train_df, test_df)
        if removed_val_overlap:
            print(f"  removed train-val overlaps: {removed_val_overlap}")
        if removed_test_overlap:
            print(f"  removed train-test overlaps: {removed_test_overlap}")

    tv_overlap = overlap_count(train_df, val_df)
    tt_overlap = overlap_count(train_df, test_df)
    vt_overlap = overlap_count(val_df, test_df)
    if tv_overlap or tt_overlap or vt_overlap:
        print("  warning: duplicate cleaned reviews still overlap across splits")
        print(f"    train-val overlap: {tv_overlap}")
        print(f"    train-test overlap: {tt_overlap}")
        print(f"    val-test overlap: {vt_overlap}")

    if val_df is not None and val_df.empty:
        print("  warning: validation split became empty after overlap filtering; skipping validation metrics.")
        val_df = None
    if test_df is not None and test_df.empty:
        print("  warning: test split became empty after overlap filtering; skipping test metrics.")
        test_df = None

    val_tune_df = None
    val_eval_df = val_df
    if val_df is not None and args.threshold_strategy != "none":
        val_tune_df, val_eval_df = split_validation_for_threshold_tuning(
            val_df,
            tune_ratio=args.threshold_tune_ratio,
            seed=args.seed,
        )
        print(f"  val tune rows:  {len(val_tune_df):,}")
        print(f"  val eval rows:  {len(val_eval_df):,}")

    word_ngram_range = (1, max(1, args.word_ngram_max))
    char_ngram_range = (3, max(3, args.char_ngram_max))
    threshold_values = np.arange(args.threshold_min, args.threshold_max + args.threshold_step, args.threshold_step)

    print("\n  [Experiment settings]")
    print(f"  threshold_strategy={args.threshold_strategy}")
    print(f"  fallback_mode={args.fallback_mode}")
    print(f"  threshold_tune_ratio={args.threshold_tune_ratio}")
    print(f"  precision_guardrail={args.precision_guardrail}")
    print(f"  aspect_c={args.aspect_c}, sentiment_c={args.sentiment_c}")
    print(f"  svm_max_iter={args.svm_max_iter}")
    print(f"  word_ngram_range={word_ngram_range}, char_ngram_range={char_ngram_range}")
    print(f"  min_df={args.min_df}, max_df={args.max_df}, sublinear_tf={args.sublinear_tf}")

    print("\n[3/6] Training aspect detection model...")
    mlb = MultiLabelBinarizer(classes=TRAIN_ASPECTS)
    X_aspect = train_df[FEATURE_COLS]
    y_aspect = mlb.fit_transform(train_df["aspects_list"])

    aspect_model = build_aspect_model(
        c=args.aspect_c,
        max_iter=args.svm_max_iter,
        word_ngram_range=word_ngram_range,
        char_ngram_range=char_ngram_range,
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=args.sublinear_tf,
    )
    aspect_model.fit(X_aspect, y_aspect)

    aspect_thresholds = {aspect: 0.0 for aspect in TRAIN_ASPECTS}
    selected_fallback_mode = args.fallback_mode
    aspect_metrics_default = None
    aspect_metrics_tuned = None
    absa_tune_overall = None
    absa_tune_per_aspect = None

    if val_df is not None:
        print("\n  [Aspect model - validation results]")
        X_val_aspect_eval = val_eval_df[FEATURE_COLS]
        y_val_aspect_eval = mlb.transform(val_eval_df["aspects_list"])
        y_pred_aspect = aspect_model.predict(X_val_aspect_eval)

        aspect_metrics_default = multilabel_metrics(y_val_aspect_eval, y_pred_aspect)
        print_metric_table("  Aspect metrics (eval split)", aspect_metrics_default)
        print(classification_report(y_val_aspect_eval, y_pred_aspect, target_names=TRAIN_ASPECTS, zero_division=0))

        if args.threshold_strategy in {"binary_f1", "micro_f1"}:
            print("\n  [Tuning aspect thresholds on tune split]")
            X_val_aspect_tune = val_tune_df[FEATURE_COLS]
            y_val_aspect_tune = mlb.transform(val_tune_df["aspects_list"])

            if args.threshold_strategy == "binary_f1":
                tuned_thresholds, y_pred_tuned = tune_aspect_thresholds_binary_f1(
                    aspect_model,
                    X_val_aspect_tune,
                    y_val_aspect_tune,
                    TRAIN_ASPECTS,
                    threshold_values,
                )
            else:
                tuned_thresholds, y_pred_tuned = tune_aspect_thresholds_micro_f1(
                    aspect_model,
                    X_val_aspect_tune,
                    y_val_aspect_tune,
                    TRAIN_ASPECTS,
                    threshold_values,
                )

            eval_scores = aspect_model.decision_function(X_val_aspect_eval)
            y_pred_tuned_eval = np.zeros_like(y_val_aspect_eval)
            for i, aspect in enumerate(TRAIN_ASPECTS):
                y_pred_tuned_eval[:, i] = (eval_scores[:, i] >= tuned_thresholds.get(aspect, 0.0)).astype(int)

            aspect_metrics_tuned = multilabel_metrics(y_val_aspect_eval, y_pred_tuned_eval)
            print_metric_table("  Tuned aspect metrics (eval split)", aspect_metrics_tuned)

            default_micro = next(row["f1"] for row in aspect_metrics_default if row["average"] == "micro")
            tuned_micro = next(row["f1"] for row in aspect_metrics_tuned if row["average"] == "micro")
            default_precision = next(row["precision"] for row in aspect_metrics_default if row["average"] == "micro")
            tuned_precision = next(row["precision"] for row in aspect_metrics_tuned if row["average"] == "micro")
            default_recall = next(row["recall"] for row in aspect_metrics_default if row["average"] == "micro")
            tuned_recall = next(row["recall"] for row in aspect_metrics_tuned if row["average"] == "micro")

            if (tuned_micro, tuned_precision + args.precision_guardrail, tuned_recall) >= (
                default_micro,
                default_precision,
                default_recall,
            ):
                aspect_thresholds = tuned_thresholds
                print("  tuned thresholds selected.")
            else:
                print("  tuned thresholds rejected (default thresholds kept).")

    print("\n[4/6] Training sentiment model...")
    sentiment_train_df = build_sentiment_training_df(train_df)
    if sentiment_train_df.empty:
        raise SystemExit("No sentiment rows were produced from the training data.")

    print(f"  sentiment training rows: {len(sentiment_train_df):,}")
    print(f"  label distribution:\n{sentiment_train_df['aspect_sentiment'].value_counts().to_string()}")

    X_sent = sentiment_train_df[SENTIMENT_FEATURE_COLS]
    y_sent = sentiment_train_df["aspect_sentiment"]

    sentiment_model = build_sentiment_model(
        c=args.sentiment_c,
        max_iter=args.svm_max_iter,
        word_ngram_range=word_ngram_range,
        char_ngram_range=char_ngram_range,
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=args.sublinear_tf,
    )
    sentiment_model.fit(X_sent, y_sent)

    sentiment_rows = None
    if val_tune_df is not None and args.threshold_strategy == "absa_micro_f1":
        print("\n  [Tuning aspect thresholds for end-to-end ABSA micro-F1]")
        tune_scores = aspect_model.decision_function(val_tune_df[FEATURE_COLS])
        tune_sentiment_lookup = precompute_sentiment_predictions(val_tune_df, sentiment_model, TRAIN_ASPECTS)
        (
            tuned_thresholds,
            tuned_fallback_mode,
            absa_tune_overall,
            absa_tune_per_aspect,
        ) = tune_absa_thresholds_micro_f1(
            df=val_tune_df,
            aspect_labels=TRAIN_ASPECTS,
            scores=tune_scores,
            sentiment_lookup=tune_sentiment_lookup,
            threshold_values=threshold_values,
            initial_thresholds=aspect_thresholds,
            initial_fallback_mode=args.fallback_mode,
            precision_guardrail=args.precision_guardrail,
        )
        aspect_thresholds = tuned_thresholds
        selected_fallback_mode = tuned_fallback_mode
        print(pd.DataFrame([absa_tune_overall]).to_string(index=False))
        print_metric_table("  ABSA tune per-aspect metrics", absa_tune_per_aspect)
        print(f"  selected fallback_mode={selected_fallback_mode}")

    if val_eval_df is not None:
        val_sentiment_df = build_sentiment_training_df(val_eval_df)
        if not val_sentiment_df.empty:
            print("\n  [Sentiment model - eval split results]")
            X_val_sent = val_sentiment_df[SENTIMENT_FEATURE_COLS]
            y_val_sent = val_sentiment_df["aspect_sentiment"]
            y_pred_sent = sentiment_model.predict(X_val_sent)

            sentiment_rows = multiclass_metrics(y_val_sent, y_pred_sent)
            print_metric_table("  Sentiment metrics", sentiment_rows)
            print(classification_report(y_val_sent, y_pred_sent, zero_division=0))

        print("\n  [ABSA end-to-end validation metrics]")
        absa_pred_dicts = build_absa_predictions(
            val_eval_df,
            aspect_model=aspect_model,
            sentiment_model=sentiment_model,
            mlb=mlb,
            thresholds=aspect_thresholds,
            fallback_mode=selected_fallback_mode,
        )
        absa_overall, absa_per_aspect = evaluate_absa_tuple_metrics(val_eval_df, absa_pred_dicts)
        print(pd.DataFrame([absa_overall]).to_string(index=False))
        print_metric_table("  ABSA per-aspect metrics", absa_per_aspect)

        experiment_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_name": args.experiment_name or "train_run",
            "seed": args.seed,
            "train_path": args.train_path if not args.data_path else args.data_path,
            "val_path": args.val_path if not args.data_path else "split_from_data_path",
            "train_rows": len(train_df),
            "val_rows": len(val_df) if val_df is not None else 0,
            "val_tune_rows": len(val_tune_df) if val_tune_df is not None else 0,
            "val_eval_rows": len(val_eval_df) if val_eval_df is not None else 0,
            "threshold_strategy": args.threshold_strategy,
            "fallback_mode": selected_fallback_mode,
            "threshold_tune_ratio": args.threshold_tune_ratio,
            "precision_guardrail": args.precision_guardrail,
            "threshold_range": [args.threshold_min, args.threshold_max, args.threshold_step],
            "aspect_c": args.aspect_c,
            "sentiment_c": args.sentiment_c,
            "svm_max_iter": args.svm_max_iter,
            "word_ngram_range": list(word_ngram_range),
            "char_ngram_range": list(char_ngram_range),
            "min_df": args.min_df,
            "max_df": args.max_df,
            "sublinear_tf": args.sublinear_tf,
            "aspect_metrics_default": aspect_metrics_default,
            "aspect_metrics_tuned": aspect_metrics_tuned,
            "absa_tune_overall": absa_tune_overall,
            "absa_tune_per_aspect": absa_tune_per_aspect,
            "sentiment_metrics": sentiment_rows,
            "absa_overall": absa_overall,
            "absa_per_aspect": absa_per_aspect,
            "weights_dir": args.weights_dir,
        }
        append_experiment_log(args.experiment_log, experiment_payload)
        print(f"  experiment logged → {args.experiment_log}")

    if test_df is not None:
        test_sentiment_df = build_sentiment_training_df(test_df)
        if not test_sentiment_df.empty:
            print("\n  [Sentiment model - test results]")
            X_test_sent = test_sentiment_df[SENTIMENT_FEATURE_COLS]
            y_test_sent = test_sentiment_df["aspect_sentiment"]
            y_pred_test_sent = sentiment_model.predict(X_test_sent)
            print_metric_table("  Sentiment test metrics", multiclass_metrics(y_test_sent, y_pred_test_sent))

    print("\n[5/6] Saving weights...")
    aspect_path = os.path.join(args.weights_dir, os.path.basename(ASPECT_MODEL_PATH))
    sentiment_path = os.path.join(args.weights_dir, os.path.basename(SENTIMENT_MODEL_PATH))
    mlb_path = os.path.join(args.weights_dir, os.path.basename(MLB_PATH))
    thresholds_path = os.path.join(args.weights_dir, os.path.basename(ASPECT_THRESHOLDS_PATH))

    save_pickle(aspect_model, aspect_path)
    save_pickle(sentiment_model, sentiment_path)
    save_pickle(mlb, mlb_path)

    save_json(aspect_thresholds, thresholds_path)
    save_json(
        {
            "name": "aspect_detection_model",
            "type": "Pipeline(ColumnTransformer + OneVsRestClassifier(LinearSVC))",
            "labels": list(mlb.classes_),
            "feature_columns": FEATURE_COLS,
            "thresholds_file": os.path.basename(thresholds_path),
            "fallback_mode": selected_fallback_mode,
            "binary_weights_file": os.path.basename(aspect_path),
        },
        os.path.join(args.weights_dir, os.path.basename(ASPECT_MODEL_JSON_PATH)),
    )
    save_json(
        {
            "name": "aspect_sentiment_model",
            "type": "Pipeline(ColumnTransformer + LinearSVC)",
            "labels": ["positive", "negative", "neutral"],
            "feature_columns": SENTIMENT_FEATURE_COLS,
            "binary_weights_file": os.path.basename(sentiment_path),
        },
        os.path.join(args.weights_dir, os.path.basename(SENTIMENT_MODEL_JSON_PATH)),
    )
    save_json(
        {
            "name": "aspect_label_binarizer",
            "classes": list(mlb.classes_),
            "binary_weights_file": os.path.basename(mlb_path),
        },
        os.path.join(args.weights_dir, os.path.basename(MLB_JSON_PATH)),
    )

    print("\n[6/6] Done. All weights saved.")


if __name__ == "__main__":
    main()
