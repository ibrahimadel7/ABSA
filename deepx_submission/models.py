# =============================================================================
# models.py — DeepX ABSA: model architecture builders and predict functions
# =============================================================================

import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

try:
    from .preprocessing import build_keyword_feature_map, clean_review_text, safe_star_rating
    from .config import ASPECTS, ASPECT_KEYWORD_FEATURE_COLS, MULTILINGUAL_ASPECT_KEYWORDS, SENTIMENTS
except ImportError:
    from preprocessing import build_keyword_feature_map, clean_review_text, safe_star_rating
    from config import ASPECTS, ASPECT_KEYWORD_FEATURE_COLS, MULTILINGUAL_ASPECT_KEYWORDS, SENTIMENTS


# ── Aspect detection model ────────────────────────────────────────────────────

def _build_text_feature_union(
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> FeatureUnion:
    """Shared text featurization block used by both aspect and sentiment models."""
    return FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=word_ngram_range,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=sublinear_tf,
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=char_ngram_range,
                min_df=min_df,
                sublinear_tf=sublinear_tf,
            ),
        ),
    ])


def build_aspect_model(
    c: float = 1.0,
    max_iter: int = 5000,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> Pipeline:
    """
    Multi-label aspect detection pipeline.

    Features:
      text  → FeatureUnion(word TF-IDF 1-2gram + char TF-IDF 3-5gram)
      cat   → OneHotEncoder(business_category_grouped, platform)

    Classifier:
      OneVsRestClassifier(LinearSVC) — one binary SVM per aspect label
    """
    # Word features catch exact terms; character features catch dialect and typos.
    text_features = _build_text_feature_union(
        word_ngram_range=word_ngram_range,
        char_ngram_range=char_ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_features, "review_text_clean"),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["business_category_grouped", "platform"],
            ),
            ("numeric", "passthrough", ["star_rating_num"]),
            ("keyword_flags", "passthrough", ASPECT_KEYWORD_FEATURE_COLS),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                OneVsRestClassifier(LinearSVC(class_weight="balanced", C=c, max_iter=max_iter)),
            ),
        ]
    )


# ── Sentiment classification model ───────────────────────────────────────────

def build_sentiment_model(
    c: float = 1.0,
    max_iter: int = 5000,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> Pipeline:
    """
    Per-aspect sentiment classifier (positive / negative / neutral).

    Features:
      text  → FeatureUnion(word TF-IDF 1-2gram + char TF-IDF 3-5gram)
      cat   → OneHotEncoder(business_category_grouped, platform, aspect)
              ↑ 'aspect' is the key feature here — tells the model WHICH
                aspect it is scoring sentiment for

    Classifier:
      LinearSVC (multi-class, balanced weights)
    """
    # Keep the same text representation as the aspect model for consistency.
    text_features = _build_text_feature_union(
        word_ngram_range=word_ngram_range,
        char_ngram_range=char_ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_features, "review_text_clean"),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["business_category_grouped", "platform", "aspect"],
            ),
            ("numeric", "passthrough", ["star_rating_num"]),
            ("keyword_flags", "passthrough", ASPECT_KEYWORD_FEATURE_COLS),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LinearSVC(class_weight="balanced", C=c, max_iter=max_iter)),
        ]
    )


# ── Keyword fallback (defined here to keep preprocessing.py import-free) ─────

def keyword_aspect_fallback(text: str) -> list:
    """
    Rule-based fallback aspect detector using multilingual keyword lists.
    Used to supplement the ML model when confidence is low or aspects are missed.
    """
    text = str(text).lower()
    found = []
    for aspect, keywords in MULTILINGUAL_ASPECT_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text:
                found.append(aspect)
                break
    return found


def normalize_predicted_aspects(aspects: list) -> list:
    """Keep only valid aspects and avoid precision loss from spurious 'none' predictions."""
    valid = [aspect for aspect in aspects if aspect in ASPECTS]
    valid = list(dict.fromkeys(valid))

    # Treat 'none' as a noisy model artifact in this dataset and back off to general.
    valid = [aspect for aspect in valid if aspect != "none"]

    return valid if valid else ["general"]


def combine_aspect_candidates(model_aspects: list, fallback_aspects: list, fallback_mode: str = "always") -> list:
    """Merge model and keyword candidates with a controllable fallback policy."""
    model_aspects = list(dict.fromkeys(model_aspects or []))
    fallback_aspects = list(dict.fromkeys(fallback_aspects or []))

    if fallback_mode == "never":
        combined = model_aspects
    elif fallback_mode == "only_if_empty":
        combined = model_aspects if model_aspects else fallback_aspects
    else:
        combined = model_aspects + fallback_aspects

    return normalize_predicted_aspects(list(dict.fromkeys(combined)))


def predict_aspects_from_scores(aspect_model, mlb, sample: pd.DataFrame, thresholds=None) -> list:
    """Convert LinearSVC decision scores into aspect labels using tuned thresholds."""
    thresholds = thresholds or {}

    if hasattr(aspect_model, "decision_function"):
        scores = aspect_model.decision_function(sample)[0]
        aspects = [
            aspect
            for i, aspect in enumerate(mlb.classes_)
            if scores[i] >= thresholds.get(aspect, 0.0)
        ]
        return [aspect for aspect in aspects if aspect in ASPECTS]

    pred = aspect_model.predict(sample)
    return [aspect for aspect in mlb.inverse_transform(pred)[0] if aspect in ASPECTS]


# ── Inference functions ───────────────────────────────────────────────────────

def predict_aspects(
    review_text: str,
    aspect_model,
    mlb,
    business_category_grouped: str = "restaurant",
    platform: str = "google_maps",
    star_rating=None,
    thresholds: dict | None = None,
    fallback_mode: str = "always",
) -> list:
    """
    Predict which aspects are mentioned in a review.
    Combines ML model output with keyword-based fallback.
    Falls back to ['general'] if nothing is detected.
    """
    sample = pd.DataFrame({
        "review_text_clean":        [clean_review_text(review_text)],
        "business_category_grouped": [business_category_grouped],
        "platform":                  [platform],
        # A neutral midpoint works better than an out-of-range zero for free-text inference.
        "star_rating_num":           [safe_star_rating(3.0 if star_rating is None else star_rating)],
        **{column: [build_keyword_feature_map(review_text).get(column, 0.0)] for column in ASPECT_KEYWORD_FEATURE_COLS},
    })

    model_aspects = predict_aspects_from_scores(aspect_model, mlb, sample, thresholds)
    fallback = keyword_aspect_fallback(review_text) or []

    return combine_aspect_candidates(model_aspects, fallback, fallback_mode=fallback_mode)


def predict_sentiment_for_aspect(
    review_text: str,
    aspect: str,
    sentiment_model,
    business_category_grouped: str = "restaurant",
    platform: str = "google_maps",
    star_rating=None,
) -> str:
    """
    Predict sentiment (positive / negative / neutral) for a specific aspect
    within a review.
    """
    sample = pd.DataFrame({
        "review_text_clean":        [clean_review_text(review_text)],
        "business_category_grouped": [business_category_grouped],
        "platform":                  [platform],
        "aspect":                    [aspect],
        # Use a neutral default when the UI/API does not provide a real rating.
        "star_rating_num":           [safe_star_rating(3.0 if star_rating is None else star_rating)],
        **{column: [build_keyword_feature_map(review_text).get(column, 0.0)] for column in ASPECT_KEYWORD_FEATURE_COLS},
    })
    sentiment = sentiment_model.predict(sample)[0]
    return sentiment if sentiment in SENTIMENTS else "neutral"


def predict_absa(
    review_text: str,
    aspect_model,
    sentiment_model,
    mlb,
    business_category_grouped: str = "restaurant",
    platform: str = "google_maps",
    star_rating=None,
    thresholds: dict | None = None,
    fallback_mode: str = "always",
) -> dict:
    """
    Full ABSA pipeline for a single review:
      1. Detect which aspects are mentioned
      2. Predict sentiment for each detected aspect
      Returns: {aspect: sentiment, ...}
    """
    aspects = predict_aspects(
        review_text=review_text,
        aspect_model=aspect_model,
        mlb=mlb,
        business_category_grouped=business_category_grouped,
        platform=platform,
        star_rating=star_rating,
        thresholds=thresholds,
        fallback_mode=fallback_mode,
    )

    return {
        aspect: predict_sentiment_for_aspect(
            review_text=review_text,
            aspect=aspect,
            sentiment_model=sentiment_model,
            business_category_grouped=business_category_grouped,
            platform=platform,
            star_rating=star_rating,
        )
        for aspect in aspects
    }
