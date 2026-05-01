"""
Model definitions for the ABSA system.

The system uses two models:
1. Aspect detection model: predicts which aspects are mentioned.
2. Aspect sentiment model: predicts positive, negative, or neutral for each aspect.

Both models use word-level and character-level TF-IDF. Character TF-IDF is useful
for Arabic dialects, spelling variation, typos, and multilingual reviews.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

def build_text_vectorizer():
    """
    Build combined word and character TF-IDF features.

    Word TF-IDF captures meaningful terms.
    Character TF-IDF helps with typos, dialect, short reviews, and foreign words.
    """
    return FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
            ),
        ),
    ])


def build_aspect_detection_model():
    """
    Build the aspect detection pipeline.

    This is a multi-label classifier. A review may mention several aspects,
    such as food, service, and price at the same time.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", build_text_vectorizer(), "review_text_clean"),
            (
                "category_features",
                OneHotEncoder(handle_unknown="ignore"),
                ["business_category_grouped", "platform"],
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", OneVsRestClassifier(
                LinearSVC(class_weight="balanced")
            )),
        ]
    )


def build_aspect_sentiment_model():
    """
    Build the aspect sentiment pipeline.

    This model receives a review and one aspect, then predicts whether the
    sentiment for that aspect is positive, negative, or neutral.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", build_text_vectorizer(), "review_text_clean"),
            (
                "category_features",
                OneHotEncoder(handle_unknown="ignore"),
                ["business_category_grouped", "platform", "aspect"],
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LinearSVC(class_weight="balanced")),
        ]
    )


def build_aspect_sentiment_training_table(df):
    """
    Convert review-level labels into review-aspect rows.

    Original format:
        one row = one review with a dictionary of aspect sentiments

    New format:
        one row = one review + one aspect + one sentiment label

    This makes the second model easier to train.
    """
    rows = []

    for _, row in df.iterrows():
        for aspect, sentiment in row["aspect_sentiments_dict"].items():
            rows.append({
                "review_text_clean": row["review_text_clean"],
                "business_category_grouped": row["business_category_grouped"],
                "platform": row["platform"],
                "aspect": aspect,
                "aspect_sentiment": sentiment,
            })

    return pd.DataFrame(rows)