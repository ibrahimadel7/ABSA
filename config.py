"""
Configuration file for the Arabic ABSA system.

This file keeps the official challenge labels, feature columns, and default
project paths in one place. Keeping these values here reduces mistakes during
training, inference, and JSON submission generation.
"""

ASPECT_LABELS = [
    "food",
    "service",
    "price",
    "cleanliness",
    "delivery",
    "ambiance",
    "app_experience",
    "general",
    "none",
]

SENTIMENT_LABELS = [
    "positive",
    "negative",
    "neutral",
]

ALLOWED_ASPECTS = set(ASPECT_LABELS)
ALLOWED_SENTIMENTS = set(SENTIMENT_LABELS)

ASPECT_FEATURE_COLUMNS = [
    "review_text_clean",
    "business_category_grouped",
    "platform",
]

SENTIMENT_FEATURE_COLUMNS = [
    "review_text_clean",
    "business_category_grouped",
    "platform",
    "aspect",
]

MODEL_WEIGHTS_DIR = "saved_model_weights"
OUTPUT_DIR = "outputs"

ASPECT_MODEL_FILE = "aspect_detection_model.joblib"
SENTIMENT_MODEL_FILE = "aspect_sentiment_model.joblib"
LABEL_BINARIZER_FILE = "aspect_label_binarizer.joblib"