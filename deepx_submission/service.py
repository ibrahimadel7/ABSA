"""Reusable inference service for ABSA predictions."""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import joblib

try:
    from .models import predict_absa
    from .preprocessing import clean_review_text, group_business_category, safe_star_rating
except ImportError:
    from models import predict_absa
    from preprocessing import clean_review_text, group_business_category, safe_star_rating


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = BASE_DIR / "weights"
DEFAULT_BUSINESS_CATEGORY = "مطعم"
DEFAULT_PLATFORM = "google_maps"


def _load_thresholds(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class ABSAService:
    """Load trained artifacts once and serve ABSA predictions."""

    def __init__(self, weights_dir: Path | None = None, fallback_mode: str = "only_if_empty") -> None:
        self.weights_dir = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        self.fallback_mode = fallback_mode
        self.aspect_model = joblib.load(self.weights_dir / "aspect_model.joblib")
        self.sentiment_model = joblib.load(self.weights_dir / "sentiment_model.joblib")
        self.mlb = joblib.load(self.weights_dir / "mlb.joblib")
        self.thresholds = _load_thresholds(self.weights_dir / "aspect_thresholds.json")

    def predict(
        self,
        review_text: str,
        business_category: str = DEFAULT_BUSINESS_CATEGORY,
        platform: str = DEFAULT_PLATFORM,
        star_rating=None,
    ) -> dict:
        review_text = str(review_text or "").strip()
        if not review_text:
            raise ValueError("review_text must not be empty")

        business_category = str(business_category or DEFAULT_BUSINESS_CATEGORY).strip() or DEFAULT_BUSINESS_CATEGORY
        platform = str(platform or DEFAULT_PLATFORM).strip() or DEFAULT_PLATFORM

        cleaned_review = clean_review_text(review_text)
        grouped_category = group_business_category(business_category)
        star_rating_num = safe_star_rating(3.0 if star_rating is None else star_rating)
        aspect_sentiments = predict_absa(
            review_text=review_text,
            aspect_model=self.aspect_model,
            sentiment_model=self.sentiment_model,
            mlb=self.mlb,
            business_category_grouped=grouped_category,
            platform=platform,
            star_rating=star_rating_num,
            thresholds=self.thresholds,
            fallback_mode=self.fallback_mode,
        )
        aspects = list(aspect_sentiments.keys())
        return {
            "review_text": review_text,
            "review_text_clean": cleaned_review,
            "business_category": business_category,
            "business_category_grouped": grouped_category,
            "platform": platform,
            "star_rating_num": star_rating_num,
            "overall_sentiment": self._summarize_sentiment(aspect_sentiments),
            "aspects": aspects,
            "aspect_sentiments": aspect_sentiments,
        }

    def predict_batch(self, items: list[dict]) -> list[dict]:
        return [
            self.predict(
                review_text=item.get("review_text", ""),
                business_category=item.get("business_category", DEFAULT_BUSINESS_CATEGORY),
                platform=item.get("platform", DEFAULT_PLATFORM),
                star_rating=item.get("star_rating"),
            )
            for item in items
        ]

    @staticmethod
    def _summarize_sentiment(aspect_sentiments: dict[str, str]) -> str:
        if not aspect_sentiments:
            return "neutral"
        counts = Counter(aspect_sentiments.values())
        if counts["negative"] > counts["positive"]:
            return "negative"
        if counts["positive"] > counts["negative"]:
            return "positive"
        return "neutral"


@lru_cache(maxsize=1)
def get_service() -> ABSAService:
    return ABSAService()
