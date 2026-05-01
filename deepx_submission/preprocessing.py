"""
Text and metadata preprocessing for the ABSA pipeline.

The goal here is not to make the text look perfect. The goal is to make noisy
customer reviews consistent enough for the model while keeping useful signals
such as emojis, punctuation, English words, and multilingual text.
"""

import re
import json
import ast
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

try:
    from .config import ASPECTS, MULTILINGUAL_ASPECT_KEYWORDS, SENTIMENTS
except ImportError:
    from config import ASPECTS, MULTILINGUAL_ASPECT_KEYWORDS, SENTIMENTS

REFERENCE_DATE = pd.Timestamp("2026-03-15")


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_review_text(text: str) -> str:
    """
    Full Arabic text cleaning pipeline:
      1. Remove URLs
      2. Remove Arabic diacritics (tashkeel)
      3. Normalize alef variants (أإآا → ا)
      4. Normalize ى → ي, ؤ → و, ئ → ي
      5. Replace newlines/tabs with space
      6. Collapse extra whitespace
    """
    text = str(text)

    # 1. Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # 2. Remove Arabic diacritics (harakat / tashkeel)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # 3. Normalize alef variants
    text = re.sub(r"[إأآا]", "ا", text)

    # 4. Normalize other letter variants
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)

    # 5. Replace newlines / tabs with single space
    text = re.sub(r"[\n\r\t]+", " ", text)

    # 6. Collapse extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def safe_star_rating(value) -> float:
    """Convert ratings to a bounded numeric feature and preserve missingness as 0."""
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0.0
    return float(min(5.0, max(0.0, numeric)))


def build_keyword_feature_map(text: str) -> dict:
    """Create one binary feature per aspect based on multilingual keyword hits."""
    text = str(text).lower()
    feature_map = {}
    for aspect in ASPECTS:
        if aspect == "none":
            continue
        feature_map[f"kw_{aspect}"] = float(
            any(keyword.lower() in text for keyword in MULTILINGUAL_ASPECT_KEYWORDS.get(aspect, []))
        )
    return feature_map


# ── Business category grouping ────────────────────────────────────────────────

def group_business_category(category: str) -> str:
    """
    Maps raw Arabic/English business category strings to a canonical
    set of grouped labels used as model features.
    """
    category = str(category).strip()

    if "مطعم" in category:
        return "restaurant"
    if category in ["كافيه", "مقهى"]:
        return "cafe"
    if category == "ecommerce":
        return "ecommerce"
    if category == "food_delivery":
        return "food_delivery"
    if category == "entertainment":
        return "entertainment"
    if category == "travel":
        return "travel"
    if category == "transport":
        return "transport"
    if category == "real_estate":
        return "real_estate"
    if "فندق" in category:
        return "hotel"
    if any(kw in category for kw in ["مستشفى", "عيادة", "طبيب", "مركز طبي", "صيدلية"]):
        return "healthcare"
    if any(kw in category for kw in ["متجر", "سوبرماركت", "سوق", "منفذ بيع", "مول"]):
        return "retail"
    if any(kw in category for kw in ["صالون", "حلاقة", "مصفف", "أظافر"]):
        return "beauty"
    if any(kw in category for kw in ["رياضة", "لياقة", "غرفة لياقة", "جيم"]):
        return "fitness"

    return "other"


# ── Date cleaning 

def clean_arabic_relative_date(value, reference_date=REFERENCE_DATE):
    """─────────────────────────────────────────────────────────────
    Parses Arabic relative date strings (e.g. 'قبل يومين', 'قبل شهر')
    into absolute timestamps using the reference date.
    """
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.to_datetime(value)
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip().replace("تاريخ التعديل:", "").strip()

    if text == "الآن":
        return reference_date

    match = re.search(r"\d+", text)
    number = int(match.group()) if match else 1

    # Handle dual forms (Arabic grammar)
    if "يومين"   in text: number = 2
    elif "أسبوعين" in text: number = 2
    elif "شهرين"  in text: number = 2
    elif "سنتين"  in text: number = 2
    elif "يوم واحد" in text: number = 1

    if "دقيقة"   in text: return reference_date - pd.Timedelta(minutes=number)
    if "ساعة"    in text or "ساعات" in text: return reference_date - pd.Timedelta(hours=number)
    if "يوم"     in text or "أيام"  in text: return reference_date - pd.Timedelta(days=number)
    if "أسبوع"   in text or "أسابيع" in text: return reference_date - pd.Timedelta(weeks=number)
    if "شهر"     in text or "أشهر"  in text: return reference_date - relativedelta(months=number)
    if "سنة"     in text or "سنوات" in text or "عام" in text:
        return reference_date - relativedelta(years=number)

    return pd.to_datetime(value, errors="coerce")


# ── Label parsers 

def parse_list(value) -> list:
    """Safely parse a JSON/Python-literal list from a cell value."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # Support competition files where aspects are stored as comma-separated text.
        if "," in text and not text.startswith("["):
            return [part.strip() for part in text.split(",") if part.strip()]
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass

    text = str(value).strip()
    if text and text.lower() != "nan":
        return [text]
    return []


def parse_dict(value) -> dict:
    """Safely parse a JSON/Python-literal dict from a cell value."""
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    try:
        return json.loads(value)
    except Exception:
        try:
            return ast.literal_eval(value)
        except Exception:
            return {}


def normalize_labels(aspects_list: list, sentiments_dict: dict) -> tuple[list, dict]:
    """Normalize and align noisy label fields without dropping rows."""
    norm_aspects = [str(a).strip().lower() for a in aspects_list if str(a).strip()]

    raw_norm_dict = {}
    for aspect, sentiment in (sentiments_dict or {}).items():
        a = str(aspect).strip().lower()
        s = str(sentiment).strip().lower()
        if s not in SENTIMENTS:
            s = "neutral"
        raw_norm_dict[a] = s

    # Common cleaned-data inconsistency: aspects='general' while dict has {'none': 'neutral'}.
    if len(norm_aspects) == 1 and len(raw_norm_dict) == 1:
        only_aspect = norm_aspects[0]
        only_dict_aspect = next(iter(raw_norm_dict.keys()))
        if {only_aspect, only_dict_aspect} == {"general", "none"}:
            only_sentiment = next(iter(raw_norm_dict.values()))
            raw_norm_dict = {only_aspect: only_sentiment}

    # Keep labels aligned with inference behavior where 'none' is treated as noise.
    norm_aspects = [a for a in norm_aspects if a in ASPECTS and a != "none"]

    norm_dict = {}
    for a, s in raw_norm_dict.items():
        if a not in ASPECTS or a == "none":
            continue
        norm_dict[a] = s

    return norm_aspects, norm_dict


# ── Dataset preparation ───────────────────────────────────────────────────────

def prepare_dataset(df: pd.DataFrame, has_labels: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline applied to any split (train / val / unlabeled).
    Steps:
      - Clean review text
      - Group business categories
      - Fill missing feature values
      - Parse list/dict label columns (if has_labels=True)
    """
    df = df.copy()

    df["review_text_clean"]        = df["review_text"].apply(clean_review_text)
    df["business_category_grouped"] = df["business_category"].apply(group_business_category)
    df["review_text_clean"]         = df["review_text_clean"].fillna("")
    df["business_category_grouped"] = df["business_category_grouped"].fillna("unknown")
    df["platform"]                  = df["platform"].fillna("unknown")
    df["star_rating_num"]           = df.get("star_rating", 0).apply(safe_star_rating)

    keyword_feature_rows = df["review_text"].fillna("").apply(build_keyword_feature_map)
    keyword_feature_df = pd.DataFrame(keyword_feature_rows.tolist(), index=df.index).fillna(0.0)
    df = pd.concat([df, keyword_feature_df], axis=1)

    if has_labels:
        df["aspects_list"] = df["aspects"].apply(parse_list)
        df["aspect_sentiments_dict"] = df["aspect_sentiments"].apply(parse_dict)

        aligned = [
            normalize_labels(aspects_list, sentiments_dict)
            for aspects_list, sentiments_dict in zip(df["aspects_list"], df["aspect_sentiments_dict"])
        ]
        df["aspects_list"] = [x[0] for x in aligned]
        df["aspect_sentiments_dict"] = [x[1] for x in aligned]

    return df


def remove_duplicate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate review texts, keeping the first occurrence."""
    return df.drop_duplicates(subset=["review_text_clean"], keep="first").reset_index(drop=True)


def build_sentiment_training_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explodes the per-review aspect_sentiments_dict into one row per
    (review, aspect, sentiment) triplet — the training format for the
    sentiment classifier.
    """
    rows = []
    for _, row in df.iterrows():
        for aspect, sentiment in row["aspect_sentiments_dict"].items():
            if aspect == "none":
                continue
            rows.append({
                "review_text_clean":        row["review_text_clean"],
                "business_category_grouped": row["business_category_grouped"],
                "platform":                 row["platform"],
                "star_rating_num":          row.get("star_rating_num", 0.0),
                "aspect":                   aspect,
                "aspect_sentiment":         sentiment,
                **{col: row.get(col, 0.0) for col in df.columns if col.startswith("kw_")},
            })
    return pd.DataFrame(rows)
