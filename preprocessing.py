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


def clean_review_text(text):
    """
    Clean review text while preserving sentiment signals.

    We remove URLs, Arabic diacritics, repeated spaces, and line breaks.
    We keep emojis and non-Arabic words because the hidden test contains
    mixed-language reviews.
    """
    text = str(text)

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)

    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def group_business_category(category):
    """
    Convert detailed business categories into broader sector groups.

    This helps the model generalise better instead of memorising very specific
    labels such as individual restaurant types or clinic types.
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

    if (
        "مستشفى" in category
        or "عيادة" in category
        or "طبيب" in category
        or "مركز طبي" in category
        or "صيدلية" in category
    ):
        return "healthcare"

    if (
        "متجر" in category
        or "سوبرماركت" in category
        or "سوق" in category
        or "منفذ بيع" in category
        or "مول" in category
    ):
        return "retail"

    if (
        "صالون" in category
        or "حلاقة" in category
        or "مصفف" in category
        or "أظافر" in category
    ):
        return "beauty"

    if (
        "رياضة" in category
        or "لياقة" in category
        or "غرفة لياقة" in category
        or "جيم" in category
    ):
        return "fitness"

    return "other"


def parse_list(value):
    """
    Parse a stringified list safely.

    The training labels may appear as strings like:
    '["food", "service"]'
    This function converts them into real Python lists.
    """
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        return json.loads(value)
    except Exception:
        try:
            return ast.literal_eval(value)
        except Exception:
            return []


def parse_dict(value):
    """
    Parse a stringified dictionary safely.

    The aspect sentiment labels may appear as strings like:
    '{"food": "positive"}'
    This function converts them into real Python dictionaries.
    """
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


def prepare_absa_dataframe(df, has_labels=True):
    """
    Apply the shared preprocessing steps to train, validation, or test data.

    For labeled data, this also parses aspects and aspect_sentiments.
    For hidden test data, labels are not expected, so only features are prepared.
    """
    df = df.copy()

    df["review_text_clean"] = df["review_text"].apply(clean_review_text)
    df["business_category_grouped"] = df["business_category"].apply(group_business_category)

    df["review_text_clean"] = df["review_text_clean"].fillna("")
    df["business_category_grouped"] = df["business_category_grouped"].fillna("unknown")
    df["platform"] = df["platform"].fillna("unknown")

    if has_labels:
        df["aspects_list"] = df["aspects"].apply(parse_list)
        df["aspect_sentiments_dict"] = df["aspect_sentiments"].apply(parse_dict)

    return df


def remove_duplicate_review_texts(df):
    """
    Remove duplicate cleaned review texts from the training set only.

    This reduces over-learning from repeated short reviews such as 'ممتاز',
    'جميل', or 'Nice'. It should not be used on validation or hidden test data.
    """
    return df.drop_duplicates(subset=["review_text_clean"], keep="first").reset_index(drop=True)


MULTILINGUAL_ASPECT_KEYWORDS = {
    "food": [
        "اكل", "الأكل", "طعام", "الطعام", "وجبة", "بيتزا", "كريب", "رز", "لحم", "فراخ",
        "food", "meal", "pizza", "burger", "chicken", "meat", "taste",
        "comida", "nourriture", "repas", "essen", "cibo", "yemek",
    ],
    "service": [
        "خدمة", "الخدمة", "موظف", "الموظفين", "تعامل", "استقبال", "انتظار",
        "service", "staff", "waiter", "rude", "friendly", "helpful",
        "servicio", "personal", "personnel", "mitarbeiter", "servizio", "servis", "personel",
    ],
    "price": [
        "سعر", "السعر", "اسعار", "الأسعار", "غالي", "رخيص", "فلوس",
        "price", "expensive", "cheap", "cost", "bill", "overpriced",
        "caro", "barato", "precio", "cher", "prix", "teuer", "preis", "prezzo", "pahalı", "fiyat",
    ],
    "cleanliness": [
        "نظيف", "نضيف", "نظافة", "النظافة", "وسخ", "قذر",
        "clean", "dirty", "toilet", "bathroom", "spotless",
        "limpio", "sucio", "propre", "sale", "sauber", "schmutzig",
        "pulito", "sporco", "temiz", "kirli",
    ],
    "ambiance": [
        "مكان", "جو", "قعدة", "رايق", "جميل", "منظر", "ديكور", "زحمة", "دوشة",
        "place", "ambience", "ambiance", "atmosphere", "view", "decor", "noisy", "lobby",
        "lugar", "ambiente", "bonito", "endroit", "belle", "ort", "atmosphäre", "aussicht",
        "posto", "atmosfera", "bello", "yer", "manzara",
    ],
    "delivery": [
        "توصيل", "الدليفري", "اوردر", "طلب", "اتأخر", "تأخير",
        "delivery", "order", "late", "delayed", "arrived",
        "entrega", "pedido", "livraison", "commande", "lieferung", "bestellung",
        "consegna", "ordine", "teslimat", "sipariş",
    ],
    "app_experience": [
        "تطبيق", "الابلكيشن", "ابلكيشن", "بيهنج", "كراش", "يفتح",
        "app", "application", "crash", "login", "bug", "checkout", "failed",
        "aplicación", "anwendung", "applicazione", "uygulama", "donuyor",
    ],
}


def keyword_aspect_fallback(text):
    """
    Add a light multilingual safety net for obvious aspect words.

    This helps when the hidden test contains reviews in languages that are not
    strongly represented in training, such as German, French, Spanish, Turkish,
    or Italian.
    """
    text = str(text).lower()
    found_aspects = []

    for aspect, keywords in MULTILINGUAL_ASPECT_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text:
                found_aspects.append(aspect)
                break

    return found_aspects