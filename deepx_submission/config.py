"""
Configuration file for the Arabic ABSA system.

This file keeps the official challenge labels, feature columns, and default
project paths in one place. Keeping these values here reduces mistakes during
training, inference, and JSON submission generation.
"""

import os

# ── Paths 
DATA_DIR  = os.environ.get("DATA_DIR", "data")
TRAIN_PATH  = os.path.join(DATA_DIR, "DeepX_train.xlsx")
VAL_PATH  = os.path.join(DATA_DIR, "DeepX_validation.xlsx")
UNLABELED_PATH = os.path.join(DATA_DIR, "DeepX_unlabeled.xlsx")

WEIGHTS_DIR = "weights"
ASPECT_MODEL_PATH = os.path.join(WEIGHTS_DIR, "aspect_model.joblib")
SENTIMENT_MODEL_PATH = os.path.join(WEIGHTS_DIR, "sentiment_model.joblib")
MLB_PATH = os.path.join(WEIGHTS_DIR, "mlb.joblib")
ASPECT_THRESHOLDS_PATH = os.path.join(WEIGHTS_DIR, "aspect_thresholds.json")
ASPECT_MODEL_JSON_PATH = os.path.join(WEIGHTS_DIR, "aspect_model.json")
SENTIMENT_MODEL_JSON_PATH = os.path.join(WEIGHTS_DIR, "sentiment_model.json")
MLB_JSON_PATH = os.path.join(WEIGHTS_DIR, "mlb.json")

SUBMISSION_PATH  = "submission.json"

#  Aspects & sentiments 
ASPECTS = [
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

SENTIMENTS = ["positive", "negative", "neutral"]

#  Feature column names 
FEATURE_COLS = [
    "review_text_clean",
    "business_category_grouped",
    "platform",
    "star_rating_num",
]

SENTIMENT_FEATURE_COLS = [
    "review_text_clean",
    "business_category_grouped",
    "platform",
    "star_rating_num",
    "aspect",
]

ASPECT_KEYWORD_FEATURE_COLS = [
    f"kw_{aspect}" for aspect in ASPECTS if aspect != "none"
]

FEATURE_COLS = FEATURE_COLS + ASPECT_KEYWORD_FEATURE_COLS
SENTIMENT_FEATURE_COLS = SENTIMENT_FEATURE_COLS + ASPECT_KEYWORD_FEATURE_COLS

#  Multilingual aspect keyword fallback 
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
        "limpio", "sucio", "propre", "sale", "sauber", "schmutzig", "pulito", "sporco", "temiz", "kirli",
    ],
    "ambiance": [
        "مكان", "جو", "قعدة", "رايق", "جميل", "منظر", "ديكور", "زحمة", "دوشة",
        "place", "ambience", "ambiance", "atmosphere", "view", "decor", "noisy", "lobby",
        "lugar", "ambiente", "bonito", "endroit", "belle", "ort", "atmosphäre",
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
