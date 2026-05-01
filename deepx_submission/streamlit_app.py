"""Streamlit frontend for single-review and uploaded-dataset ABSA analysis."""

from __future__ import annotations

import json
import os
from collections import Counter

import pandas as pd
import requests
import streamlit as st

try:
    from .preprocessing import group_business_category
except ImportError:
    from preprocessing import group_business_category

try:
    from .service import get_service as _get_service
except Exception:
    _get_service = None


def _resolve_api_base_url() -> str:
    configured = (
        os.environ.get("ABSA_API_BASE_URL")
        or os.environ.get("ABSA_API_URL")
        or "http://127.0.0.1:8000"
    ).rstrip("/")
    if configured.endswith("/predict"):
        return configured[: -len("/predict")]
    return configured


API_BASE_URL = _resolve_api_base_url()
PREDICT_URL = f"{API_BASE_URL}/predict"
BATCH_PREDICT_URL = f"{API_BASE_URL}/predict/batch"
HEALTH_URL = f"{API_BASE_URL}/health"
BATCH_REQUEST_SIZE = 50
DEFAULT_CATEGORY = "مطعم"
DEFAULT_PLATFORM = "google_maps"
SENTIMENT_COLORS = {
    "positive": "#D4A373",
    "neutral": "#936639",
    "negative": "#5C2A21",
}
ASPECT_LABELS = {
    "food": "Food",
    "service": "Service",
    "price": "Price",
    "cleanliness": "Cleanliness",
    "delivery": "Delivery",
    "ambiance": "Ambiance",
    "app_experience": "App Experience",
    "general": "General",
    "none": "None",
}


st.set_page_config(
    page_title="DeepX ABSA Studio",
    page_icon=":material/rate_review:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

        :root {
            --bg: #fafaf5;
            --bg-soft: #f3ede4;
            --paper: rgba(255, 252, 247, 0.96);
            --paper-strong: #fffdf9;
            --ink: #2b211d;
            --muted: #6d5a4d;
            --line: rgba(92, 42, 33, 0.14);
            --line-strong: rgba(92, 42, 33, 0.26);
            --primary: #5c2a21;
            --secondary: #d4a373;
            --tertiary: #936639;
            --shadow: 0 24px 60px rgba(92, 42, 33, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(212, 163, 115, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(147, 102, 57, 0.14), transparent 24%),
                linear-gradient(180deg, #fffdf9 0%, var(--bg-soft) 100%);
            color: var(--ink);
            font-family: "Manrope", "Segoe UI", sans-serif;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        h1, h2, h3 {
            font-family: "Fraunces", Georgia, serif;
            color: var(--ink);
            letter-spacing: -0.03em;
        }

        .hero {
            background:
                radial-gradient(circle at top right, rgba(212, 163, 115, 0.16), transparent 28%),
                linear-gradient(135deg, #472118 0%, var(--primary) 52%, var(--tertiary) 100%);
            color: #fffaf4;
            border-radius: 30px;
            padding: 2rem 2rem 1.8rem;
            box-shadow: var(--shadow);
            border: 1px solid rgba(255, 250, 244, 0.10);
            margin-bottom: 1.2rem;
        }

        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.78rem;
            opacity: 0.78;
            margin-bottom: 0.9rem;
        }

        .hero-title {
            font-size: 4rem;
            line-height: 0.94;
            margin: 0;
            color: #fffdf8;
        }

        .hero-copy {
            max-width: 720px;
            margin-top: 0.9rem;
            color: rgba(255, 248, 239, 0.84);
            font-size: 1.02rem;
            line-height: 1.7;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-top: 1.2rem;
        }

        .hero-badge {
            background: rgba(255, 250, 244, 0.10);
            border: 1px solid rgba(255, 250, 244, 0.16);
            border-radius: 999px;
            padding: 0.48rem 0.82rem;
            font-size: 0.84rem;
        }

        .panel {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1.3rem;
            box-shadow: var(--shadow);
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1.3rem;
            box-shadow: var(--shadow);
        }

        [data-testid="stVerticalBlockBorderWrapper"] > div {
            background: transparent;
        }

        .subtle-panel {
            background: rgba(255, 250, 245, 0.82);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem;
        }

        .metric-card {
            background: var(--paper-strong);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            min-height: 126px;
        }

        .metric-label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.76rem;
            margin-bottom: 0.75rem;
        }

        .metric-value {
            font-family: "Fraunces", Georgia, serif;
            font-size: 2.3rem;
            line-height: 1;
            margin-bottom: 0.3rem;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .result-card {
            background: linear-gradient(180deg, rgba(255, 253, 249, 0.98), rgba(246, 239, 230, 0.98));
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.15rem;
        }

        .review-quote {
            font-size: 1.05rem;
            line-height: 1.8;
            color: var(--ink);
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.9rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.42rem 0.78rem;
            background: rgba(212, 163, 115, 0.18);
            color: var(--primary);
            border: 1px solid rgba(147, 102, 57, 0.18);
            font-weight: 700;
            font-size: 0.84rem;
        }

        .aspect-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 1rem;
            align-items: center;
            padding: 0.8rem 0;
            border-top: 1px solid rgba(92, 42, 33, 0.10);
        }

        .aspect-row:first-child {
            border-top: none;
            padding-top: 0;
        }

        .aspect-bar {
            height: 8px;
            border-radius: 999px;
            background: rgba(147, 102, 57, 0.12);
            overflow: hidden;
            margin-top: 0.45rem;
        }

        .aspect-bar > span {
            display: block;
            height: 100%;
            background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        }

        .section-note {
            color: var(--muted);
            margin-top: 0.2rem;
            margin-bottom: 1rem;
            line-height: 1.65;
        }

        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button {
            border-radius: 16px !important;
            border: 1px solid rgba(92, 42, 33, 0.10) !important;
            background: linear-gradient(135deg, var(--primary) 0%, var(--tertiary) 100%) !important;
            color: #fffaf4 !important;
            font-weight: 700 !important;
            padding: 0.7rem 1rem !important;
            box-shadow: 0 10px 20px rgba(92, 42, 33, 0.12) !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {
            border-color: rgba(92, 42, 33, 0.16) !important;
            filter: brightness(1.02);
        }

        .stTextArea [data-baseweb="textarea"],
        .stTextInput [data-baseweb="base-input"],
        .stNumberInput [data-baseweb="base-input"],
        .stSelectbox [data-baseweb="select"] > div,
        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            border-radius: 18px !important;
            background: rgba(255, 253, 249, 0.92) !important;
            border: 1px solid var(--line) !important;
            color: var(--ink) !important;
            box-shadow: none !important;
            transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
        }

        .stTextArea textarea,
        .stTextInput input,
        .stNumberInput input {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: var(--ink) !important;
            caret-color: var(--primary) !important;
        }

        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder,
        .stNumberInput input::placeholder {
            color: rgba(109, 90, 77, 0.72) !important;
            -webkit-text-fill-color: rgba(109, 90, 77, 0.72) !important;
            opacity: 1 !important;
        }

        .stTextArea [data-baseweb="textarea"]:focus-within,
        .stTextInput [data-baseweb="base-input"]:focus-within,
        .stNumberInput [data-baseweb="base-input"]:focus-within,
        .stSelectbox [data-baseweb="select"] > div:focus-within,
        .stFileUploader [data-testid="stFileUploaderDropzone"]:focus-within {
            border-color: var(--line-strong) !important;
            box-shadow: 0 0 0 3px rgba(212, 163, 115, 0.16) !important;
        }

        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] div,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderFileName"] {
            color: var(--ink) !important;
            opacity: 1 !important;
        }

        .stTextArea label,
        .stTextInput label,
        .stNumberInput label,
        .stSelectbox label,
        .stFileUploader label {
            color: var(--ink) !important;
        }

        .stSelectbox [data-baseweb="select"] * {
            color: var(--ink) !important;
            box-shadow: none !important;
        }

        .stSelectbox svg,
        .stNumberInput button svg {
            fill: var(--tertiary) !important;
        }

        [data-testid="stTabs"] {
            margin-top: 0.2rem;
        }

        [data-testid="stTabs"] > div:first-child {
            border-bottom: none !important;
        }

        div[data-baseweb="tab-list"] {
            align-items: center;
            gap: 0.5rem;
            padding: 0 !important;
            margin: 0 !important;
            border-bottom: none !important;
        }

        div[data-baseweb="tab-border"] {
            display: none !important;
        }

        div[data-baseweb="tab-highlight"] {
            background: transparent !important;
        }

        button[data-baseweb="tab"] {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 2.1rem;
            background: rgba(255, 252, 247, 0.68);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.55rem 0.9rem;
            color: var(--muted);
            font-weight: 700;
            box-shadow: none !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: var(--primary);
            border-color: var(--primary);
            color: #fffaf4;
        }

        button[data-baseweb="tab"] p {
            color: inherit !important;
        }

        [data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        }

        [data-testid="stCodeBlock"] {
            border-radius: 22px;
            border: 1px solid var(--line);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 22px;
            overflow: hidden;
        }

        @media (max-width: 900px) {
            .hero-title {
                font-size: 2.9rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_aspect(aspect: str) -> str:
    return ASPECT_LABELS.get(aspect, aspect.replace("_", " ").title())


def sentiment_tone(sentiment: str) -> str:
    return sentiment.strip().lower() if isinstance(sentiment, str) else "neutral"


def build_api_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        detail = ""
        try:
            body = exc.response.json()
            detail = body.get("detail", "")
        except ValueError:
            detail = exc.response.text.strip()
        detail = detail or "Unexpected backend error."
        return f"Backend request failed ({exc.response.status_code}): {detail}"
    if isinstance(exc, requests.RequestException):
        return (
            f"Backend API is unavailable at {API_BASE_URL}. "
            "Start it with `uvicorn deepx_submission.api:app --reload`."
        )
    return str(exc)


def build_batch_payload(
    df: pd.DataFrame,
    review_col: str,
    category_col: str | None,
    platform_col: str | None,
    star_rating_col: str | None,
    max_rows: int,
) -> list[tuple[pd.Series, dict]]:
    working = df.copy().head(max_rows).reset_index(drop=True)
    payload_rows: list[tuple[pd.Series, dict]] = []

    for _, row in working.iterrows():
        review_text = str(row.get(review_col, "")).strip()
        if not review_text:
            continue
        business_category = (
            str(row.get(category_col, DEFAULT_CATEGORY)).strip()
            if category_col
            else DEFAULT_CATEGORY
        ) or DEFAULT_CATEGORY
        platform = (
            str(row.get(platform_col, DEFAULT_PLATFORM)).strip()
            if platform_col
            else DEFAULT_PLATFORM
        ) or DEFAULT_PLATFORM
        payload_rows.append(
            (
                row,
                {
                    "review_text": review_text,
                    "business_category": business_category,
                    "platform": platform,
                    "star_rating": row.get(star_rating_col) if star_rating_col else None,
                },
            )
        )

    return payload_rows


def chunk_items(items: list[tuple[pd.Series, dict]], chunk_size: int) -> list[list[tuple[pd.Series, dict]]]:
    return [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]


@st.cache_data(ttl=10, show_spinner=False)
def get_backend_status() -> tuple[bool, str]:
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        response.raise_for_status()
        return True, "Connected"
    except Exception as exc:
        return False, build_api_error(exc)


def render_metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    api_ready, api_status = get_backend_status()
    backend_badge = "Backend Live" if api_ready else "Backend Offline"
    backend_note = (
        f"Serving predictions from {API_BASE_URL}"
        if api_ready
        else "Frontend submissions will wait for the API to come online."
    )
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">DeepX ABSA Studio</div>
            <div class="hero-title">Analyze Reviews Without the Noise.</div>
            <div class="hero-copy">
                Paste a single review for instant aspect extraction, or upload your own dataset to generate
                sentiment and aspect predictions row by row through the FastAPI backend.
            </div>
            <div class="badge-row">
                <div class="hero-badge">Single review inference</div>
                <div class="hero-badge">CSV / XLSX upload</div>
                <div class="hero-badge">Aspect + sentiment output</div>
                <div class="hero-badge">{backend_badge}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if api_ready:
        st.caption(backend_note)
    else:
        st.warning(api_status)


def predict_via_api(
    review_text: str,
    business_category: str,
    platform: str,
    star_rating: float | None = None,
) -> dict:
    payload = {
        "review_text": review_text,
        "business_category": business_category,
        "platform": platform,
        "star_rating": star_rating,
    }
    # Try HTTP API first, fall back to in-process service when available
    try:
        response = requests.post(PREDICT_URL, json=payload, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception:
        if _get_service is not None:
            service = _get_service()
            return service.predict(
                review_text=review_text,
                business_category=business_category,
                platform=platform,
                star_rating=star_rating,
            )
        raise


def predict_batch_via_api(items: list[dict]) -> list[dict]:
    # Try HTTP batch endpoint, fall back to local service if available
    try:
        response = requests.post(BATCH_PREDICT_URL, json={"items": items}, timeout=120)
        response.raise_for_status()
        data = response.json()
        predictions = data.get("predictions", [])
        if len(predictions) != len(items):
            raise RuntimeError("Backend returned a different number of predictions than requested.")
        return predictions
    except Exception:
        if _get_service is not None:
            service = _get_service()
            return service.predict_batch(items)
        raise


def predict_review(
    review_text: str,
    business_category: str,
    platform: str,
    star_rating: float | None = None,
) -> tuple[dict | None, str]:
    try:
        return predict_via_api(review_text, business_category, platform, star_rating=star_rating), "api"
    except Exception as exc:
        return None, build_api_error(exc)


def parse_upload(uploaded_file) -> pd.DataFrame:
    suffix = uploaded_file.name.lower()
    if suffix.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if suffix.endswith(".xlsx") or suffix.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def dataset_template_bytes() -> bytes:
    sample = pd.DataFrame(
        {
            "review_text": [
                "الأكل ممتاز والخدمة سريعة",
                "التطبيق بطيء لكن التوصيل جيد",
            ],
            "business_category": ["مطعم", "مطعم"],
            "platform": ["google_maps", "talabat"],
            "star_rating": [5, 3],
        }
    )
    return sample.to_csv(index=False).encode("utf-8")


def build_result_row(source_row: pd.Series, prediction: dict) -> dict:
    return {
        **source_row.to_dict(),
        "predicted_overall_sentiment": prediction["overall_sentiment"],
        "predicted_aspects": ", ".join(prediction["aspects"]),
        "predicted_aspect_sentiments": json.dumps(prediction["aspect_sentiments"], ensure_ascii=False),
        "predicted_grouped_category": prediction["business_category_grouped"],
        "review_text_clean": prediction["review_text_clean"],
    }


def run_batch_analysis(
    df: pd.DataFrame,
    review_col: str,
    category_col: str | None,
    platform_col: str | None,
    star_rating_col: str | None,
    max_rows: int,
) -> tuple[pd.DataFrame, str]:
    payload_rows = build_batch_payload(
        df,
        review_col=review_col,
        category_col=category_col,
        platform_col=platform_col,
        star_rating_col=star_rating_col,
        max_rows=max_rows,
    )
    if not payload_rows:
        return pd.DataFrame(), "API"

    progress = st.progress(0.0, text="Preparing uploaded rows...")
    status_box = st.empty()
    results: list[dict] = []
    total = len(payload_rows)

    for batch_index, chunk in enumerate(chunk_items(payload_rows, BATCH_REQUEST_SIZE), start=1):
        try:
            predictions = predict_batch_via_api([payload for _, payload in chunk])
        except Exception as exc:
            raise RuntimeError(build_api_error(exc)) from exc

        for (row, _), prediction in zip(chunk, predictions):
            results.append(build_result_row(row, prediction))

        completed = min(batch_index * BATCH_REQUEST_SIZE, total)
        progress.progress(
            completed / total,
            text=f"Analyzing row {completed:,} of {total:,} through the backend",
        )
        status_box.caption(f"Processed {completed:,} reviews via API.")

    progress.empty()
    status_box.empty()
    return pd.DataFrame(results), "API"


def aspect_frequency_html(counter: Counter[str]) -> str:
    if not counter:
        return "<div class='section-note'>No aspects were extracted from the current batch.</div>"
    top_items = counter.most_common(6)
    max_count = top_items[0][1]
    rows = []
    for aspect, count in top_items:
        width = 0 if max_count == 0 else count / max_count * 100
        rows.append(
            f"""
            <div class="aspect-row">
                <div>
                    <div><strong>{format_aspect(aspect)}</strong></div>
                    <div class="aspect-bar"><span style="width:{width:.1f}%"></span></div>
                </div>
                <div>{count:,}</div>
            </div>
            """
        )
    return "".join(rows)


def render_single_review_tab() -> None:
    with st.container(border=True):
        st.markdown("## Single Review")
        st.markdown(
            "<div class='section-note'>Paste one customer review and extract overall sentiment, aspects, and aspect-level sentiment through the backend API.</div>",
            unsafe_allow_html=True,
        )

        with st.form("single_review_form"):
            review_text = st.text_area(
                "Review text",
                height=180,
                placeholder="اكتب المراجعة هنا... / Paste the review here...",
            )
            left, mid, right, far = st.columns([1, 1, 1, 0.8], gap="medium")
            with left:
                business_category = st.text_input("Business category", value=DEFAULT_CATEGORY)
            with mid:
                platform = st.text_input("Platform", value=DEFAULT_PLATFORM)
            with right:
                star_rating = st.number_input(
                    "Star rating",
                    min_value=0.0,
                    max_value=5.0,
                    value=3.0,
                    step=1.0,
                )
            with far:
                grouped = group_business_category(business_category)
                st.markdown(
                    f"<div class='subtle-panel' style='margin-top:1.8rem;'><strong>Grouped category</strong><br>{grouped}</div>",
                    unsafe_allow_html=True,
                )
            submitted = st.form_submit_button("Analyze Review", use_container_width=True)

        if submitted:
            if not review_text.strip():
                st.error("Please enter a review before running analysis.")
            else:
                with st.spinner("Running ABSA inference through the backend..."):
                    result, source = predict_review(
                        review_text,
                        business_category,
                        platform,
                        star_rating=float(star_rating),
                    )
                if result is None:
                    st.error(source)
                else:
                    top_left, top_mid, top_right = st.columns(3, gap="medium")
                    with top_left:
                        render_metric_card("Overall Sentiment", result["overall_sentiment"].title(), "Review-level summary")
                    with top_mid:
                        render_metric_card("Detected Aspects", str(len(result["aspects"])), "Mentioned aspect count")
                    with top_right:
                        render_metric_card("Inference Source", source.upper(), "Frontend request routed through FastAPI")

                    lower_left, lower_right = st.columns([1.15, 0.85], gap="medium")
                    with lower_left:
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="metric-label">Review</div>
                                <div class="review-quote">"{result['review_text']}"</div>
                                <div class="chip-row">
                                    <span class="chip">{result['overall_sentiment'].title()}</span>
                                    <span class="chip">{result['business_category_grouped'].replace('_', ' ').title()}</span>
                                    <span class="chip">{result['platform']}</span>
                                    <span class="chip">Stars: {result.get('star_rating_num', star_rating):.1f}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with lower_right:
                        with st.container(border=True):
                            st.markdown("### Aspect Breakdown")
                            if result["aspect_sentiments"]:
                                for aspect, sentiment in result["aspect_sentiments"].items():
                                    color = SENTIMENT_COLORS.get(sentiment_tone(sentiment), "#5d6b78")
                                    st.markdown(
                                        f"""
                                        <div class="aspect-row">
                                            <div><strong>{format_aspect(aspect)}</strong></div>
                                            <div style="color:{color}; font-weight:800;">{sentiment.title()}</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.info("No aspects were returned for this review.")

                    with st.container(border=True):
                        st.markdown("### JSON Output")
                        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")


def render_dataset_tab() -> None:
    with st.container(border=True):
        st.markdown("## Dataset Upload")
        st.markdown(
            "<div class='section-note'>Upload your own CSV or Excel file, map the review column, and generate aspect + sentiment predictions for each row through the backend API.</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download Template CSV",
            data=dataset_template_bytes(),
            file_name="deepx_absa_template.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])

        if uploaded_file is None:
            return

        try:
            uploaded_df = parse_upload(uploaded_file)
        except Exception as exc:
            st.error(str(exc))
            return

        if uploaded_df.empty:
            st.warning("The uploaded file is empty.")
            return

        columns = uploaded_df.columns.tolist()
        suggested_review = "review_text" if "review_text" in columns else columns[0]
        suggested_category = "business_category" if "business_category" in columns else None
        suggested_platform = "platform" if "platform" in columns else None
        suggested_star_rating = "star_rating" if "star_rating" in columns else None

        map_left, map_mid, map_right, map_far, map_last = st.columns([1.2, 1, 1, 1, 0.8], gap="medium")
        with map_left:
            review_col = st.selectbox("Review text column", options=columns, index=columns.index(suggested_review))
        with map_mid:
            category_options = ["<use default>"] + columns
            category_index = 0 if suggested_category is None else category_options.index(suggested_category)
            category_choice = st.selectbox("Category column", options=category_options, index=category_index)
        with map_right:
            platform_options = ["<use default>"] + columns
            platform_index = 0 if suggested_platform is None else platform_options.index(suggested_platform)
            platform_choice = st.selectbox("Platform column", options=platform_options, index=platform_index)
        with map_far:
            star_rating_options = ["<skip>"] + columns
            star_rating_index = 0 if suggested_star_rating is None else star_rating_options.index(suggested_star_rating)
            star_rating_choice = st.selectbox("Star rating column", options=star_rating_options, index=star_rating_index)
        with map_last:
            max_rows = st.number_input(
                "Rows to run",
                min_value=1,
                max_value=int(max(len(uploaded_df), 1)),
                value=min(len(uploaded_df), 200),
                step=1,
            )

        with st.container(border=True):
            st.markdown(f"Previewing `{uploaded_file.name}` with **{len(uploaded_df):,} rows** and **{len(columns)} columns**.")
            st.dataframe(uploaded_df.head(8), use_container_width=True, hide_index=True)

        if st.button("Run Dataset Analysis", use_container_width=True):
            category_col = None if category_choice == "<use default>" else category_choice
            platform_col = None if platform_choice == "<use default>" else platform_choice
            star_rating_col = None if star_rating_choice == "<skip>" else star_rating_choice
            with st.spinner("Sending uploaded rows to the backend for inference..."):
                try:
                    results_df, source_label = run_batch_analysis(
                        uploaded_df,
                        review_col=review_col,
                        category_col=category_col,
                        platform_col=platform_col,
                        star_rating_col=star_rating_col,
                        max_rows=int(max_rows),
                    )
                except Exception as exc:
                    st.error(str(exc))
                    return
            st.session_state["batch_results"] = results_df
            st.session_state["batch_source"] = source_label

        results_df = st.session_state.get("batch_results")
        if not isinstance(results_df, pd.DataFrame) or results_df.empty:
            return

        source_label = st.session_state.get("batch_source", "API")
        sentiment_counts = results_df["predicted_overall_sentiment"].value_counts()
        aspect_counter = Counter()
        for value in results_df["predicted_aspects"].fillna(""):
            for aspect in [item.strip() for item in str(value).split(",") if item.strip()]:
                aspect_counter[aspect] += 1

        summary_left, summary_mid, summary_right, summary_far = st.columns(4, gap="medium")
        with summary_left:
            render_metric_card("Rows Analyzed", f"{len(results_df):,}", "Processed from uploaded dataset")
        with summary_mid:
            render_metric_card("Positive", f"{sentiment_counts.get('positive', 0):,}", "Overall review sentiment count")
        with summary_right:
            render_metric_card("Negative", f"{sentiment_counts.get('negative', 0):,}", "Rows with negative overall signal")
        with summary_far:
            top_aspect = format_aspect(aspect_counter.most_common(1)[0][0]) if aspect_counter else "None"
            render_metric_card("Top Aspect", top_aspect, f"Predictions generated via {source_label}")

        chart_left, chart_right = st.columns([0.9, 1.1], gap="medium")
        with chart_left:
            with st.container(border=True):
                st.markdown("### Sentiment Mix")
                total = max(len(results_df), 1)
                for label in ["positive", "neutral", "negative"]:
                    count = sentiment_counts.get(label, 0)
                    pct = count / total * 100
                    color = SENTIMENT_COLORS[label]
                    st.markdown(
                        f"""
                        <div style="margin:0.85rem 0;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:0.35rem;">
                                <span><strong>{label.title()}</strong></span>
                                <span>{count:,} ({pct:.1f}%)</span>
                            </div>
                            <div class="aspect-bar" style="margin-top:0;">
                                <span style="width:{pct:.1f}%; background:{color};"></span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        with chart_right:
            with st.container(border=True):
                st.markdown("### Most Frequent Aspects")
                st.markdown(aspect_frequency_html(aspect_counter), unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("### Prediction Results")
            export_df = results_df.copy()
            export_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results CSV",
                data=export_bytes,
                file_name="deepx_absa_results.csv",
                mime="text/csv",
            )
            preview_cols = [
                col
                for col in [
                    review_col,
                    "predicted_overall_sentiment",
                    "predicted_aspects",
                    "predicted_aspect_sentiments",
                ]
                if col in export_df.columns
            ]
            st.dataframe(
                export_df[preview_cols + [col for col in export_df.columns if col not in preview_cols]],
                use_container_width=True,
                hide_index=True,
            )


def render_footer() -> None:
    st.caption(
        f"Frontend submissions are routed to the FastAPI backend at {API_BASE_URL}. "
        "Single-review and dataset predictions both use the backend model service."
    )


render_hero()

tab_single, tab_dataset = st.tabs(["Single Review", "Dataset Upload"])
with tab_single:
    render_single_review_tab()
with tab_dataset:
    render_dataset_tab()

render_footer()
