# DeepX Arabic ABSA

Arabic aspect-based sentiment analysis for customer reviews.

This package now includes:

- a FastAPI backend for single-review inference
- a Streamlit frontend for interactive analysis
- a shared local inference service used by both apps
- training and submission scripts for the original DeepX workflow

## What changed

- Added `api.py` to expose model inference over HTTP.
- Added `streamlit_app.py` for a frontend with single-review and dataset-upload flows.
- Added `service.py` so the API and frontend share the same model-loading and prediction logic.
- Updated the frontend so batch predictions run on the user's uploaded CSV or Excel file instead of relying on bundled datasets.
- Kept packaged weights in `deepx_submission/weights` so the app can run without retraining first.

## Project structure

```text
deepx_submission/
|-- __init__.py
|-- api.py
|-- config.py
|-- inference.py
|-- models.py
|-- preprocessing.py
|-- README.md
|-- requirements.txt
|-- service.py
|-- streamlit_app.py
|-- train.py
`-- weights/
```

## Setup

From the repository root:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r deepx_submission/requirements.txt
```

Installed runtime dependencies include `fastapi`, `uvicorn`, `streamlit`, `requests`, `pandas`, `scikit-learn`, `joblib`, and `openpyxl`.

## Backend usage

Start the API from the repository root:

```bash
uvicorn deepx_submission.api:app --reload
```

Available endpoints:

- `GET /health`
- `POST /predict`
- `POST /predict/batch`

Example request:

```json
{
  "review_text": "الاكل ممتاز والخدمة سريعة",
  "business_category": "مطعم",
  "platform": "google_maps",
  "star_rating": 5
}
```

Example response shape:

```json
{
  "review_text": "...",
  "review_text_clean": "...",
  "business_category": "...",
  "business_category_grouped": "...",
  "platform": "...",
  "star_rating_num": 5.0,
  "overall_sentiment": "positive",
  "aspects": ["food", "service"],
  "aspect_sentiments": {
    "food": "positive",
    "service": "positive"
  }
}
```

## Frontend usage

Start the Streamlit app from the repository root:

```bash
streamlit run deepx_submission/streamlit_app.py
```

The frontend uses `ABSA_API_BASE_URL` when it is set. `ABSA_API_URL` is still supported for compatibility. By default it posts to:

```bash
http://127.0.0.1:8000
```

The frontend is now backend-first for both single-review and uploaded-dataset predictions, so make sure the FastAPI app is running before opening Streamlit.

### Frontend features

- Single review analysis with aspect extraction and per-aspect sentiment.
- Dataset upload for `csv`, `xlsx`, and `xls` files.
- Column mapping for review text, business category, platform, and optional star rating.
- Downloadable template CSV for batch input.
- Downloadable CSV containing prediction results.
- Summary cards for sentiment mix and most frequent aspects.

### Expected upload columns

Required:

- `review_text`

Optional:

- `business_category`
- `platform`
- `star_rating`

If category or platform columns are not provided, the app uses:

- `business_category = "مطعم"`
- `platform = "google_maps"`

### Batch output columns

The uploaded dataset is preserved and extended with:

- `predicted_overall_sentiment`
- `predicted_aspects`
- `predicted_aspect_sentiments`
- `predicted_grouped_category`
- `review_text_clean`

Batch predictions are sent to `POST /predict/batch` in chunks, so uploaded frontend data is scored by the backend model service instead of an in-process fallback.

## Training

Run training from inside `deepx_submission` so the default relative paths resolve as expected:

```bash
cd deepx_submission
python train.py
```

Example with explicit inputs:

```bash
cd deepx_submission
python train.py --train_path data/DeepX_train.xlsx --val_path data/DeepX_validation.xlsx --weights_dir weights
```

Useful training options:

- `--data_path` to train from one labeled file and split it internally
- `--threshold_strategy` to tune aspect thresholds
- `--fallback_mode` to control keyword fallback behavior
- `--experiment_log` to append validation metrics to a JSONL log

Training writes updated artifacts to `weights/`, including:

- `aspect_model.joblib`
- `sentiment_model.joblib`
- `mlb.joblib`
- `aspect_thresholds.json`
- model metadata JSON files

## Competition inference

To generate `submission.json` from unlabeled review data:

```bash
cd deepx_submission
python inference.py --unlabeled_path data/DeepX_unlabeled.xlsx --weights_dir weights --output submission.json
```

Packaged weights are already included, so you can skip retraining when you only need inference.

## Model summary

The pipeline is split into two stages:

1. Aspect detection
2. Per-aspect sentiment classification

Aspect detection uses:

- word TF-IDF features
- character TF-IDF features
- grouped business category
- platform
- star rating numeric feature
- multilingual aspect keyword flags

Sentiment classification uses the same feature family plus the target `aspect` itself.

The model also supports a multilingual keyword fallback to recover aspect mentions missed by the SVM, especially for sparse wording or dialect variation.

## Aspect labels

| Aspect | Description |
|---|---|
| `food` | Food quality and taste |
| `service` | Staff behavior and wait time |
| `price` | Pricing and value |
| `cleanliness` | Hygiene and tidiness |
| `delivery` | Delivery speed and accuracy |
| `ambiance` | Atmosphere, decor, and noise |
| `app_experience` | App or ordering experience |
| `general` | Overall impression |
| `none` | No specific aspect in source labels |

## Notes and limitations

- The frontend and backend are designed for inference, not for training workflows.
- Training and inference scripts still rely on relative paths, so running them from `deepx_submission` is the safest default.
- The frontend expects the FastAPI backend to be running for interactive predictions.
- Neutral sentiment remains the hardest label because it is less represented in the training data.
