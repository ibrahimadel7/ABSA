# DeepX Arabic ABSA Workspace

Arabic aspect-based sentiment analysis for customer reviews, with:

- a FastAPI backend for inference
- a Streamlit frontend for interactive analysis
- packaged model weights for local predictions
- training and submission scripts for the original competition workflow

The production app lives in `deepx_submission/`. If you only need to run the API or UI, start there.

## Repository layout

```text
.
|-- deepx_submission/        # main package: API, Streamlit app, training, inference
|-- weights/                 # root-level experiment artifacts
|-- weights_round2_c/        # alternate saved experiment artifacts
|-- DeepX_*.xlsx             # local dataset files used during experimentation
|-- hidden_test_predictions_best.json
`-- README.md
```

Important note:

- `deepx_submission/weights/` is the weights directory used by the packaged app.
- `weights/` and `weights_round2_c/` are workspace-level experiment outputs and are not the default runtime path for the API/frontend.

## Quick start

Run these commands from the repository root:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r deepx_submission/requirements.txt
```

## Run the backend

```bash
uvicorn deepx_submission.api:app --reload
```

Default endpoints:

- `GET /health`
- `POST /predict`
- `POST /predict/batch`

## Run the frontend

```bash
streamlit run deepx_submission/streamlit_app.py
```

The Streamlit app is backend-first, so keep the FastAPI server running while using it.

By default, the frontend sends requests to:

```text
http://127.0.0.1:8000
```

You can override that with either:

- `ABSA_API_BASE_URL`
- `ABSA_API_URL`

## Frontend features

- Single-review sentiment and aspect analysis
- CSV, XLSX, and XLS dataset upload
- Column mapping for review text, category, platform, and optional star rating
- Batch scoring through the FastAPI backend
- Downloadable template CSV and results CSV
- Summary cards for sentiment mix and top detected aspects

## Training

Run training from inside `deepx_submission/` so the default relative paths resolve correctly:

```bash
cd deepx_submission
python train.py
```

Example with explicit inputs:

```bash
cd deepx_submission
python train.py --train_path data/DeepX_train.xlsx --val_path data/DeepX_validation.xlsx --weights_dir weights
```

Useful options include:

- `--data_path` to train from one labeled file and split internally
- `--threshold_strategy` to tune aspect thresholds
- `--fallback_mode` to control keyword fallback behavior
- `--experiment_log` to append validation metrics to a JSONL log

Training writes updated artifacts into the selected `weights/` directory.

## Competition inference

To generate a `submission.json` file from unlabeled review data:

```bash
cd deepx_submission
python inference.py --unlabeled_path data/DeepX_unlabeled.xlsx --weights_dir weights --output submission.json
```

Packaged weights are already included in `deepx_submission/weights/`, so retraining is optional if you only need local inference.

## API response shape

Single-review predictions return:

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

## Tech stack

- `fastapi`
- `uvicorn`
- `streamlit`
- `pandas`
- `scikit-learn`
- `joblib`
- `openpyxl`

## Where to look next

- `deepx_submission/README.md` for the package-level guide
- `deepx_submission/api.py` for backend endpoints
- `deepx_submission/streamlit_app.py` for the UI
- `deepx_submission/train.py` for model training
- `deepx_submission/inference.py` for submission generation
