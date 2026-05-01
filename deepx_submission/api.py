"""FastAPI backend for ABSA inference."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .service import DEFAULT_BUSINESS_CATEGORY, DEFAULT_PLATFORM, get_service


app = FastAPI(title="DeepX ABSA API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    review_text: str = Field(..., min_length=1, description="User review text")
    business_category: str = Field(default=DEFAULT_BUSINESS_CATEGORY, description="Raw business category")
    platform: str = Field(default=DEFAULT_PLATFORM, description="Review platform")
    star_rating: float | None = Field(default=None, description="Optional 1-5 star rating")


class PredictionResponse(BaseModel):
    review_text: str
    review_text_clean: str
    business_category: str
    business_category_grouped: str
    platform: str
    star_rating_num: float
    overall_sentiment: str
    aspects: list[str]
    aspect_sentiments: dict[str, str]


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        result = get_service().predict(
            review_text=payload.review_text,
            business_category=payload.business_category,
            platform=payload.platform,
            star_rating=payload.star_rating,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        predictions = get_service().predict_batch([item.model_dump() for item in payload.items])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**prediction) for prediction in predictions]
    )
