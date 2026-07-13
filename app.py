"""
FastAPI service for the content-staleness model.

Local run:
    uvicorn app:app --reload

Endpoints:
    GET  /health         -> liveness check
    POST /predict        -> { "needs_refresh": 0|1, "probability": float }
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from train import engineer_features, build_matrix

MODEL_PATH = Path("artifacts/model.joblib")

app = FastAPI(title="Content Staleness Prediction API", version="1.0")

_artifact = None


def get_artifact():
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"{MODEL_PATH} not found. Run `python train.py` first to create it."
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


class ContentInput(BaseModel):
    # Raw fields — same columns as content_refresh_anonymized.csv, one row per page.
    search_volume: Optional[float] = None
    competition: Optional[float] = None
    competition_level: Optional[str] = None
    cpc: Optional[float] = None
    content_type: str
    main_intent: Optional[str] = None
    word_count: Optional[float] = None
    char_count: Optional[float] = None
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    impressions_90d: float
    clicks_90d: float
    pageviews_90d: float
    sessions_90d: float
    users_90d: float
    engaged_sessions_90d: float
    ai_sessions_90d: float
    scroll_events_90d: float
    days_with_impressions: float
    days_with_sessions: float
    impressions_last_30d: float
    clicks_last_30d: float
    sessions_last_30d: float
    impressions_prev_30d: float
    clicks_prev_30d: float
    sessions_prev_30d: float
    content_age_days: float
    age_tier: Optional[str] = None
    age_tier_order: Optional[float] = None
    days_since_last_update: float
    freshness_tier: Optional[str] = None
    word_count_tier: Optional[str] = None
    char_count_tier: Optional[str] = None
    ctr: float
    avg_position: float
    engagement_rate: float
    scroll_rate: Optional[float] = None
    ai_traffic_pct: float
    impression_tier: Optional[str] = None
    position_tier: Optional[str] = None
    trend_direction: str
    trend_pct: Optional[float] = None


class PredictionResponse(BaseModel):
    needs_refresh: int
    probability: float
    threshold_used: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ContentInput):
    artifact = get_artifact()
    try:
        df = pd.DataFrame([payload.dict()])
        df = engineer_features(df, artifact["impute_medians"])
        X = build_matrix(df, feature_columns=artifact["feature_columns"])
        prob = float(artifact["model"].predict_proba(X)[:, 1][0])
        threshold = artifact["threshold"]
        return PredictionResponse(
            needs_refresh=int(prob > threshold),
            probability=round(prob, 4),
            threshold_used=threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
