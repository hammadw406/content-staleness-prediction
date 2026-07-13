"""
Trains the Random Forest content-staleness model and saves everything
the API needs to reproduce the notebook's preprocessing at inference time.

Run once (locally or in CI) whenever the model needs retraining:
    python train.py

Produces artifacts/model.joblib containing:
    - model: the fitted RandomForestClassifier
    - feature_columns: exact column order the model expects (post get_dummies)
    - impute_medians: medians used to fill missing numeric values
    - threshold: the tuned decision threshold (0.3, from the notebook's PR analysis)
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DATA_PATH = "content_refresh_anonymized.csv"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLS = [
    "content_type", "main_intent", "provider_used", "model_used",
    "competition_level", "age_tier", "freshness_tier",
    "word_count_tier", "char_count_tier", "impression_tier",
    "position_tier", "trend_direction",
]

MEDIAN_FILL_COLS = ["search_volume", "cpc", "competition", "word_count", "char_count", "scroll_rate"]


def engineer_features(df: pd.DataFrame, impute_medians: dict) -> pd.DataFrame:
    """Reproduces every feature-engineering step from CodeFile_clean.ipynb."""
    df = df.copy()

    # --- Missing value handling ---
    df["provider_used"] = df["provider_used"].fillna("not_ai_generated")
    df["model_used"] = df["model_used"].fillna("not_ai_generated")

    df["kw_data_missing"] = df["search_volume"].isnull().astype(int)
    for col in MEDIAN_FILL_COLS:
        df[col] = df[col].fillna(impute_medians[col])

    df["competition_level"] = df["competition_level"].fillna("unknown")
    df["content_not_scraped"] = df["word_count"].isnull().astype(int)
    df["word_count_tier"] = df["word_count_tier"].fillna("unknown")
    df["char_count_tier"] = df["char_count_tier"].fillna("unknown")
    df["main_intent"] = df["main_intent"].fillna("unknown")
    df["trend_pct"] = df["trend_pct"].fillna(0)

    # --- CTR scale fix (some rows recorded 0-100 instead of 0-1) ---
    df.loc[df["ctr"] > 1, "ctr"] = df.loc[df["ctr"] > 1, "ctr"] / 100

    # --- Momentum / change features ---
    df["traffic_change_pct"] = (
        (df["clicks_last_30d"] - df["clicks_prev_30d"]) / df["clicks_prev_30d"].replace(0, np.nan)
    ).fillna(0).clip(-1, 3)

    df["impressions_change_pct"] = (
        (df["impressions_last_30d"] - df["impressions_prev_30d"]) / df["impressions_prev_30d"].replace(0, np.nan)
    ).fillna(0).clip(-1, 3)

    df["sessions_change_pct"] = (
        (df["sessions_last_30d"] - df["sessions_prev_30d"]) / df["sessions_prev_30d"].replace(0, np.nan)
    ).fillna(0).clip(-1, 3)

    # --- Activity / coverage features ---
    df["clicks_per_active_day"] = (df["clicks_90d"] / df["days_with_sessions"].replace(0, np.nan)).fillna(0)
    df["impressions_per_active_day"] = (df["impressions_90d"] / df["days_with_impressions"].replace(0, np.nan)).fillna(0)
    df["impression_coverage"] = df["days_with_impressions"] / 90
    df["session_coverage"] = df["days_with_sessions"] / 90

    df["engaged_session_rate"] = (df["engaged_sessions_90d"] / df["sessions_90d"].replace(0, np.nan)).fillna(0)
    df["scroll_events_per_session"] = (df["scroll_events_90d"] / df["sessions_90d"].replace(0, np.nan)).fillna(0)
    df["pageviews_per_user"] = (df["pageviews_90d"] / df["users_90d"].replace(0, np.nan)).fillna(0)

    df["ai_session_share"] = (df["ai_sessions_90d"] / df["sessions_90d"].replace(0, np.nan)).fillna(0)
    df["ai_session_share"] = df["ai_session_share"].clip(upper=1.0)

    # --- Staleness / opportunity features ---
    df["stale_and_declining"] = (
        (df["days_since_last_update"] > impute_medians["days_since_last_update"]) &
        (df["trend_direction"] == "down")
    ).astype(int)

    df["update_lag_ratio"] = (df["days_since_last_update"] / df["content_age_days"].replace(0, np.nan)).fillna(0)

    df["opportunity_score"] = df["search_volume"] * (1 - df["ctr"])
    df["opportunity_score_log"] = np.log1p(df["opportunity_score"].clip(lower=0))

    df["visibility_gap"] = df["impressions_90d"] / (df["search_volume"] + 1)
    df["visibility_gap_log"] = np.log1p(df["visibility_gap"])

    df["low_ctr_high_volume"] = (
        (df["ctr"] < impute_medians["ctr"]) & (df["search_volume"] > impute_medians["search_volume"])
    ).astype(int)

    df["thin_content_flag"] = (df["word_count"] < impute_medians["word_count_q25"]).astype(int)
    df["engagement_per_word"] = df["engagement_rate"] / (df["word_count"] + 1)

    df["decline_and_thin"] = (df["traffic_change_pct"] < -0.10).astype(int) * df["thin_content_flag"]
    df["stale_decline_thin"] = df["stale_and_declining"] * df["thin_content_flag"]

    return df


def build_matrix(df: pd.DataFrame, feature_columns=None):
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    df_encoded = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns], drop_first=True)

    drop_cols = [
        "content_id", "client_id", "needs_refresh", "avg_position",
        "stale_and_declining", "stale_decline_thin", "decline_and_thin", "visibility_gap",
    ]
    X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])
    X = X.drop(columns=[c for c in X.columns if c.startswith(("trend_direction_", "position_tier_"))])

    if feature_columns is not None:
        # Align inference-time columns to the training-time schema
        X = X.reindex(columns=feature_columns, fill_value=0)

    return X


def main():
    df = pd.read_csv(DATA_PATH)

    # domain-rule fix used before feature engineering in the notebook
    df.loc[df["ctr"] > 1, "ctr"] = df.loc[df["ctr"] > 1, "ctr"] / 100

    impute_medians = {col: float(df[col].median()) for col in MEDIAN_FILL_COLS}
    impute_medians["days_since_last_update"] = float(df["days_since_last_update"].median())
    impute_medians["ctr"] = float(df["ctr"].median())
    impute_medians["word_count_q25"] = float(df["word_count"].quantile(0.25))

    df = engineer_features(df, impute_medians)

    df["needs_refresh"] = (
        (df["avg_position"] > 20) & (df["stale_and_declining"] == 1)
    ).astype(int)

    X = build_matrix(df)
    y = df["needs_refresh"]
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_estimators=200)
    rf.fit(X_train, y_train)

    threshold = 0.3
    probs = rf.predict_proba(X_test)[:, 1]
    preds = (probs > threshold).astype(int)
    print(classification_report(y_test, preds))

    joblib.dump(
        {
            "model": rf,
            "feature_columns": feature_columns,
            "impute_medians": impute_medians,
            "threshold": threshold,
        },
        ARTIFACT_DIR / "model.joblib",
    )
    print(f"Saved model to {ARTIFACT_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
