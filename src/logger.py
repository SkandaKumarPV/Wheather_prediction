"""Persist live predictions and score them once observations are available."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .predict_live import fetch_recent

ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / "data" / "prediction_log.csv"
LOG_COLUMNS = [
    "logged_at",
    "as_of",
    "prediction_for",
    "location",
    "latitude",
    "longitude",
    "predicted_temp_max",
    "predicted_temp_delta",
    "predicted_rain_prob",
    "rain_predicted",
    "actual_temp_max",
    "actual_rain",
    "temp_error",
    "rain_correct",
    "retrospective",
]


def _load_log() -> pd.DataFrame:
    """Load the prediction log with a stable schema."""
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    log = pd.read_csv(LOG_FILE)
    for column in LOG_COLUMNS:
        if column not in log.columns:
            log[column] = pd.NA
    return log[LOG_COLUMNS]


def _write_log(log: pd.DataFrame) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log[LOG_COLUMNS].to_csv(LOG_FILE, index=False)


def log_new_prediction(prediction: dict[str, Any]) -> bool:
    """Append a prediction unless the same location/date is already logged."""
    required = {
        "prediction_date",
        "location",
        "latitude",
        "longitude",
        "predicted_temperature",
        "rain_probability",
        "rain_predicted",
    }
    missing = required.difference(prediction)
    if missing:
        raise ValueError(f"Prediction is missing fields: {sorted(missing)}")

    log = _load_log()
    target_date = str(prediction["prediction_date"])
    duplicate = (
        (log["prediction_for"].astype(str) == target_date)
        & (pd.to_numeric(log["latitude"], errors="coerce") == float(prediction["latitude"]))
        & (pd.to_numeric(log["longitude"], errors="coerce") == float(prediction["longitude"]))
    ).any()
    if duplicate:
        return False

    row = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "as_of": prediction.get("latest_observation_date", ""),
        "prediction_for": target_date,
        "location": prediction["location"],
        "latitude": float(prediction["latitude"]),
        "longitude": float(prediction["longitude"]),
        "predicted_temp_max": float(prediction["predicted_temperature"]),
        "predicted_temp_delta": float(prediction.get("predicted_temp_delta", 0.0)),
        "predicted_rain_prob": float(prediction["rain_probability"]),
        "rain_predicted": bool(prediction["rain_predicted"]),
        "actual_temp_max": pd.NA,
        "actual_rain": pd.NA,
        "temp_error": pd.NA,
        "rain_correct": pd.NA,
        "retrospective": bool(prediction.get("retrospective", False)),
    }
    _write_log(pd.concat([log, pd.DataFrame([row])], ignore_index=True))
    return True


def backfill_actuals(lat: float, lon: float, rain_threshold: float = 1.0) -> int:
    """Fill scores for past predictions using observed weather data."""
    log = _load_log().astype(object)
    matches = (
        (pd.to_numeric(log["latitude"], errors="coerce") == float(lat))
        & (pd.to_numeric(log["longitude"], errors="coerce") == float(lon))
        & log["actual_temp_max"].isna()
    )
    if not matches.any():
        return 0

    observations = fetch_recent(lat, lon, past_days=30)
    observations["date_key"] = observations["date"].dt.strftime("%Y-%m-%d")
    observed = observations.set_index("date_key")
    scored = 0
    for index in log.index[matches]:
        target = str(log.at[index, "prediction_for"])[:10]
        if target not in observed.index:
            continue
        actual = observed.loc[target]
        predicted_rain = bool(log.at[index, "rain_predicted"])
        actual_rain = float(actual["precip"]) > rain_threshold
        predicted_temp = float(log.at[index, "predicted_temp_max"])
        actual_temp = float(actual["temp_max"])
        log.at[index, "actual_temp_max"] = actual_temp
        log.at[index, "actual_rain"] = actual_rain
        log.at[index, "temp_error"] = abs(predicted_temp - actual_temp)
        log.at[index, "rain_correct"] = predicted_rain == actual_rain
        scored += 1
    if scored:
        _write_log(log)
    return scored


def seed_historical_predictions(
    lat: float, lon: float, days: int = 14, location: str = "Historical estimate"
) -> int:
    """Create retrospective estimates for recent observations, without scoring them as forecasts."""
    if days < 1:
        raise ValueError("days must be positive.")
    observations = fetch_recent(lat, lon, past_days=days + 3)
    created = 0
    for index in range(2, len(observations) - 1):
        current = observations.iloc[index]
        next_day = observations.iloc[index + 1]
        prediction = {
            "prediction_date": next_day["date"].date().isoformat(),
            "latest_observation_date": current["date"].date().isoformat(),
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "predicted_temperature": float(current["temp_max"]),
            "predicted_temp_delta": 0.0,
            "rain_probability": 1.0 if float(current["precip"]) > 1 else 0.0,
            "rain_predicted": float(current["precip"]) > 1,
            "retrospective": True,
        }
        if log_new_prediction(prediction):
            created += 1
    return created


def get_accuracy_summary(last_n: int = 30) -> dict[str, Any]:
    """Return accuracy metrics from scored predictions only."""
    if last_n < 1:
        raise ValueError("last_n must be positive.")
    log = _load_log()
    scored = log.dropna(subset=["temp_error", "rain_correct"]).tail(last_n)
    if scored.empty:
        return {
            "scored_predictions": 0,
            "average_temperature_error": None,
            "rain_accuracy": None,
            "history": [],
        }
    history = scored[
        ["prediction_for", "predicted_temp_max", "actual_temp_max", "temp_error"]
    ].to_dict(orient="records")
    return {
        "scored_predictions": int(len(scored)),
        "average_temperature_error": float(
            pd.to_numeric(scored["temp_error"]).mean()
        ),
        "rain_accuracy": float(
            scored["rain_correct"].astype(str).str.lower().eq("true").mean()
        ),
        "history": history,
    }