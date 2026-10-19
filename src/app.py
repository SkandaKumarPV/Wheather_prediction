"""FastAPI service for live weather predictions and accuracy tracking."""

from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .logger import backfill_actuals, get_accuracy_summary, log_new_prediction
from .predict_live import (
    fetch_7day_forecast,
    geocode_city,
    get_live_prediction,
)

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Weather Predictor API", version="1.0.0")


class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    location_name: str = Field(..., min_length=1, max_length=200)


class LocationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


def _service_error(error: Exception) -> HTTPException:
    return HTTPException(status_code=502, detail=str(error))


@app.get("/", response_class=FileResponse)
def dashboard() -> Path:
    """Serve the lightweight accuracy dashboard."""
    return ROOT / "static" / "index.html"


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    """Create and persist a live prediction."""
    try:
        prediction = get_live_prediction(
            request.latitude, request.longitude, request.location_name
        )
        prediction["logged"] = log_new_prediction(prediction)
        return prediction
    except (OSError, ValueError, KeyError) as error:
        raise _service_error(error) from error


@app.post("/forecast")
def forecast(request: LocationRequest) -> dict[str, Any]:
    """Return the Open-Meteo seven-day forecast."""
    try:
        return {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "forecast": fetch_7day_forecast(request.latitude, request.longitude),
        }
    except (OSError, ValueError, KeyError) as error:
        raise _service_error(error) from error


@app.get("/geocode")
def geocode(city: str = Query(..., min_length=1)) -> dict[str, Any]:
    """Resolve a city name to coordinates."""
    try:
        return geocode_city(city)
    except (OSError, ValueError, KeyError) as error:
        raise _service_error(error) from error


@app.get("/accuracy")
def accuracy(
    latitude: float | None = None,
    longitude: float | None = None,
    last_n: int = 30,
) -> dict[str, Any]:
    """Backfill optional coordinates and return scored prediction accuracy."""
    if last_n < 1 or last_n > 1000:
        raise HTTPException(status_code=422, detail="last_n must be between 1 and 1000.")
    if (latitude is None) != (longitude is None):
        raise HTTPException(
            status_code=422,
            detail="latitude and longitude must be provided together.",
        )
    try:
        if latitude is not None and longitude is not None:
            backfill_actuals(latitude, longitude)
        return get_accuracy_summary(last_n)
    except (OSError, ValueError, KeyError) as error:
        raise _service_error(error) from error