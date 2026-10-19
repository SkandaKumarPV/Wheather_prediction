"""Live weather retrieval and inference using the trained model artifacts."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import requests

try:
    from .features import FEATURE_COLUMNS, WEATHER_COLUMNS
except ImportError:
    from features import FEATURE_COLUMNS, WEATHER_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
TIMEOUT_SECONDS = 30

WMO_CONDITIONS = {
    0: ("Clear sky", "☀️", "sunny"),
    1: ("Mainly clear", "🌤️", "clear"),
    2: ("Partly cloudy", "⛅", "clear"),
    3: ("Overcast", "☁️", "clear"),
    45: ("Fog", "🌫️", "clear"),
    48: ("Depositing rime fog", "🌫️", "clear"),
    51: ("Light drizzle", "🌦️", "rainy"),
    53: ("Moderate drizzle", "🌦️", "rainy"),
    55: ("Dense drizzle", "🌧️", "rainy"),
    61: ("Slight rain", "🌦️", "rainy"),
    63: ("Moderate rain", "🌧️", "rainy"),
    65: ("Heavy rain", "🌧️", "rainy"),
    71: ("Slight snow", "🌨️", "clear"),
    73: ("Moderate snow", "🌨️", "clear"),
    75: ("Heavy snow", "❄️", "clear"),
    80: ("Slight rain showers", "🌦️", "rainy"),
    81: ("Moderate rain showers", "🌧️", "rainy"),
    82: ("Violent rain showers", "⛈️", "rainy"),
    95: ("Thunderstorm", "⛈️", "rainy"),
    96: ("Thunderstorm with hail", "⛈️", "rainy"),
    99: ("Thunderstorm with heavy hail", "⛈️", "rainy"),
}


def _request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Weather service returned an invalid response.")
    return payload


@lru_cache(maxsize=1)
def _load_models() -> tuple[Any, Any]:
    """Load and validate the deployment model artifacts once per process."""
    temp_model = joblib.load(MODELS_DIR / "temp_model.joblib")
    rain_model = joblib.load(MODELS_DIR / "rain_model.joblib")
    for name, model in (("temperature", temp_model), ("rain", rain_model)):
        if getattr(model, "n_features_in_", None) != len(FEATURE_COLUMNS):
            raise ValueError(f"{name} model does not accept {len(FEATURE_COLUMNS)} features.")
        model_features = list(getattr(model, "feature_names_in_", []))
        if model_features and model_features != FEATURE_COLUMNS:
            raise ValueError(f"{name} model feature order is incompatible.")
    if 1 not in list(getattr(rain_model, "classes_", [])):
        raise ValueError("Rain model does not contain positive class 1.")
    return temp_model, rain_model


def _daily_frame(payload: dict[str, Any]) -> pd.DataFrame:
    daily = payload.get("daily")
    if not isinstance(daily, dict) or "time" not in daily:
        raise ValueError("Weather response is missing daily observations.")
    missing = [column for column in WEATHER_COLUMNS if column not in daily]
    if missing:
        raise ValueError(f"Weather response is missing columns: {missing}")
    frame = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for column in WEATHER_COLUMNS:
        frame[column] = pd.to_numeric(daily[column], errors="coerce")
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if frame[["date", *WEATHER_COLUMNS]].isna().any().any():
        raise ValueError("Weather response contains missing required values.")
    return frame


def _normalize_daily_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate Open-Meteo variable names into the training schema."""
    daily = payload.get("daily")
    if not isinstance(daily, dict):
        return payload
    names = {
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "precipitation_sum": "precip",
        "wind_speed_10m_max": "wind",
        "surface_pressure_mean": "pressure",
        "relative_humidity_2m_mean": "humidity",
        "cloud_cover_mean": "cloudcover",
        "shortwave_radiation_sum": "radiation",
    }
    normalized = dict(payload)
    normalized["daily"] = {
        names.get(column, column): values for column, values in daily.items()
    }
    return normalized


def fetch_recent(lat: float, lon: float, past_days: int = 30) -> pd.DataFrame:
    """Fetch recent daily observations in the training schema."""
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Coordinates are outside valid latitude/longitude ranges.")
    if past_days < 3:
        raise ValueError("At least three recent days are required for inference.")
    payload = _request_json(
        FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "past_days": past_days,
            "forecast_days": 1,
            "daily": ",".join(
                [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "wind_speed_10m_max",
                    "surface_pressure_mean",
                    "relative_humidity_2m_mean",
                    "cloud_cover_mean",
                    "shortwave_radiation_sum",
                ]
            ),
            "timezone": "auto",
        },
    )
    return _daily_frame(_normalize_daily_payload(payload))


def _feature_row(observations: pd.DataFrame) -> pd.DataFrame:
    if len(observations) < 3:
        raise ValueError("At least three observations are required to build features.")
    df = observations.sort_values("date").reset_index(drop=True)
    if not df["date"].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Recent weather observations must have consecutive daily dates.")
    current = df.iloc[-1]
    previous = df.iloc[-2]
    prior = df.iloc[-3]
    values = {
        "temp_max_lag1": current["temp_max"],
        "temp_max_lag2": previous["temp_max"],
        "temp_max_lag3": prior["temp_max"],
        "temp_min_lag1": current["temp_min"],
        "temp_min_lag2": previous["temp_min"],
        "precip_lag1": current["precip"],
        "precip_lag2": previous["precip"],
        "wind_lag1": current["wind"],
        "pressure_lag1": current["pressure"],
        "humidity_lag1": current["humidity"],
        "cloudcover_lag1": current["cloudcover"],
        "radiation_lag1": current["radiation"],
        "pressure_trend": current["pressure"] - previous["pressure"],
        "humidity_trend": current["humidity"] - previous["humidity"],
        "temp_range_lag1": current["temp_max"] - current["temp_min"],
        "precip_rolling3": df["precip"].iloc[-3:].mean(),
        "wind_rolling3": df["wind"].iloc[-3:].mean(),
        "radiation_rolling3": df["radiation"].iloc[-3:].mean(),
    }
    return pd.DataFrame([[values[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def get_live_prediction(lat: float, lon: float, location_name: str) -> dict[str, Any]:
    """Predict tomorrow's maximum temperature and rain probability."""
    observations = fetch_recent(lat, lon)
    features = _feature_row(observations)
    temp_model, rain_model = _load_models()
    temp_delta = float(temp_model.predict(features)[0])
    class_index = list(rain_model.classes_).index(1)
    rain_probability = float(rain_model.predict_proba(features)[0, class_index])
    latest_date = observations["date"].iloc[-1].date()
    prediction_date = latest_date + timedelta(days=1)
    return {
        "location": location_name,
        "latitude": lat,
        "longitude": lon,
        "latest_observation_date": latest_date.isoformat(),
        "prediction_date": prediction_date.isoformat(),
        "predicted_temperature": float(observations["temp_max"].iloc[-1] + temp_delta),
        "predicted_temp_delta": temp_delta,
        "rain_probability": rain_probability,
        "rain_predicted": rain_probability >= 0.5,
        "model_features": list(FEATURE_COLUMNS),
    }


def parse_wmo_code(code: int) -> dict[str, str]:
    """Return a display description, icon, and theme for a WMO code."""
    description, icon, theme = WMO_CONDITIONS.get(code, ("Unknown", "❔", "clear"))
    return {"description": description, "icon": icon, "theme": theme}


def fetch_7day_forecast(lat: float, lon: float) -> list[dict[str, Any]]:
    """Fetch and normalize the next seven daily forecast entries."""
    payload = _request_json(
        FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "forecast_days": 7,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "timezone": "auto",
        },
    )
    daily = payload.get("daily", {})
    required = ["time", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "weather_code"]
    if any(column not in daily for column in required):
        raise ValueError("Forecast response is missing required daily fields.")
    forecast = []
    for values in zip(*(daily[column] for column in required)):
        day, maximum, minimum, precipitation, code = values
        condition = parse_wmo_code(int(code))
        forecast.append(
            {
                "date": str(day),
                "temp_max": float(maximum),
                "temp_min": float(minimum),
                "precipitation": float(precipitation),
                "weather_code": int(code),
                **condition,
            }
        )
    return forecast


def geocode_city(city_name: str) -> dict[str, Any]:
    """Resolve a city name through Open-Meteo geocoding."""
    if not city_name.strip():
        raise ValueError("City name cannot be empty.")
    payload = _request_json(
        GEOCODING_URL,
        {"name": city_name.strip(), "count": 1, "language": "en", "format": "json"},
    )
    results = payload.get("results") or []
    if not results:
        raise ValueError(f"No location found for '{city_name}'.")
    result = results[0]
    return {
        "name": result["name"],
        "country": result.get("country", ""),
        "latitude": float(result["latitude"]),
        "longitude": float(result["longitude"]),
    }


def fetch_geospatial_weather_data() -> pd.DataFrame:
    """Fetch supported current indicators for a small set of monitored cities."""
    cities = [
        ("Nashik", "India", 19.9975, 73.7898),
        ("Mumbai", "India", 19.0760, 72.8777),
        ("Delhi", "India", 28.6139, 77.2090),
        ("London", "United Kingdom", 51.5074, -0.1278),
        ("New York", "United States", 40.7128, -74.0060),
        ("Tokyo", "Japan", 35.6762, 139.6503),
        ("Sydney", "Australia", -33.8688, 151.2093),
    ]
    rows = []
    for city, country, lat, lon in cities:
        payload = _request_json(
            FORECAST_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "forecast_days": 1,
                "daily": "temperature_2m_max,precipitation_sum,wind_speed_10m_max,weather_code",
                "timezone": "auto",
            },
        )
        daily = payload.get("daily", {})
        required = ["temperature_2m_max", "precipitation_sum", "wind_speed_10m_max", "weather_code"]
        if any(not daily.get(column) for column in required):
            raise ValueError(f"Weather response is incomplete for {city}.")
        condition = parse_wmo_code(int(daily["weather_code"][0]))
        rows.append(
            {
                "city": city,
                "country": country,
                "lat": lat,
                "lon": lon,
                "temperature": float(daily["temperature_2m_max"][0]),
                "precipitation": float(daily["precipitation_sum"][0]),
                "wind_speed": float(daily["wind_speed_10m_max"][0]),
                "condition": condition["description"],
                "icon": condition["icon"],
                "theme": condition["theme"],
            }
        )
    return pd.DataFrame(rows)