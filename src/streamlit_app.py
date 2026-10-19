"""Interactive Streamlit dashboard for the weather prediction service."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from .logger import (
        backfill_actuals,
        get_accuracy_summary,
        log_new_prediction,
        seed_historical_predictions,
    )
    from .predict_live import (
        fetch_7day_forecast,
        fetch_geospatial_weather_data,
        geocode_city,
        get_live_prediction,
    )
except ImportError:
    from logger import backfill_actuals, get_accuracy_summary, log_new_prediction, seed_historical_predictions
    from predict_live import fetch_7day_forecast, fetch_geospatial_weather_data, geocode_city, get_live_prediction

st.set_page_config(page_title="Weather Predictor", page_icon="🌦️", layout="wide")

PRESET_CITIES = {
    "Nashik": (19.9975, 73.7898, "India"),
    "Mumbai": (19.0760, 72.8777, "India"),
    "Delhi": (28.6139, 77.2090, "India"),
    "London": (51.5074, -0.1278, "United Kingdom"),
    "New York": (40.7128, -74.0060, "United States"),
    "Tokyo": (35.6762, 139.6503, "Japan"),
    "Sydney": (-33.8688, 151.2093, "Australia"),
}


@st.cache_data(ttl=600)
def cached_prediction(lat: float, lon: float, name: str) -> dict:
    return get_live_prediction(lat, lon, name)


@st.cache_data(ttl=600)
def cached_forecast(lat: float, lon: float) -> list[dict]:
    return fetch_7day_forecast(lat, lon)


@st.cache_data(ttl=600)
def cached_map_data() -> pd.DataFrame:
    return fetch_geospatial_weather_data()


def _set_location(name: str, lat: float, lon: float, country: str) -> None:
    st.session_state.location = {"name": name, "lat": lat, "lon": lon, "country": country}


def _render_forecast(forecast: list[dict]) -> None:
    columns = st.columns(min(len(forecast), 7))
    for index, item in enumerate(forecast):
        with columns[index % len(columns)]:
            st.metric(
                f"{item['icon']} {item['date']}",
                f"{item['temp_max']:.1f} °C",
                f"{item['temp_min']:.1f} °C min",
            )
            st.caption(f"{item['description']} · {item['precipitation']:.1f} mm")
    chart = pd.DataFrame(forecast).set_index("date")[["temp_max", "temp_min"]]
    st.line_chart(chart)


def main() -> None:
    if "location" not in st.session_state:
        name, (lat, lon, country) = next(iter(PRESET_CITIES.items()))
        _set_location(name, lat, lon, country)

    st.sidebar.title("🌦️ Weather Predictor")
    preset = st.sidebar.selectbox("Popular cities", list(PRESET_CITIES), index=list(PRESET_CITIES).index(st.session_state.location["name"]) if st.session_state.location["name"] in PRESET_CITIES else 0)
    if st.sidebar.button("Use selected city"):
        name, (lat, lon, country) = preset, PRESET_CITIES[preset]
        _set_location(name, lat, lon, country)
        cached_prediction.clear()
        st.rerun()
    city_query = st.sidebar.text_input("Search any city", placeholder="e.g. Paris")
    if st.sidebar.button("Search") and city_query.strip():
        try:
            result = geocode_city(city_query)
            _set_location(result["name"], result["latitude"], result["longitude"], result["country"])
            cached_prediction.clear()
            st.rerun()
        except (OSError, ValueError, KeyError) as error:
            st.sidebar.error(str(error))
    if st.sidebar.button("Refresh data"):
        cached_prediction.clear()
        cached_forecast.clear()
        cached_map_data.clear()
        st.rerun()

    location = st.session_state.location
    try:
        prediction = cached_prediction(location["lat"], location["lon"], location["name"])
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Unable to load weather data: {error}")
        return
    try:
        forecast = cached_forecast(location["lat"], location["lon"])
    except (OSError, ValueError, KeyError) as error:
        forecast = []
        st.warning(f"Seven-day forecast is temporarily unavailable: {error}")

    st.title("Weather Pattern Prediction System")
    st.caption(f"Live ML inference for {location['name']}, {location['country']}")
    if prediction["rain_predicted"]:
        st.warning(f"Rain likely tomorrow ({prediction['rain_probability']:.0%}).")
    else:
        st.success(f"Rain unlikely tomorrow ({prediction['rain_probability']:.0%}).")

    first, second, third = st.columns(3)
    first.metric("Tomorrow maximum", f"{prediction['predicted_temperature']:.1f} °C")
    second.metric("Predicted change", f"{prediction['predicted_temp_delta']:+.1f} °C")
    third.metric("Prediction date", prediction["prediction_date"])
    st.info("The ML model was trained at Nashik coordinates; predictions elsewhere are geographic extrapolations.")

    forecast_tab, accuracy_tab, map_tab = st.tabs(["📅 Seven-day forecast", "📊 Accuracy", "🗺️ Weather map"])
    with forecast_tab:
        st.subheader("Open-Meteo seven-day forecast")
        if forecast:
            _render_forecast(forecast)
            st.caption("This forecast is supplied by Open-Meteo and is separate from the ML prediction above.")
        else:
            st.info("The forecast service is temporarily unavailable. Refresh data to try again.")

    with accuracy_tab:
        st.subheader("Prediction accuracy")
        try:
            log_new_prediction(prediction)
            backfill_actuals(location["lat"], location["lon"])
            summary = get_accuracy_summary()
        except (OSError, ValueError, KeyError) as error:
            st.error(str(error))
            summary = get_accuracy_summary()
        metrics = st.columns(4)
        metrics[0].metric("Scored predictions", summary["scored_predictions"])
        metrics[1].metric("Pending predictions", summary["pending_predictions"])
        metrics[2].metric("Average temperature error", "—" if summary["average_temperature_error"] is None else f"{summary['average_temperature_error']:.2f} °C")
        metrics[3].metric("Rain accuracy", "—" if summary["rain_accuracy"] is None else f"{summary['rain_accuracy']:.1%}")
        if summary["pending_predictions"]:
            st.info(
                "Pending predictions are scored after their prediction date, "
                "when observed weather becomes available."
            )
        if st.button("Seed retrospective estimates"):
            seed_historical_predictions(location["lat"], location["lon"])
            st.rerun()
        if summary["history"]:
            st.dataframe(pd.DataFrame(summary["history"]), use_container_width=True)
        else:
            st.info("No scored predictions are available yet.")

    with map_tab:
        st.subheader("Monitored-city weather map")
        try:
            map_data = cached_map_data()
            st.plotly_chart(
                go.Figure(
                    go.Scattergeo(
                        lat=map_data["lat"],
                        lon=map_data["lon"],
                        text=map_data["city"],
                        mode="markers+text",
                        marker={"size": 14, "color": map_data["temperature"], "colorscale": "Turbo", "showscale": True},
                        hovertemplate="%{text}<br>%{marker.color:.1f} °C<extra></extra>",
                    )
                ).update_layout(geo={"showland": True, "projection_type": "natural earth"}, height=500),
                use_container_width=True,
            )
            st.dataframe(map_data, use_container_width=True, hide_index=True)
        except (OSError, ValueError, KeyError) as error:
            st.error(f"Unable to load map data: {error}")


if __name__ == "__main__":
    main()