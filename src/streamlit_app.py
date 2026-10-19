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
    from logger import (
        backfill_actuals,
        get_accuracy_summary,
        log_new_prediction,
        seed_historical_predictions,
    )
    from predict_live import (
        fetch_7day_forecast,
        fetch_geospatial_weather_data,
        geocode_city,
        get_live_prediction,
    )

st.set_page_config(
    page_title="Weather Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRESET_CITIES = {
    "Nashik": (19.9975, 73.7898, "India"),
    "Mumbai": (19.0760, 72.8777, "India"),
    "Delhi": (28.6139, 77.2090, "India"),
    "London": (51.5074, -0.1278, "United Kingdom"),
    "New York": (40.7128, -74.0060, "United States"),
    "Tokyo": (35.6762, 139.6503, "Japan"),
    "Sydney": (-33.8688, 151.2093, "Australia"),
}


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --ink: #edf4ff; --muted: #9aabc2; --blue: #60a5fa; --navy: #0b1220; --panel: #151f31; --border: #26364e; }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .stApp { background: #0b1220; color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(11,18,32,.9); }
        [data-testid="stSidebar"] { background: linear-gradient(180deg,#0a1426 0%,#101e34 100%); }
        [data-testid="stSidebar"] * { color: #e8eef8; }
        [data-testid="stSidebar"] .stCaption { color: #aebdd2; }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] input { background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.18); }
        [data-testid="stSidebar"] .stButton > button { background: #2563eb; color: white; border: 0; }
        .block-container { max-width: 1440px; padding: 2.2rem 3.2rem 3rem; }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
        h1 { letter-spacing: -.04em; font-size: 2.45rem; }
        .eyebrow { color: #60a5fa; font-size: .76rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
        .hero { background: linear-gradient(135deg,#142541,#1c416a); border: 1px solid #2a4d75; border-radius: 24px; padding: 2rem 2.3rem; color: white; box-shadow: 0 16px 40px rgba(0,0,0,.3); margin: .6rem 0 1.25rem; }
        .hero h1 { color: white; margin: .3rem 0 .2rem; font-size: 2.35rem; }
        .hero p { color: #c9d9ed; margin: 0; font-size: 1rem; }
        .hero-location { color: #93c5fd; font-weight: 600; }
        .status-pill { display: inline-block; padding: .42rem .75rem; border-radius: 99px; background: rgba(255,255,255,.12); color: #dbeafe; font-size: .8rem; font-weight: 600; }
        .metric-card { background: var(--panel); border: 1px solid var(--border); border-radius: 18px; padding: 1.1rem 1.25rem; min-height: 126px; box-shadow: 0 5px 18px rgba(0,0,0,.18); }
        .metric-label { color: var(--muted); font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
        .metric-value { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.85rem; font-weight: 700; margin-top: .4rem; }
        .metric-help { color: #8294ad; font-size: .78rem; margin-top: .25rem; }
        .section-note { color: var(--muted); font-size: .92rem; margin-top: -.55rem; margin-bottom: 1.1rem; }
        div[data-testid="stTabs"] button { font-weight: 600; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
        .stButton > button { background: #1b2a41; color: #e8eef8; border: 1px solid #344966; border-radius: 10px; font-weight: 600; min-height: 2.5rem; }
        .stButton > button:hover { border-color: #60a5fa; color: #bfdbfe; }
        [data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; }
        [data-baseweb="tab-list"] { gap: .35rem; }
        [data-baseweb="tab"] { color: #9aabc2; }
        [data-baseweb="tab"][aria-selected="true"] { color: #93c5fd; }
        [data-testid="stAlert"] { background: #17263a; border-color: #31557c; color: #dbeafe; }
        [data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def _metric_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-help">{help_text}</div></div>',
        unsafe_allow_html=True,
    )


def _render_forecast(forecast: list[dict]) -> None:
    st.markdown("#### Coming week")
    columns = st.columns(min(len(forecast), 7))
    for index, item in enumerate(forecast):
        with columns[index % len(columns)]:
            day = pd.to_datetime(item["date"]).strftime("%a")
            date_label = pd.to_datetime(item["date"]).strftime("%d %b")
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{day} · {date_label}</div>'
                f'<div style="font-size:1.7rem;margin:.45rem 0">{item["icon"]}</div>'
                f'<div class="metric-value">{item["temp_max"]:.1f}°</div>'
                f'<div class="metric-help">{item["temp_min"]:.1f}° min · {item["description"]}</div>'
                f'<div class="metric-help">{item["precipitation"]:.1f} mm rain</div></div>',
                unsafe_allow_html=True,
            )

    frame = pd.DataFrame(forecast)
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=frame["date"], y=frame["temp_max"], name="High",
        mode="lines+markers", line={"color": "#3b82f6", "width": 3},
    ))
    figure.add_trace(go.Scatter(
        x=frame["date"], y=frame["temp_min"], name="Low",
        mode="lines+markers", line={"color": "#9bbce8", "width": 3},
    ))
    figure.update_layout(
        height=300, margin={"l": 10, "r": 10, "t": 25, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Temperature (°C)", xaxis_title=None,
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="#e7ebf2", zeroline=False)
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def _sidebar() -> None:
    st.sidebar.markdown("## 🌦️ Weather IQ")
    st.sidebar.caption("Machine-learning weather intelligence")
    st.sidebar.divider()
    st.sidebar.markdown("**Choose a location**")
    current_name = st.session_state.location["name"]
    preset = st.sidebar.selectbox(
        "Popular cities", list(PRESET_CITIES),
        index=list(PRESET_CITIES).index(current_name) if current_name in PRESET_CITIES else 0,
        label_visibility="collapsed",
    )
    if st.sidebar.button("Use selected city", use_container_width=True):
        name, (lat, lon, country) = preset, PRESET_CITIES[preset]
        _set_location(name, lat, lon, country)
        cached_prediction.clear()
        st.rerun()
    city_query = st.sidebar.text_input("Search any city", placeholder="e.g. Paris")
    if st.sidebar.button("Search location", use_container_width=True) and city_query.strip():
        try:
            result = geocode_city(city_query)
            _set_location(result["name"], result["latitude"], result["longitude"], result["country"])
            cached_prediction.clear()
            st.rerun()
        except (OSError, ValueError, KeyError) as error:
            st.sidebar.error(str(error))
    st.sidebar.divider()
    if st.sidebar.button("↻  Refresh weather data", use_container_width=True):
        cached_prediction.clear()
        cached_forecast.clear()
        cached_map_data.clear()
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.caption("Data powered by Open-Meteo")
    st.sidebar.caption("Forecasts refresh every 10 minutes.")


def main() -> None:
    _apply_theme()
    if "location" not in st.session_state:
        name, (lat, lon, country) = next(iter(PRESET_CITIES.items()))
        _set_location(name, lat, lon, country)
    _sidebar()

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

    st.markdown(
        f'<div class="hero"><div class="status-pill">● LIVE MODEL INFERENCE</div>'
        f'<h1>Weather, made clearer.</h1>'
        f'<p>Tomorrow’s outlook for <span class="hero-location">{location["name"]}, {location["country"]}</span> '
        f'· updated from the latest observations</p></div>',
        unsafe_allow_html=True,
    )

    first, second, third, fourth = st.columns(4)
    with first:
        _metric_card("Tomorrow's high", f'{prediction["predicted_temperature"]:.1f} °C', "ML temperature prediction")
    with second:
        _metric_card("Temperature change", f'{prediction["predicted_temp_delta"]:+.1f} °C', "Compared with latest observation")
    with third:
        _metric_card("Rain probability", f'{prediction["rain_probability"]:.0%}', "Probability of measurable rain")
    with fourth:
        _metric_card("Prediction date", pd.to_datetime(prediction["prediction_date"]).strftime("%d %b"), "Forecast horizon: tomorrow")

    if prediction["rain_predicted"]:
        st.warning(f'☔ Rain is likely tomorrow in {location["name"]}. Consider carrying an umbrella.')
    else:
        st.success(f'☀️ Rain is unlikely tomorrow in {location["name"]}. Conditions look favorable.')

    st.markdown(
        '<div class="section-note">The model was trained on historical weather around Nashik. '
        'Predictions for other cities are geographic extrapolations.</div>',
        unsafe_allow_html=True,
    )
    forecast_tab, accuracy_tab, map_tab = st.tabs(
        ["📅 7-day outlook", "📊 Model performance", "🗺️ Global weather map"]
    )

    with forecast_tab:
        st.subheader("Seven-day forecast")
        st.markdown('<div class="section-note">Daily outlook from Open-Meteo, separate from the ML prediction above.</div>', unsafe_allow_html=True)
        if forecast:
            _render_forecast(forecast)
        else:
            st.info("The forecast service is temporarily unavailable. Refresh data to try again.")

    with accuracy_tab:
        st.subheader("Model performance")
        st.markdown('<div class="section-note">Accuracy is calculated as predictions become eligible for observed-weather scoring.</div>', unsafe_allow_html=True)
        try:
            log_new_prediction(prediction)
            backfill_actuals(location["lat"], location["lon"])
            summary = get_accuracy_summary()
        except (OSError, ValueError, KeyError) as error:
            st.error(str(error))
            summary = get_accuracy_summary()
        metrics = st.columns(4)
        values = [
            ("Scored predictions", str(summary["scored_predictions"]), "Completed evaluations"),
            ("Pending predictions", str(summary["pending_predictions"]), "Awaiting observations"),
            ("Avg. temperature error", "—" if summary["average_temperature_error"] is None else f'{summary["average_temperature_error"]:.2f} °C', "Lower is better"),
            ("Rain accuracy", "—" if summary["rain_accuracy"] is None else f'{summary["rain_accuracy"]:.1%}', "Correct rain decisions"),
        ]
        for column, (label, value, help_text) in zip(metrics, values):
            with column:
                _metric_card(label, value, help_text)
        if summary["pending_predictions"]:
            st.info("Pending predictions are scored after their prediction date, when observed weather becomes available.")
        with st.expander("Create retrospective estimates"):
            st.caption("Useful for building a historical baseline without treating estimates as live forecasts.")
            if st.button("Seed retrospective estimates"):
                seed_historical_predictions(location["lat"], location["lon"])
                st.rerun()
        if summary["history"]:
            st.dataframe(pd.DataFrame(summary["history"]), use_container_width=True, hide_index=True)
        else:
            st.info("No scored predictions are available yet.")

    with map_tab:
        st.subheader("Monitored-city weather")
        st.markdown('<div class="section-note">A live snapshot across cities monitored by the application.</div>', unsafe_allow_html=True)
        try:
            map_data = cached_map_data()
            figure = go.Figure(go.Scattergeo(
                lat=map_data["lat"], lon=map_data["lon"], text=map_data["city"],
                mode="markers+text", textposition="top center",
                marker={"size": 13, "color": map_data["temperature"], "colorscale": "Blues", "showscale": True, "colorbar": {"title": "°C"}},
                hovertemplate="<b>%{text}</b><br>%{marker.color:.1f} °C<extra></extra>",
            ))
            figure.update_layout(
                geo={"showland": True, "landcolor": "#e7edf5", "showocean": True, "oceancolor": "#f4f8fc", "projection_type": "natural earth"},
                height=500, margin={"l": 0, "r": 0, "t": 0, "b": 0}, paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
            st.dataframe(map_data, use_container_width=True, hide_index=True)
        except (OSError, ValueError, KeyError) as error:
            st.error(f"Unable to load map data: {error}")


if __name__ == "__main__":
    main()
