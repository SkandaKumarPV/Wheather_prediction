
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "historical.csv"

LATITUDE = 19.9975
LONGITUDE = 73.7898

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"

API_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "surface_pressure_mean",
    "relative_humidity_2m_mean",
    "cloud_cover_mean",
    "shortwave_radiation_sum",
]


def fetch_historical_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Asia/Kolkata",
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()

    if "daily" not in payload:
        raise ValueError(
            f"Daily data missing from API response: "
            f"{payload}"
        )

    daily = payload["daily"]

    df = pd.DataFrame(daily)
    df = df.rename(
        columns={
            "temperature_2m_max": "temp_max",
            "temperature_2m_min": "temp_min",
            "precipitation_sum": "precip",
            "wind_speed_10m_max": "wind",
            "surface_pressure_mean": "pressure",
            "relative_humidity_2m_mean": "humidity",
            "cloud_cover_mean": "cloudcover",
            "shortwave_radiation_sum": "radiation",
        }
    )

    df["date"] = pd.to_datetime(
        df["time"], errors="coerce"
    )

    df = df.drop(columns=["time"])
    df = df.sort_values("date")
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["date"])

    numeric_columns = [
        "temp_max",
        "temp_min",
        "precip",
        "wind",
        "pressure",
        "humidity",
        "cloudcover",
        "radiation",
    ]

    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    df = df.dropna(subset=numeric_columns)
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid weather observations downloaded.")

    if (df["temp_max"] < df["temp_min"]).any():
        raise ValueError("Found maximum temperature below minimum.")

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved dataset: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print("\nFirst five rows:")
    print(df.head())


if __name__ == "__main__":
    fetch_historical_data()
