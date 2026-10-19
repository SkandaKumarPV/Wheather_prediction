"""Build leakage-safe next-day weather training examples."""

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "historical.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "training.csv"

WEATHER_COLUMNS = [
    "temp_max",
    "temp_min",
    "precip",
    "wind",
    "pressure",
    "humidity",
    "cloudcover",
    "radiation",
]
FEATURE_COLUMNS = [
    "temp_max_lag1",
    "temp_max_lag2",
    "temp_max_lag3",
    "temp_min_lag1",
    "temp_min_lag2",
    "precip_lag1",
    "precip_lag2",
    "wind_lag1",
    "pressure_lag1",
    "humidity_lag1",
    "cloudcover_lag1",
    "radiation_lag1",
    "pressure_trend",
    "humidity_trend",
    "temp_range_lag1",
    "precip_rolling3",
    "wind_rolling3",
    "radiation_rolling3",
]
TARGET_COLUMNS = ["temp_delta", "rain_target"]


def load_raw_data(path: Path = RAW_FILE) -> pd.DataFrame:
    """Load and validate daily observations."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    required = {"date", *WEATHER_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing raw weather columns: {sorted(missing)}")
    if df["date"].duplicated().any():
        raise ValueError("Raw data contains duplicate dates.")
    expected = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    actual = pd.DatetimeIndex(df["date"])
    if not actual.equals(expected):
        missing_dates = expected.difference(actual)
        raise ValueError(
            f"Raw data must contain consecutive daily dates; missing {len(missing_dates)} date(s)."
        )
    return df.reset_index(drop=True)


def build_training_table(
    df: pd.DataFrame, rain_threshold: float = 1.0
) -> pd.DataFrame:
    """Create features from observations through day t and targets for day t+1."""
    result = pd.DataFrame({"date": df["date"].shift(-1)})
    result["temp_max_lag1"] = df["temp_max"]
    result["temp_max_lag2"] = df["temp_max"].shift(1)
    result["temp_max_lag3"] = df["temp_max"].shift(2)
    result["temp_min_lag1"] = df["temp_min"]
    result["temp_min_lag2"] = df["temp_min"].shift(1)
    result["precip_lag1"] = df["precip"]
    result["precip_lag2"] = df["precip"].shift(1)
    result["wind_lag1"] = df["wind"]
    result["pressure_lag1"] = df["pressure"]
    result["humidity_lag1"] = df["humidity"]
    result["cloudcover_lag1"] = df["cloudcover"]
    result["radiation_lag1"] = df["radiation"]
    result["pressure_trend"] = df["pressure"] - df["pressure"].shift(1)
    result["humidity_trend"] = df["humidity"] - df["humidity"].shift(1)
    result["temp_range_lag1"] = df["temp_max"] - df["temp_min"]
    result["precip_rolling3"] = df["precip"].rolling(3).mean()
    result["wind_rolling3"] = df["wind"].rolling(3).mean()
    result["radiation_rolling3"] = df["radiation"].rolling(3).mean()
    result["temp_delta"] = df["temp_max"].shift(-1) - df["temp_max"]
    result["rain_target"] = (df["precip"].shift(-1) > rain_threshold).astype("Int64")
    result = result.dropna().reset_index(drop=True)
    result["rain_target"] = result["rain_target"].astype(int)
    return result[["date", *FEATURE_COLUMNS, *TARGET_COLUMNS]]


def main() -> None:
    """Build and save the processed training table."""
    with (ROOT / "params.yaml").open(encoding="utf-8") as file:
        params = yaml.safe_load(file)
    training = build_training_table(
        load_raw_data(),
        rain_threshold=params["data"]["rain_threshold_mm"],
    )
    if len(training) == 0:
        raise ValueError("Feature engineering produced no training rows.")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(training)} rows with {len(FEATURE_COLUMNS)} features to {OUTPUT_FILE}")
    print(training.isna().sum().to_string())


if __name__ == "__main__":
    main()