import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, build_training_table, load_raw_data


def weather_frame(days: int = 6) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days)
    return pd.DataFrame(
        {
            "date": dates,
            "temp_max": range(30, 30 + days),
            "temp_min": range(20, 20 + days),
            "precip": [0.0, 2.0, 0.0, 3.0, 0.0, 1.0][:days],
            "wind": [10.0] * days,
            "pressure": [1000.0 + i for i in range(days)],
            "humidity": [50.0 + i for i in range(days)],
            "cloudcover": [20.0] * days,
            "radiation": [100.0] * days,
        }
    )


def test_training_table_has_expected_alignment_and_columns():
    result = build_training_table(weather_frame())
    assert len(FEATURE_COLUMNS) == 18
    assert list(result.columns) == ["date", *FEATURE_COLUMNS, "temp_delta", "rain_target"]
    assert result.iloc[0]["date"] == pd.Timestamp("2025-01-04")
    assert result.iloc[0]["temp_delta"] == 1
    assert result.iloc[0]["rain_target"] == 1
    assert result.isna().sum().sum() == 0


def test_missing_daily_date_is_rejected(tmp_path):
    frame = weather_frame()
    frame = frame.drop(index=2)
    path = tmp_path / "historical.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="consecutive daily dates"):
        load_raw_data(path)
