from pathlib import Path

import pandas as pd
import numpy as np

from src import logger, predict_live


def test_live_prediction_uses_class_one_and_tomorrow_date(monkeypatch):
    observations = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3),
            "temp_max": [30.0, 31.0, 32.0],
            "temp_min": [20.0, 21.0, 22.0],
            "precip": [0.0, 0.0, 2.0],
            "wind": [10.0, 10.0, 10.0],
            "pressure": [1000.0, 1001.0, 1002.0],
            "humidity": [50.0, 51.0, 52.0],
            "cloudcover": [20.0, 20.0, 20.0],
            "radiation": [100.0, 100.0, 100.0],
        }
    )

    class TempModel:
        def predict(self, values):
            assert list(values.columns) == predict_live.FEATURE_COLUMNS
            return [-0.5]

    class RainModel:
        classes_ = [0, 1]

        def predict_proba(self, values):
            return np.array([[0.8, 0.2]])

    monkeypatch.setattr(predict_live, "fetch_recent", lambda *args: observations)
    monkeypatch.setattr(predict_live, "_load_models", lambda: (TempModel(), RainModel()))
    result = predict_live.get_live_prediction(1, 2, "Test")
    assert result["prediction_date"] == "2025-01-04"
    assert result["predicted_temperature"] == 31.5
    assert result["rain_probability"] == 0.2
    assert result["rain_predicted"] is False


def test_logger_deduplicates_and_scores_predictions(monkeypatch, tmp_path):
    log_path = Path(tmp_path) / "prediction_log.csv"
    monkeypatch.setattr(logger, "LOG_FILE", log_path)
    observations = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3),
            "temp_max": [30.0, 31.0, 32.0],
            "precip": [0.0, 2.0, 0.0],
        }
    )
    monkeypatch.setattr(logger, "fetch_recent", lambda *args, **kwargs: observations)
    prediction = {
        "prediction_date": "2025-01-02",
        "latest_observation_date": "2025-01-01",
        "location": "Test",
        "latitude": 1,
        "longitude": 2,
        "predicted_temperature": 31.5,
        "rain_probability": 0.8,
        "rain_predicted": True,
    }
    assert logger.log_new_prediction(prediction) is True
    assert logger.log_new_prediction(prediction) is False
    assert logger.backfill_actuals(1, 2) == 1
    summary = logger.get_accuracy_summary()
    assert summary["scored_predictions"] == 1
    assert summary["average_temperature_error"] == 0.5
    assert summary["rain_accuracy"] == 1.0
