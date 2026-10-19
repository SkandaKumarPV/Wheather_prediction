from pathlib import Path

from src import app, logger


def test_api_routes_with_mocked_services(monkeypatch, tmp_path):
    monkeypatch.setattr(logger, "LOG_FILE", Path(tmp_path) / "prediction_log.csv")
    monkeypatch.setattr(
        app,
        "get_live_prediction",
        lambda lat, lon, name: {
            "location": name,
            "latitude": lat,
            "longitude": lon,
            "prediction_date": "2025-01-02",
            "latest_observation_date": "2025-01-01",
            "predicted_temperature": 30.0,
            "predicted_temp_delta": 0.2,
            "rain_probability": 0.25,
            "rain_predicted": False,
        },
    )
    monkeypatch.setattr(app, "fetch_7day_forecast", lambda lat, lon: [{"date": "2025-01-02"}])
    monkeypatch.setattr(app, "geocode_city", lambda city: {"name": city})

    assert app.health() == {"status": "ok"}
    prediction = app.predict(
        app.PredictionRequest(latitude=1, longitude=2, location_name="Test")
    )
    assert prediction["logged"] is True
    assert app.forecast(app.LocationRequest(latitude=1, longitude=2))["forecast"]
    assert app.geocode("Test")["name"] == "Test"
    assert app.accuracy()["scored_predictions"] == 0
    assert {"/", "/health", "/predict", "/forecast", "/geocode", "/accuracy"} <= set(
        app.app.openapi()["paths"]
    )
