"""Run the daily prediction logging and backfill workflow."""

from .logger import backfill_actuals, get_accuracy_summary, log_new_prediction
from .predict_live import get_live_prediction


def run_daily_prediction(
    latitude: float, longitude: float, location: str
) -> dict:
    """Backfill prior results, log today's forecast, and return the summary."""
    backfill_actuals(latitude, longitude)
    prediction = get_live_prediction(latitude, longitude, location)
    log_new_prediction(prediction)
    return {
        "prediction": prediction,
        "accuracy": get_accuracy_summary(),
    }