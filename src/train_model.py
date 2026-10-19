"""Train, compare, evaluate, and save the weather models."""

from pathlib import Path
import json

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "training.csv"
MODELS_DIR = ROOT / "models"
METRICS_FILE = ROOT / "reports" / "metrics.json"
PARAMS_FILE = ROOT / "params.yaml"


def make_regressor(name: str, params: dict) -> object:
    if name == "random_forest":
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=42, **params)
    raise ValueError(f"Unknown regressor: {name}")


def main() -> None:
    with PARAMS_FILE.open(encoding="utf-8") as file:
        params = yaml.safe_load(file)
    data = pd.read_csv(DATA_FILE, parse_dates=["date"])
    x = data[FEATURE_COLUMNS]
    y_temp = data["temp_delta"]
    y_rain = data["rain_target"]

    test_size = max(1, int(len(data) * params["data"]["test_fraction"]))
    split_at = len(data) - test_size
    x_train, x_test = x.iloc[:split_at], x.iloc[split_at:]
    y_temp_train, y_temp_test = y_temp.iloc[:split_at], y_temp.iloc[split_at:]
    y_rain_train, y_rain_test = y_rain.iloc[:split_at], y_rain.iloc[split_at:]

    regressors = {
        "random_forest": params["random_forest"],
        "hist_gradient_boosting": params["hist_gradient_boosting"],
    }
    cv = TimeSeriesSplit(n_splits=params["data"]["cv_splits"])
    cv_mae = {}
    for name, model_params in regressors.items():
        scores = []
        for train_index, validation_index in cv.split(x_train):
            model = make_regressor(name, model_params)
            model.fit(x_train.iloc[train_index], y_temp_train.iloc[train_index])
            prediction = model.predict(x_train.iloc[validation_index])
            scores.append(mean_absolute_error(y_temp_train.iloc[validation_index], prediction))
        cv_mae[name] = float(sum(scores) / len(scores))

    selected_name = min(cv_mae, key=cv_mae.get)
    temp_holdout_model = make_regressor(selected_name, regressors[selected_name])
    temp_holdout_model.fit(x_train, y_temp_train)
    temp_prediction = temp_holdout_model.predict(x_test)
    persistence_prediction = [0.0] * len(y_temp_test)

    rain_holdout_model = RandomForestClassifier(
        n_estimators=params["random_forest"]["n_estimators"],
        max_depth=params["random_forest"]["max_depth"],
        min_samples_leaf=params["random_forest"]["min_samples_leaf"],
        random_state=42,
        n_jobs=-1,
    )
    rain_holdout_model.fit(x_train, y_rain_train)
    rain_prediction = rain_holdout_model.predict(x_test)
    rain_class_index = list(rain_holdout_model.classes_).index(1)
    rain_probability = rain_holdout_model.predict_proba(x_test)[:, rain_class_index]

    model_mae = mean_absolute_error(y_temp_test, temp_prediction)
    baseline_mae = mean_absolute_error(y_temp_test, persistence_prediction)
    metrics = {
        "selected_temperature_model": selected_name,
        "cross_validated_mae": cv_mae,
        "test_temperature_mae": float(model_mae),
        "persistence_baseline_mae": float(baseline_mae),
        "temperature_beats_persistence": bool(model_mae < baseline_mae),
        "rain_accuracy": float(accuracy_score(y_rain_test, rain_prediction)),
        "rain_precision": float(precision_score(y_rain_test, rain_prediction, zero_division=0)),
        "rain_recall": float(recall_score(y_rain_test, rain_prediction, zero_division=0)),
        "rain_f1": float(f1_score(y_rain_test, rain_prediction, zero_division=0)),
        "rain_brier_score": float(brier_score_loss(y_rain_test, rain_probability)),
        "rain_roc_auc": float(roc_auc_score(y_rain_test, rain_probability)),
        "training_rows": int(split_at),
        "test_rows": int(test_size),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_model = make_regressor(selected_name, regressors[selected_name])
    temp_model.fit(x, y_temp)
    rain_model = RandomForestClassifier(
        n_estimators=params["random_forest"]["n_estimators"],
        max_depth=params["random_forest"]["max_depth"],
        min_samples_leaf=params["random_forest"]["min_samples_leaf"],
        random_state=42,
        n_jobs=-1,
    )
    rain_model.fit(x, y_rain)
    joblib.dump(temp_model, MODELS_DIR / "temp_model.joblib")
    joblib.dump(rain_model, MODELS_DIR / "rain_model.joblib")
    METRICS_FILE.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()