# Weather Predictor

An end-to-end weather machine-learning project with DVC data versioning, live
Open-Meteo inference, prediction tracking, a FastAPI service, and a Streamlit
dashboard.

## Features

- Historical daily weather ingestion and validation
- Leakage-safe next-day feature engineering
- Random Forest and HistGradientBoosting model comparison
- Next-day maximum-temperature and rain predictions
- Seven-day Open-Meteo forecast
- City geocoding and preset locations
- Prediction logging and scored accuracy tracking
- FastAPI JSON API and lightweight HTML accuracy dashboard
- Streamlit forecast, accuracy, and monitored-city map views
- Reproducible DVC feature and training stages
- MLflow experiment tracking for parameters, metrics, models, and artifacts

## Architecture

```text
Open-Meteo archive data
        |
        v
data/raw/historical.csv
        |
        v
src/features.py -> data/processed/training.csv
        |
        v
src/train_model.py -> models/*.joblib + reports/metrics.json
        |
        +--> src/predict_live.py
        |        +--> FastAPI: src/app.py
        |        +--> Streamlit: src/streamlit_app.py
        |        +--> logger: src/logger.py
        |
        +--> DVC remote storage
```

## Project structure

```text
data/raw/             DVC-tracked raw observations
data/processed/       DVC-generated training table
models/               DVC-generated deployment models
reports/              verified evaluation metrics
mlflow.db             local MLflow tracking database (ignored)
src/fetch_historical.py
src/features.py       leakage-safe feature engineering
src/train_model.py    training, selection, and evaluation
src/predict_live.py   live inference and weather service adapters
src/logger.py         prediction log and accuracy scoring
src/daily_log.py      daily prediction workflow
src/app.py            FastAPI backend
src/streamlit_app.py  Streamlit dashboard
static/index.html     lightweight accuracy dashboard
dvc.yaml              reproducible feature/train pipeline
params.yaml           tracked pipeline parameters
tests/                application tests
```

## Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

The repository was validated with Python 3.11.

## Retrieve DVC data and models

The configured exercise remote is a local directory one level above the
repository:

```powershell
.\.venv\Scripts\dvc.exe pull
```

For a new checkout, configure a shared or cloud DVC remote instead of relying
on the local exercise path.

## Run the pipeline

```powershell
.\.venv\Scripts\python.exe src\fetch_historical.py
.\.venv\Scripts\dvc.exe repro
.\.venv\Scripts\dvc.exe status
.\.venv\Scripts\dvc.exe metrics show
.\.venv\Scripts\dvc.exe push
```

The feature stage requires consecutive daily dates and creates 18 input
features. The training target `temp_delta` is the next day's maximum
temperature minus the latest observed maximum temperature. `rain_target` is 1
when next-day precipitation exceeds the configured 1 mm threshold.

## Run the applications

FastAPI:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.app:app --reload
```

Open `http://127.0.0.1:8000/` for the HTML accuracy dashboard or
`http://127.0.0.1:8000/docs` for interactive API documentation.

Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\streamlit_app.py
```

The Streamlit dashboard distinguishes the trained next-day ML prediction from
the separate Open-Meteo seven-day forecast. Use the Refresh action to bypass
the ten-minute weather-response cache.

## Daily prediction tracking

The daily workflow can be called from a scheduler:

```powershell
.\.venv\Scripts\python.exe -c "from src.daily_log import run_daily_prediction; print(run_daily_prediction(19.9975, 73.7898, 'Nashik'))"
```

Predictions are stored in `data/prediction_log.csv`, which is intentionally
ignored because it is a generated runtime log. Only predictions with observed
outcomes are included in accuracy summaries.

## Current verified evaluation

The current holdout metrics are stored in
[reports/metrics.json](reports/metrics.json):

```text
Selected temperature model: Random Forest
Temperature test MAE:       0.84707 °C
Persistence baseline MAE:   0.90302 °C
Rain accuracy:               0.91209
Rain F1:                     0.89041
Rain Brier score:            0.05956
Rain ROC-AUC:                0.97777
```

The holdout models are evaluated on the final chronological 20% split. The
saved deployment models are then retrained on all available training rows.

## MLflow monitoring

Training logs the selected model, pipeline parameters, cross-validation scores,
holdout metrics, final models, metrics report, and parameter file to the
`weather-prediction` MLflow experiment. By default, tracking uses the ignored
local SQLite database `mlflow.db`.

Run training or the DVC pipeline:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
.\.venv\Scripts\dvc.exe repro
```

Start the local MLflow UI:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` to compare runs and inspect metrics. For team
monitoring, set `MLFLOW_TRACKING_URI` to a shared MLflow tracking server and
configure its artifact store; do not commit `mlflow.db` or model artifacts.

## Geographic limitation

The historical training data was collected for Nashik-area coordinates
(approximately latitude 19.9975, longitude 73.7898). Live weather can be
requested for other cities, but ML predictions outside that training geography
are extrapolations and should be interpreted cautiously.

## Tests

Run the application tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe -m pytest --cov=src --cov-report=term-missing
```

The tests mock weather services and use temporary prediction logs, so normal
test execution does not require internet access or modify the real runtime log.

## Continuous integration

GitHub Actions is configured in
[.github/workflows/ci.yml](.github/workflows/ci.yml). Every push and pull
request to `main` installs the pinned dependencies, runs the full test suite
with coverage, validates the DVC graph, pulls DVC data when the
`DVC_REMOTE_URL` repository secret is configured, reproduces the DVC pipeline
when raw data is available, and validates model feature compatibility.

For full CI pipeline reproduction, add a shared DVC remote URL as the
`DVC_REMOTE_URL` GitHub Actions repository secret. The current local exercise
remote is machine-specific and is intentionally not usable by hosted runners.
