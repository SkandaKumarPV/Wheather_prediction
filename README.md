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

If PowerShell blocks activation, either run this once as the current user or
use the explicit `.venv\Scripts\python.exe` commands shown below:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Confirm the environment:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
```

## Retrieve DVC data and models

The configured exercise remote is a local directory one level above the
repository:

```powershell
.\.venv\Scripts\dvc.exe pull
```

For a new checkout, configure a shared or cloud DVC remote instead of relying
on the local exercise path.

## Run the pipeline

Run the complete local data and model pipeline:

```powershell
.\.venv\Scripts\python.exe src\fetch_historical.py
.\.venv\Scripts\dvc.exe repro
.\.venv\Scripts\dvc.exe status
.\.venv\Scripts\dvc.exe metrics show
.\.venv\Scripts\dvc.exe push
```

Useful pipeline inspection commands:

```powershell
.\.venv\Scripts\dvc.exe dag
.\.venv\Scripts\dvc.exe diff
Get-ChildItem data\raw, data\processed, models, reports
```

`dvc push` uploads to the configured DVC remote. The development remote in
this checkout is a local directory; configure a shared remote before using
`dvc push` from another machine.

The feature stage requires consecutive daily dates and creates 18 input
features. The training target `temp_delta` is the next day's maximum
temperature minus the latest observed maximum temperature. `rain_target` is 1
when next-day precipitation exceeds the configured 1 mm threshold.

Run the stages individually when debugging:

```powershell
.\.venv\Scripts\python.exe src\features.py
.\.venv\Scripts\python.exe src\train_model.py
```

The training script writes deployment models to `models\`, evaluation metrics
to `reports\metrics.json`, and an MLflow run to the configured tracking store.

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

Check that Streamlit is responding:

```powershell
Invoke-WebRequest http://127.0.0.1:8501/_stcore/health -UseBasicParsing
```

The Streamlit dashboard distinguishes the trained next-day ML prediction from
the separate Open-Meteo seven-day forecast. Use the Refresh action to bypass
the ten-minute weather-response cache. Open-Meteo may temporarily return HTTP
502/503/504 responses; the dashboard retries transient failures and displays a
warning instead of hiding the ML prediction.

To stop a locally started server, identify its process and stop only that
specific process:

```powershell
Get-NetTCPConnection -LocalPort 8501 | Select-Object OwningProcess
Stop-Process -Id <process-id>
```

### API smoke tests

Start FastAPI in one terminal, then run these commands in another:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:8000/ -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:8000/docs -UseBasicParsing
```

The live `/predict` and `/forecast` endpoints require the Open-Meteo service.
They can return a clear service error when that external API is unavailable.

## API reference

The project uses two API layers:

1. **Open-Meteo external APIs** provide weather observations, forecasts, and
   city coordinates.
2. **The project FastAPI service** calls Open-Meteo, applies the trained
   models, stores predictions, and exposes JSON endpoints for the dashboard or
   other clients.

### External Open-Meteo APIs

The application uses these public Open-Meteo endpoints:

| Service | URL | Purpose |
| --- | --- | --- |
| Forecast | `https://api.open-meteo.com/v1/forecast` | Recent daily observations, seven-day forecasts, and monitored-city weather |
| Geocoding | `https://geocoding-api.open-meteo.com/v1/search` | Converts a city name into latitude, longitude, and country |
| Archive | `https://archive-api.open-meteo.com/v1/archive` | Historical weather download used by `src/fetch_historical.py` |

The forecast API returns daily fields such as maximum and minimum
temperature, precipitation, wind speed, pressure, humidity, cloud cover,
shortwave radiation, and WMO weather codes. Open-Meteo names are normalized
inside the project before feature engineering:

```text
temperature_2m_max        -> temp_max
temperature_2m_min        -> temp_min
precipitation_sum         -> precip
wind_speed_10m_max        -> wind
surface_pressure_mean     -> pressure
relative_humidity_2m_mean -> humidity
cloud_cover_mean          -> cloudcover
shortwave_radiation_sum   -> radiation
```

The application requests data using latitude, longitude, `daily` variables,
`past_days` or `forecast_days`, and `timezone=auto`. Open-Meteo can
temporarily return HTTP 502, 503, or 504; the client retries these transient
responses and the UI reports an informative warning if the service remains
unavailable.

### FastAPI endpoints

Start the service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

Interactive documentation is available at
`http://127.0.0.1:8000/docs`. The JSON endpoints are:

| Method and path | Input | Response |
| --- | --- | --- |
| `GET /` | None | Serves the HTML accuracy dashboard |
| `GET /health` | None | `{"status": "ok"}` |
| `POST /predict` | JSON latitude, longitude, and location name | Tomorrow's predicted maximum temperature, temperature change, rain probability, prediction date, model feature list, and whether the prediction was logged |
| `POST /forecast` | JSON latitude and longitude | Seven daily entries with dates, maximum/minimum temperatures, precipitation, WMO code, description, icon, and theme |
| `GET /geocode?city=Nashik` | City query | City name, country, latitude, and longitude |
| `GET /accuracy` | Optional latitude, longitude, and `last_n` | Scored count, pending count, average temperature error, rain accuracy, and score history |

Example `/predict` request:

```powershell
$body = @{
    latitude = 19.9975
    longitude = 73.7898
    location_name = "Nashik"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
    -ContentType "application/json" -Body $body
```

Example `/forecast` request:

```powershell
$body = @{
    latitude = 19.9975
    longitude = 73.7898
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/forecast `
    -ContentType "application/json" -Body $body
```

Example `/geocode` and `/accuracy` requests:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/geocode?city=Nashik"
Invoke-RestMethod "http://127.0.0.1:8000/accuracy?last_n=30"
Invoke-RestMethod "http://127.0.0.1:8000/accuracy?latitude=19.9975&longitude=73.7898&last_n=30"
```

Invalid coordinates, missing fields, and invalid accuracy parameters return
HTTP 422. If the external weather service is unavailable, weather-dependent
endpoints return HTTP 502 with a readable error detail.

## Daily prediction tracking

The daily workflow can be called from a scheduler:

```powershell
.\.venv\Scripts\python.exe -c "from src.daily_log import run_daily_prediction; print(run_daily_prediction(19.9975, 73.7898, 'Nashik'))"
```

Predictions are stored in `data/prediction_log.csv`, which is intentionally
ignored because it is a generated runtime log. Only predictions with observed
outcomes are included in accuracy summaries.

Inspect the runtime log:

```powershell
if (Test-Path data\prediction_log.csv) {
    Get-Content data\prediction_log.csv
} else {
    Write-Output "No prediction log has been created yet."
}
```

The log distinguishes:

- **Pending predictions**: a forecast exists, but its target date has not yet
  been observed or the weather service has not returned the actual data.
- **Scored predictions**: the target date has passed and actual temperature and
  rainfall are available.

Therefore, `Scored predictions: 0` does not mean that inference failed. It
means that no forecast has been evaluated yet.

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

Inspect the active tracking URI from PowerShell:

```powershell
if ($env:MLFLOW_TRACKING_URI) {
    $env:MLFLOW_TRACKING_URI
} else {
    Write-Output "Default tracking URI: sqlite:///mlflow.db"
}
```

Start the local MLflow UI:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` to compare runs and inspect metrics. For team
monitoring, set `MLFLOW_TRACKING_URI` to a shared MLflow tracking server and
configure its artifact store; do not commit `mlflow.db` or model artifacts.

When troubleshooting MLflow locally, confirm the tracking database exists:

```powershell
Test-Path mlflow.db
Get-ChildItem mlruns -ErrorAction SilentlyContinue
```

## Geographic limitation

The historical training data was collected for Nashik-area coordinates
(approximately latitude 19.9975, longitude 73.7898). Live weather can be
requested for other cities, but ML predictions outside that training geography
are extrapolations and should be interpreted cautiously.

## Tests

Run all application tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe -m pytest --cov=src --cov-report=term-missing
```

The tests mock weather services and use temporary prediction logs, so normal
test execution does not require internet access or modify the real runtime log.

Run focused test groups:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_features.py
.\.venv\Scripts\python.exe -m pytest -q tests\test_live_and_logger.py
.\.venv\Scripts\python.exe -m pytest -q tests\test_api.py
```

Run the same checks used by CI locally:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests --cov=src --cov-report=term-missing
.\.venv\Scripts\dvc.exe dag
.\.venv\Scripts\dvc.exe status
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m pip check
```

`compileall` creates ignored `__pycache__` directories. Remove generated local
test files when a clean working tree is needed:

```powershell
Get-ChildItem -Path . -Directory -Filter __pycache__ -Recurse |
    Remove-Item -Recurse -Force
Remove-Item .pytest_cache, .coverage -Recurse -Force -ErrorAction SilentlyContinue
```

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

To inspect the latest workflow run on GitHub, open the repository's **Actions**
tab and select **Weather pipeline CI**. A run in `Queued` status has not started
on a runner yet; it is not a test failure. Do not repeatedly push commits while
the runner queue is blocked because each push creates another run.

## Clean validation checklist

Use this checklist before pushing a change:

```powershell
git status --short --branch
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q tests --cov=src --cov-report=term-missing
.\.venv\Scripts\dvc.exe dag
.\.venv\Scripts\dvc.exe status
git diff --check
git status --short --branch
```

Generated runtime files such as `data\prediction_log.csv`, `mlflow.db`,
`mlruns\`, coverage output, and Python caches are ignored by Git. Do not force
add them unless there is a deliberate reason to version one of these artifacts.
