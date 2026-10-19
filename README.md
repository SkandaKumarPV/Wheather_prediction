# Weather Predictor

Weather prediction project with a trained machine-learning model, FastAPI
service, and Streamlit dashboard.

## Install

Open PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

If the virtual environment is already available, activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run the Streamlit application

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\streamlit_app.py
```

Open this URL in your browser:

```text
http://127.0.0.1:8501
```

The Streamlit dashboard provides:

- Tomorrow's predicted maximum temperature
- Tomorrow's rain probability
- Seven-day weather forecast
- Prediction accuracy
- Monitored-city weather map

## Run with Docker

Docker Compose runs the FastAPI service and Streamlit dashboard in separate
containers built from the same image. The local `data/` directory is mounted
into both containers so prediction history is retained on the host.

The model artifacts in `models/` must be present before building the image. If
they are not available, pull or reproduce the DVC pipeline first.

Build and start both services:

```powershell
docker compose up --build -d
```

Open the applications:

- Streamlit dashboard: `http://localhost:8501`
- FastAPI documentation: `http://localhost:8000/docs`

Check service status and logs:

```powershell
docker compose ps
docker compose logs -f
```

Stop the containers:

```powershell
docker compose down
```

To check that Streamlit is running:

```powershell
Invoke-WebRequest http://127.0.0.1:8501/_stcore/health -UseBasicParsing
```

The response should be:

```text
ok
```

## Run the FastAPI application

Open a second PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.app:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

Check the API health:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
```

The response should contain:

```json
{"status":"ok"}
```

## Run tests

Run all tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
```

Run tests with coverage:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests --cov=src --cov-report=term-missing
```

Run one test file:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_features.py
.\.venv\Scripts\python.exe -m pytest -q tests\test_live_and_logger.py
.\.venv\Scripts\python.exe -m pytest -q tests\test_api.py
```

The tests mock external weather requests, so they do not require the
Open-Meteo service.

## Run the DVC pipeline

If the DVC data is available:

```powershell
.\.venv\Scripts\dvc.exe pull
.\.venv\Scripts\dvc.exe repro
.\.venv\Scripts\dvc.exe status
```

The pipeline creates the processed training data, model files, and evaluation
metrics.

## Stop the applications

Press `Ctrl+C` in the terminal running Streamlit or FastAPI.

If a process must be stopped by ID:

```powershell
Get-NetTCPConnection -LocalPort 8501 | Select-Object OwningProcess
Stop-Process -Id <process-id>
```
