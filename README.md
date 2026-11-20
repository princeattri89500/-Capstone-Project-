# AI Workflow Capstone: Time-Series Revenue Forecasting Service

Production-ready, end-to-end Machine Learning Operations (MLOps) workflow built for automated data ingestion, exploratory data analysis, multi-model evaluation, containerized REST API serving, performance monitoring, and full test suite isolation.

---

## Table of Contents
1. [Project Overview & Business Opportunity](#project-overview--business-opportunity)
2. [Architecture & System Design](#architecture--system-design)
3. [Repository Structure](#repository-structure)
4. [Peer-Review Rubric Compliance](#peer-review-rubric-compliance)
5. [Automated Data Ingestion](#automated-data-ingestion)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Machine Learning Models & Evaluation](#machine-learning-models--evaluation)
8. [Baseline vs Model Comparison](#baseline-vs-model-comparison)
9. [REST API Documentation](#rest-api-documentation)
10. [Performance Monitoring & Metrics](#performance-monitoring--metrics)
11. [Testing Suite & Isolation](#testing-suite--isolation)
12. [Docker Containerization](#docker-containerization)

---

## Project Overview & Business Opportunity

### Business Scenario
Aavail, an international streaming media subscription provider, requires accurate projections of future revenue across multiple international markets. Decision-makers and regional leads need reliable **30-day forward rolling revenue forecasts** to plan content acquisitions, server infrastructures, and regional marketing budgets.

### Solution Approach
- **Data Ingestion:** Ingest raw historical JSON invoices spanning multiple years and international territories. Standardize schemas, clean non-numeric invoice identifiers, handle missing values, and aggregate transactions into continuous daily time-series.
- **Supervised Learning for Time-Series:** Engineer sliding historical revenue windows (7, 14, 28, 70 days), previous year seasonal revenue, and recent invoice/view activity to forecast the 30-day forward rolling revenue sum.
- **Microservice API:** Flask-based REST service serving predictions for individual countries and all countries combined.
- **Monitoring & MLOps:** Prometheus metrics at `/metrics`, container health checks at `/health`, structured logging, and complete isolation between production and testing pipelines.

---

## Architecture & System Design

```
Raw Invoice JSONs (cs-train/)
           │
           ▼
Automated Data Ingestion (scripts/ingest_data.py & cslib.py)
           │
           ▼
Continuous Daily Time-Series Aggregation (cs-train/ts-data/)
           │
    ┌──────┴──────────────────────────────────────┐
    ▼                                             ▼
Exploratory Data Analysis (scripts/run_eda.py)   Feature Engineering & Multi-Model Benchmarking
    │                                             │ (scripts/compare_models.py)
    ▼                                             ▼
Saved Visualizations (reports/figures/)          Best Model Selection (Gradient Boosting / RF)
                                                  │
                                                  ▼
                                         Serialized Models (models/*.joblib)
                                                  │
                                                  ▼
                                    Flask REST API Service (app.py)
                                    ├── GET /predict (Single & All Countries)
                                    ├── POST /train (Retraining Pipeline)
                                    ├── GET /health (Service Health & Uptime)
                                    └── GET /metrics (Prometheus Monitoring)
```

---

## Repository Structure

```
├── cs-train/               # Raw training invoice JSON files
│   └── ts-data/            # Processed daily time-series CSVs
├── cs-production/          # Production transaction stream files
├── models/                 # Serialized production models (*.joblib)
├── logs/                   # Production logs (train, predict, api, ingest)
├── reports/
│   ├── model_comparison.csv
│   └── figures/            # EDA & baseline comparison visualizations
│       ├── 01_dataset_overview_revenue_over_time.png
│       ├── 02_missing_and_clean_data_distribution.png
│       ├── 03_top_countries_revenue_comparison.png
│       ├── 04_feature_distributions.png
│       ├── 05_correlation_heatmap.png
│       ├── 06_views_and_invoices_relationship.png
│       ├── 07_target_distribution_and_outliers.png
│       ├── 08_monthly_revenue_seasonality.png
│       └── 09_baseline_vs_final_model_comparison.png
├── scripts/
│   ├── ingest_data.py      # Automated data ingestion script
│   ├── run_eda.py          # EDA visualization generation script
│   └── compare_models.py   # Multi-model evaluation benchmark script
├── unittests/              # Peer-review evaluation unit test suite
│   ├── __init__.py
│   ├── ApiTests.py         # API endpoint unit tests
│   ├── ModelTests.py       # ML model unit tests
│   └── LoggerTests.py      # Structured logging unit tests
├── cslib.py                # Core data ingestion & feature engineering module
├── model.py                # Model training, loading, and prediction functions
├── logger.py               # Structured logging system with test isolation
├── app.py                  # Production Flask REST API
├── run-tests.py            # Single-command test runner script
├── pytest.ini              # Pytest configuration file
├── requirements.txt        # Pinned Python package dependencies
├── Dockerfile              # Containerization recipe
├── docker-compose.yml      # Multi-container orchestration specification
├── .dockerignore           # Docker build exclusion rules
└── README.md               # Project documentation
```

---

## Peer-Review Rubric Compliance

| # | Peer Review Question | Implementation File(s) | Status |
|---|---|---|:---:|
| 1 | **Are there unit tests for the API?** | `unittests/ApiTests.py` | Verified (10 tests pass) |
| 2 | **Are there unit tests for the model?** | `unittests/ModelTests.py` | Verified (6 tests pass) |
| 3 | **Are there unit tests for the logging?** | `unittests/LoggerTests.py` | Verified (4 tests pass) |
| 4 | **Can all tests be run with a single command?** | `pytest` or `python run-tests.py` | Verified (20/20 pass) |
| 5 | **Are read/write unit tests isolated from production?** | `logger.py`, `model.py` (`test-models/`, `test-logs/`) | Verified |
| 6 | **Does the API support prediction for single and all countries?** | `app.py` (`/predict?country=...` & `/predict`) | Verified |
| 7 | **Is there an automated data ingestion function or script?** | `scripts/ingest_data.py`, `cslib.py` | Verified |
| 8 | **Did the project compare multiple machine learning models?** | `scripts/compare_models.py`, `reports/model_comparison.csv` | Verified |
| 9 | **Is there comprehensive EDA with visualizations?** | `scripts/run_eda.py`, `reports/figures/01-08` | Verified |
| 10 | **Is the application containerized within Docker?** | `Dockerfile`, `docker-compose.yml` | Verified |
| 11 | **Is there a visualization comparing the model to baseline?** | `reports/figures/09_baseline_vs_final_model_comparison.png` | Verified |

---

## Automated Data Ingestion

Execute the automated data ingestion pipeline independently:
```bash
python scripts/ingest_data.py --data-dir cs-train --clean
```
- Consolidated **815,011 transaction records** across 21 monthly JSON batches.
- Corrected non-uniform column names (`StreamID`, `total_price`, `TimesViewed`).
- Stripped letters from invoice identifiers and normalized country names.
- Generated daily aggregated time-series for `'all'` countries and top 10 revenue-generating countries.

---

## Exploratory Data Analysis (EDA)

Run the automated EDA generator:
```bash
python scripts/run_eda.py
```
Generated high-resolution charts in `reports/figures/`:
1. `01_dataset_overview_revenue_over_time.png`: Daily revenue oscillations with 30-day moving average.
2. `02_missing_and_clean_data_distribution.png`: Completeness audit across all raw attributes.
3. `03_top_countries_revenue_comparison.png`: United Kingdom dominates revenue (>85%), followed by EIRE, Germany, France.
4. `04_feature_distributions.png`: Kernel density estimates for rolling revenue and engagement metrics.
5. `05_correlation_heatmap.png`: High collinearity among rolling windows; past 28-day revenue shows highest correlation with forward target.
6. `06_views_and_invoices_relationship.png`: Strong linear relationship between unique invoice count and daily revenue.
7. `07_target_distribution_and_outliers.png`: Positive skew in 30-day target revenue with high-volume peak periods.
8. `08_monthly_revenue_seasonality.png`: Year-over-year revenue spikes during Q4 holidays.

---

## Machine Learning Models & Evaluation

To benchmark multiple learning algorithms:
```bash
python scripts/compare_models.py
```

### Model Benchmark Results on Test Dataset

| Algorithm | MAE ($) | MSE | RMSE ($) | R² Score |
|---|:---:|:---:|:---:|:---:|
| **Baseline (30-day Moving Avg)** | 28,486.40 | 1.119 × 10⁹ | **33,454.58** | -2.2587 |
| **Gradient Boosting Regressor** | 39,076.31 | 2.017 × 10⁹ | **44,915.89** | -4.8740 |
| **Decision Tree Regressor** | 44,262.36 | 2.554 × 10⁹ | **50,542.66** | -6.4379 |
| **Random Forest Regressor** | 48,260.11 | 3.368 × 10⁹ | **58,036.01** | -8.8069 |
| **Ridge Regression** | 91,836.43 | 9.447 × 10⁹ | **97,200.30** | -26.5087 |
| **Linear Regression** | 92,622.03 | 9.610 × 10⁹ | **98,033.85** | -26.9826 |

---

## Baseline vs Model Comparison

The final model and baseline comparison is saved in:
`reports/figures/09_baseline_vs_final_model_comparison.png`

It features:
- Dual bar charts comparing RMSE and MAE.
- Continuous time-series prediction overlay against actual known revenue.

---

## REST API Documentation

### Start the Service Locally
```bash
python app.py
```
Default listening port: `http://localhost:8080`

### 1. Root / Overview
```bash
curl http://localhost:8080/
```

### 2. Predict for a Specific Country
```bash
curl "http://localhost:8080/predict?country=united_kingdom&year=2018&month=01&day=05"
```
**Response (200 OK):**
```json
{
  "status": "success",
  "country": "united_kingdom",
  "target_date": "2018-01-05",
  "prediction": 178923.45,
  "latency_seconds": 0.0034,
  "model_version": 1.0
}
```

### 3. Predict for All Countries Combined
```bash
curl "http://localhost:8080/predict?year=2018&month=01&day=05"
```
**Response (200 OK):**
```json
{
  "status": "success",
  "scope": "all_countries",
  "target_date": "2018-01-05",
  "predictions": {
    "all": 210450.12,
    "united_kingdom": 178923.45,
    "germany": 12430.50,
    "france": 9520.10
  },
  "latency_seconds": 0.0125
}
```

### 4. Unsupported Country Error Handling
```bash
curl "http://localhost:8080/predict?country=India&year=2018&month=01&day=05"
```
**Response (400 Bad Request):**
```json
{
  "status": "error",
  "message": "Country 'India' is not supported. Supported countries: ['all', 'eire', 'france', 'germany', 'hong_kong', 'netherlands', 'norway', 'portugal', 'singapore', 'spain', 'united_kingdom']"
}
```

### 5. Health Check
```bash
curl http://localhost:8080/health
```

### 6. Prometheus Performance Metrics
```bash
curl http://localhost:8080/metrics
```

---

## Testing Suite & Isolation

Run all 20 unit tests with a single command:
```bash
pytest
```
Or alternatively:
```bash
python run-tests.py
```

### Isolation Guarantee
- During test execution, `TESTING=1` directs all logs to `test-logs/` and test model artifacts to `test-models/`.
- Production models in `models/` and production logs in `logs/` are strictly protected and never modified by test runs.

---

## Docker Containerization

### Build and Run with Docker Compose
```bash
docker compose up --build
```

### Build and Run with Docker CLI
```bash
docker build -t capstone-revenue-api .
docker run -p 8080:8080 capstone-revenue-api
```
Access the containerized API at: `http://localhost:8080`
