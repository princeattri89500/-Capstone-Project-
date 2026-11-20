#!/usr/bin/env python
"""
app.py
Production REST API for AI Workflow Capstone project.
Provides endpoints for training, revenue prediction (single country and all countries),
performance monitoring (/metrics), health check (/health), and logging.
"""

import os
import sys
import time
import psutil
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, Response

from model import model_train, model_load, model_predict, MODEL_DIR, MODEL_VERSION
from logger import update_api_log, get_log_dir

app = Flask(__name__)
START_TIME = time.time()

# In-memory metrics counters for performance monitoring
REQUEST_COUNTS = {}
REQUEST_LATENCIES = []
ERROR_COUNTS = {}
PREDICTION_LATENCIES = []

# Preload models into memory for low latency
ALL_DATA = None
ALL_MODELS = None


def get_models():
    global ALL_DATA, ALL_MODELS
    if ALL_MODELS is None or len(ALL_MODELS) == 0:
        model_dir = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
        data_dir = os.getenv("DATA_DIR", "cs-train")
        try:
            ALL_DATA, ALL_MODELS = model_load(data_dir=data_dir, model_dir=model_dir, training=False)
        except Exception:
            # Fallback to base models directory if test-models doesn't have them
            base_dir = os.path.join(os.path.dirname(__file__), "models")
            try:
                ALL_DATA, ALL_MODELS = model_load(data_dir=data_dir, model_dir=base_dir, training=False)
            except Exception as e2:
                print(f"Warning: Models could not be loaded: {e2}")
                ALL_DATA, ALL_MODELS = {}, {}
    return ALL_DATA, ALL_MODELS


@app.before_request
def start_timer():
    request.start_time = time.time()


@app.after_request
def record_metrics(response):
    runtime = time.time() - getattr(request, 'start_time', time.time())
    endpoint = request.endpoint or request.path
    status_code = response.status_code

    REQUEST_COUNTS[endpoint] = REQUEST_COUNTS.get(endpoint, 0) + 1
    REQUEST_LATENCIES.append(runtime)
    if len(REQUEST_LATENCIES) > 1000:
        REQUEST_LATENCIES.pop(0)

    if status_code >= 400:
        ERROR_COUNTS[endpoint] = ERROR_COUNTS.get(endpoint, 0) + 1

    runtime_str = f"{runtime:.4f}s"
    test_mode = os.getenv("TESTING", "0") in ["1", "true"]
    update_api_log(request.path, status_code, runtime_str, method=request.method, test=test_mode)
    return response


@app.route("/", methods=["GET"])
def index():
    """API Overview and Documentation"""
    return jsonify({
        "service": "AI Workflow Capstone - Revenue Prediction API",
        "version": str(MODEL_VERSION),
        "status": "online",
        "endpoints": {
            "GET /": "API Overview and Documentation",
            "GET /health": "System health, uptime, and model status",
            "GET /metrics": "Prometheus & performance monitoring metrics",
            "GET /predict": "Predict revenue for a specific country (?country=United+Kingdom) or all countries combined",
            "POST /predict": "Predict revenue via JSON payload",
            "POST /train": "Trigger model retraining (?test=true for test mode)",
            "GET /train": "Trigger model retraining via GET"
        }
    }), 200


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for container orchestrators and monitoring"""
    _, models = get_models()
    uptime_sec = round(time.time() - START_TIME, 2)
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": uptime_sec,
        "models_loaded": len(models),
        "supported_countries": list(models.keys()),
        "system_resources": {
            "memory_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
            "cpu_percent": process.cpu_percent(interval=None)
        }
    }), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus-compatible monitoring endpoint"""
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    cpu_pct = process.cpu_percent(interval=None)
    avg_latency = float(sum(REQUEST_LATENCIES) / len(REQUEST_LATENCIES)) if REQUEST_LATENCIES else 0.0
    p95_latency = float(np.percentile(REQUEST_LATENCIES, 95)) if REQUEST_LATENCIES else 0.0

    lines = [
        "# HELP api_requests_total Total number of HTTP requests received.",
        "# TYPE api_requests_total counter"
    ]
    for ep, count in REQUEST_COUNTS.items():
        lines.append(f'api_requests_total{{endpoint="{ep}"}} {count}')

    lines.extend([
        "# HELP api_errors_total Total number of HTTP error responses.",
        "# TYPE api_errors_total counter"
    ])
    for ep, count in ERROR_COUNTS.items():
        lines.append(f'api_errors_total{{endpoint="{ep}"}} {count}')

    lines.extend([
        "# HELP api_latency_seconds Average request latency in seconds.",
        "# TYPE api_latency_seconds gauge",
        f"api_latency_seconds {avg_latency:.6f}",
        "# HELP api_latency_p95_seconds 95th percentile latency in seconds.",
        "# TYPE api_latency_p95_seconds gauge",
        f"api_latency_p95_seconds {p95_latency:.6f}",
        "# HELP process_resident_memory_bytes Resident memory size in bytes.",
        "# TYPE process_resident_memory_bytes gauge",
        f"process_resident_memory_bytes {mem_bytes}",
        "# HELP process_cpu_percent Process CPU utilization percentage.",
        "# TYPE process_cpu_percent gauge",
        f"process_cpu_percent {cpu_pct}"
    ])

    return Response("\n".join(lines) + "\n", mimetype="text/plain; version=0.0.4")


@app.route("/train", methods=["GET", "POST"])
def train():
    """Retrain models"""
    is_test = request.args.get("test", "false").lower() in ["true", "1"]
    data_dir = os.getenv("DATA_DIR", "cs-train")

    try:
        results = model_train(data_dir=data_dir, test=is_test)
        # Reload models in memory
        global ALL_DATA, ALL_MODELS
        ALL_DATA, ALL_MODELS = model_load(data_dir=data_dir, training=False)

        return jsonify({
            "status": "success",
            "message": "Models trained and loaded successfully",
            "test_mode": is_test,
            "trained_countries": list(results.keys())
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Predict 30-day forward revenue.
    Supports single country (?country=United_Kingdom) and all countries (omitted country or country=all).
    """
    time_pred_start = time.time()
    all_data, all_models = get_models()

    if not all_models:
        return jsonify({
            "status": "error",
            "message": "No models loaded. Please trigger /train first."
        }), 503

    # Extract query params or JSON body
    params = request.get_json(silent=True) or {}
    args = request.args

    country = params.get("country", args.get("country", None))
    year = params.get("year", args.get("year", "2018"))
    month = params.get("month", args.get("month", "01"))
    day = params.get("day", args.get("day", "05"))

    # Empty request validation
    if request.method == "POST" and not params and not args:
        return jsonify({
            "status": "error",
            "message": "Empty request payload provided."
        }), 400

    # Validate date numbers
    for field_name, val in [("year", year), ("month", month), ("day", day)]:
        if not str(val).isdigit():
            return jsonify({
                "status": "error",
                "message": f"Invalid date parameter '{field_name}': must be numeric."
            }), 400

    is_test = os.getenv("TESTING", "0") in ["1", "true"]

    # Case 1: Predict for ALL countries combined or omitted
    if country is None or str(country).strip().lower() in ["all", "combined", ""]:
        predictions = {}
        for c_tag in all_models.keys():
            try:
                res = model_predict(c_tag, year, month, day, all_models=all_models, all_data=all_data, test=is_test)
                predictions[c_tag] = res["prediction"]
            except Exception as e:
                predictions[c_tag] = None

        pred_latency = time.time() - time_pred_start
        PREDICTION_LATENCIES.append(pred_latency)

        return jsonify({
            "status": "success",
            "scope": "all_countries",
            "target_date": f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}",
            "predictions": predictions,
            "latency_seconds": round(pred_latency, 4)
        }), 200

    # Case 2: Predict for a SPECIFIC country
    country_norm = str(country).strip().lower().replace(" ", "_")
    if country_norm not in all_models:
        return jsonify({
            "status": "error",
            "message": f"Country '{country}' is not supported. Supported countries: {list(all_models.keys())}"
        }), 400

    try:
        res = model_predict(country_norm, year, month, day, all_models=all_models, all_data=all_data, test=is_test)
        pred_latency = time.time() - time_pred_start
        PREDICTION_LATENCIES.append(pred_latency)
        res["latency_seconds"] = round(pred_latency, 4)
        return jsonify(res), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
