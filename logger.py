#!/usr/bin/env python
"""
Logging module for AI Workflow Capstone project.
Provides structured logging for training, prediction, API requests, and data ingestion.
Supports full isolation between production and testing environments.
"""

import os
import sys
import time
import csv
import uuid
import logging
from datetime import datetime

# Base directories
DEFAULT_LOG_DIR = os.getenv("LOG_DIR", os.path.join(os.path.dirname(__file__), "logs"))
TEST_LOG_DIR = os.getenv("TEST_LOG_DIR", os.path.join(os.path.dirname(__file__), "test-logs"))


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_log_dir(test=False):
    """
    Return the appropriate log directory based on testing mode or environment.
    """
    is_test_env = os.getenv("TESTING", "0") in ["1", "true", "True"] or os.getenv("CS_ENV") == "test"
    if test or is_test_env:
        _ensure_dir(TEST_LOG_DIR)
        return TEST_LOG_DIR
    _ensure_dir(DEFAULT_LOG_DIR)
    return DEFAULT_LOG_DIR


def get_logger(name="ai_workflow", test=False):
    """
    Get configured standard Python logger writing to a structured log file.
    """
    log_dir = get_log_dir(test=test)
    prefix = "test-" if (test or os.getenv("TESTING") in ["1", "true"]) else ""
    log_file = os.path.join(log_dir, f"{prefix}app.log")

    logger = logging.getLogger(f"{name}_{'test' if test else 'prod'}")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


def update_train_log(tag, dates, eval_test, runtime, model_version, model_version_note, test=False):
    """
    Update training log file with training metadata, date range, evaluation metrics, and runtime.
    """
    log_dir = get_log_dir(test=test)
    prefix = "test-" if test else ""
    log_file = os.path.join(log_dir, f"{prefix}train.log")
    csv_file = os.path.join(log_dir, f"{prefix}train.csv")

    timestamp = datetime.utcnow().isoformat()
    start_date, end_date = dates if isinstance(dates, (tuple, list)) and len(dates) == 2 else ("", "")
    rmse = eval_test.get('rmse', '') if isinstance(eval_test, dict) else eval_test
    mae = eval_test.get('mae', '') if isinstance(eval_test, dict) else ''
    r2 = eval_test.get('r2', '') if isinstance(eval_test, dict) else ''

    # Write human-readable log
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"TIMESTAMP: {timestamp} | TAG: {tag} | START_DATE: {start_date} | END_DATE: {end_date} | "
            f"METRICS: {eval_test} | RUNTIME: {runtime} | VERSION: {model_version} | NOTE: {model_version_note}\n"
        )

    # Write structured CSV
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["unique_id", "timestamp", "tag", "start_date", "end_date", "rmse", "mae", "r2", "runtime", "model_version", "note"])
        writer.writerow([str(uuid.uuid4())[:8], timestamp, tag, start_date, end_date, rmse, mae, r2, runtime, model_version, model_version_note])


def update_predict_log(country, y_pred, y_proba, target_date, runtime, model_version, test=False):
    """
    Update prediction log file with input country, prediction output, timestamp, and latency.
    """
    log_dir = get_log_dir(test=test)
    prefix = "test-" if test else ""
    log_file = os.path.join(log_dir, f"{prefix}predict.log")
    csv_file = os.path.join(log_dir, f"{prefix}predict.csv")

    timestamp = datetime.utcnow().isoformat()
    pred_val = float(y_pred[0]) if hasattr(y_pred, "__len__") and len(y_pred) > 0 else float(y_pred)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"TIMESTAMP: {timestamp} | COUNTRY: {country} | TARGET_DATE: {target_date} | "
            f"PREDICTION: {pred_val:.2f} | PROBA: {y_proba} | RUNTIME: {runtime} | VERSION: {model_version}\n"
        )

    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["unique_id", "timestamp", "country", "target_date", "y_pred", "y_proba", "runtime", "model_version"])
        writer.writerow([str(uuid.uuid4())[:8], timestamp, country, target_date, pred_val, str(y_proba), runtime, model_version])


def update_api_log(endpoint, status_code, runtime, method="GET", error=None, test=False):
    """
    Update API access and error log.
    """
    log_dir = get_log_dir(test=test)
    prefix = "test-" if test else ""
    log_file = os.path.join(log_dir, f"{prefix}api.log")

    timestamp = datetime.utcnow().isoformat()
    status_str = "SUCCESS" if 200 <= status_code < 400 else "ERROR"
    error_msg = f" | ERROR: {error}" if error else ""

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"TIMESTAMP: {timestamp} | METHOD: {method} | ENDPOINT: {endpoint} | "
            f"STATUS: {status_code} ({status_str}) | RUNTIME: {runtime}{error_msg}\n"
        )


def update_ingest_log(status, records_count, files_count, runtime, error=None, test=False):
    """
    Update data ingestion log.
    """
    log_dir = get_log_dir(test=test)
    prefix = "test-" if test else ""
    log_file = os.path.join(log_dir, f"{prefix}ingest.log")

    timestamp = datetime.utcnow().isoformat()
    error_msg = f" | ERROR: {error}" if error else ""

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"TIMESTAMP: {timestamp} | STATUS: {status} | RECORDS: {records_count} | "
            f"FILES: {files_count} | RUNTIME: {runtime}{error_msg}\n"
        )
