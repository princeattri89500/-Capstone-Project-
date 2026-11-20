#!/usr/bin/env python
"""
model.py
Machine learning modeling module for the AI Workflow Capstone project.
Handles model training, evaluation, saving, loading, and time-series revenue predictions.
"""

import os
import re
import sys
import time
import uuid
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from logger import update_predict_log, update_train_log
from cslib import fetch_ts, engineer_features

# Base directories and metadata
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
MODEL_VERSION = 1.0
MODEL_VERSION_NOTE = "Production supervised time-series revenue forecasting model"


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _model_train(df, tag, test=False, algorithm="rf", model_dir=None):
    """
    Train a machine learning model on a single country's time-series dataframe.
    """
    time_start = time.time()
    save_dir = model_dir or os.getenv("MODEL_DIR", MODEL_DIR)
    _ensure_dir(save_dir)

    X, y, dates = engineer_features(df, training=True)

    if len(X) == 0:
        raise ValueError(f"Feature matrix for tag '{tag}' is empty. Cannot train.")

    if test:
        # Fast subset for testing purposes
        n_samples = max(5, int(np.round(0.4 * X.shape[0])))
        indices = np.arange(X.shape[0])[:n_samples]
        X = X.iloc[indices].copy()
        y = y[indices]
        dates = dates[indices]

    test_size = 0.25 if len(X) >= 8 else 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    if algorithm == "rf":
        param_grid = {
            'model__n_estimators': [10, 20, 30] if not test else [5],
            'model__max_depth': [5, 10, None] if not test else [3]
        }
        base_estimator = RandomForestRegressor(random_state=42)
    elif algorithm == "gb":
        param_grid = {
            'model__n_estimators': [20, 40] if not test else [5],
            'model__learning_rate': [0.05, 0.1]
        }
        base_estimator = GradientBoostingRegressor(random_state=42)
    elif algorithm == "dt":
        param_grid = {'model__max_depth': [3, 5, 8]}
        base_estimator = DecisionTreeRegressor(random_state=42)
    else:
        param_grid = {'model__alpha': [0.1, 1.0, 10.0]}
        base_estimator = Ridge()

    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', base_estimator)
    ])

    cv_folds = min(3, len(X_train)) if len(X_train) >= 3 else 2
    if cv_folds > 1 and len(param_grid['model__' + list(param_grid.keys())[0].split('__')[-1]]) > 1:
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv_folds, n_jobs=-1)
    else:
        grid = pipe

    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    eval_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred))) if len(y_test) > 0 else 0.0
    eval_mae = float(mean_absolute_error(y_test, y_pred)) if len(y_test) > 0 else 0.0
    eval_r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 1 else 1.0

    # Retrain on full dataset
    grid.fit(X, y)

    model_name = re.sub(r"\.", "_", str(MODEL_VERSION))
    prefix = "test" if test else "sl"
    saved_model_path = os.path.join(save_dir, f"{prefix}-{tag}-{model_name}.joblib")
    joblib.dump(grid, saved_model_path)

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    metrics = {'rmse': round(eval_rmse, 2), 'mae': round(eval_mae, 2), 'r2': round(eval_r2, 4)}
    update_train_log(tag, (str(dates[0]), str(dates[-1])), metrics, runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE, test=test)

    return grid, saved_model_path, metrics


def model_train(data_dir="cs-train", test=False, algorithm="rf", model_dir=None):
    """
    Train time-series revenue models for all supported countries.
    """
    save_dir = model_dir or os.getenv("MODEL_DIR", MODEL_DIR)
    _ensure_dir(save_dir)

    print(f"Loading data from '{data_dir}' (test={test})")
    ts_data = fetch_ts(data_dir, clean=False)

    trained_models = {}
    for country, df in ts_data.items():
        if test and country not in ['all', 'united_kingdom']:
            continue
        print(f"... training model for {country}")
        grid, model_path, metrics = _model_train(df, country, test=test, algorithm=algorithm, model_dir=save_dir)
        trained_models[country] = {"path": model_path, "metrics": metrics}

    return trained_models


def model_load(prefix="sl", data_dir="cs-train", model_dir=None, training=False):
    """
    Load serialized trained models and corresponding feature dictionaries.
    """
    target_model_dir = model_dir or os.getenv("MODEL_DIR", MODEL_DIR)
    if not os.path.isdir(target_model_dir):
        raise FileNotFoundError(f"Model directory '{target_model_dir}' does not exist.")

    model_files = [f for f in os.listdir(target_model_dir) if f.startswith(prefix) and f.endswith(".joblib")]
    if not model_files:
        # Fallback to any joblib model in directory
        model_files = [f for f in os.listdir(target_model_dir) if f.endswith(".joblib")]
        if not model_files:
            raise FileNotFoundError(f"No models with prefix '{prefix}' found in '{target_model_dir}'. Run training first.")

    all_models = {}
    for mf in model_files:
        parts = mf.split("-")
        if len(parts) >= 3:
            country_tag = parts[1]
        else:
            country_tag = parts[0].replace(".joblib", "")
        all_models[country_tag] = joblib.load(os.path.join(target_model_dir, mf))

    # Load time-series data for feature vectors
    ts_data = fetch_ts(data_dir, clean=False)
    all_data = {}
    for country, df in ts_data.items():
        if country in all_models:
            X, y, dates = engineer_features(df, training=training)
            all_data[country] = {"X": X, "y": y, "dates": np.array([str(d) for d in dates])}

    return all_data, all_models


def model_predict(country, year, month, day, all_models=None, all_data=None, data_dir="cs-train", model_dir=None, test=False):
    """
    Generate 30-day forward revenue prediction for a given country and date.
    """
    time_start = time.time()
    prefix = "test" if test else "sl"

    if all_models is None or all_data is None:
        try:
            all_data, all_models = model_load(prefix=prefix, data_dir=data_dir, model_dir=model_dir, training=False)
        except Exception:
            all_data, all_models = model_load(prefix="sl", data_dir=data_dir, model_dir=model_dir, training=False)

    country_norm = str(country).strip().lower().replace(" ", "_")
    if country_norm not in all_models:
        supported = list(all_models.keys())
        raise ValueError(f"Model for country '{country}' not found. Supported countries: {supported}")

    # Validate date numbers
    try:
        y_int, m_int, d_int = int(year), int(month), int(day)
        target_date = f"{y_int}-{str(m_int).zfill(2)}-{str(d_int).zfill(2)}"
    except Exception:
        raise ValueError(f"Invalid date components provided: year={year}, month={month}, day={day}")

    model = all_models[country_norm]
    data = all_data.get(country_norm)

    if data is None or len(data['dates']) == 0:
        raise ValueError(f"No feature matrix data available for country '{country_norm}'")

    # If exact date exists in features, use it; otherwise use closest or most recent feature row
    dates_list = list(data['dates'])
    if target_date in dates_list:
        date_idx = dates_list.index(target_date)
        query = data['X'].iloc[[date_idx]]
    else:
        # Fallback to closest available date
        target_dt = pd.to_datetime(target_date)
        dt_series = pd.to_datetime(data['dates'])
        diffs = (dt_series - target_dt).abs()
        date_idx = int(diffs.argmin())
        query = data['X'].iloc[[date_idx]]

    y_pred = model.predict(query)
    pred_val = float(y_pred[0])
    # Revenue cannot be negative
    pred_val = max(0.0, round(pred_val, 2))

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    update_predict_log(country_norm, [pred_val], None, target_date, runtime, MODEL_VERSION, test=test)

    return {
        "status": "success",
        "country": country_norm,
        "target_date": target_date,
        "prediction": pred_val,
        "y_pred": [pred_val],
        "runtime": runtime,
        "model_version": MODEL_VERSION
    }
