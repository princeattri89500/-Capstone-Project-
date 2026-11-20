#!/usr/bin/env python
"""
cslib.py
Core utility library for data ingestion, cleaning, time-series conversion,
and feature engineering for the AI Workflow Capstone project.
"""

import os
import sys
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd


def fetch_data(data_dir):
    """
    Load all JSON formatted invoice files in data_dir into a single consolidated, cleaned DataFrame.
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"Specified data directory does not exist: {data_dir}")

    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")]
    if len(file_list) == 0:
        raise ValueError(f"Specified data directory does not contain any JSON files: {data_dir}")

    correct_columns = sorted([
        'country', 'customer_id', 'day', 'invoice', 'month',
        'price', 'stream_id', 'times_viewed', 'year'
    ])

    all_dfs = []
    for file_name in file_list:
        try:
            df = pd.read_json(file_name)
        except Exception as e:
            print(f"Warning: Failed to read {file_name}: {e}")
            continue

        cols = df.columns.tolist()
        rename_map = {}
        for c in cols:
            cl = c.lower().strip()
            if cl in ['streamid', 'stream_id']:
                rename_map[c] = 'stream_id'
            elif cl in ['timesviewed', 'times_viewed']:
                rename_map[c] = 'times_viewed'
            elif cl in ['total_price', 'price']:
                rename_map[c] = 'price'
            elif cl == 'country':
                rename_map[c] = 'country'
            elif cl in ['customer_id', 'customerid']:
                rename_map[c] = 'customer_id'
            elif cl == 'invoice':
                rename_map[c] = 'invoice'
            elif cl == 'year':
                rename_map[c] = 'year'
            elif cl == 'month':
                rename_map[c] = 'month'
            elif cl == 'day':
                rename_map[c] = 'day'

        df.rename(columns=rename_map, inplace=True)

        for col in correct_columns:
            if col not in df.columns:
                df[col] = np.nan

        df = df[correct_columns]
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No valid data frames could be loaded from files.")

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Clean missing / invalid records
    df_all['price'] = pd.to_numeric(df_all['price'], errors='coerce').fillna(0.0)
    df_all['times_viewed'] = pd.to_numeric(df_all['times_viewed'], errors='coerce').fillna(0)
    df_all['country'] = df_all['country'].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    # Clean invoice IDs: remove non-numeric chars
    df_all['invoice'] = df_all['invoice'].astype(str).apply(lambda x: re.sub(r"\D+", "", x))
    df_all['invoice'] = df_all['invoice'].replace('', '0')

    # Construct datetime column
    years = pd.to_numeric(df_all['year'], errors='coerce').fillna(2018).astype(int)
    months = pd.to_numeric(df_all['month'], errors='coerce').fillna(1).astype(int)
    days = pd.to_numeric(df_all['day'], errors='coerce').fillna(1).astype(int)

    dates = [
        f"{y}-{str(m).zfill(2)}-{str(d).zfill(2)}"
        for y, m, d in zip(years, months, days)
    ]
    df_all['invoice_date'] = pd.to_datetime(dates, errors='coerce')
    df_all.dropna(subset=['invoice_date'], inplace=True)

    df_all.sort_values(by='invoice_date', inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    return df_all


def convert_to_ts(df_orig, country=None):
    """
    Given the consolidated DataFrame from fetch_data(),
    aggregate by calendar day into a continuous daily time-series DataFrame.
    """
    if country:
        country_norm = str(country).strip().lower().replace(" ", "_")
        avail_countries = set(df_orig['country'].unique())
        if country_norm not in avail_countries and country not in avail_countries:
            raise ValueError(f"Country '{country}' not found in dataset")
        mask = (df_orig['country'] == country_norm) | (df_orig['country'] == country)
        df = df_orig[mask]
    else:
        df = df_orig

    if df.empty:
        raise ValueError("Cannot convert empty DataFrame to time-series")

    # Full continuous daily date range
    min_date = df['invoice_date'].min().floor('D')
    max_date = df['invoice_date'].max().floor('D')
    days = pd.date_range(start=min_date, end=max_date, freq='D')

    df_agg = df.groupby(df['invoice_date'].dt.floor('D')).agg(
        purchases=('price', 'count'),
        unique_invoices=('invoice', 'nunique'),
        unique_streams=('stream_id', 'nunique'),
        total_views=('times_viewed', 'sum'),
        revenue=('price', 'sum')
    ).reindex(days, fill_value=0)

    df_agg.reset_index(names='date', inplace=True)
    df_agg['year_month'] = df_agg['date'].dt.strftime('%Y-%m')
    df_agg['date'] = df_agg['date'].dt.strftime('%Y-%m-%d')
    return df_agg


def fetch_ts(data_dir, clean=False):
    """
    Convenience function to load or generate daily aggregated time-series datasets
    for 'all' countries combined and the top 10 individual countries.
    """
    ts_data_dir = os.path.join(data_dir, "ts-data")

    if clean and os.path.exists(ts_data_dir):
        shutil.rmtree(ts_data_dir)

    os.makedirs(ts_data_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(ts_data_dir) if f.endswith(".csv")]
    if len(existing_files) > 0 and not clean:
        print("... loading ts data from existing cache")
        dfs = {}
        for cf in existing_files:
            key = re.sub(r"^ts-", "", cf)
            key = re.sub(r"\.csv$", "", key)
            dfs[key] = pd.read_csv(os.path.join(ts_data_dir, cf))
        return dfs

    print("... processing raw data into time-series")
    df = fetch_data(data_dir)

    # Identify top 10 countries by total revenue
    rev_by_country = df.groupby('country')['price'].sum().sort_values(ascending=False)
    top_ten_countries = rev_by_country.head(10).index.tolist()

    dfs = {}
    dfs['all'] = convert_to_ts(df)
    dfs['all'].to_csv(os.path.join(ts_data_dir, "ts-all.csv"), index=False)

    for country in top_ten_countries:
        country_id = re.sub(r"\s+", "_", str(country).lower())
        df_c = convert_to_ts(df, country=country)
        dfs[country_id] = df_c
        df_c.to_csv(os.path.join(ts_data_dir, f"ts-{country_id}.csv"), index=False)

    return dfs


def engineer_features(df, training=True):
    """
    Engineer time-series features for supervised learning.
    For each target day, target 'y' is the cumulative sum of revenue over the NEXT 30 days.
    Features include past revenue sums (7, 14, 28, 70 days), previous year revenue,
    and recent invoice and viewing volume.
    """
    dates_dt = pd.to_datetime(df['date'])
    dates_str = dates_dt.dt.strftime('%Y-%m-%d').values
    revenue = df['revenue'].values.astype(float)
    invoices = df['unique_invoices'].values.astype(float)
    views = df['total_views'].values.astype(float)

    eng_features = defaultdict(list)
    previous_windows = [7, 14, 28, 70]
    n_days = len(dates_dt)
    y = np.zeros(n_days)

    for d in range(n_days):
        curr_date = dates_dt.iloc[d]

        # Backward rolling revenue windows
        for num in previous_windows:
            prev_date = curr_date - pd.Timedelta(days=num)
            mask = (dates_dt >= prev_date) & (dates_dt < curr_date)
            eng_features[f"previous_{num}"].append(float(revenue[mask].sum()) if mask.any() else 0.0)

        # 30-day forward target revenue
        plus_30 = curr_date + pd.Timedelta(days=30)
        mask_fwd = (dates_dt >= curr_date) & (dates_dt < plus_30)
        y[d] = float(revenue[mask_fwd].sum()) if mask_fwd.any() else 0.0

        # Prior year corresponding 30-day revenue
        start_prev_year = curr_date - pd.Timedelta(days=365)
        stop_prev_year = plus_30 - pd.Timedelta(days=365)
        mask_yr = (dates_dt >= start_prev_year) & (dates_dt < stop_prev_year)
        eng_features['previous_year'].append(float(revenue[mask_yr].sum()) if mask_yr.any() else 0.0)

        # Recent 30-day non-revenue activity
        minus_30 = curr_date - pd.Timedelta(days=30)
        mask_act = (dates_dt >= minus_30) & (dates_dt < curr_date)
        eng_features['recent_invoices'].append(float(invoices[mask_act].mean()) if mask_act.any() else 0.0)
        eng_features['recent_views'].append(float(views[mask_act].mean()) if mask_act.any() else 0.0)

    X = pd.DataFrame(eng_features).fillna(0.0)
    non_zero_mask = (X.sum(axis=1) > 0).values
    X = X[non_zero_mask].copy()
    y = y[non_zero_mask]
    out_dates = dates_str[non_zero_mask]
    X.reset_index(drop=True, inplace=True)

    if training:
        if len(X) > 30:
            X = X.iloc[:-30].copy()
            y = y[:-30]
            out_dates = out_dates[:-30]
            X.reset_index(drop=True, inplace=True)

    return X, y, out_dates
