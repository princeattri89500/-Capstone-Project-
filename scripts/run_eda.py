#!/usr/bin/env python
"""
scripts/run_eda.py
Automated Exploratory Data Analysis (EDA) script.
Generates comprehensive visualizations across dataset overviews, missing values,
country breakdowns, feature distributions, correlation heatmaps, and target variables.
Saves all figures into reports/figures/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cslib import fetch_data, fetch_ts, engineer_features


def run_eda(data_dir="cs-train", output_dir="reports/figures"):
    """
    Generate all EDA figures based on actual project data.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print(f"Loading raw and time-series data for EDA from '{data_dir}'...")
    raw_df = fetch_data(data_dir)
    ts_data = fetch_ts(data_dir, clean=False)
    df_all = ts_data['all']
    X, y, dates = engineer_features(df_all, training=False)
    X['target_revenue_30d'] = y

    # 1. Dataset Overview: Daily Revenue & 30-Day Moving Average
    plt.figure(figsize=(12, 5))
    df_plot = df_all.copy()
    df_plot['rolling_30'] = df_plot['revenue'].rolling(30, min_periods=1).mean()
    plt.plot(pd.to_datetime(df_plot['date']), df_plot['revenue'], label="Daily Revenue ($)", alpha=0.5, color="#3498db")
    plt.plot(pd.to_datetime(df_plot['date']), df_plot['rolling_30'], label="30-Day Rolling Mean ($)", color="#e74c3c", linewidth=2)
    plt.title("Aavail Daily Revenue Trends Over Time", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Revenue ($)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_dataset_overview_revenue_over_time.png"), dpi=200)
    plt.close()

    # 2. Missing-Value and Transaction Audit
    plt.figure(figsize=(10, 4.5))
    cols_to_check = ['country', 'customer_id', 'day', 'invoice', 'month', 'price', 'stream_id', 'times_viewed', 'year']
    missing_rates = [raw_df[c].isna().mean() * 100 for c in cols_to_check]
    bars = plt.bar(cols_to_check, missing_rates, color="#2ecc71")
    plt.title("Data Completeness Audit: Missing Value Percentages", fontsize=13, fontweight="bold")
    plt.ylabel("Missing (%)", fontsize=11)
    plt.ylim(0, 10)
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, f"{b.get_height():.1f}%", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_missing_and_clean_data_distribution.png"), dpi=200)
    plt.close()

    # 3. Country-wise Revenue Comparison
    plt.figure(figsize=(12, 6))
    top_countries = raw_df.groupby('country')['price'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_countries.values, y=top_countries.index, palette="viridis")
    plt.title("Top 10 Countries by Total Revenue Contribution", fontsize=14, fontweight="bold")
    plt.xlabel("Total Revenue ($)", fontsize=11)
    plt.ylabel("Country", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_top_countries_revenue_comparison.png"), dpi=200)
    plt.close()

    # 4. Feature Distributions (Histograms & KDE)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(X['previous_7'], kde=True, ax=axes[0, 0], color="#9b59b6")
    axes[0, 0].set_title("Previous 7-Day Revenue Distribution", fontweight="bold")
    sns.histplot(X['previous_28'], kde=True, ax=axes[0, 1], color="#3498db")
    axes[0, 1].set_title("Previous 28-Day Revenue Distribution", fontweight="bold")
    sns.histplot(X['recent_invoices'], kde=True, ax=axes[1, 0], color="#e67e22")
    axes[1, 0].set_title("Recent Invoices (Daily Mean)", fontweight="bold")
    sns.histplot(X['recent_views'], kde=True, ax=axes[1, 1], color="#1abc9c")
    axes[1, 1].set_title("Recent Views (Daily Mean)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_feature_distributions.png"), dpi=200)
    plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, cbar_kws={'label': 'Pearson Correlation'})
    plt.title("Feature & Target Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_correlation_heatmap.png"), dpi=200)
    plt.close()

    # 6. Views and Invoices Relationship
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df_all, x="unique_invoices", y="revenue", scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
    plt.title("Daily Unique Invoices vs Revenue Relationship", fontsize=13, fontweight="bold")
    plt.xlabel("Unique Invoices Per Day", fontsize=11)
    plt.ylabel("Daily Revenue ($)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_views_and_invoices_relationship.png"), dpi=200)
    plt.close()

    # 7. Target Variable Distribution & Outliers
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(y, kde=True, ax=axes[0], color="#e74c3c")
    axes[0].set_title("Distribution of 30-Day Forward Revenue (Target)", fontweight="bold")
    axes[0].set_xlabel("30-Day Forward Revenue ($)")
    sns.boxplot(y=y, ax=axes[1], color="#f39c12")
    axes[1].set_title("Boxplot: 30-Day Target Revenue Outliers", fontweight="bold")
    axes[1].set_ylabel("30-Day Forward Revenue ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_target_distribution_and_outliers.png"), dpi=200)
    plt.close()

    # 8. Monthly Revenue Seasonality
    plt.figure(figsize=(12, 5))
    monthly_rev = df_all.groupby('year_month')['revenue'].sum().reset_index()
    sns.barplot(data=monthly_rev, x="year_month", y="revenue", color="#2980b9")
    plt.xticks(rotation=45, ha='right')
    plt.title("Aggregated Monthly Revenue Seasonality", fontsize=14, fontweight="bold")
    plt.xlabel("Year-Month", fontsize=11)
    plt.ylabel("Total Revenue ($)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_monthly_revenue_seasonality.png"), dpi=200)
    plt.close()

    print(f"All 8 EDA figures successfully saved to '{output_dir}'.")


if __name__ == "__main__":
    run_eda()
