#!/usr/bin/env python
"""
scripts/ingest_data.py
Automated data ingestion script for the AI Workflow Capstone project.
Loads raw JSON files, validates records, handles missing and duplicate values,
aggregates daily time-series by country, saves processed datasets, and logs results.
"""

import os
import sys
import time
import argparse

# Add parent directory to sys.path so cslib and logger can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cslib import fetch_data, convert_to_ts, fetch_ts
from logger import update_ingest_log, get_logger


def run_ingestion(data_dir="cs-train", clean=False, test=False):
    """
    Execute full data ingestion and validation pipeline.
    """
    logger = get_logger("data_ingestion", test=test)
    time_start = time.time()
    logger.info(f"Starting automated data ingestion from: {data_dir} (clean={clean})")

    if not os.path.isdir(data_dir):
        err = f"Data directory '{data_dir}' does not exist"
        logger.error(err)
        update_ingest_log("FAILURE", 0, 0, "00:00:00", error=err, test=test)
        raise FileNotFoundError(err)

    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    files_count = len(json_files)
    if files_count == 0:
        err = f"No JSON invoice files found in '{data_dir}'"
        logger.error(err)
        update_ingest_log("FAILURE", 0, 0, "00:00:00", error=err, test=test)
        raise ValueError(err)

    try:
        # Load and clean consolidated raw records
        df = fetch_data(data_dir)
        records_count = len(df)
        logger.info(f"Successfully cleaned and consolidated {records_count} records across {files_count} files")

        # Create time-series aggregations
        ts_dfs = fetch_ts(data_dir, clean=clean)
        countries_processed = list(ts_dfs.keys())
        logger.info(f"Successfully generated daily time-series datasets for: {countries_processed}")

        m, s = divmod(time.time() - time_start, 60)
        h, m = divmod(m, 60)
        runtime = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

        update_ingest_log("SUCCESS", records_count, files_count, runtime, test=test)
        logger.info(f"Data ingestion completed successfully in {runtime}")

        return {
            "status": "SUCCESS",
            "records_count": records_count,
            "files_count": files_count,
            "countries": countries_processed,
            "runtime": runtime
        }

    except Exception as e:
        m, s = divmod(time.time() - time_start, 60)
        h, m = divmod(m, 60)
        runtime = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
        logger.error(f"Ingestion failed: {e}")
        update_ingest_log("FAILURE", 0, files_count, runtime, error=str(e), test=test)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Data Ingestion Script")
    parser.add_argument("--data-dir", type=str, default="cs-train", help="Path to raw data directory")
    parser.add_argument("--clean", action="store_true", help="Clean and reprocess cached time-series files")
    parser.add_argument("--test", action="store_true", help="Run in test mode (isolated test logs)")
    args = parser.parse_args()

    try:
        results = run_ingestion(data_dir=args.data_dir, clean=args.clean, test=args.test)
        print("\nData Ingestion Summary:")
        for k, v in results.items():
            print(f"  {k}: {v}")
        sys.exit(0)
    except Exception as exc:
        print(f"\nERROR: Ingestion process failed: {exc}", file=sys.stderr)
        sys.exit(1)
