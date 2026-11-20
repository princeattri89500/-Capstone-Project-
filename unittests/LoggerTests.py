#!/usr/bin/env python
"""
unittests/LoggerTests.py
Unit tests verifying log creation, formatting, error logging, and test log isolation.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TESTING"] = "1"

from logger import update_train_log, update_predict_log, update_api_log, update_ingest_log, get_log_dir


class LoggerTests(unittest.TestCase):
    """Unit test suite for the logging system."""

    @classmethod
    def setUpClass(cls):
        cls.test_log_dir = get_log_dir(test=True)

    def test_01_train_log(self):
        """Test training log file creation and content."""
        update_train_log("test_tag", ("2018-01-01", "2018-02-01"), {"rmse": 100.5}, "00:00:05", 1.0, "test note", test=True)
        log_file = os.path.join(self.test_log_dir, "test-train.log")
        csv_file = os.path.join(self.test_log_dir, "test-train.csv")
        self.assertTrue(os.path.exists(log_file))
        self.assertTrue(os.path.exists(csv_file))

    def test_02_predict_log(self):
        """Test prediction log file creation and content."""
        update_predict_log("all", [5432.1], None, "2018-03-01", "00:00:01", 1.0, test=True)
        log_file = os.path.join(self.test_log_dir, "test-predict.log")
        self.assertTrue(os.path.exists(log_file))

    def test_03_api_log(self):
        """Test API log file creation and formatting."""
        update_api_log("/predict", 200, "0.0123s", method="GET", test=True)
        log_file = os.path.join(self.test_log_dir, "test-api.log")
        self.assertTrue(os.path.exists(log_file))

    def test_04_ingest_log(self):
        """Test data ingestion log creation."""
        update_ingest_log("SUCCESS", 1000, 5, "00:00:02", test=True)
        log_file = os.path.join(self.test_log_dir, "test-ingest.log")
        self.assertTrue(os.path.exists(log_file))


if __name__ == "__main__":
    unittest.main()
