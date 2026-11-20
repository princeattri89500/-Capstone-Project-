#!/usr/bin/env python
"""
unittests/ModelTests.py
Unit tests for model training, loading, prediction, consistency, and serialization.
Uses isolated test directories to ensure zero contamination of production models.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TESTING"] = "1"
os.environ["MODEL_DIR"] = os.path.join(os.path.dirname(__file__), "..", "test-models")

from model import model_train, model_load, model_predict, _model_train
from cslib import fetch_ts, engineer_features


class ModelTests(unittest.TestCase):
    """Unit test suite for Machine Learning models."""

    @classmethod
    def setUpClass(cls):
        cls.test_model_dir = os.environ["MODEL_DIR"]
        os.makedirs(cls.test_model_dir, exist_ok=True)
        # Train lightweight test models
        cls.trained_models = model_train(data_dir="cs-train", test=True, model_dir=cls.test_model_dir)

    def test_01_model_train_creates_files(self):
        """Test model training generates serialized joblib files."""
        model_files = [f for f in os.listdir(self.test_model_dir) if f.endswith(".joblib")]
        self.assertGreater(len(model_files), 0)

    def test_02_model_load(self):
        """Test model_load successfully deserializes trained models."""
        all_data, all_models = model_load(prefix="test", data_dir="cs-train", model_dir=self.test_model_dir)
        self.assertIn("all", all_models)
        self.assertIn("united_kingdom", all_models)

    def test_03_model_predict_output_structure(self):
        """Test model_predict returns valid positive float prediction."""
        res = model_predict("united_kingdom", "2018", "01", "05", model_dir=self.test_model_dir, test=True)
        self.assertEqual(res["status"], "success")
        self.assertIn("prediction", res)
        self.assertIsInstance(res["prediction"], (int, float))
        self.assertGreaterEqual(res["prediction"], 0.0)

    def test_04_prediction_consistency(self):
        """Test identical inputs generate consistent deterministic predictions."""
        res1 = model_predict("all", "2018", "02", "01", model_dir=self.test_model_dir, test=True)
        res2 = model_predict("all", "2018", "02", "01", model_dir=self.test_model_dir, test=True)
        self.assertEqual(res1["prediction"], res2["prediction"])

    def test_05_invalid_country_error(self):
        """Test predicting for nonexistent country raises ValueError."""
        with self.assertRaises(ValueError):
            model_predict("Atlantis", "2018", "01", "05", model_dir=self.test_model_dir, test=True)

    def test_06_invalid_date_error(self):
        """Test predicting with invalid non-numeric date raises ValueError."""
        with self.assertRaises(ValueError):
            model_predict("all", "year_xyz", "01", "05", model_dir=self.test_model_dir, test=True)


if __name__ == "__main__":
    unittest.main()
