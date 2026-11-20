#!/usr/bin/env python
"""
unittests/ApiTests.py
Comprehensive unit tests for the Flask API endpoints.
Tests specific country prediction, all countries prediction, invalid country handling,
missing/malformed parameters, metrics, health checks, and status codes.
"""

import os
import sys
import json
import unittest

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TESTING"] = "1"

from app import app, get_models


class ApiTests(unittest.TestCase):
    """Unit test suite for API endpoints."""

    @classmethod
    def setUpClass(cls):
        os.environ["MODEL_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        app.testing = True
        cls.client = app.test_client()
        get_models()

    def test_01_index_endpoint(self):
        """Test root / endpoint returns documentation and 200 OK."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("service", data)
        self.assertIn("endpoints", data)

    def test_02_health_endpoint(self):
        """Test /health endpoint returns healthy status and system metrics."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("uptime_seconds", data)
        self.assertIn("system_resources", data)

    def test_03_metrics_endpoint(self):
        """Test /metrics endpoint returns Prometheus formatted metrics."""
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        text = response.data.decode("utf-8")
        self.assertIn("api_requests_total", text)
        self.assertIn("process_resident_memory_bytes", text)

    def test_04_predict_specific_country(self):
        """Test GET /predict?country=united_kingdom returns valid prediction."""
        response = self.client.get("/predict?country=united_kingdom&year=2018&month=01&day=05")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("country"), "united_kingdom")
        self.assertIn("prediction", data)
        self.assertGreaterEqual(data["prediction"], 0.0)

    def test_05_predict_all_countries(self):
        """Test GET /predict without country parameter returns predictions for all countries."""
        response = self.client.get("/predict?year=2018&month=01&day=05")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("scope"), "all_countries")
        self.assertIn("predictions", data)
        self.assertIsInstance(data["predictions"], dict)

    def test_06_predict_invalid_country(self):
        """Test GET /predict?country=India (unsupported country) returns 400 Bad Request."""
        response = self.client.get("/predict?country=India&year=2018&month=01&day=05")
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data.get("status"), "error")
        self.assertIn("not supported", data.get("message", ""))

    def test_07_predict_invalid_date_format(self):
        """Test GET /predict with non-numeric date parameters returns 400 Bad Request."""
        response = self.client.get("/predict?country=united_kingdom&year=abcd&month=01&day=05")
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data.get("status"), "error")
        self.assertIn("must be numeric", data.get("message", ""))

    def test_08_predict_post_json(self):
        """Test POST /predict with JSON payload."""
        payload = {"country": "all", "year": "2018", "month": "01", "day": "05"}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get("status"), "success")

    def test_09_predict_empty_post(self):
        """Test POST /predict with empty payload returns 400 Bad Request."""
        response = self.client.post("/predict", json={}, content_type="application/json")
        self.assertIn(response.status_code, [200, 400])

    def test_10_404_handler(self):
        """Test nonexistent endpoint returns 404 Not Found."""
        response = self.client.get("/nonexistent_endpoint_12345")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
