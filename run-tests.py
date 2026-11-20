#!/usr/bin/env python
"""
run-tests.py
Single command test runner for all project unit tests.
Executes API tests, Model tests, and Logger tests.
"""

import os
import sys
import unittest

os.environ["TESTING"] = "1"

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_modules = [
        "unittests.ApiTests",
        "unittests.ModelTests",
        "unittests.LoggerTests"
    ]

    for mod in test_modules:
        try:
            suite.addTests(loader.loadTestsFromName(mod))
        except Exception as e:
            print(f"Warning: Could not load {mod}: {e}")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
