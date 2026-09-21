"""
Shared test configuration.

Tests marked `needs_ml` load real models (embeddings, FAISS, EasyOCR, or the orchestrator that
imports them). CI installs requirements-ci.txt, which leaves that runtime out, so there they are
skipped - visibly, with a reason in the summary - while everything else (routing rules, web-search
parsing, action security, workspace confinement, tracing, evaluation metrics) still gates the merge.
In the backend container the runtime is present and nothing is skipped.
"""
import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ML_MODULES = ("torch", "sentence_transformers", "faiss", "easyocr")
MISSING_ML = [m for m in ML_MODULES if importlib.util.find_spec(m) is None]


def pytest_collection_modifyitems(config, items):
    if not MISSING_ML:
        return
    skip = pytest.mark.skip(reason=f"needs the ML runtime (missing: {', '.join(MISSING_ML)}); runs in the backend container")
    for item in items:
        if "needs_ml" in item.keywords:
            item.add_marker(skip)


def pytest_report_header(config):
    return f"ML runtime: {'present - nothing skipped for it' if not MISSING_ML else 'ABSENT (' + ', '.join(MISSING_ML) + ') - needs_ml tests will be skipped'}"
