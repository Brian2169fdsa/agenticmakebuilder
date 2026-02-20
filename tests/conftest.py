"""
Test configuration — mock database before any app code loads.
"""

import os
import sys
from unittest.mock import MagicMock, patch

# Set dummy DATABASE_URL before any imports
os.environ["DATABASE_URL"] = "sqlite://"

# Pre-mock the psycopg2 import since it may not connect
import importlib

# Create mock engine and session before db.session is imported
_mock_engine = MagicMock()
_mock_session_local = MagicMock()


def _patched_create_engine(*args, **kwargs):
    return _mock_engine


# Patch create_engine before db.session imports it
with patch("sqlalchemy.create_engine", _patched_create_engine):
    # Force-load db.session with our mocked engine
    if "db.session" in sys.modules:
        del sys.modules["db.session"]
    if "db" in sys.modules:
        del sys.modules["db"]

    import db.session
    db.session.engine = _mock_engine
    db.session.SessionLocal = _mock_session_local
    db.session.check_db = lambda: None

# Now import the app — it will use our mocked db.session
if "app" in sys.modules:
    del sys.modules["app"]

import app as app_module
app_module._registry = {"registry_version": "1", "modules": {"http": {}}, "module_count": 1}
