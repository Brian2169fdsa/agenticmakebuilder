"""
Structured Logger

Thin wrapper around Python logging for structured event output.
Used by the execution poller and monitor loop.
"""

import json
import logging
import sys
from datetime import datetime, timezone

_logger = logging.getLogger("agenticmakebuilder")
if not _logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)

_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def log(event: str, level: str = "info", **kwargs):
    """
    Emit a structured log line as JSON.

    Args:
        event:  Dot-separated event name (e.g. "monitor.poll_complete")
        level:  Log level string (debug, info, warning, error, critical)
        **kwargs: Additional key-value data to include
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "level": level,
    }
    entry.update(kwargs)
    py_level = _LEVEL_MAP.get(level, logging.INFO)
    _logger.log(py_level, json.dumps(entry, default=str))


if __name__ == "__main__":
    print("=== Logger Self-Check ===\n")

    print("Test 1: Basic log output")
    log("test.basic", level="info", message="hello")
    print("  [OK]")

    print("Test 2: Warning level")
    log("test.warning", level="warning", count=5)
    print("  [OK]")

    print("Test 3: Error with extra data")
    log("test.error", level="error", slug="demo", error="something broke")
    print("  [OK]")

    print("\n=== All logger checks passed ===")
