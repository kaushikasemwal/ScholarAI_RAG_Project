"""
observability/__init__.py — Observability Package
==================================================
Structured logging, metrics, and health checks for ScholarAI.
"""

from .health import HealthChecker, get_health_checker
from .logging import RequestIDMiddleware, StructuredLogger, get_logger, setup_logging
from .metrics import (
    MetricsCollector,
    get_metrics_collector,
    record_generation,
    record_model_load,
    record_request,
    record_upload,
)

__all__ = [
    "setup_logging",
    "get_logger",
    "StructuredLogger",
    "RequestIDMiddleware",
    "MetricsCollector",
    "get_metrics_collector",
    "record_request",
    "record_generation",
    "record_model_load",
    "record_upload",
    "HealthChecker",
    "get_health_checker",
]
