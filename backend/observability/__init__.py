"""
observability/__init__.py — Observability Package
==================================================
Structured logging, metrics, and health checks for ScholarAI.
"""

from .logging import setup_logging, get_logger, StructuredLogger
from .metrics import MetricsCollector, get_metrics_collector, record_request, record_generation
from .health import HealthChecker, get_health_checker

__all__ = [
    "setup_logging",
    "get_logger",
    "StructuredLogger",
    "MetricsCollector",
    "get_metrics_collector",
    "record_request",
    "record_generation",
    "HealthChecker",
    "get_health_checker",
]