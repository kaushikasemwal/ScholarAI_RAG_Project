"""
test_observability.py — Observability Tests
============================================
Tests for structured logging, metrics, and health checks.
"""

import sys
import os
import json
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# Import observability modules directly (no app dependency)
from backend.observability.metrics import MetricsCollector
from backend.observability.logging import StructuredLogger, set_request_id, get_request_id
from backend.observability.health import (
    HealthChecker, HealthStatus, check_disk_space, check_memory, HealthCheckResult
)

# Only import app-dependent tests conditionally
try:
    from fastapi.testclient import TestClient
    from backend.app import app
    HAS_FASTAPI = True
    client = TestClient(app)
except ImportError:
    HAS_FASTAPI = False
    client = None


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
class TestHealthEndpoints:
    """Health check endpoint tests."""

    def test_liveness(self):
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Liveness" in data["message"]

    def test_readiness(self):
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Readiness" in data["message"]

    def test_startup(self):
        response = client.get("/health/startup")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Startup" in data["message"]

    def test_full_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
class TestMetricsEndpoint:
    """Prometheus metrics endpoint tests."""

    def test_metrics_endpoint_exists(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")

    def test_metrics_format(self):
        response = client.get("/metrics")
        text = response.text
        # Should contain Prometheus format markers
        assert "# HELP" in text
        assert "# TYPE" in text

    def test_metrics_contains_http_requests(self):
        # Make a request first
        client.get("/health/live")
        response = client.get("/metrics")
        text = response.text
        # Should have http_requests_total counter
        assert "scholarai_http_requests_total" in text


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
class TestRequestIDMiddleware:
    """Request ID middleware tests."""

    def test_request_id_header_returned(self):
        response = client.get("/health/live", headers={"X-Request-ID": "test-123"})
        assert response.status_code == 200
        assert response.headers.get("x-request-id") == "test-123"

    def test_request_id_generated_if_missing(self):
        response = client.get("/health/live")
        assert response.status_code == 200
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0


class TestStructuredLogging:
    """Structured logging tests."""

    def test_logger_output_format(self):
        from backend.observability.logging import StructuredLogger
        logger = StructuredLogger("test")
        # Just verify it doesn't crash
        logger.info("test message", extra_field="value")


class TestMetricsCollection:
    """Metrics collection tests."""

    def test_counter_increment(self):
        from backend.observability.metrics import MetricsCollector
        collector = MetricsCollector(namespace="test")
        counter = collector.counter("test_counter", "Test counter")
        counter.inc()
        counter.inc(5)
        assert counter.get() == 6

    def test_histogram_observe(self):
        from backend.observability.metrics import MetricsCollector
        collector = MetricsCollector(namespace="test")
        hist = collector.histogram("test_hist", "Test histogram", buckets=[0.1, 0.5, 1.0])
        hist.observe(0.05)
        hist.observe(0.3)
        hist.observe(0.8)
        assert hist.get_count() == 3
        buckets = hist.get_buckets()
        # 0.05 <= 0.1, 0.3 <= 0.5, 0.8 <= 1.0
        assert buckets[0] >= 1  # 0.1 bucket
        assert buckets[1] >= 1  # 0.5 bucket
        assert buckets[2] == 3  # 1.0 bucket (all)

    def test_gauge_set(self):
        from backend.observability.metrics import MetricsCollector
        collector = MetricsCollector(namespace="test")
        gauge = collector.gauge("test_gauge", "Test gauge")
        gauge.set(10)
        assert gauge.get() == 10
        gauge.inc(5)
        assert gauge.get() == 15
        gauge.dec(3)
        assert gauge.get() == 12

    def test_prometheus_output_format(self):
        from backend.observability.metrics import MetricsCollector
        collector = MetricsCollector(namespace="test")
        counter = collector.counter("requests_total", "Total requests", labels={"method": "GET"})
        counter.inc()
        output = collector.generate_prometheus_output()
        assert "test_requests_total" in output
        assert 'method="GET"' in output
        assert "# TYPE test_requests_total counter" in output


class TestHealthChecker:
    """Health checker tests."""

    def test_health_checker_creation(self):
        from backend.observability.health import HealthChecker, HealthStatus
        checker = HealthChecker()
        report = checker.liveness()
        assert report.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert isinstance(report.checks, list)

    def test_disk_space_check(self):
        from backend.observability.health import check_disk_space, HealthStatus
        result = check_disk_space("/")
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert result.name == "disk_space"

    def test_memory_check(self):
        from backend.observability.health import check_memory, HealthStatus
        result = check_memory()
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert result.name == "memory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])