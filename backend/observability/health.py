"""
observability/health.py — Health Checks
========================================
Comprehensive health checks for Kubernetes/container readiness and liveness probes.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import psutil

log = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    details: dict[str, Any] | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


@dataclass
class HealthReport:
    status: HealthStatus
    checks: list[HealthCheckResult]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    version: str = "1.0.0"
    uptime_seconds: float = 0


class HealthChecker:
    """
    Health check registry with support for:
    - Liveness checks (is process alive?)
    - Readiness checks (can serve traffic?)
    - Startup checks (is initialization complete?)
    - Custom dependency checks (DB, models, disk, etc.)
    """

    def __init__(self, app_start_time: float = None):
        self._liveness_checks: list[Callable[[], HealthCheckResult]] = []
        self._readiness_checks: list[Callable[[], HealthCheckResult]] = []
        self._startup_checks: list[Callable[[], HealthCheckResult]] = []
        self._app_start_time = app_start_time or time.time()

    def add_liveness_check(self, name: str, check_fn: Callable[[], HealthCheckResult]):
        """Add a liveness check (process is alive)."""
        self._liveness_checks.append(check_fn)

    def add_readiness_check(self, name: str, check_fn: Callable[[], HealthCheckResult]):
        """Add a readiness check (can serve requests)."""
        self._readiness_checks.append(check_fn)

    def add_startup_check(self, name: str, check_fn: Callable[[], HealthCheckResult]):
        """Add a startup check (initialization complete)."""
        self._startup_checks.append(check_fn)

    def run_checks(self, checks: list[Callable[[], HealthCheckResult]]) -> list[HealthCheckResult]:
        """Run a list of checks and collect results."""
        results = []
        for check_fn in checks:
            start = time.perf_counter()
            try:
                result = check_fn()
                result.duration_ms = (time.perf_counter() - start) * 1000
            except Exception as e:
                result = HealthCheckResult(
                    name=check_fn.__name__,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {e}",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            results.append(result)
        return results

    def liveness(self) -> HealthReport:
        """Run liveness checks - lightweight, should never fail if process is running."""
        checks = self.run_checks(self._liveness_checks)
        overall = HealthStatus.HEALTHY
        for c in checks:
            if c.status == HealthStatus.UNHEALTHY:
                overall = HealthStatus.UNHEALTHY
                break
        return HealthReport(
            status=overall,
            checks=checks,
            uptime_seconds=time.time() - self._app_start_time,
        )

    def readiness(self) -> HealthReport:
        """Run readiness checks - can we serve traffic?"""
        checks = self.run_checks(self._readiness_checks)
        overall = HealthStatus.HEALTHY
        for c in checks:
            if c.status == HealthStatus.UNHEALTHY:
                overall = HealthStatus.UNHEALTHY
                break
            elif c.status == HealthStatus.DEGRADED and overall == HealthStatus.HEALTHY:
                overall = HealthStatus.DEGRADED
        return HealthReport(
            status=overall,
            checks=checks,
            uptime_seconds=time.time() - self._app_start_time,
        )

    def startup(self) -> HealthReport:
        """Run startup checks - is initialization complete?"""
        checks = self.run_checks(self._startup_checks)
        overall = HealthStatus.HEALTHY
        for c in checks:
            if c.status == HealthStatus.UNHEALTHY:
                overall = HealthStatus.UNHEALTHY
                break
        return HealthReport(
            status=overall,
            checks=checks,
            uptime_seconds=time.time() - self._app_start_time,
        )


# ─── BUILT-IN CHECKS ─────────────────────────────────────────────

def check_disk_space(path: str = "/", min_free_gb: float = 1.0) -> HealthCheckResult:
    """Check available disk space."""
    try:
        usage = psutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < min_free_gb:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNHEALTHY,
                message=f"Low disk space: {free_gb:.1f} GB free (minimum: {min_free_gb} GB)",
                duration_ms=0,
                details={"free_gb": free_gb, "min_free_gb": min_free_gb, "path": path}
            )
        elif free_gb < min_free_gb * 2:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.DEGRADED,
                message=f"Disk space warning: {free_gb:.1f} GB free",
                duration_ms=0,
                details={"free_gb": free_gb, "path": path}
            )
        return HealthCheckResult(
            name="disk_space",
            status=HealthStatus.HEALTHY,
            message=f"Disk space OK: {free_gb:.1f} GB free",
            duration_ms=0,
            details={"free_gb": free_gb, "path": path}
        )
    except Exception as e:
        return HealthCheckResult(
            name="disk_space",
            status=HealthStatus.UNHEALTHY,
            message=f"Disk check failed: {e}",
            duration_ms=0,
        )


def check_memory(max_percent: float = 90.0) -> HealthCheckResult:
    """Check memory usage."""
    try:
        mem = psutil.virtual_memory()
        if mem.percent > max_percent:
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"High memory usage: {mem.percent:.1f}% (max: {max_percent}%)",
                duration_ms=0,
                details={"percent": mem.percent, "available_gb": mem.available / (1024**3)}
            )
        elif mem.percent > max_percent * 0.8:
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.DEGRADED,
                message=f"Memory usage warning: {mem.percent:.1f}%",
                duration_ms=0,
                details={"percent": mem.percent}
            )
        return HealthCheckResult(
            name="memory",
            status=HealthStatus.HEALTHY,
            message=f"Memory OK: {mem.percent:.1f}% used",
            duration_ms=0,
            details={"percent": mem.percent, "available_gb": mem.available / (1024**3)}
        )
    except Exception as e:
        return HealthCheckResult(
            name="memory",
            status=HealthStatus.UNHEALTHY,
            message=f"Memory check failed: {e}",
            duration_ms=0,
        )


def check_model_files(models_dir: str = "models") -> HealthCheckResult:
    """Check that required model files exist."""
    try:
        path = Path(models_dir)
        required = ["autoencoder_weights.pt", "encryption.key"]
        missing = [f for f in required if not (path / f).exists()]

        if missing:
            return HealthCheckResult(
                name="model_files",
                status=HealthStatus.DEGRADED,
                message=f"Missing model files: {', '.join(missing)}",
                duration_ms=0,
                details={"missing": missing, "models_dir": str(path)}
            )
        return HealthCheckResult(
            name="model_files",
            status=HealthStatus.HEALTHY,
            message="All model files present",
            duration_ms=0,
            details={"models_dir": str(path)}
        )
    except Exception as e:
        return HealthCheckResult(
            name="model_files",
            status=HealthStatus.UNHEALTHY,
            message=f"Model file check failed: {e}",
            duration_ms=0,
        )


def check_directories(dirs: list[str] = None) -> HealthCheckResult:
    """Check that required directories exist and are writable."""
    dirs = dirs or ["uploads", "outputs", "models"]
    results = []
    for d in dirs:
        path = Path(d)
        try:
            path.mkdir(parents=True, exist_ok=True)
            # Test write
            test_file = path / ".health_check_write_test"
            test_file.write_text("ok")
            test_file.unlink()
            results.append({"dir": d, "status": "ok"})
        except Exception as e:
            results.append({"dir": d, "status": "failed", "error": str(e)})

    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        return HealthCheckResult(
            name="directories",
            status=HealthStatus.UNHEALTHY,
            message=f"Directory check failed: {', '.join(f['dir'] for f in failed)}",
            duration_ms=0,
            details={"results": results}
        )
    return HealthCheckResult(
        name="directories",
        status=HealthStatus.HEALTHY,
        message="All directories accessible",
        duration_ms=0,
        details={"results": results}
    )


# ─── DEFAULT HEALTH CHECKER ──────────────────────────────────────

_default_checker: HealthChecker | None = None
_checker_lock = threading.Lock()


def get_health_checker(app_start_time: float = None) -> HealthChecker:
    """Get or create the default health checker with built-in checks."""
    global _default_checker
    if _default_checker is None:
        with _checker_lock:
            if _default_checker is None:
                _default_checker = HealthChecker(app_start_time)
                # Register built-in checks
                _default_checker.add_liveness_check("disk", lambda: check_disk_space())
                _default_checker.add_liveness_check("memory", lambda: check_memory())
                _default_checker.add_readiness_check("disk", lambda: check_disk_space())
                _default_checker.add_readiness_check("memory", lambda: check_memory())
                _default_checker.add_readiness_check("directories", lambda: check_directories())
                _default_checker.add_readiness_check("model_files", lambda: check_model_files())
                _default_checker.add_startup_check("directories", lambda: check_directories())
                _default_checker.add_startup_check("model_files", lambda: check_model_files())
    return _default_checker
