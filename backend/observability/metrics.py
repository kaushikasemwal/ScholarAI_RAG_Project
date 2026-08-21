"""
observability/metrics.py — Prometheus Metrics Collection
=========================================================
Thread-safe metrics collector with counters, histograms, and gauges.
Compatible with Prometheus text exposition format.
"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Counter:
    """Thread-safe counter metric."""
    name: str
    help_text: str
    labels: dict[str, str] = field(default_factory=dict)
    _value: float = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def inc(self, value: float = 1, labels: dict[str, str] | None = None):
        with self._lock:
            self._value += value

    def get(self) -> float:
        with self._lock:
            return self._value


@dataclass
class Histogram:
    """Thread-safe histogram with configurable buckets."""
    name: str
    help_text: str
    buckets: list[float] = field(default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    labels: dict[str, str] = field(default_factory=dict)
    _counts: list[int] = field(default_factory=list, init=False)
    _sum: float = 0
    _count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        self._counts = [0] * len(self.buckets)

    def observe(self, value: float, labels: dict[str, str] | None = None):
        with self._lock:
            self._count += 1
            self._sum += value
            for i, bucket in enumerate(self.buckets):
                if value <= bucket:
                    self._counts[i] += 1

    def get_count(self) -> int:
        with self._lock:
            return self._count

    def get_sum(self) -> float:
        with self._lock:
            return self._sum

    def get_buckets(self) -> list[int]:
        with self._lock:
            return self._counts.copy()


@dataclass
class Gauge:
    """Thread-safe gauge metric."""
    name: str
    help_text: str
    labels: dict[str, str] = field(default_factory=dict)
    _value: float = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def set(self, value: float, labels: dict[str, str] | None = None):
        with self._lock:
            self._value = value

    def inc(self, value: float = 1, labels: dict[str, str] | None = None):
        with self._lock:
            self._value += value

    def dec(self, value: float = 1, labels: dict[str, str] | None = None):
        with self._lock:
            self._value -= value

    def get(self) -> float:
        with self._lock:
            return self._value


class MetricsCollector:
    """
    Centralized metrics registry for Prometheus exposition.
    
    Usage:
        collector = MetricsCollector()
        collector.counter("http_requests_total", "Total HTTP requests").inc(labels={"method": "POST", "endpoint": "/upload"})
        collector.histogram("http_request_duration_seconds", "Request latency").observe(0.123)
        collector.gauge("active_connections", "Active connections").set(5)
        
        # Export for Prometheus
        print(collector.generate_prometheus_output())
    """

    def __init__(self, namespace: str = "scholarai"):
        self.namespace = namespace
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._gauges: dict[str, Gauge] = {}
        self._lock = threading.Lock()

    def _make_key(self, name: str, labels: dict[str, str]) -> str:
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}" if label_str else name

    def counter(self, name: str, help_text: str, labels: dict[str, str] | None = None) -> Counter:
        """Get or create a counter."""
        full_name = f"{self.namespace}_{name}"
        key = self._make_key(full_name, labels or {})
        with self._lock:
            if key not in self._counters:
                self._counters[key] = Counter(full_name, help_text, labels or {})
            return self._counters[key]

    def histogram(self, name: str, help_text: str, buckets: list[float] | None = None, labels: dict[str, str] | None = None) -> Histogram:
        """Get or create a histogram."""
        full_name = f"{self.namespace}_{name}"
        key = self._make_key(full_name, labels or {})
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = Histogram(full_name, help_text, buckets or [], labels or {})
            return self._histograms[key]

    def gauge(self, name: str, help_text: str, labels: dict[str, str] | None = None) -> Gauge:
        """Get or create a gauge."""
        full_name = f"{self.namespace}_{name}"
        key = self._make_key(full_name, labels or {})
        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = Gauge(full_name, help_text, labels or {})
            return self._gauges[key]

    def generate_prometheus_output(self) -> str:
        """Generate Prometheus text format output."""
        lines = []

        # Counters
        for counter in self._counters.values():
            lines.append(f"# HELP {counter.name} {counter.help_text}")
            lines.append(f"# TYPE {counter.name} counter")
            label_str = self._format_labels(counter.labels)
            lines.append(f"{counter.name}{label_str} {counter.get()}")

        # Histograms
        for hist in self._histograms.values():
            lines.append(f"# HELP {hist.name} {hist.help_text}")
            lines.append(f"# TYPE {hist.name} histogram")
            label_str = self._format_labels(hist.labels)

            # Bucket counts
            for i, bucket in enumerate(hist.buckets):
                if hist.labels:
                    label_parts = [f'{k}="{v}"' for k, v in sorted(hist.labels.items())]
                    label_parts.append(f'le="{bucket:g}"')
                    bucket_label = f"{hist.name}{{{','.join(label_parts)}}}"
                else:
                    bucket_label = f"{hist.name}{{le=\"{bucket:g}\"}}"
                lines.append(f"{bucket_label} {hist._counts[i]}")

            # +Inf bucket
            inf_label = self._format_histogram_label(hist, "+Inf")
            lines.append(f"{inf_label} {hist.get_count()}")

            # Sum and count
            lines.append(f"{hist.name}_sum{label_str} {hist.get_sum()}")
            lines.append(f"{hist.name}_count{label_str} {hist.get_count()}")

        # Gauges
        for gauge in self._gauges.values():
            lines.append(f"# HELP {gauge.name} {gauge.help_text}")
            lines.append(f"# TYPE {gauge.name} gauge")
            label_str = self._format_labels(gauge.labels)
            lines.append(f"{gauge.name}{label_str} {gauge.get()}")

        return "\n".join(lines) + "\n"

    def _format_labels(self, labels: dict[str, str]) -> str:
        if not labels:
            return ""
        return "{" + ",".join(f'{k}="{v}"' for k, v in sorted(labels.items())) + "}"

    def _format_histogram_label(self, hist: Histogram, le_value: str) -> str:
        if not hist.labels:
            return f"{hist.name}{{le=\"{le_value}\"}}"
        label_parts = [f'{k}="{v}"' for k, v in sorted(hist.labels.items())]
        label_parts.append(f'le="{le_value}"')
        return f"{hist.name}{{{','.join(label_parts)}}}"


# Global metrics collector
_metrics_collector: MetricsCollector | None = None
_metrics_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        with _metrics_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector()
                # Initialize default metrics so /metrics always returns valid output
                _metrics_collector.counter("http_requests_total", "Total HTTP requests")
                _metrics_collector.histogram("http_request_duration_seconds", "HTTP request latency")
                _metrics_collector.gauge("http_requests_in_flight", "In-flight HTTP requests")
                _metrics_collector.counter("generation_total", "Total content generations")
                _metrics_collector.histogram("generation_duration_seconds", "Generation latency")
                _metrics_collector.histogram("model_load_duration_seconds", "Model loading latency")
                _metrics_collector.counter("file_uploads_total", "Total file uploads")
                _metrics_collector.histogram("file_upload_size_bytes", "Uploaded file size")
    return _metrics_collector


# ─── CONVENIENCE FUNCTIONS ───────────────────────────────────────

# HTTP request metrics
_request_counter = None
_request_duration = None
_request_in_flight = None


def _get_request_metrics():
    global _request_counter, _request_duration, _request_in_flight
    collector = get_metrics_collector()
    if _request_counter is None:
        _request_counter = collector.counter("http_requests_total", "Total HTTP requests")
        _request_duration = collector.histogram("http_request_duration_seconds", "HTTP request latency")
        _request_in_flight = collector.gauge("http_requests_in_flight", "In-flight HTTP requests")
    return _request_counter, _request_duration, _request_in_flight


@contextmanager
def record_request(method: str, endpoint: str, status_code: int = 200):
    """
    Context manager to record HTTP request metrics.
    
    Usage:
        with record_request("POST", "/generate/summary", 200):
            # handle request
    """
    counter, duration, in_flight = _get_request_metrics()
    labels = {"method": method, "endpoint": endpoint, "status": str(status_code)}
    start = time.perf_counter()
    in_flight.inc(labels={"method": method, "endpoint": endpoint})
    try:
        yield
    except Exception:
        counter.inc(labels={**labels, "status": "500"})
        raise
    finally:
        elapsed = time.perf_counter() - start
        duration.observe(elapsed, labels={"method": method, "endpoint": endpoint})
        counter.inc(labels=labels)
        in_flight.dec(labels={"method": method, "endpoint": endpoint})


# Generation metrics
_generation_counter = None
_generation_duration = None


def _get_generation_metrics():
    global _generation_counter, _generation_duration
    collector = get_metrics_collector()
    if _generation_counter is None:
        _generation_counter = collector.counter("generation_total", "Total content generations")
        _generation_duration = collector.histogram("generation_duration_seconds", "Generation latency")
    return _generation_counter, _generation_duration


@contextmanager
def record_generation(generation_type: str, success: bool = True):
    """
    Context manager to record content generation metrics.
    
    Usage:
        with record_generation("summary", True):
            summary = generate_summary(text)
    """
    counter, duration = _get_generation_metrics()
    labels = {"type": generation_type, "success": str(success).lower()}
    start = time.perf_counter()
    try:
        yield
    except Exception:
        counter.inc(labels={**labels, "success": "false"})
        raise
    finally:
        elapsed = time.perf_counter() - start
        duration.observe(elapsed, labels={"type": generation_type})
        counter.inc(labels=labels)


# Model loading metrics
_model_load_duration = None


def _get_model_metrics():
    global _model_load_duration
    collector = get_metrics_collector()
    if _model_load_duration is None:
        _model_load_duration = collector.histogram("model_load_duration_seconds", "Model loading latency")
    return _model_load_duration


@contextmanager
def record_model_load(model_name: str):
    """Context manager to record model loading time."""
    hist = _get_model_metrics()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        hist.observe(elapsed, labels={"model": model_name})


# File upload metrics
_upload_counter = None
_upload_size = None


def _get_upload_metrics():
    global _upload_counter, _upload_size
    collector = get_metrics_collector()
    if _upload_counter is None:
        _upload_counter = collector.counter("file_uploads_total", "Total file uploads")
        _upload_size = collector.histogram("file_upload_size_bytes", "Uploaded file size")
    return _upload_counter, _upload_size


def record_upload(file_type: str, size_bytes: int, success: bool = True):
    """Record a file upload."""
    counter, size_hist = _get_upload_metrics()
    labels = {"type": file_type, "success": str(success).lower()}
    counter.inc(labels=labels)
    if success:
        size_hist.observe(size_bytes, labels={"type": file_type})
