import json
import os
import random
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Gauge, Histogram, generate_latest

# -----------------------------
# App & Service Identity
# -----------------------------
SERVICE_NAME = os.getenv("SERVICE_NAME", "simulator")
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv(
    "OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4318/v1/traces"
)

app = FastAPI(title=f"Simulator Service ({SERVICE_NAME})")

# -----------------------------
# OpenTelemetry (traces -> Jaeger via OTLP HTTP)
# -----------------------------
resource = Resource.create({"service.name": SERVICE_NAME})
provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)
FastAPIInstrumentor.instrument_app(app)

# -----------------------------
# Prometheus metrics
# -----------------------------
REQUEST_COUNT = Counter(
    "app_requests_total", "Total de peticiones", ["service", "endpoint"]
)
ERROR_COUNT = Counter("app_errors_total", "Errores simulados", ["service", "endpoint"])
LATENCY = Histogram(
    "app_request_latency_seconds",
    "Latencia de solicitudes",
    ["service", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0, 5.0),
)
FAULT_ACTIVE = Gauge(
    "app_fault_active",
    "Fault activo por tipo (1=activo, 0=inactivo)",
    ["service", "fault_type"],
)


# -----------------------------
# Structured logs (JSON to stdout)
# -----------------------------
def log_json(level: str, message: str, **fields) -> None:
    payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "service": SERVICE_NAME,
        "level": level,
        "message": message,
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


# -----------------------------
# Fault state (reproducible failures)
# -----------------------------
class FaultState:
    """
    Mantiene faults activos con expiración (end_time).
    Se aplica a /simulate y se refleja en métricas y logs.
    """

    def __init__(self) -> None:
        self.extra_latency_s: float = 0.0
        self.error_rate: float = 0.1  # base (normal)
        self.cpu_spike: bool = False
        self.memory_leak: bool = False
        self._leak: list[bytes] = []
        self.end_time: Optional[float] = None
        self.active_faults: set[str] = set()

    def clear(self) -> None:
        self.extra_latency_s = 0.0
        self.error_rate = 0.1
        self.cpu_spike = False
        self.memory_leak = False
        self._leak = []
        self.end_time = None
        self.active_faults.clear()
        self._update_fault_gauges()

    def set_fault(self, fault_type: str, duration_s: int) -> None:
        if duration_s <= 0:
            raise ValueError("duration_s must be > 0")
        self.end_time = time.time() + duration_s
        self.active_faults.add(fault_type)
        self._update_fault_gauges()

    def is_expired(self) -> bool:
        return self.end_time is not None and time.time() > self.end_time

    def expire_if_needed(self) -> None:
        if self.is_expired():
            # Log and clear only the fault-related knobs, restore "normal"
            log_json(
                "INFO", "fault_end", active_faults=sorted(list(self.active_faults))
            )
            self.clear()

    def _update_fault_gauges(self) -> None:
        # Set all known faults to 0, then mark active ones as 1
        known = ["latency", "errors", "cpu", "memory"]
        for f in known:
            FAULT_ACTIVE.labels(service=SERVICE_NAME, fault_type=f).set(
                1.0 if f in self.active_faults else 0.0
            )


FAULT = FaultState()


def burn_cpu(duration_s: int) -> None:
    """
    Simula un spike de CPU por duration_s segundos con un loop "busy".
    """
    end = time.time() + duration_s
    x = 0.0
    while time.time() < end:
        x = (x + 1.2345) * 0.9999  # operación dummy
    _ = x


def leak_memory(mb_per_sec: int, duration_s: int) -> None:
    """
    Simula un memory leak (controlado): agrega bytes a una lista.
    """
    total_chunks = max(1, duration_s)
    bytes_per_sec = max(1, mb_per_sec) * 1024 * 1024
    chunk = b"x" * bytes_per_sec
    for _ in range(total_chunks):
        FAULT._leak.append(chunk)
        time.sleep(1)


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/simulate")
def simulate_request():
    """
    Genera comportamiento normal y con fallos:
    - latencia extra controlada
    - tasa de error controlada
    - (opcional) spike CPU o memory leak programados desde /fault/*
    """
    FAULT.expire_if_needed()

    endpoint = "/simulate"
    REQUEST_COUNT.labels(service=SERVICE_NAME, endpoint=endpoint).inc()

    base_latency = random.uniform(0.05, 0.5)
    latency = base_latency + FAULT.extra_latency_s

    with LATENCY.labels(service=SERVICE_NAME, endpoint=endpoint).time():
        # Latency simulation
        time.sleep(latency)

        # Error simulation
        if random.random() < FAULT.error_rate:
            ERROR_COUNT.labels(service=SERVICE_NAME, endpoint=endpoint).inc()
            log_json(
                "ERROR",
                "request_failed",
                endpoint=endpoint,
                latency=latency,
                base_latency=base_latency,
                error_rate=FAULT.error_rate,
                active_faults=sorted(list(FAULT.active_faults)),
            )
            return {"status": "error", "latency": latency, "service": SERVICE_NAME}

        log_json(
            "INFO",
            "request_ok",
            endpoint=endpoint,
            latency=latency,
            base_latency=base_latency,
            error_rate=FAULT.error_rate,
            active_faults=sorted(list(FAULT.active_faults)),
        )
        return {"status": "ok", "latency": latency, "service": SERVICE_NAME}


@app.get("/metrics")
def metrics():
    return generate_latest()


@app.post("/fault/latency")
def fault_latency(ms: int = 300, duration: int = 60):
    """
    Inyecta latencia adicional (ms) por duration (s).
    Ej: /fault/latency?ms=500&duration=60
    """
    if ms < 0 or duration <= 0:
        raise HTTPException(status_code=400, detail="Invalid ms/duration")
    FAULT.extra_latency_s = ms / 1000.0
    FAULT.set_fault("latency", duration)
    log_json("WARN", "fault_start", fault_type="latency", ms=ms, duration_s=duration)
    return {
        "status": "ok",
        "fault": "latency",
        "ms": ms,
        "duration_s": duration,
        "service": SERVICE_NAME,
    }


@app.post("/fault/errors")
def fault_errors(rate: float = 0.5, duration: int = 60):
    """
    Aumenta tasa de errores (0..1) por duration (s).
    Ej: /fault/errors?rate=0.3&duration=90
    """
    if not (0.0 <= rate <= 1.0) or duration <= 0:
        raise HTTPException(status_code=400, detail="Invalid rate/duration")
    FAULT.error_rate = rate
    FAULT.set_fault("errors", duration)
    log_json("WARN", "fault_start", fault_type="errors", rate=rate, duration_s=duration)
    return {
        "status": "ok",
        "fault": "errors",
        "rate": rate,
        "duration_s": duration,
        "service": SERVICE_NAME,
    }


@app.post("/fault/cpu")
def fault_cpu(duration: int = 30):
    """
    Simula spike de CPU por duration (s).
    Nota: esto bloquea el worker mientras corre (válido para sim).
    """
    if duration <= 0 or duration > 300:
        raise HTTPException(status_code=400, detail="Invalid duration (1..300)")
    FAULT.set_fault("cpu", duration)
    log_json("WARN", "fault_start", fault_type="cpu", duration_s=duration)
    burn_cpu(duration)
    FAULT.expire_if_needed()
    return {
        "status": "ok",
        "fault": "cpu",
        "duration_s": duration,
        "service": SERVICE_NAME,
    }


@app.post("/fault/memory")
def fault_memory(mb_per_sec: int = 5, duration: int = 30):
    """
    Simula memory leak controlado.
    Recomendación: limitar recursos del contenedor.
    """
    if mb_per_sec <= 0 or duration <= 0 or duration > 300:
        raise HTTPException(status_code=400, detail="Invalid mb_per_sec/duration")
    FAULT.set_fault("memory", duration)
    log_json(
        "WARN",
        "fault_start",
        fault_type="memory",
        mb_per_sec=mb_per_sec,
        duration_s=duration,
    )
    leak_memory(mb_per_sec, duration)
    FAULT.expire_if_needed()
    return {
        "status": "ok",
        "fault": "memory",
        "mb_per_sec": mb_per_sec,
        "duration_s": duration,
        "service": SERVICE_NAME,
    }


@app.post("/fault/clear")
def fault_clear():
    """
    Limpia cualquier fault activo y vuelve a parámetros normales.
    """
    if FAULT.active_faults:
        log_json(
            "INFO", "fault_cleared", active_faults=sorted(list(FAULT.active_faults))
        )
    FAULT.clear()
    return {"status": "ok", "service": SERVICE_NAME}
