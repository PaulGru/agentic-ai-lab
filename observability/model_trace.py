"""Metadata-only tracing at the agent/provider boundary."""
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from time import perf_counter_ns, time_ns
from uuid import uuid4


_sink = ContextVar("model_trace_sink", default=None)


@contextmanager
def capture_model_trace(sink):
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def prompt_identity(agent, version, instructions):
    return {"prompt_id": agent, "prompt_version": version,
            "instructions_sha256": sha256(instructions.encode("utf-8")).hexdigest()}


def invoke_traced(operation, call, *, prompt, tracer):
    sink = _sink.get()
    if sink is None and tracer is None:
        return call()
    identity = {"call_id": uuid4().hex, "operation": operation, "started_at_ns": time_ns(), **prompt}
    if sink is not None:
        sink({**identity, "status": "started"})
    start = perf_counter_ns()

    def finish(status):
        record = {**identity, "status": status, "latency_ms": (perf_counter_ns() - start) // 1_000_000}
        if sink is not None:
            sink(record)
        if tracer is not None:
            tracer.record("model_call", **record)

    try:
        result = call()
    except Exception:
        finish("error")
        raise
    finish("returned")
    return result
