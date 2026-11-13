"""Shared API infrastructure: security, lifecycle and checkpoints."""

import os
from contextlib import asynccontextmanager
from secrets import compare_digest
from threading import Lock

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from langgraph.checkpoint.postgres import PostgresSaver

THREAD_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided_key: str | None = Depends(api_key_header)) -> None:
    expected = os.environ["AGENTIC_API_KEY"]
    if provided_key is None or not compare_digest(provided_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def make_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 60,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    key = os.environ.get("AGENTIC_API_KEY", "")
    if len(key) < 32:
        raise RuntimeError("AGENTIC_API_KEY must contain at least 32 characters")

    with PostgresSaver.from_conn_string(os.environ["DATABASE_URL"]) as saver:
        saver.setup()
        app.state.saver = saver
        app.state.workflow_lock = Lock()
        yield
