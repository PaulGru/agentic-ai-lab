import os
from contextlib import asynccontextmanager
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException, Path, Request
from pydantic import BaseModel, Field

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from demo_v45 import build_graph

from secrets import compare_digest

from fastapi import Depends
from fastapi.security import APIKeyHeader

# Nous conservons le format de thread_id de la V4.5.
THREAD_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"

# Security - V5.4

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def require_api_key(
    provided_key: str | None = Depends(api_key_header),
) -> None:
    expected_key = os.environ["AGENTIC_API_KEY"]

    if provided_key is None or not compare_digest(
        provided_key, expected_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )


# 1. Modèles des requêtes HTTP

class StartRequest(BaseModel):
    thread_id: str = Field(pattern=THREAD_PATTERN)


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject", "edit"]
    content: str | None = Field(default=None, max_length=10000)


# 2. Cycle de vie de l'application

@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.environ.get("AGENTIC_API_KEY", "")

    if len(api_key) < 32:
        raise RuntimeError(
            "AGENTIC_API_KEY must contain at least 32 characters"
        )

    database_url = os.environ["DATABASE_URL"]

    with PostgresSaver.from_conn_string(database_url) as saver:
        saver.setup()

        app.state.saver = saver
        app.state.workflow_lock = Lock()

        yield


app = FastAPI(
    title="Agentic AI Lab",
    version="5.4",
    lifespan=lifespan,
)


# 3. Fonctions utilitaires

def make_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 60,
    }


def describe_workflow(graph, config: dict, thread_id: str):
    snapshot = graph.get_state(config)

    if not snapshot.values:
        raise HTTPException(404, "Workflow not found")

    approval = (
        snapshot.interrupts[0].value
        if snapshot.interrupts
        else None
    )

    if approval is not None:
        status = "pending_approval"
    elif not snapshot.next:
        status = "completed"
    else:
        status = "in_progress"

    return {
        "thread_id": thread_id,
        "status": status,
        "approval": approval,
        "answer": snapshot.values.get("answer") or None,
    }


# 4. Endpoint de santé

@app.get("/health")
def health():
    return {"status": "ok"}


# 5. Démarrer un nouveau workflow

@app.post("/workflows", status_code=201)
def start_workflow(body: StartRequest, request: Request, _auth: None = Depends(require_api_key)):
    with request.app.state.workflow_lock:
        saver = request.app.state.saver
        config = make_config(body.thread_id)

        graph, _, _ = build_graph(
            saver, body.thread_id, start=True
        )

        if graph.get_state(config).values:
            raise HTTPException(409, "Thread already exists")

        graph.invoke(
            {"prompt": "Calculate 125 * 48 and save it to a note"},
            config=config,
        )

        return describe_workflow(graph, config, body.thread_id)


# 6. Consulter l'état d'un workflow

@app.get("/workflows/{thread_id}")
def get_workflow(
    request: Request,
    thread_id: str = Path(pattern=THREAD_PATTERN),
    _auth: None = Depends(require_api_key),
):
    with request.app.state.workflow_lock:
        saver = request.app.state.saver
        config = make_config(thread_id)

        graph, _, _ = build_graph(
            saver, thread_id, start=False
        )

        return describe_workflow(graph, config, thread_id)


# 7. Approuver, refuser ou modifier une action

@app.post("/workflows/{thread_id}/decision")
def decide_workflow(
    body: DecisionRequest,
    request: Request,
    thread_id: str = Path(pattern=THREAD_PATTERN),
    _auth: None = Depends(require_api_key),
):
    with request.app.state.workflow_lock:
        saver = request.app.state.saver
        config = make_config(thread_id)

        graph, _, _ = build_graph(
            saver, thread_id, start=False
        )

        snapshot = graph.get_state(config)

        if not snapshot.values:
            raise HTTPException(404, "Workflow not found")

        if not snapshot.interrupts:
            raise HTTPException(409, "No pending approval")

        if body.decision == "edit":
            if not body.content or not body.content.strip():
                raise HTTPException(
                    422, "Non-empty content required for edit"
                )

            original = snapshot.interrupts[0].value["action"]

            if original["name"] != "create_note_once":
                raise HTTPException(
                    422, "Editing this tool is not supported"
                )

            arguments = dict(original["arguments"])
            arguments["content"] = body.content

            decision = {
                "decision": "edit",
                "arguments": arguments,
            }
        else:
            if body.content is not None:
                raise HTTPException(
                    422, "Content is only allowed for edit"
                )
            decision = {"decision": body.decision}

        graph.invoke(
            Command(resume=decision),
            config=config,
        )

        return describe_workflow(graph, config, thread_id)