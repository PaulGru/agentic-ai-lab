
from fastapi import Depends, HTTPException, Path, Request
from pydantic import BaseModel, Field, field_validator

from api_v54 import (
    app,
    make_config,
    require_api_key,
    THREAD_PATTERN,
)
from research.factory import build_research_graph
from research.pipeline import build_research_pipeline


app.version = "6.4"


class ResearchRequest(BaseModel):
    thread_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=THREAD_PATTERN,
    )
    question: str = Field(
        min_length=1,
        max_length=4000,
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be empty")

        return value.strip()


def research_config(thread_id: str) -> dict:
    # Isoler nos workflows V6 des anciens workflows V5.
    return make_config(f"v6-research:{thread_id}")


def describe_research(graph, config, thread_id):
    snapshot = graph.get_state(config)

    if not snapshot.values:
        raise HTTPException(
            status_code=404,
            detail="Research workflow not found",
        )

    values = snapshot.values

    return {
        "thread_id": thread_id,
        "status": (
            "completed" if not snapshot.next
            else "in_progress"
        ),
        "answer": values.get("answer") or None,
        "web_search_used": values.get(
            "web_search_used", False
        ),
        "sources": values.get("sources", []),

        # V6.3 - Rapport d'audit
        "source_audit": values.get("source_audit"),
    }


@app.post("/research", status_code=201)
def start_research(
    body: ResearchRequest,
    request: Request,
    _auth: None = Depends(require_api_key),
):
    with request.app.state.workflow_lock:
        saver = request.app.state.saver
        config = research_config(body.thread_id)

        graph = build_research_graph(saver)

        if graph.get_state(config).values:
            raise HTTPException(
                status_code=409,
                detail="Research thread already exists",
            )

        graph.invoke(
            {"prompt": body.question},
            config=config,
        )

        return describe_research(
            graph, config, body.thread_id
        )


@app.get("/research/{thread_id}")
def get_research(
    request: Request,
    thread_id: str = Path(
        min_length=1,
        max_length=64,
        pattern=THREAD_PATTERN,
    ),
    _auth: None = Depends(require_api_key),
):
    with request.app.state.workflow_lock:
        saver = request.app.state.saver
        config = research_config(thread_id)

        graph = build_research_graph(saver)

        return describe_research(
            graph, config, thread_id
        )



# V6.4 - Pipeline de recherche multi-agent

def pipeline_config(thread_id: str) -> dict:
    return make_config(f"v64-pipeline:{thread_id}")


def describe_pipeline(graph, config, thread_id):
    snapshot = graph.get_state(config)

    if not snapshot.values:
        raise HTTPException(
            status_code=404,
            detail="Research pipeline not found",
        )

    values = snapshot.values

    return {
        "thread_id": thread_id,
        "status": (
            "completed" if not snapshot.next
            else "in_progress"
        ),
        "question": values.get("question"),
        "plan": values.get("plan", []),
        "findings": values.get("findings", []),
        "answer": values.get("answer") or None,
        "web_search_used": values.get(
            "web_search_used", False
        ),
        "sources": values.get("sources", []),
        "source_audit": values.get("source_audit"),
    }


@app.post("/research/pipeline", status_code=201)
def start_research_pipeline(
    body: ResearchRequest,
    request: Request,
    _auth: None = Depends(require_api_key),
):
    with request.app.state.workflow_lock:
        saver = request.app.state.saver
        config = pipeline_config(body.thread_id)

        graph = build_research_pipeline(saver)

        if graph.get_state(config).values:
            raise HTTPException(
                status_code=409,
                detail="Research pipeline already exists",
            )

        graph.invoke(
            {"question": body.question},
            config=config,
        )

        return describe_pipeline(
            graph, config, body.thread_id
        )


@app.get("/research/pipeline/{thread_id}")
def get_research_pipeline(
    request: Request,
    thread_id: str = Path(
        min_length=1,
        max_length=64,
        pattern=THREAD_PATTERN,
    ),
    _auth: None = Depends(require_api_key),
):
    with request.app.state.workflow_lock:
        saver = request.app.state.saver
        config = pipeline_config(thread_id)

        graph = build_research_pipeline(saver)

        return describe_pipeline(
            graph, config, thread_id
        )
