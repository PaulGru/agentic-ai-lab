
from fastapi import Depends, HTTPException, Path, Request
from pydantic import BaseModel, Field, field_validator

from api_v54 import (
    app,
    make_config,
    require_api_key,
    THREAD_PATTERN,
)
from research.factory import build_research_graph


app.version = "6.1"


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

    return {
        "thread_id": thread_id,
        "status": (
            "completed" if not snapshot.next
            else "in_progress"
        ),
        "answer": snapshot.values.get("answer") or None,
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
