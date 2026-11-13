"""Compatibility HTTP endpoints for the V4.5/V5 HITL workflow."""

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from langgraph.types import Command

from api.core import THREAD_PATTERN, make_config, require_api_key
from api.models import DecisionRequest, StartRequest
from workflows.legacy import build_graph

router = APIRouter()


def describe_workflow(graph, config: dict, thread_id: str) -> dict:
    snapshot = graph.get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Workflow not found")

    approval = snapshot.interrupts[0].value if snapshot.interrupts else None
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


@router.post("/workflows", status_code=201)
def start_workflow(
    body: StartRequest,
    request: Request,
    _auth: None = Depends(require_api_key),
):
    with request.app.state.workflow_lock:
        saver = request.app.state.saver
        config = make_config(body.thread_id)
        graph, _, _ = build_graph(saver, body.thread_id, start=True)

        if graph.get_state(config).values:
            raise HTTPException(status_code=409, detail="Thread already exists")

        graph.invoke(
            {"prompt": "Calculate 125 * 48 and save it to a note"},
            config=config,
        )
        return describe_workflow(graph, config, body.thread_id)


@router.get("/workflows/{thread_id}")
def get_workflow(
    request: Request,
    thread_id: str = Path(pattern=THREAD_PATTERN),
    _auth: None = Depends(require_api_key),
):
    with request.app.state.workflow_lock:
        saver = request.app.state.saver
        config = make_config(thread_id)
        graph, _, _ = build_graph(saver, thread_id, start=False)
        return describe_workflow(graph, config, thread_id)


@router.post("/workflows/{thread_id}/decision")
def decide_workflow(
    body: DecisionRequest,
    request: Request,
    thread_id: str = Path(pattern=THREAD_PATTERN),
    _auth: None = Depends(require_api_key),
):
    with request.app.state.workflow_lock:
        saver = request.app.state.saver
        config = make_config(thread_id)
        graph, _, _ = build_graph(saver, thread_id, start=False)
        snapshot = graph.get_state(config)

        if not snapshot.values:
            raise HTTPException(status_code=404, detail="Workflow not found")
        if not snapshot.interrupts:
            raise HTTPException(status_code=409, detail="No pending approval")

        if body.decision == "edit":
            if not body.content or not body.content.strip():
                raise HTTPException(status_code=422, detail="Non-empty content required for edit")

            original = snapshot.interrupts[0].value["action"]
            if original["name"] != "create_note_once":
                raise HTTPException(status_code=422, detail="Editing this tool is not supported")

            arguments = dict(original["arguments"])
            arguments["content"] = body.content
            decision = {"decision": "edit", "arguments": arguments}
        else:
            if body.content is not None:
                raise HTTPException(status_code=422, detail="Content is only allowed for edit")
            decision = {"decision": body.decision}

        graph.invoke(Command(resume=decision), config=config)
        return describe_workflow(graph, config, thread_id)
