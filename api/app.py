"""Stable HTTP entrypoint. No versioned API modules."""

from fastapi import FastAPI

from api.core import lifespan
from api.routes.research import router as research_router
from api.routes.workflows import router as workflows_router

app = FastAPI(title="Agentic AI Lab", version="6.6", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(workflows_router)
app.include_router(research_router)
