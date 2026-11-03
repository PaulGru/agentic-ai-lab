"""Create-once note tool: retry-safe for a deterministic note ID and content.

This does not promise exactly-once semantics for arbitrary external tools.
A partially written file after a crash is treated as a conflict, not overwritten.
"""
import re
from pathlib import Path
from typing import Any

from agent.models import ToolDefinition
from tools.base import Tool


class CreateNoteOnceTool(Tool):
    def __init__(self, directory: str = "v45_notes"):
        self.directory = Path(directory)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="create_note_once",
            description="Create a new note once. Never overwrite a different note.",
            input_schema={
                "type": "object",
                "properties": {
                    "note_id": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["note_id", "content"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: dict[str, Any]) -> str:
        if set(arguments) != {"note_id", "content"}:
            raise ValueError("Expected note_id and content")
        note_id, content = arguments["note_id"], arguments["content"]
        if not isinstance(note_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", note_id):
            raise ValueError("Invalid note_id")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Invalid content")

        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{note_id}.txt"
        try:
            # Atomic exclusive creation (O_CREAT|O_EXCL) prevents replacement.
            with path.open("x", encoding="utf-8", newline="") as file:
                file.write(content)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != content:
                raise ValueError("Note ID already exists with different content")
            # Same content = successful replay of the same desired operation.
        return f"Note saved: {path}"
