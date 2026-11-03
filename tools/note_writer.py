
import re
from pathlib import Path
from typing import Any

from agent.models import ToolDefinition
from tools.base import Tool


class WriteNoteTool(Tool):
    def __init__(self, directory="v42_notes"):
        self.directory = Path(directory)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_note",
            description="Write a note to a local file.",
            input_schema={
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "Identifier of the note.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text to write.",
                    },
                },
                "required": ["note_id", "content"],
                "additionalProperties": False,
            },
        )

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        note_id = arguments.get("note_id")
        content = arguments.get("content")

        if (
            not isinstance(note_id, str)
            or not re.fullmatch(
                r"[a-zA-Z0-9_-]{1,64}",
                note_id,
            )
        ):
            raise ValueError("Invalid note_id")

        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise ValueError("Invalid content")

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = self.directory / f"{note_id}.txt"
        path.write_text(content, encoding="utf-8")

        return f"Note saved: {path}"
