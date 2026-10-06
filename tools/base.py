from abc import ABC, abstractmethod
from typing import Any

from agent.models import ToolDefinition


class Tool(ABC):
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        pass

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> str:
        pass