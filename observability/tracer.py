from dataclasses import dataclass
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class TraceEvent:
    run_id: str
    type: str
    data: dict[str, Any]


class Tracer:
    def __init__(
        self,
        print_events: bool = True,
    ):
        self.print_events = print_events
        self.run_id: str | None = None
        self.events: list[TraceEvent] = []

    def start_run(self) -> str:
        self.run_id = str(uuid4())
        self.events = []

        return self.run_id

    def record(
        self,
        event_type: str,
        **data,
    ) -> None:
        if self.run_id is None:
            self.start_run()

        event = TraceEvent(
            run_id=self.run_id,
            type=event_type,
            data=data,
        )

        self.events.append(event)

        if self.print_events:
            print(
                f"[trace:{self.run_id[:8]}] "
                f"{event.type} {event.data}"
            )