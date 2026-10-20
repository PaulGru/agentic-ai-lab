from agent.models import AgentOutcome
from agent.runner import AgentRunner
from orchestration.router import Router


class Orchestrator:
    def __init__(
        self,
        router: Router,
        max_handoff_depth: int = 3,
        verbose: bool = True,
    ):
        if max_handoff_depth <= 0:
            raise ValueError(
                "max_handoff_depth must be greater than 0"
            )

        self.router = router
        self.max_handoff_depth = max_handoff_depth
        self.verbose = verbose

    def run(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        agent = self.router.route(prompt)

        return self._execute(
            agent=agent,
            prompt=prompt,
            depth=0,
            active_agents=(),
        )

    def _execute(
        self,
        agent: AgentRunner,
        prompt: str,
        depth: int,
        active_agents: tuple[str, ...],
    ) -> str:
        if agent.name in active_agents:
            raise RuntimeError(
                f"Handoff cycle detected for "
                f"agent '{agent.name}'"
            )

        active_agents = (
            *active_agents,
            agent.name,
        )

        outcome = agent.run(prompt)

        return self._resolve_outcome(
            agent=agent,
            outcome=outcome,
            depth=depth,
            active_agents=active_agents,
        )

    def _resolve_outcome(
        self,
        agent: AgentRunner,
        outcome: AgentOutcome,
        depth: int,
        active_agents: tuple[str, ...],
    ) -> str:
        while outcome.handoff is not None:
            if depth >= self.max_handoff_depth:
                raise RuntimeError(
                    "Maximum handoff depth reached"
                )

            request = outcome.handoff

            target_agent = self.router.agents.get(
                request.target_agent
            )

            if target_agent is None:
                raise RuntimeError(
                    "Unknown handoff target: "
                    f"'{request.target_agent}'"
                )

            self._log(
                f"[orchestrator] handoff "
                f"{agent.name} -> {target_agent.name}"
            )

            delegated_result = self._execute(
                agent=target_agent,
                prompt=request.task,
                depth=depth + 1,
                active_agents=active_agents,
            )

            self._log(
                f"[orchestrator] return "
                f"{target_agent.name} -> {agent.name}"
            )

            outcome = agent.resume_handoff(
                request=request,
                result=delegated_result,
            )

        if outcome.text is None:
            raise RuntimeError(
                "Agent produced no final text"
            )

        return outcome.text

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)