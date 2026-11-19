"""Resumable, human-supervised workflow from a brief to a verified pilot paper."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Literal

from pydantic import Field

from analysis.contracts import HEX_SHA, fingerprint
from analysis.retry_pilot import CalibrationFailed, analyze_retry_pilot, verify_retry_calibration
from execution.backends.local import LocalDockerBackend
from execution.campaign import CampaignBlocked, PilotProtocol, build_report, run_campaign, save_report
from execution.contracts import StrictModel
from execution.job_manager import DurableJobManager
from execution.workloads import WORKLOAD_NAME as SYNTHETIC_WORKLOAD
from experiments.retry_pilot import PilotCorpus
from experiments.retry_pilot_runner import PhaseApproval, PilotJournal, approval_target, final_plan, run_phase
from research.brief_workflow_store import WorkflowStore
from research.critique.contracts import HumanDecision, LLMSettings
from research.critique.evidence import read_document
from research.critique.llm import bounded_provider
from research.critique.store import ReviewStore
from research.design_planner import ExperimentalDesignPlanner
from research.experimental_design import validate_approved_literature, validate_design_against_inputs
from research.literature import LiteratureCatalog, digest_json, render_markdown, validate_approved_mission
from research.literature_analyst import LiteratureAnalyst
from research.literature_search import InsufficientLiteratureError, OpenAlexClient, OpenAlexRateLimitError, retrieve_catalog
from research.manuscript import export_manuscript, record_manuscript, verify_manuscript_draft
from research.memory.contracts import MemoryBinding
from research.memory.integration import open_session
from research.mission import mission_digest, parse_brief, sha256_text
from research.mission_planner import MissionPlanner
from research.retry_study import WORKLOAD, check_study, protocol_for
from scripts.demo_v9_review import draft_pilot

ROOT = Path(__file__).resolve().parents[1]
MAX_REVISIONS = 4
MAX_INPUT_BYTES = 131_072
BRIEF_WORKLOADS = (WORKLOAD, SYNTHETIC_WORKLOAD, "unmapped")
SYNTHETIC_STAGE = "synthetic_execution"
STAGES = ("mission", "catalog", "literature", "protocol", SYNTHETIC_STAGE, "calibration", "measurement", "draft")
ORDER = {stage: index for index, stage in enumerate(STAGES)}
APPROVALS = {"review_mission": "mission", "review_literature": "literature", "review_protocol": "protocol",
             "review_manuscript": "draft"}
PROMPTS = {
    "mission": "Autoriser un appel du planificateur pour transformer le brief en mission ?",
    "review_mission": "Approuver la question, les hypothèses et le périmètre proposés ?",
    "catalog": "Autoriser la recherche OpenAlex avec les requêtes et les limites affichées ?",
    "literature": "Autoriser un appel de l'analyste pour synthétiser les notices récupérées ?",
    "review_literature": "Approuver la synthèse et les références proposées ?",
    "protocol": "Autoriser un appel du planificateur pour proposer le protocole ?",
    "review_protocol": "Approuver le protocole et le périmètre de l'expérience proposée ?",
    SYNTHETIC_STAGE: "Autoriser ces calculs synthétiques dans Docker local avec les paramètres affichés ?",
    "calibration": "Autoriser la calibration avec le budget affiché ?",
    "measurement": "Autoriser la mesure avec le budget affiché et la calibration vérifiée ?",
    "review_manuscript": "Approuver cette version du manuscrit pour l'export final ?",
}
MODEL_REPAIRS = {
    "mission": "Reformuler une mission conforme au schéma en conservant le périmètre et les contraintes du brief.",
    "literature": "Refaire la synthèse avec le catalogue déjà collecté. Pour chaque citation, reprendre un identifiant "
                  "du catalogue et un passage exact du résumé correspondant.",
    "protocol": "Revoir les hypothèses, les références et les contraintes de budget pour produire un protocole compatible "
                "avec la mission et la synthèse approuvées.",
}


class Decision(StrictModel):
    target_sha256: str = Field(pattern=HEX_SHA)
    actor: str = Field(min_length=1, max_length=100)


class Revision(Decision):
    stage: Literal["mission", "catalog", "literature", "protocol", "draft"]
    reason: str = Field(min_length=1, max_length=2000)
    brief: str | None = Field(default=None, max_length=12_000)
    queries: list[str] | None = Field(default=None, min_length=1, max_length=4)
    workload: Literal["retry_catalog", "synthetic.mean", "unmapped"] | None = None
    corpus: dict | None = None


class _OneCall:
    def __init__(self, factory, settings):
        self.factory = factory
        self.settings = settings
        self.sent = False
        self.received = False

    def start(self, prompt, tools=None, instructions=None):
        if self.sent or tools or len((prompt + (instructions or "")).encode("utf-8")) > MAX_INPUT_BYTES:
            raise ValueError("Planner request exceeds its approved scope")
        provider = self.factory(self.settings)
        self.sent = True
        reply = provider.start(prompt, tools=[], instructions=instructions)
        self.received = True
        if reply.web_search_used or reply.sources:
            raise ValueError("Planner reply contains unexpected external sources")
        return reply


class BriefWorkflow:
    def __init__(self, directory, *, provider_factory=bounded_provider, search_factory=OpenAlexClient,
                 experiment_client_factory=None, synthetic_transport=None):
        self.root = Path(directory).resolve()
        self.store = WorkflowStore(self.root / "workflow.sqlite3")
        self.provider_factory = provider_factory
        self.search_factory = search_factory
        self.experiment_client_factory = experiment_client_factory
        self.synthetic_transport = synthetic_transport

    def start(self, run_id, brief, settings, workload=None, corpus=None):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,23}", run_id):
            raise ValueError("Use a lowercase run identifier of at most 24 characters")
        sections = parse_brief(brief)
        declared = sections.get("Exécuteur", "").strip()
        if workload and declared and workload != declared:
            raise ValueError("The explicit workload differs from the brief")
        selected = workload or declared or "unmapped"
        if selected not in BRIEF_WORKLOADS:
            raise ValueError("Unknown workload. Use retry_catalog, synthetic.mean or leave the brief unmapped.")
        if corpus is not None:
            if selected != WORKLOAD:
                raise ValueError("A custom corpus requires the retry_catalog workload")
            corpus = PilotCorpus.model_validate(corpus).model_dump(mode="json")
        settings = LLMSettings.model_validate(settings.model_dump(mode="json"))
        self.store.append(None, {"id": run_id, "root": str(self.root), "version": 1, "stage": "mission",
                                 "artifacts": {}, "feedback": {}, "queries": None, "status": "waiting",
                                 "brief": brief, "llm": settings.model_dump(mode="json"), "workload": selected,
                                 "retry_corpus": corpus},
                          action="start")
        return self.status()

    def _current(self):
        event = self.store.current()
        if event["state"]["root"] != str(self.root):
            raise ValueError("Workflow directory differs from its recorded location")
        return event

    def _trial_paths(self, state):
        protocol = state["artifacts"]["protocol"]
        trial_dir = self.root / f"pilot-{protocol['version']:02}"
        paper_name = "paper" if state["version"] == protocol["version"] else f"paper-v{state['version']:02}"
        return trial_dir / "study.json", trial_dir / "pilot.sqlite3", trial_dir / paper_name

    def _draft_digest(self, state):
        _, _, paper = self._trial_paths(state)
        draft = state["artifacts"]["draft"]
        store = ReviewStore(paper / "review.sqlite3", read_only=True)
        manuscript = verify_manuscript_draft(store, draft["review_id"], paper / "draft")
        digest = manuscript.fingerprint()
        if digest != draft["manuscript_sha256"]:
            raise ValueError("Draft differs from the version awaiting approval")
        return digest

    def _study(self, state):
        artifacts = state["artifacts"]
        protocol = artifacts["protocol"]
        scoped = protocol["document"]
        source = {"kind": "brief_retry_study", "corpus": protocol["corpus"], "mission": artifacts["mission"],
                  "literature": artifacts["literature"], "protocol": scoped, "protocol_sha256": fingerprint(scoped)}
        return source, check_study(source)

    def _prepare_synthetic_protocol(self, state):
        artifacts = state["artifacts"]
        mission, _, _ = validate_approved_literature(artifacts["mission"], artifacts["literature"])
        template = read_document(ROOT / "examples" / "v846_protocol.json")
        campaign = PilotProtocol.model_validate({**template, "campaign_id": f"{state['id']}-v{state['version']}"})
        campaign.jobs()
        return {"version": state["version"], "document": {
            "workload": SYNTHETIC_WORKLOAD,
            "requested_question": mission.research_question,
            "measured_question": "Quelles sont les moyennes de deux listes définies et leur différence ?",
            "scope": "Contrôle du moteur sur des valeurs fictives, sans inférence sur la question du brief.",
            "mission_sha256": fingerprint(artifacts["mission"]),
            "literature_sha256": fingerprint(artifacts["literature"]),
            "campaign": campaign.model_dump(mode="json"),
        }}

    def _synthetic_campaign(self, state):
        protocol = state["artifacts"]["protocol"]
        path = self.root / "versions" / f"v{protocol['version']:02}" / "protocol.json"
        if fingerprint(read_document(path)) != fingerprint(protocol):
            raise ValueError("Exported synthetic protocol differs from the approved version")
        return PilotProtocol.model_validate(protocol["document"]["campaign"])

    @staticmethod
    def _synthetic_details(state, campaign):
        return {
            "backend": "docker-local", "network": False, "scientific_evidence": False,
            "protocol_sha256": fingerprint(state["artifacts"]["protocol"]),
            "campaign_sha256": campaign.fingerprint(),
            "jobs": [{"spec": job.model_dump(mode="json"), "sha256": job.fingerprint()} for job in campaign.jobs()],
        }

    def _synthetic_directory(self, state):
        return self.root / f"synthetic-{state['artifacts']['protocol']['version']:02}"

    def _complete_synthetic_report(self, state, campaign, approval):
        directory = self._synthetic_directory(state)
        manager = DurableJobManager(path=directory / "jobs.sqlite3", read_only=True)
        report = build_report(campaign, manager)
        report.update(brief_protocol_sha256=fingerprint(state["artifacts"]["protocol"]),
                      execution_approval=approval)
        path = save_report(report, directory / "report.json")
        state["report"] = {"path": str(path), "sha256": fingerprint(report)}
        state.update(stage="complete", status="complete")

    def _gate(self, event):
        state = event["state"]
        stage = state["stage"]
        if state["status"] != "waiting":
            return None
        artifacts = state["artifacts"]
        details = {}
        if stage == "draft" or (stage == "protocol" and state["workload"] in {WORKLOAD, SYNTHETIC_WORKLOAD}):
            return None
        if stage in {"mission", "literature", "protocol"}:
            details.update(llm=state["llm"], max_calls=1, max_input_bytes=MAX_INPUT_BYTES)
            if stage == "mission":
                details["brief"] = parse_brief(state["brief"])
            elif stage == "literature":
                details["catalog_sha256"] = fingerprint(artifacts["catalog"])
                details["papers"] = len(artifacts["catalog"]["papers"])
            else:
                details["mission_sha256"] = fingerprint(artifacts["mission"])
                details["literature_sha256"] = fingerprint(artifacts["literature"])
        elif stage == "catalog":
            mission = validate_approved_mission(artifacts["mission"])
            queries = state["queries"] or list(dict.fromkeys(mission.literature_queries))[:4]
            details.update(target="OpenAlex", queries=queries, results_per_query=5, max_http_requests=3 * len(queries))
        elif stage in APPROVALS:
            details["proposal"] = artifacts[APPROVALS[stage]]
            if stage == "review_manuscript":
                details["manuscript_sha256"] = self._draft_digest(state)
        elif stage == SYNTHETIC_STAGE:
            details = self._synthetic_details(state, self._synthetic_campaign(state))
        elif stage in {"calibration", "measurement"}:
            _, corpus = self._study(state)
            _, path, _ = self._trial_paths(state)
            journal = PilotJournal(path)
            details = approval_target(final_plan(corpus), journal, stage)
            if stage == "measurement" and details["calibration_sha256"] != verify_retry_calibration(corpus, journal):
                raise ValueError("Measurement authorization differs from verified calibration")
            details = {**details, "phase_target_sha256": details["target_sha256"],
                       "protocol_sha256": fingerprint(artifacts["protocol"]["document"])}
            del details["target_sha256"]
        gate = {"checkpoint_sha256": event["sha256"], "stage": stage, "version": state["version"],
                "kind": "review" if stage in APPROVALS else "authorization", "prompt": PROMPTS[stage],
                "details": details, "feedback": state["feedback"].get(APPROVALS.get(stage, stage), "")}
        return {**gate, "target_sha256": fingerprint(gate)}

    def _broader_queries(self, state):
        mission = validate_approved_mission(state["artifacts"]["mission"])
        queries = state["queries"] or list(dict.fromkeys(mission.literature_queries))[:4]
        tried = {query.casefold() for query in queries}
        for event in self.store.history():
            if event["action"] == "reserve" and event["detail"]["gate"]["stage"] == "catalog":
                tried.update(query.casefold() for query in event["detail"]["gate"]["details"]["queries"])
        alternatives = []
        for query in queries:
            words = query.split()
            if len(words) < 3:
                continue
            for terms in (words[:-1], words[1:]):
                candidate = " ".join(terms).strip(" ?.,")
                if 5 <= len(candidate) <= 500 and candidate.casefold() not in tried:
                    alternatives.append(candidate)
                    tried.add(candidate.casefold())
        return alternatives[:4]

    def _revision_gate(self, event, block):
        state = event["state"]
        if not block or block["uncertain"] or state["status"] != "blocked" or state["version"] > MAX_REVISIONS:
            return None
        cause, stage = block["reason"], state["stage"]
        if cause == "InsufficientLiteratureError" and stage == "catalog":
            queries = self._broader_queries(state)
            if not queries:
                return None
            proposal = {"stage": "catalog", "queries": queries,
                        "reason": "Le catalogue ne contient pas assez de notices ou de résumés. "
                                  "Essayer des requêtes plus courtes issues des termes déjà approuvés."}
        elif cause == "invalid_model_output" and stage in MODEL_REPAIRS:
            proposal = {"stage": stage, "reason": MODEL_REPAIRS[stage]}
        else:
            return None
        gate = {
            "kind": "revision", "stage": stage, "version": state["version"],
            "checkpoint_sha256": event["sha256"],
            "prompt": f"Accepter le retour proposé vers l'étape « {proposal['stage']} » ?",
            "details": {"cause": cause, "proposal": proposal, "external_calls_authorized": 0},
        }
        return {**gate, "target_sha256": fingerprint(gate)}

    @staticmethod
    def _attention(block, state):
        if not block:
            return None
        if block["uncertain"]:
            return "Examiner les reçus de l'opération. La reprise peut utiliser un reçu complet déjà enregistré."
        if state["version"] > MAX_REVISIONS:
            return "La limite de révisions est atteinte. Examiner l'historique avant de définir une nouvelle série."
        messages = {
            "no_compatible_executor": "Choisir ou fournir un exécuteur compatible avec le protocole.",
            "synthetic_execution_failed": "Examiner le statut du job avant de proposer un nouveau protocole.",
            "invalid_synthetic_report": "Le rapport exporté a changé ou manque. Vérifier les fichiers du pilote.",
            "calibration_did_not_pass": "Examiner les essais de calibration et définir ce qui doit changer dans le protocole.",
            "OpenAlexRateLimitError": "Vérifier le quota OpenAlex et choisir quand reprendre la recherche.",
            "InsufficientLiteratureError": "Préciser un nouveau vocabulaire de recherche ou reprendre la question.",
        }
        return messages.get(block["reason"], "Examiner l'étape bloquée et les options de retour affichées.")

    def status(self):
        event = self._current()
        state = event["state"]
        block = state.get("block")
        report = state.get("report")
        pending = None
        try:
            pending = self._gate(event)
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            block = {"reason": type(error).__name__, "return_to": ["protocol", "literature", "catalog", "mission"],
                     "uncertain": False}
        if state["status"] == "running":
            block = {"reason": "operation_reserved", "return_to": [], "uncertain": True}
        if report:
            try:
                if fingerprint(read_document(report["path"])) != report["sha256"]:
                    raise ValueError("Synthetic report digest mismatch")
            except (OSError, ValueError):
                report = None
                block = {"reason": "invalid_synthetic_report", "return_to": ["protocol"], "uncertain": False}
        if pending is None:
            pending = self._revision_gate(event, block)
        result = {
            "run_id": state["id"], "version": state["version"], "checkpoint_sha256": event["sha256"],
            "status": "blocked" if block and state["status"] in {"waiting", "complete"} else state["status"],
            "stage": state["stage"], "block": block, "pending": pending,
            "attention": self._attention(block, state) if pending is None else None,
            "artifacts": list(state["artifacts"]),
            "draft": state["artifacts"].get("draft"), "export": state.get("export"),
        }
        if state["workload"] == SYNTHETIC_WORKLOAD:
            result["report"] = report
        return result

    def revise(self, revision):
        return self._revise(revision)

    def _revise(self, revision, *, approved_gate=None):
        revision = Revision.model_validate(revision.model_dump(mode="json"))
        event = self._current()
        state = event["state"]
        if revision.target_sha256 != event["sha256"] or not revision.actor.strip() or not revision.reason.strip():
            raise PermissionError("Revision must target the current checkpoint and include a reason")
        if state["status"] == "running" or state.get("block", {}).get("uncertain"):
            raise PermissionError("An uncertain operation requires reconciliation before any other action")
        current = APPROVALS.get(state["stage"], state["stage"])
        if ORDER[revision.stage] > ORDER.get(current, ORDER["draft"]):
            raise ValueError("Revision cannot skip pending stages")
        if revision.stage == "draft" and state["workload"] == SYNTHETIC_WORKLOAD:
            raise ValueError("The synthetic calculation has no manuscript to revise")
        if state["version"] > MAX_REVISIONS:
            raise PermissionError("Revision limit reached")
        if revision.brief is not None:
            if revision.stage != "mission":
                raise ValueError("Brief changes require returning to the mission")
            declared = parse_brief(revision.brief).get("Exécuteur", "").strip()
            if declared and declared not in BRIEF_WORKLOADS:
                raise ValueError("Unknown workload in revised brief")
            state["workload"] = declared or "unmapped"
            state["retry_corpus"] = None
            state["brief"] = revision.brief
        if revision.queries is not None:
            clean = [query.strip() for query in revision.queries]
            if (revision.stage != "catalog" or len(set(clean)) != len(clean)
                    or any(not 5 <= len(query) <= 500 or "\n" in query or "\r" in query for query in clean)):
                raise ValueError("Provide up to four search queries of 5–500 characters when revising the catalog")
            state["queries"] = clean
        elif revision.stage == "mission":
            state["queries"] = None
        if revision.workload is not None:
            if revision.stage != "protocol":
                raise ValueError("Executor changes require returning to the protocol")
            state["workload"] = revision.workload
        if revision.corpus is not None:
            if revision.stage != "protocol" or state["workload"] != WORKLOAD:
                raise ValueError("A custom retry corpus requires revising the retry_catalog protocol")
            state["retry_corpus"] = PilotCorpus.model_validate(revision.corpus).model_dump(mode="json")
        prefix = ORDER[revision.stage]
        state["artifacts"] = {key: value for key, value in state["artifacts"].items() if ORDER[key] < prefix}
        state["feedback"] = {key: value for key, value in state["feedback"].items() if ORDER[key] < prefix}
        state["feedback"][revision.stage] = revision.reason.strip()
        state.pop("block", None)
        state.pop("export", None)
        state.pop("report", None)
        state.update(version=state["version"] + 1, stage=revision.stage, status="waiting")
        detail = revision.model_dump(mode="json")
        if approved_gate is not None:
            detail = {"revision": detail, "approved_gate": approved_gate}
        self.store.append(event["sha256"], state, action="approve_revision" if approved_gate else "revise", detail=detail)
        return self._finish_local()

    def _publish(self, state, document, filename):
        path = self.root / "versions" / f"v{state['version']:02}" / filename
        save_report(document, path)
        return path

    def _accept(self, state, stage, decision):
        artifacts = state["artifacts"]
        if stage == "review_mission":
            path = self._publish(state, artifacts["mission"], "mission.json")
            self._index(state, "mission", path)
            state["stage"] = "catalog"
        elif stage == "review_literature":
            mission = validate_approved_mission(artifacts["mission"])
            literature = {**artifacts["literature"], "status": "approved"}
            _, catalog, review = validate_approved_literature(artifacts["mission"], literature)
            path = self._publish(state, literature, "literature.json")
            md = path.with_suffix(".md")
            content = render_markdown(catalog, review, mission)
            if not md.exists():
                with md.open("x", encoding="utf-8") as handle:
                    handle.write(content)
            elif md.read_text(encoding="utf-8") != content:
                raise ValueError("Literature export differs")
            mission_path = self._publish(state, artifacts["mission"], "mission.json")
            self._index(state, "literature", path, parent_path=str(mission_path))
            artifacts["literature"] = literature
            state["stage"] = "protocol"
        elif stage == "review_protocol":
            self._publish(state, artifacts["protocol"], "protocol.json")
            if state["workload"] == SYNTHETIC_WORKLOAD:
                state["stage"] = SYNTHETIC_STAGE
            elif state["workload"] == WORKLOAD:
                source, _ = self._study(state)
                path, _, _ = self._trial_paths(state)
                save_report(source, path)
                state["stage"] = "calibration"
            else:
                self._block(state, "no_compatible_executor", ["protocol", "literature", "catalog", "mission"])
        else:
            result = artifacts["draft"]
            _, _, paper = self._trial_paths(state)
            store = ReviewStore(paper / "review.sqlite3")
            manuscript = verify_manuscript_draft(store, result["review_id"], paper / "draft")
            record_manuscript(store, manuscript, HumanDecision(
                decision="approve", actor=decision.actor, target_sha256=result["manuscript_sha256"],
            ))
            state["export"] = str(export_manuscript(store, result["review_id"], result["manuscript_sha256"],
                                                    paper / "approved"))
            state.update(stage="complete", status="complete")

    def _index(self, state, kind, path, **options):
        from research.memory.contracts import SourceOrigin
        binding = self._memory_binding(state)
        session = open_session(binding, initialize=True)
        session.memory.ingest(binding.project_id, SourceOrigin(kind=kind, path=str(path), **options))

    def _memory_binding(self, state):
        return MemoryBinding(project_id=state["id"], path=str(self.root / "memory.sqlite3"))

    @staticmethod
    def _block(state, reason, back, *, uncertain=False):
        state.update(status="blocked", block={"reason": reason, "return_to": back, "uncertain": uncertain})

    def _execute(self, state, stage, gate, decision, provider):
        artifacts = state["artifacts"]
        feedback = state["feedback"].get(stage, "")
        if stage in APPROVALS:
            self._accept(state, stage, decision)
        elif stage == "mission":
            sections = parse_brief(state["brief"])
            mission = MissionPlanner(provider).propose(sections, feedback=feedback).model_dump()
            artifacts["mission"] = {"schema_version": "8.1", "brief_sha256": sha256_text(state["brief"]),
                                    "brief_sections": sections, "mission": mission, "mission_sha256": mission_digest(mission)}
            state["stage"] = "review_mission"
        elif stage == "catalog":
            mission = validate_approved_mission(artifacts["mission"])
            catalog = retrieve_catalog(mission, self.search_factory(), max_queries=4, per_query=5,
                                       queries_override=state["queries"])
            artifacts["catalog"] = catalog.model_dump()
            state["stage"] = "literature"
        elif stage == "literature":
            mission = validate_approved_mission(artifacts["mission"])
            catalog = LiteratureCatalog.model_validate(artifacts["catalog"])
            review = LiteratureAnalyst(provider).analyze(mission, catalog, feedback=feedback).model_dump()
            artifacts["literature"] = {"schema_version": "8.2", "status": "pending",
                                       "mission_sha256": artifacts["mission"]["mission_sha256"],
                                       "catalog": artifacts["catalog"], "catalog_sha256": digest_json(artifacts["catalog"]),
                                       "review": review, "review_sha256": digest_json(review),
                                       "verification_level": "catalog_metadata_and_exact_abstract_excerpts_only"}
            state["stage"] = "review_literature"
        elif stage == "protocol":
            mission, _, literature = validate_approved_literature(artifacts["mission"], artifacts["literature"])
            design = ExperimentalDesignPlanner(provider).propose(mission, literature, feedback=feedback)
            validate_design_against_inputs(design, mission, literature)
            artifacts["protocol"] = {"version": state["version"], "document": design.model_dump(),
                                     "workload": "unmapped"}
            state["stage"] = "review_protocol"
        elif stage == SYNTHETIC_STAGE:
            campaign = self._synthetic_campaign(state)
            if gate["details"] != self._synthetic_details(state, campaign):
                raise PermissionError("Synthetic campaign differs from the authorized jobs")
            backend = LocalDockerBackend(execution_authorized=True, transport=self.synthetic_transport)
            manager = DurableJobManager(backend, self._synthetic_directory(state) / "jobs.sqlite3")
            run_campaign(campaign, manager)
            self._complete_synthetic_report(state, campaign, decision.model_dump(mode="json"))
        elif stage in {"calibration", "measurement"}:
            source, corpus = self._study(state)
            source_path, journal, _ = self._trial_paths(state)
            if read_document(source_path) != source:
                raise ValueError("Approved pilot source differs from the exported protocol")
            options = {"client_factory": self.experiment_client_factory} if self.experiment_client_factory else {}
            run_phase(corpus, PilotJournal(journal), PhaseApproval(
                phase=stage, actor=decision.actor, target_sha256=gate["details"]["phase_target_sha256"],
            ), **options)
            if stage == "calibration":
                try:
                    verify_retry_calibration(corpus, PilotJournal(journal))
                except CalibrationFailed:
                    self._block(state, "calibration_did_not_pass", ["protocol", "literature", "catalog", "mission"])
                    return
                state["stage"] = "measurement"
            else:
                state["stage"] = "draft"

    def approve(self, decision):
        decision = Decision.model_validate(decision.model_dump(mode="json"))
        event = self._current()
        gate = self._gate(event) or self._revision_gate(event, event["state"].get("block"))
        if gate is None or decision.target_sha256 != gate["target_sha256"] or not decision.actor.strip():
            raise PermissionError("Approval does not target the current action")
        if gate["kind"] == "revision":
            return self._revise(Revision(target_sha256=event["sha256"], actor=decision.actor,
                                         **gate["details"]["proposal"]), approved_gate=gate)
        state = event["state"]
        stage = state["stage"]
        state["status"] = "running"
        reserved = self.store.append(event["sha256"], state, action="reserve",
                                     detail={"decision": decision.model_dump(mode="json"), "gate": gate})
        provider = _OneCall(self.provider_factory, LLMSettings.model_validate(state["llm"]))
        try:
            state["status"] = "waiting"
            self._execute(state, stage, gate, decision, provider)
        except (InsufficientLiteratureError, OpenAlexRateLimitError) as error:
            self._block(state, type(error).__name__, ["catalog", "mission"])
            state["block"]["message"] = str(error)[:1500]
        except CampaignBlocked as error:
            self._block(state, "synthetic_execution_failed",
                        [] if error.uncertain else ["protocol", "literature", "catalog", "mission"],
                        uncertain=error.uncertain)
            state["block"]["message"] = str(error)
        except Exception as error:
            uncertain = (provider.sent and not provider.received) or stage in {
                SYNTHETIC_STAGE, "calibration", "measurement",
            }
            back = list(reversed(STAGES[:min(ORDER.get(APPROVALS.get(stage, stage), 3), 3) + 1]))
            reason = ("invalid_model_output" if provider.received and stage in MODEL_REPAIRS
                      and isinstance(error, ValueError) else type(error).__name__)
            self._block(state, reason, [] if uncertain else back, uncertain=uncertain)
        self.store.append(reserved["sha256"], state, action="result", detail={"stage": stage})
        return self._finish_local()

    def _finish_local(self):
        while True:
            event = self._current()
            state = event["state"]
            if state["status"] != "waiting":
                break
            stage = state["stage"]
            if stage == "protocol" and state["workload"] in {WORKLOAD, SYNTHETIC_WORKLOAD}:
                try:
                    artifacts = state["artifacts"]
                    if state["workload"] == SYNTHETIC_WORKLOAD:
                        artifacts["protocol"] = self._prepare_synthetic_protocol(state)
                    else:
                        template = state["retry_corpus"] or read_document(ROOT / "examples" / "v9_retry_pilot.json")
                        corpus = PilotCorpus.model_validate(template).model_copy(
                            update={"pilot_id": f"{state['id']}-v{state['version']}"})
                        artifacts["protocol"] = {"version": state["version"], "corpus": corpus.model_dump(mode="json"),
                                                 "document": protocol_for(artifacts["mission"], artifacts["literature"],
                                                                          corpus)}
                    state["stage"] = "review_protocol"
                except (ValueError, RuntimeError, OSError, KeyError) as error:
                    self._block(state, type(error).__name__, ["protocol", "literature", "catalog", "mission"])
                self.store.append(event["sha256"], state, action="prepare_protocol")
            elif stage == "draft":
                source, journal, paper = self._trial_paths(state)
                try:
                    expected, _ = self._study(state)
                    if read_document(source) != expected:
                        raise ValueError("Pilot source no longer matches its approved protocol")
                    review_id = f"pilot-review-v{state['version']}"
                    result = draft_pilot(source, journal, paper, review_id, state["id"],
                                         memory_path=self._memory_binding(state).path)
                    state["artifacts"]["draft"] = result
                    state["stage"] = "review_manuscript"
                except (ValueError, RuntimeError, OSError, KeyError) as error:
                    self._block(state, type(error).__name__, ["draft", "protocol", "literature", "catalog", "mission"])
                self.store.append(event["sha256"], state, action="prepare_draft")
            else:
                break
        return self.status()

    def recover(self, decision):
        decision = Decision.model_validate(decision.model_dump(mode="json"))
        event = self._current()
        state = event["state"]
        if decision.target_sha256 != event["sha256"] or not decision.actor.strip():
            raise PermissionError("Recovery must target the current checkpoint")
        stage = state["stage"]
        if stage == SYNTHETIC_STAGE and (state["status"] == "running" or state.get("block", {}).get("uncertain")):
            reservation = event
            if event["action"] != "reserve":
                reservation = next(item for item in self.store.history() if item["sha256"] == event["previous_sha256"])
            reserved = reservation["detail"]
            campaign = self._synthetic_campaign(state)
            if (reservation["action"] != "reserve" or reserved["gate"]["stage"] != SYNTHETIC_STAGE
                    or reserved["decision"]["target_sha256"] != reserved["gate"]["target_sha256"]
                    or reserved["gate"]["details"] != self._synthetic_details(state, campaign)):
                raise PermissionError("Synthetic receipts do not match the reserved operation")
            try:
                self._complete_synthetic_report(state, campaign, reserved["decision"])
            except (FileNotFoundError, RuntimeError) as error:
                raise PermissionError("No complete synthetic job receipts. Reconcile manually; no job was replayed.") from error
            state.pop("block", None)
            self.store.append(event["sha256"], state, action="recover_synthetic_report",
                              detail=decision.model_dump(mode="json"))
            return self.status()
        if stage in {"calibration", "measurement"} and state["status"] == "running":
            source, corpus = self._study(state)
            source_path, path, _ = self._trial_paths(state)
            if read_document(source_path) != source:
                raise ValueError("Pilot source differs from its approved protocol")
            journal = PilotJournal(path)
            events = [item for item in journal.events() if item["phase"] == stage]
            if not events or events[-1]["step"] != "completed":
                raise PermissionError("No complete receipt. Reconcile manually; no experiment was replayed.")
            reserved = event["detail"]
            recorded = events[0]["document"]["approval"]
            if recorded != {"phase": stage, "actor": reserved["decision"]["actor"],
                            "target_sha256": reserved["gate"]["details"]["phase_target_sha256"]}:
                raise PermissionError("Receipt approval does not match the reserved operation")
            if stage == "measurement":
                analyze_retry_pilot(corpus, journal)
                state["stage"] = "draft"
            else:
                try:
                    verify_retry_calibration(corpus, journal)
                except CalibrationFailed:
                    self._block(state, "calibration_did_not_pass", ["protocol", "literature", "catalog", "mission"])
                    self.store.append(event["sha256"], state, action="recover_completed_calibration",
                                      detail=decision.model_dump(mode="json"))
                    return self.status()
                state["stage"] = "measurement"
        elif (state["status"] not in {"blocked", "waiting"} or not (
                stage == "draft" or (stage == "protocol" and state["workload"] in {WORKLOAD, SYNTHETIC_WORKLOAD}))):
            raise PermissionError("Recovery requires a safe local step or a complete existing pilot receipt")
        state.pop("block", None)
        state["status"] = "waiting"
        self.store.append(event["sha256"], state, action="recover_local", detail=decision.model_dump(mode="json"))
        return self._finish_local()
