"""Resumable, human-supervised workflow from a brief to a verified pilot paper."""
from __future__ import annotations

import json
from pathlib import Path
import re
from subprocess import TimeoutExpired
from typing import Literal

from pydantic import Field

from analysis.agentic_gpu import SOURCE_KIND as AGENTIC_SOURCE_KIND, analyze_agentic
from analysis.contracts import HEX_SHA, fingerprint
from analysis.executable_protocol import SOURCE_KIND as PROGRAM_SOURCE_KIND, analyze_protocol
from analysis.ood_replay import GPU_SOURCE_KIND, SOURCE_KIND as OOD_SOURCE_KIND, analyze_ood_gpu, analyze_ood_replay
from analysis.retry_gpu import SOURCE_KIND as RETRY_GPU_SOURCE_KIND, analyze_retry_gpu
from analysis.retry_pilot import CalibrationFailed, analyze_retry_pilot, verify_retry_calibration
from execution.agentic_gpu import CAPABILITY as AGENTIC_CAPABILITY, AgenticGPUBackend, AgenticGPUConfig
from execution.agentic_gpu_worker import worker_sha256 as agentic_worker_sha256
from execution.backends.local import LocalDockerBackend
from execution.campaign import CampaignBlocked, PilotProtocol, build_report, run_campaign, save_report
from execution.contracts import StrictModel
from execution.job_manager import DurableJobManager
from execution.ood_gpu import OODGPUConfig, OODGPUProtocol, WORKLOAD as GPU_WORKLOAD, run_ood_gpu
from execution.ood_replay import WORKLOAD as OOD_WORKLOAD, run_ood_replay
from execution.protocol_cpu import ProtocolCPUBackend, run_program_plan
from execution.protocol_gpu import CAPABILITY as PROGRAM_GPU_CAPABILITY, ProtocolGPUBackend, ProtocolGPUConfig, worker_sha256
from execution.retry_gpu import RetryGPUConfig, WORKLOAD as RETRY_GPU_WORKLOAD, run_retry_gpu
from execution.retry_gpu_worker import worker_sha256 as retry_worker_sha256
from execution.workloads import WORKLOAD_NAME as SYNTHETIC_WORKLOAD
from experiments.agent import ExperimentAgent
from experiments.protocol_program import EvaluationSet, interpreter_sha256
from experiments.retry_pilot import PilotCorpus
from experiments.retry_pilot_runner import PhaseApproval, PilotJournal, approval_target, final_plan, run_phase
from observability.model_trace import capture_model_trace
from providers.openai_provider import reasoning_for
from research import agentic_study
from research.brief_workflow_store import WorkflowStore
from research.capability_resolution import (
    propose_resolution, resolution_record, resolution_scope, resolve_resource, verify_resolved_resource,
)
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
from research.ood_study import GPU_STUDY_KIND, REPLAY_STUDY_KIND, check_ood_study, ood_protocol_for
from research.pilot_manuscript import draft_pilot
from research.protocol_study import (
    STUDY_KIND as PROGRAM_STUDY_KIND, compile_study, design_constraints, implementation_request, validate_capability,
)
from research.resource_catalog import choose_resource, resources_for
from research.retry_gpu_study import STUDY_KIND as RETRY_GPU_STUDY_KIND, check_study as check_retry_gpu_study
from research.retry_gpu_study import protocol_for as retry_gpu_protocol_for
from research.retry_study import WORKLOAD, check_study, protocol_for

ROOT = Path(__file__).resolve().parents[1]
MAX_REVISIONS = 4
MAX_INPUT_BYTES = 131_072
OOD_WORKLOADS = {OOD_WORKLOAD, GPU_WORKLOAD}
GPU_CONFIGS = {GPU_WORKLOAD: OODGPUConfig, RETRY_GPU_WORKLOAD: RetryGPUConfig}
RETRY_WORKLOADS = {WORKLOAD, RETRY_GPU_WORKLOAD}
PREDEFINED_PROTOCOL_WORKLOADS = {SYNTHETIC_WORKLOAD, *OOD_WORKLOADS, *RETRY_WORKLOADS}
BRIEF_WORKLOADS = (WORKLOAD, SYNTHETIC_WORKLOAD, OOD_WORKLOAD, GPU_WORKLOAD, RETRY_GPU_WORKLOAD, "unmapped")
SYNTHETIC_STAGE = "synthetic_execution"
OOD_STAGE = "ood_execution"
GPU_STAGE = "ood_gpu_execution"
RETRY_GPU_STAGE = "retry_gpu_execution"
RESOURCE_STAGE = "resource_selection"
RESOLUTION_STAGE = "capability_resolution"
IMPLEMENTATION_STAGE = "implementation"
PROGRAM_STAGE = "protocol_execution"
OOD_STAGES = {OOD_STAGE, GPU_STAGE}
STAGES = ("mission", "catalog", "literature", RESOURCE_STAGE, RESOLUTION_STAGE, "protocol", IMPLEMENTATION_STAGE, PROGRAM_STAGE,
          SYNTHETIC_STAGE, OOD_STAGE, GPU_STAGE, RETRY_GPU_STAGE,
          "calibration", "measurement", "draft")
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
    RESOURCE_STAGE: "Autoriser la sélection d'un corpus et du modèle compatible dans le catalogue affiché ?",
    RESOLUTION_STAGE: "Autoriser une tentative de résolution du manque dans le périmètre affiché ?",
    SYNTHETIC_STAGE: "Autoriser ces calculs synthétiques dans Docker local avec les paramètres affichés ?",
    OOD_STAGE: "Autoriser la répétition OOD avec les huit réponses fictives prévues ?",
    GPU_STAGE: "Autoriser le lot de calibration SSH sur le GPU et la révision affichés ?",
    RETRY_GPU_STAGE: "Autoriser ce plan d'essais appariés sur le GPU et la révision affichés ?",
    IMPLEMENTATION_STAGE: "Autoriser l'agent de code à préparer l'expérience depuis le protocole approuvé ?",
    PROGRAM_STAGE: "Autoriser la mesure avec le programme, le corpus et les jobs affichés ?",
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
    RESOURCE_STAGE: "Choisir uniquement un identifiant du catalogue et citer exactement la question approuvée.",
}


class Decision(StrictModel):
    target_sha256: str = Field(pattern=HEX_SHA)
    actor: str = Field(min_length=1, max_length=100)


class Revision(Decision):
    stage: Literal["mission", "catalog", "literature", "resource_selection", "protocol", "implementation", "draft"]
    reason: str = Field(min_length=1, max_length=2000)
    brief: str | None = Field(default=None, max_length=12_000)
    queries: list[str] | None = Field(default=None, min_length=1, max_length=4)
    workload: Literal["retry_catalog", "synthetic.mean", "ood_replay", "ood_gpu", "retry_gpu", "unmapped"] | None = None
    corpus: dict | None = None
    evaluation: dict | None = None
    gpu_config: OODGPUConfig | RetryGPUConfig | ProtocolGPUConfig | AgenticGPUConfig | None = None


def _is_agentic(evaluation):
    return bool(evaluation and evaluation["schema_version"] == AGENTIC_CAPABILITY)


def _gpu_config_model(workload, evaluation, *, optional_target=False, target=None):
    if workload == "unmapped" and (_is_agentic(evaluation) or (
            optional_target and isinstance(target, dict) and target.get("worker_path", "").endswith("/agentic_gpu_worker.py"))):
        return AgenticGPUConfig
    if workload == "unmapped" and (optional_target or (evaluation and evaluation["schema_version"] == PROGRAM_GPU_CAPABILITY)):
        return ProtocolGPUConfig
    return GPU_CONFIGS.get(workload)


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

    def start(self, run_id, brief, settings, workload=None, corpus=None, *, gpu_config=None, evaluation=None,
              auto_resources=None):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,23}", run_id):
            raise ValueError("Use a lowercase run identifier of at most 24 characters")
        sections = parse_brief(brief)
        declared = sections.get("Exécuteur", "").strip()
        if workload and declared and workload != declared:
            raise ValueError("The explicit workload differs from the brief")
        selected = workload or declared or "unmapped"
        if selected not in BRIEF_WORKLOADS:
            raise ValueError(f"Unknown workload. Choose one of {', '.join(BRIEF_WORKLOADS)}.")
        if evaluation is not None:
            if selected != "unmapped":
                raise ValueError("An independent evaluation set requires a newly designed protocol")
            evaluation = EvaluationSet.model_validate(evaluation).model_dump(mode="json")
        if auto_resources is None:
            auto_resources = selected == "unmapped" and evaluation is None and (not workload or gpu_config is not None)
        if auto_resources and (selected != "unmapped" or evaluation is not None):
            raise ValueError("Automatic selection requires a new protocol without an explicit evaluation set")
        config_model = _gpu_config_model(selected, evaluation, optional_target=auto_resources and gpu_config is not None,
                                         target=gpu_config)
        if (config_model is not None) != (gpu_config is not None):
            raise ValueError("GPU workloads require an explicit GPU configuration")
        gpu_settings = ({"gpu_config": config_model.model_validate(gpu_config).model_dump(mode="json")}
                        if gpu_config is not None else {})
        if selected == RETRY_GPU_WORKLOAD and corpus is None:
            raise ValueError("retry_gpu requires an explicit catalog corpus")
        if corpus is not None:
            if selected not in RETRY_WORKLOADS:
                raise ValueError("A custom corpus requires a catalog retry workload")
            corpus = PilotCorpus.model_validate(corpus).model_dump(mode="json")
        settings = LLMSettings.model_validate(settings.model_dump(mode="json"))
        self.store.append(None, {"id": run_id, "root": str(self.root), "version": 1, "stage": "mission",
                                 "artifacts": {}, "feedback": {}, "queries": None, "status": "waiting",
                                 "brief": brief, "llm": settings.model_dump(mode="json"), "workload": selected,
                                 "retry_corpus": corpus, "evaluation": evaluation, "auto_resources": auto_resources, **gpu_settings},
                          action="start")
        return self.status()

    def _current(self):
        event = self.store.current()
        if event["state"]["root"] != str(self.root):
            raise ValueError("Workflow directory differs from its recorded location")
        return event

    def _trial_paths(self, state):
        protocol = state["artifacts"]["protocol"]
        implementation = state["artifacts"].get(IMPLEMENTATION_STAGE)
        version = implementation["version"] if implementation else protocol["version"]
        trial_dir = self.root / f"pilot-{version:02}"
        paper_name = "paper" if state["version"] == version else f"paper-v{state['version']:02}"
        jobs = implementation is not None or _is_agentic(state.get("evaluation")) or state["workload"] in {
            *OOD_WORKLOADS, RETRY_GPU_WORKLOAD,
        }
        ledger_name = "jobs.sqlite3" if jobs else "pilot.sqlite3"
        return trial_dir / "study.json", trial_dir / ledger_name, trial_dir / paper_name

    def _implementation_request(self, state):
        protocol = state["artifacts"]["protocol"]
        path = self.root / "versions" / f"v{protocol['version']:02}" / "protocol.json"
        evaluation = EvaluationSet.model_validate(state["evaluation"])
        if (read_document(path) != protocol or "evaluation_sha256" not in protocol
                or fingerprint(evaluation.model_dump(mode="json")) != protocol["evaluation_sha256"]
                or protocol["interpreter_sha256"] != interpreter_sha256()):
            raise ValueError("Implementation inputs differ from the approved protocol")
        request = implementation_request(protocol["document"], evaluation,
                                         feedback=state["feedback"].get(IMPLEMENTATION_STAGE, ""))
        selection = self._verified_resolution(state)
        if selection:
            if protocol["resource_selection"] != selection:
                raise ValueError("Implementation inputs differ from the verified resource")
            request["resource_description"] = selection["resource"]["description"]
        return request

    def _verified_resolution(self, state):
        artifacts = state["artifacts"]
        selection = artifacts.get(RESOURCE_STAGE)
        if not selection or not selection.get("capability_resolution"):
            return None
        record = artifacts[RESOLUTION_STAGE]
        directory = self.root / "versions" / f"v{record['version']:02}"
        resolution = selection["capability_resolution"]
        if (record["document"] != resolution or read_document(directory / "capability-resolution.json") != resolution
                or read_document(directory / "resource-selection.json") != resolution["gap"]
                or read_document(directory / "resolved-resource.json") != selection
                or read_document(directory / "resolved-evaluation.json") != state["evaluation"]
                or selection["mission_sha256"] != fingerprint(artifacts["mission"])
                or selection["literature_sha256"] != fingerprint(artifacts["literature"])):
            raise ValueError("Resolution exports differ from the recorded study")
        verify_resolved_resource(selection, EvaluationSet.model_validate(state["evaluation"]),
                                 artifacts["mission"]["mission"]["research_question"])
        return selection

    def _program_source(self, state):
        artifacts = state["artifacts"]
        if _is_agentic(state.get("evaluation")):
            return {"kind": agentic_study.STUDY_KIND, "run_id": state["id"], "version": artifacts["protocol"]["version"],
                    "mission": artifacts["mission"], "literature": artifacts["literature"],
                    "protocol": artifacts["protocol"], "evaluation": state["evaluation"]}
        implementation = artifacts[IMPLEMENTATION_STAGE]
        return {"kind": PROGRAM_STUDY_KIND, "run_id": state["id"], "version": implementation["version"],
                "mission": artifacts["mission"], "literature": artifacts["literature"],
                "protocol": artifacts["protocol"], "evaluation": state["evaluation"],
                "proposal": implementation["proposal"]}

    def _program_inputs(self, state):
        if _is_agentic(state.get("evaluation")):
            source = self._program_source(state)
            _, jobs, plan = agentic_study.compile_study(source)
            path, ledger, _ = self._trial_paths(state)
            approved = self.root / "versions" / f"v{source['version']:02}" / "protocol.json"
            if (read_document(approved) != source["protocol"] or read_document(path) != source
                    or read_document(path.with_name("action-plan.json")) != plan):
                raise ValueError("Agentic exports differ from the approved protocol")
            return source, jobs, plan, ledger
        self._implementation_request(state)
        source = self._program_source(state)
        _, _, jobs, plan = compile_study(source)
        path, ledger, _ = self._trial_paths(state)
        candidate = self.root / "versions" / f"v{source['version']:02}" / "candidate.json"
        if (read_document(path) != source or read_document(path.with_name("action-plan.json")) != plan
                or read_document(candidate) != source["proposal"]):
            raise ValueError("Execution exports differ from the approved implementation")
        return source, jobs, plan, ledger

    def _complete_program(self, state, source, ledger, approval):
        manager = DurableJobManager(path=ledger, read_only=True)
        agentic = source["kind"] == agentic_study.STUDY_KIND
        analysis = analyze_agentic(source, manager) if agentic else analyze_protocol(source, manager)
        report = {**analysis.model_dump(mode="json"), "approval": approval}
        preparation_path = ledger.with_name("model-preparation.json")
        if agentic:
            preparation = read_document(preparation_path)
            _, jobs, plan = agentic_study.compile_study(source)
            expected = {job.workload.parameters["model_key"]: job for job in jobs}
            if (set(preparation) != expected.keys() or any(
                row.get("ready") is not True or row.get("model_key") != key
                or row.get("job_sha256") != expected[key].fingerprint() or row.get("worker_sha256") != plan["worker_sha256"]
                or row.get("manifest") != {k: v for k, v in plan["models"][key].items() if k != "snapshot"}
                for key, row in preparation.items()
            )):
                raise ValueError("Model preparation receipts differ from the agentic plan")
            report["model_preparation_sha256"] = fingerprint(preparation)
        elif analysis.gpu_model and preparation_path.is_file():
            preparation = read_document(preparation_path)
            protocol = source["protocol"]
            if (preparation.get("job_sha256") != analysis.trials[0].job_sha256
                    or preparation.get("worker_sha256") != protocol["worker_sha256"]
                    or preparation.get("snapshot") != protocol["gpu_config"]["snapshot_path"]
                    or preparation.get("ready") is not True):
                raise ValueError("Model preparation receipt differs from the executed plan")
            report["model_preparation_sha256"] = fingerprint(preparation)
        path = save_report(report, ledger.with_name("report.json"))
        state["report"] = {"path": str(path), "sha256": fingerprint(report)}
        state.update(stage="draft", status="waiting")

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

    def _retry_gpu_study(self, state):
        artifacts = state["artifacts"]
        protocol = artifacts["protocol"]
        document = protocol["document"]
        source = {"kind": RETRY_GPU_STUDY_KIND, "corpus": protocol["corpus"], "mission": artifacts["mission"],
                  "literature": artifacts["literature"], "protocol": document, "protocol_sha256": fingerprint(document)}
        corpus, plan = check_retry_gpu_study(source)
        return source, corpus, plan

    def _retry_gpu_inputs(self, state):
        source, corpus, plan = self._retry_gpu_study(state)
        study_path, ledger, _ = self._trial_paths(state)
        version = state["artifacts"]["protocol"]["version"]
        approved_protocol = self.root / "versions" / f"v{version:02}" / "protocol.json"
        if (read_document(approved_protocol) != state["artifacts"]["protocol"]
                or read_document(study_path) != source
                or read_document(study_path.with_name("action-plan.json")) != plan):
            raise ValueError("Catalog GPU exports differ from the approved protocol and action plan")
        return source["protocol"], corpus, plan, ledger

    def _complete_retry_gpu(self, state, protocol, corpus, plan, ledger, approval):
        manager = DurableJobManager(path=ledger, read_only=True)
        report = analyze_retry_gpu(protocol, corpus, plan, manager).model_dump(mode="json")
        report["approval"] = approval
        path = save_report(report, ledger.with_name("report.json"))
        state["report"] = {"path": str(path), "sha256": fingerprint(report)}
        state.update(stage="draft", status="waiting")

    @staticmethod
    def _ood_study(state):
        artifacts = state["artifacts"]
        document = artifacts["protocol"]["document"]
        source = {
            "kind": GPU_STUDY_KIND if state["workload"] == GPU_WORKLOAD else REPLAY_STUDY_KIND,
            "metric_name": "ood.exact_match", "mission": artifacts["mission"],
            "literature": artifacts["literature"], "protocol": document, "protocol_sha256": fingerprint(document),
        }
        return source, check_ood_study(source)

    def _ood_execution_inputs(self, state):
        expected, protocol = self._ood_study(state)
        source, ledger, _ = self._trial_paths(state)
        if read_document(source) != expected:
            raise ValueError("OOD study source differs from the approved protocol")
        return protocol, ledger

    @staticmethod
    def _ood_details(protocol):
        gpu = isinstance(protocol, OODGPUProtocol)
        jobs = [protocol.job()] if gpu else protocol.jobs()
        runtime = {"gpu_config": protocol.config.model_dump(mode="json"), "worker_sha256": protocol.worker_sha256} if gpu else {}
        return {
            "backend": "ood-gpu-ssh" if gpu else "ood_replay", "simulated": not gpu, "network": gpu, "gpu": gpu,
            **runtime,
            "protocol_sha256": protocol.fingerprint(),
            "jobs": [{"spec": job.model_dump(mode="json"), "sha256": job.fingerprint()} for job in jobs],
        }

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
        if stage == "draft" or (stage == "protocol" and state["workload"] in PREDEFINED_PROTOCOL_WORKLOADS):
            return None
        if stage in {"mission", "literature", "protocol"}:
            details.update(llm=state["llm"], max_calls=1, max_input_bytes=MAX_INPUT_BYTES)
            if state["llm"]["provider"] == "openai":
                reasoning = reasoning_for(state["llm"]["model"])
                if reasoning is not None:
                    details["reasoning"] = reasoning
            if stage == "mission":
                details["brief"] = parse_brief(state["brief"])
            elif stage == "literature":
                details["catalog_sha256"] = fingerprint(artifacts["catalog"])
                details["papers"] = len(artifacts["catalog"]["papers"])
            else:
                details["mission_sha256"] = fingerprint(artifacts["mission"])
                details["literature_sha256"] = fingerprint(artifacts["literature"])
                if state.get("evaluation"):
                    details["execution_constraints"] = (agentic_study.design_constraints() if _is_agentic(state["evaluation"])
                                                        else design_constraints(EvaluationSet.model_validate(state["evaluation"])))
                    if selection := self._verified_resolution(state):
                        details["resource_description"] = selection["resource"]["description"]
        elif stage == RESOURCE_STAGE:
            catalog = resources_for(state.get("gpu_config"))
            details = {"llm": state["llm"], "max_calls": 1, "max_input_bytes": MAX_INPUT_BYTES,
                       "mission_sha256": fingerprint(artifacts["mission"]),
                       "literature_sha256": fingerprint(artifacts["literature"]),
                       "catalog": [metadata for metadata, _ in catalog.values()]}
        elif stage == RESOLUTION_STAGE:
            gap = artifacts[RESOURCE_STAGE]
            directory = self.root / "versions" / f"v{state['version']:02}"
            if (read_document(directory / "resource-selection.json") != gap
                    or gap["decision"]["resource_id"] is not None
                    or gap["mission_sha256"] != fingerprint(artifacts["mission"])
                    or gap["literature_sha256"] != fingerprint(artifacts["literature"])):
                raise ValueError("Catalog gap differs from the approved study")
            details = {"llm": state["llm"], "max_calls": 1, "max_input_bytes": MAX_INPUT_BYTES,
                       "gap": gap, "resolution_scope": resolution_scope()}
        elif stage == "catalog":
            mission = validate_approved_mission(artifacts["mission"])
            queries = state["queries"] or list(dict.fromkeys(mission.literature_queries))[:4]
            details.update(target="OpenAlex", queries=queries, results_per_query=5, max_http_requests=3 * len(queries))
        elif stage in APPROVALS:
            details["proposal"] = artifacts[APPROVALS[stage]]
            if stage == "review_manuscript":
                details["manuscript_sha256"] = self._draft_digest(state)
        elif stage == IMPLEMENTATION_STAGE:
            request = self._implementation_request(state)
            details = {"llm": state["llm"], "max_calls": 1, "max_input_bytes": MAX_INPUT_BYTES,
                       "protocol_sha256": fingerprint(artifacts["protocol"]),
                       "evaluation_sha256": artifacts["protocol"]["evaluation_sha256"],
                       "capability": request["capability"], "request_sha256": fingerprint(request)}
            if selection := artifacts["protocol"].get("resource_selection"):
                details["resource_id"] = selection["resource"]["id"]
                if "capability_resolution" in selection:
                    details["capability_resolution"] = selection["capability_resolution"]
        elif stage == PROGRAM_STAGE:
            _, _, details, _ = self._program_inputs(state)
            expected_worker = agentic_worker_sha256 if details["capability"] == AGENTIC_CAPABILITY else worker_sha256
            if details["gpu"] and details["worker_sha256"] != expected_worker():
                raise ValueError("Local GPU worker differs from the source approved for the action plan")
        elif stage == SYNTHETIC_STAGE:
            details = self._synthetic_details(state, self._synthetic_campaign(state))
        elif stage in OOD_STAGES:
            protocol, _ = self._ood_execution_inputs(state)
            details = self._ood_details(protocol)
        elif stage == RETRY_GPU_STAGE:
            _, _, details, _ = self._retry_gpu_inputs(state)
            if details["worker_sha256"] != retry_worker_sha256():
                raise ValueError("Local GPU worker differs from the source approved for the action plan")
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
        elif cause == "invalid_implementation" and stage == IMPLEMENTATION_STAGE:
            proposal = {"stage": stage, "reason": state["block"]["message"]}
        elif cause == "incompatible_evaluation" and stage == "review_protocol":
            proposal = {"stage": "protocol", "reason": state["block"]["message"]}
        elif cause == "gpu_snapshot_unavailable" and stage == PROGRAM_STAGE:
            proposal = {"stage": "protocol", "reason": state["block"]["message"]}
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
            "incompatible_evaluation": "Revoir le protocole ou fournir des données d'évaluation compatibles.",
            "no_compatible_resource": "Le catalogue ne couvre pas la question. Fournir un corpus vérifié ou une capacité compatible.",
            "capability_unresolved": "Consulter le prérequis indiqué avant de reprendre la sélection ou le protocole.",
            "invalid_capability_resolution": "Revenir à la sélection ou fournir un corpus vérifié.",
            "gpu_preflight_failed": "Vérifier le worker, l'accès réseau et le cache de modèles sur la cible approuvée.",
            "synthetic_execution_failed": "Examiner le statut du job avant de proposer un nouveau protocole.",
            "invalid_synthetic_report": "Le rapport exporté a changé ou manque. Vérifier les fichiers du pilote.",
            "invalid_gpu_report": "Le rapport GPU a changé ou manque. Vérifier les fichiers du pilote.",
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
                reason = "invalid_gpu_report" if state["workload"] == RETRY_GPU_WORKLOAD else "invalid_synthetic_report"
                block = {"reason": reason, "return_to": ["protocol"], "uncertain": False}
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
        if state["workload"] in {SYNTHETIC_WORKLOAD, RETRY_GPU_WORKLOAD} or state.get("evaluation"):
            result["report"] = report
        if event["action"] == "autonomy_pause":
            result["autonomy_stop"] = event["detail"]["reason"]
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
        if revision.stage == RESOURCE_STAGE and not state.get("auto_resources"):
            raise ValueError("Resource selection is unavailable for this mission")
        supply_resource = (current in {RESOURCE_STAGE, RESOLUTION_STAGE} and revision.stage == "protocol"
                           and (revision.evaluation is not None or revision.workload is not None))
        if ORDER[revision.stage] > ORDER.get(current, ORDER["draft"]) and not supply_resource:
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
            state["evaluation"] = None
            state["auto_resources"] = state["workload"] == "unmapped"
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
            if revision.stage != "protocol" or state["workload"] not in RETRY_WORKLOADS:
                raise ValueError("A custom retry corpus requires revising a catalog retry protocol")
            state["retry_corpus"] = PilotCorpus.model_validate(revision.corpus).model_dump(mode="json")
        if revision.evaluation is not None:
            if revision.stage != "protocol" or state["workload"] != "unmapped":
                raise ValueError("Evaluation changes require revising a newly designed protocol")
            state["evaluation"] = EvaluationSet.model_validate(revision.evaluation).model_dump(mode="json")
            state["auto_resources"] = False
            state["artifacts"].pop(RESOURCE_STAGE, None)
            state["artifacts"].pop(RESOLUTION_STAGE, None)
        if state["workload"] != "unmapped":
            state.pop("evaluation", None)
            state["auto_resources"] = False
            state["artifacts"].pop(RESOURCE_STAGE, None)
            state["artifacts"].pop(RESOLUTION_STAGE, None)
        optional_target = bool(state.get("auto_resources") and (state.get("gpu_config") or revision.gpu_config))
        target = revision.gpu_config.model_dump(mode="json") if revision.gpu_config else state.get("gpu_config")
        config_model = _gpu_config_model(state["workload"], state.get("evaluation"), optional_target=optional_target, target=target)
        if revision.gpu_config is not None:
            if revision.stage != "protocol" or config_model is None:
                raise ValueError("GPU configuration changes require revising a GPU protocol")
            state["gpu_config"] = revision.gpu_config.model_dump(mode="json")
        if config_model is not None:
            if not state.get("gpu_config"):
                raise ValueError("GPU workloads require an explicit GPU configuration")
            config_model.model_validate(state["gpu_config"])
        else:
            state.pop("gpu_config", None)
        if state["workload"] == RETRY_GPU_WORKLOAD and state.get("retry_corpus") is None:
            raise ValueError("retry_gpu requires an explicit catalog corpus")
        prefix = ORDER[revision.stage]
        if state.get("auto_resources") and prefix <= ORDER[RESOURCE_STAGE]:
            state["evaluation"] = None
        state["artifacts"] = {key: value for key, value in state["artifacts"].items() if ORDER[key] < prefix}
        state["feedback"] = {key: value for key, value in state["feedback"].items() if ORDER[key] < prefix}
        state["feedback"][revision.stage] = revision.reason.strip()
        state.pop("block", None)
        state.pop("export", None)
        if revision.stage != "draft" or (state["workload"] != RETRY_GPU_WORKLOAD and not state.get("evaluation")):
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
            state["stage"] = RESOURCE_STAGE if state.get("auto_resources") and not state.get("evaluation") else "protocol"
        elif stage == "review_protocol":
            self._publish(state, artifacts["protocol"], "protocol.json")
            if state["workload"] == SYNTHETIC_WORKLOAD:
                state["stage"] = SYNTHETIC_STAGE
            elif state["workload"] == WORKLOAD:
                source, _ = self._study(state)
                path, _, _ = self._trial_paths(state)
                save_report(source, path)
                state["stage"] = "calibration"
            elif state["workload"] == RETRY_GPU_WORKLOAD:
                source, _, plan = self._retry_gpu_study(state)
                study_path, _, _ = self._trial_paths(state)
                save_report(source, study_path)
                save_report(plan, study_path.with_name("action-plan.json"))
                state["stage"] = RETRY_GPU_STAGE
            elif state["workload"] in OOD_WORKLOADS:
                source, _ = self._ood_study(state)
                path, _, _ = self._trial_paths(state)
                save_report(source, path)
                state["stage"] = GPU_STAGE if state["workload"] == GPU_WORKLOAD else OOD_STAGE
            elif state.get("evaluation"):
                try:
                    if _is_agentic(state["evaluation"]):
                        source = self._program_source(state)
                        _, _, plan = agentic_study.compile_study(source)
                    else:
                        validate_capability(artifacts["protocol"]["document"], EvaluationSet.model_validate(state["evaluation"]))
                except ValueError as error:
                    self._block(state, "incompatible_evaluation", ["protocol", "literature", "mission"])
                    state["block"]["message"] = str(error)[:1000]
                    return
                if _is_agentic(state["evaluation"]):
                    study_path, _, _ = self._trial_paths(state)
                    save_report(source, study_path)
                    save_report(plan, study_path.with_name("action-plan.json"))
                    state["stage"] = PROGRAM_STAGE
                else:
                    state["stage"] = IMPLEMENTATION_STAGE
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
        elif stage == RESOURCE_STAGE:
            catalog = resources_for(state.get("gpu_config"))
            if gate["details"]["catalog"] != [metadata for metadata, _ in catalog.values()]:
                raise PermissionError("Resource catalogue differs from the authorized selection inputs")
            choice = choose_resource(provider, artifacts["mission"], artifacts["literature"], catalog)
            selection = {"decision": choice.model_dump(mode="json"),
                         "mission_sha256": fingerprint(artifacts["mission"]),
                         "literature_sha256": fingerprint(artifacts["literature"])}
            if choice.resource_id is None:
                self._publish(state, selection, "resource-selection.json")
                artifacts[RESOURCE_STAGE] = selection
                state["stage"] = RESOLUTION_STAGE
                return
            metadata, evaluation = catalog[choice.resource_id]
            selection["resource"] = metadata
            self._publish(state, selection, "resource-selection.json")
            state["evaluation"] = evaluation.model_dump(mode="json")
            artifacts[RESOURCE_STAGE] = selection
            state["stage"] = "protocol"
        elif stage == RESOLUTION_STAGE:
            gap = artifacts[RESOURCE_STAGE]
            if gap != gate["details"]["gap"] or resolution_scope() != gate["details"]["resolution_scope"]:
                raise PermissionError("Resolution inputs differ from the authorized scope")
            mission = validate_approved_mission(artifacts["mission"])
            try:
                resolution = propose_resolution(provider, mission.model_dump(), gap)
            except ValueError as error:
                if not provider.received:
                    raise
                self._block(state, "invalid_capability_resolution", [RESOURCE_STAGE, "literature", "mission"])
                state["block"]["message"] = str(error)[:800]
                return
            record = resolution_record(resolution, gap)
            self._publish(state, record, "capability-resolution.json")
            artifacts[RESOLUTION_STAGE] = {"version": state["version"], "document": record}
            if resolution.kind != "local_recipe":
                self._block(state, "capability_unresolved", [RESOURCE_STAGE, "literature", "mission"])
                state["block"]["message"] = resolution.next_requirement
                return
            selection, evaluation = resolve_resource(resolution, gap)
            corpus = evaluation.model_dump(mode="json")
            self._publish(state, selection, "resolved-resource.json")
            self._publish(state, corpus, "resolved-evaluation.json")
            artifacts[RESOURCE_STAGE] = selection
            state["evaluation"] = corpus
            state["stage"] = "protocol"
        elif stage == "protocol":
            mission, _, literature = validate_approved_literature(artifacts["mission"], artifacts["literature"])
            constraints = gate["details"].get("execution_constraints")
            planning_constraints = constraints
            if description := gate["details"].get("resource_description"):
                planning_constraints = {**constraints, "resource_description": description}
            design = ExperimentalDesignPlanner(provider).propose(
                mission, literature, feedback=feedback, execution_constraints=planning_constraints)
            validate_design_against_inputs(design, mission, literature)
            artifacts["protocol"] = {"version": state["version"], "document": design.model_dump(),
                                     "workload": "unmapped"}
            if state.get("evaluation") is not None:
                artifacts["protocol"].update(evaluation_sha256=fingerprint(state["evaluation"]),
                                             execution_constraints=constraints, interpreter_sha256=interpreter_sha256())
                if _is_agentic(state["evaluation"]):
                    artifacts["protocol"].update(gpu_config=state["gpu_config"], worker_sha256=agentic_worker_sha256())
                elif _gpu_config_model(state["workload"], state["evaluation"]) is ProtocolGPUConfig:
                    artifacts["protocol"].update(gpu_config=state["gpu_config"], worker_sha256=worker_sha256())
                if selection := artifacts.get(RESOURCE_STAGE):
                    artifacts["protocol"]["resource_selection"] = selection
            state["stage"] = "review_protocol"
        elif stage == IMPLEMENTATION_STAGE:
            request = self._implementation_request(state)
            if fingerprint(request) != gate["details"]["request_sha256"]:
                raise PermissionError("Code request differs from the authorized inputs")
            try:
                proposal = ExperimentAgent(provider).propose(json.dumps(request, ensure_ascii=False)).model_dump(mode="json")
                self._publish(state, proposal, "candidate.json")
                artifacts[IMPLEMENTATION_STAGE] = {"version": state["version"], "proposal": proposal}
                source = self._program_source(state)
                _, _, _, plan = compile_study(source)
            except ValueError as error:
                if not provider.received:
                    raise
                artifacts.pop(IMPLEMENTATION_STAGE, None)
                self._block(state, "invalid_implementation", [IMPLEMENTATION_STAGE, "protocol"])
                state["block"]["message"] = ("Corriger le programme sans modifier le protocole. " + str(error))[:1000]
                return
            study_path, _, _ = self._trial_paths(state)
            save_report(source, study_path)
            save_report(plan, study_path.with_name("action-plan.json"))
            state["stage"] = PROGRAM_STAGE
        elif stage == PROGRAM_STAGE:
            source, jobs, plan, ledger = self._program_inputs(state)
            if plan != gate["details"]:
                raise PermissionError("Execution plan differs from the authorized jobs")
            if plan["capability"] == AGENTIC_CAPABILITY:
                backend = AgenticGPUBackend(plan)
                try:
                    receipts = backend.prepare_models(jobs)
                except (OSError, RuntimeError, ValueError, TimeoutExpired) as error:
                    self._block(state, "gpu_preflight_failed", ["protocol"])
                    state["block"]["message"] = str(error)[:200] if isinstance(error, FileNotFoundError) else type(error).__name__
                    return
                save_report({row["model_key"]: row for row in receipts}, ledger.with_name("model-preparation.json"))
            elif plan["gpu"]:
                backend = ProtocolGPUBackend(plan)
                try:
                    preparation = backend.prepare_model(jobs[0])
                except (OSError, RuntimeError, ValueError, TimeoutExpired) as error:
                    self._block(state, "gpu_preflight_failed", ["protocol"])
                    state["block"]["message"] = type(error).__name__
                    return
                if not preparation["ready"]:
                    self._block(state, "gpu_snapshot_unavailable", ["protocol"])
                    state["block"]["message"] = "Le snapshot approuvé est incomplet sur l'hôte. Revoir la faisabilité du protocole."
                    return
                save_report(preparation, ledger.with_name("model-preparation.json"))
            else:
                backend = ProtocolCPUBackend(jobs)
            run_program_plan(jobs, ledger, backend)
            self._complete_program(state, source, ledger, decision.model_dump(mode="json"))
        elif stage == SYNTHETIC_STAGE:
            campaign = self._synthetic_campaign(state)
            if gate["details"] != self._synthetic_details(state, campaign):
                raise PermissionError("Synthetic campaign differs from the authorized jobs")
            backend = LocalDockerBackend(execution_authorized=True, transport=self.synthetic_transport)
            manager = DurableJobManager(backend, self._synthetic_directory(state) / "jobs.sqlite3")
            run_campaign(campaign, manager)
            self._complete_synthetic_report(state, campaign, decision.model_dump(mode="json"))
        elif stage in OOD_STAGES:
            protocol, ledger = self._ood_execution_inputs(state)
            if gate["details"] != self._ood_details(protocol):
                raise PermissionError("OOD rehearsal differs from the authorized jobs")
            if isinstance(protocol, OODGPUProtocol):
                run_ood_gpu(protocol, ledger)
            else:
                run_ood_replay(protocol, ledger)
            state["stage"] = "draft"
        elif stage == RETRY_GPU_STAGE:
            protocol, corpus, plan, ledger = self._retry_gpu_inputs(state)
            if gate["details"] != plan:
                raise PermissionError("Catalog GPU action plan differs from the authorized job")
            run_retry_gpu(plan, ledger, approved_plan_sha256=gate["details"]["plan_sha256"])
            self._complete_retry_gpu(state, protocol, corpus, plan, ledger, decision.model_dump(mode="json"))
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
            with capture_model_trace(lambda trace: self.store.record_model_trace(reserved["sha256"], trace)):
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
                SYNTHETIC_STAGE, GPU_STAGE, RETRY_GPU_STAGE, PROGRAM_STAGE, "calibration", "measurement",
            }
            protocol_position = ORDER["protocol"]
            back = list(reversed(STAGES[:min(ORDER.get(APPROVALS.get(stage, stage), protocol_position), protocol_position) + 1]))
            back = [name for name in back if name != RESOLUTION_STAGE
                    and (state.get("auto_resources") or name != RESOURCE_STAGE)]
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
            if stage == "protocol" and state["workload"] in PREDEFINED_PROTOCOL_WORKLOADS:
                try:
                    artifacts = state["artifacts"]
                    if state["workload"] == SYNTHETIC_WORKLOAD:
                        artifacts["protocol"] = self._prepare_synthetic_protocol(state)
                    elif state["workload"] in OOD_WORKLOADS:
                        gpu_config = (OODGPUConfig.model_validate(state["gpu_config"])
                                      if state["workload"] == GPU_WORKLOAD else None)
                        protocol = ood_protocol_for(artifacts["mission"], artifacts["literature"],
                                                    f"{state['id']}-v{state['version']}", gpu_config=gpu_config)
                        artifacts["protocol"] = {"version": state["version"], "document": protocol.model_dump(mode="json")}
                    else:
                        template = state["retry_corpus"] or read_document(ROOT / "examples" / "v9_retry_pilot.json")
                        corpus = PilotCorpus.model_validate(template).model_copy(
                            update={"pilot_id": f"{state['id']}-v{state['version']}"})
                        mission, literature = artifacts["mission"], artifacts["literature"]
                        if state["workload"] == RETRY_GPU_WORKLOAD:
                            config = RetryGPUConfig.model_validate(state["gpu_config"])
                            document = retry_gpu_protocol_for(mission, literature, corpus, config)
                        else:
                            document = protocol_for(mission, literature, corpus)
                        artifacts["protocol"] = {"version": state["version"], "corpus": corpus.model_dump(mode="json"),
                                                 "document": document}
                    state["stage"] = "review_protocol"
                except (ValueError, RuntimeError, OSError, KeyError) as error:
                    self._block(state, type(error).__name__, ["protocol", "literature", "catalog", "mission"])
                self.store.append(event["sha256"], state, action="prepare_protocol")
            elif stage == "draft":
                source, journal, paper = self._trial_paths(state)
                try:
                    options = {}
                    if state["workload"] in OOD_WORKLOADS:
                        expected, _ = self._ood_study(state)
                        options["source_kind"] = GPU_SOURCE_KIND if state["workload"] == GPU_WORKLOAD else OOD_SOURCE_KIND
                    elif state["workload"] == RETRY_GPU_WORKLOAD:
                        self._retry_gpu_inputs(state)
                        expected, _, _ = self._retry_gpu_study(state)
                        options["source_kind"] = RETRY_GPU_SOURCE_KIND
                    elif state.get("evaluation"):
                        expected, _, _, _ = self._program_inputs(state)
                        options["source_kind"] = (AGENTIC_SOURCE_KIND if _is_agentic(state["evaluation"]) else PROGRAM_SOURCE_KIND)
                    else:
                        expected, _ = self._study(state)
                    if read_document(source) != expected:
                        raise ValueError("Pilot source no longer matches its approved protocol")
                    review_id = f"pilot-review-v{state['version']}"
                    result = draft_pilot(source, journal, paper, review_id, state["id"],
                                         memory_path=self._memory_binding(state).path, **options)
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
        if stage in OOD_STAGES and (state["status"] == "running" or state.get("block", {}).get("uncertain")):
            protocol, ledger = self._ood_execution_inputs(state)
            reservation = event
            if event["action"] != "reserve":
                reservation = next(item for item in self.store.history() if item["sha256"] == event["previous_sha256"])
            reserved = reservation["detail"]
            if (reservation["action"] != "reserve" or reserved["gate"]["stage"] != stage
                    or reserved["gate"]["target_sha256"] != reserved["decision"]["target_sha256"]
                    or reserved["gate"]["details"] != self._ood_details(protocol)):
                raise PermissionError("OOD receipts do not match the reserved operation")
            try:
                manager = DurableJobManager(path=ledger, read_only=True)
                if isinstance(protocol, OODGPUProtocol):
                    analyze_ood_gpu(protocol, manager)
                else:
                    analyze_ood_replay(protocol, manager)
            except (FileNotFoundError, RuntimeError) as error:
                raise PermissionError("No complete OOD receipts. No job was replayed.") from error
            state["stage"] = "draft"
        elif stage == RETRY_GPU_STAGE and (state["status"] == "running" or state.get("block", {}).get("uncertain")):
            protocol, corpus, plan, ledger = self._retry_gpu_inputs(state)
            reservation = event
            if event["action"] != "reserve":
                reservation = next(item for item in self.store.history() if item["sha256"] == event["previous_sha256"])
            reserved = reservation["detail"]
            if (reservation["action"] != "reserve" or reserved["gate"]["stage"] != stage
                    or reserved["gate"]["details"] != plan
                    or reserved["gate"]["target_sha256"] != reserved["decision"]["target_sha256"]):
                raise PermissionError("Catalog GPU receipts do not match the reserved operation")
            try:
                self._complete_retry_gpu(state, protocol, corpus, plan, ledger, reserved["decision"])
            except (FileNotFoundError, RuntimeError) as error:
                raise PermissionError("No complete catalog GPU receipt. No job was replayed.") from error
            state.pop("block", None)
            self.store.append(event["sha256"], state, action="recover_retry_gpu_report",
                              detail=decision.model_dump(mode="json"))
            return self._finish_local()
        elif stage == PROGRAM_STAGE and (state["status"] == "running" or state.get("block", {}).get("uncertain")):
            source, _, plan, ledger = self._program_inputs(state)
            reserved = event if event["action"] == "reserve" else next(
                item for item in self.store.history() if item["sha256"] == event["previous_sha256"])
            if (reserved["action"] != "reserve" or reserved["detail"]["gate"]["stage"] != stage
                    or reserved["detail"]["gate"]["details"] != plan
                    or reserved["detail"]["decision"]["target_sha256"] != reserved["detail"]["gate"]["target_sha256"]):
                raise PermissionError("Receipts do not match the reserved operation")
            try:
                self._complete_program(state, source, ledger, reserved["detail"]["decision"])
            except (FileNotFoundError, RuntimeError) as error:
                raise PermissionError("No complete job receipts. No job was replayed.") from error
        elif stage in {"calibration", "measurement"} and state["status"] == "running":
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
                stage == "draft" or (stage == "protocol" and state["workload"] in PREDEFINED_PROTOCOL_WORKLOADS))):
            raise PermissionError("Recovery requires a safe local step or a complete existing pilot receipt")
        state.pop("block", None)
        state["status"] = "waiting"
        self.store.append(event["sha256"], state, action="recover_local", detail=decision.model_dump(mode="json"))
        return self._finish_local()
