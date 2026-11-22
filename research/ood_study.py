"""Bind the bounded OOD profiles to a reviewed brief and literature."""
from analysis.contracts import fingerprint
from execution.ood_gpu import OODGPUProtocol
from execution.ood_replay import OODReplayProtocol, fixture_sha256
from research.experimental_design import validate_approved_literature


REPLAY_STUDY_KIND = "brief_ood_replay"
GPU_STUDY_KIND = "brief_ood_gpu"


def ood_protocol_for(mission, literature, campaign_id, *, gpu_config=None):
    approved, _, _ = validate_approved_literature(mission, literature)
    details = dict(
        campaign_id=campaign_id, requested_question=approved.research_question,
        mission_sha256=fingerprint(mission), literature_sha256=fingerprint(literature),
    )
    if gpu_config is not None:
        return OODGPUProtocol(**details, config=gpu_config)
    return OODReplayProtocol(**details, fixture_sha256=fixture_sha256())


def check_ood_study(source) -> OODReplayProtocol | OODGPUProtocol:
    if (set(source) != {"kind", "metric_name", "mission", "literature", "protocol", "protocol_sha256"}
            or source["kind"] not in {REPLAY_STUDY_KIND, GPU_STUDY_KIND} or source["metric_name"] != "ood.exact_match"):
        raise ValueError("Invalid OOD study source envelope")
    protocol_type = OODGPUProtocol if source["kind"] == GPU_STUDY_KIND else OODReplayProtocol
    protocol = protocol_type.model_validate(source["protocol"])
    expected = ood_protocol_for(source["mission"], source["literature"], protocol.campaign_id,
                                gpu_config=getattr(protocol, "config", None))
    if source["protocol"] != expected.model_dump(mode="json") or source["protocol_sha256"] != expected.fingerprint():
        raise ValueError("OOD protocol differs from the approved mission, literature or fixture")
    return protocol
