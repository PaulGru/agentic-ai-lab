"""V8.5.4 — fail-closed validity gate; scientific inference not yet enabled."""
from __future__ import annotations

import math
import statistics

from analysis.contracts import AnalysisDataset, QualityAssessment, verify_export
from analysis.stat_contracts import PedagogicalPairedFixture, ValidityReport


def assess_verified_pilot(dataset: AnalysisDataset, quality: QualityAssessment) -> ValidityReport:
    verify_export(dataset, quality)
    return ValidityReport(
        source_kind="verified_pilot_aggregates",
        source_sha256=dataset.fingerprint(),
        ready_for_pedagogical_inference=False,
        blockers=list(quality.blockers) + ["Only one aggregate per run; cannot estimate sampling uncertainty"],
        cautions=list(quality.warnings),
    )


def assess_paired_fixture(fixture: PedagogicalPairedFixture) -> ValidityReport:
    differences = [p.variant - p.baseline for p in fixture.pairs]
    blockers = []
    if len(differences) < 8:
        blockers.append("At least 8 synthetic task-pairs are required for this demonstration")
    # Floating point scale-aware check; exact identical offsets offer no empirical dispersion.
    scale = max(1., *(abs(d) for d in differences))
    if statistics.pstdev(differences) <= 1e-12 * scale:
        blockers.append("Paired differences have no measurable variation; standardized effect undefined")
    if any(not math.isfinite(d) for d in differences):
        blockers.append("Non-finite paired differences")
    return ValidityReport(
        source_kind="pedagogical_paired_fixture",
        source_sha256=fixture.fingerprint(),
        ready_for_pedagogical_inference=not blockers,
        blockers=blockers,
        cautions=[
            "Synthetic fixture only; never evidence of real-world performance",
            "Independence and sign symmetry are assumptions, not established by inspecting JSON",
            "Exactly specified sign-flip test is exploratory; bootstrap percentile CI is illustrative",
            "Do not interpret a small p-value as causal evidence or a published result",
        ],
    )
