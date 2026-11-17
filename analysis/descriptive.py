"""V8.5.2 — deterministic descriptive summaries with explicit observation unit."""
from __future__ import annotations

import statistics

from analysis.contracts import AnalysisDataset, QualityAssessment, verify_export
from analysis.stat_contracts import (
    DescriptiveResult, NumericSummary, PedagogicalPairedFixture,
)


def _summary(values: list[float]) -> NumericSummary:
    if not values:
        raise ValueError("Cannot describe an empty sample")
    return NumericSummary(
        n=len(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        minimum=min(values), maximum=max(values),
        sample_std_dev=statistics.stdev(values) if len(values) > 1 else None,
    )


def describe_verified_pilot(dataset: AnalysisDataset, quality: QualityAssessment) -> DescriptiveResult:
    """One aggregate per run; do not infer variance from the input value count."""
    verify_export(dataset, quality)
    obs = dataset.observations
    groups = {r.run_id: _summary([r.value]) for r in obs}
    return DescriptiveResult(
        kind="job_aggregates", source_sha256=dataset.fingerprint(),
        observation_unit="one_job_level_aggregate",
        groups=groups,
        mean_difference_variant_minus_baseline=obs[-1].value - obs[0].value,
        cautions=[
            "The contrast is arithmetic only; run order is not a randomized condition assignment",
            "input_value_count is not the number of independently replicated experiments",
            "No standard deviation or confidence interval is estimated across one run-level aggregate",
        ],
    )


def describe_paired_fixture(fixture: PedagogicalPairedFixture) -> DescriptiveResult:
    base = [p.baseline for p in fixture.pairs]
    variant = [p.variant for p in fixture.pairs]
    differences = [v - b for b, v in zip(base, variant)]
    return DescriptiveResult(
        kind="paired_fixture", source_sha256=fixture.fingerprint(),
        observation_unit="synthetic_paired_task",
        n_independent_pairs=len(fixture.pairs),
        groups={
            "baseline": _summary(base),
            "variant": _summary(variant),
            "paired_difference": _summary(differences),
        },
        mean_difference_variant_minus_baseline=statistics.fmean(differences),
        cautions=["Fixture values are deliberately synthetic; independent sampling cannot be demonstrated from JSON"],
    )
