"""V8.5.3 — deterministic, exact *pedagogical* paired sign-flip analysis.

No SciPy required. The test enumerates all 2^n possible signs for n <= 16.
Sign-flip inference requires an exchangeability/symmetry assumption and does
not establish it. Bootstrap interval uses fixed RNG seed; not publication CI.
"""
from __future__ import annotations

import itertools
import random
import statistics

from analysis.stat_contracts import InferenceResult, PedagogicalPairedFixture, ValidityReport
from analysis.validity import assess_paired_fixture


def _percentile(sorted_values: list[float], q: float) -> float:
    at = (len(sorted_values) - 1) * q
    lower = int(at)
    frac = at - lower
    if lower == len(sorted_values) - 1:
        return sorted_values[lower]
    return sorted_values[lower] + frac * (sorted_values[lower + 1] - sorted_values[lower])


def infer_paired_fixture(
    fixture: PedagogicalPairedFixture,
    *, validity: ValidityReport | None = None,
    bootstrap_seed: int = 42,
    bootstrap_resamples: int = 2000,
) -> InferenceResult:
    if type(bootstrap_seed) is not int or not (0 <= bootstrap_seed <= 2**32 - 1):
        raise ValueError("bootstrap_seed must be a nonnegative 32-bit integer")
    if type(bootstrap_resamples) is not int or not (1000 <= bootstrap_resamples <= 5000):
        raise ValueError("bootstrap_resamples must be between 1000 and 5000")
    actual = assess_paired_fixture(fixture)
    # Check current values rather than trust an LLM's or caller's asserted readiness.
    if validity is not None and validity != actual:
        raise ValueError("Stale or forged validity report")
    if not actual.ready_for_pedagogical_inference:
        raise ValueError("Pedagogical inference blocked: " + "; ".join(actual.blockers))
    diffs = [p.variant - p.baseline for p in fixture.pairs]
    n = len(diffs)
    observed = abs(sum(diffs))
    epsilon = 1e-12 * max(1.0, observed)
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=n):
        if abs(sum(s * d for s, d in zip(signs, diffs))) + epsilon >= observed:
            extreme += 1
    permutations = 1 << n
    p_value = extreme / permutations
    rng = random.Random(bootstrap_seed)
    samples = sorted(statistics.fmean([rng.choice(diffs) for _ in range(n)])
                     for _ in range(bootstrap_resamples))
    mean = statistics.fmean(diffs)
    deviation = statistics.stdev(diffs)
    return InferenceResult(
        source_sha256=fixture.fingerprint(), n_pairs=n,
        mean_difference=mean,
        exact_two_sided_p=p_value, sign_flips_enumerated=permutations,
        bootstrap_percentile_ci_95=[_percentile(samples, .025), _percentile(samples, .975)],
        bootstrap_resamples=bootstrap_resamples, bootstrap_seed=bootstrap_seed,
        paired_standardized_effect_dz=mean / deviation,
        caveats=actual.cautions,
    )
