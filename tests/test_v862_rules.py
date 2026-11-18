import unittest

from research.critique.contracts import Claim, CriticSuggestions, ReviewReport
from research.critique.evidence import build_evidence
from research.critique.rules import deterministic_review, verify_review, with_suggestions
from tests.v86_helpers import fixture_document, sample_bundle


class CriticRulesTests(unittest.TestCase):
    def test_repeatable_review_separates_observations_and_hypotheses(self):
        bundle = sample_bundle()
        report = deterministic_review(bundle)
        self.assertEqual(report, deterministic_review(bundle))
        self.assertTrue(report.observation_refs)
        self.assertEqual(len(report.hypothesis_refs), 1)
        codes = {issue.issue_id for issue in report.issues}
        self.assertTrue({"unlinked_design", "engineering_scope", "abstract_only", "assumed_independence"} <= codes)
        self.assertFalse(report.ready_for_scientific_inference)
        verify_review(bundle, report)

    def test_small_fixture_preserves_existing_pedagogical_blocker(self):
        doc = fixture_document()
        doc["pairs"] = doc["pairs"][:6]
        report = deterministic_review(build_evidence("pedagogical_paired_fixture", doc))
        self.assertIn("pedagogical_inference_blocked", {i.issue_id for i in report.issues})

    def test_claims_do_not_promote_hypotheses_or_fiction_to_science(self):
        claims = [
            Claim(claim_id="value", kind="observation", statement="A numerical observation",
                  evidence_refs=["/analysis/descriptive/groups/paired_difference/n"], claimed_value=10.0),
            Claim(claim_id="wrong", kind="observation", statement="Incorrect claimed value",
                  evidence_refs=["/analysis/descriptive/groups/paired_difference/n"], claimed_value=11.0),
            Claim(claim_id="absent", kind="observation", statement="Unreferenced measurement"),
            Claim(claim_id="missing", kind="observation", statement="Missing referenced value",
                  evidence_refs=["/analysis/missing"], claimed_value=1.0),
            Claim(claim_id="hyp", kind="hypothesis", statement="An untested hypothesis"),
            Claim(claim_id="science", kind="scientific_conclusion", statement="Model robustness demonstrated"),
        ]
        report = deterministic_review(sample_bundle(False, claims))
        self.assertEqual([x.status for x in report.claim_assessments],
                         ["supported_numeric", "unsupported", "unsupported", "unsupported",
                          "hypothesis_only", "outside_scientific_scope"])

    def test_llm_opinion_cannot_change_facts_and_requires_human_review(self):
        bundle = sample_bundle(False)
        report = deterministic_review(bundle)
        suggestions = CriticSuggestions.model_validate({"issues": [{
            "category": "methodology", "severity": "minor", "finding": "Potential limitation to discuss",
            "recommendation": "Ask a human to assess this limitation",
            "evidence_refs": ["/analysis/validity/cautions"],
        }]})
        assisted = with_suggestions(bundle, report, suggestions)
        self.assertEqual(assisted.issues[:-1], report.issues)
        self.assertTrue(assisted.issues[-1].requires_human_review)
        verify_review(bundle, assisted)
        bad = assisted.model_dump(mode="json")
        bad["issues"][0]["finding"] = "A model replaced a deterministic result"
        with self.assertRaisesRegex(ValueError, "Deterministic"):
            verify_review(bundle, ReviewReport.model_validate(bad))
        bad = suggestions.model_dump(mode="json")
        bad["issues"][0]["evidence_refs"] = ["/unknown"]
        with self.assertRaises(ValueError):
            with_suggestions(bundle, report, CriticSuggestions.model_validate(bad))


if __name__ == "__main__":
    unittest.main()
