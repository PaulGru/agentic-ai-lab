from copy import deepcopy
import unittest

from research.critique.contracts import RevisionDraft, RevisionProposal
from research.critique.revision import build_revision, revision_matrix, verify_revision
from research.critique.rules import deterministic_review
from tests.v86_helpers import revision_draft, sample_bundle


class RevisionTests(unittest.TestCase):
    def test_justified_diff_matrix_and_parent_preservation(self):
        bundle = sample_bundle()
        original = deepcopy(bundle.model_dump(mode="json"))
        report = deterministic_review(bundle)
        proposal = build_revision(bundle, report, revision_draft(bundle), review_id="review")
        self.assertTrue(proposal.reviewed_after_results)
        self.assertEqual(proposal.basis, "design_review_only_unlinked_results")
        self.assertEqual([change.path for change in proposal.changes], ["/analysis_plan/power_limitation"])
        self.assertEqual(revision_matrix(proposal)["execution_status"], "not_executed")
        self.assertEqual(bundle.model_dump(mode="json"), original)
        verify_revision(bundle, report, proposal)

    def test_unknown_issue_missing_justification_invalid_design_and_unchanged(self):
        bundle = sample_bundle()
        report = deterministic_review(bundle)
        for mutation in ("issue", "justification", "budget", "results", "scope", "unchanged"):
            draft = revision_draft(bundle).model_dump(mode="json")
            if mutation == "issue":
                draft["justifications"]["/analysis_plan/power_limitation"]["issue_ids"] = ["invented"]
            elif mutation == "justification":
                draft["justifications"]["/limitations"] = draft["justifications"].pop("/analysis_plan/power_limitation")
            elif mutation == "budget":
                draft["candidate_design"]["resource_limits"]["max_total_llm_calls"] = 1
            elif mutation == "results":
                draft["candidate_design"]["results_status"] = "executed"
            elif mutation == "scope":
                draft["candidate_design"]["objective"] = "A different research objective requiring a new mission."
                draft["justifications"]["/objective"] = draft["justifications"]["/analysis_plan/power_limitation"]
            else:
                draft["candidate_design"] = deepcopy(bundle.context.design_document["design"])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_revision(bundle, report, RevisionDraft.model_validate(draft), review_id="review")

    def test_changed_proposal_or_parent_is_rejected(self):
        bundle = sample_bundle()
        report = deterministic_review(bundle)
        proposal = build_revision(bundle, report, revision_draft(bundle), review_id="review")
        altered = proposal.model_dump(mode="json")
        altered["changes"][0]["before"] = "Wrong previous content"
        with self.assertRaises(ValueError):
            verify_revision(bundle, report, RevisionProposal.model_validate(altered))
        bundle.context.design_document["design"]["title"] = "Changed parent"
        with self.assertRaises(ValueError):
            verify_revision(bundle, report, proposal)

    def test_cycle_requires_link_to_previous_proposal(self):
        bundle = sample_bundle()
        report = deterministic_review(bundle)
        with self.assertRaises(ValueError):
            build_revision(bundle, report, revision_draft(bundle), review_id="review", cycle=2)
        first = build_revision(bundle, report, revision_draft(bundle), review_id="review")
        second = build_revision(bundle, report, revision_draft(bundle), review_id="review", cycle=2,
                                previous_proposal_sha256=first.fingerprint())
        self.assertEqual(second.previous_proposal_sha256, first.fingerprint())


if __name__ == "__main__":
    unittest.main()
