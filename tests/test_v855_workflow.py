"""V8.5.5 LangGraph read-only analysis; skipped when not installed."""
import json
import unittest
from pathlib import Path

try:
    from langgraph.checkpoint.memory import MemorySaver
    from analysis.graph import AnalysisWorkflow
except ImportError:
    MemorySaver = None

from analysis.pipeline import build_fixture_analysis
from analysis.stat_contracts import AnalysisReport, PedagogicalPairedFixture

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "v85_paired_fixture.json"


@unittest.skipUnless(MemorySaver is not None, "LangGraph not installed in this environment")
class AnalysisWorkflowTests(unittest.TestCase):
    def test_fixture_graph_matches_direct_pipeline(self):
        fixture = PedagogicalPairedFixture.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
        graph = AnalysisWorkflow().build(MemorySaver())
        result = graph.invoke({"source_kind": "pedagogical_paired_fixture",
                               "fixture_document": fixture.model_dump(mode="json")},
                              config={"configurable": {"thread_id": "v855-fixture"}})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(AnalysisReport.model_validate(result["report_document"]),
                         build_fixture_analysis(fixture))

    def test_pilot_without_ledger_is_rejected(self):
        graph = AnalysisWorkflow().build(MemorySaver())
        with self.assertRaises(ValueError):
            graph.invoke({"source_kind": "verified_pilot_aggregates", "protocol_document": {}},
                         config={"configurable": {"thread_id": "v855-no-ledger"}})

    def test_invalid_fixture_rejected_before_descriptive(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        fixture["pairs"][0]["baseline"] = float("nan")
        graph = AnalysisWorkflow().build(MemorySaver())
        with self.assertRaises(ValueError):
            graph.invoke({"source_kind": "pedagogical_paired_fixture",
                          "fixture_document": fixture},
                         config={"configurable": {"thread_id": "v855-bad"}})

    def test_graph_blocks_small_fixture_inference(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        fixture["pairs"] = fixture["pairs"][:6]
        graph = AnalysisWorkflow().build(MemorySaver())
        result = graph.invoke({"source_kind": "pedagogical_paired_fixture",
                               "fixture_document": fixture},
                              config={"configurable": {"thread_id": "v855-small"}})
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["inference_document"])
        self.assertFalse(result["report_document"]["validity"]["ready_for_pedagogical_inference"])


if __name__ == "__main__":
    unittest.main()
