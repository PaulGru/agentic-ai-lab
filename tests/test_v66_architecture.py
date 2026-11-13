import unittest

from api.core import make_config
from api.routes.research import pipeline_config, research_config
from workflows.legacy import build_graph


class V66ArchitectureTests(unittest.TestCase):
    def test_checkpoint_namespaces_unchanged(self):
        self.assertEqual(
            research_config("abc")["configurable"]["thread_id"],
            "v6-research:abc",
        )
        self.assertEqual(
            pipeline_config("abc")["configurable"]["thread_id"],
            "v64-pipeline:abc",
        )
        self.assertEqual(
            make_config("abc")["configurable"]["thread_id"],
            "abc",
        )

    def test_legacy_factory_not_imported_from_demo(self):
        self.assertEqual(build_graph.__module__, "workflows.legacy")
