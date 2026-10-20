import unittest

from observability.tracer import Tracer


class TracerTests(unittest.TestCase):
    def test_events_share_same_run_id(self):
        tracer = Tracer(
            print_events=False
        )

        run_id = tracer.start_run()

        tracer.record(
            "route_selected",
            target="general",
        )

        tracer.record(
            "agent_final",
            agent="general",
        )

        self.assertEqual(
            len(tracer.events),
            2,
        )

        for event in tracer.events:
            self.assertEqual(
                event.run_id,
                run_id,
            )


if __name__ == "__main__":
    unittest.main()