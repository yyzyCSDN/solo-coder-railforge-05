import unittest
from datetime import datetime, timezone
from railforge.api import RailForgeService
from railforge.yard.retrack import (RetrackCut, RetrackTrack, Swap, occupancy, replan, violations)

U = timezone.utc
AT = datetime(2026, 3, 1, 8, tzinfo=U)


def cut(cid, dest="D1", wagons=5, priority=0, hazard=None):
    return RetrackCut(cid, dest, wagons, priority, hazard)


def track(tid, cap=10, dests=("D1",), forbidden=(), locked=False):
    return RetrackTrack(tid, cap, frozenset(dests), frozenset(forbidden), locked)


class RetrackDomainTests(unittest.TestCase):
    def test_direct_replan_keeps_existing_and_respects_rules(self):
        tracks = [track("T1"), track("T2", dests=("D1", "D2"), forbidden=("HAZ-A",))]
        cuts = [cut("C1"), cut("C2", hazard="HAZ-A"), cut("C3", dest="D2")]
        plan = replan(cuts, tracks, {"C1": "T1"})
        self.assertEqual(plan.status, "replanned")
        self.assertEqual(plan.assignment["C1"], "T1")
        self.assertEqual(plan.assignment["C2"], "T1")  # T2 forbids HAZ-A
        self.assertEqual(plan.assignment["C3"], "T2")  # T1 does not serve D2
        self.assertEqual(plan.moved, ())
        self.assertEqual(violations(plan.assignment, cuts, tracks), ())

    def test_hazard_incompatibility_separates_cuts(self):
        conflicts = frozenset({frozenset({"HAZ-A", "HAZ-B"})})
        tracks = [track("T1"), track("T2")]
        cuts = [cut("C1", hazard="HAZ-A"), cut("C2", hazard="HAZ-B")]
        plan = replan(cuts, tracks, {"C1": "T1"}, conflicts)
        self.assertEqual(plan.assignment["C2"], "T2")
        bad = violations({"C1": "T1", "C2": "T1"}, cuts, tracks, conflicts)
        self.assertIn("hazard-conflict:T1:C1:C2", bad)

    def test_invalid_existing_cut_is_unseated_by_priority(self):
        tracks = [track("T1"), track("T2")]
        cuts = [cut("C1", wagons=6, priority=1), cut("C2", wagons=6, priority=9)]
        plan = replan(cuts, tracks, {"C1": "T1", "C2": "T1"})  # 12 > 10 overflow
        self.assertEqual(plan.status, "replanned")
        self.assertEqual(plan.assignment, {"C1": "T2", "C2": "T1"})
        self.assertEqual(plan.moved, ("C1",))

    def test_new_hazard_rule_unseats_lower_priority_cut(self):
        conflicts = frozenset({frozenset({"HAZ-A", "HAZ-B"})})
        tracks = [track("T1"), track("T2")]
        cuts = [cut("C1", priority=1, hazard="HAZ-A"), cut("C2", priority=9, hazard="HAZ-B")]
        plan = replan(cuts, tracks, {"C1": "T1", "C2": "T1"}, conflicts)
        self.assertEqual(plan.assignment, {"C1": "T2", "C2": "T1"})
        self.assertEqual(plan.moved, ("C1",))

    def test_local_swap_displaces_lower_priority_cut(self):
        tracks = [track("T1"), track("T2")]
        cuts = [cut("C1", wagons=4, priority=1), cut("C2", wagons=6, priority=1),
                cut("C3", wagons=8, priority=5)]
        plan = replan(cuts, tracks, {"C1": "T1", "C2": "T2"})
        self.assertEqual(plan.status, "replanned")
        self.assertEqual(plan.assignment, {"C1": "T2", "C2": "T2", "C3": "T1"})
        self.assertEqual(plan.swaps, (Swap("C3", "C1", "T1", "T2"),))
        self.assertEqual(plan.moved, ("C1",))
        self.assertEqual(violations(plan.assignment, cuts, tracks), ())

    def test_swap_never_displaces_equal_or_higher_priority(self):
        tracks = [track("T1")]
        cuts = [cut("C1", wagons=6, priority=5), cut("C2", wagons=8, priority=5)]
        plan = replan(cuts, tracks, {"C1": "T1"})
        self.assertEqual(plan.status, "rolled_back")
        self.assertEqual(plan.assignment, {"C1": "T1"})
        self.assertEqual(plan.unplaced, ("C2",))

    def test_switch_locked_track_neither_gains_nor_loses(self):
        tracks = [track("T1", locked=True), track("T2", cap=4)]
        cuts = [cut("C1"), cut("C2")]
        plan = replan(cuts, tracks, {"C1": "T1"})
        self.assertEqual(plan.status, "rolled_back")
        self.assertEqual(plan.assignment, {"C1": "T1"})
        self.assertEqual(plan.unplaced, ("C2",))

    def test_locked_track_infeasible_raises(self):
        tracks = [track("T1", cap=4, locked=True)]
        with self.assertRaises(ValueError):
            replan([cut("C1")], tracks, {"C1": "T1"})

    def test_rollback_restores_original_assignment(self):
        tracks = [track("T1"), track("T2")]
        cuts = [cut("C1"), cut("C9", wagons=20, priority=9)]
        plan = replan(cuts, tracks, {"C1": "T1"})
        self.assertEqual(plan.status, "rolled_back")
        self.assertEqual(plan.assignment, {"C1": "T1"})
        self.assertEqual(plan.moved, ())
        self.assertEqual(plan.swaps, ())
        self.assertEqual(plan.unplaced, ("C9",))

    def test_departed_cuts_and_unknown_tracks_are_dropped(self):
        tracks = [track("T1")]
        plan = replan([cut("C1")], tracks, {"C1": "T1", "GONE": "T1", "C2": "TX"})
        self.assertEqual(plan.assignment, {"C1": "T1"})

    def test_occupancy_and_input_validation(self):
        cuts = [cut("C1", wagons=3), cut("C2", wagons=4)]
        self.assertEqual(occupancy({"C1": "T1", "C2": "T1"}, cuts), {"T1": 7})
        with self.assertRaises(ValueError):
            replan([cut("C1"), cut("C1")], [track("T1")], {})
        with self.assertRaises(ValueError):
            replan([cut("C1", wagons=0)], [track("T1")], {})
        with self.assertRaises(ValueError):
            replan([cut("C1")], [track("T1"), track("T1")], {})


class RetrackWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.service = RailForgeService()
        self.control = self.service.workflow()
        self.tracks = [track("T1"), track("T2")]

    def test_consumers_see_consistent_version_sequence(self):
        c1, c2 = cut("C1", priority=5), cut("C2", wagons=2)
        r1 = self.control.retrack_yard("op-1", [c1, c2], self.tracks, {}, frozenset(), "YARD-1", AT)
        self.assertEqual(r1.result.status, "replanned")
        oversized = cut("C3", wagons=20, priority=9)
        r2 = self.control.retrack_yard("op-2", [c1, c2, oversized], self.tracks,
                                       r1.result.assignment, frozenset(), "YARD-1", AT)
        self.assertEqual(r2.result.status, "rolled_back")
        self.assertEqual(r2.result.assignment, r1.result.assignment)
        c4, c5 = cut("C4", wagons=4, priority=1), cut("C5", wagons=8, priority=9)
        current = dict(r1.result.assignment, C4="T2")
        r3 = self.control.retrack_yard("op-3", [c1, c2, c4, c5], self.tracks,
                                       current, frozenset(), "YARD-1", AT)
        self.assertEqual(r3.result.status, "replanned")
        self.assertEqual(r3.result.swaps, (Swap("C5", "C1", "T1", "T2"),))
        self.assertEqual((r1.version, r2.version, r3.version), (1, 2, 3))
        page = self.control.stream.page(0, topics=["yard.retracked"])
        self.assertFalse(page.gap)
        self.assertEqual([e.payload["version"] for e in page.events], [1, 2, 3])
        self.assertEqual([e.payload["result"]["status"] for e in page.events],
                         ["replanned", "rolled_back", "replanned"])
        follow = self.control.stream.page(page.next_cursor, topics=["yard.retracked"])
        self.assertEqual(follow.events, ())
        self.assertEqual(self.control.store.read("YARD-1").version, 3)
        self.assertTrue(self.control.stream.verify_chain())
        self.assertTrue(self.control.trail.verify())

    def test_retry_is_idempotent_and_publishes_no_duplicate(self):
        c1 = cut("C1")
        first = self.control.retrack_yard("op-9", [c1], self.tracks, {}, frozenset(), "YARD-2", AT)
        head = self.control.stream.head
        again = self.control.retrack_yard("op-9", [c1], self.tracks, {}, frozenset(), "YARD-2", AT)
        self.assertIs(first, again)
        self.assertEqual(self.control.stream.head, head)

    def test_feature_entrypoint_commits_replanned_and_rolled_back(self):
        c1, c2 = cut("C1", priority=5), cut("C2", wagons=2)
        ok = self.service.features.retrack_yard_dynamic("f-1", [c1, c2], self.tracks, {}, frozenset(), AT)
        self.assertEqual(ok.event.topic, "feature.yard.retracked")
        self.assertEqual(ok.result["status"], "replanned")
        blocked = self.service.features.retrack_yard_dynamic(
            "f-2", [c1, c2, cut("C3", wagons=20, priority=9)], self.tracks,
            ok.result["assignment"], frozenset(), AT)
        self.assertEqual(blocked.result["status"], "rolled_back")
        self.assertEqual(blocked.result["unplaced"], ["C3"])
        events = self.control.stream.page(0, topics=["feature.yard.retracked"]).events
        self.assertEqual([e.sequence for e in events], [1, 2])
        self.assertEqual([e.payload["version"] for e in events], [1, 1])  # per-subject versions
        self.assertTrue(self.control.stream.verify_chain())


if __name__ == "__main__":
    unittest.main()
