import unittest
from datetime import datetime, timezone

from railforge.api import RailForgeService
from railforge.core.versioning import VersionConflict
from railforge.events.cursor import CursorConsumer, FeedEvent
from railforge.yard.track_rearrangement import (
    YardCut, YardSnapshot, YardTrack, rearrange,
)

U = timezone.utc
DEST = frozenset({"D1", "D2", "DX", "DY"})


def snap(tracks, placement, cuts, switches=None, track_switches=None, incompatible=frozenset()):
    ids = [t.track_id for t in tracks]
    return YardSnapshot(
        tracks=tuple(tracks),
        placement={tid: tuple(placement.get(tid, ())) for tid in ids},
        cuts=cuts,
        switches=switches or {},
        track_switches=track_switches or {},
        incompatible_groups=incompatible,
    )


class RearrangementDomainTests(unittest.TestCase):
    def test_direct_reassignment_respects_capacity_and_priority(self):
        tracks = (
            YardTrack("T1", 10, DEST),
            YardTrack("T2", 10, DEST),
        )
        committed = {
            "c0": YardCut("c0", "OLD", "D1", 6, 1),
        }
        c_high = YardCut("c1", "N1", "D1", 4, 9)
        c_low = YardCut("c2", "N2", "D1", 7, 1)
        s = snap(tracks, {"T1": ["c0"]}, committed)
        plan = rearrange(s, [c_high, c_low], "crew-A")
        self.assertEqual(plan.outcome, "rearranged")
        # T1 has 6 used: c1 (4) fits, c2 (7) does not
        self.assertEqual(sorted(plan.placement["T1"]), ["c0", "c1"])
        self.assertEqual(plan.placement["T2"], ["c2"])

    def test_hazmat_pairwise_segregation_forces_move(self):
        tracks = (
            YardTrack("T1", 10, DEST),
            YardTrack("T2", 10, DEST),
        )
        resident = YardCut("r1", "OLD", "D1", 2, 1, hazard_group="B")
        incoming = YardCut("n1", "N1", "D1", 2, 9, hazard_group="A")
        s = snap(tracks, {"T1": ["r1"], "T2": []}, {"r1": resident},
                 incompatible=frozenset({frozenset({"A", "B"})}))
        # n1 cannot share T1 with the B-group resident, so it must take T2
        plan = rearrange(s, [incoming], "crew-A")
        self.assertEqual(plan.outcome, "rearranged")
        self.assertEqual(plan.placement["T2"], ["n1"])
        self.assertNotIn("n1", plan.placement["T1"])

    def test_hazmat_track_ban_forces_alternative_track(self):
        tracks = (
            YardTrack("T1", 10, DEST, forbidden_hazard=frozenset({"B"})),
            YardTrack("T2", 10, DEST),
        )
        s = snap(tracks, {"T1": ["r1"], "T2": []},
                 {"r1": YardCut("r1", "OLD", "D1", 2, 1)})
        plan = rearrange(s, [YardCut("n1", "N1", "D1", 2, 9, hazard_group="B")],
                         "crew-A")
        self.assertEqual(plan.placement["T2"], ["n1"])
        self.assertEqual(plan.placement["T1"], ["r1"])

    def test_local_swap_with_lower_priority_committed_resident(self):
        tracks = (
            YardTrack("T1", 8, DEST),
            YardTrack("T2", 8, DEST),
        )
        # T2 is full and pairs the committed n1 with a B-group resident, so
        # re-declaring n1 as hazmat A makes staying put pairwise-incompatible.
        # T1 holds a low-priority B resident with just enough room for n1; a
        # direct entry is blocked by the A/B pair, so b1 is displaced to T2.
        current = YardCut("n1", "N1", "DX", 3, 5)
        b1 = YardCut("b1", "OLD", "DY", 5, 1, hazard_group="B")
        z1 = YardCut("z1", "OLD", "DY", 3, 1, hazard_group="B")
        s = YardSnapshot(
            tracks=tracks,
            placement={"T1": ("b1",), "T2": ("n1", "z1")},
            cuts={"b1": b1, "z1": z1, "n1": current},
            switches={},
            track_switches={},
            incompatible_groups=frozenset({frozenset({"A", "B"})}),
        )
        requested = YardCut("n1", "N1", "DX", 3, 5, hazard_group="A")
        plan = rearrange(s, [requested], "crew-A")
        self.assertEqual(plan.outcome, "local_swaps")
        self.assertEqual(plan.swaps[0].incoming_cut, "n1")
        self.assertEqual(plan.swaps[0].displaced_cut, "b1")
        self.assertEqual(plan.placement["T1"], ["n1"])
        self.assertEqual(sorted(plan.placement["T2"]), ["b1", "z1"])
        swapped_moves = {m.cut_id: m for m in plan.moves if m.swapped}
        self.assertIn("n1", swapped_moves)
        self.assertIn("b1", swapped_moves)
        self.assertEqual(swapped_moves["b1"].to_track, "T2")

    def test_swap_does_not_preempt_higher_priority_and_rolls_order_back(self):
        tracks = (
            YardTrack("T1", 8, DEST),
            YardTrack("T2", 8, DEST),
        )
        # n1's current track pairs it with B traffic, so its new A declaration
        # cannot stay. T1 is full of higher-priority traffic and displacement
        # is forbidden, so the whole order rolls back to the committed state.
        current = YardCut("n1", "N1", "DX", 3, 5)
        z1 = YardCut("z1", "OLD", "DY", 5, 1, hazard_group="B")
        f1 = YardCut("f1", "OLD", "DY", 8, 9)
        s = YardSnapshot(
            tracks=tracks,
            placement={"T1": ("f1",), "T2": ("n1", "z1")},
            cuts={"f1": f1, "z1": z1, "n1": current},
            switches={}, track_switches={},
            incompatible_groups=frozenset({frozenset({"A", "B"})}),
        )
        requested = YardCut("n1", "N1", "DX", 3, 5, hazard_group="A")
        plan = rearrange(s, [requested], "crew-A")
        self.assertEqual(plan.outcome, "order_rolled_back")
        self.assertEqual(plan.rolled_back_orders, ["N1"])
        self.assertEqual(plan.placement, {"T1": ["f1"], "T2": ["n1", "z1"]})
        self.assertEqual(plan.moves, [])
        self.assertTrue(plan.rejections)
        reasons = plan.rejections[0].reasons
        self.assertIn("capacity", reasons)
        self.assertTrue(any(r.startswith("hazard_pair") for r in reasons))

    def test_switch_lock_blocks_entry_and_triggers_rollback(self):
        tracks = (
            YardTrack("T1", 10, DEST),
            YardTrack("T2", 10, DEST),
        )
        resident = YardCut("r1", "OLD", "D1", 9, 9)
        incoming = YardCut("n1", "N1", "D1", 2, 5)
        s = snap(tracks, {"T1": ["r1"], "T2": []},
                 {"r1": resident},
                 switches={"SW2": "signal-box"},
                 track_switches={"T2": frozenset({"SW2"})})
        plan = rearrange(s, [incoming], "crew-A")
        self.assertEqual(plan.outcome, "order_rolled_back")
        self.assertEqual(plan.placement, {"T1": ["r1"], "T2": []})
        self.assertIn("switch_locked:T2", plan.rejections[0].reasons)
        # owner holding the lock may enter
        plan2 = rearrange(s, [incoming], "signal-box")
        self.assertEqual(plan2.outcome, "rearranged")
        self.assertEqual(plan2.placement["T2"], ["n1"])

    def test_rollback_undoes_earlier_moves_of_same_order(self):
        tracks = (
            YardTrack("T1", 10, DEST),
            YardTrack("T2", 6, DEST),
        )
        committed = {
            "a1": YardCut("a1", "N1", "D1", 2, 5),
            "b1": YardCut("b1", "OLD", "D1", 8, 1),
        }
        # a1 committed on T1 (2 free, cannot host c1); T2 (6) cannot host c1
        # either. a1 can move to T2 first, but then c1 still has nowhere to go,
        # so the whole N1 order must roll back including a1's move.
        requested = [
            YardCut("a1", "N1", "D1", 2, 5),
            YardCut("c1", "N1", "D1", 9, 4),
        ]
        s = snap(tracks, {"T1": ["a1", "b1"], "T2": []}, committed)
        plan = rearrange(s, requested, "crew-A")
        self.assertEqual(plan.outcome, "order_rolled_back")
        self.assertEqual(sorted(plan.placement["T1"]), ["a1", "b1"])
        self.assertEqual(plan.placement["T2"], [])
        self.assertEqual(plan.moves, [])


class RearrangementWorkflowTests(unittest.TestCase):
    def _snapshot(self):
        tracks = (YardTrack("T1", 10, frozenset({"D1"})),
                  YardTrack("T2", 10, frozenset({"D1"})))
        return snap(tracks, {"T1": ["r1"]},
                    {"r1": YardCut("r1", "OLD", "D1", 3, 1)})

    def test_commits_event_with_contiguous_versions(self):
        service = RailForgeService()
        at = datetime(2026, 1, 1, 8, tzinfo=U)
        s = self._snapshot()
        r1 = service.features.reassign_yard_tracks(
            "op-1", s, [YardCut("n1", "N1", "D1", 2, 9)], "crew-A", at)
        self.assertEqual(r1.event.payload["status"], "committed")
        self.assertEqual(r1.event.payload["version"], 1)
        self.assertEqual(r1.event.payload["base_version"], 0)
        self.assertEqual(r1.result.outcome, "rearranged")

        r2 = service.features.reassign_yard_tracks(
            "op-2", s, [YardCut("n2", "N2", "D1", 2, 9)], "crew-A", at)
        self.assertEqual(r2.event.payload["version"], 2)
        self.assertEqual(r2.event.payload["base_version"], 1)

        self.assertTrue(service.workflow().stream.verify_chain())
        self.assertTrue(service.workflow().trail.verify())

    def test_stale_base_version_rejects_without_publishing(self):
        service = RailForgeService()
        at = datetime(2026, 1, 1, 8, tzinfo=U)
        s = self._snapshot()
        first = service.features.reassign_yard_tracks(
            "op-1", s, [YardCut("n1", "N1", "D1", 2, 9)], "crew-A", at)
        self.assertEqual(first.version, 1)
        head_before = service.workflow().stream.head

        with self.assertRaises(VersionConflict):
            service.features.reassign_yard_tracks(
                "op-stale", s, [YardCut("n2", "N2", "D1", 2, 9)], "crew-A", at,
                expected_version=0)
        # no event published: consumers see no gap
        self.assertEqual(service.workflow().stream.head, head_before)
        self.assertTrue(service.workflow().stream.verify_chain())

        fresh = service.features.reassign_yard_tracks(
            "op-3", s, [YardCut("n3", "N3", "D1", 2, 9)], "crew-A", at,
            expected_version=1)
        self.assertEqual(fresh.version, 2)

    def test_idempotent_retry_returns_same_receipt(self):
        service = RailForgeService()
        at = datetime(2026, 1, 1, 8, tzinfo=U)
        s = self._snapshot()
        kwargs = dict(requested=[YardCut("n1", "N1", "D1", 2, 9)], owner="crew-A", at=at)
        r1 = service.features.reassign_yard_tracks("op-idem", s, **kwargs)
        r2 = service.features.reassign_yard_tracks("op-idem", s, **kwargs)
        self.assertIs(r1, r2)

    def test_consumer_sees_unbroken_sequence_on_dedicated_topic(self):
        service = RailForgeService()
        at = datetime(2026, 1, 1, 8, tzinfo=U)
        s = self._snapshot()
        for i in range(3):
            service.features.reassign_yard_tracks(
                f"op-{i}", s, [YardCut(f"n{i}", f"N{i}", "D1", 1, 9)],
                "crew-A", at)
        page = service.workflow().stream.page(
            0, limit=10, topics=["feature.yard.tracks.reassigned"])
        consumer = CursorConsumer()
        versions = []
        for e in page.events:
            accepted = consumer.accept(FeedEvent("yard:crew-A", e.sequence,
                                                 e.event_id, e.payload))
            self.assertTrue(accepted)
            versions.append(e.payload["version"])
        self.assertEqual(versions, [1, 2, 3])

    def test_rollback_plan_publishes_committed_consistent_state(self):
        service = RailForgeService()
        at = datetime(2026, 1, 1, 8, tzinfo=U)
        tracks = (YardTrack("T1", 10, DEST), YardTrack("T2", 10, DEST))
        resident = YardCut("r1", "OLD", "D1", 9, 9)
        s = snap(tracks, {"T1": ["r1"], "T2": []}, {"r1": resident},
                 switches={"SW2": "signal-box"},
                 track_switches={"T2": frozenset({"SW2"})})
        receipt = service.features.reassign_yard_tracks(
            "op-rb", s, [YardCut("n1", "N1", "D1", 2, 5)], "crew-A", at)
        payload = receipt.event.payload
        self.assertEqual(payload["status"], "committed")
        self.assertEqual(payload["result"]["outcome"], "order_rolled_back")
        self.assertEqual(payload["result"]["placement"],
                         {"T1": ["r1"], "T2": []})
        self.assertTrue(service.workflow().stream.verify_chain())


if __name__ == "__main__":
    unittest.main()
