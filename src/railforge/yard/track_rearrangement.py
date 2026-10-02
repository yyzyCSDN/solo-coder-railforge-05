from __future__ import annotations

from dataclasses import dataclass

# Dynamic classification-yard track rearrangement.
#
# A rearrangement reallocates requested cuts across classification tracks while
# honouring five constraints at once:
#   1. dangerous-goods incompatibility (track bans and pairwise hazard groups);
#   2. per-track wagon capacity;
#   3. switch locking (a route held by another owner cannot be entered);
#   4. existing ("committed") cuts already sitting in the yard;
#   5. cut priority (higher priority is served first and may pre-empt).
#
# When a cut cannot be placed directly the planner first tries a LOCAL SWAP with
# a static, lower-priority resident; if that is impossible it rolls the whole
# order back to its committed positions. Rollback replays the order's own
# actions in reverse, so every surviving placement was validated against the
# restored cuts and the resulting plan always satisfies every hard invariant.


@dataclass(frozen=True)
class YardCut:
    cut_id: str
    order_id: str
    destination: str
    wagons: int
    priority: int
    hazard_group: str | None = None


@dataclass(frozen=True)
class YardTrack:
    track_id: str
    capacity: int
    destinations: frozenset[str]
    forbidden_hazard: frozenset[str] = frozenset()


@dataclass(frozen=True)
class YardSnapshot:
    tracks: tuple[YardTrack, ...]
    # committed placement: track_id -> ordered cut ids
    placement: dict[str, tuple[str, ...]]
    cuts: dict[str, YardCut]
    # switch_id -> owner holding the lock (None = free to set)
    switches: dict[str, str | None]
    # switch ids that must be set to enter each track
    track_switches: dict[str, frozenset[str]]
    # hazard group pairs that may never share a track
    incompatible_groups: frozenset[frozenset[str]] = frozenset()


@dataclass(frozen=True)
class TrackMove:
    cut_id: str
    order_id: str
    from_track: str | None
    to_track: str | None
    priority: int
    swapped: bool = False


@dataclass(frozen=True)
class LocalSwap:
    incoming_cut: str
    displaced_cut: str
    incoming_track: str
    displaced_track: str


@dataclass(frozen=True)
class Rejection:
    cut_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RearrangementPlan:
    # 'rearranged' | 'local_swaps' | 'order_rolled_back'
    outcome: str
    placement: dict[str, list[str]]
    moves: list[TrackMove]
    swaps: list[LocalSwap]
    rolled_back_orders: list[str]
    rejections: list[Rejection]


@dataclass
class _Action:
    order_id: str
    kind: str  # 'move' | 'swap'
    cut: str
    frm: str | None
    to: str | None
    victim: str | None = None
    victim_from: str | None = None
    victim_to: str | None = None


def _groups_conflict(a: str | None, b: str | None,
                     incompatible: frozenset[frozenset[str]]) -> bool:
    if not a or not b:
        return False
    return frozenset({a, b}) in incompatible


def _reachable(track_id: str, snapshot: YardSnapshot, owner: str) -> bool:
    for sid in snapshot.track_switches.get(track_id, frozenset()):
        locker = snapshot.switches.get(sid)
        if locker is not None and locker != owner:
            return False
    return True


def _blockers(track: YardTrack, cut: YardCut, residents: list[str],
              catalog: dict[str, YardCut], snapshot: YardSnapshot) -> list[str]:
    reasons: list[str] = []
    if cut.destination not in track.destinations:
        reasons.append('destination')
    used = sum(catalog[c].wagons for c in residents)
    if used + cut.wagons > track.capacity:
        reasons.append('capacity')
    if cut.hazard_group and cut.hazard_group in track.forbidden_hazard:
        reasons.append('hazard_track')
    for cid in residents:
        if _groups_conflict(cut.hazard_group, catalog[cid].hazard_group,
                            snapshot.incompatible_groups):
            reasons.append('hazard_pair:' + cid)
            break
    return reasons


def _find_track(cut, current, tracks, placement, catalog, snapshot, owner):
    feasible = []
    for tid in sorted(tracks):
        residents = [c for c in placement[tid] if c != cut.cut_id]
        reasons = _blockers(tracks[tid], cut, residents, catalog, snapshot)
        if tid != current and not _reachable(tid, snapshot, owner):
            reasons.append('switch_locked')
        if not reasons:
            free = tracks[tid].capacity - sum(catalog[c].wagons for c in residents) - cut.wagons
            feasible.append((free, tid))
    if not feasible:
        return None
    # staying put is preferred; otherwise best fit like hump classification
    if current is not None and any(tid == current for _, tid in feasible):
        return current
    return min(feasible)[1]


def _find_swap(cut, frm, tracks, placement, catalog, snapshot, owner, request_orders):
    if frm is None:
        return None
    for tid in sorted(t for t in tracks if t != frm):
        track = tracks[tid]
        for victim_id in sorted(placement[tid], key=lambda c: (catalog[c].priority, c)):
            victim = catalog[victim_id]
            if victim.priority >= cut.priority:
                continue
            # only committed, static residents may be displaced: cuts belonging
            # to this request could themselves be rolled back later
            if victim.order_id in request_orders:
                continue
            incoming_residents = [c for c in placement[tid] if c != victim_id]
            victim_residents = [c for c in placement[frm] if c != cut.cut_id]
            if _blockers(track, cut, incoming_residents, catalog, snapshot):
                continue
            if _blockers(tracks[frm], victim, victim_residents, catalog, snapshot):
                continue
            if not _reachable(tid, snapshot, owner) or not _reachable(frm, snapshot, owner):
                continue
            return (tid, victim_id)
    return None


def _validate_final(plan_tracks, placement, catalog, snapshot):
    for tid, residents in placement.items():
        track = plan_tracks[tid]
        if sum(catalog[c].wagons for c in residents) > track.capacity:
            raise ValueError('inconsistent rearrangement: capacity on ' + tid)
        for i, a in enumerate(residents):
            ca = catalog[a]
            if ca.hazard_group and ca.hazard_group in track.forbidden_hazard:
                raise ValueError('inconsistent rearrangement: hazard ban on ' + tid)
            for b in residents[i + 1:]:
                if _groups_conflict(ca.hazard_group, catalog[b].hazard_group,
                                    snapshot.incompatible_groups):
                    raise ValueError('inconsistent rearrangement: hazard pair on ' + tid)


def rearrange(snapshot: YardSnapshot, requested: list[YardCut],
              owner: str) -> RearrangementPlan:
    tracks = {t.track_id: t for t in snapshot.tracks}
    catalog = dict(snapshot.cuts)
    placement = {tid: list(ids) for tid, ids in snapshot.placement.items()}
    for tid in tracks:
        placement.setdefault(tid, [])

    committed_origin = {cid: tid for tid, ids in snapshot.placement.items() for cid in ids}
    for cid in committed_origin:
        if cid not in catalog:
            raise ValueError('placement references unknown cut ' + cid)
    requested_ids = [c.cut_id for c in requested]
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError('duplicate requested cut')
    request_orders = {c.order_id for c in requested}

    def locate(cid):
        return next((tid for tid, ids in placement.items() if cid in ids), None)

    actions: list[_Action] = []
    swaps: list[LocalSwap] = []
    rejections: list[Rejection] = []
    rolled: set[str] = set()

    def undo(action: _Action):
        if action.kind == 'move':
            if action.to is not None:
                placement[action.to].remove(action.cut)
            if action.frm is not None:
                placement[action.frm].append(action.cut)
        else:
            placement[action.victim_to].remove(action.victim)
            placement[action.victim_from].append(action.victim)
            placement[action.to].remove(action.cut)
            placement[action.frm].append(action.cut)

    def rollback_order(order_id: str):
        kept: list[_Action] = []
        undone_cuts: set[str] = set()
        for action in reversed(actions):
            if action.order_id == order_id:
                undo(action)
                undone_cuts.add(action.cut)
            else:
                kept.append(action)
        actions.clear()
        actions.extend(reversed(kept))
        swaps[:] = [s for s in swaps if s.incoming_cut not in undone_cuts]
        # restore the committed declaration of every cut of this order, so the
        # final validation runs against the version that remains in effect
        for rc in requested:
            if rc.order_id != order_id:
                continue
            committed = snapshot.cuts.get(rc.cut_id)
            if committed is not None:
                catalog[rc.cut_id] = committed
            else:
                catalog.pop(rc.cut_id, None)

    for cut in sorted(requested, key=lambda c: (-c.priority, c.cut_id)):
        if cut.order_id in rolled:
            continue
        catalog[cut.cut_id] = cut
        frm = locate(cut.cut_id)
        target = _find_track(cut, frm, tracks, placement, catalog, snapshot, owner)
        if target is not None:
            if target != frm:
                if frm is not None:
                    placement[frm].remove(cut.cut_id)
                placement[target].append(cut.cut_id)
                actions.append(_Action(cut.order_id, 'move', cut.cut_id, frm, target))
            continue

        swap = _find_swap(cut, frm, tracks, placement, catalog, snapshot, owner, request_orders)
        if swap is not None:
            tid, victim_id = swap
            victim = catalog[victim_id]
            placement[tid].remove(victim_id)
            placement[frm].remove(cut.cut_id)
            placement[tid].append(cut.cut_id)
            placement[frm].append(victim_id)
            actions.append(_Action(cut.order_id, 'swap', cut.cut_id, frm, tid,
                                   victim_id, tid, frm))
            swaps.append(LocalSwap(cut.cut_id, victim_id, frm, tid))
            continue

        reasons: list[str] = []
        for tid in sorted(tracks):
            residents = [c for c in placement[tid] if c != cut.cut_id]
            reasons.extend(_blockers(tracks[tid], cut, residents, catalog, snapshot))
            if tid != frm and not _reachable(tid, snapshot, owner):
                reasons.append('switch_locked:' + tid)
        rejections.append(Rejection(cut.cut_id, tuple(dict.fromkeys(reasons))))
        rolled.add(cut.order_id)
        rollback_order(cut.order_id)

    swapped_ids = {p for s in swaps for p in (s.incoming_cut, s.displaced_cut)}

    moves: list[TrackMove] = []
    seen = set()
    for cut in sorted(requested, key=lambda c: (-c.priority, c.cut_id)):
        cid = cut.cut_id
        origin = committed_origin.get(cid)
        now = locate(cid)
        if origin != now:
            moves.append(TrackMove(cid, cut.order_id, origin, now, cut.priority,
                                   cid in swapped_ids))
        seen.add(cid)
    for s in swaps:
        victim = catalog[s.displaced_cut]
        if s.displaced_cut not in seen:
            moves.append(TrackMove(victim.cut_id, victim.order_id,
                                   s.displaced_track, s.incoming_track,
                                   victim.priority, True))

    final_placement = {tid: list(ids) for tid, ids in placement.items()}
    _validate_final(tracks, final_placement, catalog, snapshot)

    if rolled:
        outcome = 'order_rolled_back'
    elif swaps:
        outcome = 'local_swaps'
    else:
        outcome = 'rearranged'

    return RearrangementPlan(
        outcome=outcome,
        placement=final_placement,
        moves=moves,
        swaps=list(swaps),
        rolled_back_orders=sorted(rolled),
        rejections=rejections,
    )
