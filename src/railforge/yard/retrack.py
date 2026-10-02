from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RetrackCut:
    cut_id: str
    destination: str
    wagons: int
    priority: int = 0
    hazard_group: str | None = None

@dataclass(frozen=True)
class RetrackTrack:
    track_id: str
    capacity: int
    destinations: frozenset[str]
    forbidden_hazard: frozenset[str] = frozenset()
    switch_locked: bool = False

@dataclass(frozen=True)
class Swap:
    incoming: str
    displaced: str
    from_track: str
    to_track: str

@dataclass(frozen=True)
class RetrackPlan:
    status: str
    assignment: dict[str, str]
    moved: tuple[str, ...]
    swaps: tuple[Swap, ...]
    unplaced: tuple[str, ...]

def _maps(cuts: list[RetrackCut], tracks: list[RetrackTrack]):
    cut_map = {}
    for c in cuts:
        if c.cut_id in cut_map:
            raise ValueError('duplicate cut')
        if c.wagons <= 0:
            raise ValueError('wagons')
        cut_map[c.cut_id] = c
    track_map = {}
    for t in tracks:
        if t.track_id in track_map:
            raise ValueError('duplicate track')
        if t.capacity <= 0:
            raise ValueError('capacity')
        track_map[t.track_id] = t
    return (cut_map, track_map)

def _tenants(assignment: dict[str, str], track_id: str):
    return sorted((cid for cid, tid in assignment.items() if tid == track_id))

def _hazard_clash(a: RetrackCut, b: RetrackCut, hazard_conflicts):
    return (a.hazard_group is not None and b.hazard_group is not None and
            frozenset((a.hazard_group, b.hazard_group)) in hazard_conflicts)

def _fits(cut: RetrackCut, track: RetrackTrack, tenant_ids, cut_map, hazard_conflicts):
    if cut.destination not in track.destinations:
        return False
    if cut.hazard_group and cut.hazard_group in track.forbidden_hazard:
        return False
    load = cut.wagons + sum((cut_map[t].wagons for t in tenant_ids))
    if load > track.capacity:
        return False
    return not any((_hazard_clash(cut, cut_map[t], hazard_conflicts) for t in tenant_ids))

def _track_violations(track: RetrackTrack, tenant_ids, cut_map, hazard_conflicts):
    out = []
    known = [cid for cid in tenant_ids if cid in cut_map]
    for cid in known:
        c = cut_map[cid]
        if c.destination not in track.destinations:
            out.append('destination:' + cid + ':' + track.track_id)
        if c.hazard_group and c.hazard_group in track.forbidden_hazard:
            out.append('forbidden-hazard:' + cid + ':' + track.track_id)
    if sum((cut_map[cid].wagons for cid in known)) > track.capacity:
        out.append('capacity:' + track.track_id)
    for i, a in enumerate(known):
        for b in known[i + 1:]:
            if _hazard_clash(cut_map[a], cut_map[b], hazard_conflicts):
                out.append('hazard-conflict:' + track.track_id + ':' + a + ':' + b)
    return out

def violations(assignment: dict[str, str], cuts: list[RetrackCut], tracks: list[RetrackTrack],
               hazard_conflicts=frozenset()):
    cut_map, track_map = _maps(cuts, tracks)
    conflicts = frozenset((frozenset(p) for p in hazard_conflicts))
    out = []
    for cid in sorted(assignment):
        tid = assignment[cid]
        if cid not in cut_map:
            out.append('unknown-cut:' + cid)
        elif tid not in track_map:
            out.append('unknown-track:' + cid + ':' + tid)
    for tid in sorted(track_map):
        out.extend(_track_violations(track_map[tid], _tenants(assignment, tid), cut_map, conflicts))
    return tuple(sorted(out))

def occupancy(assignment: dict[str, str], cuts: list[RetrackCut]):
    cut_map = {c.cut_id: c for c in cuts}
    out = {}
    for cid, tid in assignment.items():
        if cid in cut_map:
            out[tid] = out.get(tid, 0) + cut_map[cid].wagons
    return out

def _involved(track: RetrackTrack, tenant_ids, cut_map, hazard_conflicts):
    bad = set()
    for cid in tenant_ids:
        c = cut_map[cid]
        if c.destination not in track.destinations:
            bad.add(cid)
        if c.hazard_group and c.hazard_group in track.forbidden_hazard:
            bad.add(cid)
    if sum((cut_map[c].wagons for c in tenant_ids)) > track.capacity:
        bad.update(tenant_ids)
    for i, a in enumerate(tenant_ids):
        for b in tenant_ids[i + 1:]:
            if _hazard_clash(cut_map[a], cut_map[b], hazard_conflicts):
                bad.add(a)
                bad.add(b)
    return bad

def _direct(cut: RetrackCut, assignment, cut_map, track_map, hazard_conflicts):
    choices = []
    for tid, track in track_map.items():
        if track.switch_locked:
            continue
        tenants = _tenants(assignment, tid)
        if _fits(cut, track, tenants, cut_map, hazard_conflicts):
            free = track.capacity - sum((cut_map[c].wagons for c in tenants))
            choices.append((free - cut.wagons, tid))
    return min(choices)[1] if choices else None

def _swap(cut: RetrackCut, assignment, cut_map, track_map, hazard_conflicts):
    options = []
    for from_tid in sorted(track_map):
        track = track_map[from_tid]
        if track.switch_locked:
            continue
        for pid in _tenants(assignment, from_tid):
            p = cut_map[pid]
            if p.priority >= cut.priority:
                continue
            rest = [c for c in _tenants(assignment, from_tid) if c != pid]
            if not _fits(cut, track, rest, cut_map, hazard_conflicts):
                continue
            for to_tid in sorted(track_map):
                if to_tid == from_tid:
                    continue
                target = track_map[to_tid]
                if target.switch_locked:
                    continue
                if _fits(p, target, _tenants(assignment, to_tid), cut_map, hazard_conflicts):
                    options.append((p.priority, p.wagons, pid, from_tid, to_tid))
    if not options:
        return None
    _, _, pid, from_tid, to_tid = min(options)
    return (pid, from_tid, to_tid)

def replan(cuts: list[RetrackCut], tracks: list[RetrackTrack], current: dict[str, str],
           hazard_conflicts=frozenset()):
    cut_map, track_map = _maps(cuts, tracks)
    conflicts = frozenset((frozenset(p) for p in hazard_conflicts))
    assignment = {cid: tid for cid, tid in current.items() if cid in cut_map and tid in track_map}
    original = dict(assignment)
    unseated = []
    for tid in sorted(track_map):
        track = track_map[tid]
        if track.switch_locked:
            if _track_violations(track, _tenants(assignment, tid), cut_map, conflicts):
                raise ValueError('locked track infeasible:' + tid)
            continue
        while True:
            tenants = _tenants(assignment, tid)
            if not _track_violations(track, tenants, cut_map, conflicts):
                break
            involved = _involved(track, tenants, cut_map, conflicts)
            victim = min(involved, key=lambda cid: (cut_map[cid].priority, cut_map[cid].wagons, cid))
            assignment.pop(victim)
            unseated.append(victim)
    queue = sorted(set(unseated) | (set(cut_map) - set(assignment)),
                   key=lambda cid: (-cut_map[cid].priority, -cut_map[cid].wagons, cid))
    swaps = []
    unplaced = []
    for cid in queue:
        cut = cut_map[cid]
        tid = _direct(cut, assignment, cut_map, track_map, conflicts)
        if tid is None:
            option = _swap(cut, assignment, cut_map, track_map, conflicts)
            if option is None:
                unplaced.append(cid)
                continue
            displaced, from_tid, to_tid = option
            assignment[displaced] = to_tid
            swaps.append(Swap(cid, displaced, from_tid, to_tid))
            tid = from_tid
        assignment[cid] = tid
    if unplaced:
        return RetrackPlan('rolled_back', original, (), (), tuple(unplaced))
    moved = tuple(sorted((cid for cid in assignment if cid in original and original[cid] != assignment[cid])))
    return RetrackPlan('replanned', assignment, moved, tuple(swaps), ())
