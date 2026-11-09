"""Time grouping inside each row set, group ids, and the group edit commands."""
from datetime import timedelta

from .plan import Group
from .inputs import InputError, chip_overlaps, parse_chip_names


def greedy_groups(items, window_days):
    """items: list of (obsid, start, end). sort by start; add the next while
    max(end incl. new) - first start <= window, else start a new group. every obsid exactly once."""
    items = sorted(items, key=lambda t: (t[1], t[0]))
    win = timedelta(days=window_days)
    out, cur, first, last_end = [], [], None, None
    for oid, st, en in items:
        if not cur:
            cur, first, last_end = [oid], st, en
            continue
        new_end = max(last_end, en)
        if new_end - first <= win:
            cur.append(oid); last_end = new_end
        else:
            out.append(cur)
            cur, first, last_end = [oid], st, en
    if cur:
        out.append(cur)
    return out


def build_groups(plan):
    """automatic groups for every row set, ids g01.. in time order across the run, s<obsid> for singles,
    'all' (or all_<letter> with several sets) for window 0"""
    obs = plan.observations
    raw = []  # (row_set letter, [obsids])
    for rs in plan.row_sets:
        members = [o for o in plan.flux_obsids() if plan.row_of(o) in rs.rows]
        if not members:
            continue
        if plan.window_days == 0:
            raw.append((rs.letter, sorted(members, key=lambda o: obs[o].best_start), "all"))
            continue
        items = [(o, obs[o].best_start, obs[o].end) for o in members]
        for g in greedy_groups(items, plan.window_days):
            raw.append((rs.letter, g, None))
    many_sets = len({r[0] for r in raw}) > 1
    raw.sort(key=lambda r: (min(obs[o].best_start for o in r[1]), r[0]))
    groups, n = [], 0
    for letter, members, kind in raw:
        if kind == "all":
            gid = f"all_{letter}" if many_sets else "all"
        elif len(members) == 1:
            gid = f"s{members[0]}"
        else:
            n += 1
            gid = f"g{n:02d}"
        groups.append(Group(gid, list(members), letter))
    return groups


def span_days(plan, group):
    obs = [plan.observations[o] for o in group.obsids]
    st = min(o.best_start for o in obs)
    en = max(o.end for o in obs)
    return (en - st).total_seconds() / 86400.0


def next_gid(plan):
    used = [int(g.group_id[1:]) for g in plan.groups if g.group_id.startswith("g") and g.group_id[1:].isdigit()]
    return f"g{(max(used) + 1 if used else 1):02d}"


def _need_group(plan, gid):
    g = plan.group(gid)
    if g is None:
        raise InputError(f"there is no group {gid}")
    return g


def _need_flux(plan, obsid):
    if obsid not in plan.observations:
        raise InputError(f"{obsid} is not in this search")
    if plan.observations[obsid].stage != "flux":
        raise InputError(f"{obsid} is not going to flux (stage {plan.observations[obsid].stage}); set its stage first")


def _letter_of(plan, obsid):
    rs = plan.row_set_of_row(plan.row_of(obsid))
    return rs.letter if rs else ""


def apply_edit(plan, cmd):
    """apply one parsed group edit in place; returns a message. only move and drop take an observation out
    of a group; new and add never remove it from anywhere (user rule 2026-11-04)."""
    kind = cmd[0]
    if kind == "move":
        _, oid, target = cmd
        _need_flux(plan, oid)
        olds = plan.groups_of(oid)
        if target == "s":
            gid = f"s{oid}"
            g = plan.group(gid)
            if g and g.obsids == [oid]:
                raise InputError(f"{oid} is already alone in {gid}")
            if g is None:
                g = Group(gid, [], _letter_of(plan, oid), manual=True)
                plan.groups.append(g)
        else:
            g = _need_group(plan, target)
        if oid in g.obsids and len(olds) == 1:
            raise InputError(f"{oid} is already in {g.group_id}")
        for o in olds:
            if o is not g:
                o.obsids.remove(oid); o.manual = True
        if oid not in g.obsids:
            g.obsids.append(oid)
        g.manual = True
        return f"{oid} moved to {g.group_id}"
    if kind == "add":
        _, oid, gid = cmd
        _need_flux(plan, oid)
        g = _need_group(plan, gid)
        if oid in g.obsids:
            raise InputError(f"{oid} is already in {gid}")
        g.obsids.append(oid); g.manual = True
        return f"{oid} added to {gid} (it stays in its other groups)"
    if kind == "new":
        _, oids = cmd
        for o in oids:
            _need_flux(plan, o)
        if len(set(oids)) != len(oids):
            raise InputError("an obsid is written twice")
        gid = next_gid(plan)
        plan.groups.append(Group(gid, list(oids), _letter_of(plan, oids[0]), manual=True))
        return f"new group {gid}: {', '.join(oids)} (they stay in their other groups)"
    if kind == "drop":
        _, oid, gid = cmd
        if gid is None:
            gs = plan.groups_of(oid)
            if not gs:
                raise InputError(f"{oid} is in no group")
            for g in gs:
                g.obsids.remove(oid); g.manual = True
            return f"{oid} dropped from {', '.join(g.group_id for g in gs)}"
        g = _need_group(plan, gid)
        if oid not in g.obsids:
            raise InputError(f"{oid} is not in {gid}")
        g.obsids.remove(oid); g.manual = True
        return f"{oid} dropped from {gid}"
    if kind == "drop_all":
        g = _need_group(plan, cmd[1])
        g.obsids.clear(); g.manual = True
        return f"{g.group_id} is now empty (delete {g.group_id} removes it)"
    if kind == "delete":
        g = _need_group(plan, cmd[1])
        if g.obsids:
            raise InputError(f"{g.group_id} still has observations; empty it first (drop all {g.group_id})")
        plan.groups.remove(g)
        return f"{g.group_id} deleted"
    if kind == "chips":
        _, gid, text = cmd
        g = _need_group(plan, gid)
        if any(plan.observations[o].category.is_hrc for o in g.obsids):
            raise InputError("chips cannot be chosen for HRC in this version")
        g.chip_sets = parse_chip_names(text); g.manual = True
        ov = chip_overlaps(g.chip_sets)
        return f"{gid} chips: {' / '.join(s.names() for s in g.chip_sets)}" + (f" ({', '.join(ov)} in more than one chip set)" if ov else "")
    raise InputError("unknown edit")


def finish_edits(plan):
    """after 'done'. empty groups block it: only the user deletes a group (delete gNN).
    an s-group that gained members gets a g id."""
    empty = [g.group_id for g in plan.groups if not g.obsids]
    if empty:
        raise InputError(f"{', '.join(empty)} {'is' if len(empty) == 1 else 'are'} empty: write "
                         f"{' and '.join('delete ' + e for e in empty)}, or add observations, then done")
    notes = []
    for g in plan.groups:
        if g.group_id.startswith("s") and len(g.obsids) > 1:
            old = g.group_id
            g.group_id = next_gid(plan)
            notes.append(f"{old} now has {len(g.obsids)} observations and is renamed {g.group_id}")
    return notes


def obs_in_no_group(plan):
    """flux observations left in no group: they are not downloaded (user rule 2026-11-04)"""
    inside = {o for g in plan.groups for o in g.obsids}
    return [o for o in plan.flux_obsids() if o not in inside]


def no_group_split(plan):
    """observations in no group, split into (already reprocessed in observations/, not there).
    the first are only not used; saying 'not downloaded' for them was wrong (test e, second-pass edit screen)"""
    import os
    from .ciao import REPRO_MARKER
    on, off = [], []
    for o in obs_in_no_group(plan):
        d = os.path.join(plan.root or "", "observations", plan.observations[o].folder, "repro", REPRO_MARKER)
        (on if os.path.exists(d) else off).append(o)
    return on, off
