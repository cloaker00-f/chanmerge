"""check_plan: what CIAO cannot do is rejected; doubtful choices are warnings the user accepts or edits.
never prompts, only returns issues."""
from dataclasses import dataclass, field
from typing import List

from .plan import CHIP_NAMES, BI_CHIPS, FI_CHIPS
from .grouping import span_days, obs_in_no_group, no_group_split
from .inputs import chip_overlaps


@dataclass
class Issue:
    level: str        # reject, warn, sure (shared data: 'Are you sure?')
    text: str
    group: str = ""


@dataclass
class CheckResult:
    issues: List[Issue] = field(default_factory=list)

    @property
    def rejects(self):
        return [i for i in self.issues if i.level == "reject"]

    @property
    def warnings(self):
        return [i for i in self.issues if i.level == "warn"]

    @property
    def sure(self):
        return [i for i in self.issues if i.level == "sure"]

    def add(self, level, text, group=""):
        self.issues.append(Issue(level, text, group))


def _chips_used(ob, cs):
    """chips of this observation that a chip set really uses (None set = all active)"""
    act = set(ob.active_chips) if ob.active_chips else None
    if cs.chips is None:
        return act
    if act is None:
        return set(cs.chips)
    return act & set(cs.chips)


def check_plan(plan):
    res = CheckResult()
    obs = plan.observations
    if not plan.groups:
        res.add("reject", "there is no group to merge")
    for g in plan.groups:
        gid = g.group_id
        if not g.obsids:
            res.add("reject", f"{gid} is empty", gid)
            continue
        cats = [obs[o].category for o in g.obsids]
        dets = {c.detector for c in cats}
        # rejected: merge_obs would silently skip these
        if any(c.is_acis for c in cats) and any(c.is_hrc for c in cats):
            res.add("reject", f"{gid}: ACIS and HRC cannot be merged (merge_obs keeps only the first instrument)", gid)
        if {"HRC-I", "HRC-S"} <= dets:
            res.add("reject", f"{gid}: HRC-I and HRC-S cannot be merged (merge_obs skips the other one)", gid)
        cc = [o for o in g.obsids if obs[o].category.is_cc]
        if cc:
            res.add("reject", f"{gid}: {', '.join(cc)} {'is' if len(cc) == 1 else 'are'} CC mode; merge_obs cannot use CC observations", gid)
        # warnings, accepted or edited
        if {"ACIS-I", "ACIS-S"} <= dets:
            res.add("warn", f"{gid}: ACIS-I and ACIS-S observations together. Flux differs by chip type and spectrum; "
                            f"a false brightness step may appear at the boundary.", gid)
        if dets == {"HRC-S"} and len(g.obsids) > 1:
            res.add("warn", f"{gid}: several HRC-S observations together; HRC-S merging is untested here.", gid)
        grat = {c.grating for c in cats}
        if len(grat) > 1:
            res.add("warn", f"{gid}: {'grating and non-grating' if 'NONE' in grat else 'HETG and LETG'} observations together "
                            f"({', '.join(sorted(grat))}); depth and the arm treatment differ between them.", gid)
        wins = {c.window for c in cats if c.window in ("full", "sub")}
        if wins == {"full", "sub"}:
            res.add("warn", f"{gid}: full-frame and subarray observations together; the subarray covers a narrow strip only.", gid)
        if g.manual and plan.window_days > 0 and len(g.obsids) > 1:
            sp = span_days(plan, g)
            if sp > plan.window_days:
                res.add("warn", f"{gid}: spans {sp:.1f} days, more than the {plan.window_days:g}-day window.", gid)
        letters = set()
        for o in g.obsids:
            rs = plan.row_set_of_row(plan.row_of(o))
            letters.add(rs.streak_action if rs else "leave")
        if len(letters) > 1:
            res.add("warn", f"{gid}: observations from sets with different streak settings ({', '.join(sorted(letters))}); "
                            f"the group uses the setting of set {g.row_set}.", gid)
        if g.chip_sets is None and all(c.is_acis for c in cats):
            choices = {}
            for r in plan.chip_rows(g):
                if r in plan.chips_by_row:
                    choices.setdefault(plan.chips_by_row[r].text(), []).append(r)
            if len(choices) > 1:
                used = plan.chips_by_row[plan.chip_rows(g)[0]].text()
                res.add("warn", f"{gid}: its observations come from rows with different chip choices "
                                f"({'; '.join(f'row {r[0]}: {t}' for t, r in choices.items())}); the group uses {used}. "
                                f"Change it with '{gid} chips ...'.", gid)
        # chip classes, known before download from the active chips
        if all(c.is_acis for c in cats):
            for cs in plan.chip_sets_for(g):
                used = set()
                for o in g.obsids:
                    u = _chips_used(obs[o], cs)
                    if u:
                        used |= u
                    elif u is not None:
                        res.add("warn", f"{gid}: chip set {cs.names()} has no active chip in {o}; that member adds nothing.", gid)
                if used & set(BI_CHIPS) and used & set(FI_CHIPS):
                    bi = ",".join(CHIP_NAMES[c] for c in sorted(used & set(BI_CHIPS)))
                    fi = ",".join(CHIP_NAMES[c] for c in sorted(used & set(FI_CHIPS)))
                    res.add("warn", f"{gid}: chip set {cs.names()} mixes chip classes (back-illuminated {bi}, front-illuminated {fi}); "
                                    f"the flux can step at chip borders. Accept, or change the chip choice.", gid)
            ov = chip_overlaps(plan.chip_sets_for(g))
            if ov:
                res.add("sure", f"{', '.join(ov)} {'is' if len(ov) == 1 else 'are'} in more than one chip set ({gid}).", gid)
    # the same observation in several groups with overlapping chips (add / new)
    seen = {}
    for g in plan.groups:
        for o in g.obsids:
            seen.setdefault(o, []).append(g)
    for o, gs in seen.items():
        if len(gs) < 2:
            continue
        res.add("sure", f"{o} is in {', '.join(g.group_id for g in gs)}.")
    on, off = no_group_split(plan)
    for o in on:
        res.add("warn", f"{o} is in no group and will not be used (it is already on disk).")
    for o in off:
        res.add("warn", f"{o} is in no group and will not be downloaded.")
    return res
