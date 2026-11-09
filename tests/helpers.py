# small builders for plans used by the unit tests
from datetime import datetime, timedelta, timezone

from chanmerge.plan import RunPlan, Observation, Category, RowSet, ChipChoice, ChipSet

T0 = datetime(2010, 1, 1, tzinfo=timezone.utc)
ACIS_I = Category("ACIS-I", "NONE", "TE", "full")
ACIS_S = Category("ACIS-S", "NONE", "TE", "full")
ACIS_S_HETG = Category("ACIS-S", "HETG", "TE", "full")
ACIS_S_HETG_SUB = Category("ACIS-S", "HETG", "TE", "sub")
ACIS_S_CC = Category("ACIS-S", "NONE", "CC", "full")
HRC_I = Category("HRC-I", "NONE", "-", "-")
HRC_S = Category("HRC-S", "LETG", "-", "-")


def obs(obsid, cat, day, exp_ks=10.0, stage="flux", chips=(5, 6, 7, 8)):
    return Observation(str(obsid), cat, T0 + timedelta(days=day), exp_ks * 1000, stage=stage, active_chips=chips)


def make_plan(observations, window=0.0, row_sets=None, tmp="/tmp/x"):
    p = RunPlan(ra=201.365, dec=-43.019, radius_arcmin=5, target="T", root=tmp)
    for o in observations:
        p.observations[o.obsid] = o
    p.categories = sorted({o.category for o in observations}, key=lambda c: c.sort_key())
    for r, c in enumerate(p.categories, 1):
        if c.is_acis:
            p.chips_by_row[r] = ChipChoice("all", [ChipSet(None)])
    p.window_days = window
    if row_sets is None:
        row_sets = [[r] for r in range(1, len(p.categories) + 1)]
    p.row_sets = [RowSet(chr(65 + i), rows) for i, rows in enumerate(row_sets)]
    return p
