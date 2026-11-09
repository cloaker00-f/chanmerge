from datetime import timedelta

import pytest

from chanmerge.grouping import (greedy_groups, build_groups, apply_edit, finish_edits, obs_in_no_group, span_days)
from chanmerge.inputs import InputError, parse_group_edit
from chanmerge.plan import window_tag
from helpers import T0, obs, make_plan, ACIS_S, ACIS_I, ACIS_S_HETG


def _items(days, exp_days=0.1):
    return [(str(i), T0 + timedelta(days=d), T0 + timedelta(days=d + exp_days)) for i, d in enumerate(days)]


def test_greedy_reference_case():
    # a, a+3, a+7, a+10 with window 7 -> {a, a+3, a+7}, {a+10}
    g = greedy_groups(_items([0, 3, 7, 10], exp_days=0.0), 7)
    assert g == [["0", "1", "2"], ["3"]]


def test_greedy_end_counts_and_long_obs_alone():
    # the third ends after the window -> new group
    g = greedy_groups(_items([0, 3, 6.95]), 7)
    assert g == [["0", "1"], ["2"]]
    long = [("a", T0, T0 + timedelta(days=9)), ("b", T0 + timedelta(days=1), T0 + timedelta(days=1.1))]
    assert greedy_groups(long, 7) == [["a"], ["b"]]


def test_greedy_every_obs_once_unsorted_input():
    items = _items([10, 0, 5, 20, 21, 3])
    g = greedy_groups(items, 4)
    flat = [o for grp in g for o in grp]
    assert sorted(flat) == sorted(i[0] for i in items) and len(flat) == len(set(flat))


def test_build_groups_ids_and_singles():
    # 950, 6688, 9052 with window 7 -> three s-groups (they are years apart)
    p = make_plan([obs(950, ACIS_I, 0), obs(6688, ACIS_I, 700), obs(9052, ACIS_I, 1300)], window=7)
    p.groups = build_groups(p)
    assert [g.group_id for g in p.groups] == ["s950", "s6688", "s9052"]
    p = make_plan([obs(1, ACIS_S, 0), obs(2, ACIS_S, 1), obs(3, ACIS_S, 30), obs(4, ACIS_S, 31), obs(5, ACIS_S, 90)], window=3)
    p.groups = build_groups(p)
    assert [(g.group_id, g.obsids) for g in p.groups] == [("g01", ["1", "2"]), ("g02", ["3", "4"]), ("s5", ["5"])]


def test_groups_stay_inside_row_sets():
    p = make_plan([obs(1, ACIS_I, 0), obs(2, ACIS_S, 0.5)], window=3)
    p.groups = build_groups(p)                       # separate rows
    assert sorted(g.group_id for g in p.groups) == ["s1", "s2"]
    p = make_plan([obs(1, ACIS_I, 0), obs(2, ACIS_S, 0.5)], window=3, row_sets=[[1, 2]])
    p.groups = build_groups(p)
    assert [g.obsids for g in p.groups] == [["1", "2"]]


def test_window_zero_all():
    p = make_plan([obs(1, ACIS_S, 0), obs(2, ACIS_S, 500)], window=0)
    p.groups = build_groups(p)
    assert [(g.group_id, g.obsids) for g in p.groups] == [("all", ["1", "2"])]
    p = make_plan([obs(1, ACIS_S, 0), obs(2, ACIS_S_HETG, 500)], window=0)
    p.groups = build_groups(p)
    assert sorted(g.group_id for g in p.groups) == ["all_A", "all_B"]


def test_only_flux_stage_grouped():
    p = make_plan([obs(1, ACIS_S, 0), obs(2, ACIS_S, 0.5, stage="repro")], window=3)
    p.groups = build_groups(p)
    assert [g.obsids for g in p.groups] == [["1"]]


def test_window_tag():
    assert window_tag(7) == "w7d" and window_tag(0) == "w0d" and window_tag(1.5) == "w1p5d"


def _plan3():
    p = make_plan([obs(101, ACIS_S, 0), obs(102, ACIS_S, 1), obs(103, ACIS_S, 20), obs(104, ACIS_S, 40)], window=3)
    p.groups = build_groups(p)   # g01 {101, 102}, s103, s104
    return p


def edit(p, text):
    return apply_edit(p, parse_group_edit(text))


def test_move_takes_out_of_old_group():
    p = _plan3()
    edit(p, "move 102 s103")
    assert p.group("g01").obsids == ["101"] and p.group("s103").obsids == ["103", "102"]
    assert p.group("g01").manual and p.group("s103").manual
    notes = finish_edits(p)
    assert any("renamed" in n for n in notes)          # s103 has two now -> g id
    assert all(not g.group_id.startswith("s") or len(g.obsids) == 1 for g in p.groups)


def test_move_to_own_single_group():
    p = _plan3()
    edit(p, "move 102 s")
    assert p.group("s102").obsids == ["102"] and "102" not in p.group("g01").obsids
    with pytest.raises(InputError):
        edit(p, "move 102 s")


def test_add_and_new_never_remove():
    p = _plan3()
    edit(p, "add 103 g01")
    assert p.group("g01").obsids == ["101", "102", "103"] and p.group("s103").obsids == ["103"]
    msg = edit(p, "new 101 104")
    assert "g02" in msg and p.group("g02").obsids == ["101", "104"]
    assert "101" in p.group("g01").obsids and p.group("s104").obsids == ["104"]
    with pytest.raises(InputError):
        edit(p, "add 103 g01")


def test_drop_and_delete_rules():
    p = _plan3()
    edit(p, "drop 102 g01")
    assert p.group("g01").obsids == ["101"]
    assert obs_in_no_group(p) == ["102"]                # not downloaded
    with pytest.raises(InputError, match="still has observations"):
        edit(p, "delete g01")
    edit(p, "drop all g01")
    with pytest.raises(InputError, match="empty"):
        finish_edits(p)                                 # empty groups block done
    edit(p, "delete g01")
    assert p.group("g01") is None
    edit(p, "drop 103")
    assert p.group("s103").obsids == []


def test_drop_everywhere():
    p = _plan3()
    edit(p, "add 101 s104")
    edit(p, "drop 101")
    assert "101" not in p.group("g01").obsids and "101" not in p.group("s104").obsids


def test_edit_errors():
    p = _plan3()
    for bad in ["move 999 g01", "move 101 g09", "add 101 g01", "drop 101 s104", "delete g07"]:
        with pytest.raises(InputError):
            edit(p, bad)
    p.observations["104"].stage = "repro"
    with pytest.raises(InputError, match="not going to flux"):
        edit(p, "add 104 g01")


def test_chips_edit():
    p = _plan3()
    msg = edit(p, "g01 chips S3 / S1,S3")
    assert "S3 in more than one chip set" in msg
    assert [s.chips for s in p.chip_sets_for(p.group("g01"))] == [(7,), (5, 7)]


def test_span_days():
    p = _plan3()
    assert span_days(p, p.group("g01")) == pytest.approx(1 + 10 / 86.4, rel=1e-3)
