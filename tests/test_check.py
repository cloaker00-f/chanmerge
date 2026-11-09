from chanmerge.check import check_plan
from chanmerge.grouping import build_groups, apply_edit
from chanmerge.inputs import parse_group_edit
from chanmerge.plan import Group, ChipChoice, ChipSet
from helpers import obs, make_plan, ACIS_I, ACIS_S, ACIS_S_HETG, ACIS_S_HETG_SUB, ACIS_S_CC, HRC_I, HRC_S


def texts(res, level):
    return " | ".join(i.text for i in res.issues if i.level == level)


def plan_with(*observations, window=3, sets=None):
    p = make_plan(list(observations), window=window, row_sets=sets)
    p.groups = build_groups(p)
    return p


def test_clean_plan_has_no_issues():
    p = plan_with(obs(1, ACIS_S, 0, chips=(7,)), obs(2, ACIS_S, 1, chips=(7,)))
    assert check_plan(p).issues == []


def test_all_chips_on_s1_to_s4_mix_classes():
    # s1, s3 are back-illuminated, s2, s4 front: 'all' really mixes them
    p = plan_with(obs(1, ACIS_S, 0), obs(2, ACIS_S, 1))
    assert "mixes chip classes" in texts(check_plan(p), "warn")


def test_rejects_acis_hrc_and_hrc_i_s_and_cc():
    p = plan_with(obs(1, ACIS_S, 0), obs(2, HRC_I, 1, chips=None), sets=[[1, 2]])
    assert "ACIS and HRC" in texts(check_plan(p), "reject")
    p = plan_with(obs(1, HRC_I, 0, chips=None), obs(2, HRC_S, 1, chips=None), sets=[[1, 2]])
    assert "HRC-I and HRC-S" in texts(check_plan(p), "reject")
    p = plan_with(obs(1, ACIS_S, 0), obs(2, ACIS_S_CC, 1), sets=[[1, 2]])
    assert "CC mode" in texts(check_plan(p), "reject")


def test_hrc_i_with_hrc_i_allowed():
    p = plan_with(obs(1, HRC_I, 0, chips=None), obs(2, HRC_I, 1, chips=None))
    assert check_plan(p).rejects == []


def test_i_plus_s_warning_and_separate_chip_class_warning():
    p = plan_with(obs(1, ACIS_I, 0, chips=(0, 1, 2, 3)), obs(2, ACIS_S, 1, chips=(6, 8)), sets=[[1, 2]])
    w = texts(check_plan(p), "warn")
    assert "ACIS-I and ACIS-S" in w and "false brightness step" in w
    assert "mixes chip classes" not in w            # i chips (fi) + s2, s4 (fi): same class
    p = plan_with(obs(1, ACIS_I, 0, chips=(0, 1, 2, 3, 6, 7)), obs(2, ACIS_I, 1, chips=(0, 1, 2, 3)))
    w = texts(check_plan(p), "warn")
    assert "mixes chip classes" in w and "ACIS-I and ACIS-S" not in w   # an acis-i obs with s3 on


def test_chip_choice_removes_class_warning():
    p = plan_with(obs(1, ACIS_I, 0, chips=(0, 1, 2, 3, 6, 7)), obs(2, ACIS_I, 1, chips=(0, 1, 2, 3)))
    p.chips_by_row[1] = ChipChoice("i", [ChipSet((0, 1, 2, 3))])
    assert "mixes chip classes" not in texts(check_plan(p), "warn")


def test_chip_off_in_member_warns():
    p = plan_with(obs(1, ACIS_S, 0, chips=(7,)), obs(2, ACIS_S, 1, chips=(5, 6)))
    p.chips_by_row[1] = ChipChoice("custom", [ChipSet((7,))])
    assert "no active chip in 2" in texts(check_plan(p), "warn")


def test_grating_mix_and_full_sub_warn():
    p = plan_with(obs(1, ACIS_S, 0), obs(2, ACIS_S_HETG, 1), sets=[[1, 2]])
    assert "grating and non-grating" in texts(check_plan(p), "warn")
    p = plan_with(obs(1, ACIS_S_HETG, 0), obs(2, ACIS_S_HETG_SUB, 1), sets=[[1, 2]])
    assert "full-frame and subarray" in texts(check_plan(p), "warn")


def test_shared_observations_ask_are_you_sure():
    p = plan_with(obs(1, ACIS_S, 0), obs(2, ACIS_S, 1), obs(3, ACIS_S, 20))
    apply_edit(p, parse_group_edit("add 3 g01"))
    s = texts(check_plan(p), "sure")
    assert "3 is in g01, s3" in s
    apply_edit(p, parse_group_edit("g01 chips S3 / S1,S3"))
    assert "S3 is in more than one chip set" in texts(check_plan(p), "sure")


def test_manual_group_longer_than_window_warns():
    p = plan_with(obs(1, ACIS_S, 0), obs(2, ACIS_S, 10))
    apply_edit(p, parse_group_edit("move 2 s1"))
    assert "more than the 3-day window" in texts(check_plan(p), "warn")


def test_different_streak_settings_warn():
    p = plan_with(obs(1, ACIS_S, 0), obs(2, ACIS_S_HETG, 0.5))
    p.row_sets[0].streak_action = "fill"
    apply_edit(p, parse_group_edit("add 2 s1"))
    assert "different streak settings" in texts(check_plan(p), "warn")


def test_obs_in_no_group_warns():
    p = plan_with(obs(1, ACIS_S, 0), obs(2, ACIS_S, 1))
    apply_edit(p, parse_group_edit("drop 2"))
    assert "2 is in no group and will not be downloaded" in texts(check_plan(p), "warn")


def test_obs_in_no_group_already_on_disk(tmp_path):
    # test e: an observation already reprocessed is 'not used', not 'not downloaded'
    from chanmerge.ciao import REPRO_MARKER
    p = plan_with(obs(1, ACIS_S, 0), obs(2, ACIS_S, 1))
    p.root = str(tmp_path)
    repro = tmp_path / "observations" / p.observations["2"].folder / "repro"
    repro.mkdir(parents=True)
    (repro / REPRO_MARKER).write_text("")
    apply_edit(p, parse_group_edit("drop 2"))
    w = texts(check_plan(p), "warn")
    assert "2 is in no group and will not be used (it is already on disk)" in w
    assert "will not be downloaded" not in w


def test_empty_plan_rejected():
    p = make_plan([obs(1, ACIS_S, 0)])
    p.groups = []
    assert check_plan(p).rejects
