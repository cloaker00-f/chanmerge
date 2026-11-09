# the terminal flow with scripted answers and a fake archive; no network, no ciao
import glob
import os
import types

import pytest

import chanmerge.archive as A
from chanmerge.cli import Terminal, chip_warnings
from chanmerge.inputs import chip_choice
from chanmerge.run import Report, HeaderProblem

LT = "/Users/cloaker/Desktop/CAFIG/chanmerge/latest-test"
FOV = {
    "950": f"{LT}/chanmergetest/obs_181431.08_-170926.2/950/primary/acisf00950_000N005_fov1.fits",
    "6688": f"{LT}/chanmergetest/obs_181431.08_-170926.2/6688/primary/acisf06688_000N003_fov1.fits",
    "27522": f"{LT}/te-grating-test/27522_g/primary/acisf27522_000N002_fov1.fits",
}
OCAT = """OBSID\tINSTR\tGRAT\tMODE\tDATAMODE\tREADOUT_DETECTOR\tSUBARY\tROW_CNT
-----\t-----\t----\t----\t--------\t-----\t----\t---
950\tACIS-I\tNONE\tTE\tFAINT\tACIS-012367\tNONE\t
6688\tACIS-S\tNONE\tTE\tVFAINT\tACIS-235678\tNONE\t
27522\tACIS-S\tHETG\tTE\tFAINT\tACIS-5678\tNONE\t
13197\tACIS-S\tHETG\tCC\tCC33_FAINT\tACIS-5678\tNONE\t
463\tHRC-I\tNONE\t\t\tHRC-I\t\t
"""
HROWS = [
    dict(obsid="950", status="archived", name="GX 13+1", ra="273.6316", dec="-17.1574", time="51000.5", detector="ACIS-I", grating="NONE", exposure="9600", public_date="51400", data_mode="TE_00AE"),
    dict(obsid="6688", status="archived", name="GPSR5", ra="273.934167", dec="-16.878333", time="53800.1", detector="ACIS-S", grating="NONE", exposure="5000", public_date="54200", data_mode="TE_00AE"),
    dict(obsid="27522", status="archived", name="Cen X-3", ra="170.312917", dec="-60.623789", time="59909.77", detector="ACIS-S", grating="HETG", exposure="9755", public_date="60300", data_mode="TE_0040"),
    dict(obsid="13197", status="archived", name="GX", ra="273.631667", dec="-17.157417", time="55900", detector="ACIS-S", grating="HETG", exposure="10000", public_date="56300", data_mode="CC_0010"),
    dict(obsid="463", status="archived", name="x", ra="201.365", dec="-43.019", time="51400", detector="HRC-I", grating="NONE", exposure="20000", public_date="51800", data_mode=None),
    dict(obsid="999", status="scheduled", name="x", time=None, detector="ACIS-S", grating="NONE", exposure=None, public_date=None, data_mode=None),
]


def fake_archive(heasarc_error=0, ocat_down=False, rows=HROWS):
    state = {"h": heasarc_error}

    def query(ra, dec, r):
        if state["h"] > 0:
            state["h"] -= 1
            raise A.ArchiveError("HEASARC query failed: ConnectionError: down")
        return [dict(x) for x in rows]

    def ocat(ids):
        if ocat_down:
            raise A.ArchiveError("Chandra Ocat unreachable: down")
        return A.parse_ocat_text(OCAT)

    def fovs(dest):
        return {o: (FOV[o] if os.path.exists(FOV.get(o, "")) else None) for o in dest}

    ns = types.SimpleNamespace(**{k: getattr(A, k) for k in dir(A) if not k.startswith("_")})
    ns.query_heasarc, ns.fetch_ocat, ns.fetch_fovs = query, ocat, fovs
    ns.download_sizes = lambda obs, evt_sizes=None: ({o.obsid: 100_000_000 for o in obs}, True)
    return ns


def drive(answers, tmp_path, archive=None, run_fn=None, cwd=None):
    it = iter(answers)
    out = []
    prompts = []

    def inp(prompt):
        prompts.append(prompt)
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    t = Terminal(input_fn=inp, out=lambda s="": out.append(str(s)), archive=archive or fake_archive(),
                 run_fn=run_fn, cwd=cwd or str(tmp_path))
    t.main()
    return t, "\n".join(out), prompts


# answers up to the groups, as in the example session (with mistakes typed on purpose)
BASE = [
    "1",                       # new target
    "273.6295", "-17.1573", "5",
    "Cen A", "CenA",           # space -> error, then fine
    "", "",                    # no date limits
    "flux", "flux", "flux", "flux", "none", "none",   # rows: 1 ACIS-I, 2 ACIS-S, 3 HETG TE, 4 HETG CC (flux refused), 5 HRC
    "done",
    "0.5:7", "0.5:7:2.3",      # missing eff -> asked again, then renamed to broad
    "2",                       # row 1 chips: i
    "7", "S3 / S1,S3",         # row 2: custom; 6688's own target is on S3 -> no warning
    "7", "S3",                 # row 3: 27522's target on S3
    "5", "4", "y",             # arms: 5 invalid, 4 = mask + fill, continue
    "3",                       # window 3 days
    "1+3", "separate", "y",    # missing row 2 -> asked again
    "all", "fill", "fill", "mask",   # streak: all sets, one action each
]


def test_full_flow_until_start_no(tmp_path):
    answers = BASE + ["move 999 g01", "add 950 s6688", "done", "y", "y", "n", "3"]
    t, out, prompts = drive(answers, tmp_path)
    p = t.last_plan
    assert "the folder name contains a space" in out and p.target == "CenA"
    assert "RA = 273.63 deg (18:14:31.1)" in out
    assert "1 not archived or not public dropped" in out
    assert p.observations["13197"].stage == "none"             # cc cannot be flux
    assert "identical to ciao's 'broad'" in out and [b.name for b in p.acis_bands] == ["broad"]
    assert "Reading chip layout (fov1" in out and "target on" in out
    # the search point (gx 13+1) lies on 950's field only: shown there, the targets decide the warnings
    assert "search point on" in out and p.observations["950"].point_chips == (3,)
    assert p.observations["6688"].source_chips == (7,) and p.observations["6688"].point_chips == ()
    assert p.chips_by_row[1].mode == "i" and [s.chips for s in p.chips_by_row[2].sets] == [(7,), (5, 7)]
    assert "falls on no chip" not in out and "WARNING: 6688" not in out
    assert p.arm_method_by_row[3] == 4
    assert "row 2 is missing" in out and "understood: 1, 2, 3" in out
    assert [rs.streak_action for rs in p.row_sets] == ["fill", "fill", "mask"]
    assert "there is no group g01" in out or "999 is not in this search" in out
    assert "renamed g01" in out                                   # s6688 got a second member
    assert "ACIS-I and ACIS-S" in out and "950 is in s950, g01" in out
    g01 = p.group("g01")
    assert [cs.chips for cs in p.chip_sets_for(g01)] == [(7,), (5, 7)]          # the group's own row set decides
    assert "different chip choices" in out
    assert "Download: 3 observations, 300 MB" in out and "Nothing was downloaded" in out


def test_run_is_called_and_report_printed(tmp_path):
    seen = {}

    def fake_run(plan, on_header_problems=None, say=print, only_groups=None):
        seen["plan"] = plan
        r = Report()
        r.images = [{}]
        r.streak = {"950": "fill"}
        r.run_dirs = [os.path.join(plan.root, plan.target, "merged_final_image_broad_w3d")]
        return r
    answers = BASE + ["done", "y", "y", "3"]   # warnings, start
    t, out, _ = drive(answers, tmp_path, run_fn=fake_run)
    assert "setup is complete. Processing now runs automatically." in out
    assert "streak filled: 950" in out and "chanmerge_index_members.csv" in out
    assert seen["plan"].window_days == 3


def test_header_problem_callback_edits_and_continues(tmp_path):
    def fake_run(plan, on_header_problems=None, say=print, only_groups=None):
        hp = HeaderProblem("full_sub", "950,6688", "s950", False, "s950 now mixes full-frame and subarray observations")
        assert on_header_problems(plan, [hp]) is True
        return Report()
    answers = BASE + ["done", "y", "y", "done", "y", "y", "3"]   # done (groups) / warnings y / start y / header: done, warnings y, continue y
    t, out, _ = drive(answers[:-1] + ["y", "3"], tmp_path, run_fn=fake_run)
    assert "--- Header check ---" in out and "setup is complete again" in out


def test_heasarc_error_then_menu(tmp_path):
    answers = ["1", "10", "10", "5", "", "", "", "1",          # error -> new coordinates
               "10", "10", "5", "", "", "", "2", "3"]          # error again -> menu -> exit
    t, out, _ = drive(answers, tmp_path, archive=fake_archive(heasarc_error=2))
    assert out.count("HEASARC query failed") == 2


def test_ocat_down_continue_without(tmp_path):
    answers = ["1", "273.6295", "-17.1573", "5", "", "", "", "2",
               "flux", "none", "none", "none", "none", "none", "done"]
    t, out, prompts = drive(answers, tmp_path, archive=fake_archive(ocat_down=True))
    p = t.last_plan
    assert "Chandra Ocat unreachable" in out and p.ocat_ok is False
    assert any(c.window == "?" for c in p.categories)


def test_folder_path_with_space_stops(tmp_path):
    d = tmp_path / "my data"
    d.mkdir()
    t, out, _ = drive(["1"], tmp_path, cwd=str(d))
    assert "contains a space" in out and "Move to a folder" in out


def test_local_directory_not_available(tmp_path):
    t, out, _ = drive(["2", "x", "3"], tmp_path)
    assert "not available in this version" in out and "choose 1, 2 or 3" in out


def test_chip_warnings_unit():
    from chanmerge.plan import Observation, Category
    from datetime import datetime, timezone
    o = Observation("1", Category("ACIS-S", "NONE", "TE", "full"), datetime.now(timezone.utc), 1, active_chips=(5, 6, 7, 8),
                    source_chips=(6, 7), source_near_edge=True)
    w = chip_warnings([o], chip_choice("custom", "S3"))
    assert any("near the S2/S3 edge; S2 is in no chip set" in x for x in w)
    w = chip_warnings([o], chip_choice("custom", "I3"))
    assert any("no active chip" in x for x in w) and any("in no chip set" in x for x in w)
    assert chip_warnings([o], chip_choice("all")) == []


def test_collapse_issues():
    from chanmerge.cli import collapse_issues
    from chanmerge.check import Issue
    t = "chip set all mixes chip classes (back-illuminated S1,S3, front-illuminated S2,S4)"
    iss = [Issue("warn", f"{g}: {t}", g) for g in ("s1", "s2", "s3", "g01")] + [Issue("warn", "x is in no group")]
    out = collapse_issues(iss)
    assert out == [f"4 groups (s1, s2, s3, g01): {t}", "x is in no group"]


def test_subarray_grating_arm_warning():
    from chanmerge.cli import subarray_arm_warning
    from helpers import obs, make_plan, ACIS_S_HETG, ACIS_S_HETG_SUB
    a, b = obs(16702, ACIS_S_HETG_SUB, 0), obs(17701, ACIS_S_HETG, 1)
    a.subarray_rows = 350
    p = make_plan([a, b])
    sub_row = p.categories.index(ACIS_S_HETG_SUB) + 1
    full_row = p.categories.index(ACIS_S_HETG) + 1
    w = subarray_arm_warning(p, sub_row, 4)
    assert "row %d is a grating subarray (16702: 350 rows)" % sub_row in w and "69%" in w
    assert "simulated" in w and "0.3%" in w and "25-27%" in w                 # option 4: filled, not lost
    w2 = subarray_arm_warning(p, sub_row, 2)
    assert "no data" in w2 and "simulated" not in w2 and "26% at 512" in w2      # options 2-3: really lost
    assert subarray_arm_warning(p, sub_row, 1) is None                # filter only: nothing removed
    assert subarray_arm_warning(p, full_row, 4) is None               # full frame: no warning
    a.subarray_rows = None
    assert "16702: rows unknown" in subarray_arm_warning(p, sub_row, 2)
    # in the terminal: shown after the option's own warning; n shows the options again
    out, answers = [], iter(["4", "n", "1", "y"])
    t = Terminal(input_fn=lambda prompt: next(answers), out=lambda s="": out.append(str(s)), archive=fake_archive(), cwd="/tmp")
    t.ask_arms(p, sub_row)
    text = "\n".join(out)
    assert text.count("grating subarray") == 1 and text.count("Grating arms for row") == 2
    assert p.arm_method_by_row[sub_row] == 1


def test_letg_arm_warning_says_a_box_covering_the_image_darkens_it_for_every_option():
    # test w (session 007): inside ciao's letg box the filter removes most events, background included; a box that covers the whole image
    # darkens the whole image, so the warning is shown for every arm option, option 1 included; no fixed width factor any more
    from chanmerge.cli import subarray_arm_warning
    from chanmerge.plan import Category
    from helpers import obs, make_plan
    letg = Category("ACIS-S", "LETG", "TE", "full")
    p = make_plan([obs(1796, letg, 0)])
    w = subarray_arm_warning(p, 1, 2)
    assert w and "row 1 is LETG" in w and "40%" in w and "narrow box chosen per observation" in w and "never merged with other observations" in w
    assert "not be used for spectra" in w
    assert "width_factor_letg 10" not in w
    w1 = subarray_arm_warning(p, 1, 1)
    assert w1 and "order-sorts the LETG events again" in w1 and "40%" not in w1
    sub = make_plan([obs(23307, Category("ACIS-S", "LETG", "TE", "sub"), 0)])
    assert "LETG subarray strip" in subarray_arm_warning(sub, 1, 4)
    hrc = make_plan([obs(14324, Category("HRC-S", "LETG", "-", "-"), 0)])
    assert subarray_arm_warning(hrc, 1, 2) is None         # no tg_m filter in hrc images


def test_row_set_chip_mix_counts_inside_one_image_only():
    # test d: 'separate' (bi / fi in different images) must not warn; a custom set holding S3 and S2 must
    from chanmerge.cli import row_set_checks
    from chanmerge.inputs import chip_choice
    from chanmerge.plan import RunPlan, Category, Observation
    p = RunPlan(ra=0, dec=0, radius_arcmin=1, target="t", root=".")
    p.categories = [Category("ACIS-I", "NONE", "TE", "full")]
    p.chips_by_row[1] = chip_choice("separate")
    _, warn = row_set_checks(p, [[1]])
    assert not any("mixes back- and front" in w for w in warn)
    p.chips_by_row[1] = chip_choice("custom", "S3,S2")
    _, warn = row_set_checks(p, [[1]])
    assert any("mixes back- and front" in w for w in warn)
    p.chips_by_row[1] = chip_choice("custom", "S3 / S2")
    _, warn = row_set_checks(p, [[1]])
    assert not any("mixes back- and front" in w for w in warn)



def test_manual_problem_edit_reruns_new_groups_too(tmp_path):
    # test e (2026-11-04): option 1, the edit drops the cc observation and makes a new group; the new group
    # used to be left out of the rerun without a word
    calls = []

    def fake_run(plan, on_header_problems=None, say=print, only_groups=None):
        calls.append(only_groups)
        r = Report()
        if len(calls) == 1:
            g = next(g for g in plan.groups if "6688" in g.obsids and len(g.obsids) > 1)
            r.manual_problems = [HeaderProblem("cc", "6688", g.group_id, True, "6688 is CC mode in its header")]
            r.failed_groups = {g.group_id: "manual group with a header problem; edit it and run it again"}
        else:
            r.images, r.kept_images = [{}, {}], 3
        return r
    answers = BASE + ["add 950 s6688", "done", "y", "y", "y",          # groups (s6688 -> g01 = 6688 + 950), warnings, start
                      "1", "drop 6688 g01", "new 950 27522", "done", "y", "y", "y", "y", "3"]
    t, out, prompts = drive(answers, tmp_path, run_fn=fake_run)
    assert len(calls) == 2 and calls[0] is None
    assert {"g01", "g02"} <= calls[1]                          # the problem group and the group the edit made
    assert "images: 2 (3 kept from the earlier pass)" in out
