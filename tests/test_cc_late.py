# an observation asked for flux whose header says cc (ocat down or disagreeing, so the mode was '?'): joins the table's cc row at the stage
# asked there, or gets stage none and is offered for deletion; manual groups with the problem wait (standby), the user is told first
import os

import numpy as np
import pytest
from astropy.io import fits

import chanmerge.run as R
from chanmerge.plan import Category, Group
from helpers import obs, make_plan, ACIS_S

Q = Category("ACIS-S", "HETG", "?", "?")              # mode and window unknown (ocat was down)
CC = Category("ACIS-S", "HETG", "CC", "full")         # a cc row of the category table
CC_OTHER = Category("ACIS-S", "NONE", "CC", "full")   # a cc row of another grating
CC_IDS = {"13197"}


def write_evt2(root, oid, grating=True, cc=False, size=2000):
    d = os.path.join(root, "observations", oid + ("_g" if grating else ""), "primary")
    os.makedirs(d, exist_ok=True)
    h = fits.Header()
    h["DATE-OBS"], h["DATE-END"], h["EXPOSURE"] = "2011-02-17T01:00:00", "2011-02-17T04:00:00", 10000.0
    h["INSTRUME"], h["READMODE"], h["DATAMODE"] = "ACIS", ("CONTINUOUS" if cc else "TIMED"), ("CC33_GRADED" if cc else "VFAINT")
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([fits.Column(name="x", format="E", array=np.zeros(size, "f4"))], header=h, name="EVENTS")])
    path = os.path.join(d, f"acisf{oid}_evt2.fits")
    hdul.writeto(path, overwrite=True)
    return path


class FakeCiao:
    def __init__(self):
        self.log_path = None
        self.events = []

    def download(self, parent, oid, grating):
        self.events.append(("download", oid))
        write_evt2(os.path.dirname(parent), oid, grating, cc=oid in CC_IDS)

    def repro(self, d):
        self.events.append(("repro", os.path.basename(d)))
        return "done"


def plan_with(tmp_path, cc_rows=(), manual=True):
    """13197 (mode '?', flux) in automatic s13197 and, if manual, in manual g01 together with 6688; optional cc rows with their stages"""
    obss = [obs(13197, Q, 0), obs(6688, ACIS_S, 1)]
    for k, (cat, stage) in enumerate(cc_rows):
        obss.append(obs(2000 + k, cat, 5 + k, stage=stage))
    p = make_plan(obss, window=7, tmp=str(tmp_path))
    p.ocat_ok = False
    p.groups = [Group("s13197", ["13197"], "A"), Group("s6688", ["6688"], "A")]
    if manual:
        p.groups.append(Group("g01", ["13197", "6688"], "A", manual=True))
    p.observations["13197"].stage = p.observations["6688"].stage = "flux"
    return p


def runner(plan):
    ciao = FakeCiao()
    lines = []
    return R.Runner(plan, ciao=ciao, say=lines.append), ciao, lines


def test_late_cc_joins_the_cc_row_at_the_stage_asked_there(tmp_path):
    for stage in ("repro", "download"):
        p = plan_with(tmp_path / stage, cc_rows=[(CC, stage)])
        r, ciao, lines = runner(p)
        r.fetch()
        o = p.observations["13197"]
        assert o.category == CC and o.stage == stage and o.mode_source == "header"     # next to the other cc observations
        assert any("joined the CC row" in n and stage in n for n in r.report.notes)
        assert r.report.cc_unwanted == []
        assert p.group("s13197") is None                                                # automatic group: left without it
        assert "13197" in p.group("g01").obsids                                         # manual group: untouched ...
        assert [(hp.group, hp.obsid) for hp in r.report.manual_problems] == [("g01", "13197")]    # ... and reported
        assert "13197" in r.needed(skip_standby=True)                                   # it is processed up to the stage asked


def test_the_highest_stage_of_the_cc_row_counts_and_a_row_of_another_grating_does_not(tmp_path):
    p = plan_with(tmp_path / "a", cc_rows=[(CC, "download"), (CC, "repro")])
    r, _, _ = runner(p)
    r.fetch()
    assert p.observations["13197"].stage == "repro"
    p2 = plan_with(tmp_path / "b", cc_rows=[(CC_OTHER, "repro")])                       # same detector, other grating: not this observation's row
    r2, _, _ = runner(p2)
    r2.fetch()
    assert p2.observations["13197"].stage == "none" and p2.observations["13197"].category == Q and len(r2.report.cc_unwanted) == 1


def test_without_a_cc_row_nothing_is_done_and_deleting_is_offered_only_for_what_this_run_downloaded(tmp_path):
    p = plan_with(tmp_path)
    r, ciao, lines = runner(p)
    r.fetch()
    o = p.observations["13197"]
    assert o.stage == "none" and o.category == Q
    (d,) = r.report.cc_unwanted
    assert d["obsid"] == "13197" and d["deletable"] and d["bytes"] > 0 and d["had_data"] is False
    assert "13197" not in r.needed() and "13197" not in r.needed(skip_standby=True)
    assert any("no CC row" in n for n in r.report.notes)
    text = R.cc_delete_question(p, d)
    assert "13197" in text and "No CC row" in text and "Delete these files?" in text
    assert R.delete_downloaded(p, d) is True
    assert not os.path.exists(d["path"]) and os.path.isdir(os.path.join(str(tmp_path), "observations"))
    assert R.delete_downloaded(p, d) is False                                          # already gone: nothing to do


def test_data_that_was_on_disk_before_the_run_is_never_deleted(tmp_path):
    p = plan_with(tmp_path)
    write_evt2(str(tmp_path), "13197", cc=True)                                        # e.g. downloaded earlier by another run
    r, ciao, _ = runner(p)
    r.fetch()
    (d,) = r.report.cc_unwanted
    assert d["had_data"] is True and d["deletable"] is False
    assert any("left untouched" in n for n in r.report.notes)
    assert R.delete_downloaded(p, d) is False and os.path.exists(d["path"])
    # and even a marked-deletable entry is refused when a finished repro is inside or the path is not directly in observations/
    d2 = dict(d, deletable=True)
    os.makedirs(os.path.join(d["path"], "repro"), exist_ok=True)
    open(os.path.join(d["path"], "repro", R.C.REPRO_MARKER), "w").write("x")
    assert R.delete_downloaded(p, d2) is False
    outside = tmp_path / "somewhere_else"
    outside.mkdir()
    assert R.delete_downloaded(p, dict(d, path=str(outside), deletable=True)) is False and outside.exists()


def test_a_known_cc_observation_or_a_non_flux_stage_is_left_alone(tmp_path):
    p = plan_with(tmp_path)
    p.observations["13197"].stage = "repro"                                            # asked for repro in the '?' row: cc is fine for that
    r, _, _ = runner(p)
    r.fetch()
    assert p.observations["13197"].stage == "repro" and p.observations["13197"].category == Q and r.report.cc_unwanted == []
    assert [hp.group for hp in r.report.manual_problems] == ["g01"]                    # but it still cannot sit in a manual merge group


def test_header_check_twice_does_not_repeat_notes_or_questions(tmp_path):
    p = plan_with(tmp_path)
    r, _, _ = runner(p)
    r.fetch()
    notes, n_unwanted = list(r.report.notes), len(r.report.cc_unwanted)
    r.header_check()
    assert r.report.notes == notes and len(r.report.cc_unwanted) == n_unwanted == 1
    assert [hp.group for hp in r.report.manual_problems] == ["g01"]                    # the manual problem is still there


def test_standby_group_does_not_pull_observations_into_repro(tmp_path):
    p = plan_with(tmp_path, cc_rows=[(CC, "repro")])
    p.groups = [Group("g01", ["13197", "6688"], "A", manual=True)]                      # 6688 only lives in the group with the problem
    r, ciao, _ = runner(p)
    r.fetch()
    assert "6688" in r.needed() and "6688" not in r.needed(skip_standby=True)          # waits with its group
    assert "13197" in r.needed(skip_standby=True)                                       # the cc observation is processed as its row asks
    p.groups.append(Group("s6688", ["6688"], "A"))                                      # an observation that another, fine group needs
    assert "6688" in r.needed(skip_standby=True)


def test_standby_notice_comes_before_any_repro(tmp_path, monkeypatch):
    p = plan_with(tmp_path, cc_rows=[(CC, "repro")])
    p.groups = [Group("g01", ["13197", "6688"], "A", manual=True)]
    monkeypatch.setattr(R.Runner, "merge_all", lambda self, only_groups=None: None)
    ciao, lines = FakeCiao(), []
    rep = R.run(p, ciao=ciao, say=lambda s: (lines.append(s), ciao.events.append(("say", s))))
    kinds = [e[0] for e in ciao.events]
    notice = next(i for i, e in enumerate(ciao.events) if e[0] == "say" and "NOT processed now" in e[1])
    assert notice < kinds.index("repro") and kinds.index("download") < notice          # after the download and header check, before repro
    assert any("g01: 13197 is CC mode in its header" in l for l in lines) and any("asked what to do" in l for l in lines)
    assert [hp.group for hp in rep.manual_problems] == ["g01"] and ("repro", "13197_g") in ciao.events
    assert ("repro", "6688") not in ciao.events                                          # the standby group's own member waits


def test_no_notice_when_no_manual_group_has_a_problem(tmp_path, monkeypatch):
    p = plan_with(tmp_path, cc_rows=[(CC, "repro")], manual=False)
    monkeypatch.setattr(R.Runner, "merge_all", lambda self, only_groups=None: None)
    ciao, lines = FakeCiao(), []
    R.run(p, ciao=ciao, say=lines.append)
    assert not any("NOT processed now" in l for l in lines)


def test_terminal_explains_asks_and_deletes_only_on_yes(tmp_path):
    from chanmerge.cli import Terminal
    for answer, left in (("y", False), ("n", True)):
        p = plan_with(tmp_path / answer)
        r, _, _ = runner(p)
        r.fetch()
        out = []
        asked = []
        t = Terminal(input_fn=lambda prompt: (asked.append(prompt), answer)[1], out=out.append, cwd=str(tmp_path))
        t.ask_cc_deletes(p, r.report)
        text = "\n".join(out)
        assert "13197 turned out to be CC mode" in text and "Delete these files?" in text and len(asked) == 1
        assert os.path.exists(r.report.cc_unwanted[0]["path"]) is left and ("kept" in text) is left and ("deleted" in text) is not left
    # data that was there before the run is never even asked about
    p = plan_with(tmp_path / "pre")
    write_evt2(str(tmp_path / "pre"), "13197", cc=True)
    r, _, _ = runner(p)
    r.fetch()
    out = []
    Terminal(input_fn=lambda prompt: pytest.fail("must not ask"), out=out.append, cwd=str(tmp_path)).ask_cc_deletes(p, r.report)
    assert out == []


def test_run_keeps_the_standby_line_up_during_the_run_and_takes_it_down_at_the_end(tmp_path, monkeypatch):
    p = plan_with(tmp_path, cc_rows=[(CC, "repro")])
    p.groups = [Group("g01", ["13197", "6688"], "A", manual=True)]
    monkeypatch.setattr(R.Runner, "merge_all", lambda self, only_groups=None: None)
    ciao = FakeCiao()
    shown = []
    R.run(p, ciao=ciao, say=lambda s: None, status=lambda text: (shown.append(text), ciao.events.append(("status", text))))
    kinds = [e[0] for e in ciao.events]
    assert shown[0] and "g01 is on hold" in shown[0] and "You will be asked" in shown[0] and shown[-1] is None
    assert kinds.index("status") < kinds.index("repro")                                   # up before the first repro starts
    assert ciao.events[-1] == ("status", None)                                            # down only when the run is over


def test_run_without_a_manual_problem_never_puts_a_line_up_and_always_takes_it_down(tmp_path, monkeypatch):
    p = plan_with(tmp_path, cc_rows=[(CC, "repro")], manual=False)
    monkeypatch.setattr(R.Runner, "merge_all", lambda self, only_groups=None: None)
    shown = []
    R.run(p, ciao=FakeCiao(), say=lambda s: None, status=shown.append)
    assert shown == [None]
    # even a crash inside the run takes the line down (the terminal must not keep a fixed scroll region)
    def boom(self, only_groups=None):
        raise RuntimeError("boom")
    monkeypatch.setattr(R.Runner, "merge_all", boom)
    shown.clear()
    with pytest.raises(RuntimeError):
        R.run(plan_with(tmp_path / "x", cc_rows=[(CC, "repro")]), ciao=FakeCiao(), say=lambda s: None, status=shown.append)
    assert shown[-1] is None


# ---------- ocat and the header disagree on the window (test x, session 007: ngc 1068 329 is a 128-row subarray, ocat says none) ----------
def test_header_window_wins_over_ocat_and_is_reported(tmp_path):
    from helpers import ACIS_S_HETG
    p = make_plan([obs(329, ACIS_S_HETG, 0), obs(16703, ACIS_S_HETG, 1)], window=7, tmp=str(tmp_path))
    p.groups = [Group("g01", ["329", "16703"], "A")]
    for o in ("329", "16703"):
        p.observations[o].stage = "flux"
    path = write_evt2(str(tmp_path), "329")
    with fits.open(path, mode="update") as h:
        h["EVENTS"].header["NROWS"] = 128
    write_evt2(str(tmp_path), "16703")
    r = R.Runner(p, ciao=FakeCiao(), say=lambda *a: None)
    r._read_header("329"); r._read_header("16703")
    assert r.window_of("329") == "sub" and r.window_of("16703") == "full"
    assert p.observations["329"].category.window == "full"                      # stays in its table row
    assert any("329: Ocat says full frame but its header says a 128-row subarray" in n for n in r.report.notes)
    probs = r.header_check()
    assert [hp.kind for hp in probs] == ["full_sub"] and probs[0].group == "g01"  # the mix only the header shows is caught
