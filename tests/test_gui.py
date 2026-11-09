# drives the window programmatically with a fake archive and scripted dialogs; skipped without a display
import time

import pytest

tk = pytest.importorskip("tkinter")

from chanmerge.run import Report
from test_cli import fake_archive


class Answers:
    def __init__(self):
        self.seen = []

    def yesno(self, title, text):
        self.seen.append((title, text)); return True

    def yesnocancel(self, title, text):
        self.seen.append((title, text)); return False

    def info(self, title, text):
        self.seen.append((title, text))

    def error(self, text):
        self.seen.append(("error", text))


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    r.withdraw()
    yield r
    r.destroy()


def pump(root, cond, timeout=20):
    t0 = time.time()
    while not cond():
        root.update()
        time.sleep(0.02)
        assert time.time() - t0 < timeout, "timed out"


def test_gui_fills_the_same_plan_and_starts(root, tmp_path):
    from chanmerge.gui import App
    got = {}

    def fake_run(plan, on_header_problems=None, say=print, only_groups=None):
        got["plan"] = plan
        say("g01 merge_obs ... ok")
        r = Report(); r.images = [{}]
        return r
    ans = Answers()
    app = App(root, archive=fake_archive(), run_fn=fake_run, cwd=str(tmp_path), ask=ans)
    for k, v in dict(ra="273.6295", dec="-17.1573", radius="5", name="Cen A").items():
        app.v[k].set(v)
    app.on_search()
    assert "space" in app.target_msg.cget("text")               # forbidden character
    app.v["name"].set("CenA")
    app.on_search()
    pump(root, lambda: app.nb.tab(app.tabs["Categories"], "state") == "normal")
    p = app.plan
    assert len(p.categories) == 5
    # all rows to flux, cc refused
    app.cat_tree.selection_set([str(i) for i in range(1, 6)])
    app.stage_var.set("flux"); app.on_row_stage()
    assert p.observations["13197"].stage == "none" and "CC" in app.cat_msg.cget("text")
    app.cat_tree.selection_set(["5"]); app.stage_var.set("none"); app.on_row_stage()   # no hrc
    app.on_categories_next()
    assert app.cat_msg.cget("text") == ""                        # no stale message after a good next (test c)
    pump(root, lambda: "reading" not in app.chip_text.get("1.0", "end"))
    assert "target on" in app.chip_text.get("1.0", "end") and "unknown" not in app.chip_text.get("1.0", "end")
    app.acis_band.set("broad"); app.window.set("3")
    for r in app.chip_vars:
        app.chip_vars[r].set("all")
    app.on_flux_next()
    assert "grating arm" in app.flux_msg.cget("text")            # no default for the arms
    for r in app.arm_vars:
        app.arm_vars[r].set(app_value(app, 4))
    app.on_flux_next()
    assert p.arm_method_by_row and set(p.arm_method_by_row.values()) == {4}
    app.rowsets.set("separate"); app.streak_sets.set("all"); app.streak_actions.set("A=fill, B=fill, C=mask")
    app.on_build_groups()
    assert [rs.streak_action for rs in p.row_sets] == ["fill", "fill", "mask"] and p.groups
    app.edit.set("add 950 s6688"); app.on_edit()
    assert "added" in app.groups_msg.cget("text")
    app.on_check()
    assert app.ok_checked and any(t == "Shared data" for t, _ in ans.seen)
    app.on_start()
    pump(root, lambda: "plan" in got and "images: 1" in app.log_text.get("1.0", "end"))
    # the start question names the downloads and always shows the sizes (confirmation, test c)
    assert any(t == "Start" and "will be downloaded" in x and "Download:" in x and "free" in x for t, x in ans.seen)
    assert got["plan"] is p and p.window_days == 3


def app_value(app, n):
    from chanmerge.cli import ARM_OPTIONS
    return f"{n}. {ARM_OPTIONS[n - 1][0]}"


def test_chip_menu_text_is_complete(root, tmp_path):
    # test c found "bi (back-illuminated chips only (S1)" cut at the comma
    from chanmerge.gui import App
    app = App(root, archive=fake_archive(), cwd=str(tmp_path), ask=Answers())
    labels = []

    def walk(w):
        for c in w.winfo_children():
            try:
                labels.append(c.cget("text"))
            except Exception:
                pass
            walk(c)
    walk(app.tabs["Flux"])
    text = "\n".join(str(t) for t in labels)
    assert "back-illuminated chips only (S1, S3)" in text and "front-illuminated chips only (S0, S2, S4, S5, I0-I3)" in text



def test_manual_group_problem_is_shown_after_the_run(root, tmp_path):
    # test e (2026-11-04): a manual group left out by the header check was only one report line in the window
    from chanmerge.gui import App
    from chanmerge.run import HeaderProblem

    def fake_run(plan, on_header_problems=None, say=print, only_groups=None):
        r = Report(); r.images = [{}]
        r.manual_problems = [HeaderProblem("cc", "6688", "g01", True, "6688 is CC mode in its header")]
        r.failed_groups = {"g01": "manual group with a header problem; edit it and run it again"}
        return r
    ans = Answers()
    app = App(root, archive=fake_archive(), run_fn=fake_run, cwd=str(tmp_path), ask=ans)
    from chanmerge.plan import RunPlan
    app.plan = RunPlan(ra=1.0, dec=2.0, radius_arcmin=5, target="T", root=str(tmp_path))
    # called directly: a worker thread under a second tk root hangs root.update() in the test process
    app.after_run(app.plan, fake_run(app.plan))
    text = dict(ans.seen)["Groups not merged"]
    assert "g01: 6688 is CC mode" in text and "drop those observations" in text
    assert app.fix is not None and app.fix_frame.winfo_manager() == "pack"


def _fix_app(root, tmp_path):
    from chanmerge.gui import App
    from chanmerge.run import HeaderProblem
    from chanmerge.plan import Group
    from helpers import obs, make_plan, ACIS_S
    app = App(root, archive=fake_archive(), run_fn=lambda *a, **k: Report(), cwd=str(tmp_path), ask=Answers())
    app.plan = make_plan([obs(6688, ACIS_S, 0), obs(9052, ACIS_S, 1), obs(950, ACIS_S, 40)], window=7, row_sets=[[1]], tmp=str(tmp_path))
    app.plan.groups = [Group("g01", ["6688", "9052"], "A", manual=True), Group("g02", ["950"], "A")]
    started = []
    app._start_run = lambda only_groups=None: started.append(only_groups)      # the rerun itself is tested in test_run
    rep = Report(); rep.images = [{}]
    rep.manual_problems = [HeaderProblem("cc", "6688", "g01", True, "6688 is CC mode in its header")]
    app.after_run(app.plan, rep)
    return app, started


def test_window_choice_2_drops_the_observation_and_reruns_the_problem_group(root, tmp_path):
    app, started = _fix_app(root, tmp_path)
    app.fix_drop()
    assert app.plan.group("g01").obsids == ["9052"]
    assert started == [{"g01"}] and app.plan.overwrite and app.fix is None


def test_window_choice_1_waits_for_check_plan_then_runs_changed_groups(root, tmp_path):
    from chanmerge.grouping import apply_edit
    from chanmerge.inputs import parse_group_edit
    app, started = _fix_app(root, tmp_path)
    app.fix_edit()
    assert app.fix_go.winfo_manager() == "pack"
    apply_edit(app.plan, parse_group_edit("new 950 9052"))                      # an edit that makes a new group
    app.fix_run()                                                               # plan check not pressed yet
    assert started == [] and app.fix is not None
    app.ok_checked = True
    app.fix_run()
    assert len(started) == 1 and "g01" in started[0] and any(g.startswith("g0") and g not in ("g01", "g02") for g in started[0])


def test_window_choice_3_leaves_them(root, tmp_path):
    app, started = _fix_app(root, tmp_path)
    app.fix_leave()
    assert app.fix is None and app.fix_frame.winfo_manager() == "" and started == []


def _app_with_unwanted_cc(root, tmp_path, answer):
    import os
    from chanmerge.gui import App
    from chanmerge.plan import RunPlan
    d = tmp_path / "observations" / "13197_g" / "primary"
    d.mkdir(parents=True)
    (d / "x_evt2.fits").write_bytes(b"x" * 5000)

    class Ask(Answers):
        def yesno(self, title, text):
            self.seen.append((title, text)); return answer
    ans = Ask()
    app = App(root, archive=fake_archive(), cwd=str(tmp_path), ask=ans)
    app.plan = RunPlan(ra=1.0, dec=2.0, radius_arcmin=5, target="T", root=str(tmp_path))
    rep = Report(); rep.images = [{}]
    rep.cc_unwanted = [dict(obsid="13197", path=str(tmp_path / "observations" / "13197_g"), bytes=5000, had_data=False, deletable=True)]
    return app, ans, rep


def test_window_asks_before_deleting_a_cc_observation_nobody_asked_for(root, tmp_path):
    import os
    app, ans, rep = _app_with_unwanted_cc(root, tmp_path, answer=True)
    app.after_run(app.plan, rep)
    (title, text), = ans.seen
    assert title == "Delete downloaded data" and "13197 turned out to be CC mode" in text and "Delete these files?" in text
    assert not os.path.exists(tmp_path / "observations" / "13197_g")
    assert "deleted the data of 13197" in app.log_text.get("1.0", "end")


def test_window_keeps_the_data_when_the_answer_is_no(root, tmp_path):
    import os
    app, ans, rep = _app_with_unwanted_cc(root, tmp_path, answer=False)
    app.after_run(app.plan, rep)
    assert os.path.exists(tmp_path / "observations" / "13197_g" / "primary" / "x_evt2.fits")
    assert "kept the data of 13197" in app.log_text.get("1.0", "end")


def test_window_does_not_ask_about_data_that_was_there_before_the_run(root, tmp_path):
    app, ans, rep = _app_with_unwanted_cc(root, tmp_path, answer=True)
    rep.cc_unwanted[0].update(had_data=True, deletable=False)
    app.after_run(app.plan, rep)
    assert ans.seen == [] and (tmp_path / "observations" / "13197_g").exists()


def test_window_shows_the_standby_bar_while_the_run_goes_on_and_hides_it_after(root, tmp_path):
    from chanmerge.gui import App
    app = App(root, archive=fake_archive(), cwd=str(tmp_path), ask=Answers())
    assert app.standby_label.winfo_manager() == ""
    app.set_standby("[!] g01 is on hold (problem). You will be asked about it after the other jobs.")
    assert app.standby_label.winfo_manager() == "pack" and "g01" in app.standby_label.cget("text")
    app.set_standby(None)
    assert app.standby_label.winfo_manager() == ""


def test_window_run_reports_the_standby_line_through_the_queue_and_only_the_real_run_gets_it(root, tmp_path):
    from chanmerge.gui import App
    app = App(root, archive=fake_archive(), cwd=str(tmp_path), ask=Answers())
    kw = app.run_kw()
    assert set(kw) == {"status"}
    kw["status"]("[!] g01 is on hold (problem).")                                 # called from the run's thread: it only queues
    assert app.standby_label.winfo_manager() == ""
    kind, fn = app.q.get_nowait()
    assert kind == "call"
    fn()                                                              # the main thread runs it
    assert app.standby_label.winfo_manager() == "pack"
    kw["status"](None)
    app.q.get_nowait()[1]()
    assert app.standby_label.winfo_manager() == ""
    assert App(root, archive=fake_archive(), run_fn=lambda *a, **k: None, cwd=str(tmp_path), ask=Answers()).run_kw() == {}
