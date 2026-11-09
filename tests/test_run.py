# run(plan) end to end with a fake ciao: downloads, repro and merge_obs are replaced by links to the real
# 27522 files on disk; dmmerge is done in python. checks layout, names, corrections, tables, index, punlearn.
import glob
import os
from datetime import datetime, timezone

import numpy as np
import pytest
from astropy.io import fits

from chanmerge.ciao import Ciao
from chanmerge.plan import RunPlan, Observation, Category, RowSet, ChipChoice, ChipSet
from chanmerge.bands import parse_bands
from chanmerge.grouping import build_groups
from chanmerge.run import run
from chanmerge.tables import read_csv, INDEX_FINAL, INDEX_MEMBERS
from chanmerge import fill as F

LT = "/Users/cloaker/Desktop/CAFIG/chanmerge/latest-test"
OBS27522 = f"{LT}/te-grating-test/27522_g"
MAN = f"{LT}/te-grating-test/manual_test_27522"
pytestmark = pytest.mark.skipif(not os.path.exists(f"{MAN}/27522_reproj_evt.fits"), reason="27522 data not on disk")


class FakeCiao:
    def __init__(self):
        self.calls = []

    def __call__(self, args, cwd=None, env=None, timeout=None):
        self.calls.append(args)
        tool = args[0]
        if tool == "download_chandra_obsid":
            d = os.path.join(cwd, args[1], "primary")
            os.makedirs(d, exist_ok=True)
            for f in glob.glob(f"{OBS27522}/primary/*evt2.fits*"):
                dst = os.path.join(d, os.path.basename(f).replace("27522", f"{int(args[1]):05d}"))
                if not os.path.exists(dst):
                    os.symlink(f, dst)
            return 0, "ok"
        if tool == "chandra_repro":
            out = args[2].split("=", 1)[1]
            os.makedirs(out, exist_ok=True)
            for f in glob.glob(f"{OBS27522}/repro/*"):
                b = os.path.basename(f)
                if b.endswith((".fits",)) and ("repro_evt2" in b or "asol1" in b or "bpix1" in b or "msk1" in b or "tgmask" in b):
                    dst = os.path.join(out, b)
                    if not os.path.exists(dst):
                        os.symlink(f, dst)
            return 0, "ok"
        if tool == "merge_obs":
            outroot = args[2]
            members = [ln.split("/observations/")[1].split("/")[0].replace("_g", "") for ln in open(args[1][1:]).read().split()]
            for f in glob.glob(f"{MAN}/*"):
                b = os.path.basename(f)
                if b.startswith("27522_"):
                    for o in members:
                        os.symlink(f, f"{outroot}_{o}_" + b[len("27522_"):])
                elif b in ("broad_flux.img", "broad_thresh.img", "broad_thresh.expmap", "merged_evt.fits"):
                    os.symlink(f, f"{outroot}_{b}")
            return 0, "merged"
        if tool == "dmmerge":
            lis = [ln for ln in open(args[1][1:]).read().splitlines() if ln.strip()]
            out = args[2]
            tabs = []
            for item in lis:
                path, filt = item.split("[cols ")
                cols = filt.rstrip("]").split(",")
                with fits.open(path) as h:
                    d = h["EVENTS"].data
                    tabs.append({c: np.asarray(d[c]) for c in cols})
                    hdr = h["EVENTS"].header
            cols = list(tabs[0])
            merged = {c: np.concatenate([t[c] for t in tabs]) for c in cols}
            with fits.open(lis[0].split("[")[0]) as h:
                fmt = {c.name: c.format for c in h["EVENTS"].columns}
            t = fits.BinTableHDU.from_columns([fits.Column(name=c, format=fmt[c], array=merged[c]) for c in cols], name="EVENTS")
            fits.HDUList([fits.PrimaryHDU(), t]).writeto(out, overwrite=True)
            return 0, "ok"
        return 0, "ok"


def plan_27522(tmp_path, obsids=("27522",), arm=4, streak="fill", window=3.0, mode="TE", win="full"):
    root = str(tmp_path / "root")
    os.makedirs(root)
    p = RunPlan(ra=170.3129, dec=-60.6238, radius_arcmin=5, target="CenX3", root=root)
    cat = Category("ACIS-S", "HETG", mode, win)
    for i, o in enumerate(obsids):
        p.observations[o] = Observation(o, cat, datetime(2022, 11, 26 + i, 18, tzinfo=timezone.utc), 9755.0, stage="flux",
                                        active_chips=(5, 6, 7, 8))
    p.categories = [cat]
    p.acis_bands, _ = parse_bands("broad")
    p.chips_by_row = {1: ChipChoice("all", [ChipSet(None)])}
    p.arm_method_by_row = {1: arm}
    p.window_days = window
    p.row_sets = [RowSet("A", [1], streak)]
    p.groups = build_groups(p)
    return p


def test_single_grating_obs_arm_fill_and_streak_fill(tmp_path):
    p = plan_27522(tmp_path)
    fake = FakeCiao()
    rep = run(p, ciao=Ciao(str(tmp_path / "pf"), runner=fake), say=lambda *a: None)
    tools = [c[0] for c in fake.calls]
    i = tools.index("chandra_repro")
    assert tools[i + 1] == "punlearn" and fake.calls[i + 1] == ["punlearn", "ardlib"]
    assert rep.failed_groups == {} and rep.failed_obs == {}
    assert rep.streak.get("27522") == "fill" and rep.arms.get("27522") == "mask_fill"
    rd = os.path.join(p.root, "CenX3", "merged_final_image_broad_w3d")
    rows = read_csv(os.path.join(rd, "final_images.csv"))
    assert len(rows) == 1
    r = rows[0]
    assert r["group_id"] == "s27522" and r["type"] == "single" and r["chips"] == "all"
    assert r["flux_image"].endswith("CenX3_w3d_s27522_broad_flux_filled.img")
    assert r["flux_image_original"].endswith("CenX3_w3d_s27522_broad_flux.img")
    assert r["evt_file"].endswith("CenX3_w3d_s27522_merged_evt_filled.fits")
    assert r["evt_file_original"].endswith("CenX3_w3d_s27522_merged_evt.fits")
    assert r["repro_evt_file"] == "../../observations/27522_g/repro/acisf27522_repro_evt2.fits"
    assert os.path.isabs(r["repro_evt_file_abs"]) and os.path.exists(r["repro_evt_file_abs"])
    assert r["streak_action"] == "fill" and r["streak_obsids"] == "27522"
    assert r["arm_method"] == "mask_fill" and r["arm_filled_obsids"] == "27522" and r["arm_masked_obsids"] == ""
    assert 0.1 < float(r["filled_pixel_fraction"]) < 0.25
    assert r["categories"] == "ACIS-S/HETG/TE/full" and r["n_grating"] == "1"
    assert r["start_utc"].startswith("2022-11-26") and float(r["total_exposure_ks"]) == pytest.approx(9.8, abs=0.1)
    folder = os.path.dirname(os.path.join(rd, r["flux_image"]))
    assert os.path.basename(folder) == "s27522_2022-11-26"
    # the new merged event list bins back to the corrected merged counts
    h = fits.getheader(os.path.join(folder, "CenX3_w3d_s27522_broad_thresh_filled.img"))
    cc = fits.getdata(os.path.join(folder, "CenX3_w3d_s27522_broad_thresh_filled.img")).astype(float)
    xx = fits.getdata(os.path.join(rd, r["expmap_file"])).astype(float)
    d = fits.getdata(os.path.join(rd, r["evt_file"]), "EVENTS")
    img = F.bin_events(d, h, cc.shape, 500, 7000)
    assert np.array_equal(img[xx > 0], cc[xx > 0])
    # originals untouched: the merge_obs product is still the same file
    assert os.path.islink(os.path.join(rd, r["flux_image_original"]))
    # member table empty for a single, index written
    assert read_csv(os.path.join(rd, "member_images.csv")) == []
    idx = read_csv(os.path.join(p.root, INDEX_FINAL))
    assert idx[0]["flux_image"].startswith("CenX3/merged_final_image_broad_w3d/") and idx[0]["repro_evt_file"].startswith("observations/")
    assert os.path.exists(os.path.join(p.root, INDEX_MEMBERS))


def test_question_mark_row_still_gets_the_arm_treatment_from_the_header(tmp_path):
    # test p: a '?' row (Ocat down) kept mode '?' in the table and the arm option was recorded but never applied; the header says TE
    p = plan_27522(tmp_path, mode="?", win="?", streak="leave")
    rep = run(p, ciao=Ciao(str(tmp_path / "pf"), runner=FakeCiao()), say=lambda *a: None)
    assert rep.failed_groups == {} and rep.failed_obs == {}
    assert rep.arms.get("27522") == "mask_fill"
    rd = os.path.join(p.root, "CenX3", "merged_final_image_broad_w3d")
    r = read_csv(os.path.join(rd, "final_images.csv"))[0]
    assert r["arm_method"] == "mask_fill" and r["arm_filled_obsids"] == "27522"
    assert float(r["filled_pixel_fraction"]) > 0.1


def test_two_obs_group_member_rows_and_reuse(tmp_path):
    p = plan_27522(tmp_path, obsids=("27522", "27523"), arm=1, streak="leave")
    fake = FakeCiao()
    rep = run(p, ciao=Ciao(str(tmp_path / "pf"), runner=fake), say=lambda *a: None)
    tools = [c[0] for c in fake.calls]
    assert tools.count("chandra_repro") == 2 and tools.count("punlearn") == 2
    assert [tools[i + 1] for i, t in enumerate(tools) if t == "chandra_repro"] == ["punlearn", "punlearn"]
    rd = os.path.join(p.root, "CenX3", "merged_final_image_broad_w3d")
    f = read_csv(os.path.join(rd, "final_images.csv"))[0]
    assert f["group_id"] == "g01" and f["n_obs"] == "2" and f["obsids"] == "27522;27523"
    assert f["flux_image"] == f["flux_image_original"] and f["evt_file"] == f["evt_file_original"]   # nothing changed
    assert f["arm_method"] == "filter" and f["mask_file"] == ""
    assert len(f["repro_evt_file"].split(";")) == 2
    m = read_csv(os.path.join(rd, "member_images.csv"))
    assert [r["obsid"] for r in m] == ["27522", "27523"] and all(r["shares_photons_with"] == "" for r in m)
    # final times come from the headers (both fake members share 27522's header), not from heasarc
    assert os.path.basename(os.path.dirname(os.path.join(rd, f["flux_image"]))) == "g01_2022-11-26_2022-11-26"
    # second run: repro is reused (marker), nothing downloads, existing run folder needs overwrite
    p.overwrite = True
    fake2 = FakeCiao()
    rep2 = run(p, ciao=Ciao(str(tmp_path / "pf"), runner=fake2), say=lambda *a: None)
    t2 = [c[0] for c in fake2.calls]
    assert "chandra_repro" not in t2 and "download_chandra_obsid" not in t2
    assert sorted(rep2.reused) == ["27522", "27523"]


def test_existing_run_folder_without_overwrite_fails_cleanly(tmp_path):
    p = plan_27522(tmp_path, arm=1, streak="leave")
    run(p, ciao=Ciao(str(tmp_path / "pf"), runner=FakeCiao()), say=lambda *a: None)
    p.overwrite = False
    with pytest.raises(RuntimeError, match="overwrite"):
        run(p, ciao=Ciao(str(tmp_path / "pf"), runner=FakeCiao()), say=lambda *a: None)


def test_rerun_of_some_groups_keeps_the_others_and_clears_edited_ones(tmp_path):
    # test e (2026-11-04): after the manual-group header problem the user edits and only some groups run again.
    # new or changed groups must run too; groups edited away lose their rows and folders; the others stay as they were
    from chanmerge.run import group_state, groups_to_rerun
    p = plan_27522(tmp_path, obsids=("27522", "27523", "27524"), arm=1, streak="leave", window=0.5)
    assert [g.group_id for g in p.groups] == ["s27522", "s27523", "s27524"]
    run(p, ciao=Ciao(str(tmp_path / "pf"), runner=FakeCiao()), say=lambda *a: None)
    rd = os.path.join(p.root, "CenX3", "merged_final_image_broad_w0.5d")
    if not os.path.exists(rd):
        rd = glob.glob(os.path.join(p.root, "CenX3", "merged_final_image_broad_*"))[0]
    first = {r["group_id"]: r for r in read_csv(os.path.join(rd, "final_images.csv"))}
    keep_file = os.path.join(rd, first["s27522"]["flux_image"])
    t_keep = os.path.getmtime(keep_file)
    gone_dir = os.path.dirname(os.path.join(rd, first["s27524"]["flux_image"]))
    before = group_state(p)
    from chanmerge.plan import Group
    p.groups.append(Group("g01", ["27522", "27523"], "A", manual=True))       # new in the edit
    p.groups = [g for g in p.groups if g.group_id != "s27524"]                  # deleted in the edit
    only = groups_to_rerun(p, before, {"s27523"})
    assert only == {"g01", "s27523"}
    p.overwrite = True
    rep = run(p, ciao=Ciao(str(tmp_path / "pf"), runner=FakeCiao()), say=lambda *a: None, only_groups=only)
    rows = read_csv(os.path.join(rd, "final_images.csv"))
    assert sorted(r["group_id"] for r in rows) == ["g01", "s27522", "s27523"]
    assert sorted(r["group_id"] for r in rep.images) == ["g01", "s27523"] and rep.kept_images == 1
    assert os.path.getmtime(keep_file) == t_keep                                 # s27522 not touched
    assert not os.path.exists(gone_dir)
    assert any("s27524 is no longer in the plan" in n for n in rep.notes)
    m = read_csv(os.path.join(rd, "member_images.csv"))
    assert sorted({r["group_id"] for r in m}) == ["g01"]
    idx = read_csv(os.path.join(p.root, INDEX_FINAL))
    assert sorted(r["group_id"] for r in idx) == ["g01", "s27522", "s27523"]


# ---------- rule 3 (session 007): an observation that must not be mixed with the others leaves its group ----------
def _untreatable(monkeypatch, oid):
    import chanmerge.run as R
    real = R.correct_member

    def fake(o, *a, **k):
        if o != oid:
            return real(o, *a, **k)
        # like the real one: an untreatable observation is left exactly as merge_obs made it
        r = real(o, *a, **dict(k, arm_method=None, arm_region=None))
        r.untreatable = "test: its ciao arm box covers the whole exposed image"
        return r
    monkeypatch.setattr(R, "correct_member", fake)


def test_untreatable_member_leaves_an_automatic_group_and_gets_its_own_image(tmp_path, monkeypatch):
    _untreatable(monkeypatch, "27523")
    p = plan_27522(tmp_path, obsids=("27522", "27523"), arm=4, streak="leave")
    assert [g.group_id for g in p.groups] == ["g01"]
    fake = FakeCiao()
    rep = run(p, ciao=Ciao(str(tmp_path / "pf"), runner=fake), say=lambda *a: None)
    assert rep.failed_groups == {}
    # merge_obs: g01 once with both, then again with 27522 alone (now s27522), then s27523 alone
    merges = [c for c in fake.calls if c[0] == "merge_obs"]
    assert len(merges) == 3
    assert rep.left_out == {"27523": "g01"}
    rd = os.path.join(p.root, "CenX3", "merged_final_image_broad_w3d")
    rows = {r["group_id"]: r for r in read_csv(os.path.join(rd, "final_images.csv"))}
    assert sorted(rows) == ["s27522", "s27523"]
    assert rows["s27523"]["left_out_of"] == "g01" and rows["s27522"]["left_out_of"] == ""
    assert rows["s27523"]["flux_image"] == rows["s27523"]["flux_image_original"]     # untouched, filter only
    assert rows["s27522"]["arm_filled_obsids"] == "27522"
    assert not [d for d in os.listdir(rd) if d.startswith("g01")]                     # the first g01 folder is gone
    assert read_csv(os.path.join(rd, "member_images.csv")) == []
    idx = read_csv(os.path.join(p.root, INDEX_FINAL))
    assert {r["group_id"]: r["left_out_of"] for r in idx} == {"s27522": "", "s27523": "g01"}
    assert any("27523 left g01" in n for n in rep.notes) and any("is not arm-treated" in n and "27523" in n for n in rep.notes)
    assert not any(n.startswith("g01, ") for n in rep.notes)          # nothing from the thrown-away first pass


def test_untreatable_member_of_a_manual_group_stays_and_is_asked_about(tmp_path, monkeypatch):
    from chanmerge.plan import Group
    _untreatable(monkeypatch, "27523")
    p = plan_27522(tmp_path, obsids=("27522", "27523"), arm=4, streak="leave")
    p.groups = [Group("g01", ["27522", "27523"], "A", manual=True)]
    fake = FakeCiao()
    rep = run(p, ciao=Ciao(str(tmp_path / "pf"), runner=fake), say=lambda *a: None)
    assert len([c for c in fake.calls if c[0] == "merge_obs"]) == 1          # never changed automatically
    assert [(hp.obsid, hp.group, hp.kind) for hp in rep.arm_problems] == [("27523", "g01", "arm")]
    rd = os.path.join(p.root, "CenX3", "merged_final_image_broad_w3d")
    rows = read_csv(os.path.join(rd, "final_images.csv"))
    assert [r["group_id"] for r in rows] == ["g01"] and rows[0]["obsids"] == "27522;27523" and rows[0]["arm_filled_obsids"] == "27522"
    # the terminal's question at the end: 1 merges g01 again without 27523 and gives 27523 its own image
    from chanmerge.cli import Terminal
    answers = iter(["1"])
    t = Terminal(input_fn=lambda prompt: next(answers), out=lambda *a: None, cwd=str(tmp_path))
    t.status.set = lambda *a: None
    runs = []

    def run_fn(plan, only_groups=None, **k):
        runs.append(set(only_groups))
        return run(plan, ciao=Ciao(str(tmp_path / "pf"), runner=FakeCiao()), say=lambda *a: None, only_groups=only_groups)
    t.ask_arm_problems(p, rep, run_fn)
    assert runs == [{"g01", "s27523"}]
    rows = {r["group_id"]: r for r in read_csv(os.path.join(rd, "final_images.csv"))}
    assert sorted(rows) == ["g01", "s27523"] and rows["g01"]["obsids"] == "27522" and rows["s27523"]["left_out_of"] == "g01"
    # 2 keeps it: nothing runs
    rep.arm_problems = [rep.arm_problems[0]] if rep.arm_problems else []
    answers2 = iter(["2"])
    t2 = Terminal(input_fn=lambda prompt: next(answers2), out=lambda *a: None, cwd=str(tmp_path))
    runs2 = []
    t2.ask_arm_problems(p, rep, lambda *a, **k: runs2.append(1))
    assert runs2 == []


def test_real_leak_of_27522_is_subtracted_and_the_event_file_matches(tmp_path):
    # cen x-3 27522 leaks arm light outside the hetg box (test w: heg A 1.35, meg A 1.00): option 4 fills the box and subtracts the leak
    p = plan_27522(tmp_path, arm=4, streak="leave")
    rep = run(p, ciao=Ciao(str(tmp_path / "pf"), runner=FakeCiao()), say=lambda *a: None)
    assert any("leaks outside the arm box" in n and "subtracted" in n for n in rep.notes)
    rd = os.path.join(p.root, "CenX3", "merged_final_image_broad_w3d")
    r = read_csv(os.path.join(rd, "final_images.csv"))[0]
    codes = fits.getdata(os.path.join(rd, r["mask_file"]))
    assert (codes == 3).sum() > 1000 and (codes == 2).any()
    folder = os.path.dirname(os.path.join(rd, r["flux_image"]))
    th = os.path.join(folder, "CenX3_w3d_s27522_broad_thresh_filled.img")
    cc, h = fits.getdata(th).astype(float), fits.getheader(th)
    c0 = fits.getdata(os.path.join(rd, r["flux_image_original"]).replace("_flux.img", "_thresh.img")).astype(float)
    assert cc[codes == 3].sum() < c0[codes == 3].sum()                    # counts went down beside the box
    xx = fits.getdata(os.path.join(rd, r["expmap_file"])).astype(float)
    d = fits.getdata(os.path.join(rd, r["evt_file"]), "EVENTS")
    img = F.bin_events(d, h, cc.shape, 500, 7000)
    assert np.array_equal(img[xx > 0], cc[xx > 0])
