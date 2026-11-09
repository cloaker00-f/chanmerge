# real ciao on data already on disk (no downloads). run with: CHANMERGE_CIAO_TESTS=1 python -m pytest tests/test_ciao_local.py
# the test root holds symlinks to the existing repro/primary files and its own done-markers, so chanmerge never
# downloads, never reruns chandra_repro and never writes into the original folders.
import glob
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import pytest
from astropy.io import fits

from chanmerge import fill as F
from chanmerge.bands import parse_bands
from chanmerge.ciao import Ciao, REPRO_MARKER, ciao_available
from chanmerge.grouping import build_groups
from chanmerge.plan import RunPlan, Observation, Category, RowSet, ChipChoice, ChipSet
from chanmerge.run import run
from chanmerge.tables import read_csv

LT = "/Users/cloaker/Desktop/CAFIG/chanmerge/latest-test"
ROOT = os.environ.get("CHANMERGE_TEST_ROOT", f"{LT}/v2-test-root")
SRC = {"27522": f"{LT}/te-grating-test/27522_g", "950": f"{LT}/chanmergetest/obs_181431.08_-170926.2/950",
       "6688": f"{LT}/chanmergetest/obs_181431.08_-170926.2/6688", "9052": f"{LT}/chanmergetest/obs_181431.08_-170926.2/9052"}
pytestmark = pytest.mark.skipif(os.environ.get("CHANMERGE_CIAO_TESTS") != "1" or not ciao_available(),
                                reason="set CHANMERGE_CIAO_TESTS=1 with ciao initialised")


def seed(oid, grating):
    d = os.path.join(ROOT, "observations", oid + ("_g" if grating else ""))
    for sub in ("repro", "primary"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
        for f in glob.glob(os.path.join(SRC[oid], sub, "*")):
            dst = os.path.join(d, sub, os.path.basename(f))
            if os.path.isfile(f) and not os.path.lexists(dst):
                os.symlink(f, dst)
    open(os.path.join(d, "repro", REPRO_MARKER), "w").write("seeded for the local ciao test\n")


def base_plan(target, ra, dec, obs, window, row_sets, arm=None):
    p = RunPlan(ra=ra, dec=dec, radius_arcmin=5, target=target, root=ROOT, overwrite=True)
    for o in obs:
        p.observations[o.obsid] = o
    p.categories = sorted({o.category for o in obs}, key=lambda c: c.sort_key())
    p.acis_bands, _ = parse_bands("broad")
    p.chips_by_row = {r: ChipChoice("all", [ChipSet(None)]) for r in range(1, len(p.categories) + 1)}
    if arm:
        p.arm_method_by_row = {r: arm for r, c in enumerate(p.categories, 1) if c.is_grating}
    p.window_days = window
    p.row_sets = row_sets
    p.groups = build_groups(p)
    return p


def check_evt_matches(rd, row):
    folder = os.path.dirname(os.path.join(rd, row["flux_image"]))
    thresh = glob.glob(os.path.join(folder, "*_broad_thresh_*.img"))
    thresh = [t for t in thresh if not any(f"_{o}_" in os.path.basename(t) for o in row["obsids"].split(";"))][0]
    h = fits.getheader(thresh); cc = fits.getdata(thresh).astype(float)
    xx = np.nan_to_num(fits.getdata(os.path.join(rd, row["expmap_file"])).astype(float))
    d = fits.getdata(os.path.join(rd, row["evt_file"]), "EVENTS")
    img = F.bin_events(d, h, cc.shape, 500, 7000)
    return np.array_equal(img[xx > 0], cc[xx > 0]), int(np.abs(img - cc)[xx > 0].max())


def test_27522_real_merge_obs_arm_fill_streak_fill():
    seed("27522", True)
    cat = Category("ACIS-S", "HETG", "TE", "full")
    o = Observation("27522", cat, datetime(2022, 11, 26, 18, tzinfo=timezone.utc), 9755.0, stage="flux", active_chips=(5, 6, 7, 8))
    p = base_plan("CenX3", 170.3129, -60.6238, [o], 3.0, [RowSet("A", [1], "fill")], arm=4)
    rep = run(p, ciao=Ciao(os.path.join(ROOT, ".chanmerge", "pfiles_test")), say=print)
    assert rep.failed_groups == {}, rep.failed_groups
    rd = os.path.join(ROOT, "CenX3", "merged_final_image_broad_w3d")
    r = read_csv(os.path.join(rd, "final_images.csv"))[0]
    print(r)
    assert r["flux_image"].endswith("_broad_flux_filled.img") and r["evt_file"].endswith("_merged_evt_filled.fits")
    assert r["streak_obsids"] == "27522" and r["arm_filled_obsids"] == "27522"
    ok, mx = check_evt_matches(rd, r)
    assert ok, mx


def test_950_6688_9052_real_merge_streak_fill_and_dmmerge():
    for oid in ("950", "6688", "9052"):
        seed(oid, False)
    ci = Category("ACIS-I", "NONE", "TE", "full"); cs = Category("ACIS-S", "NONE", "TE", "full")
    obs = [Observation("950", ci, datetime(1998, 7, 19, tzinfo=timezone.utc), 9600, stage="flux", active_chips=(0, 1, 2, 3, 6, 7)),
           Observation("6688", cs, datetime(2006, 3, 1, tzinfo=timezone.utc), 5000, stage="flux", active_chips=(2, 3, 5, 6, 7, 8)),
           Observation("9052", ci, datetime(2008, 6, 1, tzinfo=timezone.utc), 5000, stage="flux", active_chips=(0, 1, 2, 3, 6, 7))]
    p = base_plan("GX13p1", 273.6295, -17.1573, obs, 0.0, [RowSet("A", [1, 2], "fill")])
    rep = run(p, ciao=Ciao(os.path.join(ROOT, ".chanmerge", "pfiles_test")), say=print)
    assert rep.failed_groups == {}, rep.failed_groups
    rd = os.path.join(ROOT, "GX13p1", "merged_final_image_broad_w0d")
    r = read_csv(os.path.join(rd, "final_images.csv"))[0]
    print(r, rep.notes)
    assert r["group_id"] == "all" and r["n_obs"] == "3" and r["streak_obsids"] == "950"
    assert r["evt_file"].endswith("_merged_evt_filled.fits")
    m = read_csv(os.path.join(rd, "member_images.csv"))
    assert len(m) == 3 and [x["evt_file"].endswith("_evt_filled.fits") for x in m] == [True, False, False]
    # the right comparison (merge_obs's own list differs from the merged counts where members were clipped):
    # new merged list == merge_obs merged list - 950's original reprojected events + 950's corrected events
    folder = os.path.dirname(os.path.join(rd, r["flux_image"]))
    h = fits.getheader(glob.glob(os.path.join(folder, "*_all_broad_thresh.img"))[0])
    shape = fits.getdata(glob.glob(os.path.join(folder, "*_all_broad_thresh.img"))[0]).shape
    b = lambda f: F.bin_events(fits.getdata(f, "EVENTS"), h, shape, 500, 7000)
    new = b(os.path.join(rd, r["evt_file"]))
    expected = b(os.path.join(rd, r["evt_file_original"])) - b(os.path.join(rd, m[0]["evt_file_original"])) + b(os.path.join(rd, m[0]["evt_file"]))
    assert np.array_equal(new, expected)
    ok, mx = check_evt_matches(rd, r)
    print("merged evt == corrected merged counts (differs only where merge_obs clipped members):", ok, mx)


def test_950_real_acisreadcorr_in_run():
    # acisreadcorr before merge_obs, position = entered coordinate on the streak line, dx=16
    seed("950", False)
    ci = Category("ACIS-I", "NONE", "TE", "full")
    o = Observation("950", ci, datetime(2000, 8, 7, tzinfo=timezone.utc), 9600, stage="flux", active_chips=(0, 1, 2, 3, 6, 7))
    p = base_plan("GX13p1_arc", 273.6295, -17.1573, [o], 3.0, [RowSet("A", [1], "acisreadcorr")])
    rep = run(p, ciao=Ciao(os.path.join(ROOT, ".chanmerge", "pfiles_test")), say=print)
    print(rep.notes, rep.failed_groups)
    assert rep.failed_groups == {} and rep.streak.get("950") == "acisreadcorr"
    rd = os.path.join(ROOT, "GX13p1_arc", "merged_final_image_broad_w3d")
    r = read_csv(os.path.join(rd, "final_images.csv"))[0]
    assert r["streak_action"] == "acisreadcorr" and r["streak_obsids"] == "950"
    # one shared acisreadcorr file per target (not inside a group folder), named in the tables
    rc = os.path.join(ROOT, "GX13p1_arc", "acisreadcorr", "GX13p1_arc_950_acisreadcorr_evt.fits")
    assert os.path.exists(rc)
    assert os.path.normpath(os.path.join(rd, r["acisreadcorr_evt_file"])) == rc and r["acisreadcorr_evt_file_abs"] == rc
    # the original repro file is untouched (still the symlink to the source data)
    assert os.path.islink(os.path.join(ROOT, "observations", "950", "repro", "acisf00950_repro_evt2.fits"))
