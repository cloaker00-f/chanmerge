# native-python checks on real chandra data already on disk (no ciao, no network). skipped when absent.
import glob
import os

import numpy as np
import pytest
from astropy.io import fits

from chanmerge import fill as F
from chanmerge import masks as M
from chanmerge.correct import correct_member, merged_recompute
from chanmerge.geometry import fov_lookup, SKY_ARCSEC

LT = "/Users/cloaker/Desktop/CAFIG/chanmerge/latest-test"
M27522 = f"{LT}/te-grating-test/manual_test_27522"
REG27522 = f"{LT}/te-grating-test/27522_g/repro/acisf27522_tgmask.fits"
M950 = f"{LT}/chanmergetest/obs_181431.08_-170926.2/merged_final_image_broad"

need27522 = pytest.mark.skipif(not os.path.exists(f"{M27522}/27522_reproj_evt.fits"), reason="27522 test data not on disk")
need950 = pytest.mark.skipif(not os.path.exists(f"{M950}/merged_obs_950_reproj_evt.fits"), reason="950 test data not on disk")


def member(folder, stem):
    c = fits.getdata(f"{folder}/{stem}_broad_thresh.img").astype(float)
    x = np.nan_to_num(fits.getdata(f"{folder}/{stem}_broad_thresh.expmap").astype(float))
    h = fits.getheader(f"{folder}/{stem}_broad_thresh.img")
    return c, x, h, f"{folder}/{stem}_reproj_evt.fits"


@need27522
@pytest.mark.parametrize("arm,streak", [(4, "fill"), (2, "fill"), (4, "mask"), (2, "mask")])
def test_27522_combinations_and_event_file(tmp_path, arm, streak):
    c, x, h, evt = member(M27522, "27522")
    reg = M.read_region_file(REG27522)
    assert reg is not None and len(reg["boxes"]) == 2
    r = correct_member("27522", c, x, h, evt, arm_method=arm, arm_region=reg, streak_action=streak)
    assert len(r.streak_lines) == 1 and r.streak_done == streak and r.arm_done
    assert r.changed
    exp = x > 0
    untouched = exp & ~r.filled & ~r.masked
    assert np.array_equal(r.counts[untouched], c[untouched])
    if arm == 2:
        assert 0.15 < r.masked.sum() / exp.sum() < 0.20            # ~17% of the area, as measured
    out = str(tmp_path / "evt.fits")
    F.correct_events(evt, h, r.counts, r.filled, r.masked, r.rpix, r.ring_of_xy(h["CDELT1P"]), out, 500, 7000)
    d2 = fits.getdata(out, "EVENTS")
    img = F.bin_events(d2, h, c.shape, 500, 7000)
    assert np.array_equal(img[r.expmap > 0], r.counts[r.expmap > 0])
    d0 = fits.getdata(evt, "EVENTS")
    assert (d2["tg_m"] == 1).sum() == (d0["tg_m"] == 1).sum()      # spectrum photons stay


@need950
def test_950_streak_fill_and_merged_recompute(tmp_path):
    c, x, h, evt = member(M950, "merged_obs_950")
    r = correct_member("950", c, x, h, evt, streak_action="fill")    # no coordinate: the source comes from the data
    assert len(r.streak_lines) == 1 and r.filled.sum() > 100 and not r.masked.any()
    mc = fits.getdata(f"{M950}/merged_obs_broad_thresh.img").astype(float)
    mx = np.nan_to_num(fits.getdata(f"{M950}/merged_obs_broad_thresh.expmap").astype(float))
    c2, x2 = merged_recompute(mc, mx, [(c, x)], [(r.counts, r.expmap)])
    assert np.array_equal(c2[~r.filled], mc[~r.filled])
    # far from the source the streak was 11-50x the ring level (measured); after the fill it must be far lower
    from chanmerge.geometry import image_sky_grid
    px, py = image_sky_grid(h, c.shape)
    far = r.filled & (np.hypot(px - r.centre[0], py - r.centre[1]) * SKY_ARCSEC > 150)
    assert far.sum() > 50
    assert c2[far].sum() < mc[far].sum() / 5


@need950
def test_fov_lookup_real_file():
    p = sorted(glob.glob(f"{LT}/chanmergetest/obs_181431.08_-170926.2/950/primary/*fov1.fits*"))[0]
    on, src, near = fov_lookup(p, 273.6295, -17.1573)
    assert on == (0, 1, 2, 3, 6, 7) and src[0] == 3 and not near


@need950
def test_streak_source_without_any_coordinate():
    # the source of 950's streak, found from the events alone, within ~1.5" of gx 13+1 (RA_TARG)
    from chanmerge.correct import load_events, image_events
    from chanmerge.streak import detect_streaks, streak_source
    from chanmerge.geometry import column_wcs, radec_to_sky
    evt = glob.glob(f"{LT}/chanmergetest/obs_181431.08_-170926.2/950/primary/*evt2.fits*")[0]
    ev, hd = load_events(evt)
    L = detect_streaks(ev)[0]
    x, y, how = streak_source(L, image_events(ev, 500, 7000))
    tx, ty = radec_to_sky(hd["RA_TARG"], hd["DEC_TARG"], *column_wcs(hd, "x", "y"))
    assert how == "psf wings on the streak line" and np.hypot(x - tx, y - ty) * SKY_ARCSEC < 1.5
