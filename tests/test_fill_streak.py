import numpy as np
import pytest
from astropy.io import fits

from chanmerge import fill as F
from chanmerge import masks as M
from chanmerge import streak as S
from chanmerge.geometry import image_sky_grid, SKY_ARCSEC


def header(shape=(100, 100), b=8.0, x0=3700.0, y0=3700.0):
    h = fits.Header()
    h["CRVAL1P"], h["CRPIX1P"], h["CDELT1P"] = x0, 0.5, b
    h["CRVAL2P"], h["CRPIX2P"], h["CDELT2P"] = y0, 0.5, b
    return h


# ---------- rings ----------
def test_ring_rates_flat_and_clip():
    rng = np.random.default_rng(0)
    x = np.full((80, 80), 100.0)
    c = rng.poisson(0.5 * x).astype(float)
    c[40, 40] += 500                                   # a point source in one ring
    yy, xx = np.indices(c.shape)
    rpix = (np.hypot(xx - 40, yy - 40) / 2).astype(int)
    R = F.ring_rates(c, x, rpix, np.zeros(c.shape, bool))
    assert np.median(list(R.values())) == pytest.approx(0.5, rel=0.05)
    assert R[0] < 2.0                                   # the 5 sigma clip removed the source pixel


def test_fallback_rule():
    x = np.ones((20, 20)); c = np.ones((20, 20))
    rpix = np.zeros((20, 20), int); rpix[:, 10:] = 1
    wide = np.zeros((20, 20), bool); wide[:, :10] = True        # ring 0 fully under the widened arms
    plain = np.zeros((20, 20), bool); plain[:, :5] = True
    R, used = F.rates_with_fallback(c, x, rpix, wide, plain)
    assert set(R) == {0, 1} and used == {"wide": 1, "plain": 1, "none": 0}
    plain[:, :10] = True
    R, used = F.rates_with_fallback(c, x, rpix, wide, plain)
    assert 0 not in R and used["none"] == 1


def test_correct_image_rules():
    rng = np.random.default_rng(1)
    x = np.full((30, 30), 50.0)
    c = rng.poisson(1.0 * x).astype(float)
    yy, xx = np.indices(c.shape)
    rpix = (np.hypot(xx - 15, yy - 15) / 2).astype(int)
    R = F.ring_rates(c, x, rpix, np.zeros(c.shape, bool))
    fill_reg = np.zeros(c.shape, bool); fill_reg[5:10, :] = True
    mask_reg = np.zeros(c.shape, bool); mask_reg[:, 5:7] = True
    cc, xx2, filled, masked = F.correct_image(c, x, rpix, [(fill_reg, 11)], [mask_reg], R)
    assert (xx2[mask_reg] == 0).all() and (cc[mask_reg] == 0).all()           # mask wins on the overlap
    assert not (filled & mask_reg).any()
    untouched = ~fill_reg & ~mask_reg
    assert np.array_equal(cc[untouched], c[untouched]) and np.array_equal(xx2[untouched], x[untouched])
    assert cc[filled].mean() == pytest.approx(50.0, rel=0.1)
    # order of the lists does not matter: one seed per region
    cc2, _, _, _ = F.correct_image(c, x, rpix, [(fill_reg, 11)], [mask_reg], R)
    assert np.array_equal(cc, cc2)
    # a fill in a ring without a rate becomes no data
    R2 = {k: v for k, v in R.items() if k != 0}
    centre = np.zeros(c.shape, bool); centre[15, 15] = True
    cc3, xx3, f3, m3 = F.correct_image(c, x, rpix, [(centre, 3)], [], R2)
    assert m3[15, 15] and xx3[15, 15] == 0 and not f3[15, 15]


def test_ring_fill_totals_fixed_and_unbiased():
    # ring totals are the expectation rounded down or up; per pixel the mean over seeds is the expectation
    rng = np.random.default_rng(5)
    lam = rng.uniform(0.02, 0.3, 400)
    ring = np.repeat(np.arange(8), 50)
    for seed in range(30):
        out = F.draw_ring_fill(lam, ring, seed)
        for k in range(8):
            e = lam[ring == k].sum()
            assert out[ring == k].sum() in (int(np.floor(e)), int(np.floor(e)) + 1)
    mean = np.mean([F.draw_ring_fill(lam, ring, s) for s in range(4000)], axis=0)
    assert mean.sum() == pytest.approx(lam.sum(), rel=0.01)
    assert np.abs(mean - lam).max() < 0.05


def test_ring_fill_same_seed_same_result_and_rings_independent():
    lam = np.full(300, 0.1)
    ring = np.repeat(np.arange(3), 100)
    a, b = F.draw_ring_fill(lam, ring, 7), F.draw_ring_fill(lam, ring, 7)
    assert np.array_equal(a, b)
    # extra pixels in ring 0 must not change rings 1 and 2 (old single row-order stream did)
    lam2 = np.concatenate([np.full(10, 0.1), lam])
    ring2 = np.concatenate([np.zeros(10, int), ring])
    c = F.draw_ring_fill(lam2, ring2, 7)
    assert np.array_equal(c[10:][ring > 0], a[ring > 0])


def test_ring_fill_cannot_run_away():
    # the 16703 case in miniature: many low-rate pixels; the old poisson draw could give 61 for 37 expected
    lam = np.full(700, 37.0 / 700)
    ring = np.zeros(700, int)
    totals = [F.draw_ring_fill(lam, ring, s).sum() for s in range(500)]
    assert set(totals) == {37}


def test_flux_and_stats():
    c = np.array([[2.0, 0.0]]); x = np.array([[4.0, 0.0]])
    f = F.flux_of(c, x)
    assert f[0, 0] == 0.5 and np.isnan(f[0, 1])
    assert F.expmap_stats(np.array([[0.0, 1.0, 2.0, 3.0]]))[1] == 2.0
    assert F.mask_codes(np.array([[True, False]]), np.array([[False, True]])).tolist() == [[2, 1]]


# ---------- masks ----------
def test_arm_mask_and_frames():
    h = header((60, 60), b=1.0, x0=0.0, y0=0.0)
    px, py = image_sky_grid(h, (60, 60))
    reg = dict(x=30.0, y=30.0, r=15.0, wcs=None,
               boxes=[dict(x=30.0, y=30.0, length=50.0, width=4.0, angle=0.0), dict(x=30.0, y=30.0, length=50.0, width=4.0, angle=90.0)])
    keep = M.arm_mask(px, py, reg, keep_source=True)
    gone = M.arm_mask(px, py, reg, keep_source=False)
    assert gone.sum() > keep.sum() and not keep[30, 30] and gone[30, 30]
    assert keep[30, 52] and keep[52, 30] and not keep[45, 45] and not keep[30, 35]
    # circle edge: source when the pixel centre is inside r (ciao's rule), even right at the edge
    d = np.hypot(px - 30.0, py - 30.0)
    inside_box = gone
    assert not keep[inside_box & (d <= 15.0)].any()
    assert keep[inside_box & (d > 15.0)].all()
    edge = inside_box & (d > 15.0 - M.ARM_MARGIN) & (d <= 15.0)
    assert edge.any() and not keep[edge].any()          # the old r - margin rule masked these
    # same frame -> unchanged
    wcs = ((200.0, -40.0), (4096.5, 4096.5), (-0.000136667, 0.000136667))
    r2 = M.region_to_frame(dict(reg, wcs=wcs), wcs)
    assert r2["x"] == pytest.approx(30.0) and r2["boxes"][1]["angle"] == pytest.approx(90.0, abs=1e-6)


def test_streak_strip_touch_rule():
    h = header((200, 200), b=8.0, x0=3200.0, y0=3200.0)
    px, py = image_sky_grid(h, (200, 200))
    line = dict(sx=4000.0, sy=4000.0, ux=0.0, uy=1.0, amin=-1000.0, amax=1000.0)
    strip, zone = M.streak_strip(line, px, py, 8.0)
    cols = np.unique(np.nonzero(strip)[1])
    # +-2.5" = +-5.1 sky px plus half a pixel -> 2 image columns around x=4000 (pixel edges at 3992-4000-4008)
    assert len(cols) == 2
    a = np.abs(py - 4000)
    assert not strip[a < 20 / SKY_ARCSEC].any()                 # starts 20" from the source
    assert zone.sum() > strip.sum()


# ---------- streak detection on synthetic events ----------
def synthetic_events(streak=True, n_bg=40000, n_streak=6000, seed=3):
    rng = np.random.default_rng(seed)
    n = n_bg + (n_streak if streak else 0)
    t = rng.uniform(0, 1, n)
    dither = 16 * np.sin(2 * np.pi * t * 7)                      # chip moves under the sky
    sx, sy = 4096.0, 4096.0
    x = rng.uniform(3600, 4600, n); y = rng.uniform(3600, 4600, n)
    if streak:
        x[n_bg:] = sx + rng.normal(0, 0.5, n_streak)               # along the readout column, through the source
        y[n_bg:] = rng.uniform(3600, 4600, n_streak)
    chipx = (x - 3600 + 12 - dither).astype(int) + 1
    chipy = (y - 3600 + 12).astype(int) + 1
    return dict(x=x, y=y, chipx=chipx.astype(float), chipy=chipy.astype(float),
                ccd_id=np.full(n, 7.0), energy=rng.uniform(500, 7000, n))


def test_detects_streak_and_line_through_source():
    lines = S.detect_streaks(synthetic_events(True))
    assert len(lines) == 1
    L = lines[0]
    assert S.line_distance(L, 4096.0, 4096.0) * SKY_ARCSEC < 1.0
    assert abs(L["ux"]) < 0.01                                   # along y


def test_no_streak_no_detection():
    assert S.detect_streaks(synthetic_events(False)) == []


def test_streak_source_rules():
    L = dict(sx=4000.0, sy=4000.0, ux=0.0, uy=1.0, amin=-500.0, amax=500.0)
    assert S.streak_source(L, zeroth_xy=(4001.0, 4100.0))[2] == "zeroth order"
    x, y, how = S.streak_source(L, obs_target_xy=(4002.0, 4050.0))      # target 1" from the line
    assert (x, y) == (4000.0, 4050.0) and "observation target" in how
    assert S.streak_source(L, obs_target_xy=(4020.0, 4050.0)) is None    # 10" away and no data: skip
    assert "rotbox(" in S.bkg_rotbox(L)


def test_streak_source_from_wings_synthetic():
    # a source with symmetric wings at y=4100 on a streak along y; no coordinate given
    rng = np.random.default_rng(4)
    n = 6000
    r = np.abs(rng.normal(0, 12, n)); t = rng.uniform(0, 2 * np.pi, n)
    ev = dict(x=4000 + r * np.cos(t), y=4100 + r * np.sin(t))
    sx = np.r_[ev["x"], 4000 + rng.normal(0, 0.5, 4000)]; sy = np.r_[ev["y"], rng.uniform(3500, 4500, 4000)]
    L = dict(sx=4000.0, sy=3900.0, ux=0.0, uy=1.0, amin=-600.0, amax=600.0)
    x, y, how = S.streak_source(L, ev=dict(x=sx, y=sy))
    assert how.startswith("psf wings") and abs(y - 4100) < 2 and x == 4000.0


# ---------- event file correction ----------
def _evt_file(path, n=4000, seed=5, grating=True):
    rng = np.random.default_rng(seed)
    x = rng.uniform(3700, 3700 + 8 * 40, n); y = rng.uniform(3700, 3700 + 8 * 40, n)
    e = rng.uniform(300, 9000, n)
    cols = [fits.Column(name="time", format="D", array=np.sort(rng.uniform(0, 1000, n))),
            fits.Column(name="x", format="E", array=x), fits.Column(name="y", format="E", array=y),
            fits.Column(name="energy", format="E", array=e),
            fits.Column(name="phas", format="9I", array=np.zeros((n, 9), np.int16))]
    if grating:
        tg = np.where(rng.random(n) < 0.3, 1, np.where(rng.random(n) < 0.5, 0, 99)).astype(np.int16)
        cols.append(fits.Column(name="tg_m", format="I", array=tg))
    t = fits.BinTableHDU.from_columns(cols, name="EVENTS")
    t.header["TSTART"], t.header["TSTOP"] = 0.0, 1000.0
    fits.HDUList([fits.PrimaryHDU(), t]).writeto(path, overwrite=True)
    return fits.getdata(path, "EVENTS")


def test_correct_events_matches_image_and_keeps_dispersed(tmp_path):
    p = str(tmp_path / "in_evt.fits")
    d = _evt_file(p)
    h = header((40, 40))
    c = F.bin_events(d, h, (40, 40), 500, 7000)
    x = np.full((40, 40), 10.0)
    px, py = image_sky_grid(h, (40, 40))
    rpix = F.ring_index(px, py, 3860.0, 3860.0, 8.0)
    R = F.ring_rates(c, x, rpix, np.zeros(c.shape, bool))
    fill_reg = np.zeros(c.shape, bool); fill_reg[10:20, 10:14] = True
    mask_reg = np.zeros(c.shape, bool); mask_reg[25:30, :] = True
    cc, xx, filled, masked = F.correct_events_inputs = F.correct_image(c, x, rpix, [(fill_reg, 2)], [mask_reg], R)
    out = str(tmp_path / "out_evt.fits")
    ring = lambda ex, ey: (np.hypot(ex - 3860.0, ey - 3860.0) / 8.0 / F.RING).astype(int)
    st = F.correct_events(p, h, cc, filled, masked, rpix, ring, out, 500, 7000)
    d2 = fits.getdata(out, "EVENTS")
    img = F.bin_events(d2, h, (40, 40), 500, 7000)
    assert np.array_equal(img, cc)                                  # bins back to the corrected image
    assert st["added"] == int((d2["FILLED"] == 1).sum()) > 0
    # dispersed events (tg_m=1) are never removed, also inside the corrected regions
    assert (d2["tg_m"] == 1).sum() == (d["tg_m"] == 1).sum()
    # masked pixels hold no image-filter events of any energy
    assert F.bin_events(d2, h, (40, 40), None, None, tg=True)[masked].sum() == 0
    # the original file is untouched
    assert np.array_equal(fits.getdata(p, "EVENTS")["x"], d["x"])


def test_common_columns_drops_phas_and_mismatch(tmp_path):
    a, b = str(tmp_path / "a.fits"), str(tmp_path / "b.fits")
    _evt_file(a, grating=False)
    cols = [fits.Column(name="time", format="D", array=np.zeros(3)), fits.Column(name="x", format="E", array=np.zeros(3)),
            fits.Column(name="y", format="E", array=np.zeros(3)), fits.Column(name="energy", format="D", array=np.zeros(3)),
            fits.Column(name="phas", format="25I", array=np.zeros((3, 25), np.int16))]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols, name="EVENTS")]).writeto(b)
    assert F.common_columns([a, b]) == ["time", "x", "y"]           # energy differs in format, phas always out
    F.add_filled_column(a, str(tmp_path / "a2.fits"))
    assert "FILLED" in fits.getdata(str(tmp_path / "a2.fits"), "EVENTS").columns.names


def test_corrected_flux_keeps_merge_obs_convention():
    orig = np.array([[0.0, 2.0, 2.0, 2.0]])          # pixel 0 outside the field: merge_obs wrote 0
    x0 = np.array([[0.0, 1.0, 1.0, 1.0]])
    c1 = np.array([[0.0, 4.0, 0.0, 6.0]]); x1 = np.array([[0.0, 2.0, 0.0, 3.0]])   # pixel 2 masked by chanmerge
    f = F.corrected_flux(orig, c1, x1, x0)
    assert f[0, 0] == 0.0 and f[0, 1] == 2.0 and np.isnan(f[0, 2]) and f[0, 3] == 2.0


def test_low_count_chip_is_searched_now():
    # 1000 background + 800 streak events (34 columns wide with this dither) = 1800, under the old 2000-event gate
    ev = synthetic_events(True, n_bg=1000, n_streak=800, seed=5)
    assert len(ev["x"]) < 2000 and S.searched_chips(ev) == (7,)
    assert len(S.detect_streaks(ev)) == 1
    assert S.detect_streaks(synthetic_events(False, n_bg=1800, seed=5)) == []


def test_report_lines_name_searched_and_found_chips():
    from chanmerge.cli import streak_search_lines
    lines = streak_search_lines({("9052", "broad"): ((0, 1, 2, 3, 6, 7), ()), ("950", "broad"): ((0, 1, 2, 3, 6, 7), (3,)),
                                 ("950", "acisreadcorr, 0.5-7 keV"): ((3,), (3,))})
    text = "\n".join(lines)
    assert "streak search (broad):" in text and "streak search (acisreadcorr, 0.5-7 keV):" in text
    assert "9052: searched I0 I1 I2 I3 S2 S3; streak found on: none" in text
    assert "950: searched I0 I1 I2 I3 S2 S3; streak found on: I3" in text


def test_region_from_table_letg_has_one_box():
    # test f: the LETG tgmask of 23307 / 23309 has a circle and ONE rotbox (HETG: two); it was refused as 'no mask'
    from astropy.io import fits
    from chanmerge.masks import region_from_table
    cols = [fits.Column(name="SHAPE", format="8A", array=np.array(["circle", "rotbox"])),
            fits.Column(name="X", format="D", array=np.array([4084.4, 4081.7])),
            fits.Column(name="Y", format="D", array=np.array([3915.3, 4095.3])),
            fits.Column(name="R", format="2D", array=np.array([[31.0, 0.0], [8178.5, 2899.2]])),
            fits.Column(name="ROTANG", format="D", array=np.array([0.0, 90.85]))]
    t = fits.BinTableHDU.from_columns(cols)
    reg = region_from_table(t.data, t.header)
    assert reg is not None and len(reg["boxes"]) == 1 and reg["boxes"][0]["width"] == pytest.approx(2899.2)
    only_circle = fits.BinTableHDU.from_columns(cols[:0] + [fits.Column(name=c.name, format=c.format, array=c.array[:1]) for c in cols])
    assert region_from_table(only_circle.data, only_circle.header) is None


def test_arm_share_counts_exposed_pixels_only():
    px, py = np.meshgrid(np.arange(0, 1000.0), np.arange(0, 1000.0))
    exp = np.ones(px.shape, bool)
    base = dict(x=500.0, y=500.0, r=20.0)
    narrow = dict(base, boxes=[dict(x=500.0, y=500.0, length=3000.0, width=100.0, angle=0.0)])
    wide = dict(base, boxes=[dict(x=500.0, y=500.0, length=3000.0, width=900.0, angle=0.0)])
    assert 0.10 < M.arm_share(px, py, narrow, exp) < 0.13        # 100 + 2 * 6 margin out of 1000
    assert M.arm_share(px, py, wide, exp) > 0.9
    assert M.arm_share(px, py, wide, np.zeros(px.shape, bool)) == 0.0


def _flat_events(tmp_path, lo, hi, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    cols = [fits.Column(name=k, format="E", array=rng.uniform(lo, hi, n)) for k in ("x", "y")]
    cols += [fits.Column(name="chipx", format="E", array=rng.uniform(1, 1024, n)),
             fits.Column(name="chipy", format="E", array=rng.uniform(1, 1024, n)),
             fits.Column(name="ccd_id", format="I", array=np.full(n, 7, np.int16)),
             fits.Column(name="energy", format="E", array=rng.uniform(500, 7000, n))]
    p = str(tmp_path / "e.fits")
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols, name="EVENTS")]).writeto(p, overwrite=True)
    return p


def test_a_box_over_40_percent_makes_the_observation_untreatable(tmp_path):
    # test w (session 007): letg 1790's ciao box covers 47% of the image and the filter darkened all of it; masking would take > 40%:
    # nothing is changed and the observation is marked untreatable (run.py takes it out of its group's merge, rule 3)
    from chanmerge.correct import correct_member, ARM_SHARE_CAP_FULL
    p = _flat_events(tmp_path, 3700, 4020)
    h = header((40, 40))                                     # 40 x 8 sky px = 320 sky px wide, from 3700
    rng = np.random.default_rng(7)
    c = rng.poisson(4.0, (40, 40)).astype(float)
    x = np.full((40, 40), 10.0)
    wide = dict(x=3860.0, y=3860.0, r=20.0, boxes=[dict(x=3860.0, y=3860.0, length=2000.0, width=200.0, angle=0.0)], wcs=None)
    narrow = dict(wide, boxes=[dict(x=3860.0, y=3860.0, length=2000.0, width=40.0, angle=0.0)])
    r = correct_member("1790", c, x, h, p, arm_method=2, arm_region=wide, arm_share_cap=ARM_SHARE_CAP_FULL)
    assert r.arm_done == "" and not r.masked.any() and np.array_equal(r.counts, c)
    assert "covers" in r.untreatable and "40%" in r.untreatable
    r = correct_member("1790", c, x, h, p, arm_method=1, arm_region=wide, arm_share_cap=ARM_SHARE_CAP_FULL)
    assert r.untreatable == "" and any("option 1 keeps it" in n for n in r.notes)    # option 1: stays, with a note
    r = correct_member("1790", c, x, h, p, arm_method=2, arm_region=narrow, arm_share_cap=ARM_SHARE_CAP_FULL)
    assert r.untreatable == "" and r.arm_done == "mask_keep_source" and r.masked.any()   # a real arm strip is still masked
    r = correct_member("1790", c, x, h, p, arm_method=2, arm_region=wide)    # no cap: subarray or unknown window
    assert r.untreatable == "" and r.arm_done == "mask_keep_source" and r.masked.sum() > 800


def test_a_box_covering_the_whole_image_is_untreatable_for_every_option(tmp_path):
    # test w: letg 2324 (box 3398 px, nothing outside it): the filter removed 50-100% of the events on every chip, background included
    from chanmerge.correct import correct_member, ARM_SHARE_CAP_FULL
    p = _flat_events(tmp_path, 3700, 4020)
    h = header((40, 40))
    c = np.random.default_rng(3).poisson(4.0, (40, 40)).astype(float)
    x = np.full((40, 40), 10.0)
    whole = dict(x=3860.0, y=3860.0, r=20.0, boxes=[dict(x=3860.0, y=3860.0, length=9000.0, width=3398.0, angle=24.0)], wcs=None)
    for method in (1, 2, 3, 4):
        for cap in (ARM_SHARE_CAP_FULL, None):        # full frame or subarray: the same
            r = correct_member("2324", c, x, h, p, arm_method=method, arm_region=whole, arm_share_cap=cap, arm_gate=True)
            assert "covers the whole exposed image" in r.untreatable and not r.changed
    r = correct_member("2324", c, x, h, p, arm_method=None, arm_region=None)
    assert r.untreatable == ""


def test_no_fixed_letg_factor_any_more():
    assert not hasattr(M, "letg_narrow") and not hasattr(M, "LETG_WIDTH_FACTOR")


def test_outside_share():
    px, py = np.meshgrid(np.arange(0, 1000.0), np.arange(0, 1000.0))
    exp = np.ones(px.shape, bool)
    base = dict(x=500.0, y=500.0, r=20.0)
    assert M.outside_share(px, py, dict(base, boxes=[dict(x=500.0, y=500.0, length=3000.0, width=100.0, angle=0.0)]), exp) == pytest.approx(0.9, abs=0.01)
    assert M.outside_share(px, py, dict(base, boxes=[dict(x=500.0, y=500.0, length=3000.0, width=2000.0, angle=30.0)]), exp) == 0.0


# ---------- is there an arm to clean? (test r, session 005) ----------
def arm_scene(arm_counts=0.0, strip_rows=None, seed=5):
    """an 800 x 400 image (sky px = image px, binsize 1), flat background of 0.05 counts per px, one arm along x through (400, 200),
    optionally with the image cut to a narrow strip around the axis (a subarray). returns px, py, counts, expmap, reg"""
    rng = np.random.default_rng(seed)
    shape = (400, 800)
    h = header(shape, b=1.0, x0=0.0, y0=0.0)
    px, py = image_sky_grid(h, shape)
    exp = np.ones(shape)
    if strip_rows is not None:
        exp[np.abs(py - 200.0) > strip_rows] = 0.0
    lam = np.full(shape, 0.05) + arm_counts * np.exp(-0.5 * ((py - 200.0) / 5.0) ** 2) / (5.0 * np.sqrt(2 * np.pi))
    counts = rng.poisson(lam * exp).astype(float)
    reg = dict(x=100.0, y=200.0, r=10.0, wcs=None, boxes=[dict(x=400.0, y=200.0, length=700.0, width=60.0, angle=0.0)])
    return px, py, counts, exp, reg


def test_arm_visibility_sees_a_clear_arm_and_a_flat_image():
    px, py, c, x, reg = arm_scene(arm_counts=2.0)
    v = M.arm_visibility(px, py, c, x, reg)
    assert v["detected"] is True and v["sigma"] > 10 and v["excess"] > 0
    px, py, c, x, reg = arm_scene(arm_counts=0.0)
    v = M.arm_visibility(px, py, c, x, reg)
    assert v["detected"] is False and abs(v["sigma"]) < M.ARM_GATE_SIGMA


def test_arm_visibility_a_dark_corridor_counts_too():
    # the tg_m filter leaves a dark corridor along the arm in HETG images (V404 Cyg): a clear deficit is also an arm to correct
    px, py, c, x, reg = arm_scene(arm_counts=0.0)
    c = c * np.where(np.abs(py - 200.0) < 25.0, 0.3, 1.0)
    v = M.arm_visibility(px, py, c, x, reg)
    assert v["detected"] is True and v["sigma"] < -M.ARM_GATE_SIGMA and v["excess"] < 0


def test_arm_visibility_undecided_without_a_usable_background():
    # a narrow subarray strip: no pixels 150-600 px from the axis, so the check cannot decide and the arm is treated as before
    px, py, c, x, reg = arm_scene(arm_counts=0.0, strip_rows=60)
    assert M.arm_visibility(px, py, c, x, reg)["detected"] is None


def test_arm_visibility_the_other_arm_does_not_count_as_background():
    # two arms at a small angle: the second arm's light must not be taken as background of the first one
    px, py, c, x, reg = arm_scene(arm_counts=2.0)
    reg["boxes"].append(dict(x=400.0, y=200.0, length=700.0, width=60.0, angle=10.0))
    v = M.arm_visibility(px, py, c, x, reg)
    assert v["detected"] is True


def test_correct_member_skips_the_arm_and_says_so_when_none_is_visible(tmp_path):
    from chanmerge.correct import correct_member
    n = 500
    rng = np.random.default_rng(9)
    cols = [fits.Column(name=k, format="E", array=rng.uniform(0, 800, n)) for k in ("x", "y")]
    cols += [fits.Column(name="chipx", format="E", array=rng.uniform(1, 1024, n)),
             fits.Column(name="chipy", format="E", array=rng.uniform(1, 1024, n)),
             fits.Column(name="ccd_id", format="I", array=np.full(n, 7, np.int16)),
             fits.Column(name="energy", format="E", array=rng.uniform(500, 7000, n))]
    p = str(tmp_path / "e.fits")
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols, name="EVENTS")]).writeto(p)
    h = header((400, 800), b=1.0, x0=0.0, y0=0.0)
    for arm_counts, visible in ((0.0, False), (2.0, True)):
        px, py, c, x, reg = arm_scene(arm_counts=arm_counts)
        res = correct_member("999", c, x, h, p, arm_method=2, arm_region=reg, arm_gate=True)
        if visible:
            assert res.arm_done == "mask_keep_source" and res.masked.any() and res.arm_visibility["detected"] is True
        else:
            assert res.arm_done == "" and not res.changed and res.arm_visibility["detected"] is False
            assert any("no grating arm is visible" in n and "no arm correction was made" in n for n in res.notes)
    px, py, c, x, reg = arm_scene(arm_counts=0.0)
    res = correct_member("999", c, x, h, p, arm_method=2, arm_region=reg)           # gate off (default): treated as before
    assert res.arm_done == "mask_keep_source"
