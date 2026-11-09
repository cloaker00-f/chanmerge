# the per-observation arm band (session 007): leak outside the ciao box measured from the member's filter-only events
import numpy as np
import pytest
from astropy.io import fits

from chanmerge import armwidth as AW
from chanmerge.geometry import image_sky_grid, sky_to_pixel_index


def header(shape, b=8.0, x0=0.0, y0=0.0):
    h = fits.Header()
    h["CRVAL1P"], h["CRPIX1P"], h["CDELT1P"] = x0, 0.5, b
    h["CRVAL2P"], h["CRPIX2P"], h["CDELT2P"] = y0, 0.5, b
    return h


def scene(A=0.0, lam=40.0, half=50.0, angles=(0.0,), density=0.004, corridor=0.15, halo=0.0, side_offset=1.0,
          strip=None, stray=None, seed=3):
    """filter-only events of a fake grating observation: zero order at (4000, 1200), arms along the given angles, flat background
    (events per sky px^2) or with a halo that falls with radius, the tg_m filter keeps only `corridor` of the events inside the boxes,
    an exponential leak outside the boxes A * exp(-(d - half) / lam), one side of the source `side_offset` times brighter (chips),
    optionally the exposure cut to a strip of +-strip sky px around the axis (subarray) and a stray point source (x, y, counts).
    returns c, x, px, py, header, reg, ex, ey"""
    rng = np.random.default_rng(seed)
    W, H = 8000.0, 2400.0
    shape = (int(H / 8), int(W / 8))
    h = header(shape)
    px, py = image_sky_grid(h, shape)
    reg = dict(x=4000.0, y=1200.0, r=30.0, wcs=None,
               boxes=[dict(x=4000.0, y=1200.0, length=7800.0, width=2 * half, angle=a) for a in angles])

    def lam_at(x, y):
        r = np.hypot(x - reg["x"], y - reg["y"])
        rate = density * (1 + halo * np.exp(-r / 1500.0))
        rate = rate * np.where(x > reg["x"], side_offset, 1.0)
        f = np.ones_like(rate)
        for b in reg["boxes"]:
            t = np.radians(b["angle"])
            d = np.abs(-(x - b["x"]) * np.sin(t) + (y - b["y"]) * np.cos(t))
            f = np.where(d <= half, np.minimum(f, corridor), f)
        leak = np.zeros_like(rate)
        for b in reg["boxes"]:
            t = np.radians(b["angle"])
            d = np.abs(-(x - b["x"]) * np.sin(t) + (y - b["y"]) * np.cos(t))
            leak = np.maximum(leak, np.where(d > half, A * np.exp(-(d - half) / lam), 0.0))
        return rate * f * (1 + leak)

    # events by thinning a uniform field at the highest rate
    top = density * (1 + halo) * max(side_offset, 1.0) * (1 + A)
    n = rng.poisson(top * W * H)
    x = rng.uniform(0, W, n); y = rng.uniform(0, H, n)
    keep = rng.uniform(0, top, n) < lam_at(x, y)
    ex, ey = x[keep], y[keep]
    if stray is not None:
        sx, sy, sn = stray
        ex = np.concatenate([ex, rng.normal(sx, 4.0, sn)]); ey = np.concatenate([ey, rng.normal(sy, 4.0, sn)])
    expo = np.ones(shape)
    if strip is not None:
        expo[np.abs(py - reg["y"]) > strip] = 0.0
        bx, by, ok = sky_to_pixel_index(ex, ey, h, shape)
        sel = ok.copy(); sel[ok] = expo[by[ok], bx[ok]] > 0
        ex, ey = ex[sel], ey[sel]
    c = np.zeros(shape)
    bx, by, ok = sky_to_pixel_index(ex, ey, h, shape)
    np.add.at(c, (by[ok], bx[ok]), 1)
    # exposure in 'counts per unit rate': the rate unit is events per sky px^2, a pixel is 64 sky px^2
    return c, expo * 64.0, px, py, h, reg, ex, ey


def test_no_leak_with_a_corridor_is_not_a_leak():
    m = AW.measure(*scene(A=0.0))
    b = m["boxes"][0]
    assert b["status"] == "none" and b["segments"] >= 4
    assert m["outside_share"] > 0.5


def test_a_cen_x3_like_leak_is_found_and_its_width_roughly_measured():
    m = AW.measure(*scene(A=0.7, lam=40.0))
    b = m["boxes"][0]
    assert b["status"] == "leak" and b["lr"] > AW.LR_DETECT
    assert 20.0 < b["lam"] < 80.0 and 0.35 < b["A"] < 1.4


def test_halo_gradient_and_a_brighter_side_do_not_fake_a_leak():
    # v404 (session 006): flat side offsets 0.65-1.9 against a ring-only reference; the local far band of the same side takes them out
    for seed in (1, 2, 3):
        m = AW.measure(*scene(A=0.0, halo=3.0, side_offset=1.6, seed=seed))
        assert m["boxes"][0]["status"] == "none"


def test_two_hetg_arms_are_measured_separately():
    m = AW.measure(*scene(A=0.7, lam=30.0, half=40.0, angles=(5.0, -5.0)))
    assert [b["status"] for b in m["boxes"]] == ["leak", "leak"]
    m = AW.measure(*scene(A=0.0, half=40.0, angles=(5.0, -5.0)))
    assert [b["status"] for b in m["boxes"]] == ["none", "none"]


def test_a_stray_point_source_beside_the_arm_is_not_a_leak():
    # test r: a point source 269 px from the ngc 4151 arm; here 60 px past the box edge, 3000 counts
    m = AW.measure(*scene(A=0.0, stray=(5500.0, 1200.0 + 110.0, 3000)))
    assert m["boxes"][0]["status"] == "none" and m["stray_px"] > 0


def test_bright_spots_on_the_axis_are_never_cut():
    # capella: emission lines are bright single spots on the arm axis; only pixels outside the box may be cut as stray sources
    c, x, px, py, h, reg, ex, ey = scene(A=0.0)
    rng = np.random.default_rng(1)
    for xs in (4600.0, 5200.0, 6100.0):
        ex = np.concatenate([ex, rng.normal(xs, 2.0, 2000)]); ey = np.concatenate([ey, rng.normal(1200.0, 2.0, 2000)])
    c = np.zeros(c.shape)
    bx, by, ok = sky_to_pixel_index(ex, ey, h, c.shape)
    np.add.at(c, (by[ok], bx[ok]), 1)
    m = AW.measure(c, x, px, py, h, reg, ex, ey)
    assert m["stray_px"] == 0 and m["boxes"][0]["status"] == "none"


def test_a_narrow_subarray_strip_cannot_be_measured():
    m = AW.measure(*scene(A=0.7, strip=120.0))
    assert m["boxes"][0]["status"] == "unmeasurable" and m["boxes"][0]["segments"] == 0


def test_a_box_that_covers_the_whole_image_leaves_nothing_outside():
    # letg 2324 / 55 / 3089: the ciao box covers the whole exposed image
    m = AW.measure(*scene(A=0.0, half=1300.0))
    assert m["outside_share"] == 0.0 and m["boxes"][0]["status"] == "unmeasurable"


def test_fit_leak_and_shares():
    rel = np.arange(-24.0, 151.0, 4.0)
    El = np.full(len(rel) - 1, 400.0)
    mids = 0.5 * (rel[:-1] + rel[1:])
    O = El * (1 + np.where(rel[:-1] >= 0, 0.8 * np.exp(-mids / 30.0), 0.0))
    f = AW.fit_leak(rel, O, El)
    assert f["detected"] and f["A"] == pytest.approx(0.8, abs=0.1) and f["lam"] == pytest.approx(30.0, rel=0.15)
    assert not AW.fit_leak(rel, El.copy(), El)["detected"]
    s = AW.leak_excess_share(np.array([10.0, 50.0, 50.0 + 30.0]), 50.0, 0.8, 30.0)
    assert s[0] == 0.0 and s[1] == pytest.approx(0.8 / 1.8) and s[2] == pytest.approx(0.8 * np.exp(-1) / (1 + 0.8 * np.exp(-1)))
    assert AW.leak_width(40.0) == pytest.approx(92.1, abs=0.1)


def test_letg_side_near_the_chip_edge_does_not_hide_a_leak():
    # test x (session 007): the letg arm runs close to the readout edge; that side has ~25% less background and no far band of its own.
    # each side is now measured on its own: the edge side is left out, the leak on the other side is found
    c, x, px, py, h, reg, ex, ey = scene(A=0.5, lam=30.0, half=100.0)
    rng = np.random.default_rng(4)
    low = (ey < reg["y"])
    drop = low & (rng.random(len(ey)) < 0.25)
    ex, ey = ex[~drop], ey[~drop]
    x = x.copy()
    x[py < reg["y"] - 100.0 - 60.0] = 0.0          # the chip ends 60 px beyond the box on the low side
    bx, by, ok = sky_to_pixel_index(ex, ey, h, c.shape)
    keep = ok.copy(); keep[ok] = x[by[ok], bx[ok]] > 0
    ex, ey = ex[keep], ey[keep]
    c = np.zeros(c.shape)
    bx, by, ok = sky_to_pixel_index(ex, ey, h, c.shape)
    np.add.at(c, (by[ok], bx[ok]), 1)
    m = AW.measure(c, x, px, py, h, reg, ex, ey)
    b = m["boxes"][0]
    # measured with the fix A 0.40-0.44, lam 33-41 for A 0.5, lam 30 (seeds 4-6); both sides against one far band: A 0.24-0.34, lam 13-20
    assert b["status"] == "leak" and b["segments"] == 6 and 0.36 < b["A"] < 0.8 and b["lam"] > 22
