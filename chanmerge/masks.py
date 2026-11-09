"""Grating arm regions and the image pixel masks for arms and streaks."""
import glob
import os

import numpy as np

from .geometry import SKY_ARCSEC, column_wcs, transform_points, transform_angle

ARM_MARGIN = 6.0         # sky px: partly covered image pixels count as inside (tested)
RING_ARM_EXTRA = 30.0    # arcsec: arm light leaks this far beyond the boxes; kept out of ring statistics
STREAK_HALF = 2.5        # arcsec: half width of the streak strip
STREAK_INNER = 20.0      # arcsec: the strip starts this far from the source


def _f(v):
    return float(np.atleast_1d(v)[0])


def region_from_table(data, header):
    """rows: 0 circle (X, Y, R), then one rotbox per arm group (X, Y, R[0] length, R[1] width, ROTANG): HETG has two
    (heg, meg), LETG has ONE (test f: 23307, 23309). returns dict or None"""
    if data is None or len(data) < 2:
        return None
    names = [n.upper() for n in data.columns.names]
    if not {"X", "Y", "R", "SHAPE"} <= set(names):
        return None
    shapes = [str(s).strip().lower() for s in data["SHAPE"]]
    if "circle" not in shapes[0]:
        return None
    boxes = []
    for i in range(1, len(data)):
        if "box" not in shapes[i]:
            continue
        r = np.atleast_1d(data["R"][i])
        if len(r) < 2:
            continue
        ang = _f(data["ROTANG"][i]) if "ROTANG" in names else 0.0
        boxes.append(dict(x=_f(data["X"][i]), y=_f(data["Y"][i]), length=float(r[0]), width=float(r[1]), angle=ang))
    if len(boxes) < 1:
        return None
    try:
        wcs = column_wcs(header, "x", "y")
    except KeyError:
        wcs = None
    return dict(x=_f(data["X"][0]), y=_f(data["Y"][0]), r=_f(np.atleast_1d(data["R"][0])[0]), boxes=boxes[:2], wcs=wcs)


# there is no fixed letg box factor any more (session 007): the letg box of tg_create_mask (width_factor_letg 300; 599-3398 sky px seen) is
# where merge_obs's tg_m filter removed the events, so it is the band (letg 1790: dark corridor exactly as wide as the box, test r / test w);
# a fixed factor 10 (session 005) left most of 1790's corridor and 9-33% of the arm light of three other letg observations untouched.


def read_region_file(path, ext="REGION"):
    from astropy.io import fits
    try:
        with fits.open(path) as h:
            if ext not in h:
                return None
            return region_from_table(h[ext].data, h[ext].header)
    except (OSError, KeyError, ValueError):
        return None


def find_region_files(repro_dir, primary_dir):
    """first two steps of the mask source order: repro *_tgmask.fits, then the primary evt2 REGION block"""
    out = []
    for p in sorted(glob.glob(os.path.join(glob.escape(repro_dir), "*_tgmask.fits"))):
        out.append(("repro tgmask", p))
    for p in sorted(glob.glob(os.path.join(glob.escape(primary_dir), "*evt2.fits*"))):
        out.append(("primary evt2 REGION", p))
    return out


def region_to_frame(reg, wcs_to):
    """region from the observation's sky frame to the merge_obs frame (tangent point differs)"""
    if reg.get("wcs") is None or wcs_to is None:
        return dict(reg)
    wf = reg["wcs"]
    x, y = transform_points(reg["x"], reg["y"], wf, wcs_to)
    boxes = []
    for b in reg["boxes"]:
        bx, by = transform_points(b["x"], b["y"], wf, wcs_to)
        boxes.append(dict(b, x=float(bx), y=float(by), angle=transform_angle(b["x"], b["y"], b["angle"], wf, wcs_to)))
    return dict(reg, x=float(x), y=float(y), boxes=boxes, wcs=wcs_to)


def arm_mask(px, py, reg, keep_source=True, extra=0.0):
    """image pixels inside the arm boxes (+ margin + extra sky px); the source circle stays out if keep_source.
    box edges: a pixel the box only touches counts as box (arm light must go). circle edge: a pixel is source when its
    centre is inside the circle, ciao's own rule for image regions (dmcopy, measured 38/38 pixels). the old r - margin
    rule removed source photons: 27522 15.2%, 7490 4%, v404 0.8-2.2%; this one 0.2-1.8% (decided 2026-11-04)"""
    m = np.zeros(px.shape, bool)
    for b in reg["boxes"]:
        t = np.radians(b["angle"])
        dx, dy = px - b["x"], py - b["y"]
        along = dx * np.cos(t) + dy * np.sin(t)
        across = -dx * np.sin(t) + dy * np.cos(t)
        m |= (np.abs(along) <= b["length"] / 2 + ARM_MARGIN + extra) & (np.abs(across) <= b["width"] / 2 + ARM_MARGIN + extra)
    if keep_source:
        m &= np.hypot(px - reg["x"], py - reg["y"]) > reg["r"]
    return m


# is there an arm to clean at all? (test r, session 005): in the filter-only image of a faint source the arm is not there (BD+30 3639 LETG,
# 337 zero-order counts: 1 sigma), masking or filling then only takes away about 10% of the area. the arm of a bright source is clear
# (PKS 2155-304 LETG, 11,280 zero-order counts: 33 sigma more counts than the surroundings). in HETG images the tg_m filter leaves a dark corridor instead
# (V404 Cyg: fewer counts than the surroundings), so the check is two sided: the rate inside the box that would be treated (half width = the box
# half width) is compared with the rate far from the axis, and either a clear excess or a clear deficit means there is something to correct.
# it returns None when it cannot decide (no usable background, e.g. a narrow subarray strip): then the arm is treated as before.
ARM_GATE_SIGMA = 5.0
ARM_ZONE = (400.0, 3500.0)    # sky px along the arm, away from the zero order
ARM_BKG = (150.0, 600.0)      # sky px across the arm: where the background rate is taken
ARM_BKG_NEAR_AXIS = 40.0      # background pixels this close to any arm axis are left out (the other HETG arm crosses at a small angle)
ARM_MIN_BKG_PX = 200


def arm_visibility(px, py, counts, expmap, reg):
    """returns dict(detected True / False / None, sigma (signed), excess (counts above, negative = below, the expected background), expected, half): the strongest of the arms decides"""
    ok = expmap > 0
    dx, dy = px - reg["x"], py - reg["y"]
    zero = np.hypot(dx, dy) < 3 * reg["r"]
    axes, halves = [], []
    for b in reg["boxes"]:
        halves.append(b["width"] / 2.0)
        t = np.radians(b["angle"])
        axes.append((dx * np.cos(t) + dy * np.sin(t), np.abs(-dx * np.sin(t) + dy * np.cos(t))))
    near_axis = np.zeros(px.shape, bool)
    for _, perp in axes:
        near_axis |= perp < ARM_BKG_NEAR_AXIS
    best = None
    for (along, perp), half in zip(axes, halves):
        zone = ok & ~zero & (np.abs(along) > ARM_ZONE[0]) & (np.abs(along) < ARM_ZONE[1])
        bkg = zone & (perp >= ARM_BKG[0]) & (perp < ARM_BKG[1]) & ~near_axis
        core = zone & (perp < half)
        if bkg.sum() < ARM_MIN_BKG_PX or core.sum() < 20 or expmap[bkg].sum() <= 0:
            continue
        e_bkg, e_core = expmap[bkg].sum(), expmap[core].sum()
        rate = counts[bkg].sum() / e_bkg
        expected = rate * e_core
        excess = counts[core].sum() - expected
        var = max(expected + expected * (e_core / e_bkg), 1.0)      # poisson in the band plus the uncertainty of the background rate
        sig = excess / np.sqrt(var)
        if best is None or abs(sig) > abs(best["sigma"]):
            best = dict(sigma=float(sig), excess=float(excess), expected=float(expected), half=float(half))
    if best is None:
        return dict(detected=None, sigma=0.0, excess=0.0, expected=0.0)
    best["detected"] = bool(abs(best["sigma"]) >= ARM_GATE_SIGMA)
    return best


def arm_share(px, py, reg, exposed):
    """share of the exposed image pixels the arm boxes cover (source circle ignored). a full-frame HETG mask measured 17%;
    LETG tg masks measured 52-100% (test j), which is no arm strip any more"""
    n = int(exposed.sum())
    if n == 0:
        return 0.0
    return float((arm_mask(px, py, reg, keep_source=True) & exposed).sum() / n)


def outside_share(px, py, reg, exposed):
    """share of the exposed pixels beyond 3 zero-order radii that lie outside every box (no margin). 0 = the box covers the whole image
    (letg 2324, 55, 3089, letg subarrays); hetg full frame about 0.8"""
    sel = exposed & (np.hypot(px - reg["x"], py - reg["y"]) > 3 * reg["r"])
    n = int(sel.sum())
    if n == 0:
        return 0.0
    inside = np.zeros(px.shape, bool)
    for b in reg["boxes"]:
        t = np.radians(b["angle"])
        dx, dy = px - b["x"], py - b["y"]
        inside |= (np.abs(dx * np.cos(t) + dy * np.sin(t)) <= b["length"] / 2) & (np.abs(-dx * np.sin(t) + dy * np.cos(t)) <= b["width"] / 2)
    return float((sel & ~inside).sum() / n)


def line_coords(line, px, py):
    du, dv = px - line["sx"], py - line["sy"]
    return line["ux"] * du + line["uy"] * dv, -line["uy"] * du + line["ux"] * dv


def streak_strip(line, px, py, binsize, half_arcsec=STREAK_HALF, inner_arcsec=STREAK_INNER, centre=None):
    """image pixels whose square touches |across| <= half; from inner_arcsec out to the chip end.
    returns (strip, zone): zone = strip plus one pixel, kept out of ring statistics"""
    a, p = line_coords(line, px, py)
    ext = binsize / 2 * (abs(line["ux"]) + abs(line["uy"]))     # half pixel size measured across the line
    w = half_arcsec / SKY_ARCSEC
    if centre is not None:
        ca = line["ux"] * (centre[0] - line["sx"]) + line["uy"] * (centre[1] - line["sy"])
        a = a - ca
        amin, amax = line["amin"] - ca, line["amax"] - ca
    else:
        amin, amax = line["amin"], line["amax"]
    strip = (np.abs(p) <= w + ext) & (np.abs(a) >= inner_arcsec / SKY_ARCSEC) & (a >= amin - binsize) & (a <= amax + binsize)
    zone = np.abs(p) <= w + ext + binsize
    return strip, zone
