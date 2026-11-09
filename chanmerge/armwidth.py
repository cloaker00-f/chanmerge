"""Grating arm band of one observation, measured from its own filter-only events (session 007).

what is known (test r, test w):
- the dark corridor the tg_m filter leaves ends exactly at the ciao box edge (hetg v404 16702/3/4, 17701; letg 1790): inside the box the
  filter removes the arm and most of the background, outside it nothing. so the ciao box is the minimum band to treat.
- a bright source can leak arm light outside the box (cen x-3 27522). the leak is measured here per arm type (hetg: heg and meg; letg: its
  one box): perpendicular distance of every filter-only event from the arm axis, along three segments of the arm (400-1000, 1000-2000,
  2000-3500 sky px from the zero order), against the counts expected from the same-radius ring rate x exposure (pixels far from every arm),
  scaled to a far band of the same side (local reference: a ring reference alone gave flat side offsets 0.65-1.9 in v404). all segments of
  one arm type are added up and the excess outside the box is fitted with A * exp(-(d - edge) / lam) (poisson likelihood on a grid,
  A >= 0); detected when the likelihood ratio against A = 0 is above 11.8 (3 sigma for 2 parameters). tested on injected leaks in real v404
  data (session 006): A 0.7 found 100%, lam within about 30-50%, false alarms 0/120 on noise re-draws; weak leaks (A <= 0.3) are often missed.
- the 8 px image is too coarse for this (the corridor smears one pixel past the edge and fakes a deficit): events, 1 sky px.
"""
import numpy as np

from .geometry import sky_to_pixel_index

RING_PX = 16.0                 # sky px, rings of the same-radius reference
REF_GAP = 250.0                # reference pixels are at least box half width + this from every arm axis (v404: 300 px for meg)
SEGMENTS = ((400.0, 1000.0), (1000.0, 2000.0), (2000.0, 3500.0))   # sky px along the arm from the zero order
FAR_OFF = (150.0, 350.0)       # far band of the local reference, sky px beyond the box edge (hetg: 200-400 px from the axis)
STEP = 4.0                     # sky px, profile bins
INNER = 24.0                   # the profile starts this far inside the box edge (shown, not fitted)
SUB = 4                        # exposure sampled SUB x SUB per image pixel
MIN_SEG_POINTS = 400           # exposure sample points a segment needs
MIN_FAR_COUNTS = 30            # counts the far band of a segment needs
MIN_REF_PX = 30                # pixels a reference ring needs
CLIP = 5.0                     # sigma clip of bright pixels in the reference rings
LAM_GRID = np.geomspace(3.0, 250.0, 60)
A_GRID = np.linspace(0.0, 3.0, 301)
LR_DETECT = 11.8


def _axes(px, py, reg, b):
    t = np.radians(b["angle"])
    dx, dy = px - reg["x"], py - reg["y"]
    return dx * np.cos(t) + dy * np.sin(t), -dx * np.sin(t) + dy * np.cos(t)


def half_arms(reg):
    """(box index, half sign along the axis, side sign across it) for every half arm. hetg: the side away from the other arm only
    (the two arms cross at a small angle, the other arm would enter the profile); letg (one box): each side on its own, so every part is
    compared with a far band of its OWN side (test x, session 007: the letg arm runs ~170-190 rows from the chips' readout edge; the rows
    there have 5-30% fewer background counts, also in non-grating data and at 9-12 keV, and the far band of that side is off the chip;
    both sides together against the other side's far band faked a 0.8-0.9 deficit and could hide a weak leak). a side without its own far
    band is left out (measure skips it)"""
    boxes = reg["boxes"]
    out = []
    for i, b in enumerate(boxes):
        t = np.radians(b["angle"])
        for h in (1, -1):
            if len(boxes) == 1:
                out.append((i, h, 1))
                out.append((i, h, -1))
                continue
            ob = boxes[1 - i]
            vx, vy = np.cos(np.radians(ob["angle"])), np.sin(np.radians(ob["angle"]))
            if vx * np.cos(t) + vy * np.sin(t) < 0:
                vx, vy = -vx, -vy
            vx, vy = h * vx, h * vy
            occ = np.sign(-vx * np.sin(t) + vy * np.cos(t))
            out.append((i, h, int(-occ) if occ != 0 else 0))
    return out


def ring_rate_map(c, x, px, py, reg, ref):
    """rate per (ring around the zero order, side of the source) from the reference pixels, one 5 sigma clip; NaN where no rate"""
    r = np.hypot(px - reg["x"], py - reg["y"])
    ring = (r / RING_PX).astype(int)
    side = np.where(_axes(px, py, reg, reg["boxes"][-1])[0] >= 0, 1, -1)
    rmap = np.full(c.shape, np.nan)
    for h in (-1, 1):
        for k in np.unique(ring[ref & (side == h)]):
            sel = ref & (side == h) & (ring == k)
            xs = x[sel].sum()
            if sel.sum() < MIN_REF_PX or xs <= 0:
                continue
            rate = c[sel].sum() / xs
            keep = sel & (c <= rate * x + CLIP * np.sqrt(rate * x + 1))
            if keep.sum() < MIN_REF_PX or x[keep].sum() <= 0:
                continue
            rmap[(side == h) & (ring == k)] = c[keep].sum() / x[keep].sum()
    return rmap


def fit_leak(rel, O, El):
    """poisson grid fit of the excess outside the box: O ~ El * (1 + A exp(-(d - edge) / lam)) for 0 <= d - edge < far band.
    rel: bin edges relative to the box edge. returns dict(A, lam, lr, detected, lam_lo, lam_hi, excess)"""
    rel = np.asarray(rel, float)
    sel = (rel[:-1] >= -1e-6) & (rel[1:] <= FAR_OFF[0] + 1e-6)
    lo, hi = rel[:-1][sel], rel[1:][sel]
    o, e = np.asarray(O, float)[sel], np.asarray(El, float)[sel]
    if len(o) == 0 or e.sum() <= 0:
        return dict(A=0.0, lam=0.0, lr=0.0, detected=False, lam_lo=0.0, lam_hi=0.0, excess=0.0)
    g = np.array([lam * (np.exp(-lo / lam) - np.exp(-hi / lam)) / (hi - lo) for lam in LAM_GRID])    # bin-averaged shape
    mu = e[None, None, :] * (1 + A_GRID[None, :, None] * g[:, None, :])
    ll = (o[None, None, :] * np.log(np.maximum(mu, 1e-12)) - mu).sum(-1)
    ll0 = (o * np.log(np.maximum(e, 1e-12)) - e).sum()
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    lr = float(2 * (ll[i, j] - ll0))
    prof = 2 * (ll[i, j] - ll.max(axis=1))
    okl = LAM_GRID[prof < 1.0]
    return dict(A=float(A_GRID[j]), lam=float(LAM_GRID[i]), lr=lr, detected=bool(lr > LR_DETECT and A_GRID[j] > 0),
                lam_lo=float(okl.min()), lam_hi=float(okl.max()), excess=float((o - e).sum()))


def _profile_edges(half):
    start = max(half - INNER, 0.0)
    near = np.arange(start, half + FAR_OFF[0], STEP)
    return np.concatenate([near, [half + FAR_OFF[0], half + FAR_OFF[1]]])


def measure(c, x, px, py, header, reg, ex, ey, only_half=None):
    """measure the leak outside every arm box of one observation.
    c, x: filter-only counts and exposure of the member (merge_obs frame), px, py: sky x, y of its pixel centres, reg: arm region in the
    same frame, ex, ey: sky positions of its filter-only events in the band (tg_m 0 or Null).
    returns dict(boxes=[per box: dict(half, status 'leak' / 'none' / 'unmeasurable', segments, A, lam, lr, excess, rel, O, El)],
                 outside_share = share of the exposed pixels in the arm zone that lie outside every box, stray_px).
    only_half +1 / -1: only that half of every arm (tests: measure on one half, check the treatment on the other)"""
    c = np.asarray(c, float)
    x = np.nan_to_num(np.asarray(x, float))
    ok = x > 0
    r = np.hypot(px - reg["x"], py - reg["y"])
    base = ok & (r > 3 * reg["r"])
    axes = [_axes(px, py, reg, b) for b in reg["boxes"]]
    halves = [b["width"] / 2.0 for b in reg["boxes"]]
    far_from_arms = np.ones(c.shape, bool)
    inside_any = np.zeros(c.shape, bool)
    for (along, perp), b, hb in zip(axes, reg["boxes"], halves):
        far_from_arms &= np.abs(perp) >= hb + REF_GAP
        inside_any |= (np.abs(perp) <= hb) & (np.abs(along) <= b["length"] / 2)
    zone = base & (r >= SEGMENTS[0][0]) & (r < SEGMENTS[-1][1])
    outside_share = float((zone & ~inside_any).sum() / max(zone.sum(), 1))
    rmap = ring_rate_map(c, x, px, py, reg, base & far_from_arms)
    good = base & np.isfinite(rmap)
    # only pixels that can enter a profile are sampled (memory: a full frame has about a million pixels)
    near = np.zeros(c.shape, bool)
    for (along, perp), hb in zip(axes, halves):
        near |= (np.abs(perp) < hb + FAR_OFF[1] + 16) & (np.abs(along) >= SEGMENTS[0][0] - 16) & (np.abs(along) < SEGMENTS[-1][1] + 16)
    good &= near
    B = float(header["CDELT1P"])
    stray = stray_sources(c, x, px, py, reg, axes, halves, good, B)
    good &= ~stray
    bx, by, inimg = sky_to_pixel_index(ex, ey, header, c.shape)
    keep = inimg.copy()
    keep[inimg] = good[by[inimg], bx[inimg]]
    ex, ey = np.asarray(ex, float)[keep], np.asarray(ey, float)[keep]
    off = ((np.arange(SUB) + 0.5) / SUB - 0.5) * B
    gy, gx = np.nonzero(good)
    sx = (px[gy, gx][:, None, None] + off[None, None, :] + 0 * off[None, :, None]).ravel()
    sy = (py[gy, gx][:, None, None] + off[None, :, None] + 0 * off[None, None, :]).ravel()
    sE = np.repeat(rmap[gy, gx] * x[gy, gx] / SUB ** 2, SUB * SUB)
    out = []
    for i, (b, hb) in enumerate(zip(reg["boxes"], halves)):
        edges = _profile_edges(hb)
        ea, ep = _axes(ex, ey, reg, b)
        sa, sp = _axes(sx, sy, reg, b)
        O = np.zeros(len(edges) - 1); El = np.zeros(len(edges) - 1)
        nseg = 0
        for (bi, h, free) in half_arms(reg):
            if bi != i or (only_half is not None and h != only_half):
                continue
            for seg in SEGMENTS:
                if seg[0] > b["length"] / 2:
                    continue
                es = (h * ea >= seg[0]) & (h * ea < seg[1])
                ss = (h * sa >= seg[0]) & (h * sa < seg[1])
                if free:
                    es &= free * ep >= 0
                    ss &= free * sp >= 0
                if ss.sum() < MIN_SEG_POINTS:
                    continue
                o = np.histogram(np.abs(ep[es]), edges)[0].astype(float)
                e = np.histogram(np.abs(sp[ss]), edges, weights=sE[ss])[0]
                if o[-1] < MIN_FAR_COUNTS or e[-1] <= 0 or e[:-1].sum() <= 0:
                    continue
                O += o
                El += e * (o[-1] / e[-1])
                nseg += 1
        rel = edges - hb
        res = dict(half=hb, segments=nseg, rel=rel, O=O, El=El)
        if nseg == 0:
            res.update(status="unmeasurable", A=0.0, lam=0.0, lr=0.0, excess=0.0)
        else:
            f = fit_leak(rel, O, El)
            res.update(f, status="leak" if f["detected"] else "none")
        out.append(res)
    return dict(boxes=out, outside_share=outside_share, stray_px=int(stray.sum()))


def stray_sources(c, x, px, py, reg, axes, halves, good, B):
    """pixels OUTSIDE the boxes that hold another source (test r: a point source 269 px from the ngc 4151 arm faked arm light). a pixel is cut
    when it is 5 sigma above the mean rate of the pixels at the same distance from the axis, on the same side, in the same segment: a leak is
    smooth along the arm, a source is not. the pixel and its 8 neighbours go. pixels inside a box are never cut: the arm's own emission lines
    are bright single spots on the axis (capella) and cutting 'bright pixels' there cuts the arm."""
    cut = np.zeros(c.shape, bool)
    for (along, perp), hb in zip(axes, halves):
        d = np.abs(perp) - hb
        sel = good & (d > 0)
        if not sel.any():
            continue
        seg = np.full(c.shape, -1)
        for k, (lo, hi) in enumerate(SEGMENTS):
            seg[(np.abs(along) >= lo) & (np.abs(along) < hi)] = k
        sel &= seg >= 0
        key = (seg * 2 + (perp > 0)) * 10000 + np.floor(d / B).astype(int)
        ks = key[sel]
        cs, xs = c[sel], x[sel]
        for _ in range(2):
            uk, inv = np.unique(ks, return_inverse=True)
            rate = np.bincount(inv, cs) / np.maximum(np.bincount(inv, xs), 1e-30)
            mu = rate[inv] * xs
            hot = cs > mu + CLIP * np.sqrt(mu + 1)
            # second pass: the rate without the hot pixels
            cs2 = np.where(hot, 0.0, cs); xs2 = np.where(hot, 0.0, xs)
            rate = np.bincount(inv, cs2) / np.maximum(np.bincount(inv, xs2), 1e-30)
            mu = rate[inv] * xs
            hot = cs > mu + CLIP * np.sqrt(mu + 1)
        m = np.zeros(c.shape, bool)
        m[sel] = hot
        cut |= m
    if cut.any():
        grown = cut.copy()
        grown[1:, :] |= cut[:-1, :]; grown[:-1, :] |= cut[1:, :]
        g2 = grown.copy()
        g2[:, 1:] |= grown[:, :-1]; g2[:, :-1] |= grown[:, 1:]
        cut = g2
    return cut


def leak_width(lam):
    """sky px beyond the box edge that hold 90% of a fitted exponential leak"""
    return float(lam * np.log(10.0))


SHARE_MIN = 0.005             # leak shares below this are left alone (an exponential never reaches 0; cen x-3 would touch 83% of the image)


def leak_excess_share(perp, half, A, lam):
    """share of the counts at perpendicular distance perp that is leak: A g / (1 + A g), only outside the box and only as far as it was
    measured (the far band ends FAR_OFF[1] beyond the edge)"""
    d = np.asarray(perp, float) - half
    g = np.where((d >= 0) & (d < FAR_OFF[1]), A * np.exp(-np.clip(d, 0, None) / max(lam, 1e-6)), 0.0)
    s = g / (1.0 + g)
    return np.where(s >= SHARE_MIN, s, 0.0)


# ---------- the two ways to handle a measured leak (compared on cen x-3 27522 in session 007, test w) ----------
def widened_region(reg, meas, scale=1.0):
    """the arm region with every leaking box made wider by scale x 2 x (90% width of its leak); boxes without a leak stay as ciao made them"""
    new = dict(reg)
    new["boxes"] = [dict(b) for b in reg["boxes"]]
    for b, m in zip(new["boxes"], meas["boxes"]):
        if m["status"] == "leak":
            b["width"] = b["width"] + 2 * scale * leak_width(m["lam"])
    return new


def leak_share_map(px, py, reg, meas, along_min=0.0):
    """per image pixel: the share of its counts that is fitted leak (0 inside the boxes and where nothing leaks); the largest of the arms.
    along_min: no leak share closer than this to the zero order along the arm"""
    s = np.zeros(px.shape)
    for b, m in zip(reg["boxes"], meas["boxes"]):
        if m["status"] != "leak":
            continue
        along, perp = _axes(px, py, reg, b)
        sel = (np.abs(along) <= b["length"] / 2) & (np.abs(along) >= along_min) & (np.hypot(px - reg["x"], py - reg["y"]) > 3 * reg["r"])
        s = np.maximum(s, np.where(sel, leak_excess_share(np.abs(perp), b["width"] / 2.0, m["A"], m["lam"]), 0.0))
    return s


def thin_counts(c, share, seed):
    """subtract the leak without losing pixels: every count is kept with probability 1 - share (binomial thinning), so the expected
    excess goes and what stays is still poisson"""
    rng = np.random.default_rng(seed)
    k = np.asarray(c).astype(np.int64)
    out = k.copy()
    sel = (share > 0) & (k > 0)
    out[sel] = rng.binomial(k[sel], 1.0 - np.clip(share[sel], 0.0, 1.0))
    return out.astype(float)
