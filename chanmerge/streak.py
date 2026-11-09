"""Readout streak: detection on chip columns, the line refitted on the streak's own profile, acisreadcorr position.

thresholds were tuned on 950, 27522, 15511 (streaks) and 6688, 9052, 7490, 17704, v404 (none) - see test results.
every chip with events is searched (the old 2000-event gate was dropped 2026-11-04: with it off, no false streak in
10 observations and 4000 synthetic chips; a streak needs ~300 events to be found).
"""
import numpy as np

from .geometry import SKY_ARCSEC

MIN_WIDTH = 15       # a dithered streak is at least this many chip columns wide
MAX_WIDTH = 70       # and at most this many
LINE_MAX_OFFSET = 3.0  # arcsec: the observation target must lie this close to the line to be the source


def searched_chips(ev, elo=500.0, ehi=7000.0):
    """the chips a search looks at: every chip with events in the band"""
    band = (ev["energy"] >= elo) & (ev["energy"] < ehi)
    return tuple(int(c) for c in np.unique(ev["ccd_id"][band]))


def find_candidates(ev, elo=500.0, ehi=7000.0):
    """column bands that stay high along the whole chip; 'flat' marks real streaks"""
    found = []
    for ccd in searched_chips(ev, elo, ehi):
        on = (ev["ccd_id"] == ccd) & (ev["energy"] >= elo) & (ev["energy"] < ehi)
        x, y = ev["chipx"][on].astype(int), ev["chipy"][on].astype(int)
        counts = np.bincount(np.clip(x, 0, 1024), minlength=1025)[1:1025].astype(float)
        # running median is the normal level, it ignores narrow bumps
        level = np.array([np.median(counts[max(i - 75, 0):i + 75]) for i in range(1024)])
        high = counts > level + 6 * np.sqrt(np.maximum(level, 4))
        i = 0
        while i < 1024:
            if not high[i]:
                i += 1
                continue
            j = i
            # small holes inside are allowed, bad columns should not split a band
            while j + 1 < 1024 and np.any(high[j + 1:j + 5]):
                j += 1
            width = j - i + 1
            if MIN_WIDTH <= width <= MAX_WIDTH:
                in_band = (x >= i + 1) & (x <= j + 1)
                side = ((x >= i + 1 - 20) & (x < i + 1)) | ((x > j + 1) & (x <= j + 1 + 20))
                # excess in 8 row slices: a streak fills all of them about evenly
                slices = []
                for start in range(1, 1025, 128):
                    r = (y >= start) & (y < start + 128)
                    slices.append(in_band[r].sum() - side[r].sum() * width / 40)
                slices = np.array(slices, float)
                middle = np.sort(slices)[1:-1]
                flat = bool(middle.min() > 0.5 * np.median(middle)
                            and np.median(middle) > 5 * np.sqrt(max(side.sum() / 8 * width / 40, 1)))
                found.append(dict(ccd=int(ccd), low=i + 1, high=j + 1, flat=flat))
            i = j + 1
    return found


def _first_guess(ev, s, elo, ehi):
    on = (ev["ccd_id"] == s["ccd"]) & (ev["chipx"] >= s["low"]) & (ev["chipx"] <= s["high"]) & (ev["energy"] >= elo) & (ev["energy"] < ehi)
    ex, ey = ev["x"][on], ev["y"][on]
    hist, xe, ye = np.histogram2d(ex, ey, bins=64)
    a, b = np.unravel_index(hist.argmax(), hist.shape)
    box = (ex >= xe[a]) & (ex < xe[a + 1]) & (ey >= ye[b]) & (ey < ye[b + 1])
    sx, sy = np.median(ex[box]), np.median(ey[box])
    far = np.hypot(ex - sx, ey - sy) > 100
    if far.sum() < 10:
        far = np.ones(len(ex), bool)
    vals, vecs = np.linalg.eigh(np.cov(np.array([ex[far] - sx, ey[far] - sy])))
    ux, uy = vecs[:, 1]
    return float(sx), float(sy), float(ux), float(uy)


def _refit(ev, s, sx, sy, ux, uy, elo, ehi):
    # move the line onto the peak of the across-profile in 100 px slices, a few iterations
    on = (ev["ccd_id"] == s["ccd"]) & (ev["energy"] >= elo) & (ev["energy"] < ehi)
    x, y = ev["x"][on], ev["y"][on]
    for _ in range(4):
        du, dv = x - sx, y - sy
        a = ux * du + uy * dv; p = -uy * du + ux * dv
        sel = (np.abs(a) > 100) & (np.abs(p) < 20)
        A, P = [], []
        for lo in np.arange(-2000, 2000, 100):
            m = sel & (a >= lo) & (a < lo + 100)
            if m.sum() < 50:
                continue
            h, e = np.histogram(p[m], bins=np.arange(-20, 20.01, 1))
            k = h.argmax(); w = slice(max(k - 3, 0), k + 4)
            c = (e[:-1] + e[1:]) / 2
            A.append(lo + 50); P.append((h[w] * c[w]).sum() / max(h[w].sum(), 1))
        if len(A) < 2:
            break
        b1, b0 = np.polyfit(A, P, 1)
        sx, sy = sx - uy * b0, sy + ux * b0
        t = np.arctan(b1)
        ux, uy = ux * np.cos(t) - uy * np.sin(t), ux * np.sin(t) + uy * np.cos(t)
    du, dv = x - sx, y - sy
    a = ux * du + uy * dv; p = -uy * du + ux * dv
    on_line = np.abs(p) < 4
    amin = float(a[on_line].min()) if on_line.any() else -2000.0
    amax = float(a[on_line].max()) if on_line.any() else 2000.0
    return dict(ccd=s["ccd"], sx=float(sx), sy=float(sy), ux=float(ux), uy=float(uy), amin=amin, amax=amax,
                low=s["low"], high=s["high"])


def detect_streaks(ev, elo=500.0, ehi=7000.0):
    """flat streaks with their line in the sky frame of these events"""
    out = []
    for s in find_candidates(ev, elo, ehi):
        if not s["flat"]:
            continue
        sx, sy, ux, uy = _first_guess(ev, s, elo, ehi)
        out.append(_refit(ev, s, sx, sy, ux, uy, elo, ehi))
    return out


def line_distance(line, x, y):
    return abs(-line["uy"] * (x - line["sx"]) + line["ux"] * (y - line["sy"]))


def project_on_line(line, x, y):
    t = line["ux"] * (x - line["sx"]) + line["uy"] * (y - line["sy"])
    return line["sx"] + line["ux"] * t, line["sy"] + line["uy"] * t


WING_RADIUS = 20.0   # arcsec: psf wings used to place the source along the line
WING_GAP = 3.0       # arcsec: events this close to the line (the streak itself) are left out
MIN_WING_EVENTS = 50


def source_from_wings(ev, line):
    """the streak's source from the data alone: across = on the line; along = centroid of the psf wings
    (events within 20" but more than 3" off the line, so the streak and a piled-up hollow core do not pull it).
    tested 2026-11-04: 950 1.1" (of it 1.09" is the line's own offset), 27522 0.01", 15511 0.46" from the truth.
    ev: image-filter events (band, and tg_m 0/Null for gratings). returns (x, y) or None"""
    du, dv = ev["x"] - line["sx"], ev["y"] - line["sy"]
    a = line["ux"] * du + line["uy"] * dv
    p = -line["uy"] * du + line["ux"] * dv
    near = np.abs(p) < 3 / SKY_ARCSEC
    if near.sum() < MIN_WING_EVENTS:
        return None
    h, e = np.histogram(a[near], bins=np.arange(a[near].min(), a[near].max() + 16, 16))
    a0 = (e[h.argmax()] + e[h.argmax() + 1]) / 2       # coarse peak along the line
    for _ in range(20):
        m = (np.abs(p) > WING_GAP / SKY_ARCSEC) & (np.hypot(a - a0, p) < WING_RADIUS / SKY_ARCSEC)
        if m.sum() < MIN_WING_EVENTS:
            return None
        n = float(a[m].mean())
        if abs(n - a0) < 0.01:
            break
        a0 = n
    return line["sx"] + line["ux"] * a0, line["sy"] + line["uy"] * a0


def streak_source(line, ev=None, zeroth_xy=None, obs_target_xy=None):
    """where the streak's source is, per observation; the search coordinate is never used.
    order: zeroth-order circle (gratings) -> psf-wing centroid on the line -> the observation's own target
    (RA_TARG) if within 3" of the line -> None (the caller skips and reports). returns (x, y, how) or None"""
    if zeroth_xy is not None:
        return float(zeroth_xy[0]), float(zeroth_xy[1]), "zeroth order"
    if ev is not None:
        w = source_from_wings(ev, line)
        if w is not None:
            return float(w[0]), float(w[1]), "psf wings on the streak line"
    if obs_target_xy is not None and line_distance(line, *obs_target_xy) * SKY_ARCSEC <= LINE_MAX_OFFSET:
        px, py = project_on_line(line, *obs_target_xy)
        return float(px), float(py), "observation target on the streak line"
    return None


def bkg_rotbox(line, offset=70.0, along=500.0, length=700.0, width=40.0):
    """ciao region for the acisreadcorr background spectrum: a box beside the streak on its chip"""
    bx = line["sx"] + line["ux"] * along - line["uy"] * offset
    by = line["sy"] + line["uy"] * along + line["ux"] * offset
    ang = float(np.degrees(np.arctan2(line["uy"], line["ux"])))
    return f"rotbox({bx:.2f},{by:.2f},{length:.1f},{width:.1f},{ang:.3f})"
