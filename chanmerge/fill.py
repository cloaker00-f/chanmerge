"""Fills and masks on merge_obs member images, the matching new event file, and the merged recompute.

rules (decided, tested on 27522 and 950):
- ring rates for every fill come from pixels outside the arm boxes (+30") and the streak strip; per ring a
  fallback to the plain boxes when fewer than MIN_PIX pixels remain; else the pixel stays no data.
- fills first, then masks; where both apply the mask wins.
- the event file: image-filter events in filled/masked pixels are removed, filled pixels get copies of real
  same-annulus events (FILLED=1); dispersed grating events stay. the original file is never touched.
"""
import numpy as np

from .geometry import sky_to_pixel_index

RING = 2          # annulus width in image pixels (7.9" at 8 sky px)
CLIP = 5.0        # sigma clip for point sources inside a ring
MIN_PIX = 20      # pixels a ring needs for its rate
EDGE_FRAC = 0.45  # fake events sit within +-0.45 pixel of the centre (no bin-edge ambiguity)


def ring_index(px, py, cx, cy, binsize):
    return (np.hypot(px - cx, py - cy) / binsize / RING).astype(int)


def ring_rates(c, x, rpix, holes):
    """counts per exposure per annulus from exposed pixels outside every hole, with one 5 sigma clip"""
    rates = {}
    use = (x > 0) & ~holes
    for k in np.unique(rpix[use]):
        sel = use & (rpix == k)
        xs = x[sel].sum()
        if xs <= 0:
            continue
        r = c[sel].sum() / xs
        keep = sel & (c <= r * x + CLIP * np.sqrt(r * x + 1))
        xk = x[keep].sum()
        rates[int(k)] = float(c[keep].sum() / xk) if xk > 0 else 0.0
    return rates


def rates_with_fallback(c, x, rpix, wide_holes, plain_holes, min_pix=MIN_PIX):
    exp = x > 0
    Rw, Rp = ring_rates(c, x, rpix, wide_holes), ring_rates(c, x, rpix, plain_holes)
    out, used = {}, {"wide": 0, "plain": 0, "none": 0}
    for k in np.unique(rpix[exp]):
        k = int(k)
        ring = (rpix == k) & exp
        if (ring & ~wide_holes).sum() >= min_pix and k in Rw:
            out[k] = Rw[k]; used["wide"] += 1
        elif (ring & ~plain_holes).sum() >= min_pix and k in Rp:
            out[k] = Rp[k]; used["plain"] += 1
        else:
            used["none"] += 1
    return out, used


def draw_ring_fill(lam, ring, seed):
    """fill counts for one region, ring by ring (decided 2026-11-04, session 002, after 16703's unlucky draw).
    per ring: the total is its expectation (sum of rate x exposure), rounded up or down at random so the mean stays
    exact; then that total is spread over the ring's pixels by a multinomial draw weighted by rate x exposure.
    a multinomial is a poisson draw conditioned on its total, so single pixels still look like real counts, but a
    ring's filled total can no longer run away (old plain poisson: 16703 1-3 keV 420-480" got 61 counts, expected 37).
    every ring has its own random stream (seed, ring), so pixels added or removed in one ring do not reshuffle the
    draws of the others (the old single stream in row order did)."""
    lam = np.asarray(lam, float)
    ring = np.asarray(ring)
    out = np.zeros(lam.shape, np.int64)
    for k in np.unique(ring):
        sel = ring == k
        e = float(lam[sel].sum())
        if e <= 0:
            continue
        rng = np.random.default_rng([int(seed), int(k)])
        n = int(np.floor(e)) + int(rng.random() < e - np.floor(e))
        out[sel] = rng.multinomial(n, lam[sel] / e)
    return out


def correct_image(c, x, rpix, fills, masks, rates, stats=None):
    """fills: list of (region, seed); masks: list of regions. returns counts, expmap, filled, masked.
    pixels of a fill whose ring has no rate become no data (counted in masked; their number goes to
    stats['norate'] when a dict is given). fill counts come from draw_ring_fill (ring totals fixed)."""
    cc, xx = c.astype(float).copy(), x.astype(float).copy()
    mask_all = np.zeros(c.shape, bool)
    for m in masks:
        mask_all |= m
    filled = np.zeros(c.shape, bool)
    n_norate = 0
    for region, seed in fills:
        reg = region & ~mask_all & (x > 0)
        norate = reg & ~np.isin(rpix, list(rates))
        reg &= ~norate
        if reg.any():
            lam = np.array([rates[int(k)] for k in rpix[reg]]) * x[reg]
            cc[reg] = draw_ring_fill(lam, rpix[reg], seed)
        filled |= reg
        mask_all |= norate
        n_norate += int(norate.sum())
    cc[mask_all] = 0
    xx[mask_all] = 0
    if stats is not None:
        stats["norate"] = n_norate
    return cc, xx, filled, mask_all


def flux_of(c, x):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x > 0, c / x, np.nan)


def corrected_flux(orig_flux, c_new, x_new, x_orig):
    """flux after corrections, in merge_obs's own convention: recomputed where there is exposure, NaN only where
    chanmerge made the pixel no data, and merge_obs's value (0) outside the field it never covered"""
    out = np.array(orig_flux, dtype=float, copy=True)
    x_orig = np.nan_to_num(np.asarray(x_orig, float))
    m = np.asarray(x_new) > 0
    out[m] = np.asarray(c_new, float)[m] / np.asarray(x_new, float)[m]
    out[(x_orig > 0) & ~m] = np.nan
    return out


def expmap_stats(x):
    v = x[x > 0]
    if v.size == 0:
        return None, None, None
    return tuple(float(q) for q in np.percentile(v, [5, 50, 95]))


def mask_codes(filled, masked, thinned=None):
    """0 untouched, 1 masked (no data), 2 filled, 3 arm leak subtracted (counts thinned, real data otherwise)"""
    m = np.zeros(filled.shape, np.uint8)
    if thinned is not None:
        m[thinned] = 3
    m[masked] = 1
    m[filled] = 2
    return m


def image_filter(d, elo_ev, ehi_ev):
    """events that enter the image: band and, for gratings, tg_m 0 or Null (99)"""
    sel = (d["energy"] >= elo_ev) & (d["energy"] <= ehi_ev)
    if "tg_m" in d.columns.names:
        sel &= np.isin(d["tg_m"], [0, 99])
    return sel


def grating_ok(d):
    return np.isin(d["tg_m"], [0, 99]) if "tg_m" in d.columns.names else np.ones(len(d), bool)


def bin_events(d, header, shape, elo_ev=None, ehi_ev=None, tg=True):
    sel = np.ones(len(d), bool)
    if elo_ev is not None:
        sel &= (d["energy"] >= elo_ev) & (d["energy"] <= ehi_ev)
    if tg:
        sel &= grating_ok(d)
    bx, by, ok = sky_to_pixel_index(d["x"][sel], d["y"][sel], header, shape)
    img = np.zeros(shape)
    np.add.at(img, (by[ok], bx[ok]), 1)
    return img


def correct_events(evt_in, header, c_corr, filled, masked, rpix, ring_of_xy, out_path, elo_ev, ehi_ev, seed=7, thinned=None):
    """write the event file that bins back to the corrected counts image. returns a small summary.
    thinned: pixels where an arm leak was subtracted; there just enough of the pixel's image events (band, tg_m 0 or Null) are dropped at
    random that the pixel bins back to c_corr; events of other energies stay"""
    from astropy.io import fits
    with fits.open(evt_in) as hl:
        ev = hl["EVENTS"].data
        rng = np.random.default_rng(seed)
        bx, by, ok = sky_to_pixel_index(ev["x"], ev["y"], header, c_corr.shape)
        in_fill = np.zeros(len(ev), bool); in_mask = np.zeros(len(ev), bool)
        in_fill[ok] = filled[by[ok], bx[ok]]
        in_mask[ok] = masked[by[ok], bx[ok]]
        tg_ok = grating_ok(ev)
        inband = (ev["energy"] >= elo_ev) & (ev["energy"] <= ehi_ev)
        er = ring_of_xy(ev["x"], ev["y"])
        donor = tg_ok & ok & ~in_fill & ~in_mask
        rows = []
        for k in np.unique(rpix[filled]):
            pix = np.argwhere(filled & (rpix == k))
            d_in = np.nonzero(donor & inband & (er == k))[0]
            d_out = np.nonzero(donor & ~inband & (er == k))[0]
            if len(d_in) == 0:
                d_in = np.nonzero(donor & inband & (np.abs(er - k) <= 2))[0]
                d_out = np.nonzero(donor & ~inband & (np.abs(er - k) <= 2))[0]
            ratio = len(d_out) / max(len(d_in), 1)
            for j, i in pix:
                nb = int(c_corr[j, i])
                no = int(rng.poisson(ratio * nb)) if len(d_out) else 0
                picks = (list(rng.choice(d_in, nb)) if nb and len(d_in) else []) + (list(rng.choice(d_out, no)) if no else [])
                rows += [(p, j, i) for p in picks]
        # only events that can enter an image are removed; dispersed grating events stay
        keep = ~((in_fill | in_mask) & tg_ok)
        if thinned is not None and thinned.any():
            in_thin = np.zeros(len(ev), bool)
            in_thin[ok] = thinned[by[ok], bx[ok]]
            cand = np.nonzero(in_thin & tg_ok & inband & keep)[0]
            pix = by[cand] * c_corr.shape[1] + bx[cand]
            order = rng.permutation(len(cand))
            cand, pix = cand[order], pix[order]
            # within each pixel keep the first c_corr events of the shuffled list
            srt = np.argsort(pix, kind="stable")
            cand, pix = cand[srt], pix[srt]
            first = np.searchsorted(pix, pix, side="left")
            rank = np.arange(len(pix)) - first
            want = c_corr.ravel()[pix].astype(int)
            keep[cand[rank >= want]] = False
        base = np.asarray(ev[keep])
        idx = np.array([p for p, _, _ in rows], int)
        fake = np.asarray(ev[idx]).copy() if len(idx) else np.asarray(ev[:0]).copy()
        if len(idx):
            jj = np.array([j for _, j, _ in rows]); ii = np.array([i for _, _, i in rows])
            fake["x"] = header["CRVAL1P"] + (ii + 1 - header["CRPIX1P"]) * header["CDELT1P"] + rng.uniform(-EDGE_FRAC, EDGE_FRAC, len(idx)) * header["CDELT1P"]
            fake["y"] = header["CRVAL2P"] + (jj + 1 - header["CRPIX2P"]) * header["CDELT2P"] + rng.uniform(-EDGE_FRAC, EDGE_FRAC, len(idx)) * header["CDELT2P"]
            fake["time"] = rng.uniform(hl["EVENTS"].header["TSTART"], hl["EVENTS"].header["TSTOP"], len(idx))
        allrows = np.concatenate([base, fake])
        flag = np.concatenate([np.zeros(len(base), np.int16), np.ones(len(fake), np.int16)])
        o = np.argsort(allrows["time"], kind="stable")
        allrows, flag = allrows[o], flag[o]
        cols = hl["EVENTS"].columns
        coldefs = [fits.Column(name=n, format=cols[n].format, unit=cols[n].unit, array=allrows[n]) for n in cols.names if n != "FILLED"]
        coldefs.append(fits.Column(name="FILLED", format="I", array=flag))
        tb = fits.BinTableHDU.from_columns(fits.ColDefs(coldefs), header=hl["EVENTS"].header)
        for kk, vv in hl["EVENTS"].header.items():
            if kk.startswith(("TCTYP", "TCRVL", "TCRPX", "TCDLT", "TLMIN", "TLMAX", "TCUNI", "TNULL")):
                tb.header[kk] = vv
        tb.header["HISTORY"] = "chanmerge: corrected events; filled pixels FILLED=1, masked pixels emptied; original untouched"
        fits.HDUList([hl[0].copy(), tb] + [h.copy() for h in hl[2:]]).writeto(out_path, overwrite=True)
    return dict(n=int(len(ev)), removed=int((~keep).sum()), added=int(len(fake)))


def add_filled_column(evt_in, out_path):
    """an unchanged member for the merged list: same events, FILLED=0, so all files share one column layout"""
    from astropy.io import fits
    with fits.open(evt_in) as hl:
        t = hl["EVENTS"]
        if "FILLED" in t.columns.names:
            hl.writeto(out_path, overwrite=True)
            return
        cols = fits.ColDefs(list(t.columns) + [fits.Column(name="FILLED", format="I", array=np.zeros(len(t.data), np.int16))])
        tb = fits.BinTableHDU.from_columns(cols, header=t.header)
        fits.HDUList([hl[0].copy(), tb] + [h.copy() for h in hl[2:]]).writeto(out_path, overwrite=True)


def common_columns(files):
    """merge_obs's rule for dmmerge: drop phas and any column missing or of another format somewhere;
    keep the first file's order"""
    from astropy.io import fits
    cs = []
    for f in files:
        with fits.open(f) as h:
            cs.append(h["EVENTS"].columns)
    keep = []
    for col in cs[0]:
        if col.name.lower() == "phas":
            continue
        if all(col.name in c.names and c[col.name].format == col.format for c in cs[1:]):
            keep.append(col.name)
    return keep


def write_image(path, data, header, dtype=np.float32, history=None):
    from astropy.io import fits
    h = header.copy()
    if history:
        h["HISTORY"] = history
    fits.writeto(path, np.asarray(data).astype(dtype), h, overwrite=True)
