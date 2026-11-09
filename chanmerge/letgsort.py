"""ACIS + LETG: the events are order-sorted again with a NARROW arm box, for images only (test x, session 007).

why: ciao's tg_create_mask draws the LETG box with width_factor_letg=300 on purpose, "large enough to accommodate source and background
regions when binning" a spectrum (ahelp; ~2940 sky px on ACIS-S, "essentially covering the full detectors"). merge_obs's image filter keeps
only tg_m 0 or Null, so with that box it removes most events of the whole image, background and halo included (test w: 50-100% per chip).
with a narrow box (tg_resolve_events run again on the repro evt2) events outside the box get tg_part 99 and stay: test x on PKS 2155-304
2324 / 1790 and Capella 55 gave 0% removed outside the box and the background back to its normal level (1790: within ~2% of its untouched
area). the box width is chosen per observation from the leak of arm light outside it (armwidth.measure): the narrowest candidate with no
leak; when the excess outside does not shrink as the box grows it is not arm light, the narrower box is kept and the excess is subtracted
later like any leak. the new event file is for images only: dispersed spectrum events outside the narrow box lose their order, so it must
never be used for spectra. the repro files (evt2, pha2, tg/) are never touched.
"""
import shutil

import numpy as np

CANDIDATE_WIDTHS = (100.0, 200.0, 300.0, 400.0)   # sky px; test x: 100 leaks in all three, 200 enough for 1790, 400 clean for capella
DEFAULT_WIDTH = 200.0          # when nothing can be measured (subarray strip, too few counts)
NOT_SHRINKING = 0.7            # the leak at the wider box is at least this share of the one before: it does not follow the box
TG_COLUMNS = ("tg_r", "tg_d", "tg_m", "tg_lam", "tg_mlam", "tg_srcid", "tg_part", "tg_smap")
IMAGING_NOTE = "chanmerge: LETG events order-sorted again with a narrow arm box; FOR IMAGES ONLY, never for spectra (use the repro files)"


def narrow_mask(src_mask, out, width):
    """copy of a tg_create_mask region file with the width of every box row set to width (sky px); the zero-order circle stays ciao's"""
    from astropy.io import fits
    shutil.copyfile(src_mask, out)
    with fits.open(out, mode="update") as h:
        d = h["REGION"].data
        shapes = [str(s).strip().lower() for s in d["SHAPE"]]
        for i, s in enumerate(shapes):
            if "box" in s:
                r = np.array(d["R"][i], dtype=float)
                r[1] = width
                d["R"][i] = r
        h["REGION"].header["HISTORY"] = f"chanmerge: box width set to {width:.0f} sky px for images"
    return out


def copy_tg_columns(evt_in, resolved, out):
    """out = evt_in with the tg_* columns taken from resolved (tg_resolve_events output on evt_in). the rows must be the same events in the
    same order (count and times compared); raises ValueError otherwise. every other column and header stays as in evt_in"""
    from astropy.io import fits
    with fits.open(evt_in, memmap=True) as h0, fits.open(resolved, memmap=True) as h1:
        t0, t1 = np.asarray(h0["EVENTS"].data["time"]), np.asarray(h1["EVENTS"].data["time"])
        if len(t0) != len(t1) or not np.array_equal(t0, t1):
            raise ValueError(f"tg_resolve_events changed the events ({len(t1)} rows for {len(t0)})")
        new = {c: np.array(h1["EVENTS"].data[c]) for c in TG_COLUMNS if c in h1["EVENTS"].columns.names}
    shutil.copyfile(evt_in, out)
    with fits.open(out, mode="update") as h:
        d = h["EVENTS"].data
        for c, v in new.items():
            if c in d.columns.names:
                d[c][:] = v
        h["EVENTS"].header["HISTORY"] = IMAGING_NOTE
    return out


def choose_width(results):
    """results: [(width, status, A)] for the candidates in increasing width (status leak / none / unmeasurable, as armwidth.measure).
    returns (width, why). the narrowest box with no leak; a leak that does not shrink from one box to the next is not arm light (the
    narrower box is kept, the excess is subtracted later); still leaking at the widest box allowed: that one, subtracted"""
    prev = None
    for w, status, a in results:
        if status == "unmeasurable":
            if prev is None:
                return DEFAULT_WIDTH, "the leak outside the box could not be measured (too few counts or no room beside the box)"
            return prev[0], f"the leak could not be measured beyond {prev[0]:.0f} px"
        if status != "leak":
            return w, f"no arm light outside the {w:.0f} px box"
        if prev is not None and a >= NOT_SHRINKING * prev[2]:
            return prev[0], (f"the excess outside did not shrink from the {prev[0]:.0f} px box to {w:.0f} px (A {prev[2]:.2f} -> {a:.2f}): "
                             f"not arm light; the {prev[0]:.0f} px box is kept and the excess is subtracted")
        prev = (w, status, a)
    if prev is None:
        return DEFAULT_WIDTH, "no candidate box fitted under the area cap"
    return prev[0], f"arm light still leaks outside the widest box allowed ({prev[0]:.0f} px); it is subtracted"


def bin_image_events(evt, header, shape, elo=500.0, ehi=7000.0):
    """counts image of the events an image would use (band, tg_m 0 or Null) on a fluximage grid"""
    from astropy.io import fits
    from .geometry import sky_to_pixel_index
    with fits.open(evt, memmap=True) as h:
        d = h["EVENTS"].data
        e = np.asarray(d["energy"], float)
        sel = (e >= elo) & (e < ehi) & np.isin(np.asarray(d["tg_m"]), [0, 99])
        x, y = np.asarray(d["x"], float)[sel], np.asarray(d["y"], float)[sel]
    bx, by, ok = sky_to_pixel_index(x, y, header, shape)
    c = np.zeros(shape)
    np.add.at(c, (by[ok], bx[ok]), 1)
    return c, x, y
