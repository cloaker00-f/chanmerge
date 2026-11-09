"""Arm and streak corrections of one merge_obs member, then the merged recompute.

works on the member's thresh image, thresh expmap and reprojected event file, all in the merge_obs frame.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import armwidth as AW
from . import masks as M
from . import streak as S
from .fill import (ring_index, rates_with_fallback, correct_image, RING)
from .geometry import image_sky_grid, column_wcs, radec_to_sky, sky_to_pixel_index, SKY_ARCSEC

# a full-frame observation whose ciao arm box covers more than this share of its exposed image is not arm-treated and is left out of
# its group's merge (rule 3, session 007): inside the box the tg_m filter removed most events (letg 1790: 88-100% on S0, S1, S3-S5, box 47%
# of the image), masking it would take away more than this. hetg full frame boxes cover 15-18%. 40% was my choice (2026-11-07, test j),
# accepted by the user; the user kept it on 2026-11-09.
ARM_SHARE_CAP_FULL = 0.40
# when less than this share of the exposed image (beyond 3 zero-order radii) lies outside every ciao box, the box covers the whole image:
# letg 2324 (box 3398 px, 0%): the filter removed 50-100% of the events on every chip, background included (test w, session 007), so the
# image is much darker than any other observation and is never mixed with them, whatever the arm option (rule 3)
COVERS_IMAGE = 0.02

@dataclass
class MemberResult:
    obsid: str
    counts: np.ndarray
    expmap: np.ndarray
    filled: np.ndarray
    masked: np.ndarray
    streak_lines: List[dict] = field(default_factory=list)
    streak_searched: Optional[tuple] = None   # chips the streak search looked at (None = no search)
    streak_done: str = ""          # fill / mask / '' (none found or leave)
    arm_done: str = ""             # mask_keep_source / mask_remove_source / mask_fill / ''
    arm_visibility: Optional[dict] = None   # what the arm check measured (None = not asked)
    arm_measure: Optional[dict] = None      # the leak measurement outside the ciao boxes (armwidth.measure), cached per observation by run.py
    untreatable: str = ""          # rule 3: why this observation cannot be mixed with others (it leaves its group's merge); '' = fine
    thinned: Optional[np.ndarray] = None    # pixels where a fitted arm leak was subtracted (counts thinned, exposure kept)
    centre: Optional[tuple] = None
    rpix: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)
    did_fill: bool = False         # a fill was chosen and applied (names get _filled)
    did_mask: bool = False         # a mask was chosen and applied (names get _masked)

    @property
    def suffix(self):
        # by the chosen actions: the unfillable centre of a fill stays no data but is not a 'mask'
        return ("_filled" if self.did_fill else "") + ("_masked" if self.did_mask else "")

    @property
    def changed(self):
        return bool(self.filled.any() or self.masked.any() or (self.thinned is not None and self.thinned.any()))

    def ring_of_xy(self, binsize):
        cx, cy = self.centre
        return lambda ex, ey: (np.hypot(ex - cx, ey - cy) / binsize / RING).astype(int)


def load_events(path, elo_ev=None, ehi_ev=None):
    from astropy.io import fits
    with fits.open(path, memmap=True) as h:
        d = h["EVENTS"].data
        hd = h["EVENTS"].header
        ev = {k: np.array(d[k], dtype=float) for k in ("x", "y", "chipx", "chipy", "ccd_id", "energy")}
        if "tg_m" in d.columns.names:
            ev["tg_m"] = np.array(d["tg_m"], dtype=float)
    return ev, hd


def correct_member(obsid, counts, expmap, header, evt_path, *, arm_method=None, arm_region=None,
                   streak_action="leave", elo_ev=500.0, ehi_ev=7000.0, seed=1, arm_share_cap=None, arm_gate=False,
                   arm_measure=None, arm_leak=False):
    """arm_method: 1 filter / 2 mask keep source / 3 mask remove source / 4 mask + fill (gratings only).
    streak_action: fill / mask / leave (acisreadcorr is applied before merge_obs, not here).
    the arm box is ciao's tgmask box, for ACIS + LETG the narrow box run.py chose (letgsort).
    arm order (session 007): (a) the arm box covers the whole exposed image (letg 2324 type): untreatable for every arm option, incl. 1;
    (b) arm_share_cap (run.py: full frame only): the ciao box covers more than that share: untreatable for options 2-4 (option 1 keeps
    the image and a note says how much of it is a dark corridor); untreatable = res.untreatable says why, nothing is changed and run.py
    takes the observation out of its group's merge; (c) arm_gate: no arm visible (masks.arm_visibility) -> nothing is done, a note says
    so; (d) the band treated is the ciao box (the dark corridor ends exactly at its edge); arm_leak: the leak outside the box is measured
    (armwidth.measure, or arm_measure when run.py has it already) and a detected leak is handled (see _leak_treatment).
    the ring centre is this observation's own source (zeroth order, or the streak source from its data);
    the search coordinate is never used."""
    c = np.asarray(counts, float); x = np.nan_to_num(np.asarray(expmap, float))
    B = float(header["CDELT1P"])
    px, py = image_sky_grid(header, c.shape)
    exp = x > 0
    res = MemberResult(obsid, c, x, np.zeros(c.shape, bool), np.zeros(c.shape, bool))
    res.arm_measure = arm_measure
    check_box = arm_method in (1, 2, 3, 4) and arm_region is not None
    need_arms = arm_method in (2, 3, 4) and arm_region is not None
    need_streak = streak_action in ("fill", "mask")
    if arm_method in (2, 3, 4) and arm_region is None:
        res.notes.append(f"{obsid}: no arm region found, arms not treated")
    if not check_box and not need_streak:
        return res
    wcs = _event_wcs(evt_path)
    reg = M.region_to_frame(arm_region, wcs) if check_box else None
    if reg is not None:
        share = M.arm_share(px, py, reg, exp)
        wide = max(b["width"] for b in reg["boxes"])
        if M.outside_share(px, py, reg, exp) < COVERS_IMAGE:
            res.untreatable = (f"its arm box ({wide:.0f} sky px wide) covers the whole exposed image; merge_obs's tg_m filter removes most events "
                               f"inside that box, background included (measured on PKS 2155-304 2324: 50-100% per chip), so this image is much "
                               f"darker than other observations and must not be mixed with them")
        elif need_arms and arm_share_cap is not None and share > arm_share_cap:
            res.untreatable = (f"its arm box ({wide:.0f} sky px wide) covers {share:.0%} of the exposed image; inside it merge_obs's filter "
                               f"removed most events (PKS 2155-304 1790: 88-100% on most chips) and masking or filling it would take away "
                               f"more than {arm_share_cap:.0%} of the image")
        elif arm_method == 1 and arm_share_cap is not None and share > arm_share_cap:
            res.notes.append(f"{obsid}: its arm box ({wide:.0f} sky px wide) covers {share:.0%} of the exposed image and merge_obs's filter "
                             f"darkened that part (option 1 keeps it as it is)")
        if res.untreatable:
            need_arms, reg = False, None
            if not need_streak:
                return res
    if not need_arms:
        reg = None
    if need_arms and arm_gate:
        vis = M.arm_visibility(px, py, c, x, reg)
        res.arm_visibility = vis
        if vis["detected"] is False:
            res.notes.append(f"{obsid}: no grating arm is visible in this image ({vis['excess']:+.0f} counts against the background in the +-{vis['half']:.0f} px band "
                             f"along the arm, {abs(vis['sigma']):.1f} sigma, {M.ARM_GATE_SIGMA:.0f} needed): no arm correction was made, because masking or filling "
                             f"would only take away area")
            need_arms, reg = False, None
    if not need_arms and not need_streak:
        return res
    ev, ehd = load_events(evt_path)
    if need_arms and arm_leak and res.arm_measure is None:
        # band independent: the 0.5-7 kev filter-only events of the member, binned on this grid for the reference rings
        iev = image_events(ev, 500.0, 7000.0)
        bx, by, ok = sky_to_pixel_index(iev["x"], iev["y"], header, c.shape)
        cm = np.zeros(c.shape)
        np.add.at(cm, (by[ok], bx[ok]), 1)
        res.arm_measure = AW.measure(cm, x, px, py, header, reg, iev["x"], iev["y"])
    # streak lines from this member's events (merge_obs frame)
    lines = []
    if need_streak:
        lines = S.detect_streaks(ev, elo_ev, ehi_ev)
        res.streak_lines = lines
        res.streak_searched = S.searched_chips(ev, elo_ev, ehi_ev)
    obs_xy = None
    if wcs is not None and ehd.get("RA_TARG") is not None and ehd.get("DEC_TARG") is not None:
        obs_xy = radec_to_sky(float(ehd["RA_TARG"]), float(ehd["DEC_TARG"]), *wcs)
    zeroth = (reg["x"], reg["y"]) if reg else None
    if zeroth is not None:
        centre = zeroth
    elif lines:
        img_ev = image_events(ev, elo_ev, ehi_ev)
        src = S.streak_source(lines[0], img_ev, None, obs_xy)
        if src is None:
            centre = (lines[0]["sx"], lines[0]["sy"])
            res.notes.append(f"{obsid}: the streak's source could not be placed; rings centred on the rough line origin")
        else:
            centre = src[:2]
    else:
        centre = None
    if centre is None:
        return res
    res.centre = (float(centre[0]), float(centre[1]))
    rpix = ring_index(px, py, centre[0], centre[1], B)
    res.rpix = rpix
    fills, masks_ = [], []
    arms_plain = np.zeros(c.shape, bool)
    arms_wide = np.zeros(c.shape, bool)
    leak_share = None
    if need_arms:
        reg, leak_share = _leak_treatment(obsid, reg, res, px, py, exp, arm_share_cap)
        keep_source = arm_method in (2, 4)
        arms = M.arm_mask(px, py, reg, keep_source=keep_source) & exp
        arms_plain = M.arm_mask(px, py, reg, keep_source=True)
        arms_wide = M.arm_mask(px, py, reg, keep_source=False, extra=M.RING_ARM_EXTRA / SKY_ARCSEC)
        if arm_method == 4:
            fills.append((arms, seed * 1000 + 22))
        else:
            masks_.append(arms)
        res.arm_done = {2: "mask_keep_source", 3: "mask_remove_source", 4: "mask_fill"}[arm_method]
    zones = np.zeros(c.shape, bool)
    for i, L in enumerate(lines):
        st_centre = zeroth if zeroth is not None else centre
        strip, zone = M.streak_strip(L, px, py, B, centre=st_centre)
        strip &= exp
        zones |= zone
        if streak_action == "fill":
            fills.insert(i, (strip, seed * 1000 + 11 + i))
        else:
            masks_.append(strip)
    if lines:
        res.streak_done = streak_action
    if not fills and not masks_:
        return res
    rates, used = rates_with_fallback(c, x, rpix, arms_wide | zones, arms_plain | zones)
    stats = {}
    cc, xx, filled, masked = correct_image(c, x, rpix, fills, masks_, rates, stats)
    if leak_share is not None:
        leak_share = np.where(filled | masked | ~(xx > 0), 0.0, leak_share)
        thin = AW.thin_counts(cc, leak_share, seed * 1000 + 33)
        res.thinned = thin < cc
        cc = thin
    res.counts, res.expmap, res.filled, res.masked = cc, xx, filled, masked
    res.did_fill = bool(fills) and bool(filled.any())
    res.did_mask = bool(masks_) and bool(masked.any())
    n_norate = stats.get("norate", 0)
    if n_norate:
        res.notes.append(f"{obsid}: {n_norate} pixels could not be filled (too few clean pixels at their radius: near the "
                         f"source, or in a narrow subarray strip along the arms) and stay no data")
    return res


# a measured leak outside the ciao box is SUBTRACTED: the fitted leak share is thinned out of the pixels beside the box, they stay real data.
# chosen on cen x-3 27522 (test w, session 007; leak measured on one half of each arm, judged on the other): rate / local reference in the
# leak zone 1.39-1.54 untreated -> 1.03-1.05 subtracted (1.07-1.23 at 150-400 px from the source), area lost unchanged (17%); widening the
# box by the 90% leak width gave 0.97-0.98 but made 56-58% of the image simulated, beyond the 40% cap. along the whole arm (only from 400 px
# on left 1.5-1.6 at 150-400 px).
LEAK_ALONG_MIN = 0.0      # sky px from the zero order along the arm where the leak handling starts


def _leak_treatment(obsid, reg, res, px, py, exp, cap):
    """returns (region to treat, leak share map or None) and writes one note per observation about the leak"""
    m = res.arm_measure
    if not m:
        return reg, None
    names = ("HEG", "MEG") if len(reg["boxes"]) == 2 else ("LETG",)
    leaks = [(n, b) for n, b in zip(names, m["boxes"]) if b["status"] == "leak"]
    quiet = [n for n, b in zip(names, m["boxes"]) if b["status"] == "none"]
    unknown = [n for n, b in zip(names, m["boxes"]) if b["status"] == "unmeasurable"]
    if quiet:
        res.notes.append(f"{obsid}: no arm light outside the arm box was seen ({', '.join(quiet)}); the box alone is treated")
    if unknown:
        res.notes.append(f"{obsid}: arm light outside the arm box could not be checked ({', '.join(unknown)}: too few counts or no room beside "
                         f"the box, e.g. a subarray strip); the box alone is treated")
    if not leaks:
        return reg, None
    txt = ", ".join(f"{n} A {b['A']:.2f} lam {b['lam']:.0f} px" for n, b in leaks)
    res.notes.append(f"{obsid}: arm light leaks outside the arm box ({txt}, fitted on the events); the fitted leak is subtracted "
                     f"beside the box (counts thinned at random, pixels kept)")
    return reg, AW.leak_share_map(px, py, reg, m, along_min=LEAK_ALONG_MIN)


def _event_wcs(evt_path):
    from astropy.io import fits
    try:
        return column_wcs(fits.getheader(evt_path, "EVENTS"), "x", "y")
    except (KeyError, OSError):
        return None


def image_events(ev, elo_ev, ehi_ev):
    """events that enter the image: the band and, for gratings, tg_m 0 or Null (99)"""
    m = (ev["energy"] >= elo_ev) & (ev["energy"] < ehi_ev)
    if "tg_m" in ev:
        m &= np.isin(ev["tg_m"], [0, 99])
    return {k: v[m] for k, v in ev.items()}


def merged_recompute(merged_counts, merged_expmap, originals, corrected):
    """merged = merge_obs merged - members' original shares + their corrected shares (same grid).
    this keeps merge_obs's own per-member clipping in pixels nobody touched."""
    c = np.asarray(merged_counts, float).copy()
    x = np.nan_to_num(np.asarray(merged_expmap, float)).copy()
    for (c0, x0), (c1, x1) in zip(originals, corrected):
        c += np.asarray(c1, float) - np.asarray(c0, float)
        x += np.nan_to_num(x1) - np.nan_to_num(x0)
    x[x < 0] = 0
    c[x <= 0] = 0
    return c, x
