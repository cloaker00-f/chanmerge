"""run(plan): everything after the questions. it never prompts.

order: downloads -> header check -> repro (punlearn ardlib after each) -> per band, group and chip set:
merge_obs -> streak / arm corrections -> new images and event files -> tables -> index -> report.
when ocat could not be reached and the headers do not fit the groups, run stops after the downloads and hands
the problems to on_header_problems(plan, problems); that callback (the terminal or gui) may edit the plan and
returns True to go on.
"""
import glob
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

from . import ciao as C
from . import armwidth as AW
from . import letgsort as LS
from . import masks as M
from . import streak as S
from .correct import correct_member, merged_recompute, load_events, image_events, ARM_SHARE_CAP_FULL
from .fill import (correct_events, add_filled_column, common_columns, corrected_flux, expmap_stats, mask_codes, write_image)
from .geometry import column_wcs, radec_to_sky, image_sky_grid
from .plan import stage_at_least, STAGE_RANK, CHIP_NAMES, ARM_METHODS
from .tables import FINAL_COLUMNS, MEMBER_COLUMNS, write_csv_atomic, rel, fill_shares_run, rebuild_index


@dataclass
class HeaderProblem:
    kind: str          # cc / full_sub
    obsid: str
    group: str
    manual: bool
    text: str


@dataclass
class Report:
    images: List[dict] = field(default_factory=list)
    members: List[dict] = field(default_factory=list)
    failed_obs: Dict[str, str] = field(default_factory=dict)
    failed_groups: Dict[str, str] = field(default_factory=dict)
    reused: List[str] = field(default_factory=list)
    streak: Dict[str, str] = field(default_factory=dict)       # obsid -> action applied
    # (obsid, band or 'acisreadcorr') -> (chips searched, chips with a streak)
    streak_search: Dict[tuple, tuple] = field(default_factory=dict)
    arms: Dict[str, str] = field(default_factory=dict)         # obsid -> arm method applied
    notes: List[str] = field(default_factory=list)
    header_problems: List[HeaderProblem] = field(default_factory=list)
    manual_problems: List[HeaderProblem] = field(default_factory=list)
    # rule 3 in a manual group (never changed automatically): the observation stayed in the image untreated; asked at the end
    arm_problems: List[HeaderProblem] = field(default_factory=list)
    left_out: Dict[str, str] = field(default_factory=dict)     # obsid -> group it left (rule 3, automatic groups)
    # cc found by the header although it was asked for flux and no cc row of the table asks for cc data: dict(obsid, path, bytes, deletable, had_data, text)
    cc_unwanted: List[dict] = field(default_factory=list)
    run_dirs: List[str] = field(default_factory=list)
    stopped: bool = False
    index_counts: tuple = (0, 0)
    kept_images: int = 0       # rerun of some groups: images of the other groups left as they were


def _iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S") if t else ""


def _date(t):
    return t.strftime("%Y-%m-%d") if t else "unknown"


class Runner:
    def __init__(self, plan, ciao=None, say=print, fetch_fov=None):
        self.plan = plan
        self.root = os.path.abspath(plan.root)
        self.obs_root = os.path.join(self.root, "observations")
        work = os.path.join(self.root, ".chanmerge")
        os.makedirs(os.path.join(work, "logs"), exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.log_path = os.path.join(work, "logs", f"{plan.target}_{stamp}.log")
        self.ciao = ciao or C.Ciao(os.path.join(work, "pfiles"), log_path=self.log_path)
        if self.ciao.log_path is None:
            self.ciao.log_path = self.log_path
        self.say = say
        self.report = Report()
        self._readcorr_cache = {}
        self._arm_cache = {}       # obsid -> leak measurement (armwidth.measure), the same for every band and group
        self._letg_cache = {}      # obsid -> letg imaging event file and its narrow box (letgsort), or None
        self._leave_now = {}       # obsid -> why it cannot be mixed (rule 3), found by the last correct_and_tables
        self._folders_made = []    # image folders made by merge_group (removed again when a group is redone without an observation)
        self._had_data = {}        # obsid -> its data was already on disk before this run (then it is never ours to delete)

    # ---------- which observations ----------
    def obs_dir(self, oid):
        return os.path.join(self.obs_root, self.plan.observations[oid].folder)

    def needed(self, skip_standby=False):
        """observations to download or process. skip_standby: a manual group with a header problem waits for the user's answer, so an
        observation that only that group needs is not reprocessed now"""
        p = self.plan
        standby = {hp.group for hp in self.report.manual_problems} if skip_standby else set()
        in_groups = {o for g in p.groups if g.group_id not in standby for o in g.obsids}
        out = []
        for oid, ob in p.observations.items():
            if ob.stage in ("download", "repro") or (ob.stage == "flux" and oid in in_groups):
                out.append(oid)
        return sorted(out, key=lambda o: p.observations[o].start)

    # ---------- phase 1: downloads and headers ----------
    def fetch(self):
        os.makedirs(self.obs_root, exist_ok=True)
        todo = self.needed()
        for i, oid in enumerate(todo, 1):
            d = self.obs_dir(oid)
            if os.path.exists(os.path.join(d, "repro", C.REPRO_MARKER)):
                self.report.reused.append(oid)
                self._had_data[oid] = True
                continue
            self._had_data[oid] = C.primary_evt2(d) is not None
            self.say(f"[{i}/{len(todo)}] {oid} download ...")
            try:
                self.ciao.download(self.obs_root, oid, self.plan.observations[oid].category.is_grating)
            except Exception as e:
                self.report.failed_obs[oid] = f"download: {e}"
                self.say(f"  {oid} download failed, skipped")
        for oid in todo:
            if oid in self.report.failed_obs:
                continue
            self._read_header(oid)
        return self.header_check()

    def _read_header(self, oid):
        ob = self.plan.observations[oid]
        evt = C.primary_evt2(self.obs_dir(oid))
        if evt is None:
            fr = C.repro_files(self.obs_dir(oid))["evt2"]
            evt = fr
        if evt is None:
            return
        try:
            h = C.read_header(evt)
        except Exception as e:
            self.report.notes.append(f"{oid}: header not readable ({e})")
            return
        ob.header_start = C.parse_header_time(h.get("DATE-OBS"))
        ob.header_end = C.parse_header_time(h.get("DATE-END"))
        ob.header_exposure_s = float(h["EXPOSURE"]) if h.get("EXPOSURE") is not None else None
        ob.header_mode, ob.header_window = C.header_mode_window(h)
        cat_w = ob.category.window
        if cat_w in ("full", "sub") and ob.header_window in ("full", "sub") and ob.header_window != cat_w:
            # test x, session 007: ngc 1068 329 is a 128-row subarray in its header while ocat says SUBARY NONE; the header describes the
            # data that was taken, so it decides every window-dependent step (the observation stays in its table row)
            rows = h.get("NROWS")
            self.report.notes.append(f"{oid}: Ocat says {'full frame' if cat_w == 'full' else 'subarray'} but its header says "
                                     f"{'a ' + str(rows) + '-row subarray' if ob.header_window == 'sub' else 'full frame'}; the header is used "
                                     f"(the observation stays in its table row)")

    def window_of(self, oid):
        """full / sub for every processing decision: the header when it was read (it describes the data), else the table row"""
        ob = self.plan.observations[oid]
        return ob.header_window if ob.header_window in ("full", "sub") else ob.category.window

    def cc_row_for(self, ob):
        """the cc row of the category table that fits this observation (same detector and grating) and the stage asked there,
        the highest one if several observations of the row differ; None when the table has no such row"""
        best = None
        for c in self.plan.categories:
            if c.is_cc and c.detector == ob.category.detector and c.grating == ob.category.grating:
                st = max((o.stage for o in self.plan.observations.values() if o.category == c), key=lambda x: STAGE_RANK[x], default="none")
                if best is None or STAGE_RANK[st] > STAGE_RANK[best[1]]:
                    best = (c, st)
        return best

    def _late_cc(self, oid, ob):
        """an observation asked for flux turned out to be cc (only the header tells when ocat is down or disagrees).
        cc has no flux, but it may be wanted as cc: the stage of the table's cc row decides (repro or download); no such row or
        stage none means it was never asked for, so nothing is done for it and, if this run downloaded it, deleting is offered"""
        row = self.cc_row_for(ob)
        wanted = row[1] if row else "none"
        if wanted == "flux":
            wanted = "repro"
        if wanted != "none":
            ob.category, ob.stage, ob.mode_source = row[0], wanted, "header"
            self.report.notes.append(f"{oid} is CC mode in its header, so no image can be made from it. It joined the CC row "
                                     f"{row[0].label()} and is processed up to {wanted}, as that row asks")
            return
        ob.stage = "none"
        path = self.obs_dir(oid)
        had = self._had_data.get(oid, True)
        size = folder_size(path)
        self.report.cc_unwanted.append(dict(obsid=oid, path=path, bytes=size, had_data=had, deletable=(not had) and os.path.isdir(path)))
        self.report.notes.append(f"{oid} is CC mode in its header and no CC row of the category table asks for CC data: nothing is done for it"
                                 + (" (its data was already on disk before this run and is left untouched)" if had else ""))

    def header_check(self):
        """only mode and window can be unknown ('?'); observations never move to another row or set.
        cc found late (asked for flux): it joins the cc row of the category table at the stage asked there, or, if no cc row asks for
        cc data, it gets stage none and is offered for deletion; either way it leaves automatic groups with a notice and manual
        groups are left alone, reported and put on standby.
        a group that now mixes full frame and subarray: handed back for an edit."""
        p, probs, manual = self.plan, [], []
        for oid, ob in list(p.observations.items()):
            if ob.header_mode == "CC" and not ob.category.is_cc and ob.stage == "flux":
                self._late_cc(oid, ob)
        for g in list(p.groups):
            for oid in list(g.obsids):
                ob = p.observations[oid]
                if ob.header_mode == "CC":
                    why = "Ocat could not be reached" if not p.ocat_ok else "Ocat and HEASARC disagreed or were empty"
                    hp = HeaderProblem("cc", oid, g.group_id, g.manual,
                                       f"{oid} is CC mode in its header; merge_obs cannot use it ({why}, so it was not caught before)")
                    if g.manual:
                        manual.append(hp)
                    else:
                        g.obsids.remove(oid)
                        self.report.notes.append(f"{oid} removed from {g.group_id}: CC mode in its header")
            wins = set()
            for oid in g.obsids:
                ob = p.observations[oid]
                w = ob.header_window or ob.category.window
                if w in ("full", "sub"):
                    wins.add(w)
            # a mix only the headers show: the window was unknown ('?') or ocat disagreed with the header (test x, 329)
            if wins == {"full", "sub"} and any(p.observations[o].category.window == "?" or self.window_of(o) != p.observations[o].category.window
                                                 for o in g.obsids):
                probs.append(HeaderProblem("full_sub", ",".join(g.obsids), g.group_id, g.manual,
                                           f"{g.group_id} now mixes full-frame and subarray observations (window known only from headers)"))
            if not g.obsids and not g.manual:
                p.groups.remove(g)
                self.report.notes.append(f"{g.group_id} is empty after the header check and was left out")
        self.report.header_problems = probs
        self.report.manual_problems = manual
        return probs

    # ---------- phase 2: repro ----------
    def repro_all(self):
        todo = [o for o in self.needed(skip_standby=True) if stage_at_least(self.plan.observations[o].stage, "repro")
                and o not in self.report.failed_obs]
        for i, oid in enumerate(todo, 1):
            d = self.obs_dir(oid)
            try:
                st = self.ciao.repro(d)          # punlearn ardlib runs inside, right after chandra_repro
                if st == "done":
                    self.say(f"[{i}/{len(todo)}] {oid} repro ... ok")
            except Exception as e:
                self.report.failed_obs[oid] = f"repro: {e}"
                self.say(f"[{i}/{len(todo)}] {oid} repro failed, skipped")

    # ---------- phase 3: merging ----------
    def run_dir(self, band):
        return os.path.join(self.root, self.plan.target, f"merged_final_image_{band.folder_label}_{self.plan.window_tag()}")

    def group_times(self, oids):
        obs = [self.plan.observations[o] for o in oids]
        st = min(o.best_start for o in obs)
        en = max(o.end for o in obs)
        return st, en

    def folder_name(self, image_id, group, st, en):
        gid = group.group_id
        if gid.startswith("all"):
            return image_id
        if gid.startswith("s") and len(group.obsids) == 1:
            return f"{image_id}_{_date(st)}"
        return f"{image_id}_{_date(st)}_{_date(en)}"

    def arm_region(self, oid):
        """repro tgmask -> primary evt2 REGION -> tgdetect2 + tg_create_mask; content checked; never prompts.
        region coordinates are in this observation's own sky frame; tgmask files carry no wcs, so the frame
        always comes from the observation's event file (needed to move the boxes into the merge_obs frame)."""
        if oid in getattr(self, "_region_cache", {}):
            return self._region_cache[oid]
        self._region_cache = getattr(self, "_region_cache", {})
        if self.is_acis_letg(oid):
            li = self.letg_imaging(oid)
            if li:
                reg = M.read_region_file(li["mask"])
                if reg:
                    from astropy.io import fits
                    reg["wcs"] = column_wcs(fits.getheader(li["evt"], "EVENTS"), "x", "y")
                    self._region_cache[oid] = (reg, f"narrow LETG box {li['width']:.0f} px")
                    return self._region_cache[oid]
        d = self.obs_dir(oid)
        evt = C.repro_files(d)["evt2"] or C.primary_evt2(d)
        frame = None
        if evt:
            try:
                from astropy.io import fits
                frame = column_wcs(fits.getheader(evt, "EVENTS"), "x", "y")
            except Exception:
                frame = None
        found = (None, "")
        for how, path in M.find_region_files(os.path.join(d, "repro"), os.path.join(d, "primary")):
            reg = M.read_region_file(path)
            if reg:
                found = (reg, how)
                break
        if found[0] is None and evt:
            try:
                mask = self.ciao.make_arm_mask(evt, os.path.join(d, "repro", "chanmerge_tgmask"))
                reg = M.read_region_file(mask)
                if reg:
                    found = (reg, "tgdetect2 + tg_create_mask")
            except Exception as e:
                self.report.notes.append(f"{oid}: tgdetect2/tg_create_mask failed ({e})")
        if found[0] is not None and frame is not None:
            found[0]["wcs"] = frame
        self._region_cache[oid] = found
        return found

    def is_acis_letg(self, oid):
        """ACIS + LETG in timed exposure: the only case whose images need the narrow re-sorting (hrc images have no tg_m filter)"""
        ob = self.plan.observations[oid]
        mode = ob.header_mode if (ob.category.mode == "?" and ob.header_mode in ("TE", "CC")) else ob.category.mode
        return ob.category.grating == "LETG" and ob.category.is_acis and mode == "TE"

    def base_evt(self, oid):
        """the event file merge_obs (and acisreadcorr) start from: the repro evt2, or for ACIS + LETG its imaging copy"""
        evt = C.repro_files(self.obs_dir(oid))["evt2"]
        if self.is_acis_letg(oid):
            li = self.letg_imaging(oid)
            if li:
                return li["evt"]
        return evt

    def letg_imaging(self, oid):
        """ACIS + LETG (session 007, test x): the repro evt2 order-sorted again with a narrow arm box chosen per observation (letgsort),
        once per run and shared by every band and group: <target>/letg_imaging/<target>_<obsid>_letg_imaging_w<width>_evt.fits and its
        _tgmask.fits. returns dict(evt, mask, width, why) or None (then ciao's own box is used and the observation gets its own image)"""
        if oid in self._letg_cache:
            return self._letg_cache[oid]
        self._letg_cache[oid] = None
        p = self.plan
        d = self.obs_dir(oid)
        files = C.repro_files(d)
        evt2 = files["evt2"]
        if not evt2:
            return None
        work = os.path.join(p.root, p.target, "letg_imaging")
        tmp = os.path.join(work, f".{oid}_work")
        try:
            from astropy.io import fits
            src = next((path for how, path in M.find_region_files(os.path.join(d, "repro"), os.path.join(d, "primary"))
                        if how == "repro tgmask"), None)
            if src is None:
                src = self.ciao.make_arm_mask(evt2, os.path.join(d, "repro", "chanmerge_tgmask"))
            ciao_reg = M.read_region_file(src)
            if not ciao_reg:
                raise RuntimeError("no readable LETG mask")
            os.makedirs(tmp, exist_ok=True)
            self.say(f"{oid} LETG: choosing a narrow arm box for the images ...")
            self.ciao.fluximage(evt2, os.path.join(tmp, "obs"), "broad", files["asol"], files["bpix"], files["msk"], files["dtf"])
            expf = glob.glob(os.path.join(glob.escape(tmp), "obs*_thresh.expmap"))[0]
            x = np.nan_to_num(fits.getdata(expf).astype(float))
            hdr = fits.getheader(expf)
            px, py = image_sky_grid(hdr, x.shape)
            ob = p.observations[oid]
            window = self.window_of(oid)
            results, made = [], {}

            def candidate(w):
                mask = LS.narrow_mask(src, os.path.join(tmp, f"w{w:.0f}_tgmask.fits"), w)
                res = os.path.join(tmp, f"w{w:.0f}_resolved.fits")
                self.ciao.tg_resolve_events(evt2, mask, files["asol"], res)
                cand = LS.copy_tg_columns(evt2, res, os.path.join(tmp, f"w{w:.0f}_evt.fits"))
                os.remove(res)
                made[w] = (cand, mask)
                return mask, cand

            for w in LS.CANDIDATE_WIDTHS:
                reg = M.read_region_file(LS.narrow_mask(src, os.path.join(tmp, f"w{w:.0f}_probe.fits"), w))
                if window == "full" and M.arm_share(px, py, reg, x > 0) > ARM_SHARE_CAP_FULL:
                    break
                mask, cand = candidate(w)
                c, ex, ey = LS.bin_image_events(cand, hdr, x.shape)
                b = AW.measure(c, x, px, py, hdr, reg, ex, ey)["boxes"][0]
                results.append((w, b["status"], b["A"]))
                width, why = LS.choose_width(results)
                if b["status"] != "leak" or width != w:
                    break
            width, why = LS.choose_width(results)
            if width not in made:
                candidate(width)
            cand, mask = made[width]
            os.makedirs(work, exist_ok=True)
            base = os.path.join(work, f"{p.target}_{oid}_letg_imaging_w{width:.0f}")
            shutil.move(cand, base + "_evt.fits")
            shutil.move(mask, base + "_tgmask.fits")
            tried = ", ".join(f"{w:.0f} px {st}" + (f" A {a:.2f}" if st == "leak" else "") for w, st, a in results)
            self.report.notes.append(f"{oid}: LETG events order-sorted again with a {width:.0f} px arm box for the images (ciao's box "
                                     f"{max(b['width'] for b in ciao_reg['boxes']):.0f} px covers far more and its filter darkens it; tried: "
                                     f"{tried or 'none'}): {why}. The repro files are untouched; the imaging event file is not for spectra")
            self._letg_cache[oid] = dict(evt=base + "_evt.fits", mask=base + "_tgmask.fits", width=width, why=why)
        except Exception as e:
            self.report.notes.append(f"{oid}: the narrow LETG re-sorting failed ({str(e)[-300:]}); ciao's own box is used")
        finally:
            if os.path.isdir(tmp):
                shutil.rmtree(tmp)
        return self._letg_cache[oid]

    def readcorr_path(self, oid):
        """one acisreadcorr file per observation and target, shared by every band and group:
        <target>/acisreadcorr/<target>_<obsid>_acisreadcorr_evt.fits (it used to land in the first group's folder and
        no table named it; test d). observations/ is never written to."""
        p = self.plan
        return os.path.join(p.root, p.target, "acisreadcorr", f"{p.target}_{oid}_acisreadcorr_evt.fits")

    def readcorr_input(self, oid, folder, prefix):
        """acisreadcorr on the repro evt2, cached per observation; returns the corrected file or None"""
        if oid in self._readcorr_cache:
            return self._readcorr_cache[oid]
        d = self.obs_dir(oid)
        files = C.repro_files(d)
        out = None
        try:
            base = self.base_evt(oid)
            ev, hd = load_events(base)
            lines = S.detect_streaks(ev)
            self.report.streak_search[(oid, "acisreadcorr, 0.5-7 keV")] = (S.searched_chips(ev), tuple(sorted({L["ccd"] for L in lines})))
            if lines:
                wcs = column_wcs(hd, "x", "y")
                zeroth = None
                if self.plan.observations[oid].category.is_grating:
                    reg, _ = self.arm_region(oid)
                    if reg:
                        zeroth = (reg["x"], reg["y"])
                obs_xy = None
                if hd.get("RA_TARG") is not None and hd.get("DEC_TARG") is not None:
                    obs_xy = radec_to_sky(float(hd["RA_TARG"]), float(hd["DEC_TARG"]), *wcs)
                pos = S.streak_source(lines[0], image_events(ev, 500.0, 7000.0), zeroth, obs_xy)
                if pos is None:
                    self.report.notes.append(f"{oid}: streak found but its source could not be placed on it; acisreadcorr skipped")
                else:
                    out = self.readcorr_path(oid)
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    self.ciao.acisreadcorr(base, out, ",".join(files["asol"]), pos[0], pos[1],
                                           S.bkg_rotbox(lines[0]), lines[0]["ccd"])
                    self.report.streak[oid] = "acisreadcorr"
        except Exception as e:
            self.report.notes.append(f"{oid}: acisreadcorr step failed ({e}); original events used")
            out = None
        self._readcorr_cache[oid] = out
        return out

    def merge_group(self, band, group, cs, image_id, run_dir):
        p = self.plan
        oids = [o for o in group.obsids if o not in self.report.failed_obs]
        if not oids:
            self.report.failed_groups[image_id] = "no usable observation"
            return None, []
        st, en = self.group_times(oids)
        folder = os.path.join(run_dir, self.folder_name(image_id, group, st, en))
        os.makedirs(folder, exist_ok=True)
        self._folders_made.append(folder)
        prefix = f"{p.target}_{p.window_tag()}_{image_id}"
        outroot = os.path.join(folder, prefix)
        rs = p.row_set(group.row_set)
        action = rs.streak_action if rs else "leave"
        inputs, asol, bpix, msk, dtf = [], [], [], [], []
        for oid in oids:
            f = C.repro_files(self.obs_dir(oid))
            evt = self.base_evt(oid)
            ob = p.observations[oid]
            if action == "acisreadcorr" and ob.category.is_acis:
                corr = self.readcorr_input(oid, folder, prefix)
                if corr:
                    evt = corr
            inputs.append(evt + cs.ciao_filter())
            asol += f["asol"]; bpix += f["bpix"]; msk += f["msk"]; dtf += f["dtf"]
        self.say(f"{image_id} merge_obs ({band.name}) ...")
        self.ciao.merge_obs(outroot, inputs, band.merge_spec, asol, bpix, msk, dtf)
        prod = self.find_products(outroot, band, oids)
        if prod is None:
            raise RuntimeError("merge_obs finished but its flux image was not found")
        return self.correct_and_tables(band, group, cs, image_id, run_dir, folder, prefix, oids, prod, action, st, en)

    def find_products(self, outroot, band, oids):
        lab = band.file_label
        merged = dict(flux=f"{outroot}_{lab}_flux.img", thresh=f"{outroot}_{lab}_thresh.img",
                      expmap=f"{outroot}_{lab}_thresh.expmap", evt=f"{outroot}_merged_evt.fits")
        if not os.path.exists(merged["flux"]):
            # merge_obs may label a numeric band differently: find it without matching member files
            cands = [c for c in glob.glob(f"{glob.escape(outroot)}_*_flux.img") if not any(f"_{o}_" in os.path.basename(c)[len(os.path.basename(outroot)):] for o in oids)]
            if not cands:
                return None
            lab = os.path.basename(cands[0])[len(os.path.basename(outroot)) + 1:-len("_flux.img")]
            merged = dict(flux=f"{outroot}_{lab}_flux.img", thresh=f"{outroot}_{lab}_thresh.img",
                          expmap=f"{outroot}_{lab}_thresh.expmap", evt=f"{outroot}_merged_evt.fits")
        members = {}
        for o in oids:
            members[o] = dict(flux=f"{outroot}_{o}_{lab}_flux.img", thresh=f"{outroot}_{o}_{lab}_thresh.img",
                              expmap=f"{outroot}_{o}_{lab}_thresh.expmap", evt=f"{outroot}_{o}_reproj_evt.fits")
        return dict(label=lab, merged=merged, members=members)

    def correct_and_tables(self, band, group, cs, image_id, run_dir, folder, prefix, oids, prod, action, st, en):
        from astropy.io import fits
        p = self.plan
        lab = prod["label"]
        elo, ehi = (band.lo * 1000, band.hi * 1000) if (band.kind == "acis" and band.lo is not None) else (None, None)
        results, member_rows = {}, []
        any_fill = any_mask = False
        self._leave_now = {}
        for oid in oids:
            ob = p.observations[oid]
            mp = prod["members"][oid]
            if not os.path.exists(mp["thresh"]):
                continue
            # a '?' row keeps '?' in the table (the observation never moves rows); the header tells the real mode and window
            mode = ob.header_mode if (ob.category.mode == "?" and ob.header_mode in ("TE", "CC")) else ob.category.mode
            window = self.window_of(oid)
            arm_method = p.arm_method_by_row.get(p.row_of(oid)) if (ob.category.is_grating and mode == "TE") else None
            if ob.category.is_grating and mode == "?" and p.arm_method_by_row.get(p.row_of(oid)) in (2, 3, 4):
                self.report.notes.append(f"{oid}: mode unknown (no readable header), arms not treated in {image_id}, {band.name}")
            streak_action = action if (action in ("fill", "mask") and ob.category.is_acis) else "leave"
            # option 1 too: the box can cover the whole image (letg 2324 type), then the image is never mixed with others (rule 3)
            if elo is None or (arm_method is None and streak_action == "leave"):
                continue
            reg = None
            if arm_method in (1, 2, 3, 4):
                reg, how = self.arm_region(oid)
                if reg is None and arm_method != 1:
                    self.report.notes.append(f"{oid}: no grating arm mask could be made; arms not treated in {image_id}, {band.name}")
            c0 = fits.getdata(mp["thresh"]).astype(float)
            x0 = np.nan_to_num(fits.getdata(mp["expmap"]).astype(float))
            h0 = fits.getheader(mp["thresh"])
            r = correct_member(oid, c0, x0, h0, mp["evt"], arm_method=arm_method, arm_region=reg,
                               streak_action=streak_action, elo_ev=elo, ehi_ev=ehi,
                               seed=int(oid) % 100000 + 1,
                               arm_share_cap=ARM_SHARE_CAP_FULL if window == "full" else None, arm_gate=True,
                               arm_measure=self._arm_cache.get(oid), arm_leak=True)
            if r.arm_measure is not None:
                self._arm_cache[oid] = r.arm_measure
            if r.untreatable:
                if len(oids) > 1:
                    # rule 3: it must not be mixed with the others; merge_all takes it out of the group (automatic) or asks (manual)
                    self._leave_now[oid] = r.untreatable
                else:
                    self.report.notes.append(f"{image_id}, {band.name}: {oid} is not arm-treated (filter only) and stays an image of its own: {r.untreatable}")
            if r.streak_searched is not None:
                self.report.streak_search[(oid, band.name)] = (r.streak_searched, tuple(sorted({L["ccd"] for L in r.streak_lines})))
            # one image per group and band, so say which one (otherwise notes look repeated)
            self.report.notes += [f"{image_id}, {band.name}: {n}" for n in r.notes]
            if r.changed:
                results[oid] = (r, c0, x0, h0)
                any_fill |= r.did_fill; any_mask |= r.did_mask
            if r.streak_done:
                self.report.streak[oid] = r.streak_done
            if r.arm_done:
                self.report.arms[oid] = r.arm_done
        gsfx = ("_filled" if any_fill else "") + ("_masked" if any_mask else "")
        mg = prod["merged"]
        # merged images: original paths, then the corrected versions when something changed
        flux_img, flux_orig = mg["flux"], mg["flux"]
        exp_img, exp_orig = mg["expmap"], mg["expmap"]
        evt_img, evt_orig = mg["evt"], mg["evt"]
        mask_file = ""
        filled_frac = 0.0
        mc = fits.getdata(mg["thresh"]).astype(float)
        mx = np.nan_to_num(fits.getdata(mg["expmap"]).astype(float))
        mh = fits.getheader(mg["thresh"])
        fc, fx = mc, mx
        if results:
            originals = [(results[o][1], results[o][2]) for o in results]
            corrected = [(results[o][0].counts, results[o][0].expmap) for o in results]
            fc, fx = merged_recompute(mc, mx, originals, corrected)
            filled_any = np.zeros(mc.shape, bool); masked_any = np.zeros(mc.shape, bool)
            for o in results:
                filled_any |= results[o][0].filled; masked_any |= results[o][0].masked
            base = f"{prefix}_{lab}"
            flux_img = os.path.join(folder, f"{base}_flux{gsfx}.img")
            exp_img = os.path.join(folder, f"{base}_thresh{gsfx}.expmap")
            write_image(flux_img, corrected_flux(fits.getdata(mg["flux"]), fc, fx, mx), fits.getheader(mg["flux"]),
                        history="chanmerge: corrected merged flux (NaN = no data made by chanmerge)")
            write_image(os.path.join(folder, f"{base}_thresh{gsfx}.img"), fc, mh, np.int32, "chanmerge: corrected counts")
            write_image(exp_img, fx, fits.getheader(mg["expmap"]), history="chanmerge: corrected exposure")
            mask_file = os.path.join(folder, f"{base}_mask.img")
            thinned_any = np.zeros(mc.shape, bool)
            for o in results:
                if results[o][0].thinned is not None:
                    thinned_any |= results[o][0].thinned
            write_image(mask_file, mask_codes(filled_any & ~masked_any, masked_any, thinned_any), mh, np.uint8,
                        "chanmerge: 0 untouched, 1 no data (masked), 2 filled, 3 arm leak subtracted")
            # share of the pixels merge_obs exposed (pixels made no data stay in the denominator)
            filled_frac = float((filled_any & (mx > 0)).sum() / max((mx > 0).sum(), 1))
            # new event files: corrected members, unchanged ones with FILLED=0, then one merged list
            evt_inputs = []
            for o in oids:
                mp = prod["members"][o]
                if not os.path.exists(mp["evt"]):
                    continue
                if o in results:
                    r, c0, x0, h0 = results[o]
                    sfx = r.suffix
                    out = os.path.join(folder, f"{prefix}_{o}_evt{sfx}.fits")
                    correct_events(mp["evt"], h0, r.counts, r.filled, r.masked, r.rpix, r.ring_of_xy(h0["CDELT1P"]),
                                   out, elo, ehi, seed=int(o) % 100000 + 7, thinned=r.thinned)
                    results[o] = (r, c0, x0, h0, out, sfx)
                    evt_inputs.append(out)
                else:
                    tmp = os.path.join(folder, f".{prefix}_{o}_evt_filled0.fits")
                    add_filled_column(mp["evt"], tmp)
                    evt_inputs.append(tmp)
            evt_img = os.path.join(folder, f"{prefix}_merged_evt{gsfx}.fits")
            try:
                self.ciao.dmmerge(evt_inputs, common_columns(evt_inputs), evt_img)
            except Exception as e:
                self.report.notes.append(f"{image_id}: merged event list not rebuilt ({e}); evt_file stays the merge_obs list")
                evt_img = mg["evt"]
            for f in evt_inputs:
                if os.path.basename(f).startswith("."):
                    os.remove(f)
        p5, pmed, p95 = expmap_stats(fx)
        obs = [p.observations[o] for o in oids]
        cats = sorted({o.category.label() for o in obs})
        repro_evts = [C.repro_files(self.obs_dir(o))["evt2"] or "" for o in oids]
        # the acisreadcorr file merge_obs used for each member ('' where none was used)
        readcorr = [(self._readcorr_cache.get(o) or "") if (action == "acisreadcorr" and p.observations[o].category.is_acis) else ""
                    for o in oids]
        letg = [((self._letg_cache.get(o) or {}).get("evt") or "") for o in oids]
        rd = run_dir
        relp = lambda x: rel(x, rd) if x else ""
        streak_obsids = [o for o in oids if self.report.streak.get(o)]
        final = dict(target=p.target, band=band.name, window_days=p.window_days, group_id=image_id, chips=cs.names(),
                     type="merged" if len(oids) > 1 else "single", n_obs=len(oids), obsids=";".join(oids),
                     categories=";".join(cats), n_grating=sum(o.category.is_grating for o in obs),
                     start_utc=_iso(st), end_utc=_iso(en), span_days=round((en - st).total_seconds() / 86400, 3),
                     total_exposure_ks=round(sum((o.header_exposure_s or o.exposure_s) for o in obs) / 1000, 1),
                     flux_image=relp(flux_img), flux_image_original=relp(flux_orig), evt_file=relp(evt_img),
                     evt_file_original=relp(evt_orig), repro_evt_file=";".join(relp(x) for x in repro_evts),
                     repro_evt_file_abs=";".join(repro_evts),
                     acisreadcorr_evt_file=";".join(relp(x) for x in readcorr) if any(readcorr) else "",
                     acisreadcorr_evt_file_abs=";".join(readcorr) if any(readcorr) else "",
                     letg_imaging_evt_file=";".join(relp(x) for x in letg) if any(letg) else "",
                     letg_imaging_evt_file_abs=";".join(letg) if any(letg) else "",
                     expmap_file=relp(exp_img), expmap_file_original=relp(exp_orig),
                     expmap_p5=p5, expmap_median=pmed, expmap_p95=p95, streak_action=action,
                     streak_obsids=";".join(streak_obsids),
                     arm_method=";".join(sorted({ARM_METHODS[p.arm_method_by_row[p.row_of(o)]] for o in oids
                                                 if p.observations[o].category.is_grating and p.row_of(o) in p.arm_method_by_row})),
                     arm_masked_obsids=";".join(o for o in oids if self.report.arms.get(o, "").startswith("mask_") and self.report.arms[o] != "mask_fill"),
                     arm_filled_obsids=";".join(o for o in oids if self.report.arms.get(o) == "mask_fill"),
                     filled_pixel_fraction=round(filled_frac, 5), mask_file=relp(mask_file), shares_photons_with="",
                     left_out_of=getattr(group, "left_out_of", ""))
        if len(oids) > 1:
            for o, rp, rc, lg in zip(oids, repro_evts, readcorr, letg):
                ob = p.observations[o]
                mp = prod["members"][o]
                m_flux, m_exp, m_evt, m_mask, m_frac = mp["flux"], mp["expmap"], mp["evt"], "", 0.0
                if o in results and len(results[o]) == 6:
                    r, c0, x0, h0, out, sfx = results[o]
                    base = f"{prefix}_{o}_{lab}"
                    m_flux = os.path.join(folder, f"{base}_flux{sfx}.img")
                    m_exp = os.path.join(folder, f"{base}_thresh{sfx}.expmap")
                    write_image(m_flux, corrected_flux(fits.getdata(mp["flux"]), r.counts, r.expmap, x0), fits.getheader(mp["flux"]),
                                history="chanmerge: corrected member flux (NaN = no data made by chanmerge)")
                    write_image(os.path.join(folder, f"{base}_thresh{sfx}.img"), r.counts, h0, np.int32)
                    write_image(m_exp, r.expmap, fits.getheader(mp["expmap"]))
                    m_mask = os.path.join(folder, f"{base}_mask.img")
                    write_image(m_mask, mask_codes(r.filled, r.masked, r.thinned), h0, np.uint8,
                                "chanmerge: 0 untouched, 1 no data, 2 filled, 3 arm leak subtracted")
                    m_evt = out
                    m_frac = float((r.filled & (x0 > 0)).sum() / max((x0 > 0).sum(), 1))
                member_rows.append(dict(
                    target=p.target, band=band.name, window_days=p.window_days, group_id=image_id, chips=cs.names(), obsid=o,
                    detector=ob.category.detector, grating=ob.category.grating, mode=ob.header_mode or ob.category.mode,
                    window=ob.header_window or ob.category.window, start_utc=_iso(ob.best_start), end_utc=_iso(ob.end),
                    exposure_ks=round((ob.header_exposure_s or ob.exposure_s) / 1000, 1),
                    flux_image=relp(m_flux), flux_image_original=relp(mp["flux"]), evt_file=relp(m_evt),
                    evt_file_original=relp(mp["evt"]), repro_evt_file=relp(rp), repro_evt_file_abs=rp,
                    acisreadcorr_evt_file=relp(rc), acisreadcorr_evt_file_abs=rc,
                    letg_imaging_evt_file=relp(lg), letg_imaging_evt_file_abs=lg,
                    expmap_file=relp(m_exp), expmap_file_original=relp(mp["expmap"]),
                    streak_action=self.report.streak.get(o, ""),
                    arm_method=ARM_METHODS.get(p.arm_method_by_row.get(p.row_of(o)), "") if ob.category.is_grating else "",
                    filled_pixel_fraction=round(m_frac, 5), mask_file=relp(m_mask), shares_photons_with=""))
        return final, member_rows

    def merge_all(self, only_groups=None):
        """only_groups (a rerun after an edit between two passes): only those groups are merged; rows and folders
        of those groups and of groups that are no longer in the plan are removed first, rows of the other groups
        are kept as they were.
        rule 3 (session 007): when a member of an AUTOMATIC group cannot be mixed with the others (its ciao arm box covers the whole image,
        or more than 40% of it with arm options 2-4), what was made for that group is removed, the observation leaves the group and gets
        a single image of its own (left_out_of names the group) and the group is merged again without it (with one observation left it
        becomes that observation's single). manual groups are never changed automatically: their image keeps the observation untreated
        and the user is asked at the end (report.arm_problems)."""
        p = self.plan
        per_dir = {}
        kept = {}
        if only_groups:
            kept = self.clear_for_rerun(only_groups)
        gi = 0
        while gi < len(p.groups):          # the list can grow (singles of observations that left a group) or shrink
            g = p.groups[gi]
            gi += 1
            if only_groups and g.group_id not in only_groups:
                continue
            if any(hp.group == g.group_id for hp in self.report.manual_problems):
                self.report.failed_groups[g.group_id] = "manual group with a header problem; edit it and run it again"
                continue
            while True:
                made, leave = self._merge_one_group(g, per_dir, only_groups)
                if not leave:
                    break
                if g.manual:
                    for oid, why in leave.items():
                        self.report.arm_problems.append(HeaderProblem("arm", oid, g.group_id, True, f"{oid} in {g.group_id}: {why}"))
                        self.report.notes.append(f"{g.group_id}: {oid} stays in this manual group untreated although {why}; you are asked about it at the end")
                    break
                self._remove_made(made)
                old = g.group_id
                gone = self._leave_group(g, leave)
                if gone:
                    gi -= 1        # g was taken out of the list: the next group now sits at its index
                    break
                self.report.notes.append(f"{old}: merged again without {', '.join(leave)}" + (f" (now {g.group_id})" if g.group_id != old else ""))
        for rd in kept:
            per_dir.setdefault(rd, ([], []))
        for rd, (finals, members) in per_dir.items():
            old_f, old_m = kept.get(rd, ([], []))
            self.report.images += finals
            self.report.members += members
            self.report.kept_images += len(old_f)
            finals, members = old_f + finals, old_m + members
            fill_shares_run(finals, members)
            write_csv_atomic(os.path.join(rd, "final_images.csv"), FINAL_COLUMNS, finals)
            write_csv_atomic(os.path.join(rd, "member_images.csv"), MEMBER_COLUMNS, members)
            self.report.run_dirs.append(rd)
        if per_dir:
            self.report.index_counts = rebuild_index(self.root)

    def _merge_one_group(self, g, per_dir, only_groups):
        """every band and chip set of one group. returns (folders made, {obsid: why} of members that must leave an automatic group).
        an automatic group stops at the first such member (its rows are not kept, merge_all redoes the group); a manual one goes on"""
        p = self.plan
        made, rows, leave = [], [], {}
        n_notes = len(self.report.notes)
        for band in p.bands_for(g):
            rd = self.run_dir(band)
            if rd not in per_dir:
                if os.path.exists(rd) and not only_groups:
                    if not p.overwrite:
                        raise RuntimeError(f"{rd} exists and overwrite was not chosen")
                    shutil.rmtree(rd)        # only this run folder; observations/ is never touched
                os.makedirs(rd, exist_ok=True)
                per_dir[rd] = ([], [])
            sets = p.chip_sets_for(g) if all(p.observations[o].category.is_acis for o in g.obsids) else [None]
            for cs in sets:
                from .plan import ChipSet
                cs = cs or ChipSet(None)
                many = len(sets) > 1
                label = cs.label or "_".join(CHIP_NAMES[c] for c in (cs.chips or ()))
                image_id = f"{g.group_id}_{label}" if many and label else g.group_id
                self._folders_made = []
                self._leave_now = {}
                try:
                    final, members = self.merge_group(band, g, cs, image_id, rd)
                except Exception as e:
                    self.report.failed_groups[image_id] = str(e)[-400:]
                    self.say(f"{image_id} failed: {str(e)[-200:]}")
                    made += self._folders_made
                    continue
                made += self._folders_made
                for oid, why in self._leave_now.items():
                    leave.setdefault(oid, why)
                if leave and not g.manual:
                    # the notes of this pass are about an image that is thrown away (session 007, real run: 'g01, broad: 1790 ...'
                    # stayed in the report although g01 was gone); the redo writes its own
                    del self.report.notes[n_notes:]
                    return made, leave
                if final:
                    rows.append((rd, final, members))
        for rd, final, members in rows:
            per_dir[rd][0].append(final); per_dir[rd][1].extend(members)
        return made, leave

    def _remove_made(self, folders):
        """image folders this run made for a group that is redone; never anything outside a run folder"""
        for f in folders:
            f = os.path.normpath(f)
            rd = os.path.dirname(f)
            if os.path.basename(rd).startswith("merged_final_image_") and os.path.isdir(f):
                shutil.rmtree(f)

    def _leave_group(self, g, leave):
        """rule 3 for an automatic group: the observations leave g, each gets its own single (appended to the plan, merged later in
        this run). returns True when g itself was removed from the plan (nothing left in it)"""
        from .plan import Group
        p = self.plan
        for oid, why in leave.items():
            if oid in g.obsids:
                g.obsids.remove(oid)
            self.report.left_out[oid] = g.group_id
            self.report.notes.append(f"{oid} left {g.group_id} and gets an image of its own (s{oid}): {why}")
            if p.group(f"s{oid}") is None:
                p.groups.append(Group(f"s{oid}", [oid], g.row_set, manual=False, chip_sets=g.chip_sets, left_out_of=g.group_id))
        if not g.obsids:
            p.groups.remove(g)
            return True
        if len(g.obsids) == 1 and p.group(f"s{g.obsids[0]}") is None:
            # one observation left: it is a single now, named like every other single
            g.group_id = f"s{g.obsids[0]}"
        return False

    def clear_for_rerun(self, only_groups):
        """before a rerun of some groups: in every run folder of this plan, drop the rows of the groups that are
        rerun or no longer in the plan (edited away or renamed) and delete their image folders, so no stale image
        stays behind. returns {run folder: (kept final rows, kept member rows)}"""
        p = self.plan
        keep = {g.group_id for g in p.groups if g.group_id not in only_groups}
        dirs = []
        for band in list(p.acis_bands) + list(p.hrc_bands):
            rd = self.run_dir(band)
            if rd not in dirs and os.path.exists(os.path.join(rd, "final_images.csv")):
                dirs.append(rd)
        kept, gone = {}, set()
        for rd in dirs:
            old_f = _read_if(os.path.join(rd, "final_images.csv"))
            old_m = _read_if(os.path.join(rd, "member_images.csv"))
            ok = lambda r: _base_group(r["group_id"], keep) is not None
            for r in old_f:
                if ok(r):
                    continue
                folder = os.path.normpath(os.path.join(rd, os.path.dirname(r.get("flux_image") or "")))
                if folder != os.path.normpath(rd) and folder.startswith(os.path.normpath(rd) + os.sep) and os.path.isdir(folder):
                    shutil.rmtree(folder)    # one group folder inside this run folder
                g = r["group_id"]
                base = _base_group(g, set(only_groups)) or g
                if base not in only_groups:
                    gone.add(base)
            kept[rd] = ([r for r in old_f if ok(r)], [r for r in old_m if ok(r)])
        for g in sorted(gone):
            self.report.notes.append(f"{g} is no longer in the plan: its images and table rows were removed")
        return kept


def _base_group(image_id, group_ids):
    """the group an image belongs to: g01 for g01 and g01_bi; None when it is none of group_ids"""
    for g in group_ids:
        if image_id == g or image_id.startswith(g + "_"):
            return g
    return None


def group_state(plan):
    """what decides a group's images (observations, row set, chips); compared before and after an edit"""
    return {g.group_id: (tuple(g.obsids), g.row_set, repr(plan.chip_sets_for(g))) for g in plan.groups}


def groups_to_rerun(plan, before, problem_groups):
    """after an edit between two passes: the problem groups still in the plan, new groups and changed groups"""
    now = group_state(plan)
    return {g for g in problem_groups if g in now} | {g for g, st in now.items() if before.get(g) != st}


def _read_if(path):
    from .tables import read_csv
    return read_csv(path) if os.path.exists(path) else []


def standby_notice(problems):
    """printed after the header check, before anything is reprocessed: manual groups with a problem wait, the user is asked at the end"""
    lines = ["[!] NOTICE: these manual groups have a problem and are NOT processed now (nothing is reprocessed or merged only for them):"]
    lines += [f"    {hp.group}: {hp.text}" for hp in problems]
    lines.append("    the other jobs run first; when they are finished you will be asked what to do with these groups.")
    return lines


def standby_status(problems):
    """the one line that stays visible (bottom line of the terminal, red bar in the window) for as long as the run goes on.
    short on purpose: an 80-column terminal cuts anything longer, so the group names come first and the plain meaning follows"""
    names = list(dict.fromkeys(hp.group for hp in problems))
    return f"[!] {', '.join(names)} {'is' if len(names) == 1 else 'are'} on hold (problem). You will be asked about it after the other jobs."


def folder_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def cc_delete_question(plan, d):
    """the text of the 'delete the data of a cc observation nobody asked for?' question (terminal and window)"""
    from .archive import human
    why = "Ocat could not be reached" if not plan.ocat_ok else "Ocat and HEASARC disagreed or were empty"
    return (f"{d['obsid']} turned out to be CC mode; only its header showed it ({why}). No CC row of the category table asks for CC data and "
            f"CC cannot make an image, so nothing was done for it. This run downloaded {human(d['bytes'])} for it "
            f"({os.path.relpath(d['path'], plan.root)}). Delete these files?")


def delete_downloaded(plan, d):
    """remove the folder this run downloaded for a cc observation nobody asked for (d is a Report.cc_unwanted entry).
    refuses anything that is not marked deletable, is not directly inside <root>/observations/, or holds a finished repro"""
    obs_root = os.path.realpath(os.path.join(plan.root, "observations"))
    path = os.path.realpath(d["path"])
    if not d.get("deletable") or os.path.dirname(path) != obs_root or not os.path.isdir(path):
        return False
    if os.path.exists(os.path.join(path, "repro", C.REPRO_MARKER)):
        return False
    shutil.rmtree(path)
    return True


def run(plan, on_header_problems=None, ciao=None, say=print, only_groups=None, status=None):
    """status(text) keeps one line visible while the run goes on (None clears it); only the real front ends pass it"""
    status = status or (lambda text: None)
    r = Runner(plan, ciao=ciao, say=say)
    try:
        probs = r.fetch()
        if probs:
            if on_header_problems is None or not on_header_problems(plan, probs):
                r.report.stopped = True
                return r.report
            # the plan may have changed: read the check again for the edited groups
            r.header_check()
        if r.report.manual_problems:
            for line in standby_notice(r.report.manual_problems):
                r.say(line)
            status(standby_status(r.report.manual_problems))
        r.repro_all()
        if any(o.stage == "flux" for o in plan.observations.values()) and plan.groups:
            r.merge_all(only_groups=only_groups)
        return r.report
    finally:
        status(None)