"""Terminal flow: asks everything before any download, fills a RunPlan, then calls run(plan).

invalid input never crashes; it explains and asks again.
"""
import atexit
import os
import shutil
import sys

from . import archive as A
from .bands import parse_bands, BandError
from .check import check_plan
from .geometry import fov_lookup
from .grouping import build_groups, apply_edit, finish_edits, span_days, obs_in_no_group, no_group_split
from .inputs import (InputError, parse_ra, parse_dec, format_ra, format_dec, parse_radius, default_target, parse_date,
                     check_date_order, check_folder_name, check_run_path, parse_yes_no, parse_stage, parse_window,
                     parse_arm_method, parse_stage_edit, parse_row_sets, format_row_sets, parse_chip_menu, chip_choice,
                     parse_streak_sets, parse_streak_action, parse_group_edit, CHIP_MENU)
from .plan import RunPlan, RowSet, CHIP_NAMES, BI_CHIPS, FI_CHIPS
from .ciao import REPRO_MARKER

STAGE_HELP = """Each stage includes the ones before it:
  none      skip this row
  download  download only
  repro     download + reprocess
  flux      download + reprocess + flux image"""

ARM_OPTIONS = [
    ("merge_obs filter only (no extra mask or fill)", "arm core and a dark corridor beside it stay in the image"),
    ("mask arms, keep the source", "strips of no data; about 17% of the area in the one test"),
    ("mask arms, remove the source", "the source is removed, images are not comparable with non-grating ones"),
    ("mask arms, fill from the same radius", "arm boxes refilled from the same radius; about 17% of the area\n"
                                             "      in the tests is simulated, marked in the tables"),
]
def subarray_arm_warning(plan, r, method):
    """grating subarray row + arm option 2-4: the strip runs along the arms (decided 2026-11-04). options 2-3 leave that
    share as no data; option 4 fills it, so the text says simulated, not lost (user, 2026-11-04 evening, after 16702).
    numbers: geometry estimate of 2026-11-03 (512/350/256/146 rows), 7490 (test a) and 16702 (session 002) measured."""
    c = plan.categories[r - 1]
    if c.grating == "LETG" and c.detector.startswith("ACIS"):
        # test w (session 007): inside ciao's letg box merge_obs's tg_m filter removes most events, background included; where the box
        # covers the whole image (2324, 55, 3089, letg subarrays) the whole image is darkened. so this warning is shown for every option
        return (f"row {r} is LETG. CIAO draws the LETG arm box very wide on purpose (for spectral background, ~2900 sky px on ACIS-S) and\n"
                f"      merge_obs's filter removes most events inside it, background included, so CIAO's own LETG image is darkened.\n"
                f"      chanmerge therefore order-sorts the LETG events again with a narrow box chosen per observation (100-400 px,\n"
                f"      from the arm light measured beside it) and uses that copy for the images only; the repro and spectral files are\n"
                f"      untouched and the copy must not be used for spectra. An observation whose box still covers its whole image\n"
                f"      (e.g. a LETG subarray strip) is never merged with other observations; it gets an image of its own.\n"
                + (f"      With this option a full-frame observation whose box covers more than 40% of its image is treated the same way.\n"
                   if method in (2, 3, 4) else "")
                + f"      Stage 'none' for this row avoids LETG data.")
    if method not in (2, 3, 4):
        return None
    if not (c.is_grating and c.window == "sub"):
        return None
    obs = [o for o in plan.observations.values() if o.category == c and o.stage == "flux"]
    if not obs:
        return None
    rows = ", ".join(f"{o.obsid}: {o.subarray_rows} rows" if o.subarray_rows else f"{o.obsid}: rows unknown" for o in obs)
    share = ("(geometry estimate: 26% at 512 rows, 37% at 350, 50% at 256, 85% at 146;\n"
             "      measured on 16702, 512 rows: 25-27%)")
    if method == 4:
        # option 4 fills the boxes, so most of that share is simulated, not lost (16702 measured, session 002)
        return (f"row {r} is a grating subarray ({rows}). Its strip runs along the grating arms, so a large share of it\n"
                f"      lies in the arm boxes and will be simulated (filled from the same radius) instead of real data,\n"
                f"      more the fewer rows {share}. In a narrow strip the part far from the source may\n"
                f"      have no clean pixels to fill from and stays no data (HETG 7490, 128 rows: 69%; HETG 16702, 512 rows: 0.3%).\n"
                f"      Option 1 or stage 'none' for this row keeps only real data.")
    return (f"row {r} is a grating subarray ({rows}). Its strip runs along the grating arms, so masking the arms\n"
            f"      leaves a large share of it as no data, more the fewer rows {share}.\n"
            f"      Option 1 or stage 'none' for this row avoids this.")


STREAK_HELP = """If a readout streak is detected:
  acisreadcorr  move streak events back to the source; a dark stripe may remain
  fill          replace the streak band with simulated counts from the same radius;
                marked in the tables
  mask          mark the streak band as no data (counts and exposure 0)"""
WINDOW0_WARNING = ("[!] WARNING: time grouping is OFF (window=0). All observations will be merged regardless of when "
                   "they were taken; for DSH work this can mix different halo states.")
OCAT_WARNING = "[!] WARNING: Chandra Ocat unreachable. CC mode will only be detected after download via FITS headers."


class Back(Exception):
    """go back to an earlier step"""
    def __init__(self, step):
        self.step = step


class Quit(Exception):
    pass


class StatusLine:
    """one line pinned to the bottom of the terminal; everything else scrolls above it, so whoever looks at the screen sees it at any
    moment. it needs a real terminal; anywhere else (tests, a pipe, a log file) it does nothing"""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.on = False
        self.text = ""
        self.rows = 0
        self.old_handler = None
        atexit.register(self.clear)           # never leave the terminal with a fixed scroll region

    def usable(self):
        return bool(getattr(self.stream, "isatty", lambda: False)()) and os.environ.get("TERM", "") not in ("", "dumb")

    def set(self, text):
        if not text:
            self.clear()
            return
        if not self.usable():
            return
        cols, rows = shutil.get_terminal_size()
        if rows < 4:
            return
        w = self.stream.write
        w("\n\033[A")                         # make sure the cursor is not on the last row (the screen scrolls one line if it was)
        w("\0337")                             # save the cursor
        w(f"\033[1;{rows - 1}r")               # scroll region: every row except the last one
        w(f"\033[{rows};1H\033[2K\033[7m{text[:cols - 1]}\033[0m")      # the last row, reversed colours so it stands out
        w("\0338")                             # put the cursor back (setting the region moves it home)
        self.stream.flush()
        self.on, self.text, self.rows = True, text, rows
        self._watch_resize(True)

    def clear(self):
        self._watch_resize(False)
        if not self.on or not self.usable():
            self.on = False
            return
        _, rows = shutil.get_terminal_size()
        w = self.stream.write
        w("\0337\033[r")                      # whole screen scrolls again
        w(f"\033[{rows};1H\033[2K")           # last row empty
        if self.rows and self.rows != rows:
            w(f"\033[{self.rows};1H\033[2K")  # the row the line had before a resize
        w("\0338")
        self.stream.flush()
        self.on = False

    def _watch_resize(self, on):
        """test r (session 005, real pseudo terminal): after a resize the line stayed on its old row in the middle of the screen. a SIGWINCH
        handler draws it again on the new last row. only in the main thread and only where the signal exists."""
        import signal
        if not hasattr(signal, "SIGWINCH"):
            return
        try:
            if on and self.old_handler is None:
                self.old_handler = signal.signal(signal.SIGWINCH, self._resized) or signal.SIG_DFL
            elif not on and self.old_handler is not None:
                signal.signal(signal.SIGWINCH, self.old_handler)
                self.old_handler = None
        except ValueError:                    # not the main thread
            pass

    def _resized(self, signum=None, frame=None):
        if not (self.on and self.text and self.usable()):
            return
        _, rows = shutil.get_terminal_size()
        old_rows, text = self.rows, self.text
        w = self.stream.write
        w("\0337\033[r")                      # scroll everything again, then clear the old row and pin the line on the new last row
        if old_rows and old_rows != rows and old_rows <= rows:
            w(f"\033[{old_rows};1H\033[2K")
        self.stream.flush()
        self.on = False
        self.set(text)


class Terminal:
    def __init__(self, input_fn=input, out=print, archive=A, run_fn=None, cwd=None):
        self.input_fn, self.out, self.A = input_fn, out, archive
        self.run_fn = run_fn
        self.cwd = cwd or os.getcwd()
        self.last_plan = None
        self.status = StatusLine() if out is print else StatusLine(stream=open(os.devnull, "w"))   # only the real screen gets the pinned line

    def run_kw(self):
        """the real run keeps a line pinned to the bottom of the screen; test doubles of run() know nothing about it"""
        return {"status": self.status.set} if self.run_fn is None else {}

    # ---------- asking ----------
    def ask(self, prompt, parse=lambda s: s):
        while True:
            try:
                text = self.input_fn(prompt)
            except EOFError:
                raise Quit()
            try:
                return parse(text)
            except (InputError, BandError) as e:
                self.out(f"  [!] {e}")

    def yes(self, prompt):
        return self.ask(prompt, parse_yes_no)

    # ---------- menu ----------
    def main(self):
        while True:
            self.out("1. New target\n2. Local directory (not available in this version)\n3. Exit")
            c = self.ask("> ").strip()
            if c == "1":
                try:
                    self.new_target()
                except Quit:
                    return
            elif c == "2":
                self.out("  Local directory mode is not available in this version.")
            elif c == "3":
                return
            else:
                self.out("  choose 1, 2 or 3")

    # ---------- the flow ----------
    def new_target(self):
        level, msg = check_run_path(self.cwd)
        if level == "error":
            self.out(f"  [!] ERROR: {msg}")
            raise Quit()
        if level == "warn":
            self.out(f"  [!] WARNING: {msg}")
            if not self.yes("Continue anyway? (y/n): "):
                return
        while True:
            plan = self.step_coordinates()
            self.last_plan = plan
            try:
                if not self.step_search(plan):
                    continue
                self.step_categories(plan)
                if plan.flux_obsids():
                    self.step_flux(plan)
                    self.step_groups(plan)
                else:
                    self.out("[!] NOTICE: No flux image will be produced.")
                if not self.step_confirm(plan):
                    return
                self.step_run(plan)
                return
            except Back as b:
                if b.step == "menu":
                    return
                continue

    def step_coordinates(self):
        self.out("\n--- Coordinates ---")
        ra = self.ask("RA (hh:mm:ss or decimal degrees): ", parse_ra)
        self.out(f"  RA = {ra:.6g} deg ({format_ra(ra)})")
        dec = self.ask("Dec (dd:mm:ss or decimal degrees): ", parse_dec)
        self.out(f"  Dec = {dec:.6g} deg ({format_dec(dec)})")
        radius = self.ask("Search radius (arcmin): ", parse_radius)
        while True:
            name = self.ask("Output folder name (blank = from coordinates): ").strip()
            if not name:
                name = default_target(ra, dec)
                self.out(f"  folder name: {name}")
                break
            level, msg = check_folder_name(name)
            if level == "error":
                self.out(f"  [!] ERROR: {msg}")
                continue
            if level == "warn":
                self.out(f"  [!] WARNING: {msg}")
                if not self.yes("Use this name anyway? (y/n): "):
                    continue
            break
        plan = RunPlan(ra=ra, dec=dec, radius_arcmin=radius, target=name, root=self.cwd)
        self.step_dates(plan)
        return plan

    def step_dates(self, plan):
        self.out("\n--- Dates (blank = no limit) ---")
        while True:
            d1 = self.ask("Start date (YYYY-MM-DD, UTC): ", parse_date)
            d2 = self.ask("End date (YYYY-MM-DD, UTC): ", parse_date)
            try:
                check_date_order(d1, d2)
                plan.date_from, plan.date_to = d1, d2
                return
            except InputError as e:
                self.out(f"  [!] {e}")

    def step_search(self, plan):
        """returns False to restart at the coordinates"""
        while True:
            self.out("\n--- Search ---")
            try:
                rows = self.A.query_heasarc(plan.ra, plan.dec, plan.radius_arcmin)
            except A.ArchiveError as e:
                self.out(f"  [!] {e}")
                c = self.ask("  1 new coordinates / 2 back to menu: ", lambda s: _choice(s, "12"))
                if c == "2":
                    raise Back("menu")
                return False
            kept, dropped = self.A.keep_archived_public(rows)
            self.out(f"  {len(rows)} observations found, {dropped} not archived or not public dropped, {len(kept)} kept")
            if not kept:
                self.out("  No observations in this radius. Enter new coordinates.")
                return False
            ocat, ok = {}, True
            ids = sorted({A.norm_obsid(r["obsid"]) for r in kept})
            while True:
                try:
                    ocat = self.A.fetch_ocat(ids)
                    ok = True
                    break
                except A.ArchiveError:
                    self.out(f"  {OCAT_WARNING}")
                    c = self.ask("  1 try Ocat again / 2 continue without it / 3 back to menu: ", lambda s: _choice(s, "123"))
                    if c == "1":
                        continue
                    if c == "3":
                        raise Back("menu")
                    ok = False
                    break
            plan.ocat_ok = ok
            all_obs = self.A.make_observations(kept, ocat, ocat_ok=ok)
            start_x, end_x = self.A.boundary_observations(all_obs, plan.date_from, plan.date_to)
            inside = [o for o in all_obs if self.A.in_date_range(o, plan.date_from, plan.date_to)]
            if start_x or end_x:
                self.out("\n--- Boundary observations ---")
                if start_x and self.yes(f"Crossing the start date: {', '.join(start_x)}. Keep them? (y/n): "):
                    inside += [o for o in all_obs if o.obsid in start_x and o not in inside]
                if end_x and not self.yes(f"Crossing the end date: {', '.join(end_x)}. Keep them? (y/n): "):
                    inside = [o for o in inside if o.obsid not in end_x]
            if not inside:
                self.out("  No observation in this date range. Choose other dates.")
                self.step_dates(plan)
                continue
            self.out(f"  {len(inside)} observations left")
            plan.observations = {o.obsid: o for o in sorted(inside, key=lambda o: o.start)}
            plan.categories = sorted({o.category for o in inside}, key=lambda c: c.sort_key())
            return True

    # ---------- tables ----------
    def category_table(self, plan, rows=None):
        rows = rows or list(range(1, len(plan.categories) + 1))
        lines = ["#  detector  grating  mode  window  count"]
        for r in rows:
            c = plan.categories[r - 1]
            n = sum(1 for o in plan.observations.values() if o.category == c)
            lines.append(f"{r:<2} {c.detector:<9} {c.grating:<8} {c.mode:<5} {c.window:<6} {n:>5}")
        self.out("\n".join(lines))

    def obs_table(self, plan):
        lines = ["obsid   detector  grating  mode  window  stage"]
        for o in plan.observations.values():
            c = o.category
            lines.append(f"{o.obsid:<7} {c.detector:<9} {c.grating:<8} {c.mode:<5} {c.window:<7} {o.stage}")
        self.out("\n".join(lines))

    def step_categories(self, plan):
        self.out("\n--- Categories ---")
        self.category_table(plan)
        self.out(STAGE_HELP)
        while True:
            for r, c in enumerate(plan.categories, 1):
                if c.is_cc:
                    st = self.ask(f"Stage for row {r} (none/download/repro): ", lambda s: parse_stage(s, allow_flux=False))
                else:
                    st = self.ask(f"Stage for row {r} (none/download/repro/flux): ", parse_stage)
                for o in plan.observations.values():
                    if o.category == c:
                        o.stage = st
            if any(o.stage != "none" for o in plan.observations.values()):
                break
            self.out("  Every row is none, so there is nothing to do. Choose the stages again.")
        self.out("\n--- Observations ---")
        self.obs_table(plan)
        ids = set(plan.observations)
        while True:
            ans = self.ask("Edit (e.g. 105 flux, 103 104 none, done): ", lambda s: parse_stage_edit(s, ids))
            if ans == "done":
                if all(o.stage == "none" for o in plan.observations.values()):
                    self.out("  Every observation is none; set at least one stage.")
                    continue
                break
            bad = [o for o, st in ans if st == "flux" and plan.observations[o].category.is_cc]
            if bad:
                self.out(f"  [!] {', '.join(bad)} is CC mode; CC has no flux stage")
                continue
            for o, st in ans:
                plan.observations[o].stage = st
            self.out("  " + ", ".join(f"{o} {st}" for o, st in ans))

    # ---------- flux settings ----------
    def flux_rows(self, plan):
        return [r for r, c in enumerate(plan.categories, 1)
                if any(o.category == c and o.stage == "flux" for o in plan.observations.values())]

    def step_flux(self, plan):
        self.out("\n--- Flux settings ---")
        rows = self.flux_rows(plan)
        cats = [plan.categories[r - 1] for r in rows]
        if any(c.is_acis for c in cats):
            plan.acis_bands = self.ask_bands("acis", "ACIS band (broad, ultrasoft, soft, medium, hard, csc, or lo:hi:eff in keV): ")
        if any(c.is_hrc for c in cats):
            plan.hrc_bands = self.ask_bands("hrc", "HRC band (wide, pilo:pihi:energy with PI channels and keV, or ::energy): ")
        acis_rows = [r for r in rows if plan.categories[r - 1].is_acis]
        if acis_rows:
            self.read_chip_layout(plan)
            self.out("\nChips for the image\nThe whole observation is still downloaded; this only selects chips for the image.")
            for i, (m, d) in enumerate(CHIP_MENU, 1):
                self.out(f"  {i}. {m:<10} {d}")
            for r in acis_rows:
                self.ask_chips(plan, r)
        for r in rows:
            c = plan.categories[r - 1]
            if c.is_grating and c.mode in ("TE", "?") and c.is_acis:
                self.ask_arms(plan, r)
        self.out("")
        plan.window_days = self.ask("Time window (days, 0 = no grouping): ", parse_window)
        if plan.window_days == 0:
            self.out(WINDOW0_WARNING)

    def ask_bands(self, kind, prompt):
        while True:
            text = self.ask(prompt)
            try:
                bands, notes = parse_bands(text, kind)
            except BandError as e:
                self.out(f"  [!] {e}")
                continue
            for n in notes:
                self.out("  " + n)
            for b in bands:
                self.out(f"  {b.describe()}")
            return bands

    def read_chip_layout(self, plan):
        todo = [o for o in plan.observations.values() if o.stage == "flux" and o.category.is_acis]
        dest = {o.obsid: os.path.join(plan.root, "observations", o.folder, "primary") for o in todo}
        self.out(f"\nReading chip layout (fov1, {len(todo)} observations) ...")
        failed = apply_chip_layout(plan, todo, self.A.fetch_fovs(dest))
        self.out("  ok" + (f" ({failed} could not be read; their chips show as unknown)" if failed else ""))

    def chips_screen(self, plan, r):
        c = plan.categories[r - 1]
        obs = [o for o in plan.observations.values() if o.category == c and o.stage == "flux"]
        self.out(f"\nRow {r} ({c.detector}, {c.grating}, {c.mode}), {len(obs)} observations")
        show_point = any(o.point_chips for o in obs)
        self.out("  obsid  target                 chips on             target on" +
                 ("                 search point on" if show_point else ""))
        for o in obs:
            on = " ".join(CHIP_NAMES[x] for x in o.active_chips) if o.active_chips else "unknown"
            if o.source_chips is None:
                src = "unknown"
            elif not o.source_chips:
                src = "none (chip gap or off the chips)"
            else:
                src = " / ".join(CHIP_NAMES[x] for x in o.source_chips) + (" (near the chip edge)" if o.source_near_edge else "")
            line = f"  {o.obsid:<6} {o.target_name[:22]:<22} {on:<20} {src:<24}"
            if show_point:
                line += " " + (" / ".join(CHIP_NAMES[x] for x in o.point_chips) if o.point_chips else "-")
            self.out(line.rstrip())
        return obs

    def ask_chips(self, plan, r):
        while True:
            obs = self.chips_screen(plan, r)
            mode = self.ask(f"Chips for row {r}: ", parse_chip_menu)
            if mode == "custom":
                choice = self.ask("  Chip names: ", lambda s: chip_choice("custom", s))
            else:
                choice = chip_choice(mode)
            warns = chip_warnings(obs, choice)
            if not warns:
                plan.chips_by_row[r] = choice
                return
            for w in warns:
                self.out(f"  [!] WARNING: {w}")
            if self.yes("Continue with these chips? (y/n): "):
                plan.chips_by_row[r] = choice
                return

    def ask_arms(self, plan, r):
        while True:
            self.out(f"\nGrating arms for row {r}:")
            for i, (t, _) in enumerate(ARM_OPTIONS, 1):
                self.out(f"  {i}. {t}")
            m = self.ask("> ", parse_arm_method)
            self.out(f"  [!] WARNING: {ARM_OPTIONS[m - 1][1]}")
            sub = subarray_arm_warning(plan, r, m)
            if sub:
                self.out(f"  [!] WARNING: {sub}")
            if self.yes("Continue with this option? (y/n): "):
                plan.arm_method_by_row[r] = m
                return

    # ---------- groups ----------
    def step_groups(self, plan):
        rows = self.flux_rows(plan)
        while True:
            if len(rows) > 1:
                self.out("\n--- Rows going to flux ---")
                self.category_table(plan, rows)
                self.out("Which table rows should be grouped together? Write every row once.\n"
                         "  separate   each row alone\n  1+3, 2     rows 1 and 3 together, row 2 alone\n"
                         "  1+2, rest  rows 1 and 2 together, every other row alone")
                sets = self.ask("> ", lambda s: parse_row_sets(s, rows))
            else:
                sets = [rows]
            rej, warn = row_set_checks(plan, sets)
            if rej:
                for t in rej:
                    self.out(f"  [!] {t}")
                continue
            if len(rows) > 1:
                self.out(f"  understood: {format_row_sets(sets)}")
                for t in warn:
                    self.out(f"  [!] WARNING: {t}")
                if not self.yes("Continue? (y/n): "):
                    continue
            break
        plan.row_sets = [RowSet(chr(65 + i), s) for i, s in enumerate(sets)]
        self.ask_streak(plan)
        plan.groups = build_groups(plan)
        self.edit_groups(plan)

    def ask_streak(self, plan):
        acis_sets = [rs for rs in plan.row_sets if any(plan.categories[r - 1].is_acis for r in rs.rows)]
        if not acis_sets:
            return
        self.out("\nSets: " + ", ".join(f"{rs.letter} = row{'s' if len(rs.rows) > 1 else ''} {'+'.join(map(str, rs.rows))}" for rs in plan.row_sets))
        letters = [rs.letter for rs in acis_sets]
        chosen = self.ask("Which sets should be corrected for a readout streak?\n(blank = none, all, or letters e.g. A): ",
                          lambda s: parse_streak_sets(s, letters))
        if chosen:
            self.out(STREAK_HELP)
        for L in chosen:
            plan.row_set(L).streak_action = self.ask(f"Action for set {L}: ", parse_streak_action)

    def groups_screen(self, plan):
        self.category_table(plan, self.flux_rows(plan))
        self.out("--- Groups (approximate times) ---")
        self.out("group     observations (table row)       span (days)  chips")
        for g in plan.groups:
            members = ", ".join(f"{o}({plan.row_of(o)})" for o in g.obsids) or "(empty)"
            is_hrc = g.obsids and all(plan.observations[o].category.is_hrc for o in g.obsids)
            chips = "-" if is_hrc else " / ".join(cs.short() for cs in plan.chip_sets_for(g))
            sp = f"{span_days(plan, g):.1f}" if g.obsids else "-"
            self.out(f"{g.group_id:<9} {members:<30} {sp:<12} {chips}" + ("  (manual)" if g.manual else ""))
        on, off = no_group_split(plan)
        if on:
            self.out(f"  in no group (not used, already on disk): {', '.join(on)}")
        if off:
            self.out(f"  in no group (not downloaded): {', '.join(off)}")

    def edit_groups(self, plan):
        while True:
            self.out("")
            self.groups_screen(plan)
            while True:
                cmd = self.ask("Edit (e.g. g01 chips I3, move 102 s, add 102 g02, new 101 201, drop 103 g01, done): ",
                               parse_group_edit)
                if cmd[0] == "done":
                    try:
                        for n in finish_edits(plan):
                            self.out(f"  {n}")
                    except InputError as e:
                        self.out(f"  [!] {e}")
                        continue
                    break
                try:
                    self.out("  " + apply_edit(plan, cmd))
                except InputError as e:
                    self.out(f"  [!] {e}")
                    continue
                self.groups_screen(plan)
            # step 10: plan check; rejects and declined warnings return to the edit
            res = check_plan(plan)
            self.out("\n--- Plan check ---")
            if not res.issues:
                self.out("  no problems found")
                return
            if res.rejects:
                for i in res.rejects:
                    self.out(f"  [!] REJECTED: {i.text}")
                self.out("  Edit the groups to continue.")
                continue
            ok = True
            if res.warnings:
                for t in collapse_issues(res.warnings):
                    self.out(f"  [!] WARNING: {t}")
                ok = self.yes("Accept these warnings? (y/n): ")
            if ok and res.sure:
                for i in res.sure:
                    self.out(f"  {i.text}")
                ok = self.yes("Are you sure? (y/n): ")
            if ok:
                return

    # ---------- confirmation and run ----------
    def step_confirm(self, plan):
        self.out("\n--- Confirmation ---")
        obs_root = os.path.join(plan.root, "observations")
        in_groups = {o for g in plan.groups for o in g.obsids}
        wanted = [o for o in plan.observations.values()
                  if o.stage in ("download", "repro") or (o.stage == "flux" and o.obsid in in_groups)]
        reused = [o.obsid for o in wanted if os.path.exists(os.path.join(obs_root, o.folder, "repro", REPRO_MARKER))]
        todo = [o for o in wanted if o.obsid not in reused]
        if reused:
            self.out(f"  Reused (already in observations/): {', '.join(reused)}")
        evt2 = {}
        sizes, exact = self.A.download_sizes(todo, evt_sizes=evt2) if todo else ({}, True)
        total = sum(sizes.values())
        approx = "" if exact else " (approximate: the archive size lookup did not finish)"
        self.out(f"  Download: {len(todo)} observations, {A.human(total)}{approx}")
        obs_disk = total * A.FOOTPRINT_FACTOR
        run_disk = A.run_footprint(plan, A.repro_evt_bytes(wanted, plan.root, sizes, evt2))
        try:
            free = shutil.disk_usage(plan.root).free
            self.out(f"  Disk after processing, at most: {A.human(obs_disk + run_disk)} "
                     f"(observations {A.human(obs_disk)}, images and event files {A.human(run_disk)}), free disk: {A.human(free)}")
            if obs_disk + run_disk > free:
                self.out("  [!] WARNING: the free disk looks smaller than the estimated footprint")
        except OSError:
            pass
        dirs = []
        for g in plan.groups:
            for b in plan.bands_for(g):
                d = os.path.join(plan.target, f"merged_final_image_{b.folder_label}_{plan.window_tag()}")
                if d not in dirs:
                    dirs.append(d)
        for d in dirs:
            self.out(f"  Output: {d}/")
        existing = [d for d in dirs if os.path.exists(os.path.join(plan.root, d))]
        if existing:
            self.out(f"  [!] {', '.join(existing)} already exist{'s' if len(existing) == 1 else ''}.")
            c = self.ask("  1 overwrite / 2 abort: ", lambda s: _choice(s, "12"))
            if c == "2":
                self.out("  Nothing was downloaded.")
                return False
            plan.overwrite = True
        if not self.yes("Start? (y/n): "):
            self.out("  Nothing was downloaded.")
            return False
        return True

    def step_run(self, plan):
        self.out("\n[!] NOTICE: setup is complete. Processing now runs automatically.")
        unknown = [o for o in plan.observations.values() if o.category.mode == "?" or o.category.window == "?"]
        if not plan.ocat_ok or unknown:
            self.out("Ocat could not be reached for some observations; if their headers do not fit the groups you made, "
                     "you will be asked to edit those groups after the downloads.")
        run_fn = self.run_fn
        if run_fn is None:
            from .run import run as run_fn
        try:
            rep = run_fn(plan, on_header_problems=self.header_problems, say=self.out, **self.run_kw())
        finally:
            self.status.clear()
        self.print_report(plan, rep)
        self.ask_cc_deletes(plan, rep)
        while rep.manual_problems and not rep.stopped:
            self.out("\nThese manual groups were not merged:")
            for hp in rep.manual_problems:
                self.out(f"  {hp.group}: {hp.text}")
            c = self.ask("  1 edit those groups and run them again / 2 drop those observations and run again / 3 leave them: ",
                         lambda s: _choice(s, "123"))
            if c == "3":
                break
            from .run import group_state, groups_to_rerun
            before = group_state(plan)
            if c == "2":
                for hp in rep.manual_problems:
                    g = plan.group(hp.group)
                    if g and hp.obsid in g.obsids:
                        g.obsids.remove(hp.obsid)
            else:
                self.edit_groups(plan)
            # the problem groups plus every group the edit made or changed (test e: a new group was not run)
            names = groups_to_rerun(plan, before, {hp.group for hp in rep.manual_problems})
            plan.overwrite = True
            try:
                rep = run_fn(plan, on_header_problems=self.header_problems, say=self.out, only_groups=names, **self.run_kw())
            finally:
                self.status.clear()
            self.print_report(plan, rep)
            self.ask_cc_deletes(plan, rep)

        self.ask_arm_problems(plan, rep, run_fn)

    def ask_arm_problems(self, plan, rep, run_fn):
        """rule 3 in a manual group (session 007): an observation that cannot be mixed with the others stayed in the image untreated,
        because manual groups are never changed automatically. ask once per pass: merge the group again without it, or keep it"""
        from .plan import Group
        while getattr(rep, "arm_problems", None) and not rep.stopped:
            self.out("\nThese manual groups keep an observation that should not be mixed with the others (it stayed in, untreated):")
            for hp in rep.arm_problems:
                self.out(f"  {hp.group}: {hp.text}")
            c = self.ask("  1 merge those groups again without it (it gets an image of its own) / 2 keep it in the group untreated: ",
                         lambda s: _choice(s, "12"))
            if c == "2":
                return
            names = set()
            for hp in rep.arm_problems:
                g = plan.group(hp.group)
                if g is None or hp.obsid not in g.obsids:
                    continue
                g.obsids.remove(hp.obsid)
                if g.obsids:
                    names.add(g.group_id)
                else:
                    plan.groups.remove(g)
                if plan.group(f"s{hp.obsid}") is None:
                    plan.groups.append(Group(f"s{hp.obsid}", [hp.obsid], g.row_set, manual=False, chip_sets=g.chip_sets, left_out_of=g.group_id))
                names.add(f"s{hp.obsid}")
            if not names:
                return
            plan.overwrite = True
            try:
                rep = run_fn(plan, on_header_problems=self.header_problems, say=self.out, only_groups=names, **self.run_kw())
            finally:
                self.status.clear()
            self.print_report(plan, rep)
            self.ask_cc_deletes(plan, rep)

    def ask_cc_deletes(self, plan, rep):
        """cc observations nobody asked for that this run downloaded: say what happened and ask before deleting anything"""
        from .run import cc_delete_question, delete_downloaded
        for d in getattr(rep, "cc_unwanted", []):
            if not d.get("deletable"):
                continue
            self.out("\n" + cc_delete_question(plan, d))
            if self.yes("Delete them? (y/n): "):
                self.out("  deleted" if delete_downloaded(plan, d) else "  not deleted (the folder is gone or not safe to remove)")
            else:
                self.out("  kept")

    def header_problems(self, plan, problems):
        self.out("\n--- Header check ---")
        for hp in problems:
            self.out(f"  [!] WARNING: {hp.text}")
        self.category_table(plan)
        self.edit_groups(plan)
        if not self.yes("Continue? (y/n): "):
            return False
        self.out("[!] NOTICE: setup is complete again. Processing now runs automatically.")
        return True

    def print_report(self, plan, rep):
        self.out("\n--- Report ---")
        if rep.stopped:
            self.out("  stopped after the downloads; nothing was merged")
            return
        kept = f" ({rep.kept_images} kept from the earlier pass)" if rep.kept_images else ""
        self.out(f"  images: {len(rep.images)}{kept}, failed: {len(rep.failed_groups)}")
        for k, v in rep.failed_groups.items():
            self.out(f"    {k}: {v}")
        for o, why in rep.failed_obs.items():
            self.out(f"  observation {o} skipped: {why}")
        for act in ("acisreadcorr", "fill", "mask"):
            got = [o for o, a in rep.streak.items() if a == act]
            if got:
                what = {"fill": "filled", "mask": "masked", "acisreadcorr": "corrected by acisreadcorr"}[act]
                self.out(f"  streak {what}: {', '.join(got)}")
        for line in streak_search_lines(rep.streak_search):
            self.out(line)
        arms = sorted(rep.arms)
        if arms:
            self.out(f"  arms treated: {', '.join(arms)}")
        left = getattr(rep, "left_out", {})
        if left:
            self.out("  left their group and got an image of their own (they must not be mixed with others, see the notes): "
                     + ", ".join(f"{o} (from {g})" for o, g in sorted(left.items())))
        for n in rep.notes:
            self.out(f"  note: {n}")
        for rd in rep.run_dirs:
            self.out(f"  Tables in {os.path.relpath(rd, plan.root)}/:")
            self.out("    final_images.csv      final images of all groups")
            self.out("    member_images.csv     each observation's own image inside multi-observation groups")
        if rep.run_dirs:
            self.out("  Index of all runs in this folder:\n    chanmerge_index_final.csv\n    chanmerge_index_members.csv")


def streak_search_lines(search):
    """report lines: per band, each observation's searched chips and the chips with a streak"""
    out, bands = [], []
    for (_, b) in search:
        if b not in bands:
            bands.append(b)
    for b in bands:
        out.append(f"  streak search ({b}):")
        for (o, bb), (searched, found) in sorted(search.items()):
            if bb != b:
                continue
            names = " ".join(CHIP_NAMES[c] for c in searched) or "no chips with events"
            hit = " ".join(CHIP_NAMES[c] for c in found) if found else "none"
            out.append(f"    {o}: searched {names}; streak found on: {hit}")
    return out


def collapse_issues(issues):
    """the same warning for many groups becomes one line that lists the groups"""
    order, by_text = [], {}
    for i in issues:
        text = i.text
        if i.group and text.startswith(i.group + ": "):
            text = text[len(i.group) + 2:]
        else:
            order.append((None, i.text))
            continue
        if text not in by_text:
            by_text[text] = []
            order.append((text, None))
        by_text[text].append(i.group)
    out = []
    for text, raw in order:
        if raw is not None:
            out.append(raw)
        else:
            gs = by_text[text]
            out.append(f"{', '.join(gs)}: {text}" if len(gs) <= 3 else f"{len(gs)} groups ({', '.join(gs)}): {text}")
    return out


def _choice(s, allowed):
    t = s.strip()
    if t not in allowed:
        raise InputError("choose " + " or ".join(allowed))
    return t


def apply_chip_layout(plan, todo, got):
    """fov1 per observation -> active chips, the chips under that observation's own target, and the chips under
    the search point (only meaningful when the point lies on that field). returns the number not read."""
    failed = 0
    for o in todo:
        path = got.get(o.obsid)
        if not path:
            failed += 1
            continue
        try:
            if o.target_ra is not None:
                on, src, near = fov_lookup(path, o.target_ra, o.target_dec)
                o.source_chips, o.source_near_edge = tuple(src), near
            _, pt, _ = fov_lookup(path, plan.ra, plan.dec)
            if o.target_ra is None:
                on = fov_lookup(path, plan.ra, plan.dec)[0]
        except Exception:
            failed += 1
            continue
        o.fov_file = path
        o.active_chips = tuple(on) or o.active_chips
        o.point_chips = tuple(pt)
    return failed


def chip_warnings(obs, choice):
    """known before download (fov1 / ocat): the target's chip left out, chosen chips that are off.
    the target is each observation's own target, never the search point."""
    out = []
    sets = choice.sets
    if any(s.chips is None for s in sets):
        return out
    union = set().union(*(set(s.chips) for s in sets))
    for o in obs:
        if o.active_chips:
            for s in sets:
                if not set(s.chips) & set(o.active_chips):
                    out.append(f"{o.obsid}: chip set {s.names()} has no active chip in this observation (on: "
                               f"{' '.join(CHIP_NAMES[c] for c in o.active_chips)})")
        if o.source_chips:
            missing = [c for c in o.source_chips if c not in union]
            if missing and len(missing) == len(o.source_chips):
                out.append(f"{o.obsid}: its target is on {'/'.join(CHIP_NAMES[c] for c in o.source_chips)}, which is in no chip set")
            elif missing and o.source_near_edge:
                out.append(f"{o.obsid}: its target is near the {'/'.join(CHIP_NAMES[c] for c in o.source_chips)} edge; "
                           f"{'/'.join(CHIP_NAMES[c] for c in missing)} is in no chip set")
        elif o.source_chips == ():
            out.append(f"{o.obsid}: its target falls on no chip (chip gap or off the chips)")
    return out


def row_set_checks(plan, sets):
    """CIAO-impossible sets are rejected; doubtful ones warn (asked once with Continue)"""
    rej, warn = [], []
    for s in sets:
        cats = [plan.categories[r - 1] for r in s]
        name = "+".join(map(str, s))
        dets = {c.detector for c in cats}
        if any(c.is_acis for c in cats) and any(c.is_hrc for c in cats):
            rej.append(f"{name}: ACIS and HRC cannot be merged")
        if {"HRC-I", "HRC-S"} <= dets:
            rej.append(f"{name}: HRC-I and HRC-S cannot be merged")
        if any(c.is_cc for c in cats):
            rej.append(f"{name}: CC observations cannot be merged")
        if {"ACIS-I", "ACIS-S"} <= dets:
            warn.append(f"{name}: ACIS-I and ACIS-S together; flux differs by chip type and spectrum, a false brightness step may appear at the boundary")
        if len({c.grating for c in cats}) > 1:
            warn.append(f"{name}: grating and non-grating rows together")
        if {"full", "sub"} <= {c.window for c in cats}:
            warn.append(f"{name}: full-frame and subarray rows together")
        # one chip set = one image: a mix only counts inside one set ('separate' puts bi and fi in different images;
        # test d found the old union over all sets warning about it)
        mixed = False
        for r in s:
            ch = plan.chips_by_row.get(r)
            if ch:
                for cs in ch.sets:
                    if cs.chips is not None and set(cs.chips) & set(BI_CHIPS) and set(cs.chips) & set(FI_CHIPS):
                        mixed = True
        if mixed:
            warn.append(f"{name}: a chip choice mixes back- and front-illuminated chips in one image")
    return rej, warn


def main():
    try:
        Terminal().main()
    except KeyboardInterrupt:
        print("\nstopped")
        sys.exit(1)


if __name__ == "__main__":
    main()
