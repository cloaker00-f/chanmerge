"""A small local window (tkinter) that fills the same RunPlan as the terminal and calls the same run(plan).

open with:  python -m chanmerge.gui   (from the folder chanmerge should run in)
network and processing run in a background thread; the window only talks to them through a queue.
"""
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from . import archive as A
from .bands import parse_bands, BandError
from .check import check_plan
from .cli import (Terminal, chip_warnings, apply_chip_layout, row_set_checks, collapse_issues, subarray_arm_warning, ARM_OPTIONS, WINDOW0_WARNING, OCAT_WARNING,
                  STREAK_HELP)
from .grouping import build_groups, apply_edit, finish_edits
from .inputs import (InputError, parse_ra, parse_dec, parse_radius, default_target, parse_date, check_date_order,
                     check_folder_name, check_run_path, parse_window, parse_row_sets, chip_choice, parse_chip_menu,
                     parse_streak_sets, parse_streak_action, parse_group_edit, format_row_sets, CHIP_MENU)
from .plan import RunPlan, RowSet, STAGES
from .ciao import REPRO_MARKER


class App:
    def __init__(self, root, archive=A, run_fn=None, cwd=None, ask=None):
        self.root, self.A, self.run_fn = root, archive, run_fn
        self.cwd = cwd or os.getcwd()
        self.plan = None
        self.q = queue.Queue()
        self.ok_checked = False
        # dialogs go through ask so tests can answer them
        self.ask = ask or _Dialogs(root)
        root.title("chanmerge 2.0")
        # two main tabs: the chanmerge steps (the notebook below, as before) and the image viewer
        self.top = ttk.Notebook(root)
        self.top.pack(fill="both", expand=True)
        self.run_page = ttk.Frame(self.top)
        self.top.add(self.run_page, text="chanmerge")
        self.nb = ttk.Notebook(self.run_page)
        self.nb.pack(fill="both", expand=True)
        self.viewer = None
        self.viewer_page = ttk.Frame(self.top)
        self.top.add(self.viewer_page, text="Viewer")
        self.top.bind("<<NotebookTabChanged>>", self.on_top_tab)
        self.tabs = {}
        for name in ("Target", "Categories", "Flux", "Groups", "Run"):
            f = ttk.Frame(self.nb, padding=8)
            self.nb.add(f, text=name)
            self.tabs[name] = f
        self.build_target()
        self.build_categories()
        self.build_flux()
        self.build_groups()
        self.build_run()
        for name in ("Categories", "Flux", "Groups", "Run"):
            self.nb.tab(self.tabs[name], state="disabled")
        root.after(100, self.poll)

    def on_top_tab(self, _e=None):
        """the viewer is built the first time its tab opens (matplotlib loads then, not at start), and re-reads the folder each time"""
        if self.top.select() != str(self.viewer_page):
            return
        if self.viewer is None:
            from .viewer import Viewer
            self.viewer = Viewer(self.viewer_page, cwd=self.cwd)
            self.viewer.pack(fill="both", expand=True)
        else:
            self.viewer.refresh()

    # ---------- background work ----------
    def bg(self, fn, done):
        def work():
            try:
                self.q.put(("done", done, fn(), None))
            except Exception as e:  # shown in the window, never a crash
                self.q.put(("done", done, None, e))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "done":
                    _, done, res, err = item
                    done(res, err)
                elif item[0] == "log":
                    self.log(item[1])
                elif item[0] == "call":
                    item[1]()
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def text_of(self, method, *args):
        buf = []
        t = Terminal(out=lambda s="": buf.append(str(s)), archive=self.A, cwd=self.cwd)
        getattr(t, method)(*args)
        return "\n".join(buf)

    def show(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    def goto(self, name):
        self.nb.tab(self.tabs[name], state="normal")
        self.nb.select(self.tabs[name])

    # ---------- target ----------
    def build_target(self):
        f = self.tabs["Target"]
        self.v = {k: tk.StringVar() for k in ("ra", "dec", "radius", "name", "d1", "d2")}
        rows = [("RA (hh:mm:ss or decimal degrees)", "ra"), ("Dec (dd:mm:ss or decimal degrees)", "dec"),
                ("Search radius (arcmin)", "radius"), ("Output folder name (blank = from coordinates)", "name"),
                ("Start date (YYYY-MM-DD, UTC, blank = no limit)", "d1"), ("End date (YYYY-MM-DD, UTC, blank = no limit)", "d2")]
        for i, (lab, k) in enumerate(rows):
            ttk.Label(f, text=lab).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(f, textvariable=self.v[k], width=30).grid(row=i, column=1, sticky="w")
        self.target_msg = ttk.Label(f, text="", foreground="#a33")
        self.target_msg.grid(row=len(rows), column=0, columnspan=2, sticky="w", pady=4)
        self.search_btn = ttk.Button(f, text="Search", command=self.on_search)
        self.search_btn.grid(row=len(rows) + 1, column=0, sticky="w")

    def on_search(self):
        level, msg = check_run_path(self.cwd)
        if level == "error":
            self.ask.error(msg)
            return
        try:
            ra, dec = parse_ra(self.v["ra"].get()), parse_dec(self.v["dec"].get())
            radius = parse_radius(self.v["radius"].get())
            d1, d2 = parse_date(self.v["d1"].get()), parse_date(self.v["d2"].get())
            check_date_order(d1, d2)
        except InputError as e:
            self.target_msg.configure(text=str(e))
            return
        name = self.v["name"].get().strip() or default_target(ra, dec)
        level, msg = check_folder_name(name)
        if level == "error":
            self.target_msg.configure(text=msg.replace("\n", " "))
            return
        if level == "warn" and not self.ask.yesno("Folder name", msg + "\nUse it anyway?"):
            return
        self.v["name"].set(name)
        self.target_msg.configure(text="searching ...")
        self.plan = RunPlan(ra=ra, dec=dec, radius_arcmin=radius, target=name, root=self.cwd, date_from=d1, date_to=d2)
        self.search_btn.state(["disabled"])

        def search():
            rows = self.A.query_heasarc(ra, dec, radius)
            kept, dropped = self.A.keep_archived_public(rows)
            ids = sorted({A.norm_obsid(r["obsid"]) for r in kept})
            try:
                ocat, ok = (self.A.fetch_ocat(ids), True) if ids else ({}, True)
            except A.ArchiveError:
                ocat, ok = {}, False
            return rows, kept, dropped, ocat, ok
        self.bg(search, self.after_search)

    def after_search(self, res, err):
        self.search_btn.state(["!disabled"])
        if err:
            self.target_msg.configure(text=str(err))
            return
        rows, kept, dropped, ocat, ok = res
        if not kept:
            self.target_msg.configure(text=f"{len(rows)} found, {dropped} dropped: no observation in this radius")
            return
        if not ok:
            c = self.ask.yesnocancel("Ocat", OCAT_WARNING + "\n\nYes: try Ocat again   No: continue without it   Cancel: stop")
            if c is None:
                self.target_msg.configure(text="stopped")
                return
            if c:
                self.on_search()
                return
        p = self.plan
        p.ocat_ok = ok
        obs = self.A.make_observations(kept, ocat, ocat_ok=ok)
        sx, ex = self.A.boundary_observations(obs, p.date_from, p.date_to)
        inside = [o for o in obs if self.A.in_date_range(o, p.date_from, p.date_to)]
        if sx and self.ask.yesno("Boundary", f"Crossing the start date: {', '.join(sx)}. Keep them?"):
            inside += [o for o in obs if o.obsid in sx and o not in inside]
        if ex and not self.ask.yesno("Boundary", f"Crossing the end date: {', '.join(ex)}. Keep them?"):
            inside = [o for o in inside if o.obsid not in ex]
        if not inside:
            self.target_msg.configure(text="no observation in this date range")
            return
        p.observations = {o.obsid: o for o in sorted(inside, key=lambda o: o.start)}
        p.categories = sorted({o.category for o in inside}, key=lambda c: c.sort_key())
        self.target_msg.configure(text=f"{len(rows)} found, {dropped} not archived or not public dropped, {len(inside)} kept")
        self.fill_categories()
        self.goto("Categories")

    # ---------- categories ----------
    def build_categories(self):
        f = self.tabs["Categories"]
        ttk.Label(f, text="Each stage includes the ones before it: none, download, repro (download + reprocess), "
                         "flux (+ flux image). CC has no flux stage.").pack(anchor="w")
        self.cat_tree = ttk.Treeview(f, columns=("detector", "grating", "mode", "window", "count", "stage"), height=6)
        self.cat_tree.heading("#0", text="#")
        self.cat_tree.column("#0", width=40)
        for c in ("detector", "grating", "mode", "window", "count", "stage"):
            self.cat_tree.heading(c, text=c)
            self.cat_tree.column(c, width=90)
        self.cat_tree.pack(fill="x")
        bar = ttk.Frame(f); bar.pack(fill="x", pady=4)
        self.stage_var = tk.StringVar(value="flux")
        ttk.Combobox(bar, textvariable=self.stage_var, values=STAGES, width=10, state="readonly").pack(side="left")
        ttk.Button(bar, text="Set stage of selected rows", command=self.on_row_stage).pack(side="left", padx=4)
        self.obs_tree = ttk.Treeview(f, columns=("detector", "grating", "mode", "window", "stage"), height=10)
        self.obs_tree.heading("#0", text="obsid")
        for c in ("detector", "grating", "mode", "window", "stage"):
            self.obs_tree.heading(c, text=c)
            self.obs_tree.column(c, width=90)
        self.obs_tree.pack(fill="both", expand=True)
        bar2 = ttk.Frame(f); bar2.pack(fill="x", pady=4)
        ttk.Button(bar2, text="Set stage of selected observations", command=self.on_obs_stage).pack(side="left")
        self.cat_msg = ttk.Label(bar2, text="", foreground="#a33"); self.cat_msg.pack(side="left", padx=8)
        ttk.Button(bar2, text="Next", command=self.on_categories_next).pack(side="right")

    def fill_categories(self):
        p = self.plan
        self.cat_tree.delete(*self.cat_tree.get_children())
        for r, c in enumerate(p.categories, 1):
            obs = [o for o in p.observations.values() if o.category == c]
            stages = sorted({o.stage for o in obs})
            self.cat_tree.insert("", "end", iid=str(r), text=str(r),
                                 values=(c.detector, c.grating, c.mode, c.window, len(obs), "/".join(stages)))
        self.obs_tree.delete(*self.obs_tree.get_children())
        for o in p.observations.values():
            c = o.category
            self.obs_tree.insert("", "end", iid=o.obsid, text=o.obsid, values=(c.detector, c.grating, c.mode, c.window, o.stage))

    def set_stage(self, obsids, stage):
        self.cat_msg.configure(text="")      # an old message must not stay on screen (seen in test c)
        bad = [o for o in obsids if stage == "flux" and self.plan.observations[o].category.is_cc]
        if bad:
            self.cat_msg.configure(text=f"{', '.join(bad)} is CC mode; CC has no flux stage")
            obsids = [o for o in obsids if o not in bad]
        for o in obsids:
            self.plan.observations[o].stage = stage
        self.fill_categories()

    def on_row_stage(self):
        rows = [int(i) for i in self.cat_tree.selection()]
        cats = [self.plan.categories[r - 1] for r in rows]
        self.set_stage([o.obsid for o in self.plan.observations.values() if o.category in cats], self.stage_var.get())

    def on_obs_stage(self):
        self.set_stage(list(self.obs_tree.selection()), self.stage_var.get())

    def on_categories_next(self):
        p = self.plan
        if all(o.stage == "none" for o in p.observations.values()):
            self.cat_msg.configure(text="every observation is none; choose at least one stage")
            return
        self.cat_msg.configure(text="")
        if p.flux_obsids():
            self.fill_flux()
            self.goto("Flux")
        else:
            self.ask.info("Flux", "No flux image will be produced.")
            self.fill_run()
            self.goto("Run")

    # ---------- flux ----------
    def build_flux(self):
        f = self.tabs["Flux"]
        top = ttk.Frame(f); top.pack(fill="x")
        self.acis_band = tk.StringVar(); self.hrc_band = tk.StringVar(); self.window = tk.StringVar()
        ttk.Label(top, text="ACIS band (broad, ultrasoft, soft, medium, hard, csc, or lo:hi:eff keV)").grid(row=0, column=0, sticky="w")
        self.acis_entry = ttk.Entry(top, textvariable=self.acis_band, width=24); self.acis_entry.grid(row=0, column=1, sticky="w")
        ttk.Label(top, text="HRC band (wide, pilo:pihi:energy, ::energy)").grid(row=1, column=0, sticky="w")
        self.hrc_entry = ttk.Entry(top, textvariable=self.hrc_band, width=24); self.hrc_entry.grid(row=1, column=1, sticky="w")
        ttk.Label(top, text="Time window (days, 0 = no grouping)").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.window, width=10).grid(row=2, column=1, sticky="w")
        # one line per choice; the old comma split cut descriptions such as "(S1, S3)" (seen in test c)
        ttk.Label(f, text="Chips (the whole observation is still downloaded; this only selects chips for the image):\n" +
                  "\n".join(f"  {m:9s} {d}" for m, d in CHIP_MENU), justify="left").pack(anchor="w", pady=(6, 0))
        self.rows_frame = ttk.Frame(f); self.rows_frame.pack(fill="x")
        self.chip_text = tk.Text(f, height=12, width=100, state="disabled"); self.chip_text.pack(fill="both", expand=True, pady=4)
        bar = ttk.Frame(f); bar.pack(fill="x")
        self.flux_msg = ttk.Label(bar, text="", foreground="#a33"); self.flux_msg.pack(side="left")
        ttk.Button(bar, text="Next", command=self.on_flux_next).pack(side="right")

    def fill_flux(self):
        p = self.plan
        for w in self.rows_frame.winfo_children():
            w.destroy()
        t = Terminal(out=lambda s="": None, archive=self.A, cwd=self.cwd)
        rows = t.flux_rows(p)
        self.acis_entry.state(["!disabled"] if any(p.categories[r - 1].is_acis for r in rows) else ["disabled"])
        self.hrc_entry.state(["!disabled"] if any(p.categories[r - 1].is_hrc for r in rows) else ["disabled"])
        self.chip_vars, self.custom_vars, self.arm_vars = {}, {}, {}
        line = 0
        for r in rows:
            c = p.categories[r - 1]
            if c.is_acis:
                ttk.Label(self.rows_frame, text=f"Row {r} ({c.detector}, {c.grating}, {c.mode}, {c.window}) chips").grid(row=line, column=0, sticky="w")
                self.chip_vars[r] = tk.StringVar(); self.custom_vars[r] = tk.StringVar()
                ttk.Combobox(self.rows_frame, textvariable=self.chip_vars[r], values=[m for m, _ in CHIP_MENU], width=10,
                             state="readonly").grid(row=line, column=1)
                ttk.Entry(self.rows_frame, textvariable=self.custom_vars[r], width=18).grid(row=line, column=2, padx=4)
                line += 1
            if c.is_acis and c.is_grating and c.mode in ("TE", "?"):
                ttk.Label(self.rows_frame, text=f"Row {r} grating arms").grid(row=line, column=0, sticky="w")
                self.arm_vars[r] = tk.StringVar()
                ttk.Combobox(self.rows_frame, textvariable=self.arm_vars[r], values=[f"{i}. {t}" for i, (t, _) in enumerate(ARM_OPTIONS, 1)],
                             width=40, state="readonly").grid(row=line, column=1, columnspan=2, sticky="w")
                line += 1
        if any(p.categories[r - 1].is_acis for r in rows):
            self.show(self.chip_text, "reading chip layout (fov1) ...")

            def fetch():
                todo = [o for o in p.observations.values() if o.stage == "flux" and o.category.is_acis]
                got = self.A.fetch_fovs({o.obsid: os.path.join(p.root, "observations", o.folder, "primary") for o in todo})
                apply_chip_layout(p, todo, got)
                return rows
            self.bg(fetch, lambda res, err: self.show(self.chip_text, "\n".join(
                self.text_of("chips_screen", p, r) for r in rows if p.categories[r - 1].is_acis) if not err else str(err)))

    def on_flux_next(self):
        p = self.plan
        t = Terminal(out=lambda s="": None, archive=self.A, cwd=self.cwd)
        rows = t.flux_rows(p)
        try:
            if any(p.categories[r - 1].is_acis for r in rows):
                p.acis_bands, notes = parse_bands(self.acis_band.get(), "acis")
                if notes:
                    self.ask.info("Bands", "\n".join(notes))
            if any(p.categories[r - 1].is_hrc for r in rows):
                p.hrc_bands, notes = parse_bands(self.hrc_band.get(), "hrc")
            for r, var in self.chip_vars.items():
                mode = parse_chip_menu(var.get())
                choice = chip_choice(mode, self.custom_vars[r].get()) if mode == "custom" else chip_choice(mode)
                obs = [o for o in p.observations.values() if o.category == p.categories[r - 1] and o.stage == "flux"]
                warns = chip_warnings(obs, choice)
                if warns and not self.ask.yesno("Chips", "\n".join(warns) + "\n\nContinue with these chips?"):
                    return
                p.chips_by_row[r] = choice
            for r, var in self.arm_vars.items():
                if not var.get():
                    raise InputError(f"choose the grating arm treatment for row {r} (there is no default)")
                m = int(var.get()[0])
                sub = subarray_arm_warning(p, r, m)
                text = ARM_OPTIONS[m - 1][1] + ("\n\n" + " ".join(sub.split()) if sub else "")
                if not self.ask.yesno("Grating arms", text + "\n\nContinue with this option?"):
                    return
                p.arm_method_by_row[r] = m
            p.window_days = parse_window(self.window.get())
        except (InputError, BandError) as e:
            self.flux_msg.configure(text=str(e))
            return
        if p.window_days == 0:
            self.ask.info("Time window", WINDOW0_WARNING)
        self.flux_msg.configure(text="")
        self.fill_groups_tab()
        self.goto("Groups")

    # ---------- groups ----------
    def build_groups(self):
        f = self.tabs["Groups"]
        self.rowsets = tk.StringVar(value="separate"); self.streak_sets = tk.StringVar(); self.edit = tk.StringVar()
        top = ttk.Frame(f); top.pack(fill="x")
        ttk.Label(top, text="Rows merged together (separate / 1+3, 2 / 1+2, rest)").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.rowsets, width=20).grid(row=0, column=1, sticky="w")
        ttk.Label(top, text="Sets to correct for a readout streak (blank = none, all, or letters)").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.streak_sets, width=20).grid(row=1, column=1, sticky="w")
        ttk.Label(top, text="Streak action per set, e.g. A=fill, B=mask (acisreadcorr / fill / mask)").grid(row=2, column=0, sticky="w")
        self.streak_actions = tk.StringVar()
        ttk.Entry(top, textvariable=self.streak_actions, width=20).grid(row=2, column=1, sticky="w")
        ttk.Button(top, text="Build groups", command=self.on_build_groups).grid(row=3, column=0, sticky="w", pady=4)
        self.groups_text = tk.Text(f, height=18, width=100, state="disabled"); self.groups_text.pack(fill="both", expand=True)
        bar = ttk.Frame(f); bar.pack(fill="x", pady=4)
        ttk.Label(bar, text="Edit").pack(side="left")
        ttk.Entry(bar, textvariable=self.edit, width=40).pack(side="left", padx=4)
        ttk.Button(bar, text="Apply", command=self.on_edit).pack(side="left")
        ttk.Button(bar, text="Check plan", command=self.on_check).pack(side="right")
        self.groups_msg = ttk.Label(f, text="", foreground="#a33"); self.groups_msg.pack(anchor="w")

    def fill_groups_tab(self):
        p = self.plan
        t = Terminal(out=lambda s="": None, archive=self.A, cwd=self.cwd)
        self.flux_rows = t.flux_rows(p)
        self.show(self.groups_text, self.text_of("category_table", p, self.flux_rows) + "\n\n" + STREAK_HELP)

    def on_build_groups(self):
        p = self.plan
        try:
            sets = parse_row_sets(self.rowsets.get(), self.flux_rows) if len(self.flux_rows) > 1 else [self.flux_rows]
            rej, warn = row_set_checks(p, sets)
            if rej:
                raise InputError("; ".join(rej))
            if warn and not self.ask.yesno("Rows", f"understood: {format_row_sets(sets)}\n" + "\n".join(warn) + "\n\nContinue?"):
                return
            p.row_sets = [RowSet(chr(65 + i), s) for i, s in enumerate(sets)]
            acis = [rs.letter for rs in p.row_sets if any(p.categories[r - 1].is_acis for r in rs.rows)]
            chosen = parse_streak_sets(self.streak_sets.get(), acis) if acis else []
            actions = {}
            for part in [x for x in self.streak_actions.get().replace(";", ",").split(",") if x.strip()]:
                if "=" not in part:
                    raise InputError("write streak actions as A=fill, B=mask")
                k, v = part.split("=", 1)
                actions[k.strip().upper()] = parse_streak_action(v)
            for L in chosen:
                if L not in actions:
                    raise InputError(f"give the streak action for set {L} (acisreadcorr, fill or mask)")
                p.row_set(L).streak_action = actions[L]
        except InputError as e:
            self.groups_msg.configure(text=str(e))
            return
        p.groups = build_groups(p)
        self.ok_checked = False
        self.groups_msg.configure(text="")
        self.refresh_groups()

    def refresh_groups(self):
        sets = ", ".join(f"{rs.letter} = rows {'+'.join(map(str, rs.rows))} (streak: {rs.streak_action})" for rs in self.plan.row_sets)
        self.show(self.groups_text, "Sets: " + sets + "\n" + self.text_of("groups_screen", self.plan))

    def on_edit(self):
        try:
            msg = apply_edit(self.plan, parse_group_edit(self.edit.get()))
            self.groups_msg.configure(text=msg)
            self.edit.set("")
            self.ok_checked = False
        except InputError as e:
            self.groups_msg.configure(text=str(e))
        self.refresh_groups()

    def on_check(self):
        p = self.plan
        try:
            notes = finish_edits(p)
        except InputError as e:
            self.groups_msg.configure(text=str(e))
            return
        self.refresh_groups()
        res = check_plan(p)
        if res.rejects:
            self.ask.error("REJECTED:\n" + "\n".join(i.text for i in res.rejects) + "\n\nEdit the groups to continue.")
            return
        if res.warnings and not self.ask.yesno("Plan check", "\n".join(collapse_issues(res.warnings)) + "\n\nAccept these warnings?"):
            return
        if res.sure and not self.ask.yesno("Shared data", "\n".join(i.text for i in res.sure) + "\n\nAre you sure?"):
            return
        self.ok_checked = True
        self.groups_msg.configure(text="; ".join(notes) or "plan check passed")
        self.fill_run()
        self.goto("Run")

    # ---------- run ----------
    def build_run(self):
        f = self.tabs["Run"]
        bar = ttk.Frame(f); bar.pack(fill="x")
        ttk.Button(bar, text="Estimate download size", command=self.on_size).pack(side="left")
        self.overwrite = tk.BooleanVar(value=False)
        self.overwrite_box = ttk.Checkbutton(bar, text="overwrite the existing run folder", variable=self.overwrite)
        self.overwrite_box.pack(side="left", padx=8)
        self.start_btn = ttk.Button(bar, text="Start", command=self.on_start)
        self.start_btn.pack(side="right")
        self.cont_btn = ttk.Button(bar, text="Continue run", command=lambda: self._answer_header(True))
        self.stop_btn = ttk.Button(bar, text="Stop", command=lambda: self._answer_header(False))
        self.size_msg = ttk.Label(f, text=""); self.size_msg.pack(anchor="w", pady=4)
        # one line that stays visible for the whole run while a manual group waits on standby (no popup): a red bar under the log
        self.standby_label = tk.Label(f, text="", bg="#a33", fg="white", anchor="w", justify="left", padx=8, pady=4, wraplength=900)
        # the terminal's '1 edit / 2 drop / 3 leave' after a header problem in a manual group (hidden until needed)
        self.fix = None
        self.fix_frame = ttk.LabelFrame(f, text="Groups not merged (manual groups with a header problem)", padding=6)
        self.fix_text = ttk.Label(self.fix_frame, text="", justify="left", wraplength=850)
        self.fix_text.pack(anchor="w")
        fb = ttk.Frame(self.fix_frame); fb.pack(anchor="w", pady=3)
        ttk.Button(fb, text="1  Edit the groups, then run them again", command=self.fix_edit).pack(side="left")
        ttk.Button(fb, text="2  Drop those observations and run again", command=self.fix_drop).pack(side="left", padx=4)
        ttk.Button(fb, text="3  Leave them", command=self.fix_leave).pack(side="left")
        self.fix_go = ttk.Button(self.fix_frame, text="Run the edited groups now", command=self.fix_run)
        self.log_text = tk.Text(f, height=24, width=110, state="disabled"); self.log_text.pack(fill="both", expand=True)
        self._header_event = threading.Event(); self._header_ok = False

    def set_standby(self, text):
        """text = the line to keep on screen, None = take it away (called from the main thread)"""
        if text:
            self.standby_label.configure(text=text)
            self.standby_label.pack(side="bottom", fill="x", before=self.log_text)
        else:
            self.standby_label.pack_forget()

    def run_kw(self):
        """the real run reports the standby line; test doubles of run() know nothing about it. the run works in a thread, so it goes through the queue"""
        if self.run_fn is not None:
            return {}
        return {"status": lambda text: self.q.put(("call", lambda: self.set_standby(text)))}

    def log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def wanted(self):
        p = self.plan
        in_groups = {o for g in p.groups for o in g.obsids}
        return [o for o in p.observations.values() if o.stage in ("download", "repro") or (o.stage == "flux" and o.obsid in in_groups)]

    def run_dirs(self):
        p, dirs = self.plan, []
        for g in p.groups:
            for b in p.bands_for(g):
                d = os.path.join(p.target, f"merged_final_image_{b.folder_label}_{p.window_tag()}")
                if d not in dirs:
                    dirs.append(d)
        return dirs

    def fill_run(self):
        existing = [d for d in self.run_dirs() if os.path.exists(os.path.join(self.plan.root, d))]
        self.size_msg.configure(text=("Output: " + ", ".join(d + "/" for d in self.run_dirs())) +
                                (f"   [!] already exists: {', '.join(existing)}" if existing else ""))

    def to_download(self):
        obs_root = os.path.join(self.plan.root, "observations")
        return [o for o in self.wanted() if not os.path.exists(os.path.join(obs_root, o.folder, "repro", REPRO_MARKER))]

    def on_size(self, then=None):
        """size lookup in the background; then(text) is called with the result line (or None on an error)"""
        todo = self.to_download()
        self.size_msg.configure(text="looking up sizes ...")

        evt2 = {}

        def done(res, err):
            if err:
                self.size_msg.configure(text=str(err))
                if then:
                    then(None)
                return
            sizes, exact = res
            total = sum(sizes.values())
            import shutil
            free = shutil.disk_usage(self.plan.root).free
            disk = total * A.FOOTPRINT_FACTOR + A.run_footprint(self.plan, A.repro_evt_bytes(self.wanted(), self.plan.root, sizes, evt2))
            text = (f"Download: {len(todo)} observations, {A.human(total)}{'' if exact else ' (approximate)'}; "
                    f"disk after processing at most {A.human(disk)}, free {A.human(free)}")
            self.size_msg.configure(text=text)
            if then:
                then(text)
        self.bg(lambda: self.A.download_sizes(todo, evt_sizes=evt2) if todo else ({}, True), done)

    def on_start(self):
        p = self.plan
        if p.flux_obsids() and not self.ok_checked:
            self.ask.error("Run the plan check on the Groups tab first.")
            return
        existing = [d for d in self.run_dirs() if os.path.exists(os.path.join(p.root, d))]
        if existing and not self.overwrite.get():
            self.ask.error(f"{', '.join(existing)} already exists: tick 'overwrite' or change the folder name.")
            return
        p.overwrite = bool(existing)
        # the confirmation of the plan (step 11): sizes are always looked up and shown before anything starts,
        # as on the terminal's confirmation screen (test c: the window only showed them on a separate button)
        self.start_btn.state(["disabled"])
        self.on_size(then=self._confirm_start)

    def _confirm_start(self, size_text):
        todo = self.to_download()
        ask = (f"{len(todo)} observation{'s' if len(todo) != 1 else ''} will be downloaded now "
               f"({', '.join(o.obsid for o in todo)}); the others are reused." if todo else
               "Nothing needs downloading (every observation is reused).")
        ask += "\n" + (size_text or "size lookup failed (see the Run tab)") + "\n\nStart processing?"
        if not self.ask.yesno("Start", ask):
            self.start_btn.state(["!disabled"])
            return
        self._start_run()

    def _start_run(self, only_groups=None):
        p = self.plan
        self.start_btn.state(["disabled"])
        if not only_groups:
            self.log("[!] NOTICE: setup is complete. Processing now runs automatically.")
        run_fn = self.run_fn
        if run_fn is None:
            from .run import run as run_fn

        def work():
            kw = dict(only_groups=only_groups) if only_groups else {}
            return run_fn(p, on_header_problems=self._header_problems, say=lambda s: self.q.put(("log", str(s))), **kw, **self.run_kw())

        def done(rep, err):
            self.start_btn.state(["!disabled"])
            if err:
                self.log(f"[!] ERROR: {err}"); return
            self.after_run(p, rep)
        self.bg(work, done)

    def after_run(self, p, rep):
        self.log(self.text_of("print_report", p, rep))
        from .run import cc_delete_question, delete_downloaded
        for d in getattr(rep, "cc_unwanted", []):
            # a cc observation nobody asked for that this run downloaded: explain, ask, only then delete
            if d.get("deletable"):
                if self.ask.yesno("Delete downloaded data", cc_delete_question(p, d)):
                    self.log(f"deleted the data of {d['obsid']}" if delete_downloaded(p, d) else f"{d['obsid']}: not deleted (folder gone or not safe to remove)")
                else:
                    self.log(f"kept the data of {d['obsid']}")
        if rep.manual_problems:
            # same choices as the terminal (1 edit and run again / 2 drop and run again / 3 leave), as buttons on the Run tab;
            # only the problem groups and the groups the edit made or changed run again (test e)
            from .run import group_state
            self.fix = dict(problems=list(rep.manual_problems), before=group_state(p))
            self.fix_text.configure(text="\n".join(f"{h.group}: {h.text}" for h in rep.manual_problems) +
                                    "\nThe other groups are finished.")
            self.fix_go.pack_forget()
            self.fix_frame.pack(fill="x", pady=4, before=self.log_text)
            self.ask.info("Groups not merged", "\n".join(f"{h.group}: {h.text}" for h in rep.manual_problems) +
                          "\n\nThe other groups are finished. Choose below, on the Run tab: edit the groups and run them again, "
                          "drop those observations and run again, or leave them.")

    def fix_edit(self):
        if not self.fix:
            return
        self.fix_go.pack(anchor="w", pady=2)
        self.ok_checked = False
        self.groups_msg.configure(text="Edit the groups, press Check plan, then press 'Run the edited groups now' on the Run tab.")
        self.refresh_groups()
        self.goto("Groups")

    def fix_drop(self):
        if not self.fix:
            return
        for hp in self.fix["problems"]:
            g = self.plan.group(hp.group)
            if g and hp.obsid in g.obsids:
                g.obsids.remove(hp.obsid)
        self._fix_rerun()

    def fix_run(self):
        if not self.fix:
            return
        if not self.ok_checked:
            self.ask.error("Press Check plan on the Groups tab first.")
            return
        self._fix_rerun()

    def fix_leave(self):
        self.fix = None
        self.fix_frame.pack_forget()

    def _fix_rerun(self):
        from .run import groups_to_rerun
        fx, p = self.fix, self.plan
        names = groups_to_rerun(p, fx["before"], {h.group for h in fx["problems"]})
        p.overwrite = True
        self.fix_leave()
        self.log(f"--- Running again: {', '.join(sorted(names)) or 'nothing changed'} ---")
        if names:
            self._start_run(only_groups=names)

    def _header_problems(self, plan, problems):
        # called in the worker thread: ask the window, wait for the answer
        self._header_event.clear()

        def show():
            self.log("--- Header check ---\n" + "\n".join(f"[!] WARNING: {h.text}" for h in problems))
            self.ask.info("Header check", "\n".join(h.text for h in problems) +
                          "\n\nEdit the groups on the Groups tab if needed (Check plan), then press 'Continue run'.")
            self.cont_btn.pack(side="right", padx=4); self.stop_btn.pack(side="right")
        self.q.put(("call", show))
        self._header_event.wait()
        return self._header_ok

    def _answer_header(self, ok):
        self._header_ok = ok
        self.cont_btn.pack_forget(); self.stop_btn.pack_forget()
        if ok:
            self.log("[!] NOTICE: setup is complete again. Processing now runs automatically.")
        self._header_event.set()


class _Dialogs:
    def __init__(self, root):
        self.root = root

    def yesno(self, title, text):
        return messagebox.askyesno(title, text, parent=self.root)

    def yesnocancel(self, title, text):
        return messagebox.askyesnocancel(title, text, parent=self.root)

    def info(self, title, text):
        messagebox.showinfo(title, text, parent=self.root)

    def error(self, text):
        messagebox.showerror("chanmerge", text, parent=self.root)


def main():
    root = tk.Tk()
    root.geometry("980x700")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
