"""viewer tab: look at the flux images chanmerge made, a bit like ds9 (colour map, scale, limits, zoom, pixel readout,
smoothing, 1 to 4 images at once).

the pure helpers (list_images, limits_of, make_norm, load_fits_image, smooth_kernel, smooth_image) have no window in them,
so tests run without a display.
the window class is at the bottom. nothing here changes any file chanmerge wrote.
"""
import os
import warnings

import numpy as np

SCALES = ("linear", "log", "sqrt", "squared", "asinh")
LIMITS = ("minmax", "99.5%", "99%", "98%", "95%", "zscale", "user")
COLORMAPS = ("gray", "viridis", "inferno", "magma", "plasma", "cividis", "hot", "cubehelix", "jet", "turbo", "coolwarm")
NAN_COLORS = ("white", "black", "gray", "red", "blue", "green")
SHOWS = ("flux", "flux (original)", "exposure map", "mask")
SMOOTH_FUNCTIONS = ("boxcar", "tophat", "gaussian", "elliptical gaussian")      # ds9: analysis > smooth parameters
PANEL_CHOICES = ("1", "2", "3", "4 (2x2)")
# what ds9 8.7 starts with (its smooth.tcl, smoothdef): gaussian, radius 3, radius minor 3, sigma 1.5, sigma minor 1.5, angle 0
SMOOTH_DEFAULTS = dict(function="gaussian", radius=3, radius_minor=3, sigma=1.5, sigma_minor=1.5, angle=0.0)


# ---------- the list of images ----------
def list_images(root, include_members=False):
    """images found through the two index files in root: one dict per (final or member) row with the files that exist.
    keys: label, kind, target, band, window, run, group_dir, group_id, chips, obsid, exposure_ks, info, files {show name: absolute path}, search.
    run and group_dir are folders relative to root, taken from the flux image path (members sit in their group's folder)"""
    from .tables import load_index
    out = []
    parts = [("final", "chanmerge_index_final.csv")] + ([("member", "chanmerge_index_members.csv")] if include_members else [])
    for kind, name in parts:
        p = os.path.join(root, name)
        if not os.path.exists(p):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows = load_index(p)
        for r in rows:
            files = {}
            for show, col in (("flux", "flux_image_abs"), ("flux (original)", "flux_image_original_abs"),
                              ("exposure map", "expmap_file_abs"), ("mask", "mask_file_abs")):
                f = r.get(col, "")
                if f and os.path.exists(f) and f not in files.values():
                    files[show] = f
            if "flux" not in files:
                continue
            gid = r.get("group_id", "")
            chips = r.get("chips", "")
            obsid = r.get("obsid", "") if kind == "member" else ""
            rel = (r.get("flux_image", "") or "").replace("\\", "/")
            group_dir = os.path.dirname(rel)
            run = os.path.dirname(group_dir)
            label = f"{r.get('target', '')} | {gid}" + (f":{obsid}" if obsid else "")
            label += f" | {r.get('band', '')} | w{r.get('window_days', '')}d" + (f" | {chips}" if chips and chips != "all" else "")
            ks = r.get("total_exposure_ks") or r.get("exposure_ks") or ""
            bits = [f"obsids {r.get('obsids') or r.get('obsid', '')}", f"{ks} ks" if ks else "", f"streak {r.get('streak_action', '')}",
                    f"arm {r.get('arm_method', '') or '-'}"]
            search = " ".join([r.get("target", ""), r.get("band", ""), f"w{r.get('window_days', '')}d", gid, chips, r.get("obsids", ""), obsid,
                               run, group_dir, r.get("categories", "")]).lower()
            out.append(dict(label=label, kind=kind, target=r.get("target", ""), band=r.get("band", ""), window=r.get("window_days", ""),
                            run=run, group_dir=group_dir, group_id=gid, chips=chips, obsid=obsid, exposure_ks=ks,
                            info="   ".join(b for b in bits if b), files=files, search=search))
    out.sort(key=lambda d: (d["target"], d["run"], d["group_dir"], d["obsid"]))
    return out


OPENED_NODE = "manually opened"
OPENED_FILE = ".chanmerge_viewer_opened.json"


def read_opened(root):
    """every path remembered for this folder, also files that are missing right now (an unplugged disk comes back later)"""
    import json
    try:
        with open(os.path.join(root, OPENED_FILE)) as f:
            paths = json.load(f).get("files", [])
    except (OSError, ValueError, AttributeError):
        return []
    seen, out = set(), []
    for p in paths:
        if isinstance(p, str) and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def save_opened(root, paths):
    """remember the list in a small hidden file in the folder the gui runs from; the only file the viewer ever writes"""
    import json
    tmp = os.path.join(root, OPENED_FILE + ".tmp")
    with open(tmp, "w") as f:
        json.dump({"files": list(paths)}, f, indent=1)
    os.replace(tmp, os.path.join(root, OPENED_FILE))          # atomic, like the index files


def opened_items(paths):
    """one item per remembered file; they live under the 'manually opened' folder of the tree.
    a file that is not there any more (moved, deleted, unplugged disk) is still listed, marked missing, so it can be removed by hand;
    it cannot be drawn, and it comes back to normal when the file is back"""
    out = []
    for p in paths:
        name = os.path.basename(p)
        gone = not os.path.exists(p)
        out.append(dict(label=f"{OPENED_NODE} | {name}", kind="file", target=OPENED_NODE, band="", window="", run="", group_dir="", group_id="",
                        chips="", obsid="", exposure_ks="", info=p, files={} if gone else {"flux": p}, missing=gone,
                        search=f"{OPENED_NODE} {name} {p}".lower()))
    return out


def filter_items(items, query):
    """indices of the items that match every word of the query (case does not matter); an empty query matches all"""
    words = (query or "").lower().split()
    return [i for i, it in enumerate(items) if all(w in it["search"] for w in words)]


def build_tree(items, keep=None):
    """nested nodes for the list on the left: target > run (band + window) > group > member observations.
    a node is dict(text, kind, index, children); index is the item's place in items (none for a folder).
    keep = indices to show (default all); a kept member also pulls in its group so it is never loose"""
    keep = set(range(len(items)) if keep is None else keep)
    by_dir = {(it["target"], it["group_dir"]): i for i, it in enumerate(items) if it["kind"] == "final"}
    show = set(keep)
    for i in keep:
        it = items[i]
        if it["kind"] == "member" and (it["target"], it["group_dir"]) in by_dir:
            show.add(by_dir[(it["target"], it["group_dir"])])
    targets, runs, groups, opened = {}, {}, {}, []

    def node(text, kind, index=None):
        return dict(text=text, kind=kind, index=index, children=[])
    for i in sorted(show):
        it = items[i]
        if it["kind"] == "file":
            opened.append(node(os.path.basename(it["info"]) + (" (not found)" if it.get("missing") else ""), "file", i))
            continue
        t = targets.setdefault(it["target"], node(it["target"] or "(no target)", "target"))
        rk = (it["target"], it["run"])
        if rk not in runs:
            runs[rk] = node(f"{it['band']} | w{it['window']}d", "run")
            t["children"].append(runs[rk])
        gk = (it["target"], it["group_dir"])
        chips = f" [{it['chips']}]" if it["chips"] and it["chips"] != "all" else ""
        if it["kind"] == "final":
            groups[gk] = node(f"{it['group_id']}{chips}", "group", i)
            runs[rk]["children"].append(groups[gk])
        else:
            if gk not in groups:                     # group row is not in the index: a plain folder keeps the member findable
                groups[gk] = node(f"{it['group_id']}{chips}", "group")
                runs[rk]["children"].append(groups[gk])
            groups[gk]["children"].append(node(f"obs {it['obsid']}", "obs", i))
    top = [targets[k] for k in sorted(targets)]
    if opened:
        folder = node(OPENED_NODE, "opened")
        folder["children"] = opened
        top.append(folder)
    return top


def load_fits_image(path):
    """(data float array, header, celestial wcs or none) of the first 2-d image in a fits file"""
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(path) as h:
        for hdu in h:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
                data = np.asarray(hdu.data, dtype=float)
                header = hdu.header
                break
        else:
            raise ValueError(f"{os.path.basename(path)} holds no 2-D image")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs = WCS(header, naxis=2)
        if not wcs.has_celestial:
            wcs = None
    except Exception:
        wcs = None
    return data, header, wcs


# ---------- scales and limits ----------
def limits_of(data, mode, user=(None, None)):
    """(lo, hi) for a display mode over the finite pixels. 'user' takes the typed numbers, a blank side falls back to the data min/max"""
    v = np.asarray(data, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    if mode == "user":
        lo = user[0] if user[0] is not None else float(v.min())
        hi = user[1] if user[1] is not None else float(v.max())
    elif mode == "zscale":
        from astropy.visualization import ZScaleInterval
        lo, hi = ZScaleInterval().get_limits(v)
    elif mode.endswith("%"):
        keep = float(mode[:-1])
        lo, hi = np.percentile(v, [(100 - keep) / 2, 100 - (100 - keep) / 2])
    else:
        lo, hi = float(v.min()), float(v.max())
    lo, hi = float(lo), float(hi)
    if not hi > lo:
        hi = lo + max(abs(lo) * 1e-6, 1e-30)
    return lo, hi


# normalised curves g(t), t in 0..1 -> 0..1, and their inverses (ds9's names; log is ds9's with a = 1000, scaled so g(1) = 1)
_CURVES = {
    "log": (lambda t: np.log10(1000.0 * t + 1.0) / np.log10(1001.0), lambda u: (1001.0 ** u - 1.0) / 1000.0),
    "sqrt": (lambda t: np.sqrt(t), lambda u: u ** 2),
    "squared": (lambda t: t ** 2, lambda u: np.sqrt(u)),
    "asinh": (lambda t: np.arcsinh(10.0 * t) / np.arcsinh(10.0), lambda u: np.sinh(u * np.arcsinh(10.0)) / 10.0),
}


def make_norm(scale, lo, hi):
    """matplotlib norm for the scale between lo and hi (values outside are clipped); colour bar labels stay in data units"""
    from matplotlib import colors
    if scale == "linear" or scale not in _CURVES:
        return colors.Normalize(vmin=lo, vmax=hi, clip=True)
    g, ginv = _CURVES[scale]
    span = hi - lo

    def fwd(x):
        return g(np.clip((x - lo) / span, 0.0, 1.0))

    def inv(u):
        return lo + span * ginv(np.clip(u, 0.0, 1.0))
    return colors.FuncNorm((fwd, inv), vmin=lo, vmax=hi)

def layout_slots(n):
    """(rows, columns, [(row0, row1, col0, col1) per picture]) of the grid for n pictures (1-4).
    3 pictures: two side by side, the third centred under them (a 4-column grid, each picture 2 columns wide)"""
    if n == 1:
        return 1, 1, [(0, 1, 0, 1)]
    if n == 2:
        return 1, 2, [(0, 1, 0, 1), (0, 1, 1, 2)]
    if n == 3:
        return 2, 4, [(0, 1, 0, 2), (0, 1, 2, 4), (1, 2, 1, 3)]
    return 2, 4, [(0, 1, 0, 2), (0, 1, 2, 4), (1, 2, 0, 2), (1, 2, 2, 4)]


# ---------- smoothing (ds9: analysis > smooth) ----------
def smooth_kernel(function="boxcar", radius=3, sigma=None, radius_minor=None, sigma_minor=None, angle=0.0):
    """normalised kernel as in ds9's smooth parameters. radius is in pixels (the kernel is (2r+1) wide); sigma defaults to radius / 2.
    boxcar: square. tophat: disc as ds9 draws it (see below). gaussian: circular gaussian cut at the radius.
    elliptical gaussian: sigma along the major axis (angle, degrees, from +x), sigma_minor across, cut by an ellipse radius x radius_minor."""
    r = int(radius)
    if r < 1:
        raise ValueError("smooth radius must be 1 or more")
    rm = int(radius_minor) if radius_minor not in (None, "", 0) else r
    if rm < 1:
        raise ValueError("smooth radius minor must be 1 or more")
    half = max(r, rm) if function == "elliptical gaussian" else r
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    if function == "boxcar":
        k = np.ones_like(x)
    elif function == "tophat":
        # ds9 8.7's disc is a little wider than x^2+y^2 <= r^2: it keeps x^2+y^2 <= r^2 + (r-1)^2 (45 pixels at r=3, not 29).
        # found by comparing its png output with this kernel for r = 1..6 (rms 0.58 gray levels = pixel exact, test h)
        k = (x * x + y * y <= r * r + (r - 1) * (r - 1)).astype(float)
    elif function == "gaussian":
        s = float(sigma) if sigma else r / 2.0
        if s <= 0:
            raise ValueError("smooth sigma must be above 0")
        k = np.exp(-(x * x + y * y) / (2.0 * s * s)) * (x * x + y * y <= r * r)
    elif function == "elliptical gaussian":
        s1 = float(sigma) if sigma else r / 2.0
        s2 = float(sigma_minor) if sigma_minor else rm / 2.0
        if s1 <= 0 or s2 <= 0:
            raise ValueError("smooth sigma must be above 0")
        th = np.radians(float(angle or 0.0))
        xr = x * np.cos(th) + y * np.sin(th)
        yr = -x * np.sin(th) + y * np.cos(th)
        k = np.exp(-(xr * xr / (2.0 * s1 * s1) + yr * yr / (2.0 * s2 * s2))) * ((xr / r) ** 2 + (yr / rm) ** 2 <= 1.0)
    else:
        raise ValueError(f"unknown smooth function {function}")
    return k / k.sum()


def smooth_image(data, kernel):
    """smooth with the kernel; no-data (nan) pixels are left out of every average (so they do not pull the edges down) and stay nan"""
    d = np.asarray(data, dtype=float)
    valid = np.isfinite(d)
    if not valid.any():
        return d.copy()
    from scipy.signal import fftconvolve
    num = fftconvolve(np.where(valid, d, 0.0), kernel, mode="same")
    den = fftconvolve(valid.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(valid & (den > 1e-9), num / den, np.nan)
    return out


def shared_limits(arrays, mode, user=(None, None)):
    """one (lo, hi) for several images (the 'same limits' option): the mode applied to all their finite pixels together"""
    parts = [np.asarray(a, dtype=float).ravel() for a in arrays if a is not None]
    if not parts:
        return 0.0, 1.0
    return limits_of(np.concatenate(parts), mode, user)


def parse_float(text):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


# ---------- the tab ----------
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:  # pragma: no cover
    tk = None


class _Panel:
    """one picture slot: the image it shows (raw and smoothed), its axes while drawn, and its own zoom"""

    def __init__(self):
        self.item = None
        self.show = "flux"
        self.path = None
        self.title = ""
        self.raw = None
        self.data = None          # what is drawn: raw, or smoothed
        self.wcs = None
        self.ax = None
        self.im = None
        self.view = None
        self.fresh = False        # a new image was just put here: do not carry the old image's zoom over
        self.lims = None


class Viewer(ttk.Frame if tk else object):
    """list of images on the left; on the right 1, 2, 3 or 4 pictures (2x2) with ds9-like controls"""

    def __init__(self, master, cwd=None):
        super().__init__(master, padding=6)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        self.cwd = cwd or os.getcwd()
        self.items = []
        self.opened = read_opened(self.cwd)     # files opened by hand, remembered between sessions
        self.notify = lambda title, text: messagebox.showwarning(title, text, parent=self)      # a test swaps this for a recorder
        self.cache = {}
        self.panels = [_Panel() for _ in range(4)]
        self.active = 0
        self._linking = False
        self._handled = False
        V = tk.StringVar
        self.vars = dict(show=V(value="flux"), cmap=V(value="viridis"), scale=V(value="linear"), limits=V(value="99.5%"),
                         lo=V(), hi=V(), nan=V(value="white"), panels=V(value="1"), search=V(),
                         reverse=tk.BooleanVar(value=False), colorbar=tk.BooleanVar(value=True),
                         lock=tk.BooleanVar(value=False), link=tk.BooleanVar(value=False),
                         smooth=tk.BooleanVar(value=False), sfunc=V(value=SMOOTH_DEFAULTS["function"]), sradius=V(value="3"),
                         ssigma=V(value="1.5"), srminor=V(value="3"), ssminor=V(value="1.5"), sangle=V(value="0"))
        self.smooth_win = None
        self.status = V(value="")
        self.info = V(value="")

        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw, padding=2)
        right = ttk.Frame(pw, padding=2)
        pw.add(left, weight=1)
        pw.add(right, weight=4)

        ttk.Label(left, text=f"Images in {self.cwd}", wraplength=260).pack(anchor="w")
        bar = ttk.Frame(left)
        bar.pack(fill="x", pady=2)
        ttk.Button(bar, text="Refresh", command=lambda: self.refresh(rescan=True)).pack(side="left")
        ttk.Button(bar, text="Open file...", command=self.open_file).pack(side="left", padx=4)
        ttk.Button(bar, text="Remove", command=self.forget_opened).pack(side="left")
        sb = ttk.Frame(left)
        sb.pack(fill="x", pady=2)
        ttk.Label(sb, text="Search").pack(side="left")
        ent = ttk.Entry(sb, textvariable=self.vars["search"])
        ent.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(sb, text="x", width=2, command=lambda: self.vars["search"].set("")).pack(side="left")
        self.vars["search"].trace_add("write", lambda *a: self.fill_tree())
        self.count = ttk.Label(left, text="", foreground="gray")
        self.count.pack(anchor="w")
        tf = ttk.Frame(left)
        tf.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tf, columns=("kind", "ks"), show="tree headings", selectmode="browse", height=18)
        self.tree.heading("#0", text="target / run / group / observation")
        self.tree.heading("kind", text="type")
        self.tree.heading("ks", text="ks")
        self.tree.column("#0", width=210)
        self.tree.column("kind", width=55)
        self.tree.column("ks", width=45)
        sy = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sy.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sy.pack(side="left", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.tree.bind("<ButtonPress-1>", lambda e: setattr(self, "_handled", False))
        self.tree.bind("<ButtonRelease-1>", self.on_tree_click)
        self.empty = ttk.Label(left, text="", wraplength=260, foreground="#a33")
        self.empty.pack(anchor="w")

        ctl = ttk.Frame(right)
        ctl.pack(fill="x")
        for i, (text, key, values, w) in enumerate((("Pictures", "panels", PANEL_CHOICES, 7), ("Show", "show", SHOWS, 14),
                                                    ("Colour map", "cmap", COLORMAPS, 10), ("Scale", "scale", SCALES, 8),
                                                    ("Limits", "limits", LIMITS, 8), ("No-data colour", "nan", NAN_COLORS, 8))):
            ttk.Label(ctl, text=text).grid(row=0, column=i, sticky="w", padx=3)
            cb = ttk.Combobox(ctl, textvariable=self.vars[key], values=values, width=w, state="readonly")
            cb.grid(row=1, column=i, padx=3)
            cb.bind("<<ComboboxSelected>>", lambda e, k=key: self.on_option(k))
        ttk.Label(ctl, text="min (Limits = user)").grid(row=0, column=6, sticky="w", padx=3)
        ttk.Label(ctl, text="max").grid(row=0, column=7, sticky="w", padx=3)
        for col, key in ((6, "lo"), (7, "hi")):
            e = ttk.Entry(ctl, textvariable=self.vars[key], width=11)
            e.grid(row=1, column=col, padx=3)
            e.bind("<Return>", lambda ev: self.on_user_limits())
        ctl2 = ttk.Frame(right)
        ctl2.pack(fill="x", pady=2)
        for text, key in (("reverse", "reverse"), ("colour bar", "colorbar"), ("same limits in all pictures", "lock"),
                          ("link zoom (same pixel grid)", "link")):
            ttk.Checkbutton(ctl2, text=text, variable=self.vars[key], command=self.redraw).pack(side="left", padx=4)
        ttk.Button(ctl2, text="Zoom to fit", command=self.zoom_fit).pack(side="left", padx=6)
        ttk.Checkbutton(ctl2, text="Smooth", variable=self.vars["smooth"], command=self.on_smooth).pack(side="left", padx=(14, 2))
        ttk.Button(ctl2, text="Smooth parameters...", command=self.open_smooth_params).pack(side="left", padx=2)

        ttk.Label(right, textvariable=self.info, wraplength=700).pack(anchor="w", pady=2)
        self.fig = Figure(figsize=(7, 5.6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        tb.update()
        tb.pack(fill="x")
        ttk.Label(right, textvariable=self.status, font=("Menlo", 10)).pack(anchor="w")
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.refresh()

    # ----- the active picture (older code and tests read these) -----
    @property
    def cur(self):
        return self.panels[self.active].item

    @property
    def data(self):
        return self.panels[self.active].data

    @property
    def path(self):
        return self.panels[self.active].path

    @property
    def wcs(self):
        return self.panels[self.active].wcs

    def n_panels(self):
        try:
            return int(str(self.vars["panels"].get()).split()[0])
        except (ValueError, IndexError):
            return 1

    # ----- list -----
    def refresh(self, rescan=False):
        """re-read the index files. rescan (the refresh button) rebuilds them from every run folder under cwd first, so new runs show up;
        without it the index is only built when there is none yet"""
        have = any(os.path.exists(os.path.join(self.cwd, n)) for n in ("chanmerge_index_final.csv", "chanmerge_index_members.csv"))
        if rescan or not have:
            try:
                from .tables import rebuild_index
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rebuild_index(self.cwd)
            except Exception as e:
                self.status.set(f"could not scan {self.cwd}: {e}")
        self.reload_list()

    def remember(self):
        try:
            save_opened(self.cwd, self.opened)
        except OSError as e:
            self.status.set(f"could not remember the list in {self.cwd}: {e}")

    def all_items(self):
        return list_images(self.cwd, include_members=True) + opened_items(self.opened)

    def reload_list(self):
        self.items = self.all_items()
        self.fill_tree()
        self.empty.config(text="" if self.items else "No images found here. Run chanmerge from this folder, or use Open file...")
        if self.items and all(p.item is None for p in self.panels):
            # no selection_set here: its event arrives later and would put the first image into whichever picture is active by then
            first = next((i for i, it in enumerate(self.items) if it["kind"] == "final"), 0)
            self.assign(self.items[first], "flux")

    def fill_tree(self):
        """rebuild the tree from the items that match the search; with a search typed everything is opened so the hits are visible"""
        query = self.vars["search"].get()
        keep = filter_items(self.items, query)
        self.tree.delete(*self.tree.get_children())
        self._n = 0

        def add(parent, nd):
            if nd["index"] is not None:
                it = self.items[nd["index"]]
                iid = str(nd["index"])
                vals = (nd["kind"], it["exposure_ks"])
            else:
                self._n += 1
                iid = f"f{self._n}"
                vals = (nd["kind"], "")
            opened = bool(query.strip()) or nd["kind"] in ("target", "run", "opened")
            self.tree.insert(parent, "end", iid=iid, text=nd["text"], values=vals, open=opened)
            for ch in nd["children"]:
                add(iid, ch)
        for top in build_tree(self.items, keep):
            add("", top)
        n_ok = lambda idx: sum(1 for k in idx if not self.items[k].get("missing"))
        n_gone = sum(1 for it in self.items if it.get("missing"))
        self.count.config(text=(f"{n_ok(keep)} of {n_ok(range(len(self.items)))} images" + (f"  ({n_gone} not found)" if n_gone else ""))
                          if self.items else "")
        if self.items and not keep:
            self.empty.config(text="Nothing matches the search.")
        elif self.items:
            self.empty.config(text="")

    def open_file(self):
        """pick any fits image; it goes in the active picture and stays in the list under 'manually opened', also next time, until removed"""
        p = filedialog.askopenfilename(title="Open a FITS image", initialdir=self.cwd,
                                       filetypes=[("FITS images", "*.img *.fits *.fits.gz *.fit"), ("all files", "*")])
        if not p:
            return
        p = os.path.abspath(p)
        if p not in self.opened:
            self.opened.append(p)
            self.remember()
        self.items = self.all_items()
        self.fill_tree()
        mine = next((it for it in self.items if it["kind"] == "file" and it["info"] == p), None)
        self.assign(mine or dict(label=os.path.basename(p), info=p, files={"flux": p}), "flux")

    def forget_opened(self):
        """take the selected 'manually opened' file out of the list (the file itself is not touched)"""
        sel = self.tree.selection()
        if not sel or not sel[0].isdigit() or self.items[int(sel[0])]["kind"] != "file":
            self.status.set("select a file under 'manually opened' to remove it from the list")
            return
        gone = self.items[int(sel[0])]["info"]
        self.opened = [q for q in self.opened if q != gone]
        self.remember()
        self.items = self.all_items()
        self.fill_tree()
        self.status.set(f"removed {os.path.basename(gone)} from the list")

    def on_select(self, _e=None):
        self._handled = True
        sel = self.tree.selection()
        if sel and sel[0].isdigit():                  # folder rows (target, run) have ids like f3 and are not images
            self.assign(self.items[int(sel[0])], self.vars["show"].get())

    def on_tree_click(self, e):
        """a click on the row that is already selected does not fire a selection event; it still puts that image in the active picture"""
        if self._handled:
            return
        iid = self.tree.identify_row(e.y)
        if iid and iid.isdigit():
            self.assign(self.items[int(iid)], self.vars["show"].get())

    def assign(self, item, show):
        """put an image in the active picture; if it cannot be read nothing changes and the user is told which file is the problem"""
        p = self.panels[self.active]
        if show not in item["files"]:
            show = "flux"
        path = item["files"].get(show) or item.get("info", "")
        if item.get("missing") or not os.path.exists(path):
            self.file_gone(item, path)
            return
        old = (p.item, p.show)
        p.item, p.show = item, show
        if self.load_into(p, path):
            self.vars["show"].set(show)
            p.view, p.fresh = None, True
        else:
            p.item, p.show = old                    # keep what was drawn before; no redraw, it would wipe the warning from the status line
            return
        self.draw()

    def file_gone(self, item, path):
        """the file of a picked item is not at its path any more. say which file, once, in a warning window and on the status line.
        a file opened by hand is taken off the 'manually opened' list right away (so it does not sit there); a chanmerge image is only
        reported, refresh rescans the run folders"""
        self.cache.pop(path, None)
        if item.get("kind") == "file":
            self.opened = [q for q in self.opened if q != path]
            self.remember()
            self.items = self.all_items()
            self.fill_tree()
            text = (f"This file is not at its old path any more:\n\n{path}\n\nIt was moved, renamed or deleted since you opened it, so it "
                    f"cannot be drawn and has been taken off the 'manually opened' list.\n\nTo get it back, put the file at the path above again "
                    f"or use Open file... on its new place.")
            self.status.set(f"not drawn: {os.path.basename(path)} is not at {os.path.dirname(path)} any more; removed from the manually opened list")
        else:
            text = (f"This chanmerge image is not at its path any more:\n\n{path}\n\n{item.get('label', '')}\n\nIt was moved or deleted after the "
                    f"index was written. Press Refresh to scan the run folders again.")
            self.status.set(f"not drawn: {os.path.basename(path)} is missing; press Refresh")
        self.notify("File not found", text)

    def load_into(self, p, path):
        if not os.path.exists(path):                # also for chanmerge outputs moved while the window is open; never draw from the cache
            self.cache.pop(path, None)
            self.status.set(f"file not found: {path} (moved or deleted?). nothing drawn")
            return False
        try:
            if path not in self.cache:
                if len(self.cache) >= 8:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[path] = load_fits_image(path)
            p.raw, _hdr, p.wcs = self.cache[path]
        except Exception as e:
            self.status.set(f"could not read {os.path.basename(path)}: {e}")
            return False
        p.path = path
        p.title = p.item["label"] if p.item else os.path.basename(path)
        p.data = p.raw
        self.apply_smooth(quiet=True)
        return True

    def on_option(self, key):
        p = self.panels[self.active]
        if key == "show":
            show = self.vars["show"].get()
            if p.item and show in p.item["files"]:
                old_show, p.show = p.show, show
                if self.load_into(p, p.item["files"][show]):
                    p.view, p.fresh = None, True
                    self.draw()
                else:
                    p.show = old_show
                    self.vars["show"].set(old_show)
            else:
                self.status.set("this image has no such file")
                self.vars["show"].set(p.show)
        elif key == "panels":
            self.draw()
        else:
            self.redraw()

    def on_user_limits(self):
        self.vars["limits"].set("user")
        self.redraw()

    # ----- smoothing -----
    def kernel(self):
        """the kernel for the smooth settings, or none when smoothing is off; valueerror with a readable text for bad numbers"""
        if not self.vars["smooth"].get():
            return None
        def num(key, kind=float):
            t = self.vars[key].get().strip()
            if not t:
                return None
            try:
                return kind(t)
            except ValueError:
                raise ValueError(f"'{t}' is not a number")
        return smooth_kernel(self.vars["sfunc"].get(), num("sradius", int) or 3, num("ssigma"), num("srminor", int), num("ssminor"),
                             num("sangle") or 0.0)

    def open_smooth_params(self):
        """ds9's 'smooth parameters' window: function, radius, sigma, radius minor, sigma minor, angle. apply turns smoothing on"""
        if self.smooth_win is not None and self.smooth_win.winfo_exists():
            self.smooth_win.lift()
            return
        w = tk.Toplevel(self)
        w.title("Smooth parameters")
        self.smooth_win = w
        f = ttk.Frame(w, padding=10)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="Function").grid(row=0, column=0, sticky="w", pady=2)
        cb = ttk.Combobox(f, textvariable=self.vars["sfunc"], values=SMOOTH_FUNCTIONS, width=20, state="readonly")
        cb.grid(row=0, column=1, pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self.on_smooth(turn_on=True))
        for r, (text, key) in enumerate((("Radius (pixels)", "sradius"), ("Sigma", "ssigma"), ("Radius minor", "srminor"),
                                         ("Sigma minor", "ssminor"), ("Angle (degrees)", "sangle")), start=1):
            ttk.Label(f, text=text).grid(row=r, column=0, sticky="w", pady=2)
            e = ttk.Entry(f, textvariable=self.vars[key], width=10)
            e.grid(row=r, column=1, sticky="w", pady=2)
            e.bind("<Return>", lambda ev: self.on_smooth(turn_on=True))
        ttk.Label(f, text="Radius minor, sigma minor and angle only matter for the elliptical gaussian.\nThe values shown are ds9's starting values.",
                  foreground="gray").grid(row=6, column=0, columnspan=2, sticky="w", pady=4)
        bar = ttk.Frame(f)
        bar.grid(row=7, column=0, columnspan=2, sticky="e")
        ttk.Button(bar, text="Reset", command=self.reset_smooth).pack(side="left", padx=3)
        ttk.Button(bar, text="Apply", command=lambda: self.on_smooth(turn_on=True)).pack(side="left", padx=3)
        ttk.Button(bar, text="Close", command=w.destroy).pack(side="left", padx=3)

    def reset_smooth(self):
        d = SMOOTH_DEFAULTS
        for key, val in (("sfunc", d["function"]), ("sradius", str(d["radius"])), ("ssigma", str(d["sigma"])),
                         ("srminor", str(d["radius_minor"])), ("ssminor", str(d["sigma_minor"])), ("sangle", str(d["angle"]))):
            self.vars[key].set(val)
        self.on_smooth(turn_on=True)

    def apply_smooth(self, quiet=False):
        try:
            k = self.kernel()
        except ValueError as e:
            self.status.set(f"smooth: {e}")
            return False
        for p in self.panels:
            if p.raw is not None:
                p.data = smooth_image(p.raw, k) if k is not None else p.raw
        return True

    def on_smooth(self, turn_on=False):
        if turn_on:
            self.vars["smooth"].set(True)
        if self.apply_smooth():
            self.draw()

    # ----- drawing -----
    def redraw(self):
        self.draw()

    def draw(self, reset_view=False):
        import matplotlib
        n = self.n_panels()
        for p in self.panels:
            if p.ax is not None and p.im is not None and not reset_view and not p.fresh:
                p.view = (p.ax.get_xlim(), p.ax.get_ylim())
            if reset_view:
                p.view = None
            p.fresh = False
        self.fig.clear()
        rows, cols, slots = layout_slots(n)
        gs = self.fig.add_gridspec(rows, cols, left=0.07 if n == 1 else 0.06, right=0.95 if n == 1 else 0.97, bottom=0.07,
                                   top=0.97 if n == 1 else 0.92, wspace=0.5 if n < 3 else 0.9, hspace=0.35)
        mode = self.vars["limits"].get()
        user = (parse_float(self.vars["lo"].get()), parse_float(self.vars["hi"].get()))
        shown = [p for p in self.panels[:n] if p.data is not None]
        same = shared_limits([p.data for p in shown], mode, user) if (self.vars["lock"].get() and shown) else None
        name = self.vars["cmap"].get() + ("_r" if self.vars["reverse"].get() else "")
        for i, p in enumerate(self.panels):
            p.ax = p.im = None
            if i >= n:
                continue
            r0, r1, c0, c1 = slots[i]
            p.ax = self.fig.add_subplot(gs[r0:r1, c0:c1])
            if p.data is None:
                p.ax.text(0.5, 0.5, "empty\nclick this picture,\nthen pick an image in the list", ha="center", va="center",
                          transform=p.ax.transAxes, fontsize=9, color="gray")
                p.ax.set_xticks([])
                p.ax.set_yticks([])
                continue
            lo, hi = same or limits_of(p.data, mode, user)
            p.lims = (lo, hi)
            cmap = matplotlib.colormaps[name].copy()
            cmap.set_bad(self.vars["nan"].get())
            p.im = p.ax.imshow(np.ma.masked_invalid(p.data), origin="lower", cmap=cmap, norm=make_norm(self.vars["scale"].get(), lo, hi),
                               interpolation="nearest", aspect="equal")
            if self.vars["colorbar"].get():
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                cax = make_axes_locatable(p.ax).append_axes("right", size=0.2 if n == 1 else 0.12, pad=0.1)   # inches: a narrow strip keeps a visible bar
                self.fig.colorbar(p.im, cax=cax)
                cax.tick_params(labelsize=8 if n == 1 else 6)
            if n > 1:
                p.ax.set_title(f"{i + 1}: {p.title}"[:46], fontsize=7.5)
                p.ax.tick_params(labelsize=6.5)
            if p.view:
                p.ax.set_xlim(p.view[0])
                p.ax.set_ylim(p.view[1])
            p.ax.callbacks.connect("xlim_changed", self._on_lim)
            p.ax.callbacks.connect("ylim_changed", self._on_lim)
        self.mark_active()
        ap = self.panels[self.active] if self.active < n else None
        if ap is not None and ap.lims and mode != "user":
            self.vars["lo"].set(f"{ap.lims[0]:.4g}")
            self.vars["hi"].set(f"{ap.lims[1]:.4g}")
        self.canvas.draw_idle()
        self.show_summary()

    def mark_active(self):
        n = self.n_panels()
        for i, p in enumerate(self.panels[:n]):
            if p.ax is None:
                continue
            on = (i == self.active and n > 1)
            for sp in p.ax.spines.values():
                sp.set_linewidth(2.6 if on else 0.8)
                sp.set_edgecolor("#e07b00" if on else "black")

    def show_summary(self):
        p = self.panels[self.active]
        self.info.set((f"[{self.active + 1}] " if self.n_panels() > 1 else "") + (f"{p.title}   {p.item['info']}" if p.item else (p.path or "")))
        if p.data is None:
            self.status.set("")
            return
        nan = int(np.sum(~np.isfinite(p.data)))
        sm = "   smoothed: " + self.vars["sfunc"].get() + f" r={self.vars['sradius'].get() or 3}" if self.vars["smooth"].get() else ""
        lim = f", limits {p.lims[0]:.4g} .. {p.lims[1]:.4g}" if p.lims else ""
        self.status.set(f"{p.data.shape[1]} x {p.data.shape[0]} pixels, {nan} no-data (NaN){lim}{sm}")

    def zoom_fit(self):
        self._linking = True
        try:
            for p in self.panels[:self.n_panels()]:
                if p.im is not None:
                    h, w = p.data.shape
                    p.ax.set_xlim(-0.5, w - 0.5)
                    p.ax.set_ylim(-0.5, h - 0.5)
                    p.view = None
        finally:
            self._linking = False
        self.canvas.draw_idle()

    def _on_lim(self, ax):
        """link zoom: the other pictures take this picture's pixel range (only meaningful when they share a pixel grid)"""
        if self._linking or not self.vars["link"].get():
            return
        self._linking = True
        try:
            xl, yl = ax.get_xlim(), ax.get_ylim()
            for p in self.panels[:self.n_panels()]:
                if p.ax is not None and p.ax is not ax and p.im is not None:
                    p.ax.set_xlim(xl)
                    p.ax.set_ylim(yl)
        finally:
            self._linking = False
        self.canvas.draw_idle()

    # ----- mouse -----
    def panel_of(self, ax):
        for i, p in enumerate(self.panels[:self.n_panels()]):
            if p.ax is ax:
                return i, p
        return None, None

    def on_press(self, e):
        """a click picks the active picture (orange frame); the list and the show menu then act on it"""
        i, p = self.panel_of(e.inaxes)
        if p is None or i == self.active:
            return
        self.active = i
        self.vars["show"].set(p.show)
        if p.lims and self.vars["limits"].get() != "user":
            self.vars["lo"].set(f"{p.lims[0]:.4g}")
            self.vars["hi"].set(f"{p.lims[1]:.4g}")
        self.mark_active()
        self.canvas.draw_idle()
        self.show_summary()

    def on_motion(self, e):
        i, p = self.panel_of(e.inaxes)
        if p is None or p.im is None or e.xdata is None:
            return
        x, y = int(round(e.xdata)), int(round(e.ydata))
        h, w = p.data.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        v = p.data[y, x]
        sky = ""
        if p.wcs is not None:
            try:
                ra, dec = p.wcs.pixel_to_world_values(x, y)
                sky = f"   RA {float(ra):.5f}  Dec {float(dec):.5f}"
            except Exception:
                pass
        pre = f"[{i + 1}] " if self.n_panels() > 1 else ""
        self.status.set(f"{pre}x {x + 1}  y {y + 1}{sky}   value " + ("no data" if not np.isfinite(v) else f"{v:.5g}"))

    def on_scroll(self, e):
        i, p = self.panel_of(e.inaxes)
        if p is None or p.im is None or e.xdata is None:
            return
        k = 0.8 if e.button == "up" else 1.25
        x0, x1 = p.ax.get_xlim()
        y0, y1 = p.ax.get_ylim()
        p.ax.set_xlim(e.xdata - (e.xdata - x0) * k, e.xdata + (x1 - e.xdata) * k)
        p.ax.set_ylim(e.ydata - (e.ydata - y0) * k, e.ydata + (y1 - e.ydata) * k)
        self.canvas.draw_idle()
