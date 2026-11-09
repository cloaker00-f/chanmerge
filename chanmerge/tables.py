"""The four tables: final_images.csv and member_images.csv in each run folder, and the two indexes at the root.

paths in run tables are relative to the run folder; in the indexes relative to the root, with _abs columns.
"""
import csv
import glob
import os
import tempfile
import warnings

from .bands import bands_overlap
from .plan import CHIP_IDS

FINAL_COLUMNS = [
    "target", "band", "window_days", "group_id", "chips", "type", "n_obs", "obsids", "categories", "n_grating",
    "start_utc", "end_utc", "span_days", "total_exposure_ks",
    "flux_image", "flux_image_original", "evt_file", "evt_file_original", "repro_evt_file", "repro_evt_file_abs",
    "acisreadcorr_evt_file", "acisreadcorr_evt_file_abs", "letg_imaging_evt_file", "letg_imaging_evt_file_abs",
    "expmap_file", "expmap_file_original", "expmap_p5", "expmap_median", "expmap_p95",
    "streak_action", "streak_obsids", "arm_method", "arm_masked_obsids", "arm_filled_obsids",
    "filled_pixel_fraction", "mask_file", "shares_photons_with",
    # rule 3 (session 007): group id this single's observation left because it could not be mixed with the others ('' normally)
    "left_out_of",
]
MEMBER_COLUMNS = [
    "target", "band", "window_days", "group_id", "chips", "obsid", "detector", "grating", "mode", "window",
    "start_utc", "end_utc", "exposure_ks",
    "flux_image", "flux_image_original", "evt_file", "evt_file_original", "repro_evt_file", "repro_evt_file_abs",
    "acisreadcorr_evt_file", "acisreadcorr_evt_file_abs", "letg_imaging_evt_file", "letg_imaging_evt_file_abs",
    "expmap_file", "expmap_file_original", "streak_action", "arm_method", "filled_pixel_fraction", "mask_file",
    "shares_photons_with",
]
# path columns that get root-relative paths and an _abs twin in the indexes
PATH_COLUMNS = ["flux_image", "flux_image_original", "evt_file", "evt_file_original", "expmap_file",
                "expmap_file_original", "mask_file", "repro_evt_file", "acisreadcorr_evt_file",
                "letg_imaging_evt_file"]
INDEX_FINAL = "chanmerge_index_final.csv"
INDEX_MEMBERS = "chanmerge_index_members.csv"


def index_columns(cols):
    out = list(cols)
    for c in PATH_COLUMNS:
        a = c + "_abs"
        if c in out and a not in out:
            out.insert(out.index(c) + 1, a)
    return out


# ---------- writing ----------
def write_csv_atomic(path, columns, rows):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=d)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: _cell(r.get(c, "")) for c in columns})
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rel(path, start):
    return os.path.relpath(path, start).replace(os.sep, "/")


# ---------- shared photons ----------
def chip_set_of(text):
    """'all' -> None, 'S3' / 'S1,S3' / 'I0-I3' -> set of ccd ids"""
    t = (text or "").strip()
    if t in ("", "all", "-"):
        return None
    if t == "I0-I3":
        return {0, 1, 2, 3}
    out = set()
    for n in t.replace(";", ",").split(","):
        n = n.strip().upper()
        if n in CHIP_IDS:
            out.add(CHIP_IDS[n])
    return out or None


def chips_overlap(a, b):
    sa, sb = chip_set_of(a), chip_set_of(b)
    if sa is None or sb is None:
        return True
    return bool(sa & sb)


def shares_map(images, key, own_group=None):
    """images: list of dicts with key, obsids (list), chips, band. two images share photons when they have an
    obsid in common, overlapping chips and overlapping bands. own_group(key) -> images that never count
    (a member and its own group's other members are not 'shared')."""
    out = {key(im): [] for im in images}
    by_obs = {}
    for im in images:
        for o in im["obsids"]:
            by_obs.setdefault(o, []).append(im)
    for o, ims in by_obs.items():
        for i, a in enumerate(ims):
            for b in ims[i + 1:]:
                ka, kb = key(a), key(b)
                if ka == kb:
                    continue
                if own_group and own_group(a) == own_group(b):
                    continue
                if chips_overlap(a["chips"], b["chips"]) and bands_overlap(a["band"], b["band"]):
                    if kb not in out[ka]:
                        out[ka].append(kb)
                    if ka not in out[kb]:
                        out[kb].append(ka)
    return {k: ";".join(sorted(v)) for k, v in out.items()}


def fill_shares_run(final_rows, member_rows):
    """inside one run folder: final rows by group_id, member rows by group_id:obsid"""
    ims = [dict(key=r["group_id"], obsids=r["obsids"].split(";"), chips=r["chips"], band=r["band"]) for r in final_rows]
    m = shares_map(ims, key=lambda d: d["key"])
    for r in final_rows:
        r["shares_photons_with"] = m.get(r["group_id"], "")
    mims = [dict(key=f'{r["group_id"]}:{r["obsid"]}', obsids=[r["obsid"]], chips=r["chips"], band=r["band"],
                 group=r["group_id"]) for r in member_rows]
    mm = shares_map(mims, key=lambda d: d["key"], own_group=lambda d: d["group"])
    for r in member_rows:
        r["shares_photons_with"] = mm.get(f'{r["group_id"]}:{r["obsid"]}', "")


# ---------- index ----------
def _run_folders(root):
    return sorted(glob.glob(os.path.join(glob.escape(root), "*", "merged_final_image_*")))


def rebuild_index(root):
    """rewrite both index files from every run folder under root, from scratch and atomically.
    malformed tables and rows whose flux image is gone are skipped with a warning."""
    root = os.path.abspath(root)
    finals, members = [], []
    for run in _run_folders(root):
        prefix = rel(run, root)
        for name, need, sink in (("final_images.csv", FINAL_COLUMNS, finals), ("member_images.csv", MEMBER_COLUMNS, members)):
            p = os.path.join(run, name)
            if not os.path.exists(p):
                continue
            try:
                rows = read_csv(p)
            except (OSError, csv.Error, UnicodeDecodeError) as e:
                warnings.warn(f"skipping unreadable table {p}: {e}")
                continue
            if rows and not {"group_id", "flux_image", "obsids" if name == "final_images.csv" else "obsid"} <= set(rows[0]):
                warnings.warn(f"skipping malformed table {p}")
                continue
            for r in rows:
                r = dict(r)
                flux = r.get("flux_image", "")
                if not flux or not os.path.exists(os.path.join(run, flux)):
                    warnings.warn(f"{prefix}: flux image of {r.get('group_id')} {r.get('obsid', '')} is missing, row skipped")
                    continue
                for c in PATH_COLUMNS:
                    v = r.get(c, "")
                    if not v:
                        r[c] = ""; r[c + "_abs"] = ""
                        continue
                    vals = v.split(";")
                    # an empty entry in a ';' list (a member without that file) stays empty
                    absv = [os.path.normpath(os.path.join(run, x)) if x else "" for x in vals]
                    r[c] = ";".join(rel(x, root) if x else "" for x in absv)
                    r[c + "_abs"] = ";".join(absv)
                r["_run"] = prefix
                sink.append(r)
    # shared photons over every run under the root
    ims = [dict(key=f'{r["_run"]}/{r["group_id"]}', obsids=r["obsids"].split(";"), chips=r.get("chips", ""), band=r.get("band", ""))
           for r in finals]
    m = shares_map(ims, key=lambda d: d["key"])
    for r in finals:
        r["shares_photons_with"] = m.get(f'{r["_run"]}/{r["group_id"]}', "")
    mims = [dict(key=f'{r["_run"]}/{r["group_id"]}:{r["obsid"]}', obsids=[r["obsid"]], chips=r.get("chips", ""),
                 band=r.get("band", ""), group=f'{r["_run"]}/{r["group_id"]}') for r in members]
    mm = shares_map(mims, key=lambda d: d["key"], own_group=lambda d: d["group"])
    for r in members:
        r["shares_photons_with"] = mm.get(f'{r["_run"]}/{r["group_id"]}:{r["obsid"]}', "")
    write_csv_atomic(os.path.join(root, INDEX_FINAL), index_columns(FINAL_COLUMNS), finals)
    write_csv_atomic(os.path.join(root, INDEX_MEMBERS), index_columns(MEMBER_COLUMNS), members)
    return len(finals), len(members)


def load_index(path):
    """read an index, recompute absolute paths from where the index is now, add 'exists'.
    warns once when the stored absolute paths are stale (root moved)."""
    path = os.path.abspath(path)
    root = os.path.dirname(path)
    rows = read_csv(path)
    stale = False
    for r in rows:
        for c in PATH_COLUMNS:
            v = r.get(c, "")
            if not v:
                continue
            new = ";".join(os.path.normpath(os.path.join(root, x)) if x else "" for x in v.split(";"))
            if r.get(c + "_abs") and r[c + "_abs"] != new:
                stale = True
            r[c + "_abs"] = new
        f = r.get("flux_image_abs", "")
        r["exists"] = bool(f) and os.path.exists(f)
    if stale:
        warnings.warn(f"index absolute paths are stale; run rebuild_index('{root}')")
    return rows
