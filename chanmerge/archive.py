"""Archive side: HEASARC chanmaster list, Chandra Ocat details, mode/window/chip rules, fov1 fetch, sizes.

every network call goes through _get/_head so tests can replace them. nothing here prompts.
"""
import glob
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from .plan import Category, Observation

HEASARC_COLUMNS = "obsid,status,name,ra,dec,time,detector,grating,exposure,type,pi,public_date,data_mode"
OCAT_URL = "https://cda.harvard.edu/srservices/ocatDetails.do?format=text&obsid={ids}"
ARCHIVE_URL = "https://cxc.cfa.harvard.edu/cdaftp/byobsid/{last}/{obsid}/"
USER_AGENT = "chanmerge/2.0 (python urllib)"
OCAT_CHUNK = 500
MJD0 = datetime(1858, 11, 17, tzinfo=timezone.utc)


class ArchiveError(RuntimeError):
    pass


class SizeLookupStopped(RuntimeError):
    pass


# ---------- tiny http layer (replaceable in tests) ----------
def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _head(url, timeout=30):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(r.headers.get("Content-Length") or 0)


HTTP = {"get": _get, "head": _head}


def _decode(v):
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    if hasattr(v, "mask") and getattr(v, "mask", False) is True:
        return None
    s = str(v).strip()
    return None if s in ("", "--", "nan", "None") else s


def mjd_to_datetime(mjd):
    return MJD0 + timedelta(days=float(mjd))


def norm_obsid(v):
    return str(int(float(str(v).strip())))


# ---------- heasarc ----------
HEASARC_TIMEOUT = 120   # seconds per try
HEASARC_TRIES = 2


def _heasarc_once(ra, dec, radius_arcmin):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.heasarc import HeasarcClass
    pos = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    # a fresh client per try, so a hung try does not share its connection with the next one
    return HeasarcClass().query_region(pos, catalog="chanmaster", radius=radius_arcmin * u.arcmin, columns=HEASARC_COLUMNS)


def query_heasarc(ra, dec, radius_arcmin, timeout=None, tries=None):
    """cone search of chanmaster; returns a list of dicts. raises ArchiveError with a readable reason.
    astroquery's heasarc goes through pyvo's tap service, which has no timeout: a server that keeps the connection
    open hangs the search forever (seen 2026-11-04, test d rerun, >10 min). each try therefore runs in a worker
    thread with a deadline; a hung try is left behind (daemon thread) and the search is tried once more."""
    import threading
    timeout = HEASARC_TIMEOUT if timeout is None else timeout
    tries = HEASARC_TRIES if tries is None else tries
    last = ""
    for _ in range(max(tries, 1)):
        box = {}

        def work():
            try:
                box["tab"] = _heasarc_once(ra, dec, radius_arcmin)
            except Exception as e:  # network, service, parse: all end up here
                box["err"] = e
        t = threading.Thread(target=work, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            last = f"no answer within {timeout:.0f} s"
            continue
        if "err" in box:
            e = box["err"]
            raise ArchiveError(f"HEASARC query failed: {type(e).__name__}: {e}")
        return rows_from_table(box["tab"])
    raise ArchiveError(f"HEASARC query failed: {last} ({tries} tries); the service may be slow, try again later")


def rows_from_table(tab):
    rows = []
    names = list(getattr(tab, "colnames", []))
    for r in tab:
        d = {}
        for n in names:
            v = r[n]
            if hasattr(v, "mask") and bool(getattr(v, "mask", False)):
                d[n.lower()] = None
            else:
                d[n.lower()] = _decode(v)
        rows.append(d)
    return rows


def keep_archived_public(rows, now=None):
    """archived and public only; returns (kept, dropped_count)"""
    now = now or datetime.now(timezone.utc)
    kept = []
    for r in rows:
        st = (r.get("status") or "").lower()
        if st != "archived":
            continue
        pd = r.get("public_date")
        if pd is not None:
            try:
                if mjd_to_datetime(float(pd)) > now:
                    continue
            except ValueError:
                pass
        kept.append(r)
    return kept, len(rows) - len(kept)


# ---------- ocat ----------
def parse_ocat_text(text):
    """ocatDetails.do?format=text: drop blank and # lines; first line header, skip the next (widths); tab split.
    rows come sorted by seqNum, so key by OBSID."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:
        return {}
    head = [h.strip() for h in lines[0].split("\t")]
    out = {}
    for ln in lines[2:]:
        vals = [v.strip() for v in ln.split("\t")]
        row = dict(zip(head, vals))
        oid = row.get("OBSID")
        if oid:
            try:
                out[norm_obsid(oid)] = row
            except ValueError:
                continue
    return out


def fetch_ocat(obsids, timeout=60):
    """details for many obsids, about 500 per request. raises ArchiveError on any failure (the caller warns)."""
    out = {}
    ids = list(obsids)
    for i in range(0, len(ids), OCAT_CHUNK):
        chunk = ",".join(ids[i:i + OCAT_CHUNK])
        try:
            txt = HTTP["get"](OCAT_URL.format(ids=chunk), timeout=timeout).decode("utf-8", "replace")
        except Exception as e:
            raise ArchiveError(f"Chandra Ocat unreachable: {type(e).__name__}: {e}")
        out.update(parse_ocat_text(txt))
    return out


# ---------- rules ----------
def mode_from_ocat(row):
    if not row:
        return None
    mode = (row.get("MODE") or "").upper()
    dm = (row.get("DATAMODE") or "").upper()
    # cc rule from the plan; empty mode (hrc, calibration 5xxxx) is not cc
    if mode in ("CC", "CONTINUOUS") or dm.startswith("CC"):
        return "CC"
    if mode in ("TE", "TIMED"):
        return "TE"
    if dm:
        return "TE"   # faint, vfaint, graded without a cc prefix are timed exposure
    return None


def mode_from_heasarc(data_mode):
    dm = (data_mode or "").upper()
    if dm.startswith("CC"):
        return "CC"
    if dm.startswith("TE"):
        return "TE"
    return None


def resolve_mode(detector, ocat_row, heasarc_mode, ocat_ok=True):
    """(mode, source). hrc has no mode. one source only -> that one. both and they differ, or none -> '?'"""
    if detector.startswith("HRC"):
        return "-", "hrc"
    o = mode_from_ocat(ocat_row) if ocat_ok else None
    h = mode_from_heasarc(heasarc_mode)
    if o and h:
        return (o, "ocat+heasarc") if o == h else ("?", "disagree")
    if o:
        return o, "ocat"
    if h:
        return h, "heasarc"
    return "?", "none"


def window_from_ocat(row):
    if not row:
        return None
    sub = (row.get("SUBARY") or "").upper()
    rc = row.get("ROW_CNT") or ""
    try:
        # ocat gives ROW_CNT 0 for full frame; only 1-1023 rows is a subarray
        n = int(float(rc)) if rc else 0
        if 0 < n < 1024:
            return "sub"
    except ValueError:
        pass
    if sub in ("", "NONE", "N", "NO"):
        return "full"
    return "sub"


def chips_from_readout(text):
    """READOUT_DETECTOR like ACIS-012367 -> (0, 1, 2, 3, 6, 7); hrc or empty -> None"""
    m = re.fullmatch(r"\s*ACIS-(\d+)\s*", text or "")
    if not m:
        return None
    return tuple(sorted({int(c) for c in m.group(1)}))


def detector_of(hrow, orow):
    d = (hrow.get("detector") or "").upper().replace("_", "-")
    if d in ("ACIS-I", "ACIS-S", "HRC-I", "HRC-S"):
        return d
    inst = (orow or {}).get("INSTR", "").upper().replace("_", "-")
    if inst in ("ACIS-I", "ACIS-S", "HRC-I", "HRC-S"):
        return inst
    return d or "?"


def grating_of(hrow, orow):
    g = (hrow.get("grating") or (orow or {}).get("GRAT") or "NONE").upper()
    return g if g in ("HETG", "LETG") else "NONE"


def make_observations(heasarc_rows, ocat, ocat_ok=True):
    """build Observation objects; each obsid once (dedupe in order)"""
    out = {}
    for r in heasarc_rows:
        try:
            oid = norm_obsid(r["obsid"])
        except (KeyError, ValueError, TypeError):
            continue
        if oid in out:
            continue
        orow = ocat.get(oid) if ocat_ok else None
        det = detector_of(r, orow)
        grat = grating_of(r, orow)
        mode, src = resolve_mode(det, orow, r.get("data_mode"), ocat_ok)
        if det.startswith("HRC"):
            win = "-"
        else:
            win = window_from_ocat(orow) if orow else None
            win = win or "?"
        start = mjd_to_datetime(r["time"]) if r.get("time") else MJD0
        exp = float(r.get("exposure") or 0.0)
        chips = chips_from_readout((orow or {}).get("READOUT_DETECTOR", "")) if det.startswith("ACIS") else None
        roll = None
        try:
            roll = float((orow or {}).get("SOE_ROLL") or "nan")
        except ValueError:
            pass
        tra = tdec = None
        try:
            tra, tdec = float(r.get("ra")), float(r.get("dec"))
        except (TypeError, ValueError):
            pass
        rows = None
        try:
            n = int(float((orow or {}).get("ROW_CNT") or 0))
            rows = n if 0 < n < 1024 else None
        except ValueError:
            pass
        out[oid] = Observation(oid, Category(det, grat, mode, win), start, exp, target_name=r.get("name") or "",
                               mode_source=src, active_chips=chips, roll=roll, target_ra=tra, target_dec=tdec,
                               subarray_rows=rows)
    return list(out.values())


def boundary_observations(observations, d_from, d_to):
    """membership is by start; return (crossing the start edge, crossing the end edge) obsids.
    start-edge crossers start before d_from and end after it; end-edge crossers start inside but end after d_to+1d."""
    start_cross, end_cross = [], []
    for o in observations:
        if d_from and o.start < d_from <= o.end:
            start_cross.append(o.obsid)
        if d_to and o.start < d_to + timedelta(days=1) < o.end:
            end_cross.append(o.obsid)
    return start_cross, end_cross


def in_date_range(o, d_from, d_to):
    if d_from and o.start < d_from:
        return False
    if d_to and o.start >= d_to + timedelta(days=1):
        return False
    return True


# ---------- archive files: fov1 and sizes ----------
def archive_url(obsid, sub=""):
    o = str(int(obsid))
    return ARCHIVE_URL.format(last=o[-1], obsid=o) + sub


def list_dir(url):
    """apache listing -> (files, folders) as names"""
    html = HTTP["get"](url).decode("utf-8", "replace")
    hrefs = re.findall(r'href="([^"?#]+)"', html)
    files, dirs = [], []
    for h in hrefs:
        if h.startswith(("/", "http", "..")):
            continue
        (dirs if h.endswith("/") else files).append(h)
    return sorted(set(files)), sorted(set(dirs))


def fetch_fov(obsid, dest_primary):
    """fetch only the fov1 file of one observation into dest_primary; returns the path or None.
    any fov1 already there (.fits or .fits.gz, file or link) is used without touching the network: the old check
    only knew the archive's exact name, so a decompressed fov1 caused a download (found in test d)."""
    have = sorted(glob.glob(os.path.join(glob.escape(dest_primary), "*fov1.fits*")))
    have = [h for h in have if not h.endswith(".part") and os.path.getsize(h) > 0]
    if have:
        return have[0]
    files, _ = list_dir(archive_url(obsid, "primary/"))
    names = [f for f in files if re.search(r"fov1\.fits(\.gz)?$", f)]
    if not names:
        return None
    os.makedirs(dest_primary, exist_ok=True)
    path = os.path.join(dest_primary, names[0])
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    data = HTTP["get"](archive_url(obsid, "primary/" + names[0]))
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def fetch_fovs(dest_by_obsid, threads=4):
    """dest_by_obsid: obsid -> primary folder. returns obsid -> path or None (failure never raises)"""
    def one(item):
        oid, dest = item
        try:
            return oid, fetch_fov(oid, dest)
        except Exception:
            return oid, None
    with ThreadPoolExecutor(threads) as ex:
        return dict(ex.map(one, list(dest_by_obsid.items())))


EVT2_ARCHIVE_RE = re.compile(r"/primary/[^/]*evt2\.fits(\.gz)?$")


def archive_size(obsid, threads=4, detail=None):
    """bytes of one observation in the archive: listing + HEAD Content-Length of every file, 4 at a time.
    detail (a dict) also gets 'evt2' = size of the archive's primary evt2 file (compressed)"""
    # walk every level (primary/responses/, secondary/aspect/, secondary/ephem/ ...), at most 4 deep
    urls, todo = [], [("", 0)]
    while todo:
        sub, depth = todo.pop()
        files, dirs = list_dir(archive_url(obsid, sub))
        urls += [archive_url(obsid, sub + f) for f in files]
        if depth < 4:
            todo += [(sub + d, depth + 1) for d in dirs]

    def head(u):
        try:
            return HTTP["head"](u)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                raise SizeLookupStopped(f"archive answered {e.code}")
            raise
    with ThreadPoolExecutor(threads) as ex:
        sizes = list(ex.map(head, urls))
    if detail is not None:
        detail["evt2"] = sum(s for u, s in zip(urls, sizes) if EVT2_ARCHIVE_RE.search(u))
    return sum(sizes)


def approx_size(exposure_s):
    """old formula, labelled approximate: 50 MB + 2 MB per ks"""
    return int((50.0 + 2.0 * exposure_s / 1000.0) * 1e6)


def download_sizes(observations, pause=0.3, max_errors=3, evt_sizes=None):
    """real sizes, one observation at a time with a short pause. on 429/503 or repeated errors stop and
    fall back to the approximate formula for the rest. returns (sizes by obsid, exact: bool).
    evt_sizes (a dict) gets each observation's archive evt2 size where the lookup worked"""
    sizes, errors, exact = {}, 0, True
    stopped = False
    for o in observations:
        if not stopped:
            try:
                d = {}
                sizes[o.obsid] = archive_size(o.obsid, detail=d)
                if evt_sizes is not None and d.get("evt2"):
                    evt_sizes[o.obsid] = d["evt2"]
                time.sleep(pause)
                continue
            except SizeLookupStopped:
                stopped = True
            except Exception:
                errors += 1
                if errors >= max_errors:
                    stopped = True
        sizes[o.obsid] = approx_size(o.exposure_s)
        exact = False
    return sizes, exact


FOOTPRINT_FACTOR = 2.2   # disk after repro / archive, worst of the five measured (1.53-2.16)
EVT2_UNPACK = 2.2        # repro evt2 / archive evt2.gz, measured 2.09-2.18 on 6 observations (2026-11-04)
EVT2_OF_ARCHIVE = 0.16   # repro evt2 / whole archive when the evt2 size is unknown, worst of 6 (0.003-0.16)
IMAGE_FILE = 1.5e6       # bytes per image file in a run folder (v404 full frame: 0.7-1.3 MB)


def repro_evt_bytes(obs, root, archive_sizes, archive_evt2):
    """expected repro evt2 size per obsid: the file itself when already reprocessed, else from the archive"""
    from .ciao import REPRO_MARKER
    out = {}
    for o in obs:
        repro = os.path.join(root, "observations", o.folder, "repro")
        found = [f for f in (os.listdir(repro) if os.path.isdir(repro) else []) if f.endswith("_repro_evt2.fits")]
        if found and os.path.exists(os.path.join(repro, REPRO_MARKER)):
            out[o.obsid] = os.path.getsize(os.path.join(repro, found[0]))
        elif o.obsid in archive_evt2:
            out[o.obsid] = archive_evt2[o.obsid] * EVT2_UNPACK
        else:
            out[o.obsid] = archive_sizes.get(o.obsid, approx_size(o.exposure_s)) * EVT2_OF_ARCHIVE
    return out


_CHIP_SHARE_CACHE = {}


def chip_shares_on_disk(path):
    """share of the events on each ccd_id in an evt2 file (read once; only the ccd_id column)"""
    if path not in _CHIP_SHARE_CACHE:
        from astropy.io import fits
        import numpy as np
        with fits.open(path, memmap=True) as h:
            ids, n = np.unique(np.asarray(h["EVENTS"].data["ccd_id"]), return_counts=True)
        tot = max(int(n.sum()), 1)
        _CHIP_SHARE_CACHE[path] = {int(i): int(c) / tot for i, c in zip(ids, n)}
    return _CHIP_SHARE_CACHE[path]


def chip_fraction(ob, chips, root):
    """share of an observation's events that a chip set keeps (the disk estimate is per chip set since test d:
    'bi' of 950 holds 5% of its events, the old estimate counted 100%).
    reprocessed on disk: exact, from the event file. not yet downloaded: chosen active chips / active chips, and 1
    when the set holds the chip under the observation's target (most events are there), so it stays an upper bound"""
    if chips is None:
        return 1.0
    from .ciao import REPRO_MARKER
    repro = os.path.join(root, "observations", ob.folder, "repro")
    found = sorted(glob.glob(os.path.join(glob.escape(repro), "*_repro_evt2.fits")))
    if found and os.path.exists(os.path.join(repro, REPRO_MARKER)):
        try:
            sh = chip_shares_on_disk(found[0])
            return float(sum(sh.get(c, 0.0) for c in chips))
        except (OSError, KeyError, ValueError):
            return 1.0
    active = set(ob.active_chips or ())
    if not active:
        return 1.0
    if ob.source_chips and set(ob.source_chips) & set(chips):
        return 1.0
    return len(active & set(chips)) / len(active)


def run_footprint(plan, evt_bytes):
    """upper estimate of the run folders: every (group, band, chip set) gets reprojected member events and a merged
    list (2x the events the chip set keeps), corrected copies of both when a fill or mask may happen (+2x), and its
    images; an observation corrected by acisreadcorr adds one full copy, once per run (it is shared by every band
    and group, <target>/acisreadcorr/). measured: test b 374 mb per band folder; test d 789 mb (old estimate 3.5 gb,
    it ignored the chip filter)"""
    total = 0.0
    readcorr = set()
    for g in plan.groups:
        obs = [plan.observations[o] for o in g.obsids]
        rs = plan.row_set(g.row_set)
        streak = rs.streak_action if rs else "leave"
        arms = any(ob.category.is_grating and ob.category.mode in ("TE", "?")
                   and plan.arm_method_by_row.get(plan.row_of(ob.obsid)) in (2, 3, 4) for ob in obs)
        k = 2 + (2 if (arms or streak in ("fill", "mask")) else 0)
        if streak == "acisreadcorr":
            readcorr |= {ob.obsid for ob in obs if ob.category.is_acis}
        sets = (plan.chip_sets_for(g) or [None]) if all(ob.category.is_acis for ob in obs) else [None]
        images = IMAGE_FILE * (9 + 5 * len(obs))
        for cs in sets:
            chips = getattr(cs, "chips", None)
            ev = sum(evt_bytes.get(ob.obsid, 0) * chip_fraction(ob, chips, plan.root) for ob in obs)
            total += len(plan.bands_for(g)) * (k * ev + images)
    total += sum(evt_bytes.get(o, 0) for o in readcorr)
    return int(total)


def human(nbytes):
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.1f} GB"
    return f"{nbytes / 1e6:.0f} MB"
