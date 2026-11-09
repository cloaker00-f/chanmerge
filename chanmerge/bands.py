"""Energy bands: ciao names, lo:hi:eff for acis, PI bands for hrc, validation and normalization."""
import re
from .plan import Band, fmt_num

# ciao's own band table (merge_obs / fluximage), keV
ACIS_BANDS = {
    "broad": (0.5, 7.0, 2.3),
    "ultrasoft": (0.2, 0.5, 0.4),
    "soft": (0.5, 1.2, 0.92),
    "medium": (1.2, 2.0, 1.56),
    "hard": (2.0, 7.0, 3.8),
}
CSC = ("soft", "medium", "hard")
HRC_WIDE_EFF = 1.5          # keV, ciao 'wide' band for hrc
ACIS_RANGE = (0.1, 15.0)    # keV, anything outside is surely a typo
HRC_EFF_RANGE = (0.06, 10.0)
TOL = 1e-6


class BandError(ValueError):
    pass


def _num(s):
    try:
        return float(s)
    except ValueError:
        raise BandError(f"'{s}' is not a number")


def _same(a, b):
    return abs(a - b) <= TOL


def acis_band(token):
    """one acis band from a name or lo:hi:eff (keV). returns (Band, note or '')"""
    t = token.strip().lower()
    if not t:
        raise BandError("empty band")
    if t in ACIS_BANDS:
        lo, hi, eff = ACIS_BANDS[t]
        return Band("acis", t, lo, hi, eff, t, t, t), ""
    if t == "wide":
        raise BandError("'wide' is an HRC band; ACIS bands: broad, ultrasoft, soft, medium, hard, csc, or lo:hi:eff")
    if ":" not in t:
        raise BandError(f"unknown ACIS band '{token.strip()}'; use broad, ultrasoft, soft, medium, hard, csc, or lo:hi:eff in keV")
    parts = t.split(":")
    if len(parts) != 3 or any(p.strip() == "" for p in parts):
        raise BandError(f"'{token.strip()}' needs three numbers lo:hi:eff in keV, e.g. 0.5:7:2.3 (the exposure-map energy eff is required)")
    lo_s, hi_s, eff_s = (p.strip() for p in parts)
    lo, hi, eff = _num(lo_s), _num(hi_s), _num(eff_s)
    if not lo < hi:
        raise BandError(f"'{token.strip()}': lo must be below hi")
    if lo < ACIS_RANGE[0] or hi > ACIS_RANGE[1]:
        raise BandError(f"'{token.strip()}': ACIS energies must lie within {ACIS_RANGE[0]}-{ACIS_RANGE[1]} keV")
    if not lo <= eff <= hi:
        raise BandError(f"'{token.strip()}': the exposure-map energy {fmt_num(eff)} keV must lie inside {fmt_num(lo)}-{fmt_num(hi)} keV")
    # identical to a ciao band -> use its name, with a visible note
    for name, (a, b, c) in ACIS_BANDS.items():
        if _same(lo, a) and _same(hi, b) and _same(eff, c):
            note = (f"[!] note: band '{token.strip()}' is identical to ciao's '{name}' band\n"
                    f"      ({fmt_num(a)}-{fmt_num(b)} keV, expmap at {fmt_num(c)} keV) -> using '{name}' for folder and file names")
            return Band("acis", name, a, b, c, name, name, name), note
    spec = f"{lo_s}:{hi_s}:{eff_s}"
    folder = safe_label(spec)
    # merge_obs labels a numeric band by its literal lo-hi strings
    file_label = f"{lo_s}-{hi_s}"
    return Band("acis", spec, lo, hi, eff, spec, folder, file_label), ""


def hrc_band(token):
    """one hrc band: wide, pilo:pihi:energy (integer PI channels, keV) or ::energy"""
    t = token.strip().lower()
    if not t:
        raise BandError("empty band")
    if t == "wide":
        return Band("hrc", "wide", None, None, HRC_WIDE_EFF, "wide", "wide", "wide"), ""
    if t in ACIS_BANDS or t == "csc":
        raise BandError(f"'{t}' is an ACIS band; HRC bands: wide, pilo:pihi:energy (PI channels, keV) or ::energy")
    parts = t.split(":")
    if len(parts) != 3:
        raise BandError(f"'{token.strip()}' needs pilo:pihi:energy, e.g. 30:200:1.1, or ::energy")
    lo_s, hi_s, eff_s = (p.strip() for p in parts)
    if eff_s == "":
        raise BandError(f"'{token.strip()}': the exposure-map energy (keV) is required")
    eff = _num(eff_s)
    if not HRC_EFF_RANGE[0] <= eff <= HRC_EFF_RANGE[1]:
        raise BandError(f"'{token.strip()}': exposure-map energy must lie within {HRC_EFF_RANGE[0]}-{HRC_EFF_RANGE[1]} keV")
    if lo_s == "" and hi_s == "":
        spec = f"::{eff_s}"
        return Band("hrc", spec, None, None, eff, spec, safe_label("all:" + eff_s), "wide"), ""
    if lo_s == "" or hi_s == "":
        raise BandError(f"'{token.strip()}': give both PI limits or neither (::energy)")
    if not (re.fullmatch(r"\d+", lo_s) and re.fullmatch(r"\d+", hi_s)):
        raise BandError(f"'{token.strip()}': HRC PI limits are integer channels")
    lo, hi = int(lo_s), int(hi_s)
    if not lo < hi:
        raise BandError(f"'{token.strip()}': pilo must be below pihi")
    spec = f"{lo}:{hi}:{eff_s}"
    return Band("hrc", spec, float(lo), float(hi), eff, spec, safe_label(spec), f"{lo}-{hi}"), ""


def safe_label(spec):
    # folder-safe: 0.5:2:1 -> 0.5_2_1
    return re.sub(r"[^A-Za-z0-9.\-]+", "_", spec).strip("_")


def parse_bands(text, kind="acis"):
    """comma list -> (bands, notes). csc expands; duplicates run once with a note. raises BandError"""
    tokens = [t for t in (x.strip() for x in text.split(",")) if t]
    if not tokens:
        raise BandError("no band given")
    expanded = []
    for t in tokens:
        if kind == "acis" and t.lower() == "csc":
            expanded += [(c, "csc") for c in CSC]
        else:
            expanded.append((t, None))
    bands, notes, seen = [], [], {}
    for tok, origin in expanded:
        b, note = (acis_band(tok) if kind == "acis" else hrc_band(tok))
        if note:
            notes.append(note)
        if b.name in seen:
            why = " (csc already includes it)" if (origin == "csc" or seen[b.name] == "csc") else ""
            notes.append(f"[!] note: band '{b.name}' requested twice{why} -> running it once")
            continue
        seen[b.name] = origin
        bands.append(b)
    return bands, notes


def energy_range_kev(band_name):
    """(lo, hi) for overlap checks from a normalized band name; hrc PI bands return PI numbers; None = all"""
    n = band_name.strip().lower()
    if n in ACIS_BANDS:
        return ACIS_BANDS[n][:2]
    if n == "wide" or n.startswith("::"):
        return None
    parts = n.replace("_", ":").split(":")
    try:
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return None


def bands_overlap(a, b):
    ra, rb = energy_range_kev(a), energy_range_kev(b)
    if ra is None or rb is None:
        return True
    return ra[0] < rb[1] and rb[0] < ra[1]
