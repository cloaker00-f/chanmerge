"""Parsing and validation of everything the user types. Pure functions, no prompts."""
import re
from datetime import datetime, timezone

from .plan import (STAGES, CHIP_IDS, CHIP_NAMES, ChipChoice, ChipSet, ACIS_I_CHIPS, ACIS_S_CHIPS,
                   BI_CHIPS, FI_CHIPS)


class InputError(ValueError):
    pass


# ---------- coordinates ----------
_SEXA = re.compile(r"^\s*([+-]?)\s*(\d+(?:\.\d*)?)[\s:hd]+(\d+(?:\.\d*)?)[\s:m']+(\d+(?:\.\d*)?)\s*[s\"]?\s*$", re.I)


def _sexa(text):
    m = _SEXA.match(text)
    if not m:
        return None
    sign = -1.0 if m.group(1) == "-" else 1.0
    a, b, c = float(m.group(2)), float(m.group(3)), float(m.group(4))
    if b >= 60 or c >= 60:
        raise InputError("minutes and seconds must be below 60")
    return sign, a, b, c


def parse_ra(text):
    """ra in degrees. plain decimal (or with a d suffix) = degrees; hh:mm:ss = hours."""
    t = text.strip()
    if not t:
        raise InputError("RA is empty")
    m = re.fullmatch(r"([+-]?\d+(?:\.\d*)?|[+-]?\.\d+)\s*d?", t, re.I)
    if m:
        ra = float(m.group(1))
    else:
        s = _sexa(t)
        if s is None:
            raise InputError(f"cannot read RA '{t}'; use hh:mm:ss (e.g. 13:25:27.6) or decimal degrees (e.g. 201.365)")
        sign, h, mi, se = s
        if sign < 0:
            raise InputError("RA cannot be negative")
        ra = (h + mi / 60 + se / 3600) * 15.0
    if not 0 <= ra < 360:
        raise InputError("RA must be within 0-360 degrees (0-24 h)")
    return ra


def parse_dec(text):
    t = text.strip()
    if not t:
        raise InputError("Dec is empty")
    m = re.fullmatch(r"([+-]?\d+(?:\.\d*)?|[+-]?\.\d+)\s*d?", t, re.I)
    if m:
        dec = float(m.group(1))
    else:
        s = _sexa(t)
        if s is None:
            raise InputError(f"cannot read Dec '{t}'; use dd:mm:ss (e.g. -43:01:09) or decimal degrees (e.g. -43.019)")
        sign, d, mi, se = s
        dec = sign * (d + mi / 60 + se / 3600)
    if not -90 <= dec <= 90:
        raise InputError("Dec must be within -90 to +90 degrees")
    return dec


def format_ra(deg):
    h = deg / 15.0
    hh = int(h); mm = int((h - hh) * 60); ss = ((h - hh) * 60 - mm) * 60
    if round(ss, 1) >= 60:
        ss = 0; mm += 1
    if mm >= 60:
        mm -= 60; hh = (hh + 1) % 24
    return f"{hh:02d}:{mm:02d}:{ss:04.1f}"


def format_dec(deg):
    sign = "-" if deg < 0 else "+"
    a = abs(deg)
    dd = int(a); mm = int((a - dd) * 60); ss = ((a - dd) * 60 - mm) * 60
    if round(ss) >= 60:
        ss = 0; mm += 1
    if mm >= 60:
        mm -= 60; dd += 1
    return f"{sign}{dd:02d}:{mm:02d}:{ss:02.0f}"


def parse_radius(text):
    try:
        r = float(text.strip())
    except ValueError:
        raise InputError("radius must be a number in arcmin, e.g. 5")
    if not r > 0:
        raise InputError("radius must be above 0 arcmin")
    if r > 120:
        raise InputError("radius above 120 arcmin is not supported")
    return r


def default_target(ra, dec):
    # from the parsed coordinates, never from what was typed
    return f"obs_{ra:.4f}_{dec:+.4f}"


# ---------- dates ----------
def parse_date(text):
    t = text.strip()
    if not t:
        return None
    try:
        d = datetime.strptime(t, "%Y-%m-%d")
    except ValueError:
        raise InputError(f"'{t}' is not a date; use YYYY-MM-DD, e.g. 2010-12-31")
    return d.replace(tzinfo=timezone.utc)


def check_date_order(d_from, d_to):
    if d_from and d_to and d_from > d_to:
        raise InputError("the start date is after the end date")


# ---------- path characters ----------
# from the path tests: these break merge_obs or chandra_repro ('=' found in test i: merge_obs says 'Arguments not ok',
# in the root path and in the output folder name alike; ) ] < > found in test o: chandra_repro fails with them in the
# root path, merge_obs / dmcopy fail with them in the output folder name; * ? found in test o: merge_obs expands them itself and fails
# when a sibling folder fits the pattern and holds the same observation; $ ` | found in test q: tgdetect2 runs 'dmlist <path>' through a shell without
# quoting and fails on them)
FORBIDDEN = set(" ,;([\\@&'\"=)]<>*?$`|")
# chandra_repro, merge_obs and the acis / grating tools passed these one by one in tests o and q: warn, ask
WARN = set("^{}")
ALLOWED_PUNCT = set("-_.+~%:#!")


def char_name(c):
    return {" ": "a space", "\\": "a backslash", "'": "a single quote", '"': "a double quote"}.get(c, f"'{c}'")


def check_folder_name(name):
    """returns (level, message): ('ok',''), ('error', msg) or ('warn', msg)"""
    if name in (".", ".."):
        return "error", f"'{name}' is not a folder name"
    if "/" in name:
        return "error", "the folder name contains '/'; give a plain folder name, not a path"
    bad = sorted({c for c in name if c in FORBIDDEN}, key=name.index)
    if bad:
        return "error", (f"the folder name contains {', '.join(char_name(c) for c in bad)}; CIAO tools fail on it.\n"
                         f"      Not allowed: space , ; ( ) [ ] \\ @ & ' \" = < > * ? $ ` |")
    odd = sorted({c for c in name if c in WARN or not (c.isalnum() or c in ALLOWED_PUNCT)}, key=name.index)
    if odd:
        return "warn", (f"the folder name contains {', '.join(char_name(c) for c in odd)}; merge_obs handled these in tests, "
                        f"chandra_repro, merge_obs and the acis / grating tools handled these in tests, other CIAO tools were not tried")
    return "ok", ""


def check_run_path(path):
    """the folder chanmerge runs from: forbidden characters cannot be fixed from inside -> error"""
    bad = sorted({c for c in path if c in FORBIDDEN}, key=path.index)
    if bad:
        return "error", (f"the current folder path contains {', '.join(char_name(c) for c in bad)}; CIAO tools fail on it. "
                         f"Move to a folder without space , ; ( ) [ ] \\ @ & ' \" = < > * ? $ ` | and start again.")
    odd = sorted({c for c in path if c != "/" and (c in WARN or not (c.isalnum() or c in ALLOWED_PUNCT))}, key=path.index)
    if odd:
        return "warn", f"the current folder path contains {', '.join(char_name(c) for c in odd)}; not every CIAO tool was tested with it"
    return "ok", ""


# ---------- small answers ----------
def parse_yes_no(text):
    t = text.strip().lower()
    if t in ("y", "yes"):
        return True
    if t in ("n", "no"):
        return False
    raise InputError("answer y or n")


def parse_stage(text, allow_flux=True):
    t = text.strip().lower()
    allowed = STAGES if allow_flux else STAGES[:3]
    if t not in allowed:
        raise InputError("choose one of: " + ", ".join(allowed))
    return t


def parse_window(text):
    t = text.strip()
    if t == "":
        return 0.0
    try:
        w = float(t)
    except ValueError:
        raise InputError("the time window is a number of days, e.g. 3 or 0")
    if w < 0:
        raise InputError("the time window cannot be negative")
    return w


def parse_arm_method(text):
    t = text.strip()
    if t not in ("1", "2", "3", "4"):
        raise InputError("choose 1, 2, 3 or 4 (there is no default)")
    return int(t)


def parse_stage_edit(text, obsids):
    """'105 flux', '103 104 none', 'done' -> 'done' or list of (obsid, stage)"""
    t = text.strip().lower()
    if t == "done":
        return "done"
    parts = t.split()
    if len(parts) < 2:
        raise InputError("write obsids then a stage, e.g. 105 flux or 103 104 none, or done")
    stage = parse_stage(parts[-1])
    out = []
    for p in parts[:-1]:
        o = norm_obsid(p)
        if o not in obsids:
            raise InputError(f"{p} is not in the list")
        out.append((o, stage))
    return out


def norm_obsid(text):
    try:
        return str(int(str(text).strip()))
    except ValueError:
        raise InputError(f"'{text}' is not an obsid")


# ---------- row sets ----------
def parse_row_sets(text, rows):
    """'separate', '1+3, 2', '1+2, rest' -> list of row lists; every row exactly once"""
    t = text.strip().lower()
    rows = list(rows)
    if t == "separate":
        return [[r] for r in rows]
    if not t:
        raise InputError("write the row sets, e.g. separate, 1+3, 2 or 1+2, rest")
    sets, used, rest = [], [], False
    for chunk in [c.strip() for c in t.split(",")]:
        if chunk == "":
            raise InputError("empty set between commas")
        if chunk == "rest":
            if rest:
                raise InputError("'rest' given twice")
            rest = True
            continue
        cur = []
        for piece in chunk.split("+"):
            piece = piece.strip()
            if not piece.isdigit():
                raise InputError(f"'{piece}' is not a row number")
            cur.append(int(piece))
        sets.append(cur)
        used += cur
    unknown = sorted({r for r in used if r not in rows})
    if unknown:
        raise InputError(f"{_rows(unknown)} not in the table")
    rep = sorted({r for r in used if used.count(r) > 1})
    if rep:
        raise InputError(f"{_rows(rep)} written more than once; write every row once")
    missing = [r for r in rows if r not in used]
    if missing and rest:
        sets += [[r] for r in missing]
    elif missing:
        raise InputError(f"{_rows(missing)} missing; write every row once (or use rest)")
    elif rest:
        pass  # rest with nothing left is harmless
    # keep the user's order but sort rows inside a set
    return [sorted(s) for s in sets]


def _rows(rs):
    return (f"row {rs[0]} is" if len(rs) == 1 else f"rows {', '.join(map(str, rs))} are")


def format_row_sets(sets):
    return ", ".join("+".join(str(r) for r in s) for s in sets)


# ---------- chips ----------
CHIP_MENU = [
    ("all", "every active chip"),
    ("i", "ACIS-I chips only (I0-I3)"),
    ("s", "ACIS-S chips only (S0-S5)"),
    ("bi", "back-illuminated chips only (S1, S3)"),
    ("fi", "front-illuminated chips only (S0, S2, S4, S5, I0-I3)"),
    ("separate", "one image for bi, one image for fi (g01_bi, g01_fi)"),
    ("custom", "chip names, e.g. S3 or I2,I3; / for separate images: S3 / S2,S4"),
]


def parse_chip_menu(text):
    """number or word -> mode name; custom needs parse_chip_names afterwards"""
    t = text.strip().lower()
    if t.isdigit() and 1 <= int(t) <= len(CHIP_MENU):
        return CHIP_MENU[int(t) - 1][0]
    names = [m for m, _ in CHIP_MENU]
    if t in names:
        return t
    raise InputError("choose 1-7 or one of: " + ", ".join(names) + " (there is no default)")


def chip_choice(mode, custom_text=None):
    if mode == "all":
        return ChipChoice("all", [ChipSet(None)])
    if mode == "i":
        return ChipChoice("i", [ChipSet(ACIS_I_CHIPS)])
    if mode == "s":
        return ChipChoice("s", [ChipSet(ACIS_S_CHIPS)])
    if mode == "bi":
        return ChipChoice("bi", [ChipSet(BI_CHIPS)])
    if mode == "fi":
        return ChipChoice("fi", [ChipSet(FI_CHIPS)])
    if mode == "separate":
        return ChipChoice("separate", [ChipSet(BI_CHIPS, "bi"), ChipSet(FI_CHIPS, "fi")])
    if mode == "custom":
        return ChipChoice("custom", parse_chip_names(custom_text or ""))
    raise InputError(f"unknown chip mode {mode}")


def parse_chip_names(text):
    """'S3 / S1,S3' -> chip sets; labels like S3, S1_S3 when there are several sets"""
    t = text.strip()
    if not t:
        raise InputError("write chip names, e.g. S3 or I2,I3, or S3 / S2,S4 for separate images")
    sets = []
    for part in t.split("/"):
        names = [n.strip().upper() for n in re.split(r"[,\s]+", part) if n.strip()]
        if not names:
            raise InputError("an empty chip set between '/'")
        ids = []
        for n in names:
            if n not in CHIP_IDS:
                raise InputError(f"'{n}' is not a chip; use I0-I3 and S0-S5")
            if CHIP_IDS[n] not in ids:
                ids.append(CHIP_IDS[n])
        sets.append(tuple(sorted(ids)))
    if len(set(sets)) != len(sets):
        raise InputError("the same chip set is written twice")
    many = len(sets) > 1
    return [ChipSet(s, "_".join(CHIP_NAMES[c] for c in s) if many else "") for s in sets]


def chip_overlaps(sets):
    """chip names that appear in more than one set"""
    seen, twice = set(), set()
    for s in sets:
        if s.chips is None:
            continue
        for c in s.chips:
            (twice if c in seen else seen).add(c)
    return [CHIP_NAMES[c] for c in sorted(twice)]


# ---------- streak ----------
def parse_streak_sets(text, letters):
    """blank = none, all, or letters 'A' / 'A,C' / 'A C'"""
    t = text.strip().upper()
    if t == "":
        return []
    if t == "ALL":
        return list(letters)
    picked = [x for x in re.split(r"[,\s]+", t) if x]
    bad = [x for x in picked if x not in letters]
    if bad:
        raise InputError(f"{', '.join(bad)} is not a set that can be corrected; choose from {', '.join(letters)} (blank = none)")
    out = []
    for x in picked:
        if x not in out:
            out.append(x)
    return out


def parse_streak_action(text):
    t = text.strip().lower()
    if t not in ("acisreadcorr", "fill", "mask"):
        raise InputError("choose acisreadcorr, fill or mask (leave = blank answer to the set question)")
    return t


# ---------- group edits ----------
def parse_group_edit(text):
    """move 102 g03 | move 102 s | add 102 g03 | new 101 102 | drop 102 [g03] | drop all g03 |
    delete g03 | g01 chips I3 | done"""
    t = text.strip()
    low = t.lower()
    p = low.split()
    if low == "done":
        return ("done",)
    if not p:
        raise InputError("write an edit, or done")
    if p[0] == "move" and len(p) == 3:
        return ("move", norm_obsid(p[1]), p[2])
    if p[0] == "add" and len(p) == 3:
        return ("add", norm_obsid(p[1]), p[2])
    if p[0] == "new" and len(p) >= 2:
        return ("new", [norm_obsid(x) for x in p[1:]])
    if p[0] == "drop" and len(p) == 3 and p[1] == "all":
        return ("drop_all", p[2])
    if p[0] == "drop" and len(p) in (2, 3):
        return ("drop", norm_obsid(p[1]), p[2] if len(p) == 3 else None)
    if p[0] == "delete" and len(p) == 2:
        return ("delete", p[1])
    if len(p) >= 3 and p[1] == "chips":
        rest = t.split(None, 2)[2]
        return ("chips", p[0], rest)
    raise InputError("unknown edit; e.g. move 102 g03, move 102 s, add 102 g03, new 101 102, "
                     "drop 102 g03, drop all g03, delete g03, g01 chips I3, done")
