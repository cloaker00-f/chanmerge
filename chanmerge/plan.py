"""RunPlan: every choice of one chanmerge run, filled by the terminal flow or the gui.

check_plan() and run() only read it and never ask questions.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

STAGES = ("none", "download", "repro", "flux")
STAGE_RANK = {s: i for i, s in enumerate(STAGES)}

# ccd_id -> chip name, and the chip classes we measured (bi = back illuminated)
CHIP_NAMES = ["I0", "I1", "I2", "I3", "S0", "S1", "S2", "S3", "S4", "S5"]
CHIP_IDS = {n: i for i, n in enumerate(CHIP_NAMES)}
BI_CHIPS = (5, 7)
FI_CHIPS = (0, 1, 2, 3, 4, 6, 8, 9)
ACIS_I_CHIPS = (0, 1, 2, 3)
ACIS_S_CHIPS = (4, 5, 6, 7, 8, 9)

STREAK_ACTIONS = ("acisreadcorr", "fill", "mask", "leave")
ARM_METHODS = {1: "filter", 2: "mask_keep_source", 3: "mask_remove_source", 4: "mask_fill"}


def stage_at_least(stage, wanted):
    return STAGE_RANK[stage] >= STAGE_RANK[wanted]


@dataclass(frozen=True)
class Category:
    # one row of the category table; disjoint, so every observation has exactly one
    detector: str   # ACIS-I, ACIS-S, HRC-I, HRC-S
    grating: str    # NONE, HETG, LETG
    mode: str       # TE, CC, '-' for hrc, '?' unknown
    window: str     # full, sub, '-' for hrc, '?' unknown

    @property
    def is_acis(self):
        return self.detector.startswith("ACIS")

    @property
    def is_hrc(self):
        return self.detector.startswith("HRC")

    @property
    def is_grating(self):
        return self.grating in ("HETG", "LETG")

    @property
    def is_cc(self):
        return self.mode == "CC"

    def label(self):
        return f"{self.detector}/{self.grating}/{self.mode}/{self.window}"

    def sort_key(self):
        det = ["ACIS-I", "ACIS-S", "HRC-I", "HRC-S"]
        grat = ["NONE", "HETG", "LETG"]
        mode = ["TE", "?", "CC", "-"]
        win = ["full", "?", "sub", "-"]
        def idx(lst, v):
            return lst.index(v) if v in lst else len(lst)
        return (idx(det, self.detector), idx(grat, self.grating), idx(mode, self.mode), idx(win, self.window))


@dataclass
class Observation:
    obsid: str                       # str(int(obsid)), never padded
    category: Category
    start: datetime                  # utc, from heasarc (approximate until headers are read)
    exposure_s: float
    target_name: str = ""
    stage: str = "none"
    mode_source: str = ""            # ocat, heasarc, header, '?'
    active_chips: Optional[Tuple[int, ...]] = None   # from ocat READOUT_DETECTOR or fov1
    target_ra: Optional[float] = None                 # this observation's own target (heasarc ra, dec)
    target_dec: Optional[float] = None
    source_chips: Optional[Tuple[int, ...]] = None   # chips under this observation's target (fov1)
    source_near_edge: bool = False
    point_chips: Optional[Tuple[int, ...]] = None    # chips under the search point, () if it is off this field
    fov_file: Optional[str] = None
    roll: Optional[float] = None
    subarray_rows: Optional[int] = None   # ocat ROW_CNT when the window is a subarray (1-1023 rows)
    # filled after the downloads, from headers
    header_start: Optional[datetime] = None
    header_end: Optional[datetime] = None
    header_exposure_s: Optional[float] = None
    header_mode: Optional[str] = None      # TE / CC from the evt2 header (header check)
    header_window: Optional[str] = None    # full / sub from the evt2 header

    @property
    def end(self):
        # approximate end before download: start + exposure
        from datetime import timedelta
        if self.header_end is not None:
            return self.header_end
        return self.start + timedelta(seconds=self.exposure_s)

    @property
    def best_start(self):
        return self.header_start or self.start

    @property
    def folder(self):
        return f"{self.obsid}_g" if self.category.is_grating else self.obsid


@dataclass(frozen=True)
class ChipSet:
    # a set of ccd ids for one image; None = every active chip (no filter)
    chips: Optional[Tuple[int, ...]]
    label: str = ""                  # folder/file label when a group has several sets

    def names(self):
        if self.chips is None:
            return "all"
        return ",".join(CHIP_NAMES[c] for c in self.chips)

    def short(self):
        # compact text for screens: I0-I3 when it is the whole acis-i array
        if self.chips is None:
            return "all"
        if tuple(sorted(self.chips)) == ACIS_I_CHIPS:
            return "I0-I3"
        return ",".join(CHIP_NAMES[c] for c in self.chips)

    def ciao_filter(self):
        if self.chips is None:
            return ""
        return "[ccd_id=" + ",".join(str(c) for c in self.chips) + "]"


@dataclass
class ChipChoice:
    # what the user picked for one acis table row
    mode: str                        # all, i, s, bi, fi, separate, custom
    sets: List[ChipSet] = field(default_factory=list)

    def text(self):
        if self.mode == "custom":
            return " / ".join(s.names() for s in self.sets)
        return self.mode


@dataclass
class Band:
    kind: str                        # acis or hrc
    name: str                        # normalized: ciao name or 'lo:hi:eff'
    lo: Optional[float]              # keV (acis) or PI (hrc); None for hrc '::energy' / wide
    hi: Optional[float]
    eff: float                       # keV, exposure map energy
    merge_spec: str                  # what merge_obs gets in bands=
    folder_label: str                # safe for paths, e.g. broad or 0.5_2_1
    file_label: str                  # how merge_obs names its files, e.g. broad or 0.5-2

    def describe(self):
        if self.kind == "acis":
            return f"ACIS band {fmt_num(self.lo)}-{fmt_num(self.hi)} keV, exposure map at {fmt_num(self.eff)} keV"
        if self.lo is None:
            return f"HRC band {self.name}, exposure map at {fmt_num(self.eff)} keV"
        return f"HRC band PI {fmt_num(self.lo)}-{fmt_num(self.hi)}, exposure map at {fmt_num(self.eff)} keV"


def fmt_num(v):
    if v is None:
        return "?"
    s = f"{v:.6g}"
    return s


@dataclass
class RowSet:
    # rows of the category table that are grouped together, with its streak setting
    letter: str
    rows: List[int]                  # 1-based table row numbers
    streak_action: str = "leave"


@dataclass
class Group:
    group_id: str                    # g01, s103, all, all_B
    obsids: List[str]
    row_set: str                     # letter of the row set it came from
    manual: bool = False
    chip_sets: Optional[List[ChipSet]] = None   # None = from the rows' chip choice
    chips_note: str = ""             # set when rows disagree on chips
    left_out_of: str = ""            # rule 3 (session 007): this single was made for an observation that left that group's merge


@dataclass
class RunPlan:
    ra: float
    dec: float
    radius_arcmin: float
    target: str
    root: str                        # folder chanmerge runs from (absolute)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    observations: Dict[str, Observation] = field(default_factory=dict)
    categories: List[Category] = field(default_factory=list)   # table rows in order (row n = index n-1)
    acis_bands: List[Band] = field(default_factory=list)
    hrc_bands: List[Band] = field(default_factory=list)
    chips_by_row: Dict[int, ChipChoice] = field(default_factory=dict)
    arm_method_by_row: Dict[int, int] = field(default_factory=dict)
    window_days: float = 0.0
    row_sets: List[RowSet] = field(default_factory=list)
    groups: List[Group] = field(default_factory=list)
    overwrite: bool = False
    ocat_ok: bool = True

    # ---- helpers that only read the plan ----
    def row_of(self, obsid):
        cat = self.observations[obsid].category
        return self.categories.index(cat) + 1

    def flux_obsids(self):
        return [o for o, ob in self.observations.items() if ob.stage == "flux"]

    def wanted(self, stage):
        return [o for o, ob in self.observations.items() if stage_at_least(ob.stage, stage)]

    def row_set_of_row(self, row):
        for rs in self.row_sets:
            if row in rs.rows:
                return rs
        return None

    def row_set(self, letter):
        for rs in self.row_sets:
            if rs.letter == letter:
                return rs
        return None

    def group(self, gid):
        for g in self.groups:
            if g.group_id == gid:
                return g
        return None

    def groups_of(self, obsid):
        return [g for g in self.groups if obsid in g.obsids]

    def bands_for(self, group):
        cats = {self.observations[o].category for o in group.obsids}
        if all(c.is_hrc for c in cats):
            return self.hrc_bands
        return self.acis_bands

    def chip_rows(self, group):
        # rows of the group's own row set first (an observation added from elsewhere does not change it)
        rows = sorted({self.row_of(o) for o in group.obsids})
        rs = self.row_set(group.row_set)
        own = [r for r in rows if rs and r in rs.rows]
        return own + [r for r in rows if r not in own]

    def chip_sets_for(self, group):
        # the group's own sets if edited, else the chip choice of its own row set's first row
        if group.chip_sets is not None:
            return group.chip_sets
        for r in self.chip_rows(group):
            if r in self.chips_by_row:
                return self.chips_by_row[r].sets
        return [ChipSet(None)]

    def window_tag(self):
        return window_tag(self.window_days)


def window_tag(days):
    # w7d, w0d, w1p5d
    if float(days).is_integer():
        return f"w{int(days)}d"
    s = f"{days:.6g}".replace(".", "p")
    return f"w{s}d"
