import gzip
import os
import urllib.error
from datetime import datetime, timezone

import pytest

import chanmerge.archive as A

OCAT_TEXT = """# comment line
OBSID\tSEQ_NUM\tINSTR\tGRAT\tMODE\tDATAMODE\tREADOUT_DETECTOR\tSUBARY\tROW_CNT\tSOE_ROLL
------\t------\t-----\t----\t----\t--------\t----------------\t------\t-------\t--------

950\t200001\tACIS-I\tNONE\tTE\tFAINT\tACIS-012367\tNONE\t\t90.1
7490\t200002\tACIS-S\tHETG\tTE\tFAINT\tACIS-7\tCUSTOM\t128\t235.8
13197\t200003\tACIS-S\tHETG\tCC\tCC33_FAINT\tACIS-5678\tNONE\t\t10
463\t200004\tHRC-I\tNONE\t\t\tHRC-I\t\t\t0
"""


def test_parse_ocat_text_keys_by_obsid():
    d = A.parse_ocat_text(OCAT_TEXT)
    assert set(d) == {"950", "7490", "13197", "463"}
    assert d["7490"]["ROW_CNT"] == "128"


def test_mode_rules():
    d = A.parse_ocat_text(OCAT_TEXT)
    assert A.mode_from_ocat(d["950"]) == "TE"
    assert A.mode_from_ocat(d["13197"]) == "CC"
    assert A.mode_from_ocat({"MODE": "", "DATAMODE": ""}) is None       # empty is not cc
    assert A.mode_from_heasarc("CC_000E") == "CC" and A.mode_from_heasarc("TE_007E2") == "TE" and A.mode_from_heasarc("") is None
    # one source only -> that one; both differ or both empty -> '?'
    assert A.resolve_mode("ACIS-S", d["950"], None) == ("TE", "ocat")
    assert A.resolve_mode("ACIS-S", None, "CC_x") == ("CC", "heasarc")
    assert A.resolve_mode("ACIS-S", d["950"], "CC_x") == ("?", "disagree")
    assert A.resolve_mode("ACIS-S", {"MODE": ""}, "") == ("?", "none")
    assert A.resolve_mode("ACIS-S", d["950"], "TE_x") == ("TE", "ocat+heasarc")
    assert A.resolve_mode("HRC-I", None, None) == ("-", "hrc")
    # ocat unreachable: heasarc only
    assert A.resolve_mode("ACIS-S", d["13197"], "TE_x", ocat_ok=False) == ("TE", "heasarc")


def test_window_and_chips():
    d = A.parse_ocat_text(OCAT_TEXT)
    assert A.window_from_ocat(d["950"]) == "full" and A.window_from_ocat(d["7490"]) == "sub"
    assert A.window_from_ocat({"SUBARY": "NONE", "ROW_CNT": "512"}) == "sub"
    # real ocat answer for full frame 950 (2026-11-04 live run): SUBARY NONE, ROW_CNT 0
    assert A.window_from_ocat({"SUBARY": "NONE", "STRT_ROW": "0", "ROW_CNT": "0"}) == "full"
    assert A.window_from_ocat({"SUBARY": "CUSTOM", "STRT_ROW": "1", "ROW_CNT": "350"}) == "sub"
    assert A.chips_from_readout("ACIS-012367") == (0, 1, 2, 3, 6, 7)
    assert A.chips_from_readout("HRC-I") is None and A.chips_from_readout("") is None


def _hrows():
    return [
        dict(obsid="950", status="archived", name="GX 13+1", time="51000.5", detector="ACIS-I", grating="NONE", exposure="9600", public_date="51400", data_mode="TE_00AE"),
        dict(obsid="7490", status="archived", name="PSR", time="54318.7", detector="ACIS-S", grating="HETG", exposure="4674", public_date="54700", data_mode="TE_0040"),
        dict(obsid="13197", status="archived", name="GX", time="55900", detector="ACIS-S", grating="HETG", exposure="10000", public_date="56300", data_mode="CC_0010"),
        dict(obsid="463", status="archived", name="Cen A", time="51400", detector="HRC-I", grating="NONE", exposure="20000", public_date="51800", data_mode=None),
        dict(obsid="999", status="scheduled", name="x", time=None, detector="ACIS-S", grating="NONE", exposure=None, public_date=None, data_mode=None),
        dict(obsid="998", status="archived", name="future", time="60900", detector="ACIS-S", grating="NONE", exposure="1000", public_date="99999", data_mode="TE_1"),
        dict(obsid="0950", status="archived", name="dup", time="51000.5", detector="ACIS-I", grating="NONE", exposure="9600", public_date="51400", data_mode="TE_00AE"),
    ]


def test_archived_public_filter():
    kept, dropped = A.keep_archived_public(_hrows(), now=datetime(2026, 11, 4, tzinfo=timezone.utc))
    assert dropped == 2 and {r["obsid"] for r in kept} == {"950", "0950", "7490", "13197", "463"}


def test_make_observations():
    kept, _ = A.keep_archived_public(_hrows(), now=datetime(2026, 11, 4, tzinfo=timezone.utc))
    obs = {o.obsid: o for o in A.make_observations(kept, A.parse_ocat_text(OCAT_TEXT))}
    assert set(obs) == {"950", "7490", "13197", "463"}               # duplicate 0950 dropped
    assert obs["950"].category.label() == "ACIS-I/NONE/TE/full"
    assert obs["7490"].category.label() == "ACIS-S/HETG/TE/sub" and obs["7490"].folder == "7490_g"
    assert obs["13197"].category.mode == "CC"
    assert obs["463"].category.label() == "HRC-I/NONE/-/-" and obs["463"].active_chips is None
    assert obs["950"].active_chips == (0, 1, 2, 3, 6, 7)
    assert obs["950"].start.year == 1998
    assert obs["7490"].subarray_rows == 128 and obs["950"].subarray_rows is None     # ocat ROW_CNT kept for subarrays


def test_make_observations_without_ocat():
    kept, _ = A.keep_archived_public(_hrows(), now=datetime(2026, 11, 4, tzinfo=timezone.utc))
    obs = {o.obsid: o for o in A.make_observations(kept, {}, ocat_ok=False)}
    assert obs["950"].category.window == "?" and obs["950"].category.mode == "TE"   # mode from heasarc, window unknown
    assert obs["13197"].category.mode == "CC"


def test_boundaries():
    from datetime import timedelta
    from chanmerge.plan import Observation, Category
    c = Category("ACIS-S", "NONE", "TE", "full")
    t = datetime(2010, 1, 1, tzinfo=timezone.utc)
    o1 = Observation("1", c, t - timedelta(hours=2), 4 * 3600)          # crosses the start edge
    o2 = Observation("2", c, t + timedelta(days=30, hours=22), 5 * 3600)  # crosses the end edge (to = day 30)
    o3 = Observation("3", c, t + timedelta(days=5), 3600)
    s, e = A.boundary_observations([o1, o2, o3], t, t + timedelta(days=30))
    assert s == ["1"] and e == ["2"]
    assert not A.in_date_range(o1, t, t + timedelta(days=30)) and A.in_date_range(o2, t, t + timedelta(days=30))


def test_heasarc_errors_are_readable(monkeypatch):
    import astroquery.heasarc
    def boom(*a, **k):
        raise ConnectionError("network is down")
    # patch the class: query_heasarc makes a fresh HeasarcClass per try (a patch on the Heasarc singleton would let
    # this test reach the real service)
    monkeypatch.setattr(astroquery.heasarc.HeasarcClass, "query_region", boom)
    with pytest.raises(A.ArchiveError, match="network is down"):
        A.query_heasarc(10, 10, 5)


def test_ocat_failure_raises_archive_error(monkeypatch):
    def boom(url, timeout=60):
        raise urllib.error.URLError("unreachable")
    monkeypatch.setitem(A.HTTP, "get", boom)
    with pytest.raises(A.ArchiveError, match="Ocat unreachable"):
        A.fetch_ocat(["950"])


def test_ocat_chunks(monkeypatch):
    calls = []
    def fake(url, timeout=60):
        calls.append(url)
        return OCAT_TEXT.encode()
    monkeypatch.setitem(A.HTTP, "get", fake)
    A.fetch_ocat([str(i) for i in range(1200)])
    assert len(calls) == 3


LISTING = b'<a href="../">up</a><a href="primary/">primary/</a><a href="secondary/">secondary/</a><a href="oif.fits">oif</a>'
PRIMARY = b'<a href="acisf00950N005_evt2.fits.gz">e</a><a href="acisf00950_000N005_fov1.fits.gz">f</a><a href="?C=N;O=D">sort</a>'


def test_fetch_fov(tmp_path, monkeypatch):
    payload = gzip.compress(b"FOVDATA")
    def fake(url, timeout=60):
        if url.endswith("/primary/"):
            return PRIMARY
        if url.endswith("fov1.fits.gz"):
            return payload
        raise AssertionError(url)
    monkeypatch.setitem(A.HTTP, "get", fake)
    got = A.fetch_fovs({"950": str(tmp_path / "950" / "primary")})
    p = got["950"]
    assert p.endswith("acisf00950_000N005_fov1.fits.gz") and open(p, "rb").read() == payload
    assert A.archive_url("950") == "https://cxc.cfa.harvard.edu/cdaftp/byobsid/0/950/"
    # a failing fetch gives None, never raises
    monkeypatch.setitem(A.HTTP, "get", lambda url, timeout=60: (_ for _ in ()).throw(OSError("x")))
    assert A.fetch_fovs({"1": str(tmp_path / "1")}) == {"1": None}


def test_fetch_fov_uses_any_local_fov_without_network(tmp_path, monkeypatch):
    # test d: a decompressed fov1 (or a link to one) was not recognised and the .gz was downloaded
    def no_net(url, timeout=60):
        raise AssertionError(f"network used: {url}")
    monkeypatch.setitem(A.HTTP, "get", no_net)
    d = tmp_path / "950" / "primary"; d.mkdir(parents=True)
    src = tmp_path / "elsewhere_fov1.fits"; src.write_bytes(b"FOV")
    (d / "acisf00950_000N005_fov1.fits").symlink_to(src)
    assert A.fetch_fov("950", str(d)).endswith("acisf00950_000N005_fov1.fits")
    assert A.fetch_fovs({"950": str(d)}) == {"950": str(d / "acisf00950_000N005_fov1.fits")}


def test_sizes_and_fallback(monkeypatch):
    def fake_get(url, timeout=60):
        if url.endswith("/950/"):
            return LISTING
        if url.endswith("primary/"):
            return PRIMARY
        return b""
    monkeypatch.setitem(A.HTTP, "get", fake_get)
    monkeypatch.setitem(A.HTTP, "head", lambda url, timeout=30: 1000)
    assert A.archive_size("950") == 3000            # oif + two primary files


def test_size_walks_nested_folders(monkeypatch):
    # the 7490 listing: primary/responses/, secondary/aspect/, secondary/ephem/ (missed before 2026-11-04)
    tree = {"": (["oif.fits"], ["primary/", "secondary/"]), "primary/": (["evt2.fits.gz"], ["responses/"]),
            "primary/responses/": (["arf.gz", "rmf.gz"], []), "secondary/": (["evt1.fits.gz"], ["aspect/", "ephem/"]),
            "secondary/aspect/": (["asol.gz", "a2.gz", "a3.gz"], []), "secondary/ephem/": (["e.gz"], [])}
    base = A.archive_url("7490")
    def fake_get(url, timeout=60):
        files, dirs = tree[url[len(base):]]
        return "".join(f'<a href="{x}">{x}</a>' for x in files + dirs).encode()
    monkeypatch.setitem(A.HTTP, "get", fake_get)
    monkeypatch.setitem(A.HTTP, "head", lambda url, timeout=30: 10)
    assert A.archive_size("7490") == 90
    from chanmerge.plan import Observation, Category
    c = Category("ACIS-S", "NONE", "TE", "full")
    o = [Observation("950", c, A.MJD0, 10000), Observation("951", c, A.MJD0, 10000)]
    def too_many(url, timeout=30):
        raise urllib.error.HTTPError(url, 429, "slow down", None, None)
    monkeypatch.setitem(A.HTTP, "head", too_many)
    sizes, exact = A.download_sizes(o, pause=0)
    assert not exact and sizes["950"] == A.approx_size(10000) == 70_000_000
    assert A.human(4.1e9) == "4.1 GB" and A.human(87.6e6) == "88 MB"


def test_archive_size_reports_the_evt2_file(monkeypatch):
    tree = {"": (["oif.fits"], ["primary/", "secondary/"]), "primary/": (["acisf01_evt2.fits.gz", "fov1.fits.gz"], []),
            "secondary/": (["acisf01_evt1.fits.gz"], [])}
    base = A.archive_url("1")
    def fake_get(url, timeout=60):
        files, dirs = tree[url[len(base):]]
        return "".join(f'<a href="{x}">{x}</a>' for x in files + dirs).encode()
    monkeypatch.setitem(A.HTTP, "get", fake_get)
    monkeypatch.setitem(A.HTTP, "head", lambda url, timeout=30: 700 if "evt2" in url else 10)
    d = {}
    assert A.archive_size("1", detail=d) == 730 and d["evt2"] == 700      # evt1 is not the evt2
    from chanmerge.plan import Observation, Category
    o = [Observation("1", Category("ACIS-S", "NONE", "TE", "full"), A.MJD0, 10000)]
    ev = {}
    sizes, exact = A.download_sizes(o, pause=0, evt_sizes=ev)
    assert exact and sizes == {"1": 730} and ev == {"1": 700}


def test_repro_evt_bytes_disk_archive_fallback(tmp_path):
    from chanmerge.ciao import REPRO_MARKER
    from helpers import obs, ACIS_S
    o1, o2, o3 = obs(101, ACIS_S, 0), obs(102, ACIS_S, 1), obs(103, ACIS_S, 2)
    repro = tmp_path / "observations" / "101" / "repro"
    repro.mkdir(parents=True)
    (repro / "acisf00101_repro_evt2.fits").write_bytes(b"x" * 1234)
    (repro / REPRO_MARKER).write_text("ok")
    out = A.repro_evt_bytes([o1, o2, o3], str(tmp_path), {"102": 1e8, "103": 1e8}, {"102": 1e6})
    assert out["101"] == 1234                                     # already reprocessed: the file itself
    assert out["102"] == 1e6 * A.EVT2_UNPACK                       # archive evt2, unpacked
    assert out["103"] == 1e8 * A.EVT2_OF_ARCHIVE                   # no evt2 size: share of the archive


def test_run_footprint_counts_bands_chip_sets_and_corrections():
    from chanmerge.bands import parse_bands
    from chanmerge.grouping import build_groups
    from chanmerge.plan import ChipChoice, ChipSet
    from helpers import obs, make_plan, ACIS_S, ACIS_S_HETG
    p = make_plan([obs(1, ACIS_S_HETG, 0), obs(2, ACIS_S_HETG, 1), obs(3, ACIS_S, 5)], window=3)
    p.acis_bands, _ = parse_bands("broad")
    p.arm_method_by_row = {2: 4}                       # hetg row: mask + fill -> corrected copies
    p.groups = build_groups(p)
    ev = {"1": 10e6, "2": 10e6, "3": 10e6}
    g01 = 4 * 20e6 + A.IMAGE_FILE * (9 + 10)           # reproj + merged + their corrected copies
    s3 = 2 * 10e6 + A.IMAGE_FILE * (9 + 5)              # no correction asked: reproj + merged only
    assert A.run_footprint(p, ev) == int(g01 + s3)
    p.acis_bands, _ = parse_bands("broad, 1:3:2")      # every band is a folder of its own
    assert A.run_footprint(p, ev) == int(2 * (g01 + s3))
    p.row_set("A").streak_action = "acisreadcorr"      # row 1 = acis-s none: one acisreadcorr copy, once per run
    p.chips_by_row[1] = ChipChoice("separate", [ChipSet((5, 7), "bi"), ChipSet((4, 6, 8, 9), "fi")])
    # obs 3 has chips S1-S4 on (5, 6, 7, 8), target chip unknown: each set gets its share of the active chips
    s3b = 2 * (2 * 10e6 * 0.5 + A.IMAGE_FILE * 14)     # bi: S1,S3 = 2 of 4; fi: S2,S4 = 2 of 4
    assert A.run_footprint(p, ev) == int(2 * (g01 + s3b) + 10e6)
    # the target sits on S3 (7): 'bi' holds it -> counted whole (upper bound); 'fi' keeps its share
    o3 = p.observations["3"]
    o3.source_chips = (7,)
    s3c = 2 * 10e6 * 1.0 + A.IMAGE_FILE * 14 + 2 * 10e6 * (2 / 4) + A.IMAGE_FILE * 14
    assert A.run_footprint(p, ev) == int(2 * (g01 + s3c) + 10e6)


def test_chip_fraction_exact_from_disk(tmp_path):
    # reprocessed on disk: the share comes from the event file itself (test d: 950 'bi' = 5% of the events)
    import numpy as np
    from astropy.io import fits
    from chanmerge.ciao import REPRO_MARKER
    from helpers import obs, ACIS_S
    o = obs(7, ACIS_S, 0)
    d = tmp_path / "observations" / o.folder / "repro"; d.mkdir(parents=True)
    ccd = np.array([7] * 5 + [6] * 90 + [5] * 5, dtype=np.int16)
    fits.BinTableHDU.from_columns([fits.Column(name="ccd_id", format="I", array=ccd)], name="EVENTS").writeto(d / "acisf00007_repro_evt2.fits")
    (d / REPRO_MARKER).write_text("ok")
    assert A.chip_fraction(o, (5, 7), str(tmp_path)) == pytest.approx(0.10)
    assert A.chip_fraction(o, (6,), str(tmp_path)) == pytest.approx(0.90)
    assert A.chip_fraction(o, None, str(tmp_path)) == 1.0


def test_heasarc_hang_times_out_and_retries(monkeypatch):
    # test d rerun: pyvo's tap search has no timeout and hung > 10 min; the search must give up and say why
    import time
    calls = []

    def hang(ra, dec, r):
        calls.append(1); time.sleep(2); return None
    monkeypatch.setattr(A, "_heasarc_once", hang)
    t0 = time.time()
    with pytest.raises(A.ArchiveError) as e:
        A.query_heasarc(1.0, 2.0, 3.0, timeout=0.2, tries=2)
    assert "no answer within" in str(e.value) and len(calls) == 2 and time.time() - t0 < 1.5

    class Tab:
        colnames = ["obsid"]
        def __iter__(self):
            return iter([{"obsid": 950}])
    seq = iter(["hang", "ok"])

    def once_then_ok(ra, dec, r):
        if next(seq) == "hang":
            time.sleep(2)
        return Tab()
    monkeypatch.setattr(A, "_heasarc_once", once_then_ok)
    assert A.query_heasarc(1.0, 2.0, 3.0, timeout=0.3, tries=2) == [{"obsid": "950"}]   # values are decoded to text
    # an ordinary error is reported at once, not retried
    monkeypatch.setattr(A, "_heasarc_once", lambda ra, dec, r: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(A.ArchiveError) as e:
        A.query_heasarc(1.0, 2.0, 3.0, timeout=1, tries=2)
    assert "ValueError: bad" in str(e.value)
