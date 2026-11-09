import os

import pytest

from chanmerge.ciao import Ciao, ToolError, REPRO_MARKER, header_mode_window, parse_header_time, repro_files, primary_evt2


class Fake:
    """records calls; fails chandra_repro for the obsids in fail"""
    def __init__(self, fail=(), outputs=None):
        self.fail, self.outputs, self.calls = set(fail), outputs or {}, []

    def __call__(self, args, cwd=None, env=None, timeout=None):
        self.calls.append(args)
        assert isinstance(args, list) and all(isinstance(a, str) for a in args)   # never a shell string
        if args[0] == "chandra_repro":
            if any(f"/{o}" in args[1] for o in self.fail):
                return 1, "chandra_repro: ERROR"
            out = args[2].split("=", 1)[1]
            os.makedirs(out, exist_ok=True)
            open(os.path.join(out, "acisf00101_repro_evt2.fits"), "w").close()
            return 0, "ok"
        if args[0] == "download_chandra_obsid":
            out = self.outputs.pop(args[1], "ok") if self.outputs else "ok"
            prim = os.path.join(cwd, args[1], "primary")
            os.makedirs(prim, exist_ok=True)
            open(os.path.join(prim, f"acisf{int(args[1]):05d}N001_evt2.fits.gz"), "w").close()
            return 0, out
        return 0, "ok"


def test_punlearn_after_every_repro_even_when_it_fails(tmp_path):
    fake = Fake(fail={"102"})
    c = Ciao(str(tmp_path / "pf"), runner=fake)
    for o in ("101", "102", "103"):
        d = tmp_path / "observations" / o
        d.mkdir(parents=True)
        if o == "102":
            with pytest.raises(ToolError):
                c.repro(str(d))
        else:
            assert c.repro(str(d)) == "done"
    tools = [a[0] for a in fake.calls]
    assert tools == ["chandra_repro", "punlearn", "chandra_repro", "punlearn", "chandra_repro", "punlearn"]
    assert all(fake.calls[i + 1] == ["punlearn", "ardlib"] for i, t in enumerate(tools) if t == "chandra_repro")
    # marker only where repro exited 0; a failed one reruns next time
    assert os.path.exists(tmp_path / "observations/101/repro" / REPRO_MARKER)
    assert not os.path.exists(tmp_path / "observations/102/repro" / REPRO_MARKER)


def test_repro_reused_with_marker_and_private_pfiles(tmp_path):
    fake = Fake()
    c = Ciao(str(tmp_path / "pf"), runner=fake)
    d = tmp_path / "o" / "101"
    (d / "repro").mkdir(parents=True)
    (d / "repro" / REPRO_MARKER).write_text("x")
    assert c.repro(str(d)) == "reused" and fake.calls == []
    assert c.env["PFILES"].startswith(str(tmp_path / "pf") + ";")


def test_download_grating_folder_and_resume(tmp_path):
    fake = Fake(outputs={"7490": "WARNING file sizes do not match"})
    c = Ciao(str(tmp_path / "pf"), runner=fake)
    parent = tmp_path / "observations"
    (parent / "7490_g" / "primary").mkdir(parents=True)
    (parent / "7490_g" / "primary" / "acisf07490_000N003_fov1.fits.gz").write_text("fov")   # fetched before
    final = c.download(str(parent), "7490", grating=True)
    assert final.endswith("7490_g") and not (parent / "7490").exists()
    assert (parent / "7490_g/primary/acisf07490_000N003_fov1.fits.gz").read_text() == "fov"
    assert [a[0] for a in fake.calls].count("download_chandra_obsid") == 2        # reran after the size warning


def test_merge_obs_uses_stacks_and_list_args(tmp_path):
    fake = Fake()
    c = Ciao(str(tmp_path / "pf"), runner=fake)
    out = str(tmp_path / "T" / "run" / "g01_x" / "T_w3d_g01")
    c.merge_obs(out, ["/a/evt.fits[ccd_id=7]", "/b/evt.fits[ccd_id=7]"], "broad", asol=["/a/asol.fits"])
    args = fake.calls[-1]
    assert args[0] == "merge_obs" and args[1].startswith("@") and "bands=broad" in args and any(a.startswith("asolfiles=@") for a in args)
    assert open(args[1][1:]).read().splitlines() == ["/a/evt.fits[ccd_id=7]", "/b/evt.fits[ccd_id=7]"]


def test_header_rules():
    assert header_mode_window({"INSTRUME": "ACIS", "READMODE": "TIMED", "DATAMODE": "FAINT", "NROWS": 128}) == ("TE", "sub")
    assert header_mode_window({"INSTRUME": "ACIS", "READMODE": "CONTINUOUS", "DATAMODE": "CC33_FAINT", "NROWS": 1024}) == ("CC", "full")
    assert header_mode_window({"INSTRUME": "HRC"}) == ("-", "-")
    assert parse_header_time("2007-08-06T17:34:16").hour == 17


def test_repro_exit_zero_without_output_is_an_error(tmp_path):
    c = Ciao(str(tmp_path / "pf"), runner=lambda args, cwd=None, env=None, timeout=None: (0, "ok"))
    d = tmp_path / "o" / "101"
    d.mkdir(parents=True)
    with pytest.raises(ToolError, match="no \\*_repro_evt2"):
        c.repro(str(d))
    assert not os.path.exists(d / "repro" / REPRO_MARKER)


def test_wildcard_characters_in_root_do_not_match_sibling_roots(tmp_path):
    # test o: a root path with * or ? used to match sibling roots that hold the same observation (wrong asol / evt2 files)
    for r in ("r*x", "r?x", "rAx", "rABx"):
        o = tmp_path / r / "observations" / "8225"
        (o / "repro").mkdir(parents=True)
        (o / "primary").mkdir()
        for f in ("repro/hrcf08225_repro_evt2.fits", "repro/pcadf08225_000N001_asol1.fits", "primary/hrcf08225N007_evt2.fits.gz"):
            (o / f).write_text("x")
    for r in ("r*x", "r?x"):
        d = str(tmp_path / r / "observations" / "8225")
        f = repro_files(d)
        assert f["asol"] == [os.path.join(d, "repro", "pcadf08225_000N001_asol1.fits")]
        assert f["evt2"] == os.path.join(d, "repro", "hrcf08225_repro_evt2.fits")
        assert primary_evt2(d) == os.path.join(d, "primary", "hrcf08225N007_evt2.fits.gz")
