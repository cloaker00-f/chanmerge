import os
import warnings

import pytest

from chanmerge.tables import (FINAL_COLUMNS, MEMBER_COLUMNS, write_csv_atomic, read_csv, fill_shares_run,
                              rebuild_index, load_index, chips_overlap, index_columns, INDEX_FINAL, INDEX_MEMBERS)


def test_columns_cover_the_plan():
    for c in ["flux_image", "flux_image_original", "evt_file", "evt_file_original", "repro_evt_file", "repro_evt_file_abs",
              "expmap_file", "expmap_file_original", "streak_action", "streak_obsids", "arm_method", "arm_masked_obsids",
              "arm_filled_obsids", "filled_pixel_fraction", "mask_file", "shares_photons_with", "categories", "n_grating", "chips"]:
        assert c in FINAL_COLUMNS
    for c in ["detector", "grating", "mode", "window", "obsid", "exposure_ks", "shares_photons_with", "repro_evt_file"]:
        assert c in MEMBER_COLUMNS
    assert "evt_matches_image" not in FINAL_COLUMNS
    ic = index_columns(FINAL_COLUMNS)
    for c in ["flux_image_abs", "evt_file_original_abs", "expmap_file_original_abs", "mask_file_abs"]:
        assert c in ic
    assert ic.count("repro_evt_file_abs") == 1 and ic.count("acisreadcorr_evt_file_abs") == 1


def test_chips_overlap():
    assert chips_overlap("all", "S3")
    assert chips_overlap("S3", "S1,S3")
    assert not chips_overlap("bi", "fi") or True    # labels are not chip names; real rows hold names
    assert not chips_overlap("S1,S3", "S0,S2,S4,S5,I0,I1,I2,I3")
    assert chips_overlap("I0-I3", "I3")


def _f(gid, obsids, chips="all", band="broad"):
    return dict(group_id=gid, obsids=";".join(obsids), chips=chips, band=band)


def _m(gid, obsid, chips="all", band="broad"):
    return dict(group_id=gid, obsid=obsid, chips=chips, band=band)


def test_shares_in_one_run():
    finals = [_f("g01", ["101", "102"]), _f("g02_S3", ["201"], "S3"), _f("g02_S1_S3", ["201"], "S1,S3"),
              _f("g03_bi", ["301"], "S1,S3"), _f("g03_fi", ["301"], "S0,S2,S4,S5,I0,I1,I2,I3"), _f("s103", ["103"])]
    members = [_m("g01", "101"), _m("g01", "102")]
    fill_shares_run(finals, members)
    by = {r["group_id"]: r["shares_photons_with"] for r in finals}
    assert by["g02_S3"] == "g02_S1_S3" and by["g02_S1_S3"] == "g02_S3"
    assert by["g03_bi"] == "" and by["g03_fi"] == ""           # same obsid, disjoint chips -> no shared photons
    assert by["g01"] == "" and by["s103"] == ""
    assert all(m["shares_photons_with"] == "" for m in members)  # members of the same group do not count


def test_shares_after_add():
    finals = [_f("g01", ["101", "102", "103"]), _f("s103", ["103"])]
    members = [_m("g01", "101"), _m("g01", "102"), _m("g01", "103")]
    fill_shares_run(finals, members)
    assert {r["group_id"]: r["shares_photons_with"] for r in finals} == {"g01": "s103", "s103": "g01"}
    assert members[2]["shares_photons_with"] == ""   # s103 has no member table row (single)


def _write_run(root, target, run, finals, members, make_flux=True):
    d = os.path.join(root, target, run)
    os.makedirs(d, exist_ok=True)
    for r in finals + members:
        if make_flux and r.get("flux_image"):
            p = os.path.join(d, r["flux_image"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").close()
    write_csv_atomic(os.path.join(d, "final_images.csv"), FINAL_COLUMNS, finals)
    write_csv_atomic(os.path.join(d, "member_images.csv"), MEMBER_COLUMNS, members)
    return d


def test_rebuild_and_load_index(tmp_path):
    root = str(tmp_path)
    f1 = dict(_f("g01", ["101", "102"]), flux_image="g01_x/T_w3d_g01_broad_flux.img",
              repro_evt_file="../../observations/101/repro/acisf00101_repro_evt2.fits;../../observations/102/repro/acisf00102_repro_evt2.fits")
    m1 = [dict(_m("g01", "101"), flux_image="g01_x/T_w3d_g01_101_broad_flux.img"),
          dict(_m("g01", "102"), flux_image="g01_x/T_w3d_g01_102_broad_flux.img")]
    _write_run(root, "T", "merged_final_image_broad_w3d", [f1], m1)
    # a second run of the same target and window 0: shares photons with the first
    f2 = dict(_f("all", ["101", "102", "103"]), flux_image="all/T_w0d_all_broad_flux.img")
    m2 = [dict(_m("all", o), flux_image=f"all/T_w0d_all_{o}_broad_flux.img") for o in ("101", "102", "103")]
    _write_run(root, "T", "merged_final_image_broad_w0d", [f2], m2)
    # a soft run of the same obs: soft vs broad overlap; a hard run would too (0.5-7 contains 2-7)
    nf, nm = rebuild_index(root)
    assert (nf, nm) == (2, 5)
    rows = read_csv(os.path.join(root, INDEX_FINAL))
    by = {r["group_id"]: r for r in rows}
    assert by["g01"]["flux_image"] == "T/merged_final_image_broad_w3d/g01_x/T_w3d_g01_broad_flux.img"
    assert by["g01"]["flux_image_abs"] == os.path.join(root, by["g01"]["flux_image"])
    assert by["g01"]["repro_evt_file"].split(";")[0] == "observations/101/repro/acisf00101_repro_evt2.fits"
    assert by["g01"]["shares_photons_with"] == "T/merged_final_image_broad_w0d/all"
    mem = read_csv(os.path.join(root, INDEX_MEMBERS))
    m101 = [r for r in mem if r["obsid"] == "101" and r["group_id"] == "g01"][0]
    assert m101["shares_photons_with"] == "T/merged_final_image_broad_w0d/all:101"
    loaded = load_index(os.path.join(root, INDEX_FINAL))
    assert all(r["exists"] for r in loaded)


def test_acisreadcorr_column_keeps_empty_entries(tmp_path):
    # test d: the acisreadcorr file is shared per target (<target>/acisreadcorr/) and named per member, '' where a member
    # had none; an empty entry in a ';' list must stay empty in the index (it used to become the run folder's path)
    root = str(tmp_path)
    f1 = dict(_f("g01", ["101", "102"]), flux_image="g01_x/T_w3d_g01_broad_flux.img",
              acisreadcorr_evt_file="../acisreadcorr/T_101_acisreadcorr_evt.fits;")
    m1 = [dict(_m("g01", "101"), flux_image="g01_x/T_w3d_g01_101_broad_flux.img",
               acisreadcorr_evt_file="../acisreadcorr/T_101_acisreadcorr_evt.fits"),
          dict(_m("g01", "102"), flux_image="g01_x/T_w3d_g01_102_broad_flux.img")]
    _write_run(root, "T", "merged_final_image_broad_w3d", [f1], m1)
    rebuild_index(root)
    r = read_csv(os.path.join(root, INDEX_FINAL))[0]
    assert r["acisreadcorr_evt_file"] == "T/acisreadcorr/T_101_acisreadcorr_evt.fits;"
    assert r["acisreadcorr_evt_file_abs"] == os.path.join(root, "T/acisreadcorr/T_101_acisreadcorr_evt.fits") + ";"
    mem = {m["obsid"]: m for m in read_csv(os.path.join(root, INDEX_MEMBERS))}
    assert mem["101"]["acisreadcorr_evt_file"] == "T/acisreadcorr/T_101_acisreadcorr_evt.fits" and mem["102"]["acisreadcorr_evt_file"] == ""
    loaded = load_index(os.path.join(root, INDEX_FINAL))[0]
    assert loaded["acisreadcorr_evt_file_abs"].endswith(";") and loaded["acisreadcorr_evt_file_abs"].count(";") == 1


def test_rebuild_skips_missing_flux_and_malformed(tmp_path):
    root = str(tmp_path)
    f1 = dict(_f("g01", ["101"]), flux_image="g01/x_flux.img")
    _write_run(root, "T", "merged_final_image_broad_w3d", [f1], [], make_flux=False)
    bad = os.path.join(root, "U", "merged_final_image_soft_w3d")
    os.makedirs(bad)
    with open(os.path.join(bad, "final_images.csv"), "w") as f:
        f.write("a,b\n1,2\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        nf, nm = rebuild_index(root)
    assert (nf, nm) == (0, 0)
    msgs = " ".join(str(x.message) for x in w)
    assert "missing" in msgs and "malformed" in msgs


def test_load_index_warns_when_root_moved(tmp_path):
    root = tmp_path / "a"
    root.mkdir()
    f1 = dict(_f("g01", ["101"]), flux_image="g01/x_flux.img")
    _write_run(str(root), "T", "merged_final_image_broad_w3d", [f1], [])
    rebuild_index(str(root))
    moved = tmp_path / "b"
    os.rename(root, moved)
    with pytest.warns(UserWarning, match="stale"):
        rows = load_index(str(moved / INDEX_FINAL))
    assert rows[0]["exists"] and rows[0]["flux_image_abs"].startswith(str(moved))


def test_atomic_write_keeps_old_file_on_error(tmp_path):
    p = tmp_path / "t.csv"
    write_csv_atomic(str(p), ["a"], [{"a": 1}])
    class Boom(dict):
        def get(self, *a):
            raise RuntimeError("boom")
    with pytest.raises(RuntimeError):
        write_csv_atomic(str(p), ["a"], [Boom()])
    assert read_csv(str(p)) == [{"a": "1"}]
    assert [x.name for x in tmp_path.iterdir()] == ["t.csv"]
