# the viewer's pure helpers (no window) and one smoke test of the tab (skipped without a display)
import csv
import os

import numpy as np
import pytest
from astropy.io import fits

from chanmerge import viewer as V


@pytest.fixture(autouse=True)
def warnings_seen(monkeypatch):
    """a real warning window would wait for a click and hang the test run: record the texts instead"""
    seen = []
    monkeypatch.setattr(V.messagebox, "showwarning", lambda title, text, **k: seen.append((title, text)))
    return seen


def test_limits_modes():
    d = np.arange(101, dtype=float)
    d[3] = np.nan
    assert V.limits_of(d, "minmax") == (0.0, 100.0)
    lo, hi = V.limits_of(d, "99%")
    assert 0.0 < lo < 1.5 and 98.5 < hi < 100.0
    assert V.limits_of(d, "user", (10.0, None)) == (10.0, 100.0)
    lo, hi = V.limits_of(np.full(5, 2.0), "minmax")
    assert hi > lo                                        # flat image: never a zero-width range
    assert V.limits_of(np.full(4, np.nan), "minmax") == (0.0, 1.0)
    zlo, zhi = V.limits_of(d, "zscale")
    assert zlo <= zhi


@pytest.mark.parametrize("scale", ["linear", "log", "sqrt", "squared", "asinh"])
def test_norm_is_monotonic_and_clipped(scale):
    n = V.make_norm(scale, 1.0, 5.0)
    x = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 9.0])
    y = np.asarray(n(x))
    assert y[0] == pytest.approx(0.0) and y[-1] == pytest.approx(1.0)      # outside the limits: clipped
    assert np.all(np.diff(y) >= 0) and y[1] == pytest.approx(0.0) and y[4] == pytest.approx(1.0)
    assert np.all((y >= 0) & (y <= 1))


def test_curves_are_inverse_pairs():
    for name, (g, ginv) in V._CURVES.items():
        t = np.linspace(0, 1, 11)
        assert np.allclose(ginv(g(t)), t), name
        assert g(0.0) == pytest.approx(0.0) and g(1.0) == pytest.approx(1.0)


def _write_run(tmp_path, with_files=True):
    img = tmp_path / "T" / "run" / "g01"
    img.mkdir(parents=True)
    for n in ("flux.img", "orig.img", "exp.img"):
        h = fits.PrimaryHDU(np.arange(12, dtype=float).reshape(3, 4))
        h.header.update(CTYPE1="RA---TAN", CTYPE2="DEC--TAN", CRVAL1=10.0, CRVAL2=20.0, CRPIX1=2, CRPIX2=2, CDELT1=-1e-3, CDELT2=1e-3)
        if with_files:
            h.writeto(img / n)
    cols = ["target", "band", "window_days", "group_id", "chips", "type", "obsids", "total_exposure_ks", "flux_image", "flux_image_original",
            "expmap_file", "mask_file"]
    row = dict(target="T", band="broad", window_days="7", group_id="g01", chips="all", type="merged", obsids="1;2", total_exposure_ks="9.5",
               flux_image="T/run/g01/flux.img", flux_image_original="T/run/g01/orig.img", expmap_file="T/run/g01/exp.img", mask_file="")
    with open(tmp_path / "chanmerge_index_final.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerow(row)


def test_list_images_reads_the_index(tmp_path):
    _write_run(tmp_path)
    items = V.list_images(str(tmp_path))
    assert len(items) == 1
    it = items[0]
    assert it["label"] == "T | g01 | broad | w7d" and it["exposure_ks"] == "9.5"
    assert set(it["files"]) == {"flux", "flux (original)", "exposure map"}      # empty mask_file: no 'mask' entry
    data, hdr, wcs = V.load_fits_image(it["files"]["flux"])
    assert data.shape == (3, 4) and wcs is not None
    ra, dec = wcs.pixel_to_world_values(1, 1)                                    # crpix 2 (1-based) is pixel index 1
    assert float(ra) == pytest.approx(10.0) and float(dec) == pytest.approx(20.0)


def test_list_images_skips_missing_files_and_missing_index(tmp_path):
    assert V.list_images(str(tmp_path)) == []                                    # no index at all
    _write_run(tmp_path, with_files=False)
    assert V.list_images(str(tmp_path)) == []                                    # rows whose images are gone


def test_load_fits_image_without_image_raises(tmp_path):
    p = tmp_path / "e.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(p)
    with pytest.raises(ValueError):
        V.load_fits_image(str(p))


def test_viewer_tab_smoke(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    try:
        _write_run(tmp_path)
        v = V.Viewer(root, cwd=str(tmp_path))
        assert len(v.items) == 1
        v.tree.selection_set("0")
        v.on_select()
        v.vars["scale"].set("log")
        v.vars["cmap"].set("gray")
        v.vars["limits"].set("user")
        v.vars["lo"].set("2")
        v.redraw()
        assert v.data.shape == (3, 4) and "no-data" in v.status.get()
        v.vars["show"].set("exposure map")
        v.on_option("show")
        assert v.path.endswith("exp.img")
    finally:
        root.destroy()


# ---------- smoothing ----------
@pytest.mark.parametrize("fn", list(V.SMOOTH_FUNCTIONS))
def test_smooth_kernels_are_normalised_and_symmetric_where_they_should_be(fn):
    k = V.smooth_kernel(fn, radius=4, radius_minor=2, angle=30.0)
    assert k.sum() == pytest.approx(1.0) and np.all(k >= 0)
    if fn != "elliptical gaussian":
        assert k.shape == (9, 9) and np.allclose(k, k.T) and np.allclose(k, k[::-1, ::-1])


def test_kernel_shapes_follow_ds9():
    box = V.smooth_kernel("boxcar", 2)
    assert box.shape == (5, 5) and np.allclose(box, 1 / 25)
    top = V.smooth_kernel("tophat", 2)
    assert top[0, 0] == 0 and top[2, 2] > 0 and np.count_nonzero(top) == 21          # ds9's disc, not x^2+y^2 <= 4 (13 pixels)
    # pixel counts measured from ds9 8.7's own output (test h): the disc keeps x^2 + y^2 <= r^2 + (r-1)^2
    assert [int(np.count_nonzero(V.smooth_kernel("tophat", r))) for r in range(1, 7)] == [5, 21, 45, 77, 117, 165]
    g = V.smooth_kernel("gaussian", 3, sigma=1.0)
    assert g[3, 3] == g.max() and g[3, 3] > g[3, 4] > g[3, 5]
    g2 = V.smooth_kernel("gaussian", 4)                                                # sigma defaults to radius / 2 = 2
    assert g2[4, 4 + 2] / g2[4, 4] == pytest.approx(np.exp(-0.5))
    e = V.smooth_kernel("elliptical gaussian", 6, sigma=3.0, radius_minor=2, sigma_minor=1.0, angle=0.0)
    assert e.shape == (13, 13) and e[6, 6 + 3] > e[6 + 3, 6]                           # long along x, short along y
    e90 = V.smooth_kernel("elliptical gaussian", 6, sigma=3.0, radius_minor=2, sigma_minor=1.0, angle=90.0)
    assert np.allclose(e90, e.T, atol=1e-12)                                           # 90 degrees turns it


@pytest.mark.parametrize("bad", [dict(radius=0), dict(radius=2, radius_minor=-1), dict(function="nope"),
                                 dict(function="gaussian", sigma=-1.0)])
def test_kernel_rejects_bad_numbers(bad):
    with pytest.raises(ValueError):
        V.smooth_kernel(**{"function": "boxcar", "radius": 3, **bad})


def test_smoothing_keeps_a_flat_image_flat_even_at_the_edge_and_around_no_data():
    d = np.full((20, 25), 5.0)
    d[8:11, 8:11] = np.nan
    out = V.smooth_image(d, V.smooth_kernel("gaussian", 3))
    ok = np.isfinite(d)
    assert np.allclose(out[ok], 5.0)                                                    # nan pixels do not pull the average down
    assert np.isnan(out[~ok]).all()                                                     # and no data stays no data


def test_smoothing_conserves_a_point_and_spreads_it():
    d = np.zeros((21, 21))
    d[10, 10] = 1.0
    out = V.smooth_image(d, V.smooth_kernel("boxcar", 2))
    assert out.sum() == pytest.approx(1.0) and out[10, 10] == pytest.approx(1 / 25) and out[8, 8] == pytest.approx(1 / 25)
    assert out[7, 10] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(V.smooth_image(np.full((3, 3), np.nan), V.smooth_kernel("boxcar", 1))).all()


def test_shared_limits_use_all_images_together():
    a, b = np.array([[0.0, 1.0]]), np.array([[10.0, np.nan]])
    assert V.shared_limits([a, b], "minmax") == (0.0, 10.0)
    assert V.shared_limits([None, a], "minmax") == (0.0, 1.0)
    assert V.shared_limits([], "minmax") == (0.0, 1.0)


# ---------- several pictures ----------
def _viewer(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    _write_run(tmp_path)
    return root, V.Viewer(root, cwd=str(tmp_path))


def test_four_pictures_fill_one_by_one_and_each_keeps_its_own_image(tmp_path):
    root, v = _viewer(tmp_path)
    try:
        it = v.items[0]
        v.vars["panels"].set("4 (2x2)")
        v.on_option("panels")
        assert v.n_panels() == 4 and v.panels[0].data is not None and v.panels[1].data is None     # picture 1 holds the first image
        for k, show in enumerate(["flux", "flux (original)", "exposure map"]):
            v.active = k + 1
            v.vars["show"].set(show)
            v.assign(it, show)
        assert [p.show for p in v.panels] == ["flux", "flux", "flux (original)", "exposure map"]
        assert all(p.ax is not None and p.im is not None for p in v.panels)
        assert v.panels[3].path.endswith("exp.img") and v.data is v.panels[v.active].data
        v.vars["panels"].set("2")
        v.on_option("panels")
        assert v.panels[2].ax is None and v.panels[0].ax is not None                              # picture 3, 4 hidden, kept
        assert v.panels[3].path.endswith("exp.img")
    finally:
        root.destroy()


def test_clicking_a_picture_makes_it_active_and_show_menu_follows(tmp_path):
    root, v = _viewer(tmp_path)
    try:
        v.vars["panels"].set("2")
        v.on_option("panels")
        v.active = 1
        v.assign(v.items[0], "exposure map")
        v.active = 0
        class E:                                  # a matplotlib mouse event, only what on_press reads
            inaxes = v.panels[1].ax
        v.on_press(E)
        assert v.active == 1 and v.vars["show"].get() == "exposure map"
    finally:
        root.destroy()


def test_smooth_changes_what_is_drawn_and_same_limits_share_one_range(tmp_path):
    root, v = _viewer(tmp_path)
    try:
        raw = v.panels[0].raw.copy()
        v.vars["sfunc"].set("boxcar")
        v.vars["sradius"].set("1")
        v.on_smooth(turn_on=True)
        assert v.vars["smooth"].get() and not np.allclose(v.data, raw) and v.data.sum() < raw.sum() * 1.01
        assert "smoothed" in v.status.get()
        v.vars["smooth"].set(False)
        v.on_smooth()
        assert np.allclose(v.data, raw)
        v.vars["sradius"].set("abc")
        v.on_smooth(turn_on=True)                                                                  # bad number: message, nothing drawn wrongly
        assert "not a number" in v.status.get()
        v.vars["smooth"].set(False)
        v.vars["panels"].set("2")
        v.vars["lock"].set(True)
        v.on_option("panels")
        v.active = 1
        v.assign(v.items[0], "exposure map")                                                       # very different values
        assert v.panels[0].lims == v.panels[1].lims
    finally:
        root.destroy()


def test_link_zoom_moves_the_other_pictures_too(tmp_path):
    root, v = _viewer(tmp_path)
    try:
        v.vars["panels"].set("2")
        v.vars["link"].set(True)
        v.on_option("panels")
        v.active = 1
        v.assign(v.items[0], "flux")
        v.panels[0].ax.set_xlim(0.5, 2.5)
        assert v.panels[1].ax.get_xlim() == pytest.approx((0.5, 2.5))
        v.vars["link"].set(False)
        v.panels[0].ax.set_xlim(1.0, 2.0)
        assert v.panels[1].ax.get_xlim() == pytest.approx((0.5, 2.5))
        v.zoom_fit()
        assert v.panels[0].ax.get_xlim() == pytest.approx((-0.5, 3.5))
    finally:
        root.destroy()


def test_new_image_does_not_inherit_the_old_images_zoom(tmp_path):
    # seen on real data: an hrc image put in a picture that had held a narrow strip came up cropped to the strip's range
    root, v = _viewer(tmp_path)
    try:
        big = tmp_path / "big.img"
        fits.PrimaryHDU(np.arange(60 * 80, dtype=float).reshape(60, 80)).writeto(big)
        v.panels[0].ax.set_xlim(0.5, 2.5)
        v.panels[0].ax.set_ylim(0.5, 1.5)                                  # zoomed on the small image
        v.assign(dict(label="big", info="", files={"flux": str(big)}), "flux")
        assert v.panels[0].ax.get_xlim() == pytest.approx((-0.5, 79.5)) and v.panels[0].ax.get_ylim() == pytest.approx((-0.5, 59.5))
        v.panels[0].ax.set_xlim(10, 20)
        v.redraw()                                                         # same image again: the zoom is kept
        assert v.panels[0].ax.get_xlim() == pytest.approx((10, 20))
    finally:
        root.destroy()


def test_start_up_does_not_assign_the_first_image_to_a_later_active_picture(tmp_path):
    root, v = _viewer(tmp_path)
    try:
        v.vars["panels"].set("2")
        v.on_option("panels")
        v.active = 1
        root.update()                                                      # lets any queued selection event run
        assert v.panels[1].data is None and v.panels[0].data is not None
    finally:
        root.destroy()


def test_smooth_box_alone_gives_ds9s_starting_smoothing(tmp_path):
    # ds9 8.7 (smooth.tcl smoothdef): gaussian, radius 3, sigma 1.5 -- one click, no parameters to choose
    root, v = _viewer(tmp_path)
    try:
        v.vars["smooth"].set(True)
        k = v.kernel()
        assert v.vars["sfunc"].get() == "gaussian" and k.shape == (7, 7)
        assert np.allclose(k, V.smooth_kernel("gaussian", 3, sigma=1.5))
        v.on_smooth()
        assert not np.allclose(v.data, v.panels[0].raw) and "gaussian" in v.status.get()
        v.vars["sfunc"].set("boxcar")
        v.vars["sradius"].set("9")
        v.reset_smooth()                                                    # reset: back to ds9's starting values, smoothing on
        assert v.vars["sfunc"].get() == "gaussian" and v.vars["sradius"].get() == "3" and v.vars["ssigma"].get() == "1.5"
    finally:
        root.destroy()


def test_smooth_parameters_window_opens_once_and_apply_turns_smoothing_on(tmp_path):
    root, v = _viewer(tmp_path)
    try:
        v.open_smooth_params()
        w = v.smooth_win
        v.open_smooth_params()
        assert v.smooth_win is w and w.winfo_exists()                       # still the same single window
        v.vars["sfunc"].set("tophat")
        v.vars["sradius"].set("2")
        v.on_smooth(turn_on=True)
        assert v.vars["smooth"].get() and "tophat" in v.status.get()
        w.destroy()
    finally:
        root.destroy()


def test_layout_slots_and_three_pictures_are_centred():
    assert V.layout_slots(1)[2] == [(0, 1, 0, 1)]
    for n in (1, 2, 3, 4):
        rows, cols, slots = V.layout_slots(n)
        assert len(slots) == n
        cells = [(r, c) for (r0, r1, c0, c1) in slots for r in range(r0, r1) for c in range(c0, c1)]
        assert len(cells) == len(set(cells)) and all(r < rows and c < cols for r, c in cells)      # no overlap, all inside
    _, cols, slots = V.layout_slots(3)
    top1, top2, bottom = slots
    assert top1[3] == top2[2] and bottom[0] == 1                       # two on top side by side, one below
    assert bottom[2] + bottom[3] == cols                               # centred: as much room left as right


def test_three_pictures_draw_two_on_top_and_one_centred_below(tmp_path):
    root, v = _viewer(tmp_path)
    try:
        v.vars["panels"].set("3")
        v.on_option("panels")
        for k in (1, 2):
            v.active = k
            v.assign(v.items[0], "flux")
        assert [p.ax is not None for p in v.panels] == [True, True, True, False]
        a0, a1, a2 = (v.panels[k].ax.get_position() for k in range(3))
        assert a0.y0 == pytest.approx(a1.y0) and a2.y1 < a0.y0 + 1e-9          # third picture sits below
        assert (a2.x0 + a2.x1) / 2 == pytest.approx((a0.x0 + a1.x1) / 2, abs=0.02)   # and in the middle
    finally:
        root.destroy()


# ---------- tree and search ----------
def _write_tree_run(tmp_path, drop_group_row=False):
    """target t: run broad_w7d with group g01 (members 1 and 2) and a one-observation group s3; run soft_w7d with g01 alone"""
    def img(rel):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(np.ones((4, 5))).writeto(path)
    fcols = ["target", "band", "window_days", "group_id", "chips", "type", "obsids", "total_exposure_ks", "flux_image", "flux_image_original",
             "expmap_file", "mask_file"]
    mcols = ["target", "band", "window_days", "group_id", "chips", "obsid", "exposure_ks", "flux_image", "flux_image_original", "expmap_file", "mask_file"]
    finals, members = [], []
    for band, groups in (("broad", [("g01", "g01_a_b", ["1", "2"]), ("s3", "s3_a", ["3"])]), ("soft", [("g01", "g01_a_b", ["1"])])):
        for gid, gdir, obs in groups:
            base = f"T/merged_final_image_{band}_w7d/{gdir}"
            img(f"{base}/f_{gid}_flux.img")
            if not (drop_group_row and band == "broad" and gid == "g01"):
                finals.append(dict(target="T", band=band, window_days="7", group_id=gid, chips="all", type="merged", obsids=";".join(obs),
                                   total_exposure_ks="5", flux_image=f"{base}/f_{gid}_flux.img", flux_image_original="", expmap_file="", mask_file=""))
            if len(obs) > 1:
                for o in obs:
                    img(f"{base}/f_{gid}_{o}_flux.img")
                    members.append(dict(target="T", band=band, window_days="7", group_id=gid, chips="all", obsid=o, exposure_ks="2.5",
                                        flux_image=f"{base}/f_{gid}_{o}_flux.img", flux_image_original="", expmap_file="", mask_file=""))
    for name, cols, rows in (("chanmerge_index_final.csv", fcols, finals), ("chanmerge_index_members.csv", mcols, members)):
        with open(tmp_path / name, "w", newline="") as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            w.writerows(rows)


def _texts(nodes):
    return [(n["text"], n["kind"], _texts(n["children"])) for n in nodes]


def test_list_images_gives_run_and_group_folders_and_members(tmp_path):
    _write_tree_run(tmp_path)
    items = V.list_images(str(tmp_path), include_members=True)
    assert [i["kind"] for i in items].count("member") == 2 and len(items) == 3 + 2
    g = next(i for i in items if i["kind"] == "final" and i["band"] == "broad" and i["group_id"] == "g01")
    assert g["run"] == "T/merged_final_image_broad_w7d" and g["group_dir"] == "T/merged_final_image_broad_w7d/g01_a_b"
    m = next(i for i in items if i["kind"] == "member" and i["obsid"] == "2")
    assert m["group_dir"] == g["group_dir"] and "2" in m["search"].split()
    assert len(V.list_images(str(tmp_path))) == 3                         # without members: only the final images


def test_build_tree_nests_members_under_their_group(tmp_path):
    _write_tree_run(tmp_path)
    tree = V.build_tree(V.list_images(str(tmp_path), include_members=True))
    assert _texts(tree) == [("T", "target", [
        ("broad | w7d", "run", [("g01", "group", [("obs 1", "obs", []), ("obs 2", "obs", [])]), ("s3", "group", [])]),
        ("soft | w7d", "run", [("g01", "group", [])])])]
    assert tree[0]["children"][0]["children"][0]["index"] is not None and tree[0]["index"] is None      # groups are images, folders are not


def test_filter_items_needs_every_word(tmp_path):
    _write_tree_run(tmp_path)
    items = V.list_images(str(tmp_path), include_members=True)
    assert V.filter_items(items, "") == list(range(len(items)))
    hits = V.filter_items(items, "BROAD 2")                               # case does not matter; the 2 is an obsid / group member
    assert {(items[i]["kind"], items[i]["obsid"] or items[i]["group_id"]) for i in hits} >= {("member", "2")}
    assert all("broad" in items[i]["search"] for i in hits)
    assert V.filter_items(items, "soft 2") == []                          # soft has no obs 2
    assert V.filter_items(items, "nothing like this") == []


def test_member_hit_pulls_in_its_group_and_a_member_without_group_row_gets_a_folder(tmp_path):
    _write_tree_run(tmp_path)
    items = V.list_images(str(tmp_path), include_members=True)
    only_member = [i for i, it in enumerate(items) if it["kind"] == "member" and it["obsid"] == "2"]
    tree = V.build_tree(items, only_member)
    run = tree[0]["children"][0]
    assert [(g["text"], g["index"] is not None, [c["text"] for c in g["children"]]) for g in run["children"]] == [("g01", True, ["obs 2"])]
    items2 = [it for it in items if not (it["kind"] == "final" and it["band"] == "broad" and it["group_id"] == "g01")]
    tree2 = V.build_tree(items2)
    grp = tree2[0]["children"][0]["children"][0]
    assert grp["text"] == "g01" and grp["index"] is None and [c["text"] for c in grp["children"]] == ["obs 1", "obs 2"]     # plain folder


def test_tree_widget_shows_search_hits_open_and_counts_them(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    try:
        _write_tree_run(tmp_path)
        v = V.Viewer(root, cwd=str(tmp_path))
        assert v.count.cget("text") == "5 of 5 images"
        targets = v.tree.get_children()
        assert len(targets) == 1 and v.tree.item(targets[0], "open")
        v.vars["search"].set("soft")
        assert v.count.cget("text") == "1 of 5 images"
        runs = v.tree.get_children(targets[0] if v.tree.exists(targets[0]) else v.tree.get_children()[0])
        assert len(runs) == 1 and "soft" in v.tree.item(runs[0], "text")
        v.vars["search"].set("obs-that-is-not-there")
        assert v.count.cget("text") == "0 of 5 images" and "Nothing matches" in v.empty.cget("text") and v.tree.get_children() == ()
        v.vars["search"].set("")
        assert v.count.cget("text") == "5 of 5 images" and v.empty.cget("text") == ""
        v.vars["search"].set("1")                                              # obsid 1: the group that holds it and its member row
        group_iids = [i for i in range(len(v.items)) if v.tree.exists(str(i))]
        assert any(v.items[i]["kind"] == "member" for i in group_iids) and any(v.items[i]["kind"] == "final" for i in group_iids)
    finally:
        root.destroy()


def test_folder_row_is_not_an_image_and_member_row_loads_its_own_file(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    try:
        _write_tree_run(tmp_path)
        v = V.Viewer(root, cwd=str(tmp_path))
        before = v.panels[0].path
        folder = v.tree.get_children()[0]
        v.tree.selection_set(folder)
        v.on_select()
        assert v.panels[0].path == before                                      # a target row does nothing
        mi = next(i for i, it in enumerate(v.items) if it["kind"] == "member" and it["obsid"] == "2")
        v.tree.selection_set(str(mi))
        v.on_select()
        assert v.panels[0].path.endswith("g01_2_flux.img") and "g01:2" in v.panels[0].title
    finally:
        root.destroy()


def test_refresh_button_rescans_the_run_folders_but_opening_the_tab_does_not(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    calls = []
    import chanmerge.tables as T
    monkeypatch.setattr(T, "rebuild_index", lambda r: calls.append(r))
    try:
        _write_tree_run(tmp_path)
        v = V.Viewer(root, cwd=str(tmp_path))
        v.refresh()
        assert calls == []                                                     # index exists: only re-read
        v.refresh(rescan=True)
        assert calls == [str(tmp_path)]
    finally:
        root.destroy()


# ---------- files opened by hand ----------
def _tk_root():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    return root


def _fits_file(path, shape=(6, 7)):
    fits.PrimaryHDU(np.arange(shape[0] * shape[1], dtype=float).reshape(shape)).writeto(path)
    return str(path)


def test_opened_list_roundtrip_dedupes_and_survives_a_damaged_file(tmp_path):
    a, b = _fits_file(tmp_path / "a.fits"), _fits_file(tmp_path / "b.fits")
    V.save_opened(str(tmp_path), [a, b, a, "/no/such/file.fits"])
    assert V.read_opened(str(tmp_path)) == [a, b, "/no/such/file.fits"]          # missing files stay remembered
    assert [it["missing"] for it in V.opened_items(V.read_opened(str(tmp_path)))] == [False, False, True]    # and are listed as missing
    assert V.read_opened(str(tmp_path / "no_such_folder")) == []
    (tmp_path / V.OPENED_FILE).write_text("not json {")
    assert V.read_opened(str(tmp_path)) == []                                    # a damaged file never crashes the viewer


def test_opened_items_mark_missing_files_instead_of_dropping_them(tmp_path):
    a = _fits_file(tmp_path / "a.fits")
    items = V.opened_items([a, str(tmp_path / "gone.fits")])
    assert [(it["info"] == a, it["missing"]) for it in items] == [(True, False), (False, True)]
    ok, gone = items
    assert ok["kind"] == "file" and ok["files"] == {"flux": a} and "a.fits" in ok["search"] and V.OPENED_NODE in ok["search"]
    assert gone["files"] == {}                                                  # nothing to draw
    tree = V.build_tree(items)
    assert [c["text"] for c in tree[0]["children"]] == ["a.fits", "gone.fits (not found)"]


def test_opened_files_form_their_own_folder_after_the_targets(tmp_path):
    _write_tree_run(tmp_path)
    f = _fits_file(tmp_path / "mine.fits")
    items = V.list_images(str(tmp_path), include_members=True) + V.opened_items([f])
    tree = V.build_tree(items)
    assert [n["text"] for n in tree] == ["T", "manually opened"] and tree[1]["kind"] == "opened"
    assert [(c["text"], c["kind"]) for c in tree[1]["children"]] == [("mine.fits", "file")] and tree[1]["index"] is None
    assert V.filter_items(items, "mine") == [len(items) - 1]


def test_open_file_lists_it_and_refresh_and_a_new_window_keep_it(tmp_path, monkeypatch):
    import chanmerge.tables as T
    import chanmerge.viewer as VM
    root = _tk_root()
    try:
        _write_tree_run(tmp_path)
        f = _fits_file(tmp_path / "far_away.fits", shape=(9, 11))
        monkeypatch.setattr(T, "rebuild_index", lambda r: None)                # the scan itself is not what is tested here
        monkeypatch.setattr(VM.filedialog, "askopenfilename", lambda **k: f)
        v = V.Viewer(root, cwd=str(tmp_path))
        v.open_file()
        assert v.data.shape == (9, 11) and v.panels[0].path == f              # shown in the active picture
        assert v.count.cget("text") == "6 of 6 images"
        folder = [i for i in v.tree.get_children() if v.tree.item(i, "text") == "manually opened"]
        assert len(folder) == 1 and v.tree.item(folder[0], "open")
        v.open_file()                                                          # same file again: not listed twice
        v.refresh(rescan=True)                                                 # the refresh button
        v.refresh()
        v.reload_list()
        assert [it["info"] for it in v.items if it["kind"] == "file"] == [f]
        v.vars["search"].set("far_away")
        assert v.count.cget("text") == "1 of 6 images"
        monkeypatch.setattr(VM.filedialog, "askopenfilename", lambda **k: "")   # cancel: nothing changes
        v.open_file()
        assert [it["info"] for it in v.items if it["kind"] == "file"] == [f]
        v.destroy()
        v2 = V.Viewer(root, cwd=str(tmp_path))                                 # the gui was closed and started again
        assert [it["info"] for it in v2.items if it["kind"] == "file"] == [f]  # still there, nobody removed it
        assert V.read_opened(str(tmp_path)) == [f]
    finally:
        root.destroy()


def test_going_to_the_chanmerge_tab_and_back_keeps_the_opened_files(tmp_path, monkeypatch):
    import chanmerge.viewer as VM
    from chanmerge.gui import App
    root = _tk_root()
    try:
        _write_tree_run(tmp_path)
        f = _fits_file(tmp_path / "mine.fits")
        monkeypatch.setattr(VM.filedialog, "askopenfilename", lambda **k: f)
        app = App(root, cwd=str(tmp_path))
        app.top.select(app.viewer_page)
        app.on_top_tab()                                                       # first visit builds the viewer
        app.viewer.open_file()
        app.top.select(app.run_page)
        app.on_top_tab()
        app.top.select(app.viewer_page)
        app.on_top_tab()                                                       # coming back re-reads the folder
        assert [it["info"] for it in app.viewer.items if it["kind"] == "file"] == [f]
        app2 = App(root, cwd=str(tmp_path))                                    # a new window = the gui was restarted
        app2.top.select(app2.viewer_page)
        app2.on_top_tab()
        assert [it["info"] for it in app2.viewer.items if it["kind"] == "file"] == [f]
    finally:
        root.destroy()


def test_remove_forgets_only_manually_opened_files(tmp_path, monkeypatch):
    import chanmerge.viewer as VM
    root = _tk_root()
    try:
        _write_tree_run(tmp_path)
        f = _fits_file(tmp_path / "mine.fits")
        monkeypatch.setattr(VM.filedialog, "askopenfilename", lambda **k: f)
        v = V.Viewer(root, cwd=str(tmp_path))
        v.open_file()
        v.tree.selection_set("0")                                              # a chanmerge group: cannot be removed
        v.forget_opened()
        assert "select a file" in v.status.get() and v.opened == [f]
        mi = next(i for i, it in enumerate(v.items) if it["kind"] == "file")
        v.tree.selection_set(str(mi))
        v.forget_opened()
        assert v.opened == [] and not any(it["kind"] == "file" for it in v.items)
        assert (tmp_path / "mine.fits").exists()                               # the file itself stays
        v.destroy()
        assert V.Viewer(root, cwd=str(tmp_path)).opened == []                  # removed by hand = gone for good
    finally:
        root.destroy()


def test_viewer_with_only_a_manually_opened_file_still_starts(tmp_path, monkeypatch):
    import chanmerge.viewer as VM
    root = _tk_root()
    try:
        f = _fits_file(tmp_path / "solo.fits")
        monkeypatch.setattr(VM.filedialog, "askopenfilename", lambda **k: f)
        v = V.Viewer(root, cwd=str(tmp_path))                                  # no chanmerge output in this folder
        assert v.items == [] and "No images found" in v.empty.cget("text")
        v.open_file()
        assert v.panels[0].path == f and v.count.cget("text") == "1 of 1 images"
    finally:
        root.destroy()


def test_a_moved_file_is_listed_as_not_found_and_picking_it_warns_with_its_path_and_removes_it(tmp_path, monkeypatch, warnings_seen):
    import chanmerge.viewer as vm
    root = _tk_root()
    try:
        _write_tree_run(tmp_path)
        a, b = _fits_file(tmp_path / "a.fits", (5, 6)), _fits_file(tmp_path / "away.fits", (8, 9))
        v = V.Viewer(root, cwd=str(tmp_path))
        for path in (b, a):
            monkeypatch.setattr(vm.filedialog, "askopenfilename", lambda path=path, **k: path)
            v.open_file()                                                        # b is drawn first (and cached), then a
        assert v.data.shape == (5, 6)
        (tmp_path / "away.fits").rename(tmp_path / "away_moved.fits")            # moved while the window is open
        mi = next(i for i, it in enumerate(v.items) if it["info"] == b)
        v.tree.selection_set(str(mi))
        v.on_select()                                                            # picking it: nothing drawn, not from the cache either
        assert v.data.shape == (5, 6) and v.panels[0].path == a and "a.fits" in v.panels[0].title
        (title, text), = warnings_seen                                           # one warning that names the file and its old path
        assert title == "File not found" and b in text and "not at its old path" in text and "taken off" in text
        assert "away.fits" in v.status.get()
        assert V.read_opened(str(tmp_path)) == [a] and [it["info"] for it in v.items if it["kind"] == "file"] == [a]     # and it is gone from the list
        # gui restarted with a moved file: listed as '(not found)' first, removed when somebody tries to draw it
        c = _fits_file(tmp_path / "later.fits")
        monkeypatch.setattr(vm.filedialog, "askopenfilename", lambda **k: c)
        v.open_file()
        v.destroy()
        (tmp_path / "later.fits").rename(tmp_path / "later_moved.fits")
        v2 = V.Viewer(root, cwd=str(tmp_path))
        gone = [it for it in v2.items if it["kind"] == "file" and it["missing"]]
        assert [it["info"] for it in gone] == [c] and "(1 not found)" in v2.count.cget("text")
        folder = next(i for i in v2.tree.get_children() if v2.tree.item(i, "text") == "manually opened")
        assert "later.fits (not found)" in [v2.tree.item(k, "text") for k in v2.tree.get_children(folder)]
        (tmp_path / "later_moved.fits").rename(tmp_path / "later.fits")          # it comes back before anybody clicks: normal again
        v2.refresh()
        assert not any(it.get("missing") for it in v2.items)
        (tmp_path / "later.fits").rename(tmp_path / "later_moved.fits")
        v2.refresh()
        v2.tree.selection_set(str(v2.items.index(next(it for it in v2.items if it["info"] == c))))
        warnings_seen.clear()
        v2.on_select()
        assert len(warnings_seen) == 1 and c in warnings_seen[0][1] and V.read_opened(str(tmp_path)) == [a]
        assert not any(it["kind"] == "file" and it["info"] == c for it in v2.items) and "not found" not in v2.count.cget("text")
    finally:
        root.destroy()


def test_a_missing_chanmerge_image_is_reported_but_not_removed_from_the_index_tree(tmp_path, warnings_seen):
    root = _tk_root()
    try:
        _write_tree_run(tmp_path)
        v = V.Viewer(root, cwd=str(tmp_path))
        gi = next(i for i, it in enumerate(v.items) if it["kind"] == "final" and it["group_id"] == "s3")
        target = v.items[gi]["files"]["flux"]
        os.remove(target)
        n = len(v.items)
        v.tree.selection_set(str(gi))
        v.on_select()
        (title, text), = warnings_seen
        assert target in text and "Refresh" in text and len(v.items) == n          # reported only; refresh rescans the run folders
        assert v.panels[0].path != target
    finally:
        root.destroy()


def test_a_new_window_starts_from_defaults_nothing_of_the_drawing_is_remembered(tmp_path, monkeypatch):
    import chanmerge.viewer as vm
    root = _tk_root()
    try:
        _write_tree_run(tmp_path)
        f = _fits_file(tmp_path / "mine.fits")
        monkeypatch.setattr(vm.filedialog, "askopenfilename", lambda **k: f)
        v = V.Viewer(root, cwd=str(tmp_path))
        v.open_file()
        v.vars["panels"].set("4 (2x2)")
        v.on_option("panels")
        v.vars["scale"].set("log")
        v.vars["cmap"].set("hot")
        v.vars["smooth"].set(True)
        v.vars["search"].set("soft")
        v.panels[0].ax.set_xlim(1, 2)
        v.destroy()                                                            # the gui is closed and started again
        v2 = V.Viewer(root, cwd=str(tmp_path))
        assert v2.n_panels() == 1 and v2.vars["scale"].get() == "linear" and v2.vars["cmap"].get() == "viridis"
        assert not v2.vars["smooth"].get() and v2.vars["search"].get() == "" and v2.active == 0
        assert [p.path is not None for p in v2.panels] == [True, False, False, False]        # only the automatic first image in picture 1
        assert v2.panels[0].item["kind"] == "final" and v2.panels[0].view is None
        assert [it["info"] for it in v2.items if it["kind"] == "file"] == [f]  # the opened-files list is the only thing kept
    finally:
        root.destroy()
