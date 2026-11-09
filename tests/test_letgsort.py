# ACIS + LETG imaging copy with a narrow arm box (session 007, test x)
import numpy as np
import pytest
from astropy.io import fits

from chanmerge import letgsort as LS
from chanmerge import masks as M


def region_file(path, width=2899.0):
    cols = [fits.Column(name="SHAPE", format="16A", array=np.array(["Circle", "Rotbox"])),
            fits.Column(name="X", format="D", array=np.array([4096.5, 4096.5])),
            fits.Column(name="Y", format="D", array=np.array([4096.5, 4096.5])),
            fits.Column(name="R", format="2D", array=np.array([[31.0, 0.0], [9888.0, width]])),
            fits.Column(name="ROTANG", format="D", array=np.array([0.0, 24.8]))]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols, name="REGION")]).writeto(path)
    return path


def test_narrow_mask_changes_only_the_box_width(tmp_path):
    src = region_file(str(tmp_path / "ciao_tgmask.fits"))
    out = LS.narrow_mask(src, str(tmp_path / "w200.fits"), 200.0)
    reg, ciao = M.read_region_file(out), M.read_region_file(src)
    assert reg["boxes"][0]["width"] == 200.0 and ciao["boxes"][0]["width"] == 2899.0      # the original is not changed
    assert reg["r"] == ciao["r"] == 31.0 and reg["boxes"][0]["length"] == 9888.0 and reg["boxes"][0]["angle"] == pytest.approx(24.8)


def events(path, tg_m, time):
    n = len(tg_m)
    cols = [fits.Column(name="time", format="D", array=np.asarray(time, float)),
            fits.Column(name="x", format="E", array=np.linspace(4000, 4200, n)),
            fits.Column(name="y", format="E", array=np.full(n, 4096.0)),
            fits.Column(name="energy", format="E", array=np.full(n, 1000.0)),
            fits.Column(name="pi", format="J", array=np.arange(n)),
            fits.Column(name="tg_m", format="I", array=np.asarray(tg_m, np.int16)),
            fits.Column(name="tg_part", format="I", array=np.where(np.asarray(tg_m) == 99, 99, 3).astype(np.int16))]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols, name="EVENTS")]).writeto(path)
    return path


def test_copy_tg_columns_keeps_everything_else_and_refuses_other_rows(tmp_path):
    t = np.arange(6) * 3.2
    orig = events(str(tmp_path / "repro_evt2.fits"), [5, -3, 2, 0, 7, 1], t)
    res = events(str(tmp_path / "resolved.fits"), [99, 99, 2, 0, 99, 1], t)
    out = LS.copy_tg_columns(orig, res, str(tmp_path / "imaging_evt.fits"))
    with fits.open(out) as h:
        d = h["EVENTS"].data
        assert list(d["tg_m"]) == [99, 99, 2, 0, 99, 1] and list(d["tg_part"]) == [99, 99, 3, 3, 99, 3]
        assert list(d["pi"]) == list(range(6)) and "FOR IMAGES ONLY" in "".join(h["EVENTS"].header["HISTORY"])
    assert list(fits.getdata(orig, "EVENTS")["tg_m"]) == [5, -3, 2, 0, 7, 1]               # the repro file is untouched
    other = events(str(tmp_path / "other.fits"), [99] * 5, t[:5])
    with pytest.raises(ValueError):
        LS.copy_tg_columns(orig, other, str(tmp_path / "x.fits"))


def test_choose_width_rules():
    # test x: 1790 leaks at 100 px, none at 200 -> 200
    assert LS.choose_width([(100.0, "leak", 0.54), (200.0, "none", 0.06)])[0] == 200.0
    # capella: the leak shrinks 100 -> 200 (0.29 -> 0.13), none at 300 -> 300
    assert LS.choose_width([(100.0, "leak", 0.29), (200.0, "leak", 0.13), (300.0, "none", 0.0)])[0] == 300.0
    # 2324: 0.51 -> 0.29 shrinks, 0.29 -> 0.29 does not: not arm light, 200 is kept and the excess subtracted
    w, why = LS.choose_width([(100.0, "leak", 0.51), (200.0, "leak", 0.29), (300.0, "leak", 0.29)])
    assert w == 200.0 and "did not shrink" in why
    # still leaking (and shrinking) at the widest allowed box
    assert LS.choose_width([(100.0, "leak", 0.9), (200.0, "leak", 0.5)])[0] == 200.0
    # nothing measurable (subarray strip) -> the default
    assert LS.choose_width([(100.0, "unmeasurable", 0.0)])[0] == LS.DEFAULT_WIDTH
    assert LS.choose_width([])[0] == LS.DEFAULT_WIDTH
    # the narrowest box already clean
    assert LS.choose_width([(100.0, "none", 0.0)])[0] == 100.0


def test_bin_image_events_uses_only_image_events(tmp_path):
    from chanmerge.geometry import image_sky_grid
    p = events(str(tmp_path / "e.fits"), [99, 0, 2, 99], [0, 1, 2, 3])
    h = fits.Header()
    h["CRVAL1P"], h["CRPIX1P"], h["CDELT1P"] = 3990.0, 0.5, 8.0
    h["CRVAL2P"], h["CRPIX2P"], h["CDELT2P"] = 4000.0, 0.5, 8.0
    c, x, y = LS.bin_image_events(p, h, (30, 30))
    assert c.sum() == 3 and len(x) == 3                     # tg_m 2 is a dispersed event and stays out
