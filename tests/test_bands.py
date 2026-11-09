import pytest

from chanmerge.bands import BandError, parse_bands, acis_band, hrc_band, bands_overlap, energy_range_kev


def test_named_and_csc():
    bands, notes = parse_bands("broad")
    assert [b.name for b in bands] == ["broad"] and notes == []
    bands, notes = parse_bands("csc")
    assert [b.name for b in bands] == ["soft", "medium", "hard"]


def test_identical_numeric_is_renamed_with_note():
    bands, notes = parse_bands("0.5:7:2.3")
    assert bands[0].name == "broad" and bands[0].folder_label == "broad"
    assert "identical to ciao's 'broad'" in notes[0]


def test_same_lo_hi_other_eff_stays_numeric_without_warning():
    bands, notes = parse_bands("0.5:7:1.5")
    b = bands[0]
    assert b.name == "0.5:7:1.5" and notes == []
    assert b.folder_label == "0.5_7_1.5" and b.file_label == "0.5-7" and b.merge_spec == "0.5:7:1.5"


def test_duplicates_run_once():
    bands, notes = parse_bands("csc, soft")
    assert [b.name for b in bands] == ["soft", "medium", "hard"]
    assert any("requested twice (csc already includes it)" in n for n in notes)
    bands, notes = parse_bands("broad, 0.5:7:2.3")
    assert [b.name for b in bands] == ["broad"]


@pytest.mark.parametrize("bad", ["0.5:7", "0.5:7:", "7:0.5:2", "0.5:7:9", "x:7:2", "huge", "wide", "", "0.01:7:2"])
def test_acis_invalid(bad):
    # 7:0.5:2 and 0.5:7:9 run in ciao without error, so chanmerge must catch them
    with pytest.raises(BandError):
        parse_bands(bad)


def test_hrc():
    b, _ = hrc_band("wide")
    assert b.name == "wide"
    b, _ = hrc_band("30:200:1.1")
    assert (b.lo, b.hi, b.eff) == (30, 200, 1.1)
    assert "PI 30-200" in b.describe() and "1.1 keV" in b.describe()
    b, _ = hrc_band("::1.1")
    assert b.lo is None
    for bad in ["broad", "csc", "30:200", "30.5:200:1", "200:30:1", "30::1", "30:200:", "30:200:50"]:
        with pytest.raises(BandError):
            hrc_band(bad)


def test_describe_units():
    b, _ = acis_band("broad")
    assert b.describe() == "ACIS band 0.5-7 keV, exposure map at 2.3 keV"


def test_overlap():
    assert bands_overlap("broad", "soft")
    assert not bands_overlap("soft", "hard")
    assert bands_overlap("0.5:2:1", "soft")
    assert not bands_overlap("0.5:1.2:0.9", "1.2:2:1.5")
    assert energy_range_kev("wide") is None and bands_overlap("wide", "30:200:1.1")
