import pytest

from chanmerge.inputs import (InputError, parse_ra, parse_dec, format_ra, format_dec, parse_radius, default_target,
                              parse_date, check_date_order, check_folder_name, check_run_path, parse_yes_no,
                              parse_stage, parse_window, parse_arm_method, parse_stage_edit, parse_row_sets,
                              format_row_sets, parse_chip_menu, chip_choice, parse_chip_names, chip_overlaps,
                              parse_streak_sets, parse_streak_action, parse_group_edit)


# ---------- coordinates ----------
def test_ra_decimal_is_degrees_not_hours():
    # the old bug: 150.2 was read as hours
    assert parse_ra("150.2") == pytest.approx(150.2)
    assert parse_ra("201.365d") == pytest.approx(201.365)


def test_ra_sexagesimal_is_hours():
    assert parse_ra("13:25:27.6") == pytest.approx(201.365, abs=1e-6)
    assert parse_ra("13 25 27.6") == pytest.approx(201.365, abs=1e-6)
    assert parse_ra("13h25m27.6s") == pytest.approx(201.365, abs=1e-6)


@pytest.mark.parametrize("bad", ["", "abc", "360", "-1", "25:00:00", "13:61:00", "-13:00:00"])
def test_ra_rejects(bad):
    with pytest.raises(InputError):
        parse_ra(bad)


def test_dec():
    assert parse_dec("-43:01:09") == pytest.approx(-(43 + 1 / 60 + 9 / 3600))
    assert parse_dec("-43.019") == pytest.approx(-43.019)
    assert parse_dec("-00:30:00") == pytest.approx(-0.5)
    assert parse_dec("+10d") == pytest.approx(10)
    for bad in ["91", "-90.5", "x", "", "10:70:00"]:
        with pytest.raises(InputError):
            parse_dec(bad)


def test_formats_round_trip():
    assert format_ra(201.365) == "13:25:27.6"
    assert format_dec(-43.019) == "-43:01:08"
    assert format_dec(0.25) == "+00:15:00"
    assert format_ra(359.99999) in ("23:59:60.0", "00:00:00.0", "23:59:59.9", "24:00:00.0") or format_ra(359.99999).startswith("23:59")


def test_radius():
    assert parse_radius("5") == 5
    for bad in ["0", "-1", "x", "", "500"]:
        with pytest.raises(InputError):
            parse_radius(bad)


def test_default_target_from_parsed_coordinates():
    t = default_target(201.365, -43.019)
    assert t == "obs_201.3650_-43.0190"
    assert check_folder_name(t)[0] == "ok"


# ---------- dates ----------
def test_dates():
    assert parse_date("") is None
    d1, d2 = parse_date("2005-01-01"), parse_date("2010-12-31")
    check_date_order(d1, d2)
    with pytest.raises(InputError):
        check_date_order(d2, d1)
    with pytest.raises(InputError):
        parse_date("2010-13-01")
    with pytest.raises(InputError):
        parse_date("31/12/2010")


# ---------- path characters ----------
@pytest.mark.parametrize("c", list(" ,;([\\@&'\"=)]<>*?$`|"))
def test_folder_name_forbidden(c):
    level, msg = check_folder_name(f"Cen{c}A")
    assert level == "error"
    assert "Not allowed" in msg


@pytest.mark.parametrize("c", list("^{}"))
def test_folder_name_warn(c):
    assert check_folder_name(f"Cen{c}A")[0] == "warn"


# non-ascii letters are allowed (s-cedilla and e-acute passed chandra_repro and merge_obs in the path tests)
@pytest.mark.parametrize("name", ["CenA", "cen-a_2", "a.b+c~d%e:f#g!k", "\u015fehir", "\u00e9"])
def test_folder_name_allowed(name):
    assert check_folder_name(name) == ("ok", "")


def test_folder_name_other():
    assert check_folder_name("a/b")[0] == "error"
    assert check_folder_name("..")[0] == "error"
    assert check_folder_name("a\tb")[0] == "warn"   # unlisted character


def test_run_path():
    assert check_run_path("/Users/x/CAFIG/run")[0] == "ok"
    assert check_run_path("/Users/x/*CAFIG/run")[0] == "error"   # merge_obs expands * and ? itself (test o)
    assert check_run_path("/Users/x/my data")[0] == "error"
    assert check_run_path("/Users/x/a=b")[0] == "error"   # merge_obs fails on '=' (test i)
    assert check_run_path("/Users/x/a^b")[0] == "warn"
    for c in ")]<>$`|":
        assert check_run_path(f"/Users/x/a{c}b")[0] == "error"   # chandra_repro fails on these in the root path (test o)


# ---------- small answers ----------
def test_small_answers():
    assert parse_yes_no(" Y ") is True and parse_yes_no("no") is False
    with pytest.raises(InputError):
        parse_yes_no("")
    assert parse_stage("FLUX") == "flux"
    with pytest.raises(InputError):
        parse_stage("flux", allow_flux=False)
    assert parse_window("") == 0 and parse_window("1.5") == 1.5
    for bad in ["-1", "x"]:
        with pytest.raises(InputError):
            parse_window(bad)
    assert parse_arm_method("4") == 4
    for bad in ["", "0", "5", "x"]:
        with pytest.raises(InputError):
            parse_arm_method(bad)


def test_stage_edit():
    ids = {"103", "104", "105"}
    assert parse_stage_edit("done", ids) == "done"
    assert parse_stage_edit("105 flux", ids) == [("105", "flux")]
    assert parse_stage_edit("103 104 none", ids) == [("103", "none"), ("104", "none")]
    assert parse_stage_edit("00105 flux", ids) == [("105", "flux")]
    for bad in ["105", "999 flux", "105 fly", ""]:
        with pytest.raises(InputError):
            parse_stage_edit(bad, ids)


# ---------- row sets ----------
def test_row_sets():
    rows = [1, 2, 3, 4]
    assert parse_row_sets("separate", rows) == [[1], [2], [3], [4]]
    assert parse_row_sets("1+3, 2, 4", rows) == [[1, 3], [2], [4]]
    assert parse_row_sets("1+2, rest", rows) == [[1, 2], [3], [4]]
    assert parse_row_sets("3+1,rest", rows) == [[1, 3], [2], [4]]
    assert parse_row_sets("1+2+3+4", rows) == [[1, 2, 3, 4]]
    assert format_row_sets([[1, 3], [2]]) == "1+3, 2"


def test_row_set_messages_plural():
    with pytest.raises(InputError, match="rows 1, 2 are missing"):
        parse_row_sets("3", [1, 2, 3])
    with pytest.raises(InputError, match="row 2 is missing"):
        parse_row_sets("1, 3", [1, 2, 3])


@pytest.mark.parametrize("bad,word", [("1+3", "missing"), ("1+3, 3, 2, 4", "more than once"), ("1, 2, 3, 5", "not in the table"),
                                      ("1,,2", "empty"), ("a+b", "not a row"), ("", "write"), ("rest, rest", "twice")])
def test_row_sets_errors(bad, word):
    with pytest.raises(InputError, match=word):
        parse_row_sets(bad, [1, 2, 3, 4])


# ---------- chips ----------
def test_chip_menu_and_choices():
    assert parse_chip_menu("2") == "i" and parse_chip_menu("custom") == "custom"
    for bad in ["", "0", "8", "x"]:
        with pytest.raises(InputError):
            parse_chip_menu(bad)
    assert chip_choice("i").sets[0].chips == (0, 1, 2, 3)
    assert chip_choice("bi").sets[0].chips == (5, 7)
    sep = chip_choice("separate").sets
    assert [s.label for s in sep] == ["bi", "fi"]
    assert chip_choice("all").sets[0].chips is None


def test_chip_names():
    one = parse_chip_names("s3")
    assert one[0].chips == (7,) and one[0].label == ""
    two = parse_chip_names("S3 / S1,S3")
    assert [s.chips for s in two] == [(7,), (5, 7)]
    assert [s.label for s in two] == ["S3", "S1_S3"]
    assert parse_chip_names("I3,I2")[0].chips == (2, 3)          # single I chips work like S chips
    assert chip_overlaps(two) == ["S3"]
    assert two[1].ciao_filter() == "[ccd_id=5,7]"
    for bad in ["", "S6", "S3 /", "S3 / S3"]:
        with pytest.raises(InputError):
            parse_chip_names(bad)


# ---------- streak ----------
def test_streak_answers():
    assert parse_streak_sets("", ["A", "B"]) == []
    assert parse_streak_sets("all", ["A", "B"]) == ["A", "B"]
    assert parse_streak_sets("a, b a", ["A", "B"]) == ["A", "B"]
    with pytest.raises(InputError):
        parse_streak_sets("C", ["A", "B"])
    assert parse_streak_action(" Mask ") == "mask"
    for bad in ["leave", "", "x"]:
        with pytest.raises(InputError):
            parse_streak_action(bad)


# ---------- group edits ----------
def test_group_edit_syntax():
    assert parse_group_edit("done") == ("done",)
    assert parse_group_edit("move 102 g03") == ("move", "102", "g03")
    assert parse_group_edit("move 102 s") == ("move", "102", "s")
    assert parse_group_edit("add 102 g03") == ("add", "102", "g03")
    assert parse_group_edit("new 101 0102") == ("new", ["101", "102"])
    assert parse_group_edit("drop 102") == ("drop", "102", None)
    assert parse_group_edit("drop 102 g03") == ("drop", "102", "g03")
    assert parse_group_edit("drop all g03") == ("drop_all", "g03")
    assert parse_group_edit("delete g03") == ("delete", "g03")
    assert parse_group_edit("g01 chips S3 / S1,S3") == ("chips", "g01", "S3 / S1,S3")
    for bad in ["", "move 102", "jump 1 2", "delete"]:
        with pytest.raises(InputError):
            parse_group_edit(bad)
