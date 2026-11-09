# the line pinned to the bottom of the terminal (and its use by the real run): escapes written, nothing outside a terminal, no leftovers
import os

import pytest

import chanmerge.cli as cli
from chanmerge.cli import StatusLine, Terminal
from chanmerge.run import standby_status, HeaderProblem


class Tty:
    def __init__(self, tty=True):
        self.out, self.tty = [], tty

    def isatty(self):
        return self.tty

    def write(self, s):
        self.out.append(s)

    def flush(self):
        pass

    def text(self):
        return "".join(self.out)


@pytest.fixture
def term(monkeypatch):
    monkeypatch.setattr(cli.shutil, "get_terminal_size", lambda *a, **k: os.terminal_size((80, 24)))
    monkeypatch.setenv("TERM", "xterm-256color")


def test_set_pins_the_text_to_the_last_row_and_keeps_the_rest_scrolling(term):
    s = Tty()
    sl = StatusLine(stream=s)
    sl.set("[!] STANDBY: g01")
    t = s.text()
    assert "\033[1;23r" in t                                   # scroll region = rows 1..23, the last row is not part of it
    assert "\033[24;1H" in t and "[!] STANDBY: g01" in t and "\033[7m" in t        # written on row 24, reversed colours
    assert t.index("\033[1;23r") < t.index("[!] STANDBY") < t.rindex("\0338")      # region set, text drawn, cursor put back
    assert sl.on


def test_clear_gives_the_whole_screen_back_and_is_safe_to_repeat(term):
    s = Tty()
    sl = StatusLine(stream=s)
    sl.clear()
    assert s.out == []                                          # nothing was set: nothing to undo
    sl.set("x")
    s.out.clear()
    sl.clear()
    t = s.text()
    assert "\033[r" in t and "\033[24;1H\033[2K" in t and not sl.on
    s.out.clear()
    sl.clear()
    assert s.out == []
    sl.set("x")
    s.out.clear()
    sl.set(None)                                                # an empty text is the same as clear
    assert "\033[r" in s.text() and not sl.on


def test_nothing_is_written_outside_a_real_terminal(term, monkeypatch):
    for stream, termvar in ((Tty(tty=False), "xterm"), (Tty(), "dumb"), (Tty(), "")):
        monkeypatch.setenv("TERM", termvar)
        sl = StatusLine(stream=stream)
        sl.set("hello")
        sl.clear()
        assert stream.out == [] and not sl.on


def test_a_long_text_is_cut_to_the_width_and_a_tiny_terminal_is_left_alone(term, monkeypatch):
    s = Tty()
    StatusLine(stream=s).set("a" * 200)
    assert "a" * 79 in s.text() and "a" * 80 not in s.text()    # one column less than the width, no wrapping
    monkeypatch.setattr(cli.shutil, "get_terminal_size", lambda *a, **k: os.terminal_size((80, 3)))
    s2 = Tty()
    StatusLine(stream=s2).set("hello")
    assert s2.out == []


def test_the_hold_text_names_every_group_once_and_says_in_plain_words_what_happens():
    one = standby_status([HeaderProblem("cc", "13197", "g01", True, "x")])
    assert one == "[!] g01 is on hold (problem). You will be asked about it after the other jobs." and len(one) < 80
    many = standby_status([HeaderProblem("cc", "1", "g01", True, "x"), HeaderProblem("cc", "2", "g02", True, "x"), HeaderProblem("cc", "3", "g01", True, "x")])
    assert many.startswith("[!] g01, g02 are on hold") and many.count("g01") == 1       # plural, every group once, groups first so a cut keeps them


def test_terminal_passes_status_only_to_the_real_run_and_only_the_real_screen_gets_a_pinned_line():
    assert "status" in Terminal().run_kw()                      # run_fn=None means the real run()
    t = Terminal(run_fn=lambda *a, **k: None)
    assert t.run_kw() == {}                                     # a test double of run() is not told about it
    quiet = Terminal(out=lambda s="": None)                     # custom output: its status line writes into the void
    quiet.status.set("never shown")
    assert not quiet.status.usable() or quiet.status.stream is not cli.sys.stdout


def test_resize_draws_the_line_again_on_the_new_last_row(monkeypatch):
    # test r (session 005, real pseudo terminal): after a resize the line stayed on its old row in the middle of the screen
    monkeypatch.setenv("TERM", "xterm-256color")
    size = {"v": (80, 24)}
    monkeypatch.setattr(cli.shutil, "get_terminal_size", lambda *a, **k: os.terminal_size(size["v"]))
    s = Tty()
    sl = StatusLine(stream=s)
    sl.set("[!] g01 is on hold")
    assert "\033[24;1H" in s.text()
    s.out.clear()
    size["v"] = (100, 36)
    sl._resized()                                              # what the SIGWINCH handler calls
    t = s.text()
    assert "\033[24;1H\033[2K" in t                            # the old row is cleared
    assert "\033[1;35r" in t and "\033[36;1H" in t and "[!] g01 is on hold" in t      # pinned again on row 36
    assert sl.on and sl.rows == 36
    sl.clear()
    assert not sl.on


def test_resize_does_nothing_when_no_line_is_up(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(cli.shutil, "get_terminal_size", lambda *a, **k: os.terminal_size((80, 24)))
    s = Tty()
    sl = StatusLine(stream=s)
    sl._resized()
    assert s.out == []
