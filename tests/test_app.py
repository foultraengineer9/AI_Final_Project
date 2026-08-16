"""Smoke tests for main_app.py using Streamlit's AppTest harness.

Run from the project root:  python -m pytest tests/ -v
These assert the app renders and degrades gracefully; they do not need any
teammate artifacts to be present.
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "main_app.py"


def _run():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    return at


def test_app_renders_without_exceptions():
    at = _run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.tabs) == 4


def test_placement_test_scores_and_places_user():
    at = _run()
    for cb in at.checkbox[:12]:
        cb.check()
    [b for b in at.button if b.label == "Score my test"][0].click()
    at.run()
    assert not at.exception
    assert any("Suggested starting level" in s.value for s in at.success)


def test_marking_a_word_learned_updates_state():
    at = _run()
    got = [b for b in at.button if b.label == "Got it"]
    assert got, "reading view should surface at least one word to learn"
    got[0].click()
    at.run()
    assert not at.exception
    assert len(at.session_state["learned"]) == 1
