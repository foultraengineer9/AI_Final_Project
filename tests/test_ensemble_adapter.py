"""Tests for the adapter onto Afia's research ensemble.

The ensemble itself needs sentence-transformers, spacy and tensorflow plus a
420 MB embedding download, so these tests exercise the adapter against a stub
predictor that returns an audit log in her documented shape. That covers the
parts we own — tier mapping, normalisation, graceful degradation, and the
agreement calculation — without requiring the heavy stack to be installed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import afia_ensemble as ae  # noqa: E402
import main_app as m  # noqa: E402


class StubPredictor:
    """Mimics WordComplexityPredictor.analyze_passage's return contract."""

    def __init__(self, audit, reconstructed=""):
        self._audit, self._reconstructed = audit, reconstructed
        self.called_with = None

    def analyze_passage(self, passage, target_comfort_level=1):
        self.called_with = target_comfort_level
        return self._reconstructed, self._audit


def _row(word, tier, status="OK", protected=False):
    return {"word": word, "pos": "NN", "predicted_tier": tier,
            "status": status, "suggestion": None, "is_protected": protected}


# --- tier mapping ------------------------------------------------------------

def test_three_tiers_map_onto_cefr_band_pairs():
    assert ae.TIER_TO_CEFR == {0: "A1-A2", 1: "B1-B2", 2: "C1-C2"}


@pytest.mark.parametrize("level,tier", [(1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (6, 2)])
def test_reader_level_maps_to_comfort_tier(level, tier):
    """Her model takes a 3-tier comfort level; our interface speaks in 6 levels."""
    stub = StubPredictor([_row("word", 1)])
    ae.analyse(stub, "word", level)
    assert stub.called_with == tier


# --- normalisation -----------------------------------------------------------

def test_punctuation_and_whitespace_are_dropped():
    audit = [_row("Space", 0), _row(".", 0), _row(" ", 0), _row("gigantic", 2)]
    words, _ = ae.analyse(StubPredictor(audit), "Space. gigantic", 3)
    assert [w.word for w in words] == ["Space", "gigantic"]


def test_challenging_status_and_band_are_exposed():
    audit = [_row("ablation", 2, status="CHALLENGING"), _row("ice", 0)]
    words, _ = ae.analyse(StubPredictor(audit), "ablation ice", 3)
    hard = {w.word: w for w in words}
    assert hard["ablation"].challenging is True
    assert hard["ablation"].band == "C1-C2"
    assert hard["ice"].challenging is False


def test_unchanged_reconstruction_is_reported_as_none():
    """No Groq key means the passage comes back untouched; that isn't a rewrite."""
    passage = "Space is gigantic."
    _, recon = ae.analyse(StubPredictor([_row("Space", 0)], passage), passage, 3)
    assert recon is None


def test_real_reconstruction_is_passed_through():
    _, recon = ae.analyse(
        StubPredictor([_row("Space", 0)], "Space is huge."), "Space is gigantic.", 3)
    assert recon == "Space is huge."


# --- agreement between the two models ---------------------------------------

def test_agreement_counts_both_directions():
    vocab = __import__("pandas").DataFrame(m.DEMO_VOCAB, columns=["word", "level"])
    tokens = m.LogicEngine(vocab, None).analyse("The gigantic distant sun.", 2, set())
    # ensemble agrees on gigantic, disagrees on sun
    audit = [_row("gigantic", 2, status="CHALLENGING"),
             _row("distant", 1), _row("sun", 2, status="CHALLENGING")]
    words, _ = ae.analyse(StubPredictor(audit), "x", 2)
    stats = ae.agreement(words, tokens)
    assert stats["compared"] == 3
    assert stats["only_ensemble"] == 1          # sun: ensemble flags, we don't
    assert 0.0 <= stats["rate"] <= 1.0


def test_agreement_on_empty_input_does_not_divide_by_zero():
    assert ae.agreement([], [])["rate"] == 0.0


# --- graceful degradation ----------------------------------------------------

def test_missing_bundle_reports_actionable_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(ae, "BUNDLE_DIR", tmp_path / "nope")
    ok, reason = ae.bundle_status()
    assert ok is False and "wordify.zip" in reason


def test_incomplete_bundle_names_the_missing_files(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(ae, "BUNDLE_DIR", tmp_path)
    ok, reason = ae.bundle_status()
    assert ok is False and "rf_clf.joblib" in reason


def test_load_predictor_raises_readable_error_when_bundle_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(ae, "BUNDLE_DIR", tmp_path / "nope")
    with pytest.raises(RuntimeError, match="not found"):
        ae.load_predictor()


# --- 3-tier mode ---------------------------------------------------------------

def test_tier_names_and_colours_cover_all_three():
    import main_app as m
    assert set(m.TIER_NAMES) == {0, 1, 2}
    assert set(m.TIER_COLOURS) == {0, 1, 2}


def test_tier_ladder_marks_the_readers_tier():
    import main_app as m, re
    html_out = m.render_tier_ladder(1)
    classes = re.findall(r"lexis-rung ([a-z]*)'", html_out)
    assert classes == ["past", "here", "next"]
    assert "Intermediate" in html_out


def test_tier_passage_highlights_only_words_above_tier():
    import main_app as m
    from dataclasses import dataclass

    @dataclass
    class T:
        text: str
        is_word: bool = False
        tier: int = 0
        status: str = "known"
        protected: bool = False

    tokens = [T("Ice", True, 0, "known"), T(" "), T("ablation", True, 2, "above")]
    out = m.render_tier_passage(tokens)
    assert out.count("w-tier") == 1
    assert "ablation" in out and ">Ice<" not in out


def test_tier_passage_escapes_html():
    import main_app as m
    from dataclasses import dataclass

    @dataclass
    class T:
        text: str
        is_word: bool = False
        tier: int = 0
        status: str = "known"
        protected: bool = False

    out = m.render_tier_passage([T("<script>", False)])
    assert "<script>" not in out and "&lt;script&gt;" in out
