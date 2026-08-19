"""Unit tests for the pure logic in main_app.py (no Streamlit runtime needed)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main_app as m  # noqa: E402


@pytest.fixture(scope="module")
def vocab():
    return pd.DataFrame(m.DEMO_VOCAB, columns=["word", "level"])


@pytest.fixture(scope="module")
def engine(vocab):
    return m.LogicEngine(vocab, None)


#tokenising...

def test_tokenising_is_lossless(engine):
    """Every character must survive, or the rendered passage won't match the source."""
    text = m.DEMO_STORIES[0]["text"]
    tokens = engine.analyse(text, user_level=2, learned=set())
    assert "".join(t.text for t in tokens) == text


def test_words_are_classified_relative_to_user_level(engine):
    text = "The gigantic distant sun."          # gigantic=B1(3), distant=B1(3)...
    at_a2 = {t.text: t.status for t in engine.analyse(text, 2, set()) if t.is_word}
    at_c2 = {t.text: t.status for t in engine.analyse(text, 6, set()) if t.is_word}
    assert at_a2["gigantic"] == "learn"          # exactly one level above = learn
    assert at_c2["gigantic"] == "known"          # below the reader = unmarked


def test_learned_words_stop_being_highlighted(engine):
    text = "The gigantic sun."
    assert [t.status for t in engine.analyse(text, 2, set()) if t.text == "gigantic"] == ["learn"]
    assert [t.status for t in engine.analyse(text, 2, {"gigantic"}) if t.text == "gigantic"] == ["known"]


#rendering...

def test_rendering_escapes_html(engine):
    """A passage containing markup must not be able to inject it into the page."""
    tokens = engine.analyse("A <script>alert(1)</script> word", 1, set())
    out = m.render_passage(tokens, {})
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_rendering_applies_both_highlight_classes(engine):
    tokens = engine.analyse(m.DEMO_STORIES[0]["text"], 2, set())
    out = m.render_passage(tokens, m.DEMO_STORIES[0]["synonyms"])
    assert "w-learn" in out and "w-stretch" in out


#features...

@pytest.mark.parametrize("word,expected", [
    ("dog", 1), ("water", 2), ("happy", 2), ("gigantic", 3), ("meticulous", 4),
])
def test_syllable_counting(word, expected):
    assert m.count_syllables(word) == expected


def test_heuristic_is_within_one_level_for_most_of_the_demo_vocab(vocab):
    """Calibration guard: the fallback was tuned to ~86% within-one-band."""
    errors = [abs(m.heuristic_level(w) - lvl) for w, lvl in zip(vocab.word, vocab.level)]
    within_one = sum(e <= 1 for e in errors) / len(errors)
    assert within_one >= 0.80, f"heuristic drifted: only {within_one:.0%} within one level"


#placement test..

def test_placement_includes_pseudoword_catch_trials(vocab):
    items = m.build_placement_items(vocab)
    assert sum(not i["real"] for i in items) == m.N_PSEUDOWORDS
    assert sum(i["real"] for i in items) > 0


@pytest.mark.parametrize("true_level", [1, 2, 3, 4, 5, 6])
def test_an_honest_taker_is_placed_at_their_true_level(vocab, true_level):
    items = m.build_placement_items(vocab)
    answers = {i["word"]: bool(i["real"] and i["level"] <= true_level) for i in items}
    assert m.score_placement(items, answers)["suggested"] == true_level


def test_over_claiming_is_detected_and_penalised(vocab):
    """Someone who ticks everything must be flagged, not placed at C2."""
    items = m.build_placement_items(vocab)
    result = m.score_placement(items, {i["word"]: True for i in items})
    assert result["reliable"] is False
    assert result["false_alarm_rate"] == 1.0


def test_a_blank_test_places_at_a1(vocab):
    items = m.build_placement_items(vocab)
    assert m.score_placement(items, {})["suggested"] == 1


#vocabulary loading (regressions found against the real dataset).

def _load_from(tmp_path, monkeypatch, csv_text):
    """Point the loader at a temporary CSV and read it."""
    path = tmp_path / "processed_vocab.csv"
    path.write_text(csv_text)
    monkeypatch.setattr(m, "VOCAB_PATH", path)
    
    m.load_vocab.clear()
    return m.load_vocab()


def test_string_cefr_levels_load(tmp_path, monkeypatch):
    """Regression: pandas >=3.0 types text columns as `str`, not `object`.

    A dtype check for `object` sent real 'A1'/'B2' values down the numeric
    branch, NaN'd every row, and silently fell back to demo data.
    """
    df, src, err = _load_from(tmp_path, monkeypatch,
                              "headword,CEFR\nabandon,B1\ndog,A1\nobfuscate,C2\n")
    assert err is None and src == "processed_vocab.csv"
    assert dict(zip(df.word, df.level)) == {"abandon": 3, "dog": 1, "obfuscate": 6}


def test_numeric_level_column_is_preferred(tmp_path, monkeypatch):
    df, _, err = _load_from(tmp_path, monkeypatch,
                            "headword,CEFR,CEFR_numeric\ndog,A1,1\ncat,B2,4\n")
    assert err is None
    assert dict(zip(df.word, df.level)) == {"dog": 1, "cat": 4}


def test_duplicate_words_resolve_to_lowest_level(tmp_path, monkeypatch):
    """'above' is listed A1 and B1; the reader shouldn't be shown it as new."""
    df, _, _ = _load_from(tmp_path, monkeypatch,
                          "headword,CEFR\nabove,B1\nabove,A1\naboard,C1\naboard,B1\n")
    assert dict(zip(df.word, df.level)) == {"above": 1, "aboard": 3}


def test_slash_variants_are_split_into_separate_entries(tmp_path, monkeypatch):
    """'adviser/advisor' is one row but two spellings a reader might meet."""
    df, _, _ = _load_from(tmp_path, monkeypatch,
                          "headword,CEFR\nadviser/advisor,B2\nagonize/agonise,C1\n")
    assert set(df.word) == {"adviser", "advisor", "agonize", "agonise"}


def test_unreadable_vocab_falls_back_with_an_error_message(tmp_path, monkeypatch):
    df, src, err = _load_from(tmp_path, monkeypatch, "colA,colB\n1,2\n")
    assert src == "demo" and err is not None and "could not be read" in err
    assert not df.empty


#simplified passage view!

@pytest.mark.parametrize("word,gloss,safe", [
    ("persistent", "lasting", True),      
    ("dense", "heavy", True),
    ("moving", "go", False),              
    ("existing", "be", False),            
    ("attached", "attach", False),        
    ("temperatures", "temperature", False),
])
def test_substitution_safety_filter(word, gloss, safe):
    assert m.substitution_is_safe(word, gloss) is safe


def test_simplified_view_only_replaces_highlighted_words(engine):
    """A word at or below the reader's level must survive untouched."""
    text = "The gigantic sun."
    tokens = engine.analyse(text, 6, set())         
    html_out, swapped = m.render_simplified(tokens, {"gigantic": "huge"})
    assert swapped == 0
    assert "gigantic" in html_out and "huge" not in html_out


def test_simplified_view_substitutes_and_counts(engine):
    text = "The gigantic sun."
    tokens = engine.analyse(text, 2, set())          
    html_out, swapped = m.render_simplified(tokens, {"gigantic": "huge"})
    assert swapped == 1
    assert "huge" in html_out
    assert 'title="gigantic"' in html_out or "title='gigantic'" in html_out


def test_simplified_view_preserves_capitalisation(engine):
    tokens = engine.analyse("Gigantic stars burn.", 2, set())
    html_out, swapped = m.render_simplified(tokens, {"gigantic": "huge"})
    assert swapped == 1 and "Huge" in html_out
