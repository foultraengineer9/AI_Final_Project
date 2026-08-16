"""
main_app.py — Lexis: an AI-powered vocabulary tutor
CS 254 Final Project | Owner: Keli (UX & Impact Specialist)

What does this file do?

-Streamlit front-end for the team's vocabulary tutor. It:
  1. Places a new user at a CEFR level (A1-C2) using a yes/No vocabulary test...
  2. Lets the user read real passages with words highlighted by difficulty.
  3. Tracks which words the user has marked as learned.
  4. Exposes a Developer Mode tab with model evaluation charts...

*Teammate artificats this utilises (view TEAM_CONTRACT.md):
  data/processed_vocab.csv   <- Jadzia   (word, cefr level, length, syllables, POS, frequency)...
  data/stories_library.json  <- Precious (title, text, synonyms)...
  models/difficulty_model.pkl<- Afia     (sklearn classifier predicting level 1-6)...
  models/model_eval.json     <- Afia     (optional: held-out metrics + confusion matrix)..


Run with:  streamlit run main_app.py
"""

from __future__ import annotations

import html
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st


# 1. The  configuration...:


APP_ROOT = Path(__file__).parent
DATA_DIR = APP_ROOT / "data"
MODEL_DIR = APP_ROOT / "models"

VOCAB_PATH = DATA_DIR / "processed_vocab.csv"
STORIES_PATH = DATA_DIR / "stories_library.json"
MODEL_PATH = MODEL_DIR / "difficulty_model.pkl"
EVAL_PATH = MODEL_DIR / "model_eval.json"

# CEFR encoding... (agreed with Jadzia: A1=1 ... C2=6)
CEFR_LEVELS = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C1", 6: "C2"}
CEFR_TO_INT = {v: k for k, v in CEFR_LEVELS.items()}
CEFR_BLURBS = {
    "A1": "Beginner",
    "A2": "Elementary",
    "B1": "Intermediate",
    "B2": "Upper-intermediate",
    "C1": "Advanced",
    "C2": "Proficient / near-native",
}

# The placement test tuning...

WORDS_PER_LEVEL = 4          # the real words that are sampled from each CEFR band..
N_PSEUDOWORDS = 6            # catching trials to detect over-claiming..
KNOWN_THRESHOLD = 0.60       # adjusted hit-rate needed to "own" a band
UNRELIABLE_FALSE_ALARM = 0.40  # above this, user is warned...

# Pseudowords: phonotactically legal English non-words used as catch trials.
# Method follows Yes/No vocabulary tests (Meara & Buxton, 1987; Lemhofer &
# Broersma's LexTALE, 2012), where false alarms correct for over-claiming.
PSEUDOWORDS = [
    "plaunch", "gorbel", "trisken", "morfid", "clarnet", "sprodge",
    "flenty", "wugsome", "brindal", "quessel", "tarnick", "shulve",
]


# 2. Design tokens...

# Accessibility note for the ethics section: colour is never the only signal!
# Each highlight class also carries a distinct underline style, so the reading view stays usable for colour-blind users and in greyscale print.

PALETTE = {
    "ink": "#16181D",
    "paper": "#FFFDF7",
    "rule": "#E4DFD2",
    "accent": "#2F5D50",     # library-binding green..
    "muted": "#6B6559",
    "learn_bg": "#FFE9A8",   # highlighter yellow — the next word to learn...
    "learn_line": "#8A6A00",
    "stretch_bg": "#DCD2F5",  # lilac — beyond the user's level..
    "stretch_line": "#4B3391",
}

CSS = f"""
<style>
  .lexis-page {{
      background: {PALETTE['paper']};
      border: 1px solid {PALETTE['rule']};
      border-left: 3px solid {PALETTE['accent']};
      border-radius: 2px;
      padding: 2rem 2.25rem;
      font-family: 'Iowan Old Style', 'Charter', 'Palatino Linotype', Palatino,
                   Georgia, 'Times New Roman', serif;
      font-size: 1.2rem;
      line-height: 2.0;
      color: {PALETTE['ink']};
      max-width: 68ch;
  }}
  .lexis-page p {{ margin: 0 0 1.1em 0; }}

  /* Word at exactly one level above the reader: the learning zone. */
  .w-learn {{
      background: {PALETTE['learn_bg']};
      border-bottom: 2px solid {PALETTE['learn_line']};
      padding: 0 2px;
      border-radius: 2px;
      cursor: help;
  }}
  /* Word two or more levels above: stretch vocabulary. */
  .w-stretch {{
      background: {PALETTE['stretch_bg']};
      border-bottom: 2px dotted {PALETTE['stretch_line']};
      padding: 0 2px;
      border-radius: 2px;
      cursor: help;
  }}
  /* Word already mastered by the reader — left completely plain on purpose. */
  .w-known {{ }}

  .lexis-swatch {{
      display: inline-block; width: 14px; height: 14px;
      border-radius: 2px; margin-right: 6px; vertical-align: -2px;
  }}
  .lexis-legend {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      font-size: 0.86rem; color: {PALETTE['muted']};
  }}
  .lexis-status {{
      font-family: ui-monospace, 'SF Mono', Menlo, monospace;
      font-size: 0.76rem; line-height: 1.7;
  }}
</style>
"""


# 3. fallback demo data...  (keeps the app runnable before my other teammates deliver)


DEMO_VOCAB = [
    ("dog", 1), ("run", 1), ("water", 1), ("happy", 1), ("book", 1), ("small", 1),
    ("kitchen", 2), ("borrow", 2), ("weather", 2), ("quiet", 2), ("journey", 2),
    ("achieve", 3), ("consider", 3), ("distant", 3), ("gigantic", 3), ("consequence", 3),
    ("substantial", 4), ("deteriorate", 4), ("inevitable", 4), ("perceive", 4),
    ("conspicuous", 5), ("ubiquitous", 5), ("meticulous", 5), ("tenacity", 5),
    ("perspicacious", 6), ("obfuscate", 6), ("recalcitrant", 6), ("ineffable", 6),
]

DEMO_STORIES = [
    {
        "title": "The Distant Light",
        "text": (
            "Space is gigantic. When you look up at night, the small points of "
            "light you see are distant suns. Some are so far away that their "
            "light began its journey before there were people to consider it.\n\n"
            "Astronomers perceive a substantial problem in this. The universe is "
            "expanding, and it is inevitable that the most distant galaxies will "
            "one day slip beyond our view. Their light will never reach us again."
        ),
        "synonyms": {"gigantic": "huge", "distant": "far away", "perceive": "notice",
                     "substantial": "large", "inevitable": "certain to happen",
                     "consider": "think about"},
    },
    {
        "title": "The Quiet Kitchen",
        "text": (
            "The kitchen was quiet in the morning. She would borrow the small "
            "radio from the shelf and listen to the weather while the water "
            "boiled.\n\n"
            "It was a meticulous routine, and she was tenacious about keeping it. "
            "Friends found the habit conspicuous, but she said the consequence of "
            "skipping it was a day that never quite started."
        ),
        "synonyms": {"meticulous": "careful", "tenacious": "determined",
                     "conspicuous": "noticeable", "consequence": "result",
                     "borrow": "take for a while"},
    },
]


# 4. Data Loading...



@dataclass
class LoadStatus:
    """Records where each artifact came from, so the sidebar can be honest."""
    vocab: str = "demo"
    stories: str = "demo"
    model: str = "none"
    evaluation: str = "none"
    notes: list = field(default_factory=list)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Locate a column by fuzzy name match.

    Jadzia's exact column names are still in flux (cleaned_file.csv ->
    label_encoded_file.csv -> processed_vocab.csv), so rather than hard-coding
    one spelling we look for any of several likely names, case-insensitively.
    """
    lookup = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    for cand in candidates:                       # substring fallback
        for low, original in lookup.items():
            if cand in low:
                return original
    return None


@st.cache_data(show_spinner=False)
def load_vocab() -> tuple[pd.DataFrame, str, str | None]:
    """Return (frame, source, error). Frame always has ['word', 'level'].

    The error is returned rather than written to session_state because this
    function is cached: it runs once per process, so a side effect written here
    would be invisible to every later session.
    """
    if VOCAB_PATH.exists():
        try:
            df = pd.read_csv(VOCAB_PATH)
            word_col = _find_column(df, ["word", "headword", "term", "lemma"])
            # Prefers an already-numeric level column when both are present...
            level_col = _find_column(
                df, ["cefr_numeric", "level", "cefr_encoded", "label",
                     "cefr_level", "cefr"]
            )
            if word_col is None or level_col is None:
                raise ValueError(
                    f"Could not find word/level columns in {list(df.columns)[:12]}"
                )

            out = df.copy()
            out["word"] = out[word_col].astype(str).str.strip().str.lower()

            # Entries like "adviser/advisor" or "agonize/agonise" hold several
            # spellings in one row. Splitting them so each spelling is matchable.
            out["word"] = out["word"].str.split("/")
            out = out.explode("word")
            out["word"] = out["word"].str.strip()

            # Accept either numeric (1-6) or string ("B1") level encodings.
            # Do not branch on dtype: pandas >=3.0 gives text columns a `str` dtype rather than `object`, so a dtype check silently sends real CEFR strings down the numeric path and NaNs out every row...
            levels = out[level_col]
            if pd.api.types.is_numeric_dtype(levels):
                out["level"] = pd.to_numeric(levels, errors="coerce")
            else:
                mapped = levels.astype(str).str.strip().str.upper().map(CEFR_TO_INT)
                if mapped.isna().all():          
                    mapped = pd.to_numeric(levels, errors="coerce")
                out["level"] = mapped

            out = out.dropna(subset=["word", "level"])
            out["level"] = out["level"].astype(int).clip(1, 6)
            out = out[out["word"].str.match(r"^[a-z][a-z'-]*$", na=False)]

            
            out = out.sort_values("level").drop_duplicates(subset="word", keep="first")
            if out.empty:
                raise ValueError("no usable rows after cleaning")
            return out.reset_index(drop=True), "processed_vocab.csv", None
        except Exception as exc:  
            demo = pd.DataFrame(DEMO_VOCAB, columns=["word", "level"])
            return demo, "demo", (
                f"`processed_vocab.csv` could not be read ({exc}). "
                "Falling back to the built-in demo vocabulary."
            )

    demo = pd.DataFrame(DEMO_VOCAB, columns=["word", "level"])
    return demo, "demo", None


@st.cache_data(show_spinner=False)
def load_stories() -> tuple[list[dict], str, str | None]:
    """Return (stories, source, error). Falls back to the demo passages."""
    if STORIES_PATH.exists():
        try:
            with open(STORIES_PATH, encoding="utf-8") as fh:
                raw = json.load(fh)
            # Accept either a list of stories or a {title: {...}} mapping.
            stories = list(raw.values()) if isinstance(raw, dict) else raw
            cleaned = []
            for s in stories:
                if not isinstance(s, dict) or not s.get("text"):
                    continue
                cleaned.append({
                    "title": s.get("title", "Untitled passage"),
                    "text": str(s["text"]),
                    "synonyms": s.get("synonyms", {}) or {},
                })
            if cleaned:
                return cleaned, "stories_library.json", None
            raise ValueError("file contained no usable passages")
        except Exception as exc:  
            return DEMO_STORIES, "demo", (
                f"`stories_library.json` could not be read ({exc}). "
                "Falling back to the built-in demo passages."
            )
    return DEMO_STORIES, "demo", None


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[object | None, str | None]:
    """Load Afia's classifier. Returns (model, error); model is None if absent."""
    if not MODEL_PATH.exists():
        return None, None
    try:
        import joblib
        return joblib.load(MODEL_PATH), None
    except Exception:  
        try:
            import pickle
            with open(MODEL_PATH, "rb") as fh:
                return pickle.load(fh), None
        except Exception as exc:  
            return None, (
                f"`difficulty_model.pkl` could not be loaded ({exc}). "
                "Word difficulty falls back to the length/syllable heuristic."
            )


@st.cache_data(show_spinner=False)
def load_evaluation() -> dict | None:
    """Load Afia's held-out evaluation report, if she has exported one."""
    if not EVAL_PATH.exists():
        return None
    try:
        with open(EVAL_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  
        return None



# 5. THE Logic Engine  (word -> difficulty -> reading status)

VOWEL_GROUPS = re.compile(r"[aeiouy]+")


def count_syllables(word: str) -> int:
    """Cheap syllable estimate, matching the feature Jadzia engineered.

    Only used for out-of-vocabulary words when the model is unavailable, so a
    rough count is acceptable. Jadzia's pipeline uses the `syllables` library
    for the training data itself.
    """
    w = word.lower().strip("'-")
    if not w:
        return 1
    groups = VOWEL_GROUPS.findall(w)
    count = len(groups)
    if w.endswith("e") and count > 1 and not w.endswith(("le", "ee", "ye")):
        count -= 1
    return max(1, count)


# Cut points calibrated by grid search against the labelled demo vocabulary...

HEURISTIC_CUTS = (4.75, 5.75, 7.0, 8.0, 9.5)


def heuristic_level(word: str) -> int:
    """Fallback difficulty estimate: longer + more syllables => harder.

    This exists so the reading view still works before difficulty_model.pkl
    arrives. It is deliberately crude, and Developer Mode labels any prediction
    made this way so nobody mistakes it for the trained model's output.
    """
    score = 0.42 * len(word) + 1.25 * count_syllables(word)
    for level, cut in enumerate(HEURISTIC_CUTS, start=1):
        if score < cut:
            return level
    return 6


@dataclass
class Token:
    """One piece of a passage: either a word (is_word=True) or the glue between words."""
    text: str
    is_word: bool = False
    level: int | None = None
    status: str = "known"      # known || learn || stretch..
    source: str = "vocab"      # vocab || model || heuristic...


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*|[^A-Za-z]+")


class LogicEngine:
    """Turns a passage into classified tokens for the reading view.

    Lookup order for each word:
      1. Jadzia's labelled vocabulary  (most trustworthy)
      2. Afia's trained classifier     (generalises to unseen words)
      3. Length/syllable heuristic     (last resort, clearly flagged)
    """

    # Maps a feature the model might ask for onto a function that computes it...

    # Keys are matched case-insensitively against model.feature_names_in_, 
    FEATURE_BUILDERS = {
        "length": len,
        "word_length": len,
        "num_letters": len,
        "syllables": count_syllables,
        "syllable_count": count_syllables,
        "n_syllables": count_syllables,
        "num_syllables": count_syllables,
    }

    def __init__(self, vocab: pd.DataFrame, model=None):
        self.lookup: dict[str, int] = dict(zip(vocab["word"], vocab["level"]))
        self.model = model
        self._cache: dict[str, tuple[int, str]] = {}

        
        names = getattr(model, "feature_names_in_", None)
        self.feature_names = [str(n) for n in names] if names is not None else None
        self.unsupported_features: list[str] = []
        if self.feature_names:
            self.unsupported_features = [
                n for n in self.feature_names
                if n.lower().strip() not in self.FEATURE_BUILDERS
            ]

    def _feature_row(self, word: str) -> pd.DataFrame:
        """Build a one-row frame matching the model's expected column schema.

        Features the model was trained on but that we can't recompute at read
        time (frequency, POS one-hots) are filled with 0. Developer Mode reports
        exactly which ones those are so the gap is visible rather than silent.
        """
        if not self.feature_names:
            # if model didn't record names; fall back to the agreed baseline order...
            return pd.DataFrame([[len(word), count_syllables(word)]])

        row = {}
        for name in self.feature_names:
            builder = self.FEATURE_BUILDERS.get(name.lower().strip())
            row[name] = builder(word) if builder else 0
        return pd.DataFrame([row], columns=self.feature_names)

    def level_of(self, word: str) -> tuple[int, str]:
        key = word.lower().strip("'-")
        if key in self._cache:
            return self._cache[key]

        if key in self.lookup:
            result = (int(self.lookup[key]), "vocab")
        elif self.model is not None:
            try:
                pred = int(self.model.predict(self._feature_row(key))[0])
                result = (max(1, min(6, pred)), "model")
            except Exception:  
                result = (heuristic_level(key), "heuristic")
        else:
            result = (heuristic_level(key), "heuristic")

        self._cache[key] = result
        return result

    def analyse(self, text: str, user_level: int, learned: set[str]) -> list[Token]:
        """Split a passage and tag every word relative to the reader's level."""
        tokens: list[Token] = []
        for chunk in TOKEN_RE.findall(text):
            if not chunk or not chunk[0].isalpha():
                tokens.append(Token(text=chunk))
                continue

            key = chunk.lower().strip("'-")
            level, source = self.level_of(chunk)

            if key in learned or level <= user_level:
                status = "known"
            elif level == user_level + 1:
                status = "learn"
            else:
                status = "stretch"

            tokens.append(Token(chunk, True, level, status, source))
        return tokens


def render_passage(tokens: list[Token], synonyms: dict[str, str]) -> str:
    """Build the highlighted HTML for a passage.

    Uses CSS classes rather than inline styles, and puts the plain-English
    synonym in a title attribute so hovering a highlighted word explains it.
    """
    parts: list[str] = ["<div class='lexis-page'><p>"]
    for tok in tokens:
        if not tok.is_word:
            # Blank lines become paragraph breaks; everything else passes through...
            if "\n\n" in tok.text:
                parts.append("</p><p>")
                parts.append(html.escape(tok.text.replace("\n\n", "")))
            else:
                parts.append(html.escape(tok.text))
            continue

        safe = html.escape(tok.text)
        if tok.status == "known":
            parts.append(safe)
            continue

        key = tok.text.lower().strip("'-")
        gloss = synonyms.get(key) or synonyms.get(tok.text)
        cefr = CEFR_LEVELS[tok.level]
        tip = f"{cefr}" + (f" \u2014 means \u201c{gloss}\u201d" if gloss else "")
        cls = "w-learn" if tok.status == "learn" else "w-stretch"
        parts.append(
            f"<span class='{cls}' title='{html.escape(tip)}'>{safe}</span>"
        )

    parts.append("</p></div>")
    return "".join(parts)



# 6. Placement test!



def build_placement_items(vocab: pd.DataFrame, seed: int = 42) -> list[dict]:
    """Sample real words across CEFR bands plus pseudoword catch trials."""
    rng = random.Random(seed)
    items: list[dict] = []

    for level in sorted(CEFR_LEVELS):
        pool = vocab.loc[vocab["level"] == level, "word"].tolist()
        if not pool:
            continue
        picked = rng.sample(pool, min(WORDS_PER_LEVEL, len(pool)))
        items.extend({"word": w, "level": level, "real": True} for w in picked)

    fakes = rng.sample(PSEUDOWORDS, min(N_PSEUDOWORDS, len(PSEUDOWORDS)))
    items.extend({"word": w, "level": None, "real": False} for w in fakes)

    rng.shuffle(items)
    return items


def score_placement(items: list[dict], answers: dict[str, bool]) -> dict:
    """Score a Yes/No vocabulary test with a false-alarm correction.

    Raw hit rates are inflated because people over-claim knowledge. We measure
    how often the reader claims a pseudoword and subtract that rate from every
    band, then place them at the highest band they still own.
    """
    fakes = [i for i in items if not i["real"]]
    false_alarms = sum(bool(answers.get(i["word"])) for i in fakes)
    fa_rate = false_alarms / len(fakes) if fakes else 0.0

    per_level: dict[int, dict] = {}
    for level in sorted(CEFR_LEVELS):
        band = [i for i in items if i["real"] and i["level"] == level]
        if not band:
            continue
        hits = sum(bool(answers.get(i["word"])) for i in band)
        hit_rate = hits / len(band)
        per_level[level] = {
            "hits": hits,
            "total": len(band),
            "hit_rate": hit_rate,
            "adjusted": max(0.0, hit_rate - fa_rate),
        }

    suggested = 1
    for level in sorted(per_level):
        if per_level[level]["adjusted"] >= KNOWN_THRESHOLD:
            suggested = level
        else:
            break  # stopping at the first band they don't own; levels are cumulative...

    return {
        "suggested": suggested,
        "per_level": per_level,
        "false_alarm_rate": fa_rate,
        "reliable": fa_rate < UNRELIABLE_FALSE_ALARM,
        "n_items": len(items),
    }



# 7. Pages...



def page_placement(vocab: pd.DataFrame) -> None:
    st.subheader("Find your level")
    st.write(
        "Tick every word you're confident you know the meaning of. Some of the "
        "words below aren't real English words \u2014 that's deliberate, and it's how "
        "we check the result is honest rather than optimistic."
    )

    items = st.session_state.setdefault("placement_items", build_placement_items(vocab))
    answers = st.session_state.setdefault("placement_answers", {})

    cols = st.columns(3)
    for idx, item in enumerate(items):
        with cols[idx % 3]:
            answers[item["word"]] = st.checkbox(
                item["word"], value=answers.get(item["word"], False), key=f"pt_{idx}"
            )

    st.write("")
    left, right = st.columns([1, 3])
    with left:
        submitted = st.button("Score my test", type="primary", width="stretch")
    with right:
        if st.button("Start over", width="content"):
            for key in ("placement_items", "placement_answers", "placement_result"):
                st.session_state.pop(key, None)
            for idx in range(len(items)):
                st.session_state.pop(f"pt_{idx}", None)
            st.rerun()

    if submitted:
        st.session_state["placement_result"] = score_placement(items, answers)

    result = st.session_state.get("placement_result")
    if not result:
        return

    st.divider()
    level = result["suggested"]
    cefr = CEFR_LEVELS[level]

    if not result["reliable"]:
        st.warning(
            f"You ticked {result['false_alarm_rate']:.0%} of the invented words, so "
            "this result probably overstates your vocabulary. Retake it and only "
            "tick words you could actually define."
        )

    st.success(f"Suggested starting level: **{cefr} \u2014 {CEFR_BLURBS[cefr]}**")

    breakdown = pd.DataFrame([
        {
            "Level": CEFR_LEVELS[lvl],
            "Known": f"{stats['hits']}/{stats['total']}",
            "Raw score": round(stats["hit_rate"], 2),
            "Adjusted": round(stats["adjusted"], 2),
            "Owned": "yes" if stats["adjusted"] >= KNOWN_THRESHOLD else "no",
        }
        for lvl, stats in sorted(result["per_level"].items())
    ])
    st.dataframe(breakdown, hide_index=True, width="stretch")
    st.caption(
        f"Adjusted score = raw score minus your false-alarm rate "
        f"({result['false_alarm_rate']:.0%}). A band counts as owned at "
        f"{KNOWN_THRESHOLD:.0%} or above."
    )

    if st.button(f"Use {cefr} and start reading", type="primary"):
        st.session_state["user_level"] = level
        st.session_state["placed_by"] = "test"
        st.rerun()


def page_read(engine: LogicEngine, stories: list[dict]) -> None:
    user_level = st.session_state["user_level"]
    learned: set[str] = st.session_state.setdefault("learned", set())

    titles = [s["title"] for s in stories]
    choice = st.selectbox("Passage", titles, key="story_choice")
    story = stories[titles.index(choice)]

    tokens = engine.analyse(story["text"], user_level, learned)

    st.markdown(
        f"<div class='lexis-legend'>"
        f"<span class='lexis-swatch' style='background:{PALETTE['learn_bg']};"
        f"border-bottom:2px solid {PALETTE['learn_line']}'></span>"
        f"Next for you ({CEFR_LEVELS[min(6, user_level + 1)]}) &nbsp;&nbsp;"
        f"<span class='lexis-swatch' style='background:{PALETTE['stretch_bg']};"
        f"border-bottom:2px dotted {PALETTE['stretch_line']}'></span>"
        f"Stretch &nbsp;&nbsp; hover any highlight for its meaning"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(render_passage(tokens, story.get("synonyms", {})), unsafe_allow_html=True)

    
    flagged = {}
    for tok in tokens:
        if tok.is_word and tok.status != "known":
            flagged.setdefault(tok.text.lower().strip("'-"), tok)

    st.write("")
    st.subheader(f"Words to learn in this passage ({len(flagged)})")

    if not flagged:
        st.info(
            "Nothing here is above your level. Try a harder passage, or raise your "
            "level in the sidebar."
        )
        return

    for key, tok in sorted(flagged.items(), key=lambda kv: (kv[1].level or 0, kv[0])):
        gloss = story.get("synonyms", {}).get(key, "")
        c1, c2, c3 = st.columns([2, 4, 1])
        c1.markdown(f"**{key}** &nbsp; `{CEFR_LEVELS[tok.level]}`")
        c2.write(gloss if gloss else "_no synonym in the library yet_")
        if c3.button("Got it", key=f"learn_{key}"):
            learned.add(key)
            st.rerun()


def page_progress(engine: LogicEngine) -> None:
    learned: set[str] = st.session_state.get("learned", set())
    user_level = st.session_state["user_level"]

    st.subheader("Your progress")

    c1, c2, c3 = st.columns(3)
    c1.metric("Current level", CEFR_LEVELS[user_level])
    c2.metric("Words marked learned", len(learned))
    c3.metric(
        "Placed by",
        "Placement test" if st.session_state.get("placed_by") == "test" else "Self-selected",
    )

    if not learned:
        st.info("Mark words as learned while reading and they'll collect here.")
        return

    rows = []
    for word in sorted(learned):
        level, source = engine.level_of(word)
        rows.append({"Word": word, "Level": CEFR_LEVELS[level], "Difficulty from": source})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if st.button("Clear learned words"):
        st.session_state["learned"] = set()
        st.rerun()


def page_developer(vocab: pd.DataFrame, model, evaluation: dict | None,
                   status: LoadStatus, engine: LogicEngine) -> None:
    st.subheader("Developer mode")
    st.caption(
        "Model diagnostics for the team and for the report. Not part of the "
        "learner-facing experience."
    )

    import matplotlib.pyplot as plt

    # artifact status...
    st.markdown("**Artifacts**")
    st.dataframe(
        pd.DataFrame([
            {"Artifact": "processed_vocab.csv", "Owner": "Jadzia", "Source": status.vocab},
            {"Artifact": "stories_library.json", "Owner": "Precious", "Source": status.stories},
            {"Artifact": "difficulty_model.pkl", "Owner": "Afia", "Source": status.model},
            {"Artifact": "model_eval.json", "Owner": "Afia", "Source": status.evaluation},
        ]),
        hide_index=True,
        width="stretch",
    )

    # feature schema agreement.
    if model is not None:
        if engine.feature_names is None:
            st.warning(
                "The model didn't record `feature_names_in_`, so predictions fall "
                "back to positional order `[length, syllables]`. If the model was "
                "trained on more or differently ordered features, every "
                "out-of-vocabulary prediction is wrong. Ask Afia to fit on a "
                "named DataFrame."
            )
        elif engine.unsupported_features:
            st.warning(
                "The model expects features this app can't recompute for a word "
                "that isn't in the vocabulary, so they are sent as 0: "
                + ", ".join(f"`{f}`" for f in engine.unsupported_features[:12])
                + (" ..." if len(engine.unsupported_features) > 12 else "")
                + ". In-vocabulary words are unaffected (their level is looked up "
                "directly), but out-of-vocabulary predictions will be degraded."
            )
        else:
            st.success(
                "Feature schema agreed: "
                + ", ".join(f"`{f}`" for f in engine.feature_names)
            )

    
    st.markdown("**Vocabulary distribution across CEFR levels**")
    counts = vocab["level"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar([CEFR_LEVELS[i] for i in counts.index], counts.values,
           color=PALETTE["accent"])
    ax.set_ylabel("Words")
    ax.set_xlabel("CEFR level")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

    imbalance = counts.max() / max(1, counts.min())
    st.caption(
        f"Largest band is {imbalance:.1f}x the smallest. Imbalance here biases the "
        "classifier toward over-represented levels \u2014 worth reporting in the "
        "ethics section, and worth checking per-class recall below rather than "
        "relying on overall accuracy."
    )

    st.divider()

    #confusion matrix.
    st.markdown("**Confusion matrix**")

    if evaluation and "confusion_matrix" in evaluation:
        cm = pd.DataFrame(evaluation["confusion_matrix"])
        labels = evaluation.get("labels", list(range(1, len(cm) + 1)))
        _plot_confusion(cm.values, labels, plt)
        if "accuracy" in evaluation:
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{evaluation['accuracy']:.3f}")
            if "macro_precision" in evaluation:
                m2.metric("Macro precision", f"{evaluation['macro_precision']:.3f}")
            if "macro_recall" in evaluation:
                m3.metric("Macro recall", f"{evaluation['macro_recall']:.3f}")
        st.caption("Computed by Afia on a held-out test set and exported to model_eval.json.")

    elif model is not None:
        st.warning(
            "No model_eval.json found, so this matrix is computed live on a random "
            "20% split of processed_vocab.csv. Those rows were almost certainly in "
            "the model's training data, so treat these numbers as a smoke test, not "
            "as the evaluation for the report."
        )
        try:
            from sklearn.metrics import confusion_matrix
            from sklearn.model_selection import train_test_split

            
            X = pd.concat(
                [engine._feature_row(w) for w in vocab["word"]], ignore_index=True
            )
            y = vocab["level"].reset_index(drop=True)
            _, X_te, _, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            y_pred = model.predict(X_te)
            labels = sorted(y.unique())
            cm = confusion_matrix(y_te, y_pred, labels=labels)
            _plot_confusion(cm, labels, plt)
            st.metric("Accuracy on this split", f"{(y_pred == y_te).mean():.3f}")
        except Exception as exc:  
            st.error(
                f"Could not run a live evaluation: {exc}\n\n"
                "This usually means the model expects different features than "
                "[length, syllables]. Ask Afia to export model_eval.json instead."
            )
    else:
        st.info(
            "difficulty_model.pkl hasn't been added yet. Drop it in `models/` "
            "(and ideally `models/model_eval.json` too) and this tab fills in."
        )

    #...feature importance...
    if model is not None and hasattr(model, "feature_importances_"):
        st.divider()
        st.markdown("**Feature importance**")
        names = getattr(model, "feature_names_in_", None)
        names = list(names) if names is not None else [
            f"feature {i}" for i in range(len(model.feature_importances_))
        ]
        fig, ax = plt.subplots(figsize=(7, 0.4 * len(names) + 1.2))
        ax.barh(names, model.feature_importances_, color=PALETTE["accent"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)


def _plot_confusion(cm, labels, plt) -> None:
    """Render a confusion matrix heatmap with counts written into each cell."""
    display = [CEFR_LEVELS.get(int(l), str(l)) for l in labels]
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(cm, cmap="YlGnBu")
    ax.set_xticks(range(len(display)), display)
    ax.set_yticks(range(len(display)), display)
    ax.set_xlabel("Predicted level")
    ax.set_ylabel("True level")
    threshold = cm.max() / 2 if hasattr(cm, "max") else 0
    for i in range(len(display)):
        for j in range(len(display)):
            value = cm[i][j]
            ax.text(j, i, f"{value}", ha="center", va="center",
                    color="white" if value > threshold else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)



# 8. The App shell...
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Lexis \u2014 vocabulary tutor",
                       page_icon="\u25e7", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    vocab, vocab_src, vocab_err = load_vocab()
    stories, stories_src, stories_err = load_stories()
    model, model_err = load_model()
    evaluation = load_evaluation()
    load_errors = [e for e in (vocab_err, stories_err, model_err) if e]

    status = LoadStatus(
        vocab=vocab_src,
        stories=stories_src,
        model="difficulty_model.pkl" if model is not None else "none",
        evaluation="model_eval.json" if evaluation else "none",
    )
    engine = LogicEngine(vocab, model)

    # sidebar...
    with st.sidebar:
        st.markdown("### Lexis")
        st.caption("Read real passages at the edge of your vocabulary.")
        st.divider()

        current = st.session_state.get("user_level")
        picked = st.selectbox(
            "Your level",
            options=list(CEFR_LEVELS),
            index=(current - 1) if current else 2,
            format_func=lambda i: f"{CEFR_LEVELS[i]} \u2014 {CEFR_BLURBS[CEFR_LEVELS[i]]}",
        )
        if picked != current:
            # manual change overrides a placement-test result, so record.
            st.session_state["user_level"] = picked
            st.session_state["placed_by"] = "self"

        st.divider()
        st.markdown("<div class='lexis-status'>", unsafe_allow_html=True)
        st.caption("Data sources")
        for label, src in (("vocab", status.vocab), ("stories", status.stories),
                           ("model", status.model)):
            mark = "\u25cf" if src not in ("demo", "none") else "\u25cb"
            st.markdown(
                f"<div class='lexis-status'>{mark} {label}: {src}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        missing = [
            name for name, src_ in (("vocabulary", status.vocab),
                                    ("passages", status.stories))
            if src_ == "demo"
        ]
        if missing:
            st.caption(
                f"Using demo {' and '.join(missing)}. Drop the real file"
                f"{'s' if len(missing) > 1 else ''} into `data/` to switch over."
            )

    for message in load_errors:
        st.warning(message, icon="\u26a0\ufe0f")

    st.session_state.setdefault("user_level", picked)

    #Main tabs.
    st.title("Lexis")
    st.caption(
        "A vocabulary tutor that finds your level, then highlights the words "
        "just past it."
    )

    tab_read, tab_place, tab_progress, tab_dev = st.tabs(
        ["Read", "Placement test", "Progress", "Developer mode"]
    )

    with tab_read:
        page_read(engine, stories)
    with tab_place:
        page_placement(vocab)
    with tab_progress:
        page_progress(engine)
    with tab_dev:
        page_developer(vocab, model, evaluation, status, engine)


if __name__ == "__main__":
    main()
