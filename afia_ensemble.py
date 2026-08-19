"""Adapter for Afia's research ensemble (wordify/export_bundle).

WHY THIS FILE EXISTS
--------------------
Afia's ensemble is the project's strongest classifier: XGBoost + Random Forest +
a Keras late-fusion network over 768-dimensional all-mpnet-base-v2 embeddings,
with Brysbaert concreteness norms and WordNet taxonomic features. It reaches 0.70
accuracy on three tiers.

It could not originally drive the reading view, for three reasons: it predicted
three tiers rather than six CEFR levels, it was not serialised, and its features
could not be recomputed per word at read time. Her `export_bundle` solves the
second problem, and this adapter solves the other two:

  * tier 0/1/2 is mapped onto CEFR band pairs, so her output can be rendered in
    the same interface as the six-level model;
  * the heavy dependencies are imported lazily and the whole module degrades to a
    clear message when they are absent, so the application still starts in
    seconds on a machine that has none of them installed.

The ensemble is offered as a SECOND OPINION rather than as a replacement. The
six-level model still drives the reading view; this shows what the stronger but
coarser classifier makes of the same passage. Keeping both visible is the honest
presentation of a genuine trade-off between accuracy and granularity.

Requires, on the machine running the app:
    pip install sentence-transformers spacy tensorflow xgboost syllables
    python -m spacy download en_core_web_sm
The first call also downloads all-mpnet-base-v2 (~420 MB) from Hugging Face.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Where Afia's bundle lives. Overridable so the bundle can sit outside the repo.
BUNDLE_DIR = Path(os.getenv("WORDIFY_BUNDLE_DIR", "wordify/export_bundle"))
PREDICTOR_DIR = BUNDLE_DIR.parent            # predictor.py sits beside the bundle

# Her three tiers, expressed in the six-level scheme the interface uses.
TIER_TO_CEFR = {0: "A1-A2", 1: "B1-B2", 2: "C1-C2"}
TIER_LABEL = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}

# A reader's CEFR level (1-6) maps onto the comfort tier her model expects.
LEVEL_TO_TIER = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}


@dataclass
class EnsembleWord:
    """One token as the ensemble sees it."""
    word: str
    tier: int
    challenging: bool
    protected: bool
    suggestion: str | None = None

    @property
    def band(self) -> str:
        return TIER_TO_CEFR.get(self.tier, "?")


def bundle_status() -> tuple[bool, str]:
    """Whether the bundle is present, and a human-readable explanation."""
    if not BUNDLE_DIR.exists():
        return False, (
            f"Afia's ensemble bundle was not found at `{BUNDLE_DIR}`. Unzip "
            "`wordify.zip` into the project root, or set WORDIFY_BUNDLE_DIR."
        )
    required = ["config.json", "xgb_clf.joblib", "rf_clf.joblib",
                "late_fusion_model.keras", "struct_scaler.joblib",
                "pipeline_scaler.joblib", "pipeline_kmeans.joblib",
                "lookups.joblib"]
    missing = [f for f in required if not (BUNDLE_DIR / f).exists()]
    if missing:
        return False, f"Bundle at `{BUNDLE_DIR}` is incomplete: {', '.join(missing)}"
    if not (PREDICTOR_DIR / "predictor.py").exists():
        return False, f"`predictor.py` was not found in `{PREDICTOR_DIR}`."
    return True, f"Bundle found at `{BUNDLE_DIR}`."


def missing_dependencies() -> list[str]:
    """Heavy imports the ensemble needs that aren't installed. Cheap to call."""
    import importlib.util
    needed = {
        "sentence_transformers": "sentence-transformers",
        "spacy": "spacy",
        "tensorflow": "tensorflow",
        "xgboost": "xgboost",
        "syllables": "syllables",
    }
    return [pip_name for module, pip_name in needed.items()
            if importlib.util.find_spec(module) is None]


def _preload_native_libs() -> None:
    """Import xgboost before anything can pull in torch.

    XGBoost and PyTorch bundle separate OpenMP runtimes; on Apple Silicon the
    second to load segfaults the process. Import order is the fix.
    """
    try:
        import xgboost  # noqa: F401
    except ImportError:
        pass


def load_predictor(groq_api_key: str | None = None):
    """Import and instantiate Afia's WordComplexityPredictor.

    Raises RuntimeError with an actionable message rather than propagating an
    ImportError, so the caller can show the reason in the interface. The Groq key
    is optional: without it her class still classifies every word and only the
    LLM phrase-rewriting is disabled.
    """
    _preload_native_libs()

    ok, reason = bundle_status()
    if not ok:
        raise RuntimeError(reason)

    missing = missing_dependencies()
    if missing:
        raise RuntimeError(
            "Missing packages for the ensemble: " + ", ".join(missing)
            + f". Install with `pip install {' '.join(missing)}`"
            + (" and `python -m spacy download en_core_web_sm`"
               if "spacy" in missing else "")
        )

    if str(PREDICTOR_DIR.resolve()) not in sys.path:
        sys.path.insert(0, str(PREDICTOR_DIR.resolve()))

    # Prefer the full three-model ensemble. Loading TensorFlow after PyTorch is
    # already resident segfaults the interpreter on Apple Silicon, so if the
    # environment sets WORDIFY_NO_TF, or TensorFlow is absent, fall back to the
    # reduced XGBoost + Random Forest ensemble instead of crashing.
    use_reduced = os.getenv("WORDIFY_NO_TF", "").strip().lower() in ("1", "true", "yes")
    if not use_reduced:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            use_reduced = True

    if use_reduced:
        try:
            from predictor_notf import ReducedEnsemblePredictor
        except ImportError as exc:
            raise RuntimeError(f"Could not import predictor_notf.py: {exc}") from exc
        return ReducedEnsemblePredictor(
            bundle_dir=str(BUNDLE_DIR.resolve()),
            module_dir=str(PREDICTOR_DIR.resolve()),
        )

    try:
        from predictor import WordComplexityPredictor
    except ImportError as exc:
        raise RuntimeError(f"Could not import predictor.py: {exc}") from exc

    return WordComplexityPredictor(
        bundle_dir=str(BUNDLE_DIR.resolve()),
        groq_api_key=groq_api_key or os.getenv("GROQ_API_KEY"),
    )


def analyse(predictor, passage: str, user_level: int
            ) -> tuple[list[EnsembleWord], str | None]:
    """Run the ensemble over a passage and normalise the audit log.

    Returns the per-word results plus her reconstructed passage, which is None
    when no Groq key is configured (classification still works in that case).
    """
    tier = LEVEL_TO_TIER.get(user_level, 1)
    reconstructed, audit = predictor.analyze_passage(
        passage=passage, target_comfort_level=tier
    )

    words: list[EnsembleWord] = []
    for row in audit:
        text = row.get("word", "")
        if not text.strip() or not any(c.isalpha() for c in text):
            continue                      # punctuation and whitespace
        words.append(EnsembleWord(
            word=text,
            tier=int(row.get("predicted_tier", 0)),
            challenging=row.get("status") == "CHALLENGING",
            protected=bool(row.get("is_protected")),
            suggestion=row.get("suggestion"),
        ))

    # Her reconstruction equals the input when nothing was rewritten.
    if reconstructed and reconstructed.strip() == passage.strip():
        reconstructed = None
    return words, reconstructed


def agreement(ensemble_words: list[EnsembleWord], app_tokens) -> dict:
    """Compare the ensemble's verdict with the six-level pipeline's, per word.

    Both answer the same question — is this word beyond the reader? — so the
    rate at which they agree is a meaningful cross-check that needs no labels.
    """
    app_flags = {}
    for tok in app_tokens:
        if getattr(tok, "is_word", False):
            app_flags[tok.text.lower().strip("'-")] = tok.status != "known"

    both = agree = 0
    only_ensemble = only_app = 0
    for w in ensemble_words:
        key = w.word.lower().strip("'-")
        if key not in app_flags:
            continue
        both += 1
        if w.challenging == app_flags[key]:
            agree += 1
        elif w.challenging:
            only_ensemble += 1
        else:
            only_app += 1

    return {
        "compared": both,
        "agree": agree,
        "rate": (agree / both) if both else 0.0,
        "only_ensemble": only_ensemble,
        "only_app": only_app,
    }

@dataclass
class TierToken:
    """One token as tier mode sees it.

    Defined at module level, not inside tier_tokens: Streamlit pickles cached
    return values, and a class defined in a function body cannot be pickled.
    """
    text: str
    is_word: bool = False
    tier: int = 0
    status: str = "known"          # known | above
    protected: bool = False


def tier_tokens(predictor, passage: str, target_tier: int):
    """Tokenise a passage the way the reading view needs, using her tiers only.

    Returns (tokens, n_flagged) where each token mirrors main_app.Token closely
    enough for the renderer: .text, .is_word, .tier, .status, .protected.
    Non-word characters are preserved so the passage reads normally.
    """
    import re

    _, audit = predictor.analyze_passage(
        passage=passage, target_comfort_level=target_tier
    )

    # Her audit is per-token in document order; index it by word occurrence so
    # repeated words keep their own verdicts.
    verdicts, seen = {}, {}
    for row in audit:
        text = str(row.get("word", ""))
        if not text.strip() or not any(c.isalpha() for c in text):
            continue
        n = seen.get(text.lower(), 0)
        verdicts[(text.lower(), n)] = row
        seen[text.lower()] = n + 1

    tokens, counter, flagged = [], {}, 0
    for chunk in re.findall(r"[A-Za-z][A-Za-z'-]*|[^A-Za-z]+", passage):
        if not chunk or not chunk[0].isalpha():
            tokens.append(TierToken(text=chunk))
            continue
        key = chunk.lower()
        n = counter.get(key, 0)
        counter[key] = n + 1
        row = verdicts.get((key, n))
        if row is None:
            tokens.append(TierToken(text=chunk, is_word=True))
            continue
        tier = int(row.get("predicted_tier", 0))
        protected = bool(row.get("is_protected"))
        above = row.get("status") == "CHALLENGING"
        if above:
            flagged += 1
        tokens.append(TierToken(chunk, True, tier,
                                "above" if above else "known", protected))
    return tokens, flagged
