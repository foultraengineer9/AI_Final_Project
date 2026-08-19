# Wordify — an AI-powered vocabulary tutor

CS 254 · Introduction to Artificial Intelligence · Final Project · Ashesi University

**Team:** Afia Otieku-Boadu · Precious Nyinerenda · Keli Kemeh · Jadzia Afia Ohenewaa Mantey

---

Wordify finds an English learner's CEFR level (A1–C2), then shows them real
passages with the words *just past* that level highlighted. It does not rewrite
the text: the reader keeps the real passage and gains a map of which words are
worth learning next.

The idea is to hold a reader in the narrow band where text is mostly understood
but still stretching — neither so easy that nothing is learned nor so hard that
the passage collapses. **Learn the word through context.**

---

## Quick start

Requires **Python 3.9 or newer**. Nothing else — the vocabulary, passages and
trained model are all in the repository.

```bash
git clone https://github.com/foultraengineer9/AI_Final_Project.git
cd AI_Final_Project

python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run main_app.py
```

The app opens at `http://localhost:8501`. No configuration, no API keys, no
downloads.

### Verify the install

```bash
pip install pytest
python -m pytest tests/ -q
```

**Expected: `57 passed`.** This runs from a fresh clone with no extra setup.

---

## What you can do in the app

**Read** — pick a passage. Words one CEFR level above you are marked with a yellow
marker stroke; words two or more above get a lilac dotted stroke. Hover any
highlight for its level and a plain-English synonym. Below the passage each
highlighted word appears as a dictionary entry you can mark as learned, after
which it stops being highlighted anywhere. There is also a collapsible
*simplified version* of the passage with graded synonyms substituted in.

*Suggested passage: **Glacier**.*

**Placement test** — tick every word you know. Some are invented (*plaunch*,
*gorbel*): your false-alarm rate on those is subtracted from every CEFR band
before a level is assigned, so the test cannot be inflated by ticking everything.

**Progress** — your level, words banked, and whether you were placed by test or
self-selection.

**Developer mode** — CEFR class distribution, confusion matrix, per-class metrics,
feature importance, and which data source each component is using.

---

## Repository layout

```
main_app.py                  Streamlit application: UI, logic engine, developer mode
train_difficulty_model.py    Trains the deployed six-level classifier
build_stories.py             Generates stories_library.json with CEFR-graded synonyms
afia_ensemble.py             Adapter for the research ensemble
predictor_notf.py            Reduced ensemble variant (no TensorFlow)

data/
  processed_vocab.csv        9,444 rows → 8,679 usable CEFR-labelled words
  stories_library.json       50 passages with 75 graded synonyms
models/
  difficulty_model.pkl       Deployed six-level Random Forest
  model_eval.json            Held-out evaluation metrics
wordify/
  export_bundle/             Serialised research ensemble
  predictor.py               Its inference class
tests/                       57 tests across three files

final_project.ipynb          Classifier development (Afia)
Final_Story_library_(1).ipynb  Passage and synonym pipeline (Precious)
dataset_cleaning.zip         Vocabulary preprocessing (Jadzia)
concreteness.csv             Brysbaert norms, a classifier feature
```

---

## How difficulty is decided

Every word in a passage is resolved in a fixed order:

| Stage | Source | Share of words | Basis |
|---|---|---|---|
| 1 | Vocabulary lookup | **76.9%** | Human CEFR label |
| 2 | Deployed classifier | **23.1%** | length · syllables · Zipf frequency |
| 3 | Length/syllable heuristic | 0% | fallback only if no model loads |

Order matters: a human label beats any prediction, so the model is consulted only
for words the vocabulary does not contain — the technical and rare terms. Every
judgement records which stage produced it, and Developer Mode reports the split,
so an estimate is never silently presented as a labelled fact.

---

## Results

**Deployed classifier** — six-level Random Forest, stratified 80/20 split,
n = 1,736 held out.

| Metric | Value |
|---|---|
| Accuracy (6 classes) | **0.359** |
| Within one CEFR band | **0.764** |
| Macro precision / recall / F1 | 0.361 / 0.401 / 0.366 |
| 5-fold CV accuracy | 0.361 ± 0.006 |

Baselines on the same split: majority class 0.283 exact / MAE 1.21; the
length-and-syllable heuristic 0.255 / MAE 1.35. The model reaches MAE **0.94**.

Feature importance: **frequency 0.765**, length 0.143, syllables 0.092 — how
common a word is predicts its CEFR level far better than how long it is.

**Research ensemble** (`final_project.ipynb`) — XGBoost, Random Forest and a
late-fusion neural network over 768-dimensional sentence embeddings plus
concreteness norms and WordNet features: **≈0.70 accuracy** on three tiers.

The two are not directly comparable — three classes versus six, where chance is
33% and 17%. On the same passage tokens the two models **agree 85%** of the time.

**Application** — 57 automated tests covering tokenisation, highlight
classification, HTML escaping, syllable counting, placement scoring at all six
levels, vocabulary loading, and four input-failure modes.

---

## Optional: running the research ensemble in the app

The base app above needs none of this. This section enables the *Compare with the
research ensemble* panel and **3-tier mode**, in which the ensemble drives the
entire reading view.

**Python 3.12 is required** — TensorFlow has no build for 3.13+.

```bash
python3.12 -m venv venv312
source venv312/bin/activate

pip install -r requirements.txt
pip install pytest sentence-transformers xgboost spacy nltk syllables groq python-dotenv
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

WORDIFY_NO_TF=1 streamlit run main_app.py
```

First run downloads the `all-mpnet-base-v2` model (~420 MB) and takes a minute or
two. Afterwards it is cached.

### Two constraints worth knowing

**Load order is critical.** XGBoost and PyTorch each bundle their own OpenMP
runtime; on Apple Silicon the second to load segfaults the interpreter. The code
imports `xgboost` and loads every `joblib` artifact *before* the
sentence-transformer. Reversing the order crashes the process.

**The neural component is omitted.** Loading TensorFlow alongside PyTorch crashes
in the same way, so `predictor_notf.py` runs a **reduced ensemble** — XGBoost and
Random Forest with weights renormalised to 0.75/0.25, 60% of the original
ensemble's weight. This is a reduced model and is reported as one.

`fileWatcherType = "none"` in `.streamlit/config.toml` is also required: without
it Streamlit's watcher walks every `transformers` submodule and the app hangs.

If the bundle or its dependencies are absent, the panel says so and the toggle is
disabled. The six-level app is unaffected.

---

## Reproducing the artifacts

```bash
python train_difficulty_model.py     # → models/difficulty_model.pkl, model_eval.json
python build_stories.py              # → data/stories_library.json
```

`train_difficulty_model.py` overwrites the committed model; expect accuracy within
±0.02 of the reported figure across scikit-learn versions.

---

## Data

| Source | Contents |
|---|---|
| `processed_vocab.csv` | 9,444 rows from a 10,000-word English CEFR dataset → 8,679 usable words with level, length, syllables, POS and Zipf frequency |
| `stories_library.json` | 50 passages from Simple English Wikipedia (MediaWiki API, 60 topic areas) with 75 CEFR-graded synonyms |
| `concreteness.csv` | Brysbaert concreteness norms, used as a classifier feature |

CEFR distribution: A1 1,061 · A2 1,229 · B1 2,120 · B2 2,454 · C1 925 · C2 890 —
a **2.8:1** imbalance, discussed in the report's ethics section.

---

## Known limitations

- **Synonym coverage is thin**: 75 glosses across 50 passages, 11 with none.
  WordNet frequently has no word that is genuinely simpler than a technical noun
  such as *astronomy*.
- **Two classifiers remain unreconciled.** The ensemble is stronger but predicts
  three tiers and needs embeddings the app cannot recompute per word at read time;
  the deployed model is weaker but integrable.
- **The ensemble runs reduced** — two of three components.
- **Version mismatch**: the ensemble's models were pickled under scikit-learn
  1.6.1 and load under a newer version, which emits warnings.
- **756 words carry conflicting CEFR labels** across senses (*above* is listed A1
  and B1); we resolve to the lowest level. **351 multi-word entries** are dropped,
  since the reading view matches single-word tokens.
- **No user study.** The system is verified to behave correctly, not to teach
  anyone. That study is the obvious next step.

---

## Team contributions

| Member | Contribution |
|---|---|
| **Jadzia** | Vocabulary pipeline: cleaning, CEFR encoding, POS tagging, frequency and syllable features |
| **Precious** | Story library: passage collection from Simple English Wikipedia, synonym mapping |
| **Afia** | Word complexity classifier: feature engineering, triple-stack ensemble, export bundle |
| **Keli** | Application, placement test, highlighting, developer mode, deployed classifier, ensemble adapter, test suite |

AI tool use is declared in `AI_Use_Declaration_Form_Wordify.docx` and the report's
AI Use Declaration appendix.
