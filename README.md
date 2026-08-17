# Wordify — an AI-powered vocabulary tutor

CS 254 · Introduction to Artificial Intelligence · Final Project · Ashesi University

Wordify finds an English learner's CEFR level (A1–C2), then shows them real
passages with the words *just past* that level highlighted. The idea is to keep a
reader in the narrow band where text is comprehensible but still stretching —
neither so easy that nothing is learned nor so hard that the passage collapses.

---

## What it does

**Places the reader.** A Yes/No vocabulary test samples real words across all six
CEFR bands alongside invented pseudowords. Ticking pseudowords reveals
over-claiming, and that false-alarm rate is subtracted from every band before a
level is assigned, so the placement cannot be gamed by ticking everything.

**Highlights the learning zone.** In any passage, words one level above the
reader are marked in highlighter yellow; words two or more levels above are
marked in lilac. Hovering a highlight shows its CEFR level and a plain-English
synonym. Words at or below the reader's level are left completely plain.

**Tracks progress.** Words can be marked as learned, after which they stop being
highlighted anywhere in the app.

**Exposes the model.** A Developer Mode tab shows the vocabulary's CEFR class
balance, the classifier's confusion matrix and per-class metrics, and flags any
mismatch between the features the model expects and the features the app can
supply at read time.

---

## Running it

Requires Python 3.9 or later.

```bash
git clone https://github.com/foultraengineer9/AI_Final_Project.git
cd AI_Final_Project

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run main_app.py
```

The app opens at `http://localhost:8501`.

No configuration is needed — `data/processed_vocab.csv` and
`data/stories_library.json` are in the repository, and the sidebar shows which
data source each component is using.

### Tests

```bash
python -m pytest tests/ -v
```

28 tests covering tokenisation, highlight classification, HTML escaping,
syllable counting, placement-test scoring, vocabulary loading, and four
failure modes (missing, malformed, or corrupt input files).

---

## Repository layout

```
main_app.py               Streamlit application — UI, logic engine, developer mode
build_stories.py          Generates stories_library.json with CEFR-graded synonyms
final_project.ipynb       Word complexity classifier (ensemble)
Final_Story_library.ipynb Passage sourcing and synonym pipeline
data/
  processed_vocab.csv     8,679 CEFR-labelled words with engineered features
  stories_library.json    50 passages with synonym maps
tests/                    pytest suite
requirements.txt          Pinned dependencies
TEAM_CONTRACT.md          Interface contract between team components
.streamlit/config.toml    Theme, so the app renders identically on any machine
```

---

## How it works

### Difficulty resolution

Every word in a passage is resolved through three stages, in order:

1. **Vocabulary lookup** — if the word is in `processed_vocab.csv`, its CEFR
   level is used directly. This covers **76.9%** of words across the 50 passages.
2. **Classifier** — otherwise the trained model predicts the level. This covers
   the remaining **23.1%**: every technical or rare term a reader meets that the
   labelled vocabulary does not contain.
3. **Heuristic** — if no model is loaded, a calibrated length-and-syllable
   estimate is used. It is clearly flagged as a fallback in Developer Mode and is not used
   while a model is present.

This ordering matters: the model is consulted only for words *outside* the
labelled vocabulary, so its influence on what the reader actually sees is
narrower than its headline accuracy suggests.

### Placement scoring

Adapted from Yes/No vocabulary tests (Meara & Buxton, 1987; Lemhöfer &
Broersma's LexTALE, 2012). For each CEFR band:

```
adjusted score = hit rate − false-alarm rate on pseudowords
```

A band is "owned" at an adjusted score of 0.60 or above, and the reader is
placed at the highest consecutive band they own. Above a 40% false-alarm rate
the result is flagged as unreliable rather than reported as fact.

### Synonym grading

`build_stories.py` searches WordNet for synonyms of every B1-or-above word, then
keeps a candidate only if it is **strictly easier** than the word it explains,
graded against the project's own CEFR vocabulary. Candidates are POS-aligned
using the vocabulary's tags, restricted to the most common senses, and required
to clear a word-frequency floor.

---

## Data

| Source | Contents |
|---|---|
| `processed_vocab.csv` | 8,679 unique words, CEFR-labelled, with length, syllable count, POS and Zipf frequency |
| `stories_library.json` | 50 passages from Simple English Wikipedia and curated backups, with 75 CEFR-graded synonyms |
| `concreteness.csv` | Brysbaert concreteness norms, used as a classifier feature |

CEFR distribution: A1 1,061 · A2 1,229 · B1 2,120 · B2 2,454 · C1 925 · C2 890 —
a 2.8× imbalance between the largest and smallest bands, discussed in the report's
ethics section.

---

## Results

**Deployed difficulty classifier** — six-level Random Forest on length, syllable
count and Zipf frequency. Stratified 80/20 split, n = 1,736 held out.

| Metric | Value |
|---|---|
| Accuracy (6 classes) | 0.359 |
| Within one CEFR band | 0.764 |
| Macro precision / recall / F1 | 0.361 / 0.401 / 0.366 |
| 5-fold CV accuracy | 0.361 ± 0.006 |

Baselines on the same split: majority class 0.283 exact, MAE 1.21; the
length/syllable heuristic 0.255 exact, MAE 1.35. The model reaches MAE 0.94.

Feature importance: frequency 0.765, length 0.143, syllable count 0.092 — how
common a word is predicts its CEFR level considerably better than how long it is.

**Research ensemble** (`final_project.ipynb`) — XGBoost, Random Forest and a
late-fusion neural network over 768-dimensional sentence embeddings plus
concreteness norms, soft-stacked: 0.70 accuracy, 0.70 macro F1 on three tiers,
n = 1,889.

The two are not directly comparable: three classes versus six, where chance is
33% and 17% respectively.

**Application** — 28 automated tests covering tokenisation, highlight
classification, HTML escaping, syllable counting, placement scoring at all six
levels, vocabulary loading, and four input-failure modes.

## Known limitations

- Synonym coverage is thin: 75 synonyms across 50 passages, with 11 passages
  having none. WordNet frequently offers no synonym that is genuinely simpler
  than a technical noun such as *astronomy* or *spacecraft*.
- Two classifiers exist. The research ensemble (`final_project.ipynb`, 0.70
  accuracy) is stronger but predicts three tiers, is not serialised, and needs
  768-dimensional sentence embeddings the app cannot recompute per word at read
  time. The deployed six-level model (`train_difficulty_model.py`, 0.359 exact /
  0.764 within one band) is weaker but integrable. The app ships the latter.
- 351 multi-word entries (*air conditioning*, *according to*) are dropped, since
  the reading view matches single-word tokens.
- No user study was conducted, so the system's pedagogical effect is untested.

---

## Team

| Member | Component |
|---|---|
| Jadzia | Data pipeline — cleaning, CEFR encoding, feature engineering (`processed_vocab.csv`) |
| Precious | Passage sourcing and synonym mapping (`Final_Story_library.ipynb`) |
| Afia | Word complexity classifier (`final_project.ipynb`) |
| Keli | Application, placement test, highlighting, developer mode, test suite (`main_app.py`) |

AI tool use is declared in the final report's AI Use Declaration appendix.
