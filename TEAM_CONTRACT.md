# Interface contract — what `main_app.py` needs from each of you

Keli owns `main_app.py`. It loads three artifacts. **The app runs today without any
of them** (it falls back to built-in demo data), so nobody is blocked — but the
closer your output matches the shapes below, the less rework we all do in Week 13.

Drop files here:

```
main_app.py
data/
  processed_vocab.csv      <- Jadzia
  stories_library.json     <- Precious
models/
  difficulty_model.pkl     <- Afia
  model_eval.json          <- Afia
```

---

## Jadzia — `data/processed_vocab.csv`

**Required columns.** The app searches case-insensitively, so any of these names work:

| What it needs | Accepted column names |
|---|---|
| the word | `word`, `headword`, `term`, `lemma` |
| the CEFR level | `level`, `cefr`, `cefr_level`, `cefr_encoded`, `label` |

The level column may be **either** the integers `1–6` **or** the strings
`A1 A2 B1 B2 C1 C2`. Both are handled. Keep the agreed encoding: A1=1 … C2=6.

Everything else you produce (`length`, `syllable_count`, `frequency`, POS one-hots)
is passed through untouched and used by Afia — the app doesn't need it directly.

**Please also:** lowercase the words and drop duplicates before export. The app does
this defensively, but doing it upstream means the row counts in our report match.

---

## Precious — `data/stories_library.json`

Either a list of objects, or an object keyed by title. Both parse.

```json
[
  {
    "title": "The Distant Light",
    "text": "Space is gigantic. Some suns are distant...\n\nSecond paragraph here.",
    "synonyms": { "gigantic": "huge", "distant": "far away" }
  }
]
```

- `text` is the only required field.
- **Use `\n\n` between paragraphs.** The reading view turns those into real
  paragraph breaks; single newlines are treated as ordinary spaces.
- `synonyms` keys should be **lowercase**. They become the hover tooltip on every
  highlighted word, so a word with no synonym still highlights but shows only its
  CEFR level — the passage looks half-finished. Coverage of B1+ words is what
  makes the demo land.

---

## Afia — `models/difficulty_model.pkl` and `models/model_eval.json`

### The model

Save with `joblib.dump(clf, "difficulty_model.pkl")`. Plain `pickle` also loads.

**Fit on a named DataFrame, not a NumPy array.** This is the one that will bite us:

```python
X = df[["length", "syllable_count"]]   # a DataFrame, keeps column names
clf.fit(X, y)                          # -> clf.feature_names_in_ is populated
```

If you fit on `X.values` or a list, `feature_names_in_` is empty and the app has to
guess column order for every unseen word. Developer Mode shows a red warning when
this happens, so you'll see it immediately.

**Feature choice matters for us specifically.** For a word already in
`processed_vocab.csv` the app just looks up its level — the model isn't consulted.
The model is only used for words in a story that aren't in the vocabulary. For
those, the app can recompute `length` and `syllable_count` on the fly, but it
**cannot** recompute `frequency` or POS one-hots without pulling in `wordfreq` and
`nltk` at read time. Any feature it can't recompute is sent as `0`.

So: if you train on frequency and POS, out-of-vocabulary predictions degrade.
Two clean options —

1. Train the model on `length` + `syllable_count` only. Simplest, and the app is
   fully correct. Report the accuracy cost in the Results section.
2. Train on everything for the best score, and tell Keli — the app can add a
   `wordfreq` lookup so `frequency` is computed live too.

Either is fine. Just tell Keli which, and Developer Mode will name any feature
it's zero-filling.

### The evaluation file

Export this so Developer Mode shows **your real held-out numbers** instead of a
leaky live split:

```python
import json
from sklearn.metrics import confusion_matrix, precision_score, recall_score

labels = sorted(y.unique())
json.dump({
    "labels": [int(l) for l in labels],
    "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
    "accuracy": float((y_pred == y_test).mean()),
    "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
    "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
}, open("models/model_eval.json", "w"), indent=2)
```

Without this file the app still draws a confusion matrix, but it computes it on a
random split of `processed_vocab.csv` — rows the model was almost certainly trained
on. It's labelled as a smoke test in the UI, and **it must not go in the report.**

Report **macro** precision and recall, not just accuracy: Developer Mode shows the
level distribution, and if our CEFR bands are imbalanced then overall accuracy will
flatter the model while the rare levels do badly. That gap is exactly what the
ethics section asks about.

### The Logic Engine

Your brief lists "a function that takes a story, looks up every word's level and
flags the ones the user won't know." That already exists as the `LogicEngine` class
in `main_app.py` — it has to live there because it drives the rendering. Rather than
writing a second copy, the useful version of that task is the feature/evaluation
work above plus a review of `LogicEngine.level_of()`, since it encodes the
lookup → model → heuristic precedence you'd otherwise be specifying.
