"""Train a CEFR difficulty classifier the application can actually use.

WHY THIS EXISTS
---------------
Afia's ensemble (final_project.ipynb) is the project's primary modelling work,
but it cannot drive the reading view as built: it predicts three broad tiers
rather than six CEFR levels, it is never serialised to disk, and it depends on
768-dimensional sentence embeddings the app cannot recompute for an unseen word
at read time.

This is a deliberately smaller model that satisfies the interface contract in
TEAM_CONTRACT.md: six classes, saved to disk, and trained ONLY on features the
application can recompute for any word it meets in a passage —

    word length, syllable count, Zipf word frequency

Frequency is recoverable because Jadzia's `frequency` column is the wordfreq
Zipf scale (verified: correlation 1.000 on a 300-word sample), so the same value
can be computed live rather than looked up.

Outputs models/difficulty_model.pkl and models/model_eval.json.
"""
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import cross_val_score, train_test_split
from wordfreq import zipf_frequency

CEFR = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C1", 6: "C2"}
VOWELS = re.compile(r"[aeiouy]+")


def count_syllables(word: str) -> int:
    """Must match main_app.count_syllables exactly, or train/serve will diverge."""
    w = word.lower().strip("'-")
    if not w:
        return 1
    n = len(VOWELS.findall(w))
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ye")):
        n -= 1
    return max(1, n)


# --- data, cleaned exactly as the application cleans it ---------------------
df = pd.read_csv("data/processed_vocab.csv")
df["word"] = df["headword"].astype(str).str.strip().str.lower().str.split("/")
df = df.explode("word")
df["word"] = df["word"].str.strip()
df["level"] = df["CEFR"].astype(str).str.strip().str.upper().map(
    {v: k for k, v in CEFR.items()})
df = df.dropna(subset=["word", "level"])
df["level"] = df["level"].astype(int)
df = df[df["word"].str.match(r"^[a-z][a-z'-]*$", na=False)]
df = df.sort_values("level").drop_duplicates("word", keep="first")

X = pd.DataFrame({
    "length": df["word"].str.len(),
    "syllable_count": df["word"].map(count_syllables),
    "frequency": df["word"].map(lambda w: zipf_frequency(w, "en")),
})
y = df["level"].values
print(f"training rows: {len(X)}   features: {list(X.columns)}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# class_weight='balanced' because the CEFR bands are imbalanced 2.8:1 and the
# rare levels are exactly the readers we care about not failing.
clf = RandomForestClassifier(
    n_estimators=120, max_depth=12, min_samples_leaf=8,
    class_weight="balanced", random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)                       # fit on a DataFrame -> feature names recorded

y_pred = clf.predict(X_te)
labels = sorted(np.unique(y))
acc = float((y_pred == y_te).mean())
within1 = float((np.abs(y_pred - y_te) <= 1).mean())
cv = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

print(f"\naccuracy            {acc:.3f}")
print(f"within one band     {within1:.3f}")
print(f"5-fold CV accuracy  {cv.mean():.3f} +/- {cv.std():.3f}")
print("\n" + classification_report(y_te, y_pred, labels=labels,
                                   target_names=[CEFR[l] for l in labels],
                                   zero_division=0))

joblib.dump(clf, "models/difficulty_model.pkl", compress=3)
json.dump({
    "labels": [int(l) for l in labels],
    "confusion_matrix": confusion_matrix(y_te, y_pred, labels=labels).tolist(),
    "accuracy": acc,
    "within_one_band": within1,
    "macro_precision": float(precision_score(y_te, y_pred, average="macro", zero_division=0)),
    "macro_recall": float(recall_score(y_te, y_pred, average="macro", zero_division=0)),
    "macro_f1": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
    "cv_accuracy_mean": float(cv.mean()),
    "cv_accuracy_std": float(cv.std()),
    "n_test": int(len(y_te)),
    "features": list(X.columns),
}, open("models/model_eval.json", "w"), indent=2)
print("\nwrote models/difficulty_model.pkl and models/model_eval.json")
print("feature importance:",
      {n: round(float(v), 3) for n, v in zip(X.columns, clf.feature_importances_)})
