"""Build stories_library.json from Precious's passages + Jadzia's CEFR vocabulary.

Precious's notebook picked the WordNet synonym with the highest spaCy cosine
similarity. Similarity does not imply simplicity, so it could gloss a hard word
with a harder one ("density" -> "denseness"). This version grades every
candidate against Jadzia's processed_vocab.csv and keeps only synonyms that are
strictly EASIER than the word they explain, which is the whole point of a gloss.

Using Jadzia's CSV rather than cefrpy also removes the second CEFR source, so
the tooltips and the app's highlighting now agree by construction.
"""
import json
import re
from collections import Counter

import pandas as pd
from nltk.corpus import wordnet as wn

CEFR = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C1", 6: "C2"}

# --- Jadzia's vocabulary, cleaned the same way the app cleans it -------------
df = pd.read_csv("data/processed_vocab.csv")
df["word"] = df["headword"].astype(str).str.strip().str.lower()
df["word"] = df["word"].str.split("/")
df = df.explode("word")
df["word"] = df["word"].str.strip()
df["level"] = df["CEFR"].astype(str).str.strip().str.upper().map(
    {v: k for k, v in CEFR.items()}
)
df = df.dropna(subset=["word", "level"])
df["level"] = df["level"].astype(int)
df = df[df["word"].str.match(r"^[a-z][a-z'-]*$", na=False)]
df = df.sort_values("level").drop_duplicates("word", keep="first")
LEVEL = dict(zip(df.word, df.level))
FREQ = dict(zip(df.word, df["frequency"])) if "frequency" in df else {}
POS = dict(zip(df.word, df["POS"].astype(str).str.lower())) if "POS" in df else {}

# Jadzia's POS labels -> WordNet POS tags, so we only consider synsets for the
# part of speech the word actually is. Without this, WordNet's rarer senses
# produce wrong glosses ("provided" -> "leave", "outer" -> "out").
POS_TO_WN = {"noun": "n", "verb": "v", "adjective": "a", "adverb": "r"}
print(f"vocabulary: {len(LEVEL)} words")

stories = json.load(open("_stories_raw.json"))
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "has", "have",
    "had", "not", "can", "will", "would", "there", "their", "them", "they",
}


MIN_CAND_FREQ = 2.8     # Zipf; below this the "simpler" word is itself obscure


def _same_stem(a: str, b: str) -> bool:
    """True for inflectional variants ("attached"/"attach"), which gloss nothing."""
    if a.startswith(b) or b.startswith(a):
        return True
    return len(a) > 4 and len(b) > 4 and a[:5] == b[:5]


def best_synonym(word: str, level: int):
    """Return the easiest WordNet synonym that is strictly simpler than `word`.

    Three filters keep the glosses usable:
      1. POS alignment  - only synsets matching Jadzia's POS for this word.
      2. Sense priority - only the two most common senses, since WordNet orders
                          synsets by frequency and the tail senses are noise.
      3. Frequency floor- the replacement must itself be a common word.
    """
    wn_pos = POS_TO_WN.get(POS.get(word, ""))
    synsets = wn.synsets(word, pos=wn_pos) if wn_pos else wn.synsets(word)
    if not synsets:
        return None

    candidates = {}
    for syn in synsets[:3]:
        for lemma in syn.lemmas():
            cand = lemma.name().replace("_", " ").lower()
            if cand == word or " " in cand or "-" in cand or len(cand) < 2:
                continue
            if _same_stem(cand, word):
                continue
            cand_level = LEVEL.get(cand)
            if cand_level is None or cand_level >= level:
                continue
            cand_freq = float(FREQ.get(cand, 0))
            if cand_freq < MIN_CAND_FREQ:
                continue
            candidates[cand] = (cand_level, -cand_freq)
    if not candidates:
        return None
    return min(candidates, key=candidates.get)


out, stats = [], Counter()
for story in stories:
    text = story["text"]
    synonyms = {}
    for token in WORD_RE.findall(text):
        w = token.lower().strip("'-")
        if w in STOP or w in synonyms or len(w) < 4:
            continue
        level = LEVEL.get(w)
        if level is None or level < 3:      # only gloss B1 and above
            continue
        syn = best_synonym(w, level)
        if syn:
            synonyms[w] = syn
            stats[CEFR[level]] += 1
    out.append({"title": story["title"], "text": text, "synonyms": synonyms})

json.dump(out, open("data/stories_library.json", "w"), indent=2, ensure_ascii=False)

covered = sum(len(s["synonyms"]) for s in out)
print(f"\npassages: {len(out)}")
print(f"synonyms mapped: {covered}  (mean {covered/len(out):.1f} per passage)")
print(f"passages with zero synonyms: {sum(1 for s in out if not s['synonyms'])}")
print("by level of the glossed word:", dict(sorted(stats.items())))
print("\nsamples:")
for s in out[:3]:
    print(f"  {s['title']}: " + ", ".join(f"{k}->{v}" for k, v in list(s["synonyms"].items())[:5]))
