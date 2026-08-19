"""predictor_notf.py — Afia's ensemble, minus the TensorFlow component.

WHY THIS FILE EXISTS
--------------------
Her ensemble fuses three classifiers with the weights in export_bundle/config.json:

    XGBoost 0.45  +  Random Forest 0.15  +  late-fusion Keras net 0.40

Loading all three in one process segfaults on Apple Silicon. sentence-transformers
pulls in PyTorch, the late-fusion model pulls in TensorFlow, and the two ship
incompatible copies of the same low-level math libraries; the interpreter is
killed at `tf.keras.models.load_model`, after the embedding model has loaded.

This variant loads the two scikit-learn components only and renormalises their
weights to sum to 1:

    XGBoost 0.75  +  Random Forest 0.25

Everything else — feature extraction, the sentence embeddings, the KMeans
cluster feature, the psycholinguistic lookups, the proper-noun and subject
protection rules — is imported unchanged from her predictor. Only the
constructor and the fusion line differ.

This is a reduced ensemble and must be reported as one. It is not her full model.
"""
from __future__ import annotations

import json
import os
import sys

# ORDER-CRITICAL IMPORT.
# XGBoost and PyTorch each bundle their own OpenMP runtime. On Apple Silicon the
# second one to load segfaults the interpreter. Importing xgboost here, before
# anything pulls in torch, makes the two coexist. Verified: loading xgboost
# after sentence-transformers dies at joblib.load; the reverse order runs.
import xgboost  # noqa: F401  - must precede any torch import

import joblib
import numpy as np


def _import_her_module(module_dir: str):
    """Import predictor.py with TensorFlow stubbed out.

    predictor.py does `import tensorflow as tf` at module level and only uses it
    in the constructor. Registering a dummy module lets us reuse every one of her
    feature-extraction methods without TensorFlow ever being loaded.
    """
    import importlib.machinery
    import types

    # xgboost is already imported at module level, so torch may now load safely.
    import sentence_transformers  # noqa: F401

    if "tensorflow" not in sys.modules:
        stub = types.ModuleType("tensorflow")
        # A bare ModuleType has __spec__ = None, which makes importlib's
        # find_spec() raise instead of returning None. Give it a real spec.
        stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
        stub.__version__ = "0.0.0-stub"
        stub.keras = types.SimpleNamespace(
            models=types.SimpleNamespace(load_model=lambda *a, **k: None)
        )
        sys.modules["tensorflow"] = stub

    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    import predictor  # noqa: E402  - imported after the stub is registered
    return predictor


class ReducedEnsemblePredictor:
    """Her WordComplexityPredictor with the neural component omitted."""

    def __init__(self, bundle_dir: str = "wordify/export_bundle",
                 module_dir: str = "wordify"):
        her = _import_her_module(module_dir)
        cls = her.WordComplexityPredictor

        # Build the object without running her __init__, then populate exactly
        # the attributes her methods expect, skipping the Keras load.
        self._inner = cls.__new__(cls)
        obj = self._inner

        with open(os.path.join(bundle_dir, "config.json")) as f:
            obj.config = json.load(f)

        # Every joblib artifact loads BEFORE the transformer, for the same
        # OpenMP reason: the sklearn/xgboost models must be in memory before
        # torch initialises.
        obj.xgb_clf = joblib.load(os.path.join(bundle_dir, "xgb_clf.joblib"))
        obj.rf_clf = joblib.load(os.path.join(bundle_dir, "rf_clf.joblib"))
        obj.late_fusion_model = None                     # deliberately absent

        obj.struct_scaler = joblib.load(os.path.join(bundle_dir, "struct_scaler.joblib"))
        obj.pipeline_scaler = joblib.load(os.path.join(bundle_dir, "pipeline_scaler.joblib"))
        obj.pipeline_kmeans = joblib.load(os.path.join(bundle_dir, "pipeline_kmeans.joblib"))

        from nltk.stem import WordNetLemmatizer
        import spacy
        from sentence_transformers import SentenceTransformer

        obj.lemmatizer = WordNetLemmatizer()
        obj.nlp = spacy.load(obj.config.get("spacy_model_name", "en_core_web_sm"))
        obj.transformer = SentenceTransformer(
            obj.config.get("embedding_model_name", "all-mpnet-base-v2"))

        lookups = joblib.load(os.path.join(bundle_dir, "lookups.joblib"))
        obj.frequency_dict = lookups["frequency_dict"]
        obj.concreteness_mean = lookups["concreteness_mean"]
        obj.concreteness_sd = lookups["concreteness_sd"]
        obj.percent_known = lookups["percent_known"]
        obj.subtlex_freq = lookups["subtlex_freq"]

        obj.n_clusters = obj.config.get("n_clusters", 6)
        obj.weights = obj.config.get(
            "ensemble_weights", {"xgb": 0.45, "rf": 0.15, "mlp": 0.40})

        obj.groq_api_key = None
        obj.groq_client = None                           # no LLM rewriting

        obj.affix_suffixes = ['characterization', 'ization', 'ability', 'tional',
                              'ative', 'istic', 'ology', 'ment', 'ship']
        obj.affix_prefixes = ['counter', 'pseudo', 'trans', 'mono', 'hyper',
                              'inter', 'retro', 'ultra', 'circum']

        # Renormalise the two surviving weights so they sum to 1.
        total = obj.weights["xgb"] + obj.weights["rf"]
        self.w_xgb = obj.weights["xgb"] / total
        self.w_rf = obj.weights["rf"] / total

    # --- prediction ---------------------------------------------------------
    def predict_word(self, word: str, tag: str = "NN") -> int:
        """Predicted tier (0/1/2) for a single word, XGBoost + RF only.

        Feature assembly mirrors her analyze_passage exactly: the sentence
        embedding concatenated with the scaled structural vector, and nothing
        else. An earlier version of this file appended a KMeans one-hot block,
        which changed the feature width and made every prediction tier 0.

        `tag` is spaCy's fine-grained tag_ (NN, VBD, JJ), not pos_ — that is
        what her _extract_word_features expects.
        """
        obj = self._inner
        sem_vec, struct_vec = obj._extract_word_features(word.lower(), tag)
        scaled_struct = obj.struct_scaler.transform([struct_vec])
        combined = np.hstack((sem_vec.reshape(1, -1), scaled_struct))

        fused = (self.w_xgb * obj.xgb_clf.predict_proba(combined)
                 + self.w_rf * obj.rf_clf.predict_proba(combined))
        return int(np.argmax(fused, axis=-1)[0])

    def analyze_passage(self, passage: str, target_comfort_level: int = 1):
        """Same contract as hers: (reconstructed_text, per-word audit list).

        Her own analyze_passage calls the Keras model directly, so the loop is
        reimplemented here using her feature extraction and protection rules.
        """
        obj = self._inner
        doc = obj.nlp(passage)
        protected_indices = {t.i for t in doc
                             if t.dep_ in ("nsubj", "nsubjpass") or t.pos_ == "PROPN"}

        audit = []
        for token in doc:
            word, tag = token.text, token.tag_

            if token.is_punct or token.is_space:
                audit.append({"word": word, "pos": tag, "predicted_tier": 0,
                              "status": "OK", "suggestion": None,
                              "is_protected": True})
                continue

            is_content = token.pos_ in ("NOUN", "VERB", "ADJ", "ADV")
            is_proper = token.pos_ == "PROPN" or tag in ("NNP", "NNPS")
            is_protected = (token.i in protected_indices
                            or (token.i > 0 and word[:1].isupper() and is_content)
                            or is_proper)

            try:
                tier = self.predict_word(word, tag)
            except Exception:                            # never break a passage
                tier = 0

            difficult = is_content and not is_protected and tier > target_comfort_level
            advanced_proper = is_proper and tier > target_comfort_level

            audit.append({
                "word": word, "pos": tag, "predicted_tier": tier,
                "status": "CHALLENGING" if (difficult or advanced_proper) else "OK",
                "suggestion": None, "is_protected": is_protected,
            })

        return passage, audit
