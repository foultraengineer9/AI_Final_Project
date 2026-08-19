# predictor.py
import os
import re
import json
import joblib
import numpy as np
import spacy
import syllables
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from groq import Groq

class WordComplexityPredictor:
    def __init__(self, bundle_dir: str = "./export_bundle", groq_api_key: str = None):
        print("Loading configuration and metadata...")
        with open(os.path.join(bundle_dir, "config.json"), "r") as f:
            self.config = json.load(f)

        print("Loading NLTK and NLP tools...")
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load(self.config.get("spacy_model_name", "en_core_web_sm"))
        self.transformer = SentenceTransformer(self.config.get("embedding_model_name", "all-mpnet-base-v2"))

        print("Loading models and scalers...")
        self.xgb_clf = joblib.load(os.path.join(bundle_dir, "xgb_clf.joblib"))
        self.rf_clf = joblib.load(os.path.join(bundle_dir, "rf_clf.joblib"))
        self.late_fusion_model = tf.keras.models.load_model(os.path.join(bundle_dir, "late_fusion_model.keras"))
        
        self.struct_scaler = joblib.load(os.path.join(bundle_dir, "struct_scaler.joblib"))
        self.pipeline_scaler = joblib.load(os.path.join(bundle_dir, "pipeline_scaler.joblib"))
        self.pipeline_kmeans = joblib.load(os.path.join(bundle_dir, "pipeline_kmeans.joblib"))

        print("Loading psycholinguistic lookup tables...")
        lookups = joblib.load(os.path.join(bundle_dir, "lookups.joblib"))
        self.frequency_dict = lookups["frequency_dict"]
        self.concreteness_mean = lookups["concreteness_mean"]
        self.concreteness_sd = lookups["concreteness_sd"]
        self.percent_known = lookups["percent_known"]
        self.subtlex_freq = lookups["subtlex_freq"]

        self.n_clusters = self.config.get("n_clusters", 6)
        self.weights = self.config.get("ensemble_weights", {"xgb": 0.45, "rf": 0.15, "mlp": 0.40})
        
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None

        self.affix_suffixes = ['characterization', 'ization', 'ability', 'tional', 'ative', 'istic', 'ology', 'ment', 'ship']
        self.affix_prefixes = ['counter', 'pseudo', 'trans', 'mono', 'hyper', 'inter', 'retro', 'ultra', 'circum']
        print("✓ System ready for deployment.")

    def _extract_word_features(self, word_clean: str, tag: str):
        # Base Lemma Resolution
        pos_char = 'n'
        if tag.startswith('JJ'): pos_char = 'a'
        elif tag.startswith('RB'): pos_char = 'r'
        elif tag.startswith('VB'): pos_char = 'v'
        orig_lemma = self.lemmatizer.lemmatize(word_clean, pos_char)

        freq_val = self.frequency_dict.get(orig_lemma, 0.0)
        if freq_val == 0.0 and orig_lemma != word_clean:
            freq_val = self.frequency_dict.get(word_clean, 0.0)

        length = len(word_clean)
        syll_count = syllables.estimate(word_clean)
        vowels = sum(1 for c in word_clean if c in 'aeiouy')
        vowel_ratio = round(vowels / length, 2) if length > 0 else 0.0
        freq_scaled = np.log1p(freq_val)
        syll_per_char = syll_count / length if length > 0 else 0.0
        freq_syll_interaction = freq_scaled * syll_count

        # Entropy & Clusters
        counts = {}
        for c in word_clean: counts[c] = counts.get(c, 0) + 1
        entropy = -sum((cnt / length) * np.log2(cnt / length) for cnt in counts.values()) if length > 0 else 0.0
        
        consonants = re.findall(r'[^aeiouy\W]+', word_clean)
        cluster_density = max(len(c) for c in consonants) if consonants else 0

        has_suffix = 1 if any(word_clean.endswith(s) for s in self.affix_suffixes) else 0
        has_prefix = 1 if any(word_clean.startswith(p) for p in self.affix_prefixes) else 0

        pos_vec = [0, 0, 0, 0, 0]
        if tag.startswith('JJ'): pos_vec[0] = 1
        elif tag.startswith('RB'): pos_vec[1] = 1
        elif tag.startswith('NN'): pos_vec[2] = 1
        elif tag.startswith('VB'): pos_vec[3] = 1
        else: pos_vec[4] = 1

        is_derived = 1 if self.lemmatizer.lemmatize(word_clean) != word_clean else 0
        letter_diversity = len(set(word_clean)) / length if length > 0 else 0.0

        conc_mean = self.concreteness_mean.get(word_clean, 3.0)
        conc_sd = self.concreteness_sd.get(word_clean, 1.0)
        p_known = self.percent_known.get(word_clean, 0.90)
        subtlex_val = self.subtlex_freq.get(word_clean, 0.0)
        subtlex_scaled = np.log1p(subtlex_val)

        synsets = wn.synsets(word_clean)
        if synsets:
            polysemy = float(len(synsets))
            max_depth = float(max(s.max_depth() for s in synsets))
            hyponyms = float(len(synsets[0].hyponyms()))
        else:
            polysemy, max_depth, hyponyms = 0.0, 0.0, 0.0

        raw_struct = [
            length, syll_count, freq_scaled, vowel_ratio,
            pos_vec[0], pos_vec[1], pos_vec[2], pos_vec[3], pos_vec[4],
            syll_per_char, freq_syll_interaction,
            round(entropy, 4), cluster_density, has_suffix, has_prefix,
            conc_mean, conc_sd, p_known, subtlex_scaled,
            polysemy, max_depth, hyponyms, is_derived, letter_diversity
        ]

        scaled_struct = self.pipeline_scaler.transform([raw_struct])
        cluster_id = self.pipeline_kmeans.predict(scaled_struct)[0]
        cluster_onehot = [0] * self.n_clusters
        cluster_onehot[cluster_id] = 1
        final_struct = raw_struct + cluster_onehot

        semantic_vector = self.transformer.encode(word_clean, convert_to_numpy=True)
        return semantic_vector, np.array(final_struct)

    def analyze_passage(self, passage: str, target_comfort_level: int = 1):
        doc = self.nlp(passage)
        protected_indices = set(token.i for token in doc if token.dep_ in ['nsubj', 'nsubjpass'] or token.pos_ == 'PROPN')

        local_results = []
        challenging_indices = []

        for token in doc:
            idx = token.i
            word = token.text
            tag = token.tag_

            if token.is_punct or token.is_space:
                local_results.append({"word": word, "pos": tag, "predicted_tier": 0, "status": "OK", "suggestion": None, "is_protected": True})
                continue

            word_lower = word.lower()
            is_content = token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']
            is_proper_noun_tag = token.pos_ == 'PROPN' or tag in ['NNP', 'NNPS']
            is_protected = (idx in protected_indices) or (idx > 0 and word[0].isupper() and is_content) or is_proper_noun_tag

            sem_vec, struct_vec = self._extract_word_features(word_lower, tag)
            scaled_struct = self.struct_scaler.transform([struct_vec])
            combined_feat = np.hstack((sem_vec.reshape(1, -1), scaled_struct))

            xgb_p = self.xgb_clf.predict_proba(combined_feat)
            rf_p = self.rf_clf.predict_proba(combined_feat)
            mlp_p = self.late_fusion_model.predict([sem_vec.reshape(1, -1), scaled_struct], verbose=0)

            fused_p = (self.weights["xgb"] * xgb_p) + (self.weights["rf"] * rf_p) + (self.weights["mlp"] * mlp_p)
            predicted_tier = int(np.argmax(fused_p, axis=-1)[0])

            is_difficult = is_content and not is_protected and (predicted_tier > target_comfort_level)
            is_advanced_proper = is_proper_noun_tag and (predicted_tier > target_comfort_level)

            local_results.append({
                "word": word,
                "pos": tag,
                "predicted_tier": predicted_tier,
                "status": "CHALLENGING" if is_difficult else "OK",
                "suggestion": None,
                "is_protected": is_protected
            })

            if is_difficult or is_advanced_proper:
                challenging_indices.append(idx)

        # Chunks grouping
        chunks = []
        if challenging_indices:
            curr_chunk = [challenging_indices[0]]
            for idx in challenging_indices[1:]:
                is_sep = any(doc[m].is_punct for m in range(curr_chunk[-1] + 1, idx))
                if idx - curr_chunk[-1] <= 2 and not is_sep:
                    for mid in range(curr_chunk[-1] + 1, idx + 1):
                        if mid not in curr_chunk: curr_chunk.append(mid)
                else:
                    chunks.append(curr_chunk)
                    curr_chunk = [idx]
            chunks.append(curr_chunk)

        phrases_llm = []
        for chunk in chunks:
            s_idx = chunk[0]
            if s_idx > 0 and not doc[s_idx - 1].is_punct and not doc[s_idx - 1].is_space: s_idx -= 1
            e_idx = chunk[-1]
            if e_idx < len(doc) - 1 and not doc[e_idx + 1].is_punct and not doc[e_idx + 1].is_space: e_idx += 1

            phrases_llm.append({
                "id": f"B{len(phrases_llm) + 1}",
                "original_bracket_phrase": "".join(doc[i].text_with_ws for i in range(s_idx, e_idx + 1)),
                "target_to_simplify": " ".join(doc[i].text for i in chunk),
                "indices": chunk,
                "start_idx": s_idx,
                "end_idx": e_idx
            })

        # LLM Simplification
        simplification_map = {}
        if phrases_llm and self.groq_client:
            payload = [{"id": p["id"], "original_bracket_phrase": p["original_bracket_phrase"].strip(), "target_to_simplify": p["target_to_simplify"]} for p in phrases_llm]
            prompt = f"""Original Passage: "{passage}"\nSimplify the target terms for a Tier {target_comfort_level} reader.\nTarget Objects:\n{json.dumps(payload, indent=2)}\nReturn ONLY a JSON object mapping each ID to the rewritten phrase."""
            try:
                res = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a vocabulary simplification API that returns ONLY valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    model="openai/gpt-oss-20b",
                    temperature=0.1,
                    max_tokens=2048,
                    extra_body={"reasoning_format": "hidden"}
                )
                raw_text = (res.choices[0].message.content or "").strip()
                json_match = re.search(r"\{[\s\S]*\}", raw_text)
                if json_match: simplification_map = json.loads(json_match.group(0))
            except Exception:
                pass

        # Reconstruction
        output_texts = [token.text_with_ws for token in doc]
        for item in phrases_llm:
            sugg = simplification_map.get(item["id"]) or simplification_map.get(item["id"].lower())
            if sugg:
                clean_sugg = re.sub(r'^["\']|["\']$', '', sugg)
                ws = doc[item["end_idx"]].text_with_ws[len(doc[item["end_idx"]].text):]
                output_texts[item["start_idx"]] = clean_sugg + ws
                for o_idx in range(item["start_idx"] + 1, item["end_idx"] + 1):
                    output_texts[o_idx] = ""
                local_results[item["indices"][0]]["suggestion"] = clean_sugg

        return "".join(output_texts), local_results