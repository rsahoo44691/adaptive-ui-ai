"""Adaptive engine for the AI-based adaptive UI demo.

This module is the "AI" behind the adaptive interface. It does three things:

1. Represents every article as a TF-IDF vector (a classic information-retrieval
   / machine-learning text representation).
2. Maintains an online user-preference model that is updated incrementally
   after every interaction (like / skip / read). This is a lightweight form of
   online learning -- the model adapts at runtime rather than being trained
   once offline.
3. Re-ranks the content feed with content-based recommendation (cosine
   similarity between the user vector and each item) plus an exploration term
   so the interface never collapses into a filter bubble.

It also tracks signals the *interface* adapts to: category affinity (which
sections to surface first) and a "deep reading" signal used to switch on an
accessibility-friendly focus reading mode.

The implementation deliberately uses only the Python standard library and
numpy so it is easy to read, has no heavy dependencies, and runs anywhere.
"""

import math
import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z]+")

# Very small English stop-word list -- enough to keep TF-IDF focused on
# meaningful terms without pulling in an external NLP package.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "be", "by", "that", "this", "it", "as", "at", "from", "how",
    "you", "your", "they", "their", "every", "into", "over", "than", "even",
    "so", "can", "make", "makes", "uses", "use", "used", "covers", "cover",
    "learn", "these", "through", "why", "what", "out", "up", "do", "does",
    "long", "read", "article", "guide", "introduction", "explains",
    "explained", "covers", "covering", "this", "that",
}


def _tokenize(text):
    return [t for t in _TOKEN_RE.findall(text.lower())
            if t not in _STOPWORDS and len(t) > 2]


class AdaptiveEngine:
    """Holds the corpus model plus the live, per-user adaptive state.

    A new instance is created per Gradio session, so each user gets an
    independent profile that adapts only to their own interactions.
    """

    # learning_rate controls how fast the profile moves toward liked content.
    LEARNING_RATE = 0.45
    # how much the profile is pushed away from skipped content.
    SKIP_RATE = 0.20
    # exploration weight: keeps some novelty in the ranking.
    EXPLORE_WEIGHT = 0.12
    # opening this many "long reads" turns on focus reading mode automatically.
    DEEP_READ_THRESHOLD = 2

    def __init__(self, articles):
        self.articles = articles
        self._build_tfidf()

        # ----- live, adaptive per-user state -----
        self.profile = np.zeros(len(self.vocab), dtype=float)
        self.category_affinity = {}      # category -> running score
        self.interactions = 0            # total interactions observed
        self.deep_reads = 0              # opened long-read articles
        self.reading_mode = False        # accessibility focus mode (auto)
        self.learning_enabled = True     # privacy control: pause learning
        self.seen = set()                # ids the user has acted on
        self.log = []                    # human-readable adaptation history

    # ------------------------------------------------------------------ #
    # Corpus model
    # ------------------------------------------------------------------ #
    def _build_tfidf(self):
        docs_tokens = [_tokenize(a["title"] + " " + a["body"])
                       for a in self.articles]

        vocab = sorted({tok for toks in docs_tokens for tok in toks})
        self.vocab = vocab
        self.index = {t: i for i, t in enumerate(vocab)}

        n_docs = len(self.articles)
        df = Counter(tok for toks in docs_tokens for tok in set(toks))
        # smoothed inverse document frequency
        self.idf = np.array(
            [math.log((1 + n_docs) / (1 + df[t])) + 1.0 for t in vocab],
            dtype=float,
        )

        self.doc_vectors = {}
        for art, toks in zip(self.articles, docs_tokens):
            self.doc_vectors[art["id"]] = self._vectorize(toks)

    def _vectorize(self, tokens):
        vec = np.zeros(len(self.vocab), dtype=float)
        if not tokens:
            return vec
        counts = Counter(tokens)
        max_c = max(counts.values())
        for tok, c in counts.items():
            i = self.index.get(tok)
            if i is not None:
                vec[i] = (c / max_c) * self.idf[i]   # tf * idf
        return _l2_normalize(vec)

    # ------------------------------------------------------------------ #
    # Online adaptation
    # ------------------------------------------------------------------ #
    def observe(self, article_id, action):
        """Update the user model from a single interaction.

        action is one of: "like", "skip", "read".
        Returns a short status string describing what adapted.
        """
        art = self._article(article_id)
        if art is None:
            return "No article selected."

        self.interactions += 1
        self.seen.add(article_id)
        vec = self.doc_vectors[article_id]
        cat = art["category"]
        notes = []

        if not self.learning_enabled:
            self.log.append(
                f"(learning paused) {action} on '{art['title']}' "
                f"was not used to adapt."
            )
            return "Learning is paused -- interaction ignored by the model."

        if action == "like":
            self.profile = _l2_normalize(
                (1 - self.LEARNING_RATE) * self.profile
                + self.LEARNING_RATE * vec
            )
            self.category_affinity[cat] = self.category_affinity.get(cat, 0) + 1.0
            notes.append(f"boosted '{cat}'")
        elif action == "skip":
            self.profile = _l2_normalize(
                self.profile - self.SKIP_RATE * vec
            )
            self.category_affinity[cat] = self.category_affinity.get(cat, 0) - 0.5
            notes.append(f"down-weighted '{cat}'")
        elif action == "read":
            # reading is a softer positive signal than an explicit like
            self.profile = _l2_normalize(
                (1 - self.LEARNING_RATE / 2) * self.profile
                + (self.LEARNING_RATE / 2) * vec
            )
            self.category_affinity[cat] = self.category_affinity.get(cat, 0) + 0.5
            notes.append(f"learned interest in '{cat}'")
            if art["reading_time"] >= 6:
                self.deep_reads += 1
                if (not self.reading_mode
                        and self.deep_reads >= self.DEEP_READ_THRESHOLD):
                    self.reading_mode = True
                    notes.append("enabled focus reading mode (larger text)")

        msg = f"{action.capitalize()} '{art['title']}' -> " + ", ".join(notes)
        self.log.append(msg)
        return msg

    def set_learning(self, enabled):
        self.learning_enabled = bool(enabled)
        state = "resumed" if enabled else "paused"
        self.log.append(f"User {state} adaptive learning (privacy control).")

    def set_reading_mode(self, enabled):
        self.reading_mode = bool(enabled)
        self.log.append(
            f"Reading mode {'on' if enabled else 'off'} (manual override)."
        )

    def reset(self):
        """Forget everything learned about the user (privacy control)."""
        self.profile = np.zeros(len(self.vocab), dtype=float)
        self.category_affinity = {}
        self.interactions = 0
        self.deep_reads = 0
        self.reading_mode = False
        self.seen = set()
        self.log = ["User reset their profile -- all learned data cleared."]

    # ------------------------------------------------------------------ #
    # Recommendation / ranking
    # ------------------------------------------------------------------ #
    def ranked(self):
        """Return articles ordered by adapted preference.

        Each entry is the article dict plus 'score' (0-1 relevance) and
        'reason' (a short transparency explanation).
        """
        has_profile = np.linalg.norm(self.profile) > 0
        scored = []
        for art in self.articles:
            vec = self.doc_vectors[art["id"]]
            relevance = float(np.dot(self.profile, vec)) if has_profile else 0.0
            # deterministic, stable "exploration" bump from category novelty
            explore = self.EXPLORE_WEIGHT * (0.0 if art["category"]
                                             in self.category_affinity else 1.0)
            score = max(0.0, relevance) + explore
            scored.append((score, relevance, art))

        # When nothing is learned yet, keep the original editorial order.
        if not has_profile:
            ordered = [(0.0, 0.0, a) for a in self.articles]
        else:
            ordered = sorted(scored, key=lambda t: t[0], reverse=True)

        results = []
        for score, relevance, art in ordered:
            item = dict(art)
            item["score"] = round(min(1.0, score), 3)
            item["reason"] = self._reason(art, relevance, has_profile)
            results.append(item)
        return results

    def _reason(self, art, relevance, has_profile):
        if not has_profile:
            return "Default editorial order -- start interacting to personalize."
        if art["id"] in self.seen:
            base = "You interacted with this; "
        else:
            base = ""
        if relevance <= 0:
            return base + "shown for variety (exploration)."
        terms = self._top_overlap_terms(art["id"])
        if terms:
            return base + "matches your interest in " + ", ".join(terms) + "."
        return base + f"related to '{art['category']}' you engaged with."

    def _top_overlap_terms(self, article_id, k=3):
        vec = self.doc_vectors[article_id]
        contrib = self.profile * vec
        if not np.any(contrib > 0):
            return []
        top_idx = np.argsort(contrib)[::-1][:k]
        return [self.vocab[i] for i in top_idx if contrib[i] > 0]

    # ------------------------------------------------------------------ #
    # Helpers for the UI
    # ------------------------------------------------------------------ #
    def affinity_summary(self):
        """Sorted (category, score) pairs for display, positives first."""
        if not self.category_affinity:
            return []
        return sorted(self.category_affinity.items(),
                      key=lambda kv: kv[1], reverse=True)

    def _article(self, article_id):
        for a in self.articles:
            if a["id"] == article_id:
                return a
        return None

    def export_profile(self):
        """Plain-text view of stored data (transparency / data portability)."""
        lines = [
            f"interactions: {self.interactions}",
            f"learning_enabled: {self.learning_enabled}",
            f"reading_mode: {self.reading_mode}",
            f"deep_reads: {self.deep_reads}",
            "category_affinity:",
        ]
        for cat, score in self.affinity_summary():
            lines.append(f"    {cat}: {round(score, 2)}")
        lines.append(f"articles_interacted: {sorted(self.seen)}")
        return "\n".join(lines)


def _l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
