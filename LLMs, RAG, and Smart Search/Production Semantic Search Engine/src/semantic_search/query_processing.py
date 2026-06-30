"""Query normalization, expansion, and correction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import get_close_matches

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


WORD_RE = re.compile(r"[a-zA-Z0-9_]+")

DEFAULT_SYNONYMS = {
    "ai": ["artificial intelligence", "machine learning"],
    "nlp": ["natural language processing"],
    "gpu": ["graphics processing unit"],
    "llm": ["large language model", "language model"],
    "doc": ["document", "article"],
}

DEFAULT_ABBREVIATIONS = {
    "api": "application programming interface",
    "etl": "extract transform load",
    "qa": "question answering",
    "rag": "retrieval augmented generation",
}


@dataclass(slots=True)
class QueryProcessor:
    """Applies normalization and light query rewriting."""

    lowercase: bool = True
    remove_stopwords: bool = True
    spell_correction: bool = True
    synonym_expansion: bool = True
    abbreviation_expansion: bool = True
    synonyms: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_SYNONYMS))
    abbreviations: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ABBREVIATIONS))
    vocabulary: set[str] = field(default_factory=set)

    def build_vocabulary(self, corpus_texts: list[str]) -> None:
        """Build vocabulary used for spell correction."""
        vocab: set[str] = set()
        for text in corpus_texts:
            vocab.update(self._tokens(text))
        self.vocabulary = vocab

    def process(self, query: str) -> str:
        """Run end-to-end query processing."""
        q = query.strip()
        if self.lowercase:
            q = q.lower()

        tokens = self._tokens(q)

        if self.abbreviation_expansion:
            expanded_tokens: list[str] = []
            for token in tokens:
                expanded_tokens.extend(self.abbreviations.get(token, token).split())
            tokens = expanded_tokens

        if self.spell_correction and self.vocabulary:
            corrected: list[str] = []
            for token in tokens:
                if token in self.vocabulary:
                    corrected.append(token)
                    continue
                matches = get_close_matches(token, self.vocabulary, n=1, cutoff=0.86)
                corrected.append(matches[0] if matches else token)
            tokens = corrected

        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]

        if self.synonym_expansion:
            extra_terms: list[str] = []
            for token in tokens:
                extra_terms.extend(self.synonyms.get(token, []))
            if extra_terms:
                tokens.extend(" ".join(extra_terms).split())

        return " ".join(tokens)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [match.group(0).lower() for match in WORD_RE.finditer(text)]
