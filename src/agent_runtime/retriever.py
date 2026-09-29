"""Small deterministic TF-IDF retriever implemented with the Python standard library."""

from __future__ import annotations

import math
import re
from collections import Counter

from .models import Document, RetrievalResult

_TOKEN = re.compile(r"[a-zA-Z0-9_]+")
_STOP_WORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of", "on", "or", "the", "to", "what", "when", "why", "with"}


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text) if t.lower() not in _STOP_WORDS]


class TfidfRetriever:
    """Cosine similarity over unigram TF-IDF vectors; scores are not calibrated."""

    def __init__(self, documents: list[Document]):
        if not documents:
            raise ValueError("documents must not be empty")
        if len({doc.doc_id for doc in documents}) != len(documents):
            raise ValueError("document ids must be unique")
        self.documents = list(documents)
        tokenized = [_tokens(doc.text) for doc in documents]
        document_frequency: Counter[str] = Counter()
        for words in tokenized:
            document_frequency.update(set(words))
        n = len(documents)
        self.idf = {word: math.log((1 + n) / (1 + df)) + 1 for word, df in document_frequency.items()}
        self.vectors = [self._vector(words) for words in tokenized]

    def _vector(self, words: list[str]) -> dict[str, float]:
        counts = Counter(words)
        vector = {word: (count / len(words)) * self.idf.get(word, 1.0) for word, count in counts.items()} if words else {}
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        return {word: weight / norm for word, weight in vector.items()} if norm else {}

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0 or not query.strip():
            return []
        query_vector = self._vector(_tokens(query))
        scored = []
        for document, vector in zip(self.documents, self.vectors):
            score = sum(weight * vector.get(word, 0.0) for word, weight in query_vector.items())
            if score > 0:
                scored.append((score, document))
        scored.sort(key=lambda pair: (-pair[0], pair[1].doc_id))
        return [
            RetrievalResult(doc.doc_id, doc.text, score, dict(doc.metadata))
            for score, doc in scored[:top_k]
        ]
