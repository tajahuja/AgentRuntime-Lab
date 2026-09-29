from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import Document, RetrievalResult

class TfidfRetriever:
    """Small deterministic lexical-semantic baseline using TF-IDF cosine similarity."""

    def __init__(self, documents: list[Document]):
        if not documents:
            raise ValueError("documents must not be empty")
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([d.text for d in documents])

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        if not query.strip():
            return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        indices = scores.argsort()[::-1][:top_k]
        return [
            RetrievalResult(
                doc_id=self.documents[i].doc_id,
                text=self.documents[i].text,
                score=float(scores[i]),
                metadata=self.documents[i].metadata,
            )
            for i in indices if scores[i] > 0
        ]
