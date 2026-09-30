"""Small dependency-free BM25 retriever over concise project context docs."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


_TOKEN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_STOP_WORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do", "does",
               "for", "from", "had", "has", "have", "how", "i", "in", "is", "it", "me", "of",
               "on", "or", "our", "the", "this", "to", "was", "we", "what", "when", "where",
               "which", "who", "why", "with", "would", "you"}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text) if token.lower() not in _STOP_WORDS]


class BM25Retriever:
    """BM25 index suitable for this small, static project knowledge base."""
    def __init__(self, documents: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = float(k1)
        self.b = float(b)
        self.tokens = [tokenize(doc.get("text", "") + " " + doc.get("title", "")) for doc in self.documents]
        self.term_counts = [Counter(tokens) for tokens in self.tokens]
        self.avg_length = sum(map(len, self.tokens)) / len(self.tokens) if self.tokens else 0.0
        self.document_frequency: Counter = Counter()
        for tokens in self.tokens:
            self.document_frequency.update(set(tokens))

    def search(self, query: str, limit: int = 4, min_score: float = 0.0) -> list[dict[str, Any]]:
        """Return highest-scoring context chunks with transparent scores."""
        if limit <= 0:
            return []
        query_terms = tokenize(query)
        if not query_terms or not self.documents:
            return []
        n_docs = len(self.documents)
        results = []
        for index, (doc, counts) in enumerate(zip(self.documents, self.term_counts)):
            doc_len = len(self.tokens[index])
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                df = self.document_frequency.get(term, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_length, 1))
                score += idf * frequency * (self.k1 + 1) / denominator
            if score >= min_score and score > 0:
                results.append({**doc, "score": score})
        return sorted(results, key=lambda item: (-item["score"], item["id"]))[:limit]
