import math
from collections import Counter
from typing import List, Dict, Any


class HybridRAGEngine:
    """Simplified hybrid-retrieval engine with RRF-style rank fusion."""

    def __init__(self, corpus: List[str]):
        self.corpus = corpus

    # 1. Simulated sparse search using keyword matching.
    def _sparse_search(self, query: str) -> List[int]:
        query_words = set(query.lower().split())
        scores = []
        for idx, doc in enumerate(self.corpus):
            doc_words = doc.lower().split()
            # Count shared words as a simplified BM25-like signal.
            score = sum(1 for word in doc_words if word in query_words)
            scores.append((idx, score))
        # Sort by the highest score.
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, score in scores if score > 0]

    # 2. Simulated dense search (vector similarity).
    def _dense_search(self, query: str) -> List[int]:
        # Use a deterministic bag-of-words cosine proxy in this dependency-free demo.
        query_counts = Counter(query.lower().split())
        query_norm = math.sqrt(sum(value ** 2 for value in query_counts.values()))
        if query_norm == 0:
            return []

        scores = []
        for idx, doc in enumerate(self.corpus):
            doc_counts = Counter(doc.lower().split())
            doc_norm = math.sqrt(sum(value ** 2 for value in doc_counts.values()))
            if doc_norm == 0:
                continue
            dot_product = sum(query_counts[word] * doc_counts[word] for word in query_counts)
            similarity = dot_product / (query_norm * doc_norm)
            if similarity > 0:
                scores.append((idx, similarity))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores]

    # 3. Merge ranks with Reciprocal Rank Fusion (RRF).
    def hybrid_retrieve(self, query: str, top_k: int = 2, k_constant: int = 60) -> List[Dict[str, Any]]:
        sparse_ranks = self._sparse_search(query)
        dense_ranks = self._dense_search(query)

        rrf_scores: Dict[int, float] = {}

        # Calculate RRF contributions for sparse results.
        for rank, doc_id in enumerate(sparse_ranks):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_constant + rank + 1))

        # Calculate RRF contributions for dense results.
        for rank, doc_id in enumerate(dense_ranks):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_constant + rank + 1))

        # Sort by the combined RRF score.
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:top_k]:
            results.append({
                "doc_id": doc_id,
                "text": self.corpus[doc_id],
                "rrf_score": round(score, 5)
            })

        return results


# --- Hybrid-retrieval demo ---
if __name__ == "__main__":
    documents = [
        "System architecture defined with deterministic FSM and durable execution.",
        "Error handling guide: HTTP 429 rate limits and exponential backoff retry.",
        "RAG context window management: combining BM25 sparse search with dense vectors.",
        "Database indexing techniques for high throughput key-value stores."
    ]

    engine = HybridRAGEngine(corpus=documents)
    
    # Combine an exact keyword query with a broad conceptual signal.
    search_query = "RAG search architecture 429"
    retrieved = engine.hybrid_retrieve(query=search_query, top_k=2)

    print(f"[HYBRID SEARCH QUERY]: '{search_query}'\n")
    for rank, res in enumerate(retrieved, start=1):
        print(f"Rank {rank} (RRF Score: {res['rrf_score']}):")
        print(f"  Content: \"{res['text']}\"\n")
