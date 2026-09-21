import os
import logging

logger = logging.getLogger(__name__)

# Knowledge-base answer floor. When even the best passage scores below this, nothing
# retrieved is about the question, and the knowledge-base path declines instead of
# generating. -5.0 is the reranker's own relevance floor (min_relevance_score): the
# reranker already logged "all documents scored below threshold" and then passed its
# top result on regardless; the floor makes the pipeline act on that verdict.
# Measured (eval/ABSTENTION.md): stops 22 of 24 unanswerable questions by itself and,
# of 64 answerable ones, only one — which was being answered wrongly anyway.
KB_ANSWER_MIN_SCORE = float(os.getenv("KB_ANSWER_MIN_SCORE", "-5.0"))


def below_answer_floor(best_score: float | None, floor: float | None = None) -> bool:
    """True when the best passage is too weak to answer from. None (reranker unavailable,
    nothing was scored) is never below the floor: absence of a score is not evidence."""
    floor = KB_ANSWER_MIN_SCORE if floor is None else floor
    return best_score is not None and best_score < floor


class RerankerModel:
    """
    Reranks retrieved documents to heavily penalize documents that aren't actually relevant to the query.
    Uses a smaller HuggingFace CrossEncoder model.
    Falls back gracefully to a pass-through if the model cannot be loaded (low-memory cloud environments).
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", min_relevance_score: float = -5.0):
        self.model = None
        self.model_name = model_name
        self.enabled = True
        self.min_relevance_score = min_relevance_score
        
        try:
            from sentence_transformers import CrossEncoder
            # max_length is overridable; measured on the eval set, 256 cost one top-5 hit
            # (recall@5 0.922 -> 0.906) and saved only ~0.15 s, so the default stays 512.
            self.model = CrossEncoder(self.model_name, max_length=int(os.getenv("RERANK_MAX_LENGTH", "512")))
        except Exception as e:
            logger.warning(f"Could not load CrossEncoder reranker ({e}). Reranking will be bypassed.")
            self.enabled = False

    def rerank_with_scores(self, query: str, documents: list[str], top_k: int = 3) -> list[tuple[float | None, str]]:
        """
        rerank(), keeping each document's cross-encoder score: [(score, doc), ...] best first.
        The score is None when the reranker is unavailable (pass-through), so callers can tell
        "scored low" from "not scored". The best score is what the knowledge-base answer gate
        reads: below KB_ANSWER_MIN_SCORE nothing retrieved is about the question.
        """
        if not documents:
            return []

        if not self.enabled or self.model is None:
            # Pass-through fallback: return top_k directly
            return [(None, doc) for doc in documents[:top_k]]

        try:
            # pairs for cross encoder: (query, doc1), (query, doc2)...
            pairs = [[query, doc] for doc in documents]

            # scores represent relevance
            scores = self.model.predict(pairs)

            # Log relevance scores for debugging and transparency
            scored_pairs = sorted(((float(sc), doc) for sc, doc in zip(scores, documents)), key=lambda x: x[0], reverse=True)
            for i, (score, doc) in enumerate(scored_pairs[:top_k + 2]):
                preview = doc[:80].replace('\n', ' ')
                logger.info(f"Reranker [{i+1}] score={score:.4f}: \"{preview}...\"")

            # Filter out documents below minimum relevance threshold
            filtered = [(score, doc) for score, doc in scored_pairs if score >= self.min_relevance_score]

            if not filtered:
                logger.warning(f"Reranker: All documents scored below threshold ({self.min_relevance_score}). Returning top result anyway.")
                filtered = [scored_pairs[0]]

            return filtered[:top_k]
        except Exception as e:
            logger.error(f"Reranking failed at runtime ({e}). Falling back to pass-through.")
            return [(None, doc) for doc in documents[:top_k]]

    def rerank(self, query: str, documents: list[str], top_k: int = 3) -> list[str]:
        return [doc for _, doc in self.rerank_with_scores(query, documents, top_k=top_k)]
