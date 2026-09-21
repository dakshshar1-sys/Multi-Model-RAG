"""
Abstention: the detector, the dataset's integrity, the gate sweep, and the scored reranker.
No models: the reranker is driven by a fake cross-encoder.
"""
import glob
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.text_support import is_abstention
from eval.abstention import summarize, sweep
from retrieval.reranker import KB_ANSWER_MIN_SCORE, RerankerModel, below_answer_floor

EVAL = os.path.join(os.path.dirname(__file__), "..", "eval")
UNANSWERABLE = [json.loads(l) for l in open(os.path.join(EVAL, "datasets", "unanswerable.jsonl")) if l.strip() and not l.startswith("#")]
CORPUS = " ".join(open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(EVAL, "corpus", "*.txt"))).lower()


@pytest.mark.parametrize("answer", [
    "The provided context does not contain information about the project budget.",
    "The documents do not mention a Kubernetes configuration [1].",
    "There is no information about BLEU scores in the report.",
    "The license is not specified in the architecture document.",
    "I couldn't find this in your documents.",
    "I cannot determine the supervisor's name from the context.",
    "### Direct Answer\nThe report doesn't specify a deadline for the final submission.",
    "No relevant context found to answer the query.",
    "This is not explicitly stated in the sources provided.",
    "No one is explicitly named as the project supervisor in Chapter 8 of the report.",   # missed by the first detector (u12)
    "None of the sources mentions a Gantt chart.",
    "The documents do not have any information on the budget.",   # still caught, via the 'documents do not' pattern
])
def test_declining_answers_are_detected(answer):
    assert is_abstention(answer)


@pytest.mark.parametrize("answer", [
    "FAISS is the vector database used by the system [1].",
    "The total project budget is ₹50,000 [2].",
    "The system uses Whisper for speech-to-text.",
    "Early RAG systems were based on rule-based methods and did not scale [1].",   # 'did not scale' is a claim, not a refusal
    "Semantic-based error detection methods are simple and do not have context depth [1].",   # a claim about the methods (q10)
    "",
])
def test_asserting_answers_are_not(answer):
    assert not is_abstention(answer)


def test_a_late_hedge_does_not_excuse_an_opening_claim():
    answer = "The project budget is ₹50,000, allocated across hardware and cloud credits. " + "x" * 300 + " The report does not mention the currency."
    assert not is_abstention(answer)


@pytest.mark.parametrize("q", UNANSWERABLE, ids=lambda q: q["id"])
def test_dataset_terms_are_really_absent_from_the_corpus(q):
    for term in q["absent_terms"]:
        assert not re.search(r"\b" + re.escape(term.lower()), CORPUS), f"{q['id']}: '{term}' occurs in the corpus; the question may be answerable"


def test_dataset_shape():
    assert len(UNANSWERABLE) == 24 and len({q["id"] for q in UNANSWERABLE}) == 24
    assert {q["kind"] for q in UNANSWERABLE} == {"near", "far"}


def _u(i, score, abstained, kind="near"):
    return {"id": i, "kind": kind, "best_score": score, "abstained": abstained}


def _a(i, score, gold, abstained=False):
    return {"id": i, "kind": "answerable", "best_score": score, "abstained": abstained, "contains_gold": gold}


def test_sweep_prices_both_sides_of_the_gate():
    un = [_u("u1", -9.0, False, "far"), _u("u2", -4.5, False), _u("u3", 2.0, True), _u("u4", 3.0, False)]
    an = [_a("q1", 5.0, True), _a("q2", -6.0, False), _a("q3", -4.2, True), _a("q4", None, True)]
    rows = {r["threshold"]: r for r in sweep(un, an, thresholds=(-5.0, -4.0))}
    assert rows[-5.0]["unanswerable_stopped_by_gate"] == 1 and rows[-5.0]["unanswerable_declined_total"] == 2
    assert rows[-5.0]["answerable_wrongly_gated"] == 1 and rows[-5.0]["of_which_were_correct"] == 0, "q2 was wrong anyway"
    assert rows[-4.0]["unanswerable_stopped_by_gate"] == 2 and rows[-4.0]["unanswerable_declined_total"] == 3
    assert rows[-4.0]["answerable_wrongly_gated"] == 2 and rows[-4.0]["of_which_were_correct"] == 1, "q3 was correct: that is the cost"
    assert all(r["answerable_n"] == 4 for r in rows.values()), "a None score (reranker off) is never gated"


def test_summary_counts_gate_and_generator_and_only_wrong_answers_as_false_abstentions():
    un = [_u("u1", -9.0, False, "far"), _u("u2", 2.0, True), _u("u3", 3.0, False)]
    an = [_a("q1", 5.0, True, abstained=True),      # hedged but correct: not a false abstention
          _a("q2", 4.0, False, abstained=True),     # declined and wrong: false abstention
          _a("q3", -7.0, True)]                     # correct but gated: the gate overrides -> not counted (it contained gold)
    s = summarize(un, an, gate=-5.0)
    assert s["unanswerable"]["declined"] == 2 and s["unanswerable"]["abstention_rate"] == 0.667
    assert s["unanswerable"]["by_generator_alone"] == 1 and s["unanswerable"]["invented"] == ["u3"]
    assert s["unanswerable"]["by_kind"] == {"far": {"n": 1, "declined": 1}, "near": {"n": 2, "declined": 1}}
    assert s["answerable"]["false_abstention_ids"] == ["q2"]
    assert summarize(un, an, gate=None)["unanswerable"]["declined"] == 1


class _FakeCE:
    def __init__(self, scores): self.scores = scores
    def predict(self, pairs): return self.scores[: len(pairs)]


def _reranker(scores, min_score=-5.0):
    r = RerankerModel.__new__(RerankerModel)
    r.model, r.enabled, r.min_relevance_score, r.model_name = _FakeCE(scores), True, min_score, "fake"
    return r


def test_rerank_with_scores_orders_filters_and_keeps_scores():
    r = _reranker([0.5, 7.0, -9.0])
    scored = r.rerank_with_scores("q", ["a", "b", "c"], top_k=5)
    assert scored == [(7.0, "b"), (0.5, "a")], "c is below the floor"
    assert r.rerank("q", ["a", "b", "c"], top_k=5) == ["b", "a"], "rerank() is unchanged"


def test_rerank_with_scores_when_everything_is_weak_or_the_model_is_off():
    r = _reranker([-9.0, -7.0])
    assert r.rerank_with_scores("q", ["a", "b"], top_k=3) == [(-7.0, "b")], "top result kept, with its (low) score for the gate to read"
    r.enabled = False
    assert r.rerank_with_scores("q", ["a", "b"], top_k=1) == [(None, "a")], "pass-through: no score, so the gate cannot fire"
    assert r.rerank_with_scores("q", [], top_k=3) == []


def test_answer_floor_is_the_rerankers_own_floor_and_none_never_trips_it():
    assert KB_ANSWER_MIN_SCORE == -5.0 == RerankerModel.__init__.__defaults__[1], "the floor acts on the reranker's existing verdict"
    assert below_answer_floor(-11.1) and below_answer_floor(-5.01)
    assert not below_answer_floor(-5.0) and not below_answer_floor(-4.5) and not below_answer_floor(4.4)
    assert not below_answer_floor(None), "reranker off -> nothing was scored -> never decline on that basis"
    assert below_answer_floor(-1.0, floor=0.0) and not below_answer_floor(-7.0, floor=-8.0)


def test_orchestrator_acts_on_the_floor_before_generating():
    """Static guard: the decline must sit between reranking and generation, and mark the response."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "orchestrator", "master_llm.py"), encoding="utf-8").read()
    i_rerank = src.index("self.reranker.rerank_with_scores")
    i_floor = src.index("below_answer_floor(best_score)")
    i_gen = src.index("self.generator.generate_answer_stream(", i_floor)   # the knowledge-base path's own generation call
    assert i_rerank < i_floor < i_gen
    assert "self.generator.generate_answer_stream(" not in src[i_rerank:i_floor], "nothing generates between reranking and the floor"
    assert '"abstained": True' in src[i_floor:i_gen] and "return" in src[i_floor:i_gen]
