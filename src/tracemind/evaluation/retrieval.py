"""Evaluate document retrieval without passing relevance labels to queries."""

from time import monotonic

from pydantic import BaseModel, Field, model_validator


class RetrievalQuestion(BaseModel):
    """Freeze a question and externally judged document relevance grades."""

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant: dict[str, int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_labels(self):
        """Require nonnegative grades and at least one relevant document."""
        if min(self.relevant.values()) < 0 or max(self.relevant.values()) < 1:
            raise ValueError(
                "relevance needs nonnegative grades and a positive")
        return self


class RetrievalSuite(BaseModel):
    """Bind judgments to the exact published snapshot being evaluated."""

    snapshot: str
    questions: list[RetrievalQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        """Prevent one question from overwriting another's measurements."""
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate question IDs")
        return self


async def evaluate_retrieval(
        runtime, suite: RetrievalSuite, *, k: int) -> dict:
    """Measure the real returned ranking with ir-measures, including misses.

    Multiple child hits of a document count once at their first appearance.
    Zero-hit questions remain in the denominator. Runtime failures propagate
    rather than being recorded as legitimate empty search results.
    """
    import ir_measures
    from ir_measures import R, RR, nDCG

    if suite.snapshot != runtime.settings.snapshot:
        raise ValueError("evaluation snapshot mismatch")
    if k < 1 or k > runtime.settings.result_limit:
        raise ValueError("evaluation k must fit the returned result limit")
    measures = [R @ k, RR @ k, nDCG @ k]
    qrels = {q.id: q.relevant for q in suite.questions}
    rankings, records = {}, []
    for question in suite.questions:
        started = monotonic()
        hits = await runtime.search(question.query)
        documents = list(dict.fromkeys(hit.chunk.doc_id for hit in hits))
        rankings[question.id] = {
            doc: float(len(documents) - position)
            for position, doc in enumerate(documents)
        }
        records.append({
            "id": question.id, "documents": documents,
            "chunks": [hit.chunk.chunk_id for hit in hits],
            "duration_ms": (monotonic() - started) * 1000,
            "model_calls": runtime.model.calls,
            "input_tokens": runtime.model.input_tokens,
            "output_tokens": runtime.model.output_tokens,
            "unreported_model_calls": runtime.model.unreported_calls,
        })
    return {
        "snapshot": suite.snapshot, "unit": "document", "k": k,
        "mode": runtime.settings.mode,
        "metrics": {str(m): v for m, v in ir_measures.calc_aggregate(
            measures, qrels, rankings).items()},
        "per_query": [{"id": m.query_id, "metric": str(m.measure),
                       "value": m.value}
                      for m in ir_measures.iter_calc(
                          measures, qrels, rankings)],
        "queries": records,
    }
