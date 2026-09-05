# Fixed-evidence claim checks

These checks call the configured real model on sanitized regression fixtures
derived from public Demo behavior. Identifiers, timestamps and timings are
illustrative; the fixtures contain no captured local run records.
They do not query live telemetry or test autonomous query selection. The script
does not send the fixture's provenance or assessment rubric to the model.

```bash
uv run --extra rag python scripts/evaluate_claim_boundaries.py \
  --fixture evals/claim-boundaries/cart-operation-scope.json \
  --output artifacts/evaluations/cart-claim-boundaries.json
```

Read the saved diagnosis against each rubric item and record the assessment.
Citation validation and the ordinary case keyword scorer cannot establish
semantic correctness. A scripted model following the desired answer would not
validate this behavior; use actual model output and keep failures visible.
