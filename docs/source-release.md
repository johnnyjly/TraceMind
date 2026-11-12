# Direct source and release access

This optional single-agent feature connects retained runtime observations to a
trusted local release snapshot, then reads immutable Git objects. It adds code
and release context to initial triage; it does not prove release causation or
production readiness. RAG remains optional reference material.

## Enable explicitly

Configure both the catalog and tool grants on the API and worker:

```dotenv
TRACEMIND_AGENT_MODE=single
TRACEMIND_AGENT_SOURCE_CATALOG=/absolute/path/releases.json
TRACEMIND_AGENT_ALLOWED_TOOLS=["query_metrics","search_traces","get_trace_detail","search_logs","read_evidence","discover_metrics","resolve_deployment","read_source","get_release_diff","read_source_reference"]
```

The default catalog is `none`, and default tool grants omit all four source
tools. A catalog alone does not grant access. Multi-agent execution rejects a
source policy. No provider, dependency or model-call budget changes are required.

The catalog is loaded at admission, validated, and saved in the run. Recovery
uses that snapshot, not the latest file. Repository paths are operator-supplied
allowlist entries; the agent cannot clone a repository, select an arbitrary
commit, fetch a URL, execute project code or modify a deployment.

## Catalog contract

Use `SourcePolicy` in `src/tracemind/models/sources.py` as the authoritative
schema. The following values are illustrative, not actual build evidence:

```json
{
  "schema_version": 1,
  "repositories": [{
    "id": "checkout",
    "path": "/srv/source/checkout",
    "service": "checkout",
    "directory": "src/checkout",
    "runtime_paths": {"/app": "src/checkout"}
  }],
  "releases": [{
    "id": "checkout-release-42",
    "service": "checkout",
    "repository_id": "checkout",
    "commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "version": "v42",
    "scope": {"environment": "production"},
    "deployed_at": "2026-01-01T10:00:00Z",
    "instances": ["checkout-pod-42"],
    "image_id": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "provenance": "Operator export of CI build and deployment capture",
    "captured_at": "2026-01-01T10:01:00Z",
    "build": {
      "record_id": "build-42",
      "producer": "ci-build-capture",
      "captured_at": "2026-01-01T09:55:00Z",
      "source_tree": "cccccccccccccccccccccccccccccccccccccccc",
      "image_id": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    }
  }]
}
```

Paths may be relative to the catalog file. `directory` and each mapped target
are repository-relative and restrict source reads to the service directory.
Absolute stack paths need an explicit `runtime_paths` mapping. Reads use the
recorded commit even if the worktree changes. Symlinks, submodules, binary files,
path traversal and missing Git objects produce explicit unavailable results.

A release can also declare `containers`, `image_digest`, `ended_at`, and
`previous_release_id`. The predecessor must be an earlier deployment of the
same service, repository and scope, and must be present in the snapshot.
Diffs compare this predecessor with the selected deployment, never `HEAD` or
an inferred latest tag. A selected diff path must exist in at least one revision;
an absent path is unavailable, never evidence of an unchanged file. Prefer the
original absolute path from the stack so configured mapping can resolve it.
A local image ID and registry repository digest are
different identities and must not be substituted for each other.

Build capture should record the clean source tree actually supplied to the
build, the resulting image, and the running instance/image association. OCI
labels, a service version string, a dirty checkout's HEAD, or an incident
reporter's deployment hint alone do not establish that association.

## Mapping and evidence meaning

`resolve_deployment` accepts an existing E handle. It checks the observation's
service, occurrence time, resource scope and available instance/container/image
identity against the frozen records. An explicitly requested scope must also
be established by the runtime resource; missing identity is not a match.

| Status | Meaning |
| --- | --- |
| `matched` | One release has matching runtime identity and a build capture whose source tree agrees with the recorded commit. This trusts the operator capture; it is not cryptographic attestation. |
| `declared` | A unique recorded mapping exists, but the runtime/build association is not established. Code can be inspected with this qualification. |
| `missing` | Required identity, scope or a candidate mapping is missing. |
| `ambiguous` | More than one release remains possible, including overlapping deployments. |
| `mismatch` | Available identities or the captured source tree disagree. |
| `unavailable` | The recorded commit cannot be read locally. |

Only `matched` and `declared` permit new file/diff reads. Unresolved mappings
never fall back to a newer release or HEAD. An investigating follow-up must
resolve its retained runtime evidence again, even for the same window. An
explanation reads saved references without opening Git or the catalog. If a
follow-up's observation budget cannot retain its prior sources, creation is
rejected rather than allowing a worker capacity failure.

References have separate roles:

- **E**: runtime telemetry used to establish what happened.
- **R**: direct deployment, source excerpt or diff, with immutable commit,
  snapshot identity, path/line or base/head metadata and runtime E linkage.
- **K**: retrieved documentation or runbook knowledge.

Models cite R handles in `source_refs`. Saved records use their content-derived
identities. `read_source_reference` pages the retained content. The initial
triage API includes `sources`; persisted candidates, items and review results
retain their R citations. The verifier classifies a code or deployment-record
fact as `source_fact`; it requires R support. A supported runtime diagnostic
still requires E. A source fact alone cannot meet `actionable_finding`, and a
diff or release timestamp alone cannot establish a causal regression.

## Bounds and data handling

The source tools share the run's overall tool allowance and have a separate
maximum of six calls. File excerpts have at most 200 lines; each retained source
response is bounded to 32 KiB including metadata, and all retained R records to
256 KiB. R storage also consumes the existing observation byte budget alongside
E. Large Git objects/output are rejected above 1 MiB. Catalog input is limited
to 1 MiB. Diff inspection is also limited to 32 changed regular files and
1 MiB of combined base/head blob content. A diff involving a file with known
credential text is unavailable, since a patch hunk can omit credential markers.
Bounds may be lowered in the snapshot.

Known credential text forms are redacted and truncation/redaction are visible.
This is not complete secret or PII detection. Allowlist repositories and service
directories appropriate for the configured model provider. Source comments,
release notes and all retrieved contents are treated as data, never instructions.

Initial triage remains available before review completes. Missing source access
is an evidence gap; the agent can still describe the observed error and propose
a precise follow-up. Source tools add no helper model. The single-agent review
workflow can request one revision and targeted source/telemetry queries, followed
by a full second review within the same budgets and deadline.

## Controlled live acceptance

`evals/release_lab/app.js` and `scripts/release_source_lab.py` provide a small
HTTP service with two actual Git revisions and Docker builds. The second
revision removes the default currency. Missing currency succeeds on v1 and
throws on v2; an object-valued currency throws on both revisions, providing a
counterexample to attributing every post-release error to the diff.

The lab reuses an already available Node image and the local Demo collector and
network. It creates only dedicated containers, captures their image identities,
submits real HTTP requests, retains normalized telemetry, checks whole-trace
fingerprints through MCP, then removes those containers. It does not change
Demo flags. Build archives, captures and every primary model attempt remain.

Start a temporary MCP using the current code (choose a free port):

```bash
PYTHONPATH=services/otel-telemetry-mcp/src TELEMETRY_MCP_PORT=8019 \
  uv run python -m otel_telemetry_mcp.server
```

Capture before spending model calls:

```bash
uv run python scripts/release_source_lab.py \
  --output artifacts/evaluations/source-release-example --capture-only \
  --mcp-url http://127.0.0.1:8019/mcp
```

After authorizing data transmission, run the five primary conditions:

```bash
uv run python scripts/release_source_lab.py \
  --output artifacts/evaluations/source-release-example --run-captured \
  --mcp-url http://127.0.0.1:8019/mcp
```

The lab reads the same repository-root `.env` as API and worker. Optional
`--env-file` arguments overlay it; `--mcp-url` overrides just the endpoint used
for capture and evaluation. The existing `--credential-file` option remains an
explicit key-only override for older invocations, but is no longer required.
Full settings/environment and credentials are never saved. Cases cover a concrete exception, two-version canary, the same canary
without source access, missing mapping, and the counterexample. API admission,
persistent worker, early triage and review all run. Measurements of first
readability use local ASGI API polling, not network SSE delivery latency.

Source acceptance requires single-agent mode and rejects multi-agent settings
before telemetry or model calls. Each condition runs once with the effective
local model, verifier and budget
configuration. Source mapping, scenario tool grants and output directories are
the per-case differences. Configure fallback, RAG and memory explicitly for the
comparison; they are no longer silently reset by the lab. The saved protocol
records effective limits and enabled features, and failures plus first-round
verdicts remain in the ledger. Earlier artifacts retain their original frozen profiles. This small synthetic comparison can
show whether direct source evidence added a useful explanation; it cannot
establish production accuracy, human time savings, general speed superiority
or causal uplift over a deterministic baseline.
