"""Idempotent setup of TraceMind's local Langfuse project resources."""

from datetime import UTC, datetime
import re

from tracemind.observability.metrics import PRICES, PRICE_VERSION, PRICE_SOURCE
from tracemind.observability.prompts import (
    INVESTIGATOR, VERIFIER, INVESTIGATOR_CONTRACT,
)

QUEUE_NAME = "TraceMind verifier calibration v1"
DASHBOARD_NAME = "TraceMind initial triage v1"
SCORE_CATEGORIES = {
    "tracemind.human.verdict": [
        "supported", "insufficient", "contradicted", "unknown",
    ],
    "tracemind.human.severity": ["none", "minor", "material", "unknown"],
    "tracemind.human.usefulness": ["useful", "limited", "misleading", "unknown"],
    "tracemind.human.verifier": ["reasonable", "too_strict", "missed", "unknown"],
}


def pages(method, *args, **kwargs):
    page = 1
    while True:
        result = method(*args, page=page, limit=100, **kwargs)
        yield from result.data
        if page >= result.meta.total_pages:
            return
        page += 1


def bootstrap(client):
    from tracemind.agent.runtime.runner import SYSTEM_PROMPT
    from tracemind.agent.verification import RUBRIC, VERSION

    prompts = []
    for name, content, contract in (
        (INVESTIGATOR, SYSTEM_PROMPT, INVESTIGATOR_CONTRACT),
        (VERIFIER, RUBRIC, VERSION),
    ):
        existing = list(pages(client.api.prompts.list, name=name))
        if not any(p.name == name for p in existing):
            prompt = client.create_prompt(
                name=name, prompt=content, type="text", labels=["production"],
                config={"contract_version": contract},
                commit_message="Import approved bundled prompt unchanged",
            )
        else:
            prompt = client.get_prompt(name, label="production", cache_ttl_seconds=0)
        prompts.append({"name": name, "version": prompt.version})

    configs = {c.name: c for c in pages(client.api.score_configs.get)}
    for name, labels in SCORE_CATEGORIES.items():
        if name not in configs:
            configs[name] = client.api.score_configs.create(
                name=name, data_type="CATEGORICAL",
                categories=[{"label": label, "value": i}
                            for i, label in enumerate(labels)],
                description=("Human assessment of the exact review candidate. "
                             "unknown is not a pass or failure. Use comments "
                             "to identify disputed item IDs and evidence."),
            )
    queues = {q.name: q for q in pages(client.api.annotation_queues.list_queues)}
    queue = queues.get(QUEUE_NAME) or client.api.annotation_queues.create_queue(
        name=QUEUE_NAME, score_config_ids=[configs[n].id for n in SCORE_CATEGORIES],
        description=("Review the candidate and frozen evidence independently. "
                     "Keep factual support separate from usefulness. "
                     "Complete the queue item after scoring; no model prefill."),
    )

    models = {m.model_name for m in pages(client.api.models.list)}
    for model, (hit, miss, output) in PRICES.items():
        name = model + " / " + PRICE_VERSION
        if name not in models:
            client.api.models.create(
                model_name=name, match_pattern="^" + re.escape(model) + "$",
                unit="TOKENS", start_date=datetime(2026, 11, 12, tzinfo=UTC),
                pricing_tiers=[{
                    "name": PRICE_VERSION, "is_default": True,
                    "priority": 0, "conditions": [],
                    "prices": {
                        "input": miss / 1_000_000,
                        "input_cache_read": hit / 1_000_000,
                        "output": output / 1_000_000,
                        "output_reasoning": output / 1_000_000,
                    },
                }],
            )

    dashboards = {d.name: d for d in pages(client.api.unstable.dashboards.list)}
    dashboard = dashboards.get(DASHBOARD_NAME)
    if dashboard is None:
        dashboard = client.api.unstable.dashboards.create(
            name=DASHBOARD_NAME,
            filters=[{"column": "metadata", "type": "stringObject",
                      "key": "run_kind", "operator": "=", "value": "investigation"}],
            description=("Logical run scores; primary attempts and retries remain "
                         "separate. Wall time includes interruptions. Costs are "
                         "peak-list estimates, not invoices. Check coverage. "
                         + PRICE_SOURCE),
        )
    widgets = {w.name: w for w in pages(client.api.unstable.dashboard_widgets.list)}
    current = client.api.unstable.dashboards.get(dashboard.id)
    placed = {p.widget_id for p in current.definition.widgets
              if p.type == "widget"}
    specs = [
        ("Run count", "run_count", "sum", "NUMBER"),
        ("First triage coverage", "has_triage", "avg", "NUMBER"),
        ("First triage P50 ms", "first_triage_ms", "p50", "LINE_TIME_SERIES"),
        ("First triage P95 ms", "first_triage_ms", "p95", "LINE_TIME_SERIES"),
        ("Final P50 ms", "final_ms", "p50", "LINE_TIME_SERIES"),
        ("Final P95 ms", "final_ms", "p95", "LINE_TIME_SERIES"),
        ("Execution failure rate", "execution_failed", "avg", "LINE_TIME_SERIES"),
        ("Review completion rate", "review_completed", "avg", "LINE_TIME_SERIES"),
        ("First review objection rate", "first_review_rejected", "avg", "LINE_TIME_SERIES"),
        ("Final review objection rate", "final_review_rejected", "avg", "LINE_TIME_SERIES"),
        ("Reported tokens", "reported_tokens", "sum", "LINE_TIME_SERIES"),
        ("Usage coverage", "usage_coverage", "avg", "LINE_TIME_SERIES"),
        ("Known model cost estimate USD", "known_cost_estimate_usd", "sum", "LINE_TIME_SERIES"),
        ("Cost coverage", "cost_coverage", "avg", "LINE_TIME_SERIES"),
        ("Unpriced model calls", "unpriced_calls", "sum", "NUMBER"),
        ("Verifier P95 ms", "verification_ms", "p95", "LINE_TIME_SERIES"),
        ("Revision P95 ms", "revision_ms", "p95", "LINE_TIME_SERIES"),
    ]
    for index, (title, metric, agg, chart) in enumerate(specs):
        name = "TraceMind / " + title
        widget = widgets.get(name)
        if widget is None:
            widget = client.api.unstable.dashboard_widgets.create(
                name=name, view="scores-numeric", dimensions=[],
                metrics=[{"measure": "value", "agg": agg}],
                filters=[{"column": "name", "type": "string", "operator": "=",
                          "value": "tracemind." + metric}],
                chart_type=chart,
                description="Missing scores are unknown; always inspect sample count/coverage.",
            )
        if widget.id not in placed:
            client.api.unstable.dashboards.add_placement(
                dashboard.id, request={"type": "widget", "widget_id": widget.id,
                                       "x": (index % 2) * 6,
                                       "y": (index // 2) * 6,
                                       "width": 6, "height": 6},
            )
    return {"prompts": prompts, "queue_id": queue.id,
            "dashboard_id": dashboard.id, "price_version": PRICE_VERSION,
            "score_configs": {n: configs[n].id for n in SCORE_CATEGORIES}}
