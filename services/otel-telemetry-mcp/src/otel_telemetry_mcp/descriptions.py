"""Model-facing tool guidance owned by the telemetry MCP service."""

DESCRIPTIONS = {
    "query_metrics": (
        "Query an instant vector at the supplied at timestamp. "
        "Use when sample values are needed to check a hypothesis. Reuse "
        "confirmed metric names and labels; discover_metrics, if available, "
        "can find unknown names and labels. Skip queries when existing "
        "evidence already answers the question.\n\n"
        "Returns normalized observations with evidence IDs, or an empty "
        "evidence list when no samples match. An empty result does not "
        "establish health: check names, labels and sample availability "
        "before retrying. Do not guess names or label variants.\n\n"
        "Counters and histogram buckets are cumulative, not counts within "
        "the incident window. For a windowed claim, use rate or increase "
        "with a range inside that window and enough samples. Buckets are "
        "cumulative across le boundaries too. Compare matching labels, "
        "units and cohorts; do not infer a specific connection failure "
        "from unrelated memory or traffic counters.\n\n"
        "Conditional example: ONLY if request_count_total and the label "
        'job="orders" are confirmed, the expression '
        "'request_count_total{job=\"orders\"}' can inspect those samples."
    ),
    "discover_metrics": (
        "Discover metric names and label examples in the supplied start/end "
        "window before writing PromQL. Use when relevant names or labels "
        "are unknown, including after an empty query with suspected "
        "mismatches. Reuse previously discovered metadata; skip discovery "
        "when names and labels are known or evidence is sufficient.\n\n"
        "Returns metrics containing metric, label_keys and example_labels. "
        "Empty example_labels means no examples were included, not that "
        "the metric has no labels; pass its exact returned name as metric "
        "to inspect it. possibly_truncated, omitted_metrics and "
        "labels_truncated mark incomplete coverage. Empty discovery may "
        "mean service labels differ; bounded global discovery can reveal "
        "them. Absence never proves health. Metadata has no evidence IDs "
        "or sample values and cannot be cited.\n\n"
        'For example, set service="orders" to find candidates for that '
        "service in the supplied window. If retained evidence already "
        "supports an explanation request, answer without another query."
    ),
    "search_traces": (
        "Find recent Jaeger traces for one service in the supplied "
        "start/end window. Copy an observed telemetry service name; do not "
        "derive it from an RPC class name or guess spelling variants. "
        "Returned attributes.services lists actual service names from the "
        "in-window spans of that trace (possibly incomplete). Use those "
        "names for downstream queries. Inspect a promising trace before "
        "searching every service; repeated trace details may add no evidence. "
        "Use min_duration_ms for slow spans or errors_only for error-tagged "
        "spans; filters run before the backend result limit. Empty searches "
        "return bounded global service-name hints without evidence IDs. "
        "Those names do not establish activity in the incident window. "
        "Summary observed_at is the earliest selected span start; "
        "duration_ms is their overall time extent, not a completion time. "
        "This is a bounded recent sample, not the slowest or failing traces "
        "across the whole window. If frontend samples do not show the "
        "reported fault, search the suspected downstream service using its "
        "observed name. For latency, inspect slow downstream spans before "
        "spending the budget on broad metric discovery. A normal recent "
        "sample cannot exclude intermittent failures."
    ),
    "get_trace_detail": (
        "Inspect a known trace: page all spans, errors first then duration. "
        "Filter by exact service/operation before pagination; fast normal "
        "controls remain accessible. Each span includes coverage with "
        "matched_span_count, page_offset, next_offset and trace_fingerprint. "
        "Use next_offset to continue with identical filters; a changed "
        "fingerprint means the normalized span observations changed "
        "between reads; volatile backend warnings are excluded. "
        "Return service, operation, duration_ms, diagnostic attributes, "
        "is_error and parent_span_id when available. observed_at is the "
        "span start time (timestamp_source=span_startTime), not collection "
        "time; its end is observed_at plus duration_ms. Cross-service clock "
        "skew can affect timestamp ordering. Use successful spans "
        "to investigate latency even when there are no errors. Parent and "
        "child durations overlap; do not sum them. Results are bounded, "
        "not a complete span tree unless coverage establishes it. "
        "This tool does not filter by time.\n\n"
        "Use the exact attributes.trace_id from trace or log evidence, "
        "not the evidence's id. If no trace ID is available, search_traces "
        "or search_logs may supply one when those tools are available. "
        "Do not invent a trace ID or reconstruct an abbreviated value."
    ),
    "search_logs": (
        "Find OpenSearch logs for one service in the supplied start/end "
        "window, optionally matching text. Results are a bounded sample; "
        "use the exact observed telemetry service name. If a filtered "
        "search is empty or fails to explain the observed operation, "
        "inspect one unfiltered sample for that service in the same "
        "window, unless already inspected. Do not cycle through guessed "
        "service names or assume the cause must appear at error severity. "
        "absence of a message does not prove absence of a fault. Event time "
        "is preferred; when it is absent or zero, observation time is used, "
        "marked by timestamp_source=observedTimestamp in evidence. "
        "Observation time is not proof of the exact event occurrence time."
    ),
}

PARAMETER_DESCRIPTIONS = {
    "query_metrics": {
        "expression": (
            "PromQL expression returning an instant vector, using confirmed "
            "metric names and label values. Not a bare range vector or scalar."
        ),
        "at": "Timezone-aware timestamp at which to evaluate the expression.",
    },
    "discover_metrics": {
        "service": (
            "Exact service value matched against service_name, service or "
            "job labels, not a substring or regex. Omit or use null for a "
            "bounded global catalogue when these labels are unknown."
        ),
        "metric": (
            "One exact metric name returned by discovery, not PromQL or a "
            "regex. Set it to inspect that metric's label examples; omit "
            "or use null to discover distinct metric names first."
        ),
        "limit": (
            "Maximum distinct names and maximum series inspected per "
            "metadata request (1-50, default 20), not an evidence count or "
            "time range. Results may be incomplete even at this limit."
        ),
    },
    "search_traces": {
        "operation": "Optional exact operation name, filtered in Jaeger "
        "before "
        "sampling.",
        "min_duration_ms": (
            "Optional positive minimum matching span duration in "
            "milliseconds. Not whole-trace duration; omit or use null for "
            "no duration filter."
        ),
        "errors_only": (
            "Match indexed error=true span tags when true. Does not cover "
            "failures without this tag. Combined with duration using AND."
        ),
    },
    "get_trace_detail": {
        "service": "Optional exact observed service, applied before paging.",
        "operation": (
            "Optional exact observed operation, applied before paging."
        ),
        "offset": "Zero-based page offset; use coverage.next_offset to "
        "continue.",
        "limit": "Spans per page, 1-100; default 25.",
        "trace_id": (
            "Copy the exact attributes.trace_id from returned trace or log "
            "evidence: a 16 or 32 character hexadecimal Jaeger ID. Not an "
            "evidence UUID or span_id. Never invent or pad an ID."
        ),
    },
    "search_logs": {
        "trace_id": "Optional exact trace ID to correlate logs within this "
        "window.",
        "service": (
            "Exact telemetry service name. Not an operation name or a "
            "list of services."
        ),
        "text": (
            "Optional log-body search terms. Omit or use null to apply no "
            "body-text filter. Use trace_id for exact trace "
            "correlation, not text."
        ),
    },
}
