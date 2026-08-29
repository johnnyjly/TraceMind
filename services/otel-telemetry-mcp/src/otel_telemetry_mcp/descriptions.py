"""Model-facing tool guidance owned by the telemetry MCP service."""

DESCRIPTIONS = {
    'query_metrics': (
        'Query an instant vector at the supplied at timestamp. '
        'Use when sample values are needed to check a hypothesis. Reuse '
        'confirmed metric names and labels; discover_metrics, if available, '
        'can find unknown names and labels. Skip queries when existing '
        'evidence already answers the question.\n\n'
        'Returns normalized observations with evidence IDs, or an empty '
        'evidence list when no samples match. An empty result does not '
        'establish health: check names, labels and sample availability '
        'before retrying. Do not guess names or label variants.\n\n'
        'Conditional example: ONLY if request_count_total and the label '
        'job="orders" are confirmed, the expression '
        '\'request_count_total{job="orders"}\' can inspect those samples.'
    ),
    'discover_metrics': (
        'Discover metric names and label examples in the supplied start/end '
        'window before writing PromQL. Use when relevant names or labels '
        'are unknown, including after an empty query with suspected '
        'mismatches. Reuse previously discovered metadata; skip discovery '
        'when names and labels are known or evidence is sufficient.\n\n'
        'Returns metrics containing metric, label_keys and example_labels. '
        'Empty example_labels means no examples were included, not that '
        'the metric has no labels; pass its exact returned name as metric '
        'to inspect it. possibly_truncated, omitted_metrics and '
        'labels_truncated mark incomplete coverage. Empty discovery may '
        'mean service labels differ; bounded global discovery can reveal '
        'them. Absence never proves health. Metadata has no evidence IDs '
        'or sample values and cannot be cited.\n\n'
        'For example, set service="orders" to find candidates for that '
        'service in the supplied window. If retained evidence already '
        'supports an explanation request, answer without another query.'
    ),
    'search_traces': (
        'Find recent Jaeger traces for one service in the supplied '
        'start/end window.'
    ),
    'get_trace_detail': (
        'Inspect a known trace: return error spans plus up to 20 longest '
        'successful spans, including service, operation, duration_ms, '
        'is_error and parent_span_id when available. Use successful spans '
        'to investigate latency even when there are no errors. Parent and '
        'child durations overlap; do not sum them. Results are bounded, '
        'not a complete span tree. This tool does not filter by time.\n\n'
        'Use the exact attributes.trace_id from trace or log evidence, '
        "not the evidence's id. If no trace ID is available, search_traces "
        'or search_logs may supply one when those tools are available. '
        'Do not invent a trace ID or reconstruct an abbreviated value.'
    ),
    'search_logs': (
        'Find OpenSearch logs for one service in the supplied start/end '
        'window, optionally matching text. Results are a bounded sample; '
        'absence of a message does not prove absence of a fault. Event time '
        'is preferred; when it is absent or zero, observation time is used, '
        'marked by timestamp_source=observedTimestamp in evidence. '
        'Observation time is not proof of the exact event occurrence time.'
    ),
}

PARAMETER_DESCRIPTIONS = {
    'query_metrics': {
        'expression': (
            'PromQL expression returning an instant vector, using confirmed '
            'metric names and label values. Not a bare range vector or scalar.'
        ),
        'at': 'Timezone-aware timestamp at which to evaluate the expression.',
    },
    'discover_metrics': {
        'service': (
            'Exact service value matched against service_name, service or '
            'job labels, not a substring or regex. Omit or use null for a '
            'bounded global catalogue when these labels are unknown.'
        ),
        'metric': (
            'One exact metric name returned by discovery, not PromQL or a '
            "regex. Set it to inspect that metric's label examples; omit "
            'or use null to discover distinct metric names first.'
        ),
        'limit': (
            'Maximum distinct names and maximum series inspected per '
            'metadata request (1-50, default 20), not an evidence count or '
            'time range. Results may be incomplete even at this limit.'
        ),
    },
    'search_traces': {},
    'get_trace_detail': {
        'trace_id': (
            'Copy the exact attributes.trace_id from returned trace or log '
            'evidence: a 16 or 32 character hexadecimal Jaeger ID. Not an '
            'evidence UUID or span_id. Never invent or pad an ID.'
        ),
    },
    'search_logs': {
        'service': (
            'Exact telemetry service name. Not an operation name or a '
            'list of services.'
        ),
        'text': (
            'Optional log-body search terms. Omit or use null to apply no '
            'body-text filter. Not a trace ID filter or a regex.'
        ),
    },
}
