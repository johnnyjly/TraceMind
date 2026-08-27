"""Model-facing tool guidance owned by the telemetry MCP service."""

DESCRIPTIONS = {
    'query_metrics': (
        'Query an instant vector at the investigation end '
        'time.\n\n        Use when sample values are needed to '
        'check a hypothesis. Reuse known\n        names and '
        'labels; discover_metrics is optional if they are '
        'confirmed.\n        Skip this tool when existing '
        'evidence already answers the question.\n        '
        'Returns normalized observations with evidence IDs, '
        'or empty-query\n        guidance without evidence '
        'IDs. An empty result does not establish\n        '
        'health: check names, labels and sample availability '
        'before retrying.\n\n        Conditional example: ONLY '
        'if discovery returned request_count_total\n        '
        'and the label job="orders", query_metrics with '
        'expression\n        '
        '\'request_count_total{job="orders"}\' can inspect '
        'those samples.\n        If names or labels are '
        'unknown, discover them instead of guessing.'
    ),
    'discover_metrics': (
        'Discover metric names and label examples before '
        'writing PromQL.\n\n        Use when relevant names or '
        'labels are unknown, including after an\n        empty'
        ' query with suspected mismatched names/labels. Reuse'
        ' previously\n        discovered metadata instead of '
        'rediscovering it for every query.\n        Skip '
        'discovery when names/labels are known or evidence is'
        ' sufficient.\n        The application fixes the '
        'investigation window; do not supply dates.\n\n        '
        'Returns metrics containing metric, label_keys and '
        'example_labels.\n        Empty example_labels means '
        'no examples were included, not that the\n        '
        'metric has no labels; pass its exact returned name '
        'as metric to inspect\n        it. possibly_truncated,'
        ' omitted_metrics and labels_truncated mark\n        '
        'incomplete coverage. Empty discovery may mean '
        'service labels differ;\n        bounded global '
        'discovery can reveal them. Absence never proves '
        'health.\n        Metadata has no evidence IDs or '
        'sample values and cannot be cited.\n\n        '
        'Examples: If the incident names orders and metric '
        'names are unknown,\n        '
        'discover_metrics(service="orders") finds candidates '
        'to query.\n        If retained evidence already '
        "supports a user's explanation request,\n        "
        'answer directly without discovery or another metric '
        'query.'
    ),
    'search_traces': (
        'Find recent Jaeger traces for one service in the '
        'incident window.'
    ),
    'get_trace_detail': (
        'Inspect a known trace: return error spans plus up to 20 '
        'longest successful spans, including service, operation, '
        'duration_ms, is_error and parent_span_id when available. '
        'Use successful spans to investigate latency even when there '
        'are no errors. Parent and child durations overlap; do not '
        'sum them. Results are bounded, not a complete span tree. '
        'TraceMind filters observations to its investigation window. '
        '\n\n        Use its returned '
        "attributes.trace_id, not the evidence's id. If no"
        '\n        trace ID is available, search_traces or '
        'search_logs first. Do not\n        reconstruct a '
        'trace ID from an abbreviated value.'
    ),
    'search_logs': (
        'Find OpenSearch logs for one service, optionally '
        'matching text. Results are a bounded sample; '
        'absence of a message does not prove absence of a fault. '
        'Event time is preferred; when it is absent '
        'or zero, use observation time, marked by '
        'timestamp_source=observedTimestamp in evidence. Observation '
        'time is not proof of the exact event occurrence time.'
    ),
}

PARAMETER_DESCRIPTIONS = {
    'query_metrics': {
        'expression': (
            'PromQL expression returning an instant vector, using'
            ' confirmed metric names and label values. Not a bare'
            ' range vector or scalar. Query time is fixed to the '
            'investigation end by the application.'
        ),
    },
    'discover_metrics': {
        'service': (
            'Exact service value matched against service_name, '
            'service or job labels, not a substring or regex. '
            'Omit or use null for a bounded global catalogue when'
            ' these labels are unknown.'
        ),
        'metric': (
            'One exact metric name returned by discovery, not '
            "PromQL or a regex. Set it to inspect that metric's "
            'label examples; omit or use null to discover '
            'distinct metric names first.'
        ),
        'limit': (
            'Maximum distinct names and maximum series inspected '
            'per metadata request (1-50, default 20), not an '
            'evidence count or time range. Results may be '
            'incomplete even at this limit.'
        ),
    },
    'search_traces': {
    },
    'get_trace_detail': {
        'trace_id': (
            'Copy the exact attributes.trace_id from returned '
            'trace or log evidence: a 16 or 32 character '
            'hexadecimal Jaeger ID. Not an evidence UUID or '
            'span_id. Never invent or pad an ID.'
        ),
    },
    'search_logs': {
        'service': (
            'Exact telemetry service name. Not an operation name or '
            'a list of services.'
        ),
        'text': (
            'Optional log-body search terms. Omit or use null to apply '
            'no body-text filter. Not a trace ID filter or a regex.'
        ),
    },
}
