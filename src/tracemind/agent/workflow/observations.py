"""Apply declared investigation scope without promoting caller hints."""

from datetime import datetime

from tracemind.models import Evidence


SCOPE_ATTRIBUTES = {
    "environment": "deployment.environment.name",
    "cluster": "k8s.cluster.name",
    "namespace": "k8s.namespace.name",
}


def scope_status(item: Evidence, scope) -> str:
    """Unknown resource identity is not a match to an explicit scope."""
    if scope is None:
        return "matched"
    resource = item.attributes.get("resource", {})
    metadata = item.attributes.get("resource_metadata", {})
    resource = resource if isinstance(resource, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    conflicts = metadata.get("conflicts", {})
    conflicts = conflicts if isinstance(conflicts, dict) else {}
    incomplete = set(metadata.get("redacted_fields") or ()) | set(
        metadata.get("truncated_fields") or (),
    )
    unknown = False
    for field, attribute in SCOPE_ATTRIBUTES.items():
        expected = getattr(scope, field)
        if expected is None:
            continue
        actual = resource.get(attribute)
        if actual is None or attribute in conflicts or attribute in incomplete:
            unknown = True
        elif actual != expected:
            return "scope_mismatch"
    return "scope_unknown" if unknown else "matched"


def mark_exception_window(item: Evidence, start, end) -> Evidence:
    """Keep exception occurrence time distinct from its enclosing span."""
    exceptions = item.attributes.get("exceptions")
    if not isinstance(exceptions, list):
        return item
    result = item.model_copy(deep=True)
    for event in result.attributes["exceptions"]:
        if not isinstance(event, dict):
            continue
        timestamp = event.get("timestamp")
        in_window = None
        if isinstance(timestamp, str):
            try:
                at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if at.utcoffset() is not None:
                    in_window = start <= at <= end
            except ValueError:
                pass  # The original timestamp remains available for review.
        event["in_investigation_window"] = in_window
    return result
