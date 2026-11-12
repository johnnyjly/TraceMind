"""Apply declared investigation scope without promoting caller hints."""

from datetime import datetime, timedelta
import math

from tracemind.models import Evidence, EvidenceSource


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


def mark_observation_window(item: Evidence, start, end) -> Evidence:
    """Derive window context without assigning a time to final span status."""
    result = item.model_copy(deep=True)
    attributes = result.attributes
    # Recompute on follow-up windows; backend-supplied derived tags are not
    # authoritative. observed_at is only a start when provenance says so.
    attributes.pop("span_window", None)
    if (item.source == EvidenceSource.TRACE
            and attributes.get("timestamp_source") == "span_startTime"):
        duration = attributes.get("duration_ms")
        finished = None
        if (type(duration) in (int, float) and duration >= 0
                and item.observed_at.utcoffset() is not None):
            try:
                if math.isfinite(duration):
                    finished = item.observed_at + timedelta(
                        milliseconds=duration,
                    )
            except OverflowError:
                pass  # Invalid remote duration cannot establish completion.
        attributes["span_window"] = {
            "end": finished.isoformat() if finished is not None else None,
            "end_in_investigation_window": (
                start <= finished <= end if finished is not None else None
            ),
        }
    exceptions = attributes.get("exceptions")
    if not isinstance(exceptions, list):
        return result
    for event in exceptions:
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
