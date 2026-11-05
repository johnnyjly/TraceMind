"""Bounded source context without exporting arbitrary attributes."""

from datetime import UTC, datetime
import json
import re


MAX_EXCEPTION_BYTES = 32 * 1024
MAX_EXCEPTION_EVENTS = 4
MAX_LOG_BODY_BYTES = 16 * 1024
REDACTED = "[REDACTED]"
_SECRET_NAME = (
    r"authorization|proxy.authorization|cookie|set.cookie|password|passwd|"
    r"api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|"
    r"secret|token"
)
_SECRET_KEY = re.compile(
    r"^(?:(?:http[._])?(?:request[._]|response[._])?headers?[._])?(?:"
    + _SECRET_NAME + r")$", re.I,
)
_ASSIGNMENT = re.compile(
    r"(?P<prefix>(?<![\w])(?:" + _SECRET_NAME
    + r")[\"']?\s*[:=]\s*)(?P<value>\"[^\"\n]*\"|'[^'\n]*'|"
    r"(?:Bearer|Basic)\s+[^\s,;\"'<>]+|[^\s,;&\"'<>}]+)", re.I,
)
_BEARER = re.compile(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", re.I)
_COOKIE = re.compile(
    r"(?P<prefix>\b(?:set[-_.]?cookie|cookie)\s*[:=]\s*)[^\r\n]+", re.I,
)
_USERINFO = re.compile(
    r"(?<![\w+.-])([a-z][a-z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@", re.I,
)
_API_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")

RESOURCE_FIELDS = {
    key: (key,) for key in (
        "service.name", "service.namespace", "service.version",
        "service.instance.id", "k8s.namespace.name", "k8s.pod.name",
        "k8s.pod.uid", "k8s.cluster.name", "container.id", "host.name",
    )
}
RESOURCE_FIELDS["deployment.environment.name"] = (
    "deployment.environment.name", "deployment.environment",
)
CODE_FIELDS = {
    "file": ("code.file.path", "code.filepath"),
    "line": ("code.line.number", "code.lineno"),
    "function": ("code.function.name", "code.function"),
    # A legacy namespace cannot safely be joined using one language's syntax.
    "namespace": ("code.namespace",),
}


def sanitize_text(value: str) -> tuple[str, bool]:
    """Redact explicit credential forms; this is not a general PII detector."""
    cleaned = _ASSIGNMENT.sub(lambda m: m["prefix"] + REDACTED, value)
    cleaned = _COOKIE.sub(lambda m: m["prefix"] + REDACTED, cleaned)
    cleaned = _BEARER.sub(lambda m: m[1] + " " + REDACTED, cleaned)
    cleaned = _USERINFO.sub(lambda m: m[1] + REDACTED + "@", cleaned)
    cleaned = _API_TOKEN.sub(REDACTED, cleaned)
    return cleaned, cleaned != value


def sanitize_value(value):
    """Sanitize both structured bodies and their rendered text copies."""
    if isinstance(value, str):
        return sanitize_text(value)[0]
    if isinstance(value, dict):
        return {
            key: REDACTED if _SECRET_KEY.fullmatch(str(key))
            else sanitize_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    return value


def _size(value) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode())


def clip_text(value: str, limit: int) -> str:
    """Retain head and tail within a UTF-8 byte allowance."""
    raw = value.encode()
    if len(raw) <= limit:
        return value
    marker = "\n[... truncated ...]\n"
    available = max(0, limit - len(marker.encode()))
    first = available * 2 // 3
    last = available - first
    return (raw[:first].decode(errors="ignore") + marker
            + (raw[-last:].decode(errors="ignore") if last else ""))


def _lookup(values: dict, key: str):
    if key in values:
        return values[key]
    current = values
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _fields(sources, aliases):
    values, metadata = {}, {"provenance": {}}
    for canonical, names in aliases.items():
        candidates = []
        raw_values = []
        for source, raw in sources:
            if not isinstance(raw, dict):
                continue
            for name in names:
                item = _lookup(raw, name)
                if canonical == "line":
                    if type(item) is not int or item < 1:
                        continue
                elif not isinstance(item, str) or not item:
                    continue
                raw_values.append(item)
                if isinstance(item, str):
                    item, redacted = sanitize_text(item)
                    if redacted:
                        metadata.setdefault("redacted_fields", []).append(
                            canonical)
                    original = len(item.encode())
                    item = clip_text(item, 1024)
                    if len(item.encode()) < original:
                        metadata.setdefault("truncated_fields", {})[
                            canonical] = {"original_bytes": original,
                                          "retained_bytes": len(item.encode())}
                candidates.append({"source": source, "field": name,
                                   "value": item})
        if not candidates:
            continue
        first = candidates[0]
        values[canonical] = first["value"]
        metadata["provenance"][canonical] = {
            k: first[k] for k in ("source", "field")}
        if any(value != raw_values[0] for value in raw_values[1:]):
            metadata.setdefault("conflicts", {})[canonical] = candidates
    if "redacted_fields" in metadata:
        metadata["redacted_fields"] = sorted(set(metadata["redacted_fields"]))
    return values, metadata


def resource_context(sources) -> dict:
    """Prefer resource scope, then modern aliases within each scope."""
    values, metadata = _fields(sources, RESOURCE_FIELDS)
    return ({"resource": values, "resource_metadata": metadata}
            if values else {})


def code_context(values, source: str) -> dict:
    code, metadata = _fields([(source, values)], CODE_FIELDS)
    return {**code, **metadata} if code else {}


def tag_values(tags) -> dict:
    if not isinstance(tags, list):
        return {}
    return {t["key"]: t.get("value") for t in tags
            if isinstance(t, dict) and isinstance(t.get("key"), str)}


def _event_timestamp(value, unit: str | None):
    if value is None:
        return None
    try:
        when = (datetime.fromtimestamp(value / 1_000_000, UTC)
                if unit == "microseconds" and type(value) in (int, float)
                else datetime.fromisoformat(str(value).replace("Z", "+00:00")))
        if when.utcoffset() is not None:
            return when.astimezone(UTC).isoformat()
    except (ValueError, TypeError, OverflowError, OSError):
        pass
    return None


def exception_context(candidates) -> dict:
    """Retain ordered exception records; never parse or invent stack frames."""
    events, count = [], 0
    event_limit = (MAX_EXCEPTION_BYTES - MAX_EXCEPTION_EVENTS - 2) // 4
    for source, values, timestamp, unit, index in candidates:
        if not isinstance(values, dict):
            continue
        if values.get("event") != "exception" and not any(
            _lookup(values, "exception." + field) is not None
            for field in ("type", "message", "stacktrace")
        ):
            continue
        count += 1
        if len(events) == MAX_EXCEPTION_EVENTS:
            continue
        event = {"source": source}
        if index is not None:
            event["event_index"] = index
        when = _event_timestamp(timestamp, unit)
        if when is not None:
            event["timestamp"] = when
        elif timestamp is not None:
            event["timestamp_invalid"] = True
        for field in ("type", "message", "stacktrace"):
            raw = _lookup(values, "exception." + field)
            if not isinstance(raw, str):
                continue
            clean, redacted = sanitize_text(raw)
            event[field] = clean
            if redacted:
                event.setdefault("redacted_fields", []).append(field)
        code = code_context(values, source)
        if code:
            event["code"] = code
        # Bound large strings before serializing; never enlarge the run budget.
        for field, limit in (("type", 512), ("message", 1024),
                             ("stacktrace", 4096)):
            if field in event and len(event[field].encode()) > limit:
                before = len(event[field].encode())
                event[field] = clip_text(event[field], limit)
                event.setdefault("truncated_fields", {})[field] = {
                    "original_bytes": before,
                    "retained_bytes": len(event[field].encode()),
                }
        # Escaping and code metadata also count toward the per-event allowance.
        if _size(event) > event_limit and "code" in event:
            event.pop("code")
            event["code_omitted"] = True
        for field in ("stacktrace", "message", "type"):
            while (_size(event) > event_limit
                   and len(event.get(field, "")) > 64):
                value = event[field]
                original = event.get("truncated_fields", {}).get(
                    field, {}).get("original_bytes", len(value.encode()))
                event[field] = clip_text(
                    value, max(64, len(value.encode()) // 2))
                event.setdefault("truncated_fields", {})[field] = {
                    "original_bytes": original,
                    "retained_bytes": len(event[field].encode()),
                }
        events.append(event)
    if not count:
        return {}
    return {"exceptions": events, "exception_coverage": {
        "total_events": count, "returned_events": len(events),
        "omitted_events": count - len(events),
        "content_limit_bytes": MAX_EXCEPTION_BYTES,
        "retained_bytes": _size(events),
        "truncated": count > len(events) or any(
            e.get("truncated_fields") or e.get("code_omitted")
            or e.get("code", {}).get("truncated_fields") for e in events),
        "redacted": any(e.get("redacted_fields")
                        or e.get("code", {}).get("redacted_fields")
                        for e in events),
    }}
