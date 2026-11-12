"""Bind human labels to exact candidates, never to a later revision."""

from collections import Counter, defaultdict
from datetime import timedelta
import json
from uuid import NAMESPACE_URL, uuid5

from tracemind.agent.verification import subject_for, subject_hash, review_packet
from tracemind.models.verification import ReviewSubject
from tracemind.observability.project import pages


def frozen_subject(state, evidence, record):
    if record.subject is not None:
        subject = record.subject
    else:
        start = next((e.sequence for e in state.events
                      if e.kind == "verification.started"
                      and e.data.get("input_sha256") == record.input_sha256), None)
        if start is None:
            raise ValueError("archived review has no identifiable start event")
        subject = subject_for(state, record.candidate, evidence).model_copy(update={
            "items": record.items,
            "execution_events": [e.model_dump(mode="json") for e in state.events
                                 if e.sequence < start],
        })
    if subject_hash(subject) != record.input_sha256:
        raise ValueError("candidate/evidence fingerprint changed; cannot label this round")
    return subject


def enqueue_reviews(client, queue_id, state, evidence, *, family, split):
    rows, cursor = [], None
    while True:
        result = client.api.observations.get_many(
            session_id=str(state.id), name="diagnosis.verification",
            from_start_time=state.started_at - timedelta(minutes=5),
            to_start_time=state.finished_at + timedelta(minutes=5),
            fields="core,basic,metadata,io",
            cursor=cursor, limit=100,
        )
        rows.extend(result.data)
        cursor = result.meta.cursor
        if not cursor:
            break
    existing = {x.object_id: x for x in pages(
        client.api.annotation_queues.list_queue_items, queue_id,
    )}
    entries, skipped = [], []
    records = [*state.verification_history,
               *([state.verification] if state.verification else [])]
    for index, record in enumerate(records):
        try:
            subject = frozen_subject(state, evidence, record)
        except ValueError as exc:
            skipped.append({"round": index, "reason": str(exc)})
            continue
        packet = review_packet(subject)
        matches = []
        for row in rows:
            raw = row.input
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except ValueError:
                    continue
            if raw == packet:
                matches.append(row)
        if len(matches) != 1:
            skipped.append({"round": index, "reason": "unique observation not found"})
            continue
        row = matches[0]
        from tracemind.observability.metrics import publish_review_score
        publish_review_score(client, trace_id=row.trace_id,
                             observation_id=row.id, record=record)
        item = existing.get(row.id) or client.api.annotation_queues.create_queue_item(
            queue_id, object_id=row.id, object_type="OBSERVATION",
        )
        entries.append({
            "run_id": str(state.id), "incident_family": family,
            "split": split, "round": index, "queue_id": queue_id,
            "queue_item_id": item.id, "trace_id": row.trace_id,
            "observation_id": row.id, "input_sha256": record.input_sha256,
            "input": subject.model_dump(mode="json"),
        })
    return {"entries": entries, "skipped": skipped}


def collect_annotations(client, queue_id):
    completed = {(queue_id, x.object_id) for x in pages(
        client.api.annotation_queues.list_queue_items, queue_id, status="COMPLETED",
    ) if x.object_type == "OBSERVATION"}
    scores, cursor = [], None
    while True:
        result = client.api.scores_v3.get_many_v3(
            source="ANNOTATION", queue_id=queue_id, cursor=cursor,
            fields="details,subject,annotation", limit=100,
        )
        scores.extend(s.model_dump(mode="json") for s in result.data)
        cursor = result.meta.cursor
        if not cursor:
            break
    return completed, scores


def annotation_dataset(entries, completed, scores):
    unique = {}
    for entry in entries:
        key = (entry["queue_id"], entry["trace_id"], entry["observation_id"])
        if key in unique and unique[key] != entry:
            raise ValueError("conflicting annotation manifest entries")
        unique[key] = entry
    entries = list(unique.values())
    families = defaultdict(set)
    for entry in entries:
        families[entry["incident_family"]].add(entry["split"])
    if any(len(splits) != 1 for splits in families.values()):
        raise ValueError("same incident family appears in multiple dataset splits")
    datasets, skipped = defaultdict(list), Counter()
    for entry in entries:
        subject = ReviewSubject.model_validate(entry["input"])
        if subject_hash(subject) != entry["input_sha256"]:
            raise ValueError("annotation manifest subject fingerprint changed")
        observation_id = entry["observation_id"]
        if (entry["queue_id"], observation_id) not in completed:
            skipped["queue_incomplete"] += 1
            continue
        labels = [s for s in scores
                  if s.get("name") == "tracemind.human.verdict"
                  and s.get("source") == "ANNOTATION"
                  and s.get("author_user_id")
                  and s.get("subject", {}).get("kind") == "observation"
                  and s["subject"].get("id") == observation_id
                  and s["subject"].get("trace_id") == entry["trace_id"]
                  and s.get("queue_id") == entry["queue_id"]]
        values = {s["value"] for s in labels}
        if not values:
            skipped["no_verified_human_label"] += 1
            continue
        if len(values) > 1:
            skipped["conflicting_labels"] += 1
            continue
        verdict = values.pop()
        if verdict not in {"supported", "insufficient", "contradicted"}:
            skipped["unknown_label"] += 1
            continue
        # Target verifier experiments select their own rubric. Reference
        # labels/comments are deliberately absent from the model input.
        model_input = subject.model_copy(update={"prompt": None}).model_dump(mode="json")
        datasets[entry["split"]].append({
            "id": str(uuid5(NAMESPACE_URL, entry["split"] + ":" + observation_id)),
            "input": model_input, "expected_output": {"verdict": verdict},
            "metadata": {
                "annotation_schema_version": "tracemind-human-v1",
                "incident_family": entry["incident_family"],
                "dataset_role": entry["split"],
                "source_subject_sha256": entry["input_sha256"],
                "source_score_ids": [s["id"] for s in labels],
                "author_user_ids": sorted({s["author_user_id"] for s in labels}),
                "source_observation_id": observation_id,
                "source_trace_id": entry["trace_id"],
            },
        })
    return {"datasets": dict(datasets), "skipped": dict(skipped),
            "candidate_count": len(entries),
            "labelled_count": sum(len(v) for v in datasets.values())}
