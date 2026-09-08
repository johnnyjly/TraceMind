"""Factories for structured proposals in existing harness scenarios."""


def proposal_payload(fields):
    if "root_cause" not in fields:
        return fields
    fields = dict(fields)
    statement = {"text": fields.pop("root_cause"),
                 "evidence_refs": fields.pop("evidence_ids", []),
                 "knowledge_refs": fields.pop("knowledge_refs", [])}
    checks = fields.pop("remediation", [])
    return {**fields, "findings": [statement], "hypotheses": [],
            "limitations": [], "proposed_checks": [
                {"text": text, "evidence_refs": [], "knowledge_refs": []}
                for text in checks]}
