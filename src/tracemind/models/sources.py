"""Pinned deployment records and direct source references, separate from E/K.
"""

from hashlib import sha256
import json
import re
from pathlib import PurePosixPath
from typing import Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

SOURCE_TOOLS = (
    "resolve_deployment",
    "read_source",
    "get_release_diff",
    "read_source_reference",
)


def content_hash(value) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def relative_path(value: str, *, directory=False) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
        or value.startswith("-")
    ):
        raise ValueError("source path must be a repository-relative path")
    if not directory and str(path) == ".":
        raise ValueError("source file path required")
    return str(path)


class SourceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    @model_validator(mode="before")
    @classmethod
    def computed_identity(cls, value):
        if isinstance(value, dict):
            return {
                k: v
                for k, v in value.items()
                if k not in cls.model_computed_fields
            }
        return value


class RepositorySource(SourceModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$", max_length=120)
    path: str = Field(min_length=1, max_length=4096)
    service: str = Field(min_length=1, max_length=120)
    directory: str = "."
    runtime_paths: dict[str, str] = Field(default_factory=dict, max_length=16)

    @model_validator(mode="after")
    def paths(self):
        self.directory = relative_path(self.directory, directory=True)
        for runtime, source in self.runtime_paths.items():
            if (
                not runtime.startswith("/")
                or ".." in PurePosixPath(runtime).parts
            ):
                raise ValueError("runtime mapping requires an absolute prefix")
            relative_path(source, directory=True)
        return self


class BuildCapture(SourceModel):
    """Operator-controlled build capture, not cryptographic attestation."""

    record_id: str = Field(min_length=1, max_length=240)
    captured_at: AwareDatetime
    producer: str = Field(min_length=1, max_length=240)
    source_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ReleaseRecord(SourceModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$", max_length=120)
    service: str = Field(min_length=1, max_length=120)
    repository_id: str = Field(min_length=1, max_length=120)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    version: str | None = Field(default=None, max_length=240)
    scope: dict[str, str] = Field(default_factory=dict, max_length=3)
    deployed_at: AwareDatetime
    ended_at: AwareDatetime | None = None
    instances: list[str] = Field(default_factory=list, max_length=100)
    containers: list[str] = Field(default_factory=list, max_length=100)
    image_id: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    image_digest: str | None = Field(default=None, max_length=512)
    previous_release_id: str | None = Field(default=None, max_length=120)
    provenance: str = Field(min_length=1, max_length=1024)
    captured_at: AwareDatetime
    build: BuildCapture | None = None

    @model_validator(mode="after")
    def valid_interval(self):
        if self.ended_at is not None and self.ended_at <= self.deployed_at:
            raise ValueError("release end must follow deployment")
        if not set(self.scope) <= {"environment", "cluster", "namespace"}:
            raise ValueError("unknown deployment scope")
        if self.build and self.build.image_id != self.image_id:
            raise ValueError("build and deployment image identity disagree")
        return self


class SourcePolicy(SourceModel):
    schema_version: Literal[1] = 1
    repositories: list[RepositorySource] = Field(min_length=1, max_length=32)
    releases: list[ReleaseRecord] = Field(default_factory=list, max_length=256)
    max_calls: int = Field(default=6, ge=1, le=6)
    max_lines: int = Field(default=200, ge=1, le=200)
    response_bytes: int = Field(default=32768, ge=1024, le=32768)
    retention_bytes: int = Field(default=262144, ge=1024, le=262144)

    @model_validator(mode="after")
    def identities(self):
        repos = {r.id: r for r in self.repositories}
        releases = {r.id: r for r in self.releases}
        if len(repos) != len(self.repositories) or len(releases) != len(
            self.releases
        ):
            raise ValueError("duplicate repository or release ID")
        for release in self.releases:
            repo = repos.get(release.repository_id)
            if repo is None or repo.service != release.service:
                raise ValueError(
                    "release repository/service is not configured"
                )
            previous = releases.get(release.previous_release_id)
            if release.previous_release_id and (
                previous is None
                or previous.id == release.id
                or previous.service != release.service
                or previous.repository_id != release.repository_id
                or previous.scope != release.scope
                or previous.deployed_at >= release.deployed_at
            ):
                raise ValueError("invalid previous deployment")
        return self

    @computed_field
    @property
    def snapshot_id(self) -> str:
        return content_hash(
            self.model_dump(mode="json", exclude={"snapshot_id"})
        )


class SourceReference(SourceModel):
    kind: Literal["deployment", "source", "diff"]
    snapshot_id: str
    service: str
    mapping_status: Literal[
        "matched",
        "declared",
        "missing",
        "ambiguous",
        "mismatch",
        "unavailable",
    ]
    evidence_ids: list[UUID] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    content: str
    truncated: bool = False
    redacted: bool = False

    @computed_field
    @property
    def id(self) -> str:
        return "source-" + content_hash(
            self.model_dump(mode="json", exclude={"id"})
        )


def source_preview(reference: SourceReference) -> dict:
    record = reference.model_dump(mode="json")
    record["content"] = reference.content[:1600]
    record["preview_truncated"] = len(reference.content) > 1600
    return record


def bound_record(record: dict, limit: int) -> dict:
    """Fit serialized content including escaping and metadata in the byte cap.
    """
    record = dict(record)
    content = record.get("content", "")

    def size():
        return len(json.dumps(record, ensure_ascii=False).encode())

    if size() <= limit:
        return record
    record["truncated"] = True
    record["content"] = ""
    if size() > limit:
        raise ValueError("source metadata exceeds response byte limit")
    low, high = 0, len(content)
    while low < high:
        middle = (low + high + 1) // 2
        record["content"] = content[:middle]
        if size() <= limit:
            low = middle
        else:
            high = middle - 1
    record["content"] = content[:low]
    return record


def bound_reference(reference: SourceReference, limit: int) -> SourceReference:
    record = bound_record(reference.model_dump(mode="json"), limit)
    if reference.kind == "source" and record["content"] != reference.content:
        content = (
            record["content"].rsplit("\n", 1)[0] + "\n"
            if "\n" in record["content"]
            else ""
        )
        if not content:
            raise ValueError("source line exceeds response byte limit")
        record["content"] = content
        numbers = re.findall(r"(?m)^(\d+):", content)
        if not numbers:
            raise ValueError("source excerpt lost its line coordinates")
        record["metadata"]["end_line"] = int(numbers[-1])
    return SourceReference.model_validate(record)
