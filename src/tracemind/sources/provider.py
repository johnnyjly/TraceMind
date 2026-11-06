"""Read immutable Git objects; never execute code or fetch model-chosen URLs.
"""

import asyncio
import json
from hashlib import sha256
from pathlib import Path, PurePosixPath

from tracemind.models.intake import redact_reported_text
from tracemind.models.sources import (
    SourcePolicy,
    SourceReference,
    relative_path,
    bound_reference,
)
from tracemind.agent.workflow.observations import scope_status

MAX_GIT_BYTES = 1048576
SCOPE = {
    "environment": "deployment.environment.name",
    "cluster": "k8s.cluster.name",
    "namespace": "k8s.namespace.name",
}


def load_source_policy(path) -> SourcePolicy | None:
    if path is None:
        return None
    path = Path(path).resolve()
    if path.stat().st_size > MAX_GIT_BYTES:
        raise ValueError("release snapshot exceeds one MiB")
    data = json.loads(path.read_text())
    # Computed identifiers are never accepted as input authority.
    data.pop("snapshot_id", None)
    policy = SourcePolicy.model_validate(data)
    for repo in policy.repositories:
        root = Path(repo.path)
        repo.path = str(
            (path.parent / root).resolve()
            if not root.is_absolute()
            else root.resolve()
        )
    return policy


async def git(repo, *args, max_bytes=MAX_GIT_BYTES):
    process = await asyncio.create_subprocess_exec(
        "git",
        "--no-pager",
        "--no-replace-objects",
        "--literal-pathspecs",
        "-C",
        repo.path,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        result = bytearray()
        while chunk := await process.stdout.read(65536):
            result.extend(chunk)
            if len(result) > max_bytes:
                raise ValueError("Git output exceeds source limit")
        if await process.wait():
            raise ValueError("requested Git object is unavailable")
        return bytes(result)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


class SourcePathNotFound(ValueError):
    """The requested path is absent from the pinned revision(s)."""


class SourceProvider:
    def __init__(self, policy):
        self.policy = policy
        self.repos = {r.id: r for r in policy.repositories}
        self.releases = {r.id: r for r in policy.releases}

    def reference(
        self,
        kind,
        release,
        content,
        *,
        evidence_ids=(),
        status="declared",
        metadata=None,
        truncated=False,
        redacted=False,
    ):
        clean = redact_reported_text(content, preserve_lines=kind == "source")
        return bound_reference(
            SourceReference(
                kind=kind,
                snapshot_id=self.policy.snapshot_id,
                service=release.service,
                mapping_status=status,
                evidence_ids=list(evidence_ids),
                content=clean,
                redacted=redacted or clean != content,
                truncated=truncated,
                metadata={
                    "release_id": release.id,
                    "repository_id": release.repository_id,
                    "commit": release.commit,
                    **(metadata or {}),
                },
            ),
            self.policy.response_bytes,
        )

    async def resolve(self, evidence, scope=None):
        resource = evidence.attributes.get("resource", {})
        metadata = evidence.attributes.get("resource_metadata", {})
        resource = dict(resource) if isinstance(resource, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        incomplete = (
            set(metadata.get("conflicts", {}))
            | set(metadata.get("redacted_fields", []))
            | set(metadata.get("truncated_fields", []))
        )
        for key in incomplete:
            resource.pop(key, None)
        scoped = scope_status(evidence, scope)
        if scoped != "matched":
            return SourceReference(
                kind="deployment",
                snapshot_id=self.policy.snapshot_id,
                service=evidence.service or "unknown",
                mapping_status=(
                    "mismatch" if scoped == "scope_mismatch" else "missing"
                ),
                evidence_ids=[evidence.id],
                metadata={"reason": scoped},
                content=(
                    "Runtime resource identity does not establish the"
                    " requested scope."
                ),
            )
        candidates = []
        mismatches = False
        for release in self.policy.releases:
            if release.service != evidence.service:
                continue
            at = evidence.observed_at
            if at.utcoffset() is None or not (
                release.deployed_at <= at
                and (release.ended_at is None or at < release.ended_at)
            ):
                continue
            required = scope.model_dump(exclude_none=True) if scope else {}
            if any(release.scope.get(k) != v for k, v in required.items()):
                continue
            if any(
                resource.get(attribute) is not None
                and release.scope.get(key) != resource[attribute]
                for key, attribute in SCOPE.items()
            ):
                continue
            checks = []
            for attribute, values in (
                ("container.id", release.containers),
                ("service.instance.id", release.instances),
                (
                    "container.image.id",
                    [release.image_id] if release.image_id else [],
                ),
            ):
                if resource.get(attribute) and values:
                    checks.append(resource[attribute] in values)
            digests = resource.get("container.image.repo_digests")
            if digests and release.image_digest:
                checks.append(
                    release.image_digest
                    in (digests if isinstance(digests, list) else [digests])
                )
            if False in checks:
                mismatches = True
                continue
            version = resource.get("service.version")
            if version and release.version and version != release.version:
                mismatches = True
                continue
            if checks or (version and version == release.version):
                candidates.append((release, bool(checks)))
        if len(candidates) != 1:
            status = (
                "ambiguous"
                if candidates
                else "mismatch" if mismatches else "missing"
            )
            return SourceReference(
                kind="deployment",
                snapshot_id=self.policy.snapshot_id,
                service=evidence.service or "unknown",
                mapping_status=status,
                evidence_ids=[evidence.id],
                metadata={
                    "candidate_release_ids": [r.id for r, _ in candidates]
                },
                content=(
                    "No unique runtime-to-source mapping was established. "
                    "Do not substitute HEAD or the newest release."
                ),
            )
        release, strong = candidates[0]
        repo = self.repos[release.repository_id]
        try:
            tree = (
                (
                    await git(
                        repo,
                        "rev-parse",
                        "--verify",
                        release.commit + "^{tree}",
                    )
                )
                .decode()
                .strip()
            )
        except (OSError, ValueError):
            return self.reference(
                "deployment",
                release,
                "The recorded commit is unavailable locally.",
                status="unavailable",
                evidence_ids=[evidence.id],
            )
        status = "declared"
        if strong and release.build:
            status = (
                "matched" if tree == release.build.source_tree else "mismatch"
            )
        data = release.model_dump(mode="json", exclude_none=True)
        return self.reference(
            "deployment",
            release,
            json.dumps(data, ensure_ascii=False),
            evidence_ids=[evidence.id],
            status=status,
            metadata={
                "source_tree": tree,
                "previous_release_id": release.previous_release_id,
                "basis": (
                    "Configured deployment/build capture plus runtime"
                    " identity; not cryptographic attestation."
                ),
            },
        )

    def release_for(self, reference):
        if (
            reference.snapshot_id != self.policy.snapshot_id
            or reference.kind != "deployment"
        ):
            raise ValueError(
                "resolve a deployment in this snapshot before reading source"
            )
        if reference.mapping_status not in {"matched", "declared"}:
            raise ValueError("deployment mapping is unresolved")
        return self.releases[reference.metadata["release_id"]]

    def path(self, repo, path):
        if path.startswith("/"):
            matches = [
                (prefix, target)
                for prefix, target in repo.runtime_paths.items()
                if path == prefix.rstrip("/")
                or path.startswith(prefix.rstrip("/") + "/")
            ]
            if not matches:
                raise ValueError(
                    "runtime path has no configured source mapping"
                )
            prefix, target = max(matches, key=lambda item: len(item[0]))
            path = str(PurePosixPath(target) / path[len(prefix):].lstrip("/"))
        path = relative_path(path)
        if not PurePosixPath(path).is_relative_to(
            PurePosixPath(repo.directory)
        ):
            raise ValueError(
                "file is outside the configured service directory"
            )
        return path

    async def read(self, reference, path, start_line, end_line):
        release = self.release_for(reference)
        repo = self.repos[release.repository_id]
        path = self.path(repo, path)
        entry = await git(repo, "ls-tree", "-z", release.commit, "--", path)
        # Symlinks and submodules are not traversed, even inside a Git tree.
        if not entry:
            raise SourcePathNotFound("source_path_not_found")
        if not entry.startswith((b"100644 blob ", b"100755 blob ")):
            raise ValueError("source must be a regular tracked file")
        blob = entry.split(b" ", 2)[2].split(b"\t", 1)[0].decode()
        size = int(await git(repo, "cat-file", "-s", blob))
        if size > MAX_GIT_BYTES:
            raise ValueError("source file exceeds one MiB")
        data = await git(repo, "cat-file", "blob", blob)
        if b"\x00" in data:
            raise ValueError("binary source is unsupported")
        original = data.decode("utf-8")
        # Detect multiline credential forms before selecting a line window.
        clean = redact_reported_text(original, preserve_lines=True)
        lines = clean.splitlines()
        if start_line > len(lines):
            raise ValueError("source line is out of range")
        end = min(end_line, start_line + self.policy.max_lines - 1, len(lines))
        selected, length = [], 0
        for number in range(start_line, end + 1):
            line = f"{number}: {lines[number - 1]}\n"
            if length + len(line.encode()) > self.policy.response_bytes:
                break
            selected.append(line)
            length += len(line.encode())
        if not selected:
            raise ValueError("source line exceeds response byte limit")
        return self.reference(
            "source",
            release,
            "".join(selected),
            status=reference.mapping_status,
            evidence_ids=reference.evidence_ids,
            metadata={
                "path": path,
                "blob": blob,
                "file_sha256": sha256(data).hexdigest(),
                "start_line": start_line,
                "end_line": start_line + len(selected) - 1,
                "total_lines": len(lines),
                "deployment_ref": reference.id,
            },
            truncated=start_line + len(selected) - 1 < end_line,
            redacted=clean != original,
        )

    async def check_diff_content(self, repo, base, head, target):
        # A hunk can omit the boundaries of a multiline credential. Refuse
        # credential-bearing files after inspecting their complete Git blobs.
        entries = await git(
            repo, "diff", "--raw", "--no-abbrev", "-z", "--no-renames",
            base, head, "--", target,
        )
        parts = entries.rstrip(b"\x00").split(b"\x00") if entries else []
        if len(parts) > 64:
            raise ValueError("release diff exceeds 32 changed files")
        objects = set()
        for header in parts[::2]:
            old_mode, new_mode, old, new, _ = header.split()
            if old_mode[1:] not in {
                b"000000",
                b"100644",
                b"100755",
            } or new_mode not in {b"000000", b"100644", b"100755"}:
                raise ValueError("diff requires regular tracked files")
            objects.update(value for value in (old, new) if set(value) != {48})
        total = 0
        for blob in sorted(objects):
            name = blob.decode("ascii")
            total += int(await git(repo, "cat-file", "-s", name))
            if total > MAX_GIT_BYTES:
                raise ValueError("diff source inspection exceeds one MiB")
            data = await git(repo, "cat-file", "blob", name)
            if b"\x00" in data:
                raise ValueError("binary diff is unsupported")
            text = data.decode("utf-8")
            if redact_reported_text(text, preserve_lines=True) != text:
                raise ValueError("diff contains credential-bearing source")

    async def diff(self, reference, path=None):
        release = self.release_for(reference)
        base = self.releases.get(release.previous_release_id)
        if base is None:
            raise ValueError("no previous actual deployment is recorded")
        repo = self.repos[release.repository_id]
        target = self.path(repo, path) if path else repo.directory
        before = await git(repo, "ls-tree", "-z", base.commit, "--", target)
        after = await git(repo, "ls-tree", "-z", release.commit, "--", target)
        if not before and not after:
            raise SourcePathNotFound("source_path_not_found")
        await self.check_diff_content(
            repo, base.commit, release.commit, target
        )
        raw = await git(
            repo,
            "-c",
            "diff.external=",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--unified=3",
            base.commit,
            release.commit,
            "--",
            target,
        )
        text = raw.decode("utf-8")
        return self.reference(
            "diff",
            release,
            text,
            status=reference.mapping_status,
            evidence_ids=reference.evidence_ids,
            metadata={
                "base_release_id": base.id,
                "base_commit": base.commit,
                "head_commit": release.commit,
                "path": target,
                "diff_sha256": sha256(raw).hexdigest(),
                "deployment_ref": reference.id,
            },
        )
