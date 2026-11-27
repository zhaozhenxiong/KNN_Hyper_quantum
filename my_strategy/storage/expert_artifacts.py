"""Immutable, verified file artifacts for expert runs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
from tempfile import NamedTemporaryFile
import time
from typing import BinaryIO

from my_strategy.core.paths import ARTIFACT_ROOT, PROJECT_ROOT


_DIGEST = re.compile(r"[0-9a-f]{64}")
_MANIFEST_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,119}")
_ROLE = re.compile(r"[A-Za-z0-9_.-]+")
_BUFFER_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ArtifactRef:
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.sha256, str) or not _DIGEST.fullmatch(self.sha256):
            raise ValueError("Artifact SHA-256 must be lowercase hexadecimal")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("Artifact size must be a nonnegative byte count")


def _reference(value: ArtifactRef | Mapping) -> ArtifactRef:
    if isinstance(value, ArtifactRef):
        return value
    if not isinstance(value, Mapping) or set(value) != {"sha256", "size_bytes"}:
        raise ValueError("Artifact reference must contain SHA-256 and size_bytes")
    return ArtifactRef(value["sha256"], value["size_bytes"])


def _digest(source: BinaryIO, target: BinaryIO | None = None) -> ArtifactRef:
    digest = sha256()
    size = 0
    for block in iter(lambda: source.read(_BUFFER_SIZE), b""):
        digest.update(block)
        size += len(block)
        if target is not None:
            target.write(block)
    return ArtifactRef(digest.hexdigest(), size)


def _require_match(path: Path, ref: ArtifactRef) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Artifact file missing or indirect: {path}")
    with path.open("rb") as source:
        if os.fstat(source.fileno()).st_size != ref.size_bytes or _digest(source) != ref:
            raise ValueError(f"Artifact checksum or length mismatch: {path}")


@contextmanager
def _staged_file(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=directory, prefix=".expert-", suffix=".tmp", delete=False) as stage:
        temporary = Path(stage.name)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def _publish(source: Path, destination: Path) -> None:
    """Publish a finished file exclusively on the same filesystem."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError as error:
        raise OSError(f"Cannot publish immutable file on this filesystem: {destination}") from error


def _unique_pairs(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def _validate_role(role: str) -> None:
    if (not isinstance(role, str) or any(not _ROLE.fullmatch(part) or part.endswith(".")
                                       for part in role.split("/"))):
        raise ValueError(f"Invalid artifact role: {role}")


class ExpertArtifactStore:
    """Store one content file per SHA-256 and multiple immutable run references."""

    def __init__(self, root: str | Path | None = None) -> None:
        path = ARTIFACT_ROOT / "expert_content" if root is None else Path(root)
        self.root = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()

    def _path(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts)
        # Resolve an existing ancestor strictly. A permissive Windows resolve
        # can return an incomplete path while another writer creates a parent.
        ancestor = path
        while ancestor != self.root:
            try:
                resolved = ancestor.resolve(strict=True)
            except FileNotFoundError:
                ancestor = ancestor.parent
                continue
            if not resolved.is_relative_to(self.root):
                raise ValueError(f"Artifact path leaves its storage root: {path}")
            break
        return path

    def _manifest_path(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not _MANIFEST_ID.fullmatch(run_id):
            raise ValueError("Expert artifact manifest requires a canonical run ID")
        return self._path("manifests", f"{run_id}.json")

    def _blob(self, ref: ArtifactRef) -> Path:
        return self._path("sha256", ref.sha256[:2], ref.sha256)

    def put_file(self, source: str | Path) -> ArtifactRef:
        """Import a stable source using a streamed hash and an exclusive publish."""
        source = Path(source).resolve(strict=True)
        with _staged_file(self._path(".staging")) as temporary:
            with source.open("rb") as original, temporary.open("wb") as staged:
                before = os.fstat(original.fileno())
                ref = _digest(original, staged)
                after = source.stat()
                if not os.path.samestat(before, after) or before.st_mtime_ns != after.st_mtime_ns or after.st_size != ref.size_bytes:
                    raise ValueError(f"Source changed during artifact import: {source}")
                staged.flush()
                os.fsync(staged.fileno())
            destination = self._blob(ref)
            try:
                _publish(temporary, destination)
            except FileExistsError:
                _require_match(destination, ref)
        return ref

    def verify(self, ref: ArtifactRef | Mapping) -> Path:
        """Return the local content path after a size and SHA-256 check."""
        ref = _reference(ref)
        path = self._blob(ref)
        _require_match(path, ref)
        return path

    def write_manifest(
        self,
        run_id: str,
        files: Mapping[str, ArtifactRef | Mapping],
        metadata: Mapping | None = None,
    ) -> Path:
        """Publish a manifest only once. Identical repeated submissions reuse it."""
        destination = self._manifest_path(run_id)
        if not files:
            raise ValueError("Artifact manifest requires at least one file")
        checked = {}
        for role, value in files.items():
            _validate_role(role)
            ref = _reference(value)
            self.verify(ref)
            checked[role] = asdict(ref)
        record = {"schema_version": 1, "run_id": run_id, "files": checked, "metadata": dict(metadata or {})}
        payload = (json.dumps(record, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        with _staged_file(self._path(".staging")) as temporary:
            with temporary.open("wb") as staged:
                staged.write(payload)
                staged.flush()
                os.fsync(staged.fileno())
            try:
                _publish(temporary, destination)
            except FileExistsError:
                if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != payload:
                    raise FileExistsError(f"Immutable artifact manifest already exists: {destination}") from None
        return destination

    def read_manifest(self, run_id: str) -> dict:
        """Read the authoritative reference map without opening SQLite or writing files."""
        path = self._manifest_path(run_id)
        if path.is_symlink():
            raise ValueError(f"Artifact manifest must be a direct file: {path}")
        record = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
        if (not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("run_id") != run_id
                or not isinstance(record.get("files"), dict) or not record["files"] or not isinstance(record.get("metadata"), dict)):
            raise ValueError(f"Malformed artifact manifest: {path}")
        for role, ref in record["files"].items():
            _validate_role(role)
            _reference(ref)
        return record

    def restore(self, ref: ArtifactRef | Mapping, destination: str | Path) -> Path:
        """Restore verified content atomically without replacing a different destination."""
        ref = _reference(ref)
        destination = Path(destination).absolute()
        source = self._blob(ref)
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"Artifact file missing or indirect: {source}")
        # Publish on the destination volume even when the content store is elsewhere.
        with _staged_file(destination.parent) as temporary:
            with source.open("rb") as original, temporary.open("wb") as staged:
                if _digest(original, staged) != ref:
                    raise ValueError(f"Artifact checksum or length mismatch: {source}")
                staged.flush()
                os.fsync(staged.fileno())
            try:
                _publish(temporary, destination)
            except FileExistsError:
                try:
                    _require_match(destination, ref)
                except (FileNotFoundError, ValueError):
                    raise FileExistsError(f"Restore destination already contains different content: {destination}") from None
        return destination

    def inventory(self, *, verify_checksums: bool = False) -> dict:
        """Inventory immutable manifests and their referenced physical content."""
        errors = []
        references: dict[str, list[dict]] = defaultdict(list)
        manifests = []
        for path in sorted(self._path("manifests").glob("*.json")):
            try:
                manifest = self.read_manifest(path.stem)
            except (OSError, ValueError) as error:
                errors.append({"path": str(path), "error": str(error)})
                continue
            manifests.append(manifest["run_id"])
            for role, ref in manifest["files"].items():
                references[ref["sha256"]].append({"run_id": path.stem, "role": role, "size_bytes": ref["size_bytes"]})

        blobs = []
        available = set()
        for candidate in sorted(self._path("sha256").glob("*/*")):
            digest = candidate.name
            if not _DIGEST.fullmatch(digest) or candidate.parent.name != digest[:2]:
                continue
            try:
                path = self._blob(ArtifactRef(digest, 0))
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"Artifact file missing or indirect: {path}")
                stat = path.stat()
                ref = ArtifactRef(digest, stat.st_size)
                if any(item["size_bytes"] != ref.size_bytes for item in references[digest]):
                    raise ValueError(f"Artifact length differs from manifest: {path}")
                if verify_checksums:
                    _require_match(path, ref)
            except (OSError, ValueError) as error:
                errors.append({"path": str(candidate), "error": str(error)})
                continue
            available.add(digest)
            blobs.append({**asdict(ref), "path": str(path), "mtime": stat.st_mtime, "references": references[digest]})
        for digest in sorted(set(references) - available):
            if references[digest]:
                errors.append({"sha256": digest, "error": "Referenced artifact missing or invalid", "references": references[digest]})
        return {
            "root": str(self.root), "manifests": manifests, "blobs": blobs, "errors": errors,
            "checksums_verified": verify_checksums,
            "reference_count": sum(len(items) for items in references.values()),
            "logical_bytes": sum(item["size_bytes"] for items in references.values() for item in items),
            "stored_bytes": sum(blob["size_bytes"] for blob in blobs),
        }

    def plan_cleanup(self, *, minimum_age_seconds: float = 0, verify_checksums: bool = False) -> dict:
        """List unreferenced content for review. This operation never deletes files."""
        if minimum_age_seconds < 0:
            raise ValueError("Minimum age must be nonnegative")
        inventory = self.inventory(verify_checksums=verify_checksums)
        cutoff = time.time() - minimum_age_seconds
        candidates = [] if inventory["errors"] else [
            blob for blob in inventory["blobs"] if not blob["references"] and blob["mtime"] <= cutoff
        ]
        return {**inventory, "mode": "dry_run", "candidates": candidates,
                "candidate_bytes": sum(blob["size_bytes"] for blob in candidates)}

    def rebuild_index(self) -> Path:
        """Publish a derived SQLite index of manifests. Manifests remain authoritative."""
        root = self.root
        destination = self._path("catalog.sqlite")
        inventory = self.inventory()
        if inventory["errors"]:
            raise ValueError("Cannot index incomplete or invalid expert artifact references")
        with _staged_file(root) as temporary:
            with closing(sqlite3.connect(temporary)) as conn, conn:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.executescript("""
                    CREATE TABLE manifests (run_id TEXT PRIMARY KEY, path TEXT NOT NULL);
                    CREATE TABLE artifact_refs (
                        run_id TEXT NOT NULL REFERENCES manifests(run_id), role TEXT NOT NULL,
                        sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL, PRIMARY KEY (run_id, role)
                    );
                    CREATE INDEX artifact_refs_hash ON artifact_refs(sha256);
                """)
                conn.executemany("INSERT INTO manifests VALUES (?, ?)", [
                    (run_id, f"manifests/{run_id}.json") for run_id in inventory["manifests"]
                ])
                conn.executemany("INSERT INTO artifact_refs VALUES (?, ?, ?, ?)", [
                    (item["run_id"], item["role"], blob["sha256"], item["size_bytes"])
                    for blob in inventory["blobs"] for item in blob["references"]
                ])
            temporary.replace(destination)
        return destination
