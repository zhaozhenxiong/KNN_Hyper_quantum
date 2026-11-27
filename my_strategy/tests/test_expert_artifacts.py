"""Verify immutable content, references and derived retention evidence."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sqlite3

import pytest

from my_strategy.storage.expert_artifacts import ArtifactRef, ExpertArtifactStore


def test_concurrent_artifact_and_manifest_publication(tmp_path):
    store = ExpertArtifactStore(tmp_path / "store")
    source = tmp_path / "source.parquet"
    content = b"verified source payload\n" * 1000
    source.write_bytes(content)

    def publish(_: int):
        ref = store.put_file(source)
        manifest = store.write_manifest("research-1", {"data_snapshots/panel.parquet": ref})
        return ref, manifest

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(publish, range(18)))
    assert len(set(results)) == 1
    ref, manifest = results[0]
    assert store.verify(ref).read_bytes() == content
    assert not list((store.root / ".staging").iterdir())
    same_payload = store.put_file(source)
    store.write_manifest("research-2", {"replay/input.parquet": same_payload})
    inventory = store.inventory(verify_checksums=True)
    assert inventory["manifests"] == ["research-1", "research-2"]
    assert inventory["errors"] == []
    assert inventory["reference_count"] == 2
    assert inventory["stored_bytes"] == len(content)
    assert inventory["logical_bytes"] == 2 * len(content)
    with pytest.raises(FileExistsError, match="manifest already exists"):
        store.write_manifest("research-1", {"data_snapshots/panel.parquet": ref}, {"changed": True})
    assert manifest.is_file()
    assert store.read_manifest("research-1")["metadata"] == {}


def test_restore_validates_content_and_does_not_overwrite(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(b"a,b\n1,2\n")
    store = ExpertArtifactStore(tmp_path / "store")
    ref = store.put_file(source)
    dest = tmp_path / "result" / "restored.csv"
    assert store.restore(ref, dest) == dest
    assert store.restore(ref, dest).read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match="checksum or length"):
        store.verify(ArtifactRef(ref.sha256, ref.size_bytes + 1))
    dest.write_bytes(b"other")
    with pytest.raises(FileExistsError, match="different content"):
        store.restore(ref, dest)
    assert dest.read_bytes() == b"other"
    blob = store.verify(ref)
    blob.write_bytes(b"X" * ref.size_bytes)
    invalid_dest = tmp_path / "invalid.csv"
    with pytest.raises(ValueError, match="checksum or length"):
        store.restore(ref, invalid_dest)
    assert not invalid_dest.exists()


def test_dry_run_reports_unreferenced_and_fails_closed_on_invalid_map(tmp_path):
    store = ExpertArtifactStore(tmp_path / "store")
    source = tmp_path / "source"
    source.write_bytes(b"retained")
    retained = store.put_file(source)
    source.write_bytes(b"unreferenced")
    unused = store.put_file(source)
    store.write_manifest("retained-run", {"evidence": retained})
    assert store.plan_cleanup(minimum_age_seconds=10_000)["candidates"] == []
    plan = store.plan_cleanup(verify_checksums=True)
    assert plan["mode"] == "dry_run"
    assert [blob["sha256"] for blob in plan["candidates"]] == [unused.sha256]
    assert plan["candidate_bytes"] == unused.size_bytes
    assert store.verify(unused).is_file()
    with pytest.raises(ValueError, match="canonical run ID"):
        store.write_manifest("../outside", {"evidence": retained})
    with pytest.raises(ValueError, match="Invalid artifact role"):
        store.write_manifest("outside", {"../evidence": retained})
    (store.root / "manifests" / "broken.json").write_text('{"files": {}, "files": {}}', encoding="utf-8")
    bad = store.plan_cleanup()
    assert bad["errors"]
    assert bad["candidates"] == []
    assert store.verify(retained).is_file()


def test_rebuilt_sqlite_catalog_has_same_references_and_is_optional(tmp_path):
    store = ExpertArtifactStore(tmp_path / "store")
    source = tmp_path / "source.parquet"
    source.write_bytes(b"payload")
    ref = store.put_file(source)
    store.write_manifest("run-a", {"panel": ref})
    catalog = store.rebuild_index()
    with closing(sqlite3.connect(catalog.as_uri() + "?mode=ro", uri=True)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute("SELECT * FROM artifact_refs").fetchall() == [("run-a", "panel", ref.sha256, ref.size_bytes)]
    store.write_manifest("run-b", {"input": ref})
    assert store.read_manifest("run-b")["files"]["input"]["sha256"] == ref.sha256
    store.rebuild_index()
    with closing(sqlite3.connect(catalog.as_uri() + "?mode=ro", uri=True)) as conn:
        assert conn.execute("SELECT count(*) FROM artifact_refs WHERE sha256 = ?", (ref.sha256,)).fetchone() == (2,)
