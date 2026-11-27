"""Run identity and artifact metadata helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from my_strategy.core.paths import PROJECT_ROOT, artifact_run_dir, artifact_subdir


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def current_git_commit(short: bool = False) -> str:
    commit = _git_output("rev-parse", "--short", "HEAD") if short else _git_output("rev-parse", "HEAD")
    return commit or "NO_GIT"


def stable_hash(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def slugify(value: str, limit: int = 48) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", str(value), flags=re.UNICODE)
    text = re.sub(r"-+", "-", text).strip("-_.")
    return (text or "run")[:limit]


@dataclass(frozen=True)
class RunContext:
    run_id: str
    as_of_date: str
    git_commit: str
    config_hash: str
    data_version: str
    seed: int = 42
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))
    task: str = ""
    artifact_root: str = ""

    def run_dir(self) -> Path:
        return artifact_run_dir(self.run_id)

    def subdir(self, *parts: str) -> Path:
        return artifact_subdir(self.run_id, *parts)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifact_root"] = str(self.run_dir())
        return data

    def write_metadata(self, extra: dict[str, Any] | None = None) -> Path:
        payload = self.to_dict()
        if extra:
            payload.update(extra)
        path = self.run_dir() / "metadata.json"
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return path


def make_run_id(task: str = "run", as_of_date: str | date | None = None, commit: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    date_part = str(as_of_date or date.today()).replace("-", "")
    git_part = (commit or current_git_commit(short=True))[:12]
    return f"{stamp}-{date_part}-{slugify(task, 36)}-{git_part}"


def create_run_context(
    *,
    task: str = "run",
    as_of_date: str | date | None = None,
    config: dict[str, Any] | None = None,
    data_version: str = "",
    seed: int = 42,
    run_id: str | None = None,
) -> RunContext:
    commit = current_git_commit()
    effective_date = str(as_of_date or date.today())
    context = RunContext(
        run_id=run_id or make_run_id(task=task, as_of_date=effective_date, commit=commit),
        as_of_date=effective_date,
        git_commit=commit,
        config_hash=stable_hash(config or {}),
        data_version=data_version or "local",
        seed=int(seed),
        task=task,
    )
    context.run_dir()
    for name in ["config", "data_snapshots", "signals", "backtests", "models", "reports", "logs", "validation"]:
        context.subdir(name)
    context.write_metadata({"config": config or {}})
    return context


def load_run_metadata(run_id: str) -> dict[str, Any]:
    path = artifact_run_dir(run_id, create=False) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))
