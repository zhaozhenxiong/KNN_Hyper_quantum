"""Schema for the unified model registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

ModelStage = Literal["candidate", "validated", "champion", "retired"]


@dataclass
class ModelEntry:
    model_id: str
    name: str
    kind: str
    stage: ModelStage
    run_id: str
    model_path: str
    report_path: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    data_range: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    git_commit: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelEntry":
        return cls(**data)
