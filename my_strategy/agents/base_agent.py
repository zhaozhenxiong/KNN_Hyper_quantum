#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base types shared by all strategy agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentSignal:
    agent_name: str
    score: float
    confidence: float
    action: str
    reason: str
    extra: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    def __init__(self, name: str, weight: float = 1.0, enabled: bool = True):
        self.name = name
        self.weight = float(weight)
        self.enabled = bool(enabled)

    @abstractmethod
    def run(self, data, context: dict[str, Any] | None = None) -> AgentSignal:
        """Run one agent against one row or dataframe."""


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(min(float(value), upper), lower)

