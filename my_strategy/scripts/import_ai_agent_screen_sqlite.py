#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command line wrapper for importing AI-Agent screen outputs into SQLite."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.storage.screen_sqlite import main  # noqa: E402


if __name__ == "__main__":
    main()
