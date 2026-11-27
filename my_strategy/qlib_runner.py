#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qlib research entrypoint for the personal strategy workspace."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QLIB_SOURCE = PROJECT_ROOT / "qlib-main"


def main():
    if importlib.util.find_spec("qlib") is None:
        print("当前 Python 环境还没有安装 qlib。")
        print("建议先执行:")
        print("  python -m pip install -e .\\qlib-main")
        print("安装后可先跑官方示例:")
        print("  python .\\qlib-main\\examples\\workflow_by_code.py")
        return

    import qlib
    from qlib.constant import REG_CN

    provider_uri = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    print(f"初始化 Qlib 数据目录: {provider_uri}")
    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    print("Qlib 已初始化。下一步可以把 my_strategy/signals.py 的因子迁移为 Qlib Dataset/Handler。")


if __name__ == "__main__":
    main()

