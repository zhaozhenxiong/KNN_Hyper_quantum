"""CLI and local HTTP entry points for the research workbench."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sqlite3
import traceback
from urllib.parse import urlsplit

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, artifact_run_dir
from my_strategy.core.run_context import make_run_id
from my_strategy.research_workbench.data import WORKBENCH_FEATURE_COLUMNS
from my_strategy.research_workbench.runner import REPORT, TASK, factor_catalog, run, write_json
from my_strategy.storage.db_paths import raw_db_path
from my_strategy.research_workbench.universe import POLICIES, load_universe, target_coverage

import pandas as pd


WEB_ROOT = Path(__file__).with_name("web")
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/sector.css": ("sector.css", "text/css; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/expert.js": ("expert.js", "text/javascript; charset=utf-8"),
    "/sector-chart.js": ("sector-chart.js", "text/javascript; charset=utf-8"),
    "/strategy-backtest.js": ("strategy-backtest.js", "text/javascript; charset=utf-8"),
    "/stock-research-chart.js": ("stock-research-chart.js", "text/javascript; charset=utf-8"),
    "/stock-chart.js": ("stock-chart.js", "text/javascript; charset=utf-8"),
    "/lightweight-charts.js": ("vendor/lightweight-charts.standalone.production.js", "text/javascript; charset=utf-8"),
}
STATUS_FILE = "workbench-status.json"
ID_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z_.-]{0,127}")
CHOICES = {
    "mode": {"baseline", "full", "seqlstm"}, "device": {"auto", "cpu", "cuda"},
    "price_basis": {"raw_research", "adjusted"}, "universe_policy": POLICIES,
}
LIMITS = {
    "max_stocks": (0, 10000), "top_n": (2, 100), "seed": (0, 2**31 - 1), "epochs": (1, 200),
    "rebalance_days": (1, 30), "score_threshold": (0, 1), "max_weight": (0.01, 1),
    "max_industry_weight": (0.01, 1), "lookback": (5, 120), "max_train_samples": (0, 2_000_000),
}
DEFAULTS = {
    "max_stocks": 40, "top_n": 8, "seed": 42, "epochs": 8,
    "mode": "baseline", "device": "auto", "price_basis": "raw_research",
    "universe_policy": "legacy", "lookback": 30, "max_train_samples": 200000, "holdout_start": "",
    "rebalance_days": 5, "score_threshold": 0.70, "max_weight": 0.15, "max_industry_weight": 0.35,
}
PATH_ARGS = ("classification_path", "benchmark_path", "adjusted_path", "execution_reference_path", "universe_path")
RETIRED_START_ROUTES = {
    "/api/runs", "/api/latent/runs", "/api/latent/paper/prepare", "/api/stock-chart/replay",
    "/api/expert/replays", "/api/expert/diagnostic-replays",
}


def _market() -> dict:
    path = raw_db_path().resolve(strict=True)
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
        first, last, count = connection.execute(
            """SELECT MIN(first_date), (SELECT MAX(date) FROM stock_daily_normalized),
                      COUNT(*) FROM securities WHERE raw_rows > 0"""
        ).fetchone()
        on_last = connection.execute(
            "SELECT COUNT(*) FROM stock_daily_normalized WHERE date=? AND has_trade_price=1", (last,)
        ).fetchone()[0]
    if not last:
        raise ValueError("本地日线仓没有可用行情。")
    return {"data_start": first, "latest_date": last, "total_stocks": count, "latest_stocks": on_last}


def _defaults(market: dict) -> dict:
    end = date.fromisoformat(market["latest_date"])
    preferred = date(end.year - 2, 1, 2)
    latest_start = end - timedelta(days=365)
    start = max(date.fromisoformat(market["data_start"]), min(preferred, latest_start))
    return {**DEFAULTS, "start": start.isoformat(), "end": end.isoformat()}


def _status(run_id: str, status: str, message: str, summary: dict | None = None) -> dict:
    payload = {
        "run_id": run_id, "status": status, "message": message,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if summary is not None:
        payload["summary"] = summary
    write_json(artifact_run_dir(run_id) / STATUS_FILE, payload)
    return payload


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_id(run_id: str) -> str:
    if not ID_PATTERN.fullmatch(run_id):
        raise ValueError("运行标识格式无效。")
    return run_id


def _runs() -> list[dict]:
    statuses = sorted(ARTIFACT_RUNS_ROOT.glob(f"*/{STATUS_FILE}"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [_read(path) for path in statuses[:50]]


def _config(values: dict, defaults: dict) -> dict:
    config = {**defaults, **{key: value for key, value in values.items() if value is not None}}
    for key, options in CHOICES.items():
        if not isinstance(config[key], str) or config[key] not in options:
            raise ValueError(f"{key} 的参数无效。")
    for key, (minimum, maximum) in LIMITS.items():
        kind = int if isinstance(DEFAULTS[key], int) else (int, float)
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, kind) or not minimum <= value <= maximum:
            raise ValueError(f"{key} 需要在 {minimum} 至 {maximum} 之间。")
    for key in ("start", "end"):
        value = config.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError(f"{key} 需要 YYYY-MM-DD 日期。")
        date.fromisoformat(value)
    if config["start"] >= config["end"]:
        raise ValueError("开始日期必须早于结束日期。")
    if 0 < config["max_stocks"] < 5 or (not config["max_stocks"] and config["universe_policy"] == "legacy"):
        raise ValueError("股票上限至少为 5。只有有名录的股票池可用 0 跑全量。")
    if config["holdout_start"]:
        cutoff = config["holdout_start"]
        if not isinstance(cutoff, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff):
            raise ValueError("留出起始日需要 YYYY-MM-DD 日期。")
        date.fromisoformat(cutoff)
        if not (config["start"] < cutoff < config["end"]):
            raise ValueError("留出起始日须在研究区间内。")
    if config["universe_policy"] == "legacy":
        config.pop("universe_path", None)
    elif not config.get("universe_path"):
        raise ValueError("所选股票池缺少服务器配置的交易所名录。")
    for key in PATH_ARGS:
        if config.get(key):
            path = Path(config[key]).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"{key} 必须是文件。")
            config[key] = str(path)
    return config


def _run_command(args: argparse.Namespace) -> int:
    market = _market() if not (args.start and args.end) else None
    defaults = _defaults(market) if market else DEFAULTS
    config = _config({key: getattr(args, key) for key in (*DEFAULTS, "start", "end", "stocks", *PATH_ARGS)}, defaults)
    run_id = _valid_id(args.run_id) if args.run_id else make_run_id(TASK, config["end"])
    folder = artifact_run_dir(run_id, create=False)
    if (folder / "reports" / REPORT).exists():
        raise ValueError(f"运行标识已有结果 {run_id}")
    try:
        result = run(config, run_id, lambda phase, message: _status(run_id, "running", message))
    except Exception as exc:
        _status(run_id, "failed", str(exc))
        traceback.print_exc()
        return 1
    _status(run_id, "completed", "研究运行完成", result["summary"])
    print(json.dumps({"run_id": run_id, "report": str(folder / "reports" / REPORT), **result["summary"]}, ensure_ascii=False))
    return 0


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "KHQuant"

    def _local_request(self) -> bool:
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in allowed:
            self._json(403, {"message": "请从本机工作台地址访问。"})
            return False
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{host}":
            self._json(403, {"message": "请求来源与工作台地址不同。"})
            return False
        return True

    def _send(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status: int, value: dict) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if not self._local_request():
            return
        request = urlsplit(self.path)
        path = request.path
        try:
            if path.startswith("/api/data/"):
                from my_strategy.research_workbench.data_api import handle_get

                if handle_get(self, path):
                    return
            if path.startswith("/api/expert/"):
                from my_strategy.research_workbench.expert_api import handle_get

                if handle_get(self, path, request.query):
                    return
            if path.startswith("/api/stock-chart/"):
                from my_strategy.research_workbench.stock_chart_api import handle_get

                if handle_get(self, path, request.query):
                    return
            if path.startswith("/api/latent/"):
                from my_strategy.research_workbench.latent_api import handle_get

                if handle_get(self, path):
                    return
            if path in STATIC:
                name, kind = STATIC[path]
                self._send(200, (WEB_ROOT / name).read_bytes(), kind)
            elif path == "/api/overview":
                defaults = dict(self.server.defaults)
                for key in PATH_ARGS:
                    defaults.pop(key, None)
                classes = [{"id": "configured", "label": "历史行业与概念资料"}] if self.server.defaults.get("classification_path") else []
                market = _market()
                overview = {
                    "data": market, "defaults": defaults,
                    "classification_options": classes, "classifications": classes,
                    "factors": factor_catalog(WORKBENCH_FEATURE_COLUMNS),
                    "runs": _runs(),
                }
                if self.server.roster is not None:
                    raw = raw_db_path().resolve(strict=True)
                    with sqlite3.connect(raw.as_uri() + "?mode=ro", uri=True) as connection:
                        coverage = target_coverage(connection, self.server.roster, pd.Timestamp(market["latest_date"]))
                    overview["universe"] = {"configured": True, **coverage}
                self._json(200, overview)
            elif path.startswith("/api/runs/"):
                run_id = _valid_id(path.removeprefix("/api/runs/"))
                folder = artifact_run_dir(run_id, create=False)
                result = _read(folder / STATUS_FILE)
                if result["status"] == "completed":
                    result = _read(folder / "reports" / REPORT)
                self._json(200, result)
            else:
                self._json(404, {"message": "页面不存在。"})
        except FileNotFoundError:
            self._json(404, {"message": "运行记录或文件不存在。"})
        except ValueError as exc:
            self._json(400, {"message": str(exc)})

    def do_POST(self) -> None:
        if not self._local_request():
            return
        path = urlsplit(self.path).path
        if path in RETIRED_START_ROUTES:
            self._json(410, {"message": "旧版启动入口已退役。请通过多模态专家或策略回测运行。", "code": "retired_entry"})
            return
        if path.startswith("/api/data/"):
            from my_strategy.research_workbench.data_api import handle_post

            if handle_post(self, path):
                return
        if path.startswith("/api/expert/"):
            from my_strategy.research_workbench.expert_api import handle_post

            if handle_post(self, path):
                return
        self._json(404, {"message": "页面不存在。"})


def _serve(args: argparse.Namespace) -> int:
    market = _market()
    defaults = _config({key: getattr(args, key) for key in (*DEFAULTS, "start", "end", "stocks", *PATH_ARGS)}, _defaults(market))
    with ThreadingHTTPServer(("127.0.0.1", args.port), WorkbenchHandler) as server:
        server.defaults = defaults
        server.roster = load_universe(defaults["universe_path"], defaults["universe_policy"]) if defaults.get("universe_path") else None
        print(f"KHQuant workbench http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            return 0
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="KHQuant local post-close research workbench")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "serve"):
        command = commands.add_parser(name)
        command.add_argument("--start")
        command.add_argument("--end")
        for key, default in DEFAULTS.items():
            command.add_argument(f"--{key.replace('_', '-')}", type=type(default), choices=sorted(CHOICES[key]) if key in CHOICES else None)
        command.add_argument("--stocks", help="Comma-separated canonical stock codes for this pilot.")
        for key in PATH_ARGS:
            command.add_argument(f"--{key.replace('_', '-')}")
    commands.choices["run"].add_argument("--run-id")
    commands.choices["serve"].add_argument("--port", type=int, default=8124)
    args = parser.parse_args()
    try:
        return _serve(args) if args.command == "serve" else _run_command(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
