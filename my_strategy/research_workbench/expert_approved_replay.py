"""Chart only the observations and continuous ledger of an approved forward version."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.latent_quant import forward
from my_strategy.latent_quant.execution import Strategy, StudyCostModel
from my_strategy.model_registry import expert_release
from my_strategy.storage.expert_artifacts import _publish

from . import expert_forward, expert_replay, latent_api, stock_chart_api
from .runner import _write_table, write_json


@dataclass(frozen=True)
class Archive:
    source: Path
    identity: dict
    config: dict
    assessment: dict
    dates: list
    raw: tuple


def _archive(spec: dict) -> Archive:
    release = spec["release"]
    source_id, symbol, cost = (spec[key] for key in ("source_run_id", "symbol", "cost"))
    start, end = (expert_replay._day(spec[key]) for key in ("start", "end"))
    if (release.get("status") != "approved" or release["source_run_id"] != source_id or spec.get("diagnostic")
            or start > end or not stock_chart_api.SYMBOL.fullmatch(symbol) or cost not in {"normal", "double"}):
        raise ValueError("已发布的专家模型或请求区间无效。")
    registry = expert_release.default_registry()
    registered = registry.find_by_name(expert_release.KIND, f"multimodal-expert:{source_id}")
    if (registered is None or registered.stage not in {"champion", "retired"}
            or registered.model_id != spec["release_id"] or registered.model_id != release["release_id"]):
        raise ValueError("找不到该已发布的专家版本。")
    manifest, digest = expert_release._verify_manifest(registered)
    if (digest != release["manifest_sha256"] or digest != registered.params.get("release_manifest_sha256")
            or manifest["arm"] != release["arm"] or manifest["arm"] != spec["arm"]
            or manifest["model_versions"] != release["model_versions"]):
        raise ValueError("批准版本的前瞻证据或策略与提交时不同。")
    source, _, config = expert_replay._source(source_id)
    past = forward._json(source / expert_forward.ACCEPTANCE)
    dates = [day for day in past["sessions"] if start <= day <= end]
    security = stock_chart_api._security(symbol)
    roster = pd.read_parquet(expert_replay._file(source, f"{expert_forward.ROOT}/lock/roster.parquet"))
    outside = symbol not in (roster["stock"].to_numpy() if "stock" in roster else roster.index)
    if dates and dates[0] == start and dates[-1] == end and not outside:
        raw = ([], [])
    else:
        panel, states, *_ = stock_chart_api._warehouse_series(security, start, end, fallback=True)
        raw = stock_chart_api._bars(panel, states, start, end)
    key = {
        "version": "approved-forward-v1", "source_run_id": source_id, "release_id": registered.model_id,
        "approval": "approved", "manifest_sha256": digest, "model_versions": manifest["model_versions"],
        "code_sha256": expert_replay._sha(Path(__file__)), "datasets": {"acceptance": manifest["forward"]["sha256"]},
        "symbol": symbol, "cost": cost, "arm": manifest["arm"], "start": start, "end": end,
        "portfolio_anchor": past["sessions"][0] if dates else None, "market_sha256": stable_hash(raw), "outside_original_pool": outside,
    }
    return Archive(source, key, config, past, dates, raw)


def identity(spec: dict) -> dict:
    return _archive(spec).identity


def _source_tables(archive: Archive, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sessions = archive.dates
    panels, statuses = [], []
    for day in sessions:
        path = archive.source / expert_forward.ROOT / "captures" / day
        record = forward._json(expert_replay._file(path, "manifest.json"))
        if forward._sha(path / "manifest.json") != archive.assessment["capture_manifests"][day]:
            raise ValueError("读取期间前瞻清单发生变化。")
        for role, values in (("panel", panels), ("market_status", statuses)):
            proof = record["files"][role]
            source = expert_replay._file(path, proof["file"])
            if forward._sha(source) != proof["sha256"]:
                raise ValueError("读取期间前瞻市场证据发生变化。")
            values.append(forward._table(source, expert_forward.PANEL_COLUMNS if role == "panel" else forward.FIELDS[role]))
    arm, cost = (archive.identity[key] for key in ("arm", "cost"))
    names = [f"{arm}_predictions.parquet", *(f"{arm}_{cost}_{name}.parquet" for name in latent_api.LEDGERS)]
    destination = root / "backtests/custom"
    for name in names:
        source = expert_replay._file(archive.source, f"{expert_forward.ROOT}/acceptance/{name}")
        if forward._sha(source) != archive.assessment["artifacts"][name]:
            raise ValueError("读取期间前瞻账本发生变化。")
        _publish(source, destination / name)
    daily = pd.read_parquet(destination / f"{arm}_{cost}_daily.parquet")
    if archive.identity["outside_original_pool"]:
        return pd.DataFrame(), pd.DataFrame(), daily
    return pd.concat(panels, ignore_index=True), pd.concat(statuses, ignore_index=True), daily


def run(*, run_id: str, spec: dict) -> Path:
    archive = _archive(spec)
    key = archive.identity
    if spec.get("input_identity", key) != key:
        raise ValueError("发布证据在任务排队后发生变化。")
    root = latent_api._run_folder(run_id)
    if root.name != run_id or (root.exists() and any(path.name != "expert.log" for path in root.iterdir())):
        raise ValueError("回放运行目录已有工件。")
    days = archive.dates
    context = create_run_context(run_id=run_id, task=expert_replay.TASK, as_of_date=days[-1] if days else key["end"],
                                 config=key, data_version=stable_hash(key["datasets"]), seed=archive.config["seed"])
    write_json(root / "status.json", {"status": "running", "phase": "verify_inputs"})
    try:
        raw = archive.raw
        coverage = {"requested_start": key["start"], "requested_end": key["end"],
                    "model_start": days[0] if days else None, "model_end": days[-1] if days else None}
        mode = "expert_portfolio" if days else "expert_market_only"
        replay = {"mode": mode, "run_id": run_id, "approval": "approved", "portfolio_anchor": key["portfolio_anchor"],
                  "source_run_id": archive.source.name, "coverage": coverage}
        warnings = []
        if not days or key["start"] < days[0] or key["end"] > days[-1]:
            warnings.append("策略标记仅来自已核验的前瞻账本。其他日期显示已到仓的本地行情。")
        frame = market = pd.DataFrame()
        equity = cash = window_return = scope = None
        if days:
            frame, market, daily = _source_tables(archive, root)
            if key["outside_original_pool"]:
                warnings.append("本股票不在已批准组合的冻结名录。")
            prior = daily.loc[daily["date"].lt(days[0])]
            initial = archive.config["initial_cash"]
            equity = float(prior.iloc[-1]["equity"]) if not prior.empty else initial
            cash = float(prior.iloc[-1]["cash"]) if not prior.empty else initial
            window_return = float(daily.loc[daily["date"].eq(days[-1]), "equity"].item()) / equity - 1
            scope = {"start": str(daily.iloc[0]["date"]), "end": str(daily.iloc[-1]["date"]), "sessions": len(daily)}
        config = archive.config
        inputs = expert_replay.Inputs(key, {}, (), config, Strategy(**config["portfolio"]),
                                      StudyCostModel(**config["expense_model"]), config["cost_multipliers"][key["cost"]])
        chart = expert_replay._chart(root, inputs, frame, market, raw, replay,
                                     has_ledger=bool(days), has_predictions=bool(days), warnings=warnings)
        if key["arm"] == "domestic_gpu":
            chart["strategies"][0]["label"] = "国内量化专家"
        metrics = archive.assessment["arms"][key["arm"]]["execution"][key["cost"]] if days else None
        if metrics:
            _write_table(root / "reports/metrics.csv", pd.DataFrame([metrics]).drop(columns=["reasons"], errors="ignore"))
        report = {
            "run": context.to_dict(), "source": key, "chart": chart, "mode": mode, "coverage": coverage,
            "window": {"start": key["start"], "end": key["end"], "sessions": len(days), "opening_cash": cash,
                       "opening_equity": equity, "window_return": window_return, "portfolio_anchor": key["portfolio_anchor"]},
            "ledger_scope": scope, "ledger_metrics": metrics,
            "qualification": "已核验前瞻期连续模拟账本，保留原始申报与模拟成交。",
        }
        path = root / "reports" / expert_replay.REPORT
        write_json(path, report)
        write_json(root / "status.json", {"status": "completed", "phase": "complete"})
        return path
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "phase": "evidence", "message": str(error)})
        raise
