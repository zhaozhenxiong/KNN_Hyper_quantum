"""Continuous, time-gated portfolio replay using frozen multimodal score versions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from hashlib import file_digest
from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pyarrow.parquet import read_schema

from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.latent_quant.execution import Strategy, StudyCostModel, execution_evidence_identity, scaled_study_cost, simulate
from my_strategy.latent_quant.runner import INITIAL_CASH, _verify_ledger
from my_strategy.model_registry import expert_fallback
from my_strategy.storage.expert_artifacts import ExpertArtifactStore

from . import expert_stock_inference, latent_api, stock_chart_api, stock_replay_data
from .runner import _write_table, write_json

TASK = "expert-stock-portfolio-replay"
REPORT = "expert_stock_replay.json"
SOURCE_REPORT = "expert_research.json"
SHANGHAI = ZoneInfo("Asia/Shanghai")
PANEL_FILES = ("panel.parquet", "benchmark.parquet", "roster.parquet", "market_status.parquet")
REQUIRED_COLUMNS = {
    "panel.parquet": {"date", "stock", "signal_eligible", "trade_open", "trade_high", "trade_low", "trade_close", "volume", "source"},
    "benchmark.parquet": {"date", "open", "close"},
    "roster.parquet": {"stock", "delist_date"},
    "market_status.parquet": {"date", "stock", "market_status", "source"},
}
QUANTILES = ["q10", "q50", "q90", "score"]
HYBRID_SCORE_COLUMNS = [
    "hybrid_score", "score", "knn_q10", "knn_q50", "knn_q90",
    "knn_raw_q10", "knn_raw_q50", "knn_raw_q90",
    "multimodal_q10", "multimodal_q50", "multimodal_q90",
    "multimodal_raw_q10", "multimodal_raw_q50", "multimodal_raw_q90",
    "knn_rank", "multimodal_rank", "effective_knn_weight", "effective_multimodal_weight",
    "fallback_reason",
]


@dataclass(frozen=True)
class Version:
    path: Path
    fold: str
    start: str
    end: str
    last_fit_label: str
    available_at: str | None
    predictions: Path


@dataclass(frozen=True)
class Inputs:
    identity: dict
    files: dict[str, Path]
    versions: tuple[Version, ...]
    config: dict
    strategy: Strategy
    cost_model: StudyCostModel
    cost_multiplier: float


@dataclass(frozen=True)
class Timeline:
    benchmark: pd.DataFrame
    sessions: pd.DatetimeIndex
    requested: pd.DatetimeIndex
    assignment: dict[pd.Timestamp, Version]

    @property
    def anchor(self) -> str | None:
        return str(self.sessions[0].date()) if len(self.sessions) else None


def _file(folder: Path, relative: str) -> Path:
    expected = folder / relative
    found = latent_api._safe(folder, expected)
    if found != expected or not found.is_file():
        raise ValueError(f"缺少可信冻结工件 {relative}。")
    return found


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def _day(value: str) -> str:
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("回放日期必须使用 YYYY-MM-DD。")
    return value


def _report_path(folder: Path) -> Path:
    standard = folder / f"reports/{SOURCE_REPORT}"
    if standard.is_file() or not (folder / expert_fallback.MANIFEST).is_file():
        return _file(folder, f"reports/{SOURCE_REPORT}")
    return _file(folder, expert_fallback.REPORT)


def _source(run_id: str) -> tuple[Path, dict, dict]:
    folder = latent_api._run_folder(run_id)
    status = latent_api._read(_file(folder, "status.json"))
    report = latent_api._read(_report_path(folder))
    if (folder / expert_fallback.MANIFEST).is_file():
        config, _ = expert_fallback.replay_config(folder)
    else:
        config = latent_api._read(_file(folder, "config/study.json"))
    if folder.name != run_id or status.get("status") != "completed" or report.get("run", {}).get("run_id") != run_id:
        raise ValueError("选中来源没有完成并带运行记录的多模态研究。")
    return folder, report, config


def _fit_end(window: dict) -> str:
    models = window.get("models", {})
    parts = ("train", "calibration") if "validation" not in window and "validation" not in models else (
        "train", "validation", "calibration",
    )
    stamps = [models.get(part, {}).get("last_label_end") or window.get(part, {}).get("last_label_end")
              for part in parts]
    if not all(stamps):
        raise ValueError(f"{window.get('period')} 缺少训练、验证或校准标签截止证据。")
    return max(_day(stamp) for stamp in stamps)


def _reported_folds(report: dict) -> list[dict]:
    review = report.get("historical_review")
    return [*report["folds"], review] if review else report["folds"]


def _content_files(folder: Path, config: dict, filenames: tuple) -> tuple[dict[str, Path], dict[str, str]]:
    store = ExpertArtifactStore()
    manifest = store.read_manifest(folder.name)
    metadata = manifest["metadata"]
    task = metadata.get("task")
    expected_source = {
        "latent-expert-research": config.get("source_run_id"),
        "expert-online-knn": config.get("source_run_id"),
        "expert-knn60-multimodal40": config.get("source_run_id"),
        "expert-knn-calibration-replay": config.get("model_source_run_id"),
    }.get(task)
    if expected_source is None or metadata.get("source_run_id") != expected_source:
        raise ValueError("研究工件清单与冻结研究的运行身份不一致。")
    files, digests = {}, {}
    for filename in filenames:
        kind = Path(filename).stem
        ref = manifest["files"].get(f"frozen_inputs/{kind}/{filename}")
        if ref is None or ref != config["sources"].get(kind):
            raise ValueError(f"研究工件清单与配置的冻结输入不一致：{filename}。")
        files[filename], digests[filename] = store.verify(ref), ref["sha256"]
    return files, digests


def _data_files(folder: Path, config: dict) -> tuple[dict[str, Path], dict[str, str]]:
    if config.get("data_source_run_id"):
        source, _, source_config = _source(config["data_source_run_id"])
        return _data_files(source, source_config)
    if config.get("model_spec") is not None:
        files, digests = _content_files(folder, config, PANEL_FILES)
        for name, path in files.items():
            if not REQUIRED_COLUMNS[name].issubset(read_schema(path).names):
                raise ValueError(f"冻结输入缺少回放字段：{name}。")
        return files, digests
    coverage = latent_api._read(_file(folder, "data_snapshots/coverage.json"))
    source = coverage.get("source_run_id")
    if not source or config.get("source_run_id") != source:
        raise ValueError("研究报告未明确绑定同一冻结面板。")
    root = latent_api._run_folder(source)
    if (root.name != source or latent_api._profile(root) != "baseline"
            or latent_api._worker_status(source).get("status") != "completed"):
        raise ValueError("指定的基线快照不存在或没有完成。")
    files = {name: _file(root, f"data_snapshots/{name}") for name in PANEL_FILES}
    digests = {name: _sha(file) for name, file in files.items()}
    pins = coverage.get("frozen_input_sha256", {})
    if any(pins.get(name) != digest for name, digest in digests.items()):
        raise ValueError("行情、交易日、状态或股票池与冻结版本不一致。")
    return files, digests


def _cutoff(stamp: str) -> datetime:
    value = datetime.fromisoformat(stamp)
    if value.tzinfo is None:
        raise ValueError("模型发布时间需要时区。")
    return value.astimezone(SHANGHAI)


def _inputs(spec: dict) -> Inputs:
    if spec.get("sector_provider", "capco") not in {"capco", "ths_fuyao"}:
        raise ValueError("板块数据提供方无效。")
    symbol = spec["symbol"]
    if not stock_chart_api.SYMBOL.fullmatch(symbol):
        raise ValueError("请输入标准股票代码。")
    start, end = _day(spec["start"]), _day(spec["end"])
    if start > end:
        raise ValueError("开始日期晚于结束日期。")
    release, arm, cost = spec["release"], spec["arm"], spec["cost"]
    approved = release.get("status") == "approved"
    fallback = release.get("status") == "production_fallback"
    pinned_fallback = fallback and release.get("production_source") != "research_candidate"
    if not approved and not fallback and not (release.get("status") == "candidate" and spec.get("diagnostic") is True):
        raise ValueError("暂无已批准发布版本。候选研究仅在明确的诊断模式下可复核。")
    if (str(release.get("release_id")) != spec["release_id"] or release.get("source_run_id") != spec["source_run_id"]
            or release.get("arm") != arm or not re.fullmatch(r"[a-z][a-z0-9_]*", arm) or cost not in {"normal", "double"}):
        raise ValueError("回放请求与批准版本或费用档不一致。")
    if approved and not re.fullmatch(r"[a-f0-9]{64}", release.get("manifest_sha256", "")):
        raise ValueError("正式版本缺少冻结发布清单的哈希。")
    source, report, config = _source(spec["source_run_id"])
    if pinned_fallback:
        proof = config.get("_fallback_release")
        if not proof or any(proof.get(key) != release.get(key) for key in (
            "status", "release_id", "source_run_id", "arm", "research_report_sha256", "fallback_manifest_sha256",
        )):
            raise ValueError("生产回退请求与冻结 KNN 清单不一致。")
    files, dataset = _data_files(source, config)
    defaults = asdict(StudyCostModel())
    expense = config["expense_model"]
    if (config["initial_cash"] != INITIAL_CASH or set(expense) != set(defaults)
            or any(type(value) not in (int, float) or not np.isfinite(value) or value < 0 for value in expense.values())
            or (approved and expense != defaults)):
        raise ValueError("归档与当前已校验的模拟账户初始资金或费用口径不一致。")
    cost_model = StudyCostModel(**expense)
    strategy = Strategy(**config["portfolio"])
    multiplier = float(config["cost_multipliers"][cost])
    if multiplier != {"normal": 1.0, "double": 2.0}[cost]:
        raise ValueError("来源版本的费用压力档与本地执行器不一致。")
    cost_rates = asdict(scaled_study_cost(multiplier, cost_model))
    if config.get("execution_costs", {}).get(cost, cost_rates) != cost_rates:
        raise ValueError("归档公布的费用档无法从冻结费用配置复算。")
    manifests = release.get("model_versions", [])
    if not manifests and not approved:
        manifests = [{
            "run_id": source.name, "fold": fold["period"], "valid_from": fold["evaluation_start"][:10],
            "valid_through": fold["evaluation_end"][:10], "training_max_label_date": _fit_end(fold),
        } for fold in _reported_folds(report)]
    if not manifests:
        raise ValueError("正式版本未提供历史模型的逐版本可用时间和适用区间。")

    versions = []
    fingerprints = {}
    for entry in manifests:
        first, last = _day(entry["valid_from"]), _day(entry["valid_through"])
        if first > last:
            raise ValueError("历史模型版本的适用区间无效。")
        if first > end:
            continue
        fold = entry["fold"]
        if fold not in latent_api.FOLDS:
            raise ValueError("未知历史模型折。")
        folder, model_report, model_config = _source(entry["run_id"])
        _, model_dataset = _data_files(folder, model_config)
        if model_dataset != dataset or any(model_config[key] != config[key] for key in (
            "portfolio", "initial_cash", "expense_model", "cost_multipliers",
        )):
            raise ValueError("区间内的模型没有共同冻结股票池、日历或统一仓位口径。")
        model_fold = next((item for item in _reported_folds(model_report) if item["period"] == fold), None)
        if not model_fold or arm not in model_fold["arms"] or cost not in model_fold["arms"][arm]["execution"]:
            raise ValueError(f"{folder.name}/{fold} 缺少发布策略的冻结分数或成本证据。")
        fit_end = _fit_end(model_fold)
        if (_day(entry["training_max_label_date"]) != fit_end
                or first < model_fold["evaluation_start"][:10] or last > model_fold["evaluation_end"][:10]
                or fit_end >= first):
            raise ValueError(f"{folder.name}/{fold} 的拟合标签或日期越过可用于回放的边界。")
        timestamp = entry.get("available_at")
        if approved and not timestamp:
            raise ValueError(f"{folder.name}/{fold} 缺少实际发布时间。")
        if timestamp and _cutoff(timestamp) > datetime.combine(date.fromisoformat(first), time(18), tzinfo=SHANGHAI):
            raise ValueError(f"{folder.name}/{fold} 在首个适用信号日尚未发布。")
        predictions = _file(folder, f"backtests/{fold}/{arm}_predictions.parquet")
        versions.append(Version(folder, fold, first, last, fit_end, timestamp, predictions))
        key = f"{folder.name}/{fold}"
        fingerprints[key] = {
            "config_sha256": _sha(_file(folder, "config/study.json")),
            "report_sha256": _sha(_report_path(folder)),
            "predictions_sha256": _sha(predictions),
        }
    identity = {
        "version": 4, "sector_provider": spec.get("sector_provider", "capco"), "code_sha256": {path.name: _sha(path) for path in (
            Path(__file__), Path(expert_stock_inference.__file__), Path(stock_replay_data.__file__),
            Path(expert_stock_inference.expert_research.__file__), Path(expert_stock_inference.models.__file__),
            Path(expert_stock_inference.models.__file__).with_name("execution.py"),
        )} | ({
            "expert_fallback.py": _sha(Path(expert_fallback.__file__)),
            "signal_experts.py": _sha(Path(expert_fallback.__file__).parents[1] / "latent_quant" / "signal_experts.py"),
            "knn_backend.py": _sha(Path(expert_fallback.__file__).parents[1] / "latent_quant" / "knn_backend.py"),
        } | ({
            "expert_hybrid_study.py": _sha(Path(__file__).with_name("expert_hybrid_study.py")),
        } if arm == expert_fallback.HYBRID_ARM else {}) if pinned_fallback else {}), "symbol": symbol, "execution_costs": cost_rates,
        "start": start, "end": end, "cost": cost, "arm": arm,
        "source_run_id": source.name, "release_id": spec["release_id"], "approval": release["status"],
        "manifest_sha256": release.get("manifest_sha256"), "fallback_manifest_sha256": release.get("fallback_manifest_sha256"),
        "selection_policy": asdict(strategy), "selection_evidence": release.get("selection_evidence"),
        "datasets": dataset,
        "execution_evidence": execution_evidence_identity(),
        "model_versions": [
            {"run_id": model.path.name, "fold": model.fold, "valid_from": model.start, "valid_through": model.end,
             "training_max_label_date": model.last_fit_label, "available_at": model.available_at}
            for model in versions
        ], "source_files": fingerprints,
    }
    return Inputs(identity, files, tuple(versions), config, strategy, cost_model, multiplier)


def replay_identity(spec: dict) -> dict:
    """Fingerprint release, model scores, selection dates and the full frozen tape."""
    if spec.get("scope") in {"watchlist", "single_stock"}:
        from .expert_personal_replay import replay_identity as personal_identity

        return personal_identity(spec)
    if _is_forward_release(spec):
        from .expert_approved_replay import identity

        return identity(spec)
    inputs = _inputs(spec)
    timeline = _timeline(inputs)
    roster = _roster(inputs)
    market = _extra_market(inputs, timeline, roster)
    identity, _ = _identity(inputs, timeline, roster, market)
    return identity


def _is_forward_release(spec: dict) -> bool:
    release = spec.get("release", {})
    return release.get("status") == "approved" and any(
        version.get("fold") == "expert_forward" for version in release.get("model_versions", [])
    )


def _timeline(inputs: Inputs) -> Timeline:
    start, end = (inputs.identity[key] for key in ("start", "end"))
    bench = pd.read_parquet(inputs.files["benchmark.parquet"]).set_index("date")
    bench.index = pd.DatetimeIndex(bench.index)
    if not bench.index.is_unique or not bench.index.is_monotonic_increasing:
        raise ValueError("冻结交易日历不唯一或未排序。")
    approved = inputs.identity["approval"] == "approved"
    if approved and (start < str(bench.index[0].date()) or end > str(bench.index[-1].date())):
        raise ValueError("自选区间超出已有验证的行情日期。")
    requested = bench.loc[start:end].index
    if approved and requested.empty:
        raise ValueError("自选区间没有真实交易日。")
    history = bench.loc[:end].index
    assignment: dict[pd.Timestamp, Version] = {}
    for model in inputs.versions:
        for session in history[history.to_series().between(model.start, model.end).to_numpy()]:
            if session in assignment:
                raise ValueError("同一历史日期存在两个相互重叠的模型版本。")
            assignment[session] = model
    covered = requested[requested.isin(assignment)]
    if approved:
        required = requested
    elif len(covered):
        required = requested[(requested >= covered[0]) & (requested <= covered[-1])]
    else:
        return Timeline(bench, covered, covered, {})
    if missing := [str(day.date()) for day in required if day not in assignment]:
        raise ValueError(f"{missing[0]} 至 {missing[-1]} 有 {len(missing)} 个交易日缺少当时可用的冻结模型分数。")
    anchor_index = history.get_loc(covered[0])
    while anchor_index and history[anchor_index - 1] in assignment:
        anchor_index -= 1
    sessions = history[anchor_index:history.get_loc(covered[-1]) + 1]
    if approved and len(sessions) < 2:
        raise ValueError("选中日是组合回放起点，至少需要包含该起点及后续一个交易日的范围。")
    if not bench.loc[sessions, ["open", "close"]].gt(0).all(axis=None):
        raise ValueError("自选区间及组合起点需要完整开盘和收盘基准。")
    return Timeline(bench, sessions, covered, {date: assignment[date] for date in sessions})


def _roster(inputs: Inputs) -> pd.DataFrame:
    roster = pd.read_parquet(inputs.files["roster.parquet"])
    if "stock" in roster:
        roster = roster.set_index("stock")
    if not roster.index.is_unique or len(roster) != int(inputs.config.get("stock_cap", 120)):
        raise ValueError("回放必须使用共同冻结名录中的完整股票池。")
    return roster


def _extra_market(inputs: Inputs, timeline: Timeline, roster: pd.DataFrame) -> tuple[list, list]:
    symbol, start, end = (inputs.identity[key] for key in ("symbol", "start", "end"))
    outside = symbol not in roster.index
    if (not outside and inputs.identity["approval"] == "approved") or (
        not outside and len(timeline.requested)
        and timeline.requested[0] == pd.Timestamp(start) and timeline.requested[-1] == pd.Timestamp(end)
    ):
        return [], []
    identity = stock_chart_api._security(symbol)
    frame, market, *_ = stock_chart_api._warehouse_series(identity, start, end, fallback=True)
    return stock_chart_api._bars(frame, market, start, end)


def _identity(inputs: Inputs, timeline: Timeline, roster: pd.DataFrame, market: tuple[list, list]) -> tuple[dict, dict]:
    outside = inputs.identity["symbol"] not in roster.index
    native = outside and inputs.identity["approval"] == "candidate"
    archives, hashes = {}, {}
    if native:
        for version in set(timeline.assignment.values()):
            _, report, config = _source(version.path.name)
            fold_report = next(row for row in _reported_folds(report) if row["period"] == version.fold)
            key = f"{version.path.name}/{version.fold}"
            try:
                archives[version], hashes[key] = expert_stock_inference.archive_files(
                    version.path, version.fold, fold_report, config, _file, _sha,
                    selected_arm=inputs.identity["arm"],
                    macro_resolver=_content_files if config.get("model_spec") is not None else None,
                )
            except (KeyError, OSError, ValueError) as error:
                hashes[key] = {"unavailable": str(error)}
    return {**inputs.identity, "portfolio_anchor": timeline.anchor, "outside_original_pool": outside,
            "market_sha256": stable_hash(market), "native_experts": hashes,
            "local_stock_input_revision": expert_stock_inference.stock_input_revision() if native and hashes else None}, archives


def _tape(
    inputs: Inputs, timeline: Timeline | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, pd.Series, pd.DataFrame]:
    timeline = timeline or _timeline(inputs)
    sessions = timeline.sessions
    frame = pd.read_parquet(inputs.files["panel.parquet"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.loc[frame["date"].isin(sessions)].reset_index(drop=True)
    if frame.duplicated(["date", "stock"]).any():
        raise ValueError("冻结面板的股票日期键重复。")
    roster = _roster(inputs)
    if not frame["stock"].isin(roster.index).all():
        raise ValueError("回放必须使用共同冻结名录中的完整股票池。")
    settled = []
    for model in inputs.versions:
        dates = {session for session, active in timeline.assignment.items() if active == model}
        if not dates:
            continue
        hybrid = inputs.identity["arm"] == expert_fallback.HYBRID_ARM
        if hybrid:
            available = read_schema(model.predictions).names
            required = {"date", "stock", "score", "hybrid_score", "knn_rank", "multimodal_rank",
                        "effective_knn_weight", "effective_multimodal_weight", "fallback_reason"}
            if not required.issubset(available):
                raise ValueError("混合回放缺少原生排名、实际权重或混合决策分。")
            columns = ["date", "stock", *(name for name in HYBRID_SCORE_COLUMNS if name in available)]
        else:
            columns = ["date", "stock", *QUANTILES]
        selected = pd.read_parquet(model.predictions, columns=columns)
        selected["date"] = pd.to_datetime(selected["date"])
        selected = selected.loc[selected["date"].isin(dates)].assign(
            model_run_id=model.path.name, model_fold=model.fold,
        )
        settled.append(selected)
    scores = pd.concat(settled, ignore_index=True)
    keys = ["date", "stock"]
    expected = pd.MultiIndex.from_frame(frame.loc[frame["signal_eligible"].eq(True), keys])
    provided = pd.MultiIndex.from_frame(scores[keys])
    covered = (not provided.is_unique or set(expected) != set(provided)
               or not scores["date"].nunique() == len(sessions))
    if covered:
        raise ValueError("冻结分数不覆盖区间内的同一股票池与合格股票日。")
    if inputs.identity["arm"] == expert_fallback.HYBRID_ARM:
        knn_weight = scores["effective_knn_weight"].to_numpy(dtype=float)
        multi_weight = scores["effective_multimodal_weight"].to_numpy(dtype=float)
        score = scores["score"].to_numpy(dtype=float)
        hybrid_score = scores["hybrid_score"].to_numpy(dtype=float)
        valid = knn_weight > 0
        complete = (knn_weight == 0.6) & (multi_weight == 0.4)
        fallback = (knn_weight == 1.0) & (multi_weight == 0.0)
        invalid = (knn_weight == 0.0) & (multi_weight == 0.0)
        expected_score = np.where(
            complete,
            0.6 * scores["knn_rank"].to_numpy(dtype=float) + 0.4 * scores["multimodal_rank"].to_numpy(dtype=float),
            scores["knn_rank"].to_numpy(dtype=float),
        )
        if not ((complete | fallback | invalid).all()
                and np.isfinite(score[valid]).all() and np.isfinite(hybrid_score[valid]).all()
                and np.isnan(score[~valid]).all() and np.isnan(hybrid_score[~valid]).all()
                and np.allclose(score[valid], hybrid_score[valid], rtol=0, atol=1e-12)
                and np.allclose(score[valid], expected_score[valid], rtol=0, atol=1e-12)):
            raise ValueError("混合回放的决策分、60/40 权重或 KNN 兜底证据无效。")
    elif not np.isfinite(scores[QUANTILES]).all(axis=None):
        raise ValueError("冻结分数不覆盖区间内的同一股票池与合格股票日。")
    index = pd.MultiIndex.from_frame(frame[keys])
    benchmark = timeline.benchmark["close"].pct_change(fill_method=None).reindex(sessions)
    status = pd.read_parquet(inputs.files["market_status.parquet"])
    status = status.loc[pd.to_datetime(status["date"]).isin(sessions)].copy()
    return frame, roster, status, sessions, benchmark, scores.set_index(keys).reindex(index).reset_index()


def _single_stock(inputs: Inputs, timeline: Timeline, archives: dict, pool: pd.DataFrame) -> tuple:
    symbol = inputs.identity["symbol"]
    end = str(timeline.sessions[-1].date())
    data = stock_replay_data.load_stock_replay_data(symbol, timeline.anchor, end)
    if not data.sessions.equals(timeline.sessions) or not np.array_equal(
        data.benchmark_close.reindex(timeline.sessions).to_numpy(dtype=float),
        timeline.benchmark.loc[timeline.sessions, "close"].to_numpy(dtype=float),
    ):
        raise ValueError("独立回放的核验日历或收盘基准与归档不一致。")
    training = pd.read_parquet(inputs.files["panel.parquet"])
    forecasts, devices = [], {}
    for version, archive in archives.items():
        dates = [day for day, active in timeline.assignment.items() if active == version]
        rows = data.frame.loc[data.frame["date"].isin(dates) & data.frame["signal_eligible"].eq(True)]
        if rows.empty:
            continue
        predicted, audit = expert_stock_inference.forecast(archive, training, rows)
        forecasts.append(rows[["date", "stock"]].join(predicted).assign(model_run_id=version.path.name, model_fold=version.fold))
        devices[version.fold] = audit
    if not forecasts:
        raise ValueError("该股票在模型覆盖期内没有通过预热的可核验特征。")
    ready = pd.concat(forecasts).sort_index()
    ranked = expert_stock_inference.original_pool_rank(
        ready, pool, top_n=inputs.strategy.top_n, min_score=inputs.strategy.min_score,
    )
    visible = ranked.loc[ranked["date"].isin(timeline.requested)]
    qualified = visible["execution_score"].gt(inputs.strategy.min_score)
    coverage = {
        "input_version": data.coverage["input_version"], "inputs": data.coverage["inputs"],
        "eligible_first_session": str(visible["date"].min().date()) if len(visible) else None,
        "eligible_last_session": str(visible["date"].max().date()) if len(visible) else None,
        "eligible_rows": len(visible), "rank_qualified_rows": int(qualified.sum()), "anchor_eligible_rows": len(ranked),
        "ranking": "same-day original pool calibrated scores, ascending symbol on ties, before industry and board limits",
        "top_n": inputs.strategy.top_n,
    }
    return data.frame, data.roster, data.market_status, data.benchmark_returns, ranked.reindex(data.frame.index), coverage, devices


def _chart(root: Path, inputs: Inputs, frame: pd.DataFrame, market: pd.DataFrame, extra: tuple[list, list],
           replay: dict, *, has_ledger: bool, has_predictions: bool, warnings: list[str]) -> dict:
    symbol, start, end, cost, arm = (inputs.identity[key] for key in ("symbol", "start", "end", "cost", "arm"))
    identity = stock_chart_api._security(symbol)
    chosen = frame.loc[frame["stock"].eq(symbol)] if "stock" in frame else frame
    market = market.loc[market["stock"].eq(symbol)] if "stock" in market else market
    frozen_bars, frozen_status = stock_chart_api._bars(chosen, market, start, end)
    bars = sorted({row["time"]: row for row in (*extra[0], *frozen_bars)}.values(), key=lambda row: row["time"])
    observed = [bar["time"] for bar in bars if bar["close"] is not None]
    observed_dates = set(observed)
    nontrading = sorted({row["time"]: row for row in (*extra[1], *frozen_status)
                         if row["time"] not in observed_dates}.values(), key=lambda row: row["time"])
    replay["coverage"].update(market_start=observed[0] if observed else None, market_end=observed[-1] if observed else None)
    prefix = f"/api/expert/replays/{root.name}"
    events = stock_chart_api._event_rows(
        root, arm, "custom", cost, symbol, start, end, download_prefix=f"{prefix}/downloads/",
    ) if has_ledger else []
    standalone = replay["mode"] == "expert_single_stock"
    unavailable = replay["mode"] == "expert_market_only" or chosen.empty
    from . import stock_chart_research

    return stock_chart_research.attach({
        **identity,
        "context": {"run_id": root.name, "profile": "expert", "fold": None, "cost": cost, "start": start, "end": end,
                    "price_basis": "raw", "price_source": "frozen_and_local_raw" if extra[0] else "frozen_raw",
                    "coverage_start": replay["coverage"]["model_start"], "coverage_end": replay["coverage"]["model_end"],
                    "replay": replay},
        "bars": bars, "nontrading": nontrading,
        "strategies": [{"id": arm, "label": ({
                            "online_knn_volume_trend": "纯在线 KNN",
                            expert_fallback.HYBRID_ARM: "KNN 60% + 多模态 40%（混合决策分）",
                        }.get(arm, "多模态专家")) + (" · 池外单股诊断" if standalone else "组合"),
                        "status": "not_covered" if unavailable else "replayed", "events": events,
                        "positions": stock_chart_api._positions(root, arm, "custom", cost, symbol, start, end) if has_ledger else [],
                        "predictions": stock_chart_api._predictions(
                            root, "baseline", arm, "custom", symbol, start, end, replay=True,
                        ) if has_predictions else []}],
        "warnings": warnings,
        "sources": [{"kind": replay["mode"], "label": "独立单股模拟" if standalone else "区间复核",
                     "run_id": root.name, "report_url": prefix}],
    })


def run(*, run_id: str, spec: dict) -> Path:
    """Replay an approved account or preserve candidate price and model coverage separately."""
    if spec.get("scope") in {"watchlist", "single_stock"}:
        from .expert_personal_replay import run as run_personal

        return run_personal(run_id=run_id, spec=spec)
    if _is_forward_release(spec):
        from .expert_approved_replay import run as run_approved

        return run_approved(run_id=run_id, spec=spec)
    root = latent_api._run_folder(run_id)
    if root.name != run_id:
        raise ValueError("回放目录身份不一致。")
    root.mkdir(parents=True, exist_ok=True)
    if any(path.name != "expert.log" for path in root.iterdir()):
        raise ValueError("回放运行目录已有工件。")
    try:
        write_json(root / "status.json", {"status": "running", "phase": "verify_inputs"})
        inputs = _inputs(spec)
        timeline = _timeline(inputs)
        roster = _roster(inputs)
        extra = _extra_market(inputs, timeline, roster)
        identity, archives = _identity(inputs, timeline, roster, extra)
        if spec.get("input_identity") is not None and spec["input_identity"] != identity:
            raise ValueError("冻结输入在任务提交后发生变化。")
        sessions = timeline.sessions
        requested = timeline.requested
        start, end, arm, cost = (identity[key] for key in ("start", "end", "arm", "cost"))
        coverage = {"requested_start": start, "requested_end": end,
                    "model_start": str(requested[0].date()) if len(requested) else None,
                    "model_end": str(requested[-1].date()) if len(requested) else None}
        replay = {
            "mode": "expert_portfolio", "run_id": run_id, "approval": identity["approval"],
            "portfolio_anchor": timeline.anchor, "source_run_id": identity["source_run_id"], "coverage": coverage,
        }
        warnings = []
        if not len(sessions):
            replay["mode"] = "expert_market_only"
            warnings.append("所选日期暂无可用的冻结专家模型。")
        elif coverage["model_start"] > start or coverage["model_end"] < end:
            warnings.append(f"可回放的冻结模型覆盖 {coverage['model_start']} 至 {coverage['model_end']}；其余日期按本地日线覆盖显示。")
        frame = market = predictions = pd.DataFrame()
        outside = identity["outside_original_pool"]
        if len(sessions):
            frame, roster, market, sessions, benchmark, predictions = _tape(inputs, timeline)
            predictions.index = frame.index
        if len(sessions) and outside and identity["approval"] == "candidate":
            frame = market = pd.DataFrame()
            try:
                missing = [record["unavailable"] for record in identity["native_experts"].values() if "unavailable" in record]
                if missing:
                    raise ValueError(missing[0])
                frame, roster, market, benchmark, predictions, details, audit = _single_stock(
                    inputs, timeline, archives, predictions,
                )
                coverage.update(details)
                effective = {}
                for fold, experts in audit.items():
                    for expert, statistics in experts.items():
                        used = [device for device, calls in statistics["calls"].items() if calls]
                        if used:
                            effective[f"{fold}/{expert}"] = " + ".join(used)
                replay.update(mode="expert_single_stock", devices={"requested_device": "GPU", "effective_devices": effective, "audit": audit})
                warnings.append("该股未进入原组合。独立账户按同日冻结池分数的排名门槛模拟，仓位、现金和成交独立计算。")
                warnings.append("池外特征根据可核验的本地历史行情重建；供应商当时版本和发布时间仍未核验。")
            except (OSError, ValueError) as error:
                replay["mode"] = "expert_market_only"
                predictions = pd.DataFrame()
                warnings.append(f"池外单股诊断缺少可核验输入。{error}")
        elif outside and len(sessions):
            warnings.append("该股票未进入发布组合，行情仍可查阅。")
        if replay_identity(spec) != identity:
            raise ValueError("冻结输入在读取时发生变化。")
        context = create_run_context(
            task=TASK, run_id=run_id, as_of_date=coverage["model_end"] or end, config=identity,
            data_version=stable_hash({"frozen": identity["datasets"], "market": identity["market_sha256"],
                                      "independent": coverage.get("input_version")}), seed=inputs.config.get("seed", 42),
        )
        has_predictions = not predictions.empty
        has_ledger = has_predictions and len(sessions) > 1
        validation = ledger_metrics = ledger_scope = opening_cash = opening_equity = window_return = None
        if has_predictions:
            fold = context.subdir("backtests", "custom")
            active = predictions.loc[predictions["score"].notna()].copy()
            active["label_end"] = pd.NaT
            active["target"] = float("nan")
            _write_table(fold / f"{arm}_predictions.parquet", active)
        if has_ledger:
            write_json(root / "status.json", {"status": "running", "phase": "portfolio"})
            score = "execution_score" if replay["mode"] == "expert_single_stock" else "score"
            outcome = simulate(
                frame, predictions[score], sessions, benchmark, run_id=run_id, strategy=inputs.strategy,
                initial_cash=inputs.config["initial_cash"], cost_multiplier=inputs.cost_multiplier,
                market_status=market, roster=roster, base_cost_model=inputs.cost_model,
            )
            validation = _verify_ledger(outcome, sessions)
            prior_day = outcome["daily"].loc[outcome["daily"]["date"].lt(coverage["model_start"])]
            opening_cash = float(prior_day["cash"].iloc[-1]) if len(prior_day) else inputs.config["initial_cash"]
            opening_equity = float(prior_day["equity"].iloc[-1]) if len(prior_day) else inputs.config["initial_cash"]
            window_return = float(outcome["daily"]["equity"].iloc[-1]) / opening_equity - 1
            for name in (*latent_api.LEDGERS, "selection_candidates", "sector_candidate_daily"):
                _write_table(fold / f"{arm}_{cost}_{name}.parquet", outcome[name])
            ledger_metrics = outcome["metrics"]
            ledger_scope = {"start": timeline.anchor, "end": coverage["model_end"], "sessions": len(sessions)}
            metrics = pd.DataFrame([ledger_metrics]).drop(columns=["reasons"]).assign(
                ledger_start=timeline.anchor, ledger_end=coverage["model_end"], window_start=start, window_return=window_return,
                opening_cash=opening_cash, opening_equity=opening_equity,
            )
            _write_table(context.subdir("reports") / "metrics.csv", metrics)
        elif has_predictions:
            warnings.append("当前仅一个可用交易日，保留预测，等待下一有证据的交易日才有模拟成交。")
        sector_report = None
        if has_ledger:
            from .expert_sector import write_sector_artifacts

            market_panel = pd.read_parquet(inputs.files["panel.parquet"])
            market_panel["date"] = pd.to_datetime(market_panel["date"], errors="raise")
            market_panel = market_panel.loc[market_panel["date"].le(sessions[-1])]
            frozen_benchmark = pd.read_parquet(inputs.files["benchmark.parquet"])
            frozen_benchmark["date"] = pd.to_datetime(frozen_benchmark["date"], errors="raise")
            sector_report = write_sector_artifacts(
                root, market_panel, frozen_benchmark.set_index("date")["close"],
                outcome["selection_candidates"], outcome["sector_candidate_daily"],
                provider=inputs.identity.get("sector_provider", "capco"),
            )
        stores_content = inputs.config.get("model_spec") is not None
        write_json(context.subdir("data_snapshots") / "source_references.json", {
            name: {
                "source_run_id": identity["source_run_id"] if stores_content else inputs.config["source_run_id"],
                "path": f"frozen_inputs/{Path(name).stem}/{name}" if stores_content else f"data_snapshots/{name}",
                "sha256": identity["datasets"][name],
                "storage": "expert_content" if stores_content else "run_snapshot",
            }
            for name in inputs.files
        })
        report = {
            "mode": replay["mode"], "run": context.to_dict(), "source": identity, "coverage": coverage,
            "window": {"start": start, "end": end,
                       "sessions": len(requested), "stock_count": len(roster) if has_predictions else 0, "portfolio_anchor": timeline.anchor,
                       "opening_cash": opening_cash, "opening_equity": opening_equity, "window_return": window_return},
            "ledger_metrics": ledger_metrics, "verification": validation, "ledger_scope": ledger_scope,
            "sector_resonance": sector_report,
            "qualification": {
                "expert_single_stock": "冻结专家版本和同日原组合排名门槛下的独立单股研究模拟。",
                "expert_portfolio": "同一冻结股票池与仓位规则下的连续组合模拟，视图按所选股票筛选。",
                "expert_market_only": "本地原始日线及已核验交易状态。",
            }[replay["mode"]],
            "chart": _chart(root, inputs, frame, market, extra, replay, has_ledger=has_ledger,
                            has_predictions=has_predictions, warnings=warnings),
        }
        path = context.subdir("reports") / REPORT
        write_json(path, report)
        write_json(root / "status.json", {"status": "completed", "phase": "complete"})
        return path
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "phase": "evidence", "message": str(error)})
        raise


def download(handler: Any, run_id: str, file_id: str) -> None:
    root = latent_api._run_folder(run_id)
    try:
        status = latent_api._read(_file(root, "status.json"))
        report = latent_api._read(_file(root, f"reports/{REPORT}"))
        if report.get("mode") in {"expert_watchlist", "expert_independent_stock"}:
            if (root.name != run_id or status.get("status") != "completed"
                    or report["run"]["run_id"] != run_id or report["run"]["task"] != TASK):
                raise ValueError("回放未完成")
            from .expert_personal_replay import authorized_download

            latent_api._download(handler, _file(root, authorized_download(report, file_id)))
            return
        source = report["source"]
        if (root.name != run_id or status.get("status") != "completed" or report["mode"] not in ("expert_portfolio", "expert_single_stock")
                or report["run"]["run_id"] != run_id or report["run"]["task"] != TASK):
            raise ValueError("回放未完成")
        arm, cost = source["arm"], source["cost"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", arm) or cost not in {"normal", "double"}:
            raise ValueError("不支持的工件上下文")
        allowed = {"metrics.csv": "reports/metrics.csv"}
        names = [f"{arm}_predictions.parquet", *(f"{arm}_{cost}_{name}.parquet" for name in (
            *latent_api.LEDGERS, "selection_candidates", "sector_candidate_daily",
        ))]
        allowed.update({f"custom--{item}": f"backtests/custom/{item}" for item in names})
        if file_id not in allowed:
            from .expert_sector import authorized_download

            relative = authorized_download(root, report, file_id)
            if relative is not None:
                allowed[file_id] = relative
        file = _file(root, allowed[file_id])
    except (KeyError, OSError, ValueError) as error:
        raise latent_api.RouteError(404, "unknown_expert_replay_artifact", "没有此项组合回放工件。") from error
    latent_api._download(handler, file)
