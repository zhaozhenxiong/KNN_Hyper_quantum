"""Sector feature archives, auditable candidate drill-downs, and read-only API payloads."""

from __future__ import annotations

from datetime import date
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from urllib.parse import parse_qs, unquote

import numpy as np
import pandas as pd

from my_strategy.latent_quant.sector_features import (
    build_native_sector_daily, build_sector_daily, strategy_confirmation, summarize_selection_candidates,
)

from . import latent_api
from .runner import _write_table, write_json


COUNT_FIELDS = ("qualified_count", "purchase_candidate_count", "target_count", "buy_intent_count", "order_count", "fill_count")
FEATURE_FIELDS = (
    "open", "high", "low", "close", "ma20", "ma60", "return_1d", "return_5d", "return_20d", "return_60d",
    "relative_strength_20d", "breadth_above_ma20", "advance_ratio", "new_high_20d_count", "new_low_20d_count",
    "volatility_20d", "drawdown_60d", "sector_amount", "market_amount_share", "amount_change_1d", "amount_change_5d",
    "amount_ma5", "amount_ma20", "activity_ratio", "turnover_median", "turnover_coverage", "constituent_count",
    "price_participant_count", "suspended_count", "missing_price_count", "unclassified_count", "price_trend",
    "breadth_trend", "relative_strength_trend", "market_trend_score", "strong_up_resonance", "strong_down_resonance",
    "resonance_state",
)


def sector_key(industry: str) -> str:
    return sha256(industry.encode("utf-8")).hexdigest()[:12]


def _benchmark_close(benchmark: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    if isinstance(benchmark, pd.DataFrame):
        if "close" not in benchmark:
            raise ValueError("板块特征缺少沪深300基准收盘价。")
        close = benchmark["close"]
    else:
        close = benchmark
    close = pd.to_numeric(close, errors="coerce")
    close.index = pd.to_datetime(close.index)
    close = close.reindex(sessions)
    if not np.isfinite(close.to_numpy(dtype=float)).all() or not close.gt(0).all():
        raise ValueError("板块特征缺少完整、为正的沪深300基准收盘价。")
    return close.rename("benchmark_close")


def _figure(industry: str, frame: pd.DataFrame, destination: Path, *, native: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    data = frame.reset_index(drop=True)
    x = np.arange(len(data))
    days = pd.to_datetime(data["date"])
    blue, orange, aqua, danger, neutral = "#2a78d6", "#eb6834", "#1baf7a", "#b4232a", "#898781"
    with plt.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "figure.facecolor": "#fcfdfa", "savefig.facecolor": "#fcfdfa",
        "text.color": "#182522", "axes.labelcolor": "#566760",
    }):
        figure, axes = plt.subplots(6, 1, figsize=(15, 19), sharex=True)
        figure.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.06, hspace=0.32)
        title = f"{industry} · 同花顺原生行业指数（价格与相对强度趋势）" if native else f"{industry} · 等权合成指数与趋势共振（研究指标，非可交易价格）"
        figure.suptitle(title, x=0.07, ha="left", fontsize=18, weight="bold")
        price = axes[0]
        valid = data[["open", "high", "low", "close"]].notna().all(axis=1)
        for position, row in data.loc[valid].iterrows():
            color = danger if row["close"] >= row["open"] else aqua
            price.vlines(position, row["low"], row["high"], color=color, linewidth=0.8)
            price.vlines(position, min(row["open"], row["close"]), max(row["open"], row["close"]), color=color, linewidth=3)
        price.plot(x, data["ma20"], label="MA20", color=orange, linewidth=1.6)
        price.plot(x, data["ma60"], label="MA60", color=blue, linewidth=1.6)
        price.plot(x, data["benchmark_normalized"], label="沪深300（同基期）", color=neutral, linewidth=1.4, linestyle="--")
        for state, color, label in (("strong_up", danger, "强上升共振"), ("strong_down", aqua, "强下降共振")):
            mask = data["resonance_state"].eq(state)
            if mask.any():
                boundaries = np.diff(np.r_[False, mask.to_numpy(), False].astype(int))
                for start, end in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1), strict=True):
                    price.axvspan(start - 0.5, end - 0.5, color=color, alpha=0.08)
                price.plot([], [], color=color, alpha=0.2, linewidth=8, label=label)
        price.legend(loc="upper left", frameon=False, ncol=3)
        price.set_ylabel("同花顺行业指数" if native else "等权合成指数")
        axes[1].bar(x, data["sector_amount"], color=blue, width=0.72, label="板块成交额")
        axes[1].plot(x, data["amount_ma5"], color=orange, linewidth=1.5, label="成交额 MA5")
        axes[1].plot(x, data["amount_ma20"], color=aqua, linewidth=1.5, label="成交额 MA20")
        axes[1].legend(loc="upper left", frameon=False, ncol=3)
        axes[1].set_ylabel("成交额（元）")
        axes[2].plot(x, data["activity_ratio"], color=aqua, linewidth=1.7, label="活跃度 MA5/MA20")
        axes[2].axhline(1, color=neutral, linewidth=1, linestyle="--")
        axes[2].legend(loc="upper left", frameon=False)
        breadth_label = "历史成分宽度不可用（不使用当前成分回填）" if native else "成分股高于各自 MA20 比例"
        axes[3].plot(x, data["breadth_above_ma20"], color=blue, linewidth=1.7, label=breadth_label)
        if native:
            axes[3].text(0.5, 0.5, "无历史成分宽度，不显示 60%／40% 共振阈值", transform=axes[3].transAxes, ha="center", color=neutral)
        else:
            axes[3].axhline(0.6, color=orange, linewidth=1, linestyle="--")
            axes[3].axhline(0.4, color=danger, linewidth=1, linestyle="--")
        axes[3].yaxis.set_major_formatter(PercentFormatter(1))
        axes[3].set_ylim(0, 1)
        axes[3].legend(loc="upper left", frameon=False)
        width = 0.38
        axes[4].bar(x - width / 2, data.get("purchase_candidate_count", 0), width=width, color=aqua, label="Top-N 购买候选")
        axes[4].bar(x + width / 2, data.get("target_count", 0), width=width, color=orange, label="实际目标")
        if native and data[["purchase_candidate_count", "target_count"]].isna().all().all():
            axes[4].text(0.5, 0.5, "无点时成分，实际候选保留在未分类桶", transform=axes[4].transAxes, ha="center", color=neutral)
        axes[4].legend(loc="upper left", frameon=False)
        axes[4].set_ylabel("股票数")
        axes[5].plot(x, data["market_trend_score"], color=blue, linewidth=1.5, drawstyle="steps-mid", label="市场趋势分")
        axes[5].axhline(0, color=neutral, linewidth=1)
        axes[5].set_yticks(range(-2, 3) if native else range(-3, 4))
        axes[5].set_ylim(-2.3, 2.3) if native else axes[5].set_ylim(-3.3, 3.3)
        axes[5].legend(loc="upper left", frameon=False)
        ticks = np.unique(np.linspace(0, max(0, len(data) - 1), min(10, len(data)), dtype=int))
        axes[5].set_xticks(ticks, [days.iloc[index].strftime("%Y-%m-%d") for index in ticks], rotation=25, ha="right")
        for axis in axes:
            axis.grid(axis="y", color="#e7ede8", linewidth=0.7)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        figure.savefig(destination.with_suffix(".png"), dpi=140)
        figure.savefig(destination.with_suffix(".svg"))
        plt.close(figure)


def write_sector_artifacts(
    root: Path, frame: pd.DataFrame, benchmark: pd.Series | pd.DataFrame,
    selection_candidates: pd.DataFrame, sector_candidates: pd.DataFrame,
    *, provider: str = "capco",
) -> dict:
    """Build read-only market features and figures from the same frozen replay inputs."""
    required = {"date", "stock", "industry", "open", "high", "low", "close", "amount"}
    if not required.issubset(frame.columns) or frame.empty:
        report = {"status": "unavailable", "reason": "冻结面板缺少板块计算所需的后复权行情或行业字段。"}
        write_json(root / "reports/sector_resonance.json", report)
        return report
    if provider not in {"capco", "ths_fuyao"}:
        raise ValueError("未知板块数据提供方")
    market = frame.copy()
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    sessions = pd.DatetimeIndex(sorted(market["date"].unique()))
    close = _benchmark_close(benchmark, sessions)
    native = provider == "ths_fuyao"
    snapshot_identity = None
    if native:
        from .fuyao_sector import apply_membership_snapshots, load_latest_snapshot

        snapshot = load_latest_snapshot()
        native_daily = snapshot["daily"].copy()
        native_daily["date"] = pd.to_datetime(native_daily["date"], errors="raise")
        native_daily = native_daily.loc[native_daily["date"].isin(sessions)]
        if native_daily.empty:
            raise ValueError("同花顺冻结快照与本地回测交易日没有交集")
        sector = build_native_sector_daily(native_daily, close, sessions=sessions)
        selection = apply_membership_snapshots(selection_candidates, snapshot["memberships"])
        rebalances = sector_candidates.loc[sector_candidates["is_rebalance"].astype(bool), "date"].unique()
        candidates = summarize_selection_candidates(selection, sessions=sessions, rebalance_dates=rebalances)
        snapshot_identity = {
            "observed_at": snapshot["manifest"]["observed_at"],
            "effective_from": snapshot["manifest"]["effective_from"],
            "manifest_sha256": sha256((snapshot["root"] / "manifest.json").read_bytes()).hexdigest(),
            "catalog_rows": snapshot["manifest"]["catalog_rows"],
            "classified_stocks": snapshot["manifest"]["classified_stocks"],
        }
    else:
        sector = build_sector_daily(market, close, sessions=sessions)
        sector["industry_name"] = sector["industry"]
        selection = selection_candidates.copy()
        candidates = sector_candidates.copy()
    sector["benchmark_close"] = sector["date"].map(close)
    sector = sector.sort_values(["industry", "date"], kind="stable")
    sector["benchmark_normalized"] = sector.groupby("industry", sort=False)["benchmark_close"].transform(
        lambda values: values / values.dropna().iloc[0] if values.notna().any() else np.nan,
    )
    first_close = sector.groupby("industry", sort=False)["close"].transform(lambda values: values.dropna().iloc[0] if values.notna().any() else np.nan)
    sector["benchmark_normalized"] = sector["benchmark_normalized"] * first_close

    candidates["date"] = pd.to_datetime(candidates["date"], errors="raise")
    candidates = strategy_confirmation(candidates)
    feature_dir = root / "features"
    figure_dir = root / "reports/figures"
    feature_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    _write_table(feature_dir / "sector_daily.parquet", sector)
    sector.to_csv(feature_dir / "sector_daily.csv", index=False, encoding="utf-8-sig")
    _write_table(feature_dir / "sector_candidate_daily.parquet", candidates)
    if native:
        _write_table(feature_dir / "sector_selection_candidates.parquet", selection)
    figures = {}
    for industry in sorted(sector.loc[sector["industry"].ne("unknown"), "industry"].unique()):
        industry_rows = sector.loc[sector["industry"].eq(industry)]
        industry_name = str(industry_rows["industry_name"].dropna().iloc[0]) if industry_rows["industry_name"].notna().any() else industry
        rows = industry_rows.merge(
            candidates.loc[candidates["industry"].eq(industry), ["date", "industry", *COUNT_FIELDS, "selection_reason", "strategy_confirmation"]
                if "strategy_confirmation" in candidates else ["date", "industry", *COUNT_FIELDS, "selection_reason"]],
            on=["date", "industry"], how="left", validate="one_to_one",
        )
        without_membership = rows["selection_reason"].isna()
        rows.loc[without_membership, "selection_reason"] = "NO_POINT_IN_TIME_MEMBERSHIP" if native else "NO_CANDIDATE_ROWS"
        if "strategy_confirmation" in rows:
            rows.loc[without_membership, "strategy_confirmation"] = "insufficient"
        if rows.empty:
            continue
        key = sector_key(industry)
        rows.to_csv(figure_dir / f"sector_{key}_chart_data.csv", index=False, encoding="utf-8-sig")
        _figure(f"{industry_name} · {industry}" if native else industry, rows, figure_dir / f"sector_{key}", native=native)
        figures[industry] = {
            "key": key, "name": industry_name, "rows": len(rows),
            "png": f"reports/figures/sector_{key}.png", "svg": f"reports/figures/sector_{key}.svg",
            "csv": f"reports/figures/sector_{key}_chart_data.csv",
        }
    report = {
        "status": "available", "schema_version": 2, "market_basis": "native_archive",
        "provider": provider,
        "price_basis": "ths_native_index" if native else "equal_weight_adjusted",
        "classification": "ths_fuyao_observed_snapshots" if native else "point_in_time_capco",
        "snapshot": snapshot_identity,
        "synthetic_index_notice": (
            "同花顺原生行业指数使用真实指数行情；历史成分宽度不可用，不以当前成分回填；成交额变化不是主力净流入。"
            if native else "等权合成指数仅供研究，不是可交易价格；成交额变化不是主力净流入。"
        ),
        "resonance_rule": {
            "price": "close>MA20>MA60 and positive MA20 five-session slope; strict reverse for down",
            "breadth": "unavailable_without_point_in_time_constituents" if native else "above-MA20 breadth >=60% / <=40%",
            "relative_strength": "20-session sector minus CSI300 return",
            "strong": "disabled_without_historical_breadth" if native else "market trend score +/-3 and amount MA5/MA20 >=1",
            "strategy_confirmation": "separate layer; never changes market score",
        },
        "sector_rows": len(sector), "industries": figures,
        "candidate_rows": len(selection), "candidate_daily_rows": len(candidates),
        "first_date": str(sector["date"].min().date()), "last_date": str(sector["date"].max().date()),
    }
    write_json(root / "reports/sector_resonance.json", report)
    return report


def _clean(value):
    if isinstance(value, pd.DataFrame):
        return _records(value)
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [_clean(row) for row in frame.to_dict(orient="records")]


def _archive(run_id: str, account: str) -> tuple[Path, dict, str]:
    if account != "full_pool":
        raise latent_api.RouteError(400, "invalid_sector_account", "板块候选目前仅提供全池账户。")
    root = latent_api._run_folder(run_id)
    try:
        status = latent_api._read(root / "status.json")
        report = latent_api._read(root / "reports/expert_stock_replay.json")
    except (OSError, ValueError) as error:
        raise latent_api.RouteError(404, "unknown_sector_run", "没有此已完成的组合回测。") from error
    if root.name != run_id or status.get("status") != "completed" or report.get("run", {}).get("run_id") != run_id:
        raise latent_api.RouteError(404, "unknown_sector_run", "没有此已完成的组合回测。")
    if report.get("mode") in {"expert_watchlist", "expert_independent_stock"}:
        if not report.get("accounts", {}).get("full_pool", {}).get("downloads"):
            raise latent_api.RouteError(404, "unknown_sector_account", "此运行没有全池账户。")
        prefix = f"full_pool_{report['source']['cost']}"
    elif report.get("mode") == "expert_portfolio":
        prefix = f"{report['source']['arm']}_{report['source']['cost']}"
    else:
        raise latent_api.RouteError(404, "unknown_sector_run", "此运行不是全市场组合回测。")
    return root, report, prefix


def _tables(root: Path, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    report = json.loads((root / "reports/sector_resonance.json").read_text(encoding="utf-8"))
    native = report.get("provider") == "ths_fuyao"
    selection_path = root / ("features/sector_selection_candidates.parquet" if native else f"backtests/custom/{prefix}_selection_candidates.parquet")
    paths = (
        root / "features/sector_daily.parquet",
        selection_path,
        root / "features/sector_candidate_daily.parquet",
    )
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("该归档未封存板块候选明细；旧回测不会伪造补齐。")
    sector, selection, candidates = (pd.read_parquet(path) for path in paths)
    for frame in (sector, selection, candidates):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    _reconcile(selection, candidates)
    return sector, selection, candidates, report


def _reconcile(selection: pd.DataFrame, candidates: pd.DataFrame) -> None:
    actual = selection.assign(
        qualified_count=selection["qualified"].astype(bool).astype(int),
        purchase_candidate_count=selection["purchase_candidate"].astype(bool).astype(int),
        target_count=selection["target_weight"].fillna(0).gt(0).astype(int),
        buy_intent_count=selection["intent_side"].eq("BUY").astype(int),
    ).groupby(["date", "industry"], as_index=False)[list(COUNT_FIELDS)].sum()
    expected = candidates[["date", "industry", *COUNT_FIELDS]].copy()
    merged = expected.merge(actual, on=["date", "industry"], how="outer", suffixes=("_expected", "_actual"), indicator=True)
    if merged["_merge"].ne("both").any():
        raise ValueError("板块候选合计与全市场逐股账本不一致。")
    for field in COUNT_FIELDS:
        if not merged[f"{field}_expected"].fillna(0).eq(merged[f"{field}_actual"].fillna(0)).all():
            raise ValueError(f"板块候选合计与全市场账本不一致：{field}。")


def _unavailable(run_id: str, industry: str | None, reason: str) -> dict:
    result = {"run_id": run_id, "status": "unavailable", "reason": reason, "downloads": {}}
    if industry is not None:
        result.update(industry=industry, series=[] if industry else [])
    return result


def _downloads(run_id: str, prefix: str, *, native: bool = False) -> dict:
    base = f"/api/expert/replays/{run_id}/downloads"
    return {
        "sector_daily": f"{base}/sector_daily.csv",
        "selection_candidates": f"{base}/sector_selection_candidates.parquet" if native else f"{base}/custom--{prefix}_selection_candidates.parquet",
        "sector_candidates": f"{base}/sector_candidate_daily.parquet" if native else f"{base}/custom--{prefix}_sector_candidate_daily.parquet",
        "report": f"{base}/sector_resonance.json",
    }


def _filters(query: str, *, chart: bool = False) -> dict:
    allowed = {"account", "date"} | ({"start", "end"} if chart else set())
    try:
        values = parse_qs(query, max_num_fields=len(allowed))
        if set(values) - allowed or any(len(items) != 1 for items in values.values()):
            raise ValueError("invalid filters")
        result = {key: items[0] for key, items in values.items()}
        for key in ("date", "start", "end"):
            value = result.get(key, "")
            if value and date.fromisoformat(value).isoformat() != value:
                raise ValueError("invalid date")
        if result.get("start") and result.get("end") and result["start"] > result["end"]:
            raise ValueError("invalid range")
        return result
    except ValueError as error:
        raise latent_api.RouteError(400, "invalid_sector_filter", "板块账户或日期筛选不合法。") from error


def overview(run_id: str, query: str) -> dict:
    filters = _filters(query)
    root, _, prefix = _archive(run_id, filters.get("account", "full_pool"))
    try:
        sector, _, candidates, report = _tables(root, prefix)
    except FileNotFoundError as error:
        return _unavailable(run_id, None, str(error)) | {"sectors": [], "dates": []}
    dates = sorted(candidates["date"].unique())
    sector_dates = sorted(sector["date"].unique())
    if candidates.empty:
        if not sector_dates:
            return _unavailable(run_id, None, "板块市场特征和候选账本均为空。") | {"sectors": [], "dates": []}
        selected = pd.Timestamp(filters["date"]) if filters.get("date") else sector_dates[-1]
        if selected not in sector_dates:
            raise latent_api.RouteError(400, "unknown_sector_date", "所选日期不在板块市场账本中。")
        daily = sector.loc[sector["date"].eq(selected)].copy()
        daily.loc[:, list(COUNT_FIELDS)] = 0
        daily["selection_reason"] = "NO_CANDIDATE_ROWS"
        daily["strategy_confirmation"] = "insufficient"
    else:
        selected = pd.Timestamp(filters["date"]) if filters.get("date") else dates[-1]
        if selected not in dates:
            raise latent_api.RouteError(400, "unknown_sector_date", "所选日期不在板块候选账本中。")
        daily = candidates.loc[candidates["date"].eq(selected)].copy()
        features = sector.loc[sector["date"].eq(selected)]
        daily = daily.merge(features, on=["date", "industry"], how="outer", validate="one_to_one", indicator=True)
        without_membership = daily["_merge"].eq("right_only")
        daily.loc[without_membership, "selection_reason"] = (
            "NO_POINT_IN_TIME_MEMBERSHIP" if report.get("provider") == "ths_fuyao" else "NO_CANDIDATE_ROWS"
        )
        daily.loc[without_membership, "strategy_confirmation"] = "insufficient"
        daily = daily.drop(columns="_merge")
    rows = []
    for row in _records(daily):
        industry = row["industry"]
        history = sector.loc[sector["industry"].eq(industry) & sector["date"].le(selected), ["date", "close"]].tail(30).copy()
        first = history["close"].dropna()
        history["value"] = history["close"] / first.iloc[0] if len(first) else np.nan
        row["sparkline"] = _records(history)
        rows.append(row)
    totals = {field: int(daily[field].fillna(0).sum()) for field in COUNT_FIELDS}
    native = report.get("provider") == "ths_fuyao"
    notice = report.get("synthetic_index_notice", "等权合成指数仅供研究；成交额变化不是主力净流入；候选与目标分开计数。")
    return _clean({
        "run_id": run_id, "account": "full_pool", "status": "available", "date": selected,
        "dates": dates, "market_basis": "native_archive", "provider": report.get("provider", "capco"),
        "classification": report.get("classification", "point_in_time_capco"),
        "price_basis": report.get("price_basis", "equal_weight_adjusted"), "snapshot": report.get("snapshot"),
        "notice": notice,
        "totals": totals, "sectors": rows, "downloads": _downloads(run_id, prefix, native=native),
    })


def _industry(value: str) -> str:
    industry = unquote(value)
    if not industry or len(industry) > 160 or "/" in industry or "\\" in industry or industry in {".", ".."}:
        raise latent_api.RouteError(400, "invalid_industry", "行业名称不合法。")
    return industry


def chart(run_id: str, industry_value: str, query: str) -> dict:
    industry = _industry(industry_value)
    filters = _filters(query, chart=True)
    root, _, prefix = _archive(run_id, filters.get("account", "full_pool"))
    try:
        sector, _, candidates, report = _tables(root, prefix)
    except FileNotFoundError as error:
        return _unavailable(run_id, industry, str(error))
    available = set(sector["industry"].unique())
    native = report.get("provider") == "ths_fuyao"
    if industry not in available:
        if native and industry == "unknown" and candidates["industry"].eq(industry).any():
            return _unavailable(
                run_id, industry,
                "快照生效前的候选仅归入未分类；没有同花顺原生未分类指数，不能用当前成分伪造历史行情。",
            ) | {"provider": "ths_fuyao", "downloads": _downloads(run_id, prefix, native=True)}
        raise latent_api.RouteError(404, "unknown_industry", "此归档没有该行业。")
    industry_dates = sector.loc[sector["industry"].eq(industry), "date"]
    if candidates.empty:
        first, last = industry_dates.min(), industry_dates.max()
    else:
        first, last = candidates["date"].min(), candidates["date"].max()
    start = pd.Timestamp(filters.get("start") or first)
    end = pd.Timestamp(filters.get("end") or last)
    features = sector.loc[sector["industry"].eq(industry) & sector["date"].between(start, end)].copy()
    daily = candidates.loc[
        candidates["industry"].eq(industry),
        ["date", "industry", *COUNT_FIELDS, "selection_reason", "strategy_confirmation"],
    ]
    rows = features.merge(daily, on=["date", "industry"], how="left", validate="one_to_one")
    without_membership = rows["selection_reason"].isna()
    rows.loc[without_membership, "selection_reason"] = (
        "NO_POINT_IN_TIME_MEMBERSHIP" if native else "NO_CANDIDATE_ROWS"
    )
    rows.loc[without_membership, "strategy_confirmation"] = "insufficient"
    if candidates.empty:
        rows.loc[:, list(COUNT_FIELDS)] = rows.loc[:, list(COUNT_FIELDS)].fillna(0)
        rows["selection_reason"] = "NO_CANDIDATE_ROWS"
        rows["strategy_confirmation"] = "insufficient"
    key = report.get("industries", {}).get(industry, {}).get("key", sector_key(industry))
    industry_name = report.get("industries", {}).get(industry, {}).get("name", industry)
    downloads = _downloads(run_id, prefix, native=native) | {"chart_csv": f"/api/expert/replays/{run_id}/downloads/sector--{key}--chart.csv"}
    return _clean({
        "run_id": run_id, "industry": industry, "industry_name": industry_name, "status": "available", "start": start, "end": end,
        "provider": report.get("provider", "capco"),
        "price_basis": report.get("price_basis", "equal_weight_adjusted"),
        "notice": report.get("synthetic_index_notice", "等权合成指数不是可交易价格；沪深300按同一基期归一化。"),
        "series": rows, "downloads": downloads,
    })


def candidates(run_id: str, industry_value: str, query: str) -> dict:
    industry = _industry(industry_value)
    filters = _filters(query)
    root, _, prefix = _archive(run_id, filters.get("account", "full_pool"))
    try:
        _, selection, daily, report = _tables(root, prefix)
    except FileNotFoundError as error:
        return _unavailable(run_id, industry, str(error)) | {"rows": []}
    dates = sorted(daily["date"].unique())
    if daily.empty:
        return _clean({
            "run_id": run_id, "industry": industry, "date": filters.get("date"), "status": "available", "rows": [],
            "downloads": _downloads(run_id, prefix, native=report.get("provider") == "ths_fuyao"),
        })
    selected = pd.Timestamp(filters["date"]) if filters.get("date") else dates[-1]
    if selected not in dates:
        raise latent_api.RouteError(400, "unknown_sector_date", "所选日期不在板块候选账本中。")
    rows = selection.loc[selection["date"].eq(selected) & selection["industry"].eq(industry)].sort_values(
        ["purchase_candidate", "global_rank", "stock"], ascending=[False, True, True], na_position="last",
    )
    return _clean({
        "run_id": run_id, "industry": industry, "date": selected, "status": "available", "rows": rows,
        "downloads": _downloads(run_id, prefix, native=report.get("provider") == "ths_fuyao"),
    })


def authorized_download(root: Path, report: dict, file_id: str) -> str | None:
    if file_id == "sector_daily.csv":
        return "features/sector_daily.csv"
    if file_id == "sector_daily.parquet":
        return "features/sector_daily.parquet"
    if file_id == "sector_candidate_daily.parquet":
        return "features/sector_candidate_daily.parquet"
    if file_id == "sector_selection_candidates.parquet":
        return "features/sector_selection_candidates.parquet"
    if file_id == "sector_resonance.json":
        return "reports/sector_resonance.json"
    match = re.fullmatch(r"sector--([a-f0-9]{12})--chart\.(csv|png|svg)", file_id)
    if match:
        suffix = "chart_data.csv" if match[2] == "csv" else f".{match[2]}"
        name = f"sector_{match[1]}_{suffix}" if match[2] == "csv" else f"sector_{match[1]}.{match[2]}"
        return f"reports/figures/{name}"
    return None


def download(run_id: str, file_id: str) -> str | None:
    root = latent_api._run_folder(run_id)
    report = latent_api._read(root / "reports/expert_stock_replay.json")
    return authorized_download(root, report, file_id)
