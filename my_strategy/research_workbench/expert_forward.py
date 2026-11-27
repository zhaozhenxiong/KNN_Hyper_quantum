"""Seal one expert candidate and measure its own contemporaneous 60-session holdout.

The final-fit producer supplies models/forward/model.json plus native quantile
models and a training cohort at models/forward/train.parquet. The manifest binds
the research report, arm, fitted label/data cutoff, availability timestamp,
asset hashes, model feature order, fusion weights, and per-arm offsets.

Each intake bundle declares trade_date and observed, SHA-256-pinned calendar,
panel, benchmark, market status, and announcements. This runner generates the
forecasts and next-open quantities from that evidence and the locked model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

from my_strategy.core.run_context import RunContext, current_git_commit, stable_hash
from my_strategy.latent_quant import forward, models
from my_strategy.latent_quant.evaluation import equity_returns, paired_block_interval, score_quantiles
from my_strategy.latent_quant.execution import Strategy, StudyCostModel, scaled_study_cost, simulate
from my_strategy.latent_quant.forward_accept import _check_orders, _holdout_labels
from my_strategy.latent_quant.runner import _verify_ledger
from my_strategy.model_registry import expert_release
from my_strategy.storage.expert_artifacts import _publish

from . import expert_replay, expert_research
from .runner import _write_table, write_json


COUNT = 60
ROOT = "expert_forward"
ACCEPTANCE = "reports/expert_forward_acceptance.json"
SCORER = "models/forward/model.json"
MARKET_ROLES = ("calendar", "panel", "benchmark", "market_status", "announcements")
PANEL_COLUMNS = (*forward.FIELDS["panel"], "trade_high", "trade_low", "volume", "source")
CORE_ARMS = {"historical_quantiles", "domestic_gpu"}
MODEL_ARMS = {"domestic_gpu", "sector_gpu", "overseas_gpu"}
Q = [*models.QUANTILE_COLUMNS, "score"]
TABLES = ("panel", "benchmark", "market_status", "predictions", "orders")


@dataclass(frozen=True)
class Lock:
    source: Path
    record: dict
    policy: forward.Policy
    model: dict
    fit_path: Path

    @property
    def path(self) -> Path:
        return self.source / ROOT / "lock/lock.json"


def _immutable_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as staging:
        temp = Path(staging) / path.name
        write_json(temp, payload)
        try:
            _publish(temp, path)
        except FileExistsError:
            if forward._json(path) != forward._json(temp):
                raise


def _historical_report(report: dict) -> list[str]:
    decision = report["decision"]
    return [message for key, message in (
        ("historical_numerical_gate", "历史样本外预测、双倍成本和风险验收未通过。"),
        ("point_in_time_verified", "历史行情首次可用时间及供应商修订尚未核验。"),
    ) if decision.get(key) is not True]


def _arms(arm: str) -> frozenset[str]:
    if arm not in expert_release.RELEASE_ARMS:
        raise ValueError("研究未指定可验收的最终策略。")
    return frozenset({*CORE_ARMS, arm})


def _model_assets(root: Path, report: dict, config: dict) -> tuple[dict, dict]:
    model = forward._json(expert_replay._file(root, SCORER))
    arm = report["final_portfolio"]["arm"]
    arms = _arms(arm)
    report_sha = forward._sha(root / expert_release.REPORT)
    if (model.get("schema_version") != 1 or model.get("run_id") != root.name or model.get("arm") != arm
            or model.get("research_report_sha256") != report_sha
            or model.get("point_in_time_verified") is not True):
        raise ValueError("候选最终模型未绑定通过核验的研究版本。")
    weights = model["weights"]
    allowed = {*arms, "supertrend_atr", "sector_gpu", "overseas_gpu"} - {"fusion"}
    if (set(weights) - allowed or not all(type(weight) in (float, int) and np.isfinite(weight) and weight > 0
                                         for weight in weights.values())
            or not np.isclose(sum(weights.values()), 1.0)
            or (arm == "domestic_gpu" and weights != {"domestic_gpu": 1.0})):
        raise ValueError("最终模型的专家权重无效。")
    needed = set(weights).intersection(MODEL_ARMS) | {"domestic_gpu"}
    features = model["features"]
    if set(features) != needed or any(features[key] != config["features"][key] for key in needed):
        raise ValueError("最终模型特征与候选研究不一致。")
    offsets = model["offsets"]
    if set(offsets) != arms or any(np.asarray(values).shape != (3,) or not np.isfinite(values).all() for values in offsets.values()):
        raise ValueError("最终模型缺少三个有序分位数的固定校准。")
    native = {"models/forward/train.parquet", *(f"models/forward/{name}/{q}.cbm"
                                               for name in needed for q in models.QUANTILE_COLUMNS)}
    if set(model["files"]) != native:
        raise ValueError("最终模型缺少规则训练样本或 CBM 文件。")
    actual = {role: forward._sha(expert_replay._file(root, role)) for role in native}
    if actual != model["files"]:
        raise ValueError("最终模型文件哈希不一致。")
    return model, {**actual, SCORER: forward._sha(root / SCORER)}


def _features(model: dict) -> tuple[str, ...]:
    columns = [column for features in model["features"].values() for column in features]
    return tuple(dict.fromkeys([*columns, *(models._TREND_COLUMNS if "supertrend_atr" in model["weights"] else ())]))


def _optional_features(model: dict) -> frozenset[str]:
    required = set(model["features"].get("domestic_gpu", ()))
    if "supertrend_atr" in model["weights"]:
        required.update(models._TREND_COLUMNS)
    return frozenset(_features(model)) - required


def _check_fit_time(model: dict, config: dict, sealed, first: date) -> None:
    published = forward._timestamp(model["sealed_at"])
    fitted = date.fromisoformat(model["training_max_label_date"])
    through = date.fromisoformat(model["training_input_through"])
    if (not fitted <= through <= published.date() <= sealed.date() < first
            or published > sealed or sealed > forward._now() or through > date.fromisoformat(config["end"])):
        raise ValueError("拟合标签、输入截止及实际封存时间须早于首个前瞻交易日。")


def _load_lock(root: Path) -> Lock:
    path = expert_replay._file(root, f"{ROOT}/lock/lock.json")
    record = forward._json(path)
    _, report, config = expert_replay._source(root.name)
    model, model_hashes = _model_assets(root, report, config)
    roster = expert_replay._file(root, f"{ROOT}/lock/roster.parquet")
    settings = root / "config/study.json"
    start = date.fromisoformat(record["first_session"])
    sealed = forward._timestamp(record["locked_at"])
    if (record.get("schema_version") != 1 or record.get("run_id") != root.name or record["arm"] != model["arm"]
            or record["sessions"] != COUNT or _historical_report(report)
            or record["config_sha256"] != forward._sha(settings)
            or record["roster_sha256"] != forward._sha(roster)
            or record["model_sha256"] != model_hashes or record["report_sha256"] != model["research_report_sha256"]):
        raise ValueError("前瞻锁或归档模型与最初哈希不一致。")
    _check_fit_time(model, config, sealed, start)
    policy = forward.Policy(start, COUNT, _features(model), _arms(model["arm"]),
                            frozenset(config["cost_multipliers"]), settings, roster,
                            optional_features=_optional_features(model))
    return Lock(root, record, policy, model, root / "models/forward/train.parquet")


def lock(run_id: str, first_session: date, *, calendar_path: Path = forward.CALENDAR) -> Path:
    """Pin the eligible final scorer, power controls and actual availability before the first session."""
    root, report, config = expert_replay._source(run_id)
    if blockers := _historical_report(report):
        raise ValueError(" ".join(blockers))
    model, hashes = _model_assets(root, report, config)
    now = forward._now()
    _check_fit_time(model, config, now, first_session)
    if not forward._calendar(calendar_path).get(first_session):
        raise ValueError("首个前瞻交易日缺少已验证的日历。")
    files, digests = expert_replay._data_files(root, config)
    destination = root / ROOT / "lock"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary)
        roster = stage / "roster.parquet"
        shutil.copyfile(files["roster.parquet"], roster)
        if forward._sha(roster) != digests["roster.parquet"]:
            raise ValueError("冻结股票池读取期间发生变化。")
        checked = forward._now()
        if checked.date() >= first_session:
            raise ValueError("冻结完成时已错过首个样本日。")
        write_json(stage / "lock.json", {
            "schema_version": 1, "run_id": run_id, "arm": model["arm"], "locked_at": checked.isoformat(),
            "first_session": str(first_session), "sessions": COUNT, "config_sha256": forward._sha(root / "config/study.json"),
            "report_sha256": model["research_report_sha256"], "model_sha256": hashes, "roster_sha256": digests["roster.parquet"],
        })
        stage.rename(destination)
    return destination / "lock.json"


def _captures(locked: Lock, calendar_path: Path | None) -> tuple[dict, dict, list[date]]:
    now = forward._now()
    calendar = forward._calendar(calendar_path) if calendar_path else {}
    records, previous = {}, None
    scope = locked.source / ROOT
    lock_hash = forward._sha(locked.path)
    for folder in sorted((scope / "captures").glob("????-??-??")):
        day = date.fromisoformat(folder.name)
        manifest = expert_replay._file(scope, f"captures/{day}/manifest.json")
        record = forward._json(manifest)
        stamp = forward._timestamp(record["captured_at"])
        files = record["files"]
        if (record.get("schema_version") != 1 or record["trade_date"] != str(day) or record["lock_sha256"] != lock_hash
                or record["previous_sha256"] != previous or stamp.date() != day
                or stamp > now or not forward.CLOSE <= stamp.time() <= forward.DEFAULT_CUTOFF
                or day < locked.policy.start or set(files) != set(forward.FIELDS)):
            raise ValueError(f"{day} 的前瞻封存时点或证据链不一致。")
        producer = expert_replay._file(scope, f"captures/{day}/market_bundle.json")
        if forward._sha(producer) != record["market_bundle_sha256"]:
            raise ValueError(f"{day} 的生产者证据包已变化。")
        for role, evidence in files.items():
            filename = evidence["file"]
            if filename not in {f"{role}.parquet", f"{role}.csv"}:
                raise ValueError(f"{day} 的证据文件名不受支持：{role}。")
            observed = forward._timestamp(evidence["observed_at"])
            source = evidence["source"]
            derived = role not in MARKET_ROLES
            if (not source.strip() or observed > stamp or (
                role != "calendar" and (observed.date() != day or observed.time() < forward.CLOSE)
            ) or (derived and (source != f"frozen_expert:{lock_hash}" or observed != stamp))):
                raise ValueError(f"{day} 的输入可用时间或来源无效：{role}。")
            file = expert_replay._file(scope, f"captures/{day}/{filename}")
            if forward._sha(file) != evidence["sha256"]:
                raise ValueError(f"{day} 的原始封存证据哈希不匹配：{role}。")
        forward._merge_calendar(calendar, forward._calendar(folder / files["calendar"]["file"]))
        records[day] = record
        previous = forward._sha(manifest)
    sessions = forward._ordered_sessions(calendar, locked.policy.start)
    if len(records) > COUNT or list(records) != sessions[:len(records)] or any(
        str(next_day) != records[day]["next_trade_date"] for day, next_day in zip(sessions, sessions[1:]) if day in records
    ) or (len(records) and len(sessions) == len(records)):
        raise ValueError("封存日期缺失、超出范围或与下一交易日不一致。")
    return records, calendar, sessions


def status(run_id: str, *, calendar_path: Path | None = forward.CALENDAR) -> dict:
    """Return eligibility and progress without creating locks, captures, or model entries."""
    root, report, config = expert_replay._source(run_id)
    blockers = _historical_report(report)
    if not (root / SCORER).is_file():
        blockers.append("该候选没有可封存的最终模型。")
    elif not blockers:
        try:
            _model_assets(root, report, config)
        except (KeyError, OSError, TypeError, ValueError) as error:
            blockers.append(f"最终模型无法核验。{error}")
    if not (root / ROOT / "lock/lock.json").is_file():
        return {"run_id": run_id, "state": "blocked" if blockers else "ready_to_lock", "blockers": blockers,
                "valid_captures": 0, "required_captures": COUNT}
    locked = _load_lock(root)
    entries, _, dates = _captures(locked, calendar_path)
    now = forward._now()
    next_capture = dates[len(entries)] if len(entries) < min(COUNT, len(dates)) else None
    phase = "ready_for_acceptance" if len(entries) == COUNT else "awaiting_capture"
    if len(entries) == COUNT and (root / ACCEPTANCE).is_file():
        assessment = forward._json(root / ACCEPTANCE)
        if assessment["qualified_forward_results"]:
            verify_release_evidence(root, assessment, forward._json(root / expert_release.MANIFEST))
            phase = "qualified"
        else:
            phase = "failed_acceptance"
    if next_capture is not None and (next_capture < now.date()
                                    or (next_capture == now.date() and now.time() > forward.DEFAULT_CUTOFF)):
        phase = "missed_capture"
    return {"run_id": run_id, "state": phase, "arm": locked.model["arm"], "blockers": blockers,
            "first_session": str(locked.policy.start), "valid_captures": len(entries), "required_captures": COUNT,
            "next_capture": str(next_capture) if next_capture else None,
            "calendar_sessions_known": min(COUNT, len(dates))}


def _forecast_assets(
    model: dict, model_root: Path, fit_path: Path, arms: frozenset[str], panel: pd.DataFrame, stamp,
    *, verify_macro_availability: bool = True,
) -> pd.DataFrame:
    """Read-only final-model inference shared by exact fit replay and sealed capture."""
    from catboost import CatBoostRegressor

    rows = panel.loc[panel["signal_eligible"].eq(True)]
    if rows.empty:
        return pd.DataFrame(columns=forward.FIELDS["predictions"])
    train = pd.read_parquet(fit_path)
    dates = pd.to_datetime(train["label_end"])
    if (train.empty or dates.isna().any() or not dates.gt(pd.to_datetime(train["date"])).all()
            or dates.max().date() > date.fromisoformat(model["training_max_label_date"])
            or not np.isfinite(train["target"]).all()):
        raise ValueError("最终模型的规则训练样本越过冻结标签时点。")
    pooled = train["target"].quantile(list(models.QUANTILE_LEVELS)).to_numpy()
    raw = {"historical_quantiles": models._historical_quantiles(train, rows, pooled)[0]}
    weights = model["weights"]
    if "supertrend_atr" in weights:
        raw["supertrend_atr"] = models._trend_quantiles(train, rows, pooled)[0]
    if "overseas_gpu" in weights and verify_macro_availability:
        fields = list(expert_research.MACRO_FEATURES)
        valid = rows["fomc_availability_basis"].eq("official_release_header")
        observed = [forward._timestamp(str(value)) for value in rows.loc[valid, "fomc_available_at"]]
        if (any(available > stamp for available in observed)
                or not np.isfinite(rows.loc[valid, fields]).all(axis=None)):
            raise ValueError("前瞻海外输入缺少信号时点前的官方可用时间。")
        rows = rows.copy()
        rows.loc[~valid, fields] = np.nan
    inference = {}
    for arm, features in sorted(model["features"].items()):
        native = []
        for quantile in models.QUANTILE_COLUMNS:
            est = CatBoostRegressor()
            est.load_model(str(model_root / arm / f"{quantile}.cbm"), format="cbm")
            if list(est.feature_names_) != features:
                raise ValueError("最终模型原生特征顺序不一致。")
            native.append(est)
        inference[arm] = expert_research._inference_audit()
        raw[arm] = expert_research._expert_raw(
            native, rows, tuple(features), inference[arm],
            fallback=raw["domestic_gpu"] if arm == "overseas_gpu" else None,
        )
    if "fusion" in arms:
        raw["fusion"] = expert_research._fuse(weights, raw)
    forecast = pd.concat([
        rows[["date", "stock"]].join(models._prediction_table(
            np.sort(np.sort(raw[arm], axis=1) + np.asarray(model["offsets"][arm]), axis=1), rows.index,
        )).assign(arm=arm)
        for arm in sorted(arms)
    ], ignore_index=True)
    forecast.attrs["inference"] = inference
    return forecast


def _forecast(locked: Lock, panel: pd.DataFrame, stamp) -> pd.DataFrame:
    return _forecast_assets(locked.model, locked.source / "models/forward", locked.fit_path,
                            locked.policy.arms, panel, stamp)


def _daily_paths(locked: Lock, day: date, record: dict) -> dict[str, Path]:
    folder = locked.source / ROOT / "captures" / str(day)
    return {role: folder / value["file"] for role, value in record["files"].items()}


def _load_tables(paths: dict[str, Path]) -> dict:
    tables = {name: forward._table(paths[name], PANEL_COLUMNS if name == "panel" else forward.FIELDS[name]) for name in TABLES}
    for table in tables.values():
        table["date"] = pd.to_datetime(table["date"])
    tables["orders"]["signal_date"] = pd.to_datetime(tables["orders"]["signal_date"])
    return tables


def _combine(days: list[dict]) -> dict:
    return {name: pd.concat([tables[name] for tables in days], ignore_index=True) for name in TABLES}


def _simulate(locked: Lock, tables: dict, dates: pd.DatetimeIndex, run_id: str, *, check: bool) -> tuple[dict, pd.DataFrame]:
    config = forward._json(locked.policy.config)
    rates = StudyCostModel(**config["expense_model"])
    if set(config["cost_multipliers"]) != {"normal", "double"} or config["cost_multipliers"] != {"normal": 1, "double": 2}:
        raise ValueError("专家前瞻需要固定的常规及双倍佣金滑点档。")
    for cost, multiple in config["cost_multipliers"].items():
        contract = asdict(scaled_study_cost(multiple, rates))
        if config.get("execution_costs", {}).get(cost, contract) != contract:
            raise ValueError("冻结费用档无法从来源配置复算。")
    frame = tables["panel"].sort_values(["date", "stock"]).reset_index(drop=True)
    if frame.duplicated(["date", "stock"]).any():
        raise ValueError("前瞻面板包含重复股票日。")
    bench = tables["benchmark"].set_index("date")["close"].pct_change(fill_method=None).reindex(dates)
    labels = _holdout_labels(frame, tables["market_status"], tables["benchmark"], dates) if check else None
    roster = pd.read_parquet(locked.policy.roster)
    if "stock" in roster:
        roster = roster.set_index("stock")
    portfolio = Strategy(**config["portfolio"])
    outputs, order_batches = {}, []
    for arm in sorted(locked.policy.arms):
        rows = tables["predictions"].loc[tables["predictions"]["arm"].eq(arm), ["date", "stock", *Q]]
        aligned = frame[["date", "stock"]].join(rows.set_index(["date", "stock"]), on=["date", "stock"], validate="one_to_one")
        entry = {"execution": {}}
        if check:
            scored = aligned.join(labels).loc[aligned["score"].notna()].copy()
            scored["scored"] = scored["target"].notna()
            entry.update(predictions=scored, prediction=score_quantiles(scored))
            if not entry["prediction"]["rows"]:
                raise ValueError(f"{arm} 没有已成熟的前瞻标签。")
        for cost, multiple in config["cost_multipliers"].items():
            outcome = simulate(
                frame, aligned["score"], dates, bench, run_id=run_id, strategy=portfolio,
                initial_cash=config["initial_cash"], cost_multiplier=multiple, base_cost_model=rates,
                market_status=tables["market_status"], roster=roster,
            )
            if check:
                sealed = tables["orders"].loc[tables["orders"]["arm"].eq(arm) & tables["orders"]["cost"].eq(cost)]
                _check_orders(outcome["orders"], sealed, dates)
                entry["execution"][cost] = {**outcome["metrics"], **_verify_ledger(outcome, dates)}
                entry[cost] = outcome
            else:
                if outcome["trades"]["date"].eq(str(dates[-1].date())).any():
                    raise ValueError("下一交易日没有行情；预申报不能记为成交。")
                positive = outcome["orders"].loc[outcome["orders"]["requested_shares"].gt(0)]
                order_batches.append(positive.assign(arm=arm, cost=cost).reindex(columns=forward.FIELDS["orders"]))
        outputs[arm] = entry
    planned = pd.concat(order_batches, ignore_index=True) if order_batches else pd.DataFrame(columns=forward.FIELDS["orders"])
    return outputs, planned


def capture(run_id: str, bundle: Path, *, calendar_path: Path | None = forward.CALENDAR) -> Path:
    """Freeze observed same-day market evidence and generate next-open quantities locally."""
    now = forward._now()
    day = now.date()
    if not forward.CLOSE <= now.time() <= forward.DEFAULT_CUTOFF:
        raise ValueError("专家前瞻封存只接受上海当天 15:00 至 18:00 的实际数据。")
    root, _, _ = expert_replay._source(run_id)
    locked = _load_lock(root)
    records, calendar, _ = _captures(locked, None)
    bundle_hash = forward._sha(bundle)
    supplied = forward._json(bundle)
    if supplied["trade_date"] != str(day) or set(supplied["inputs"]) != set(MARKET_ROLES):
        raise ValueError("证据包须声明当日且只提供五项已观察的行情证据。")
    paths, evidence = {}, {}
    destination = root / ROOT / "captures" / str(day)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary)
        shutil.copyfile(bundle, stage / "market_bundle.json")
        if forward._sha(stage / "market_bundle.json") != bundle_hash:
            raise ValueError("复制期间生产者证据包发生变化。")
        for role in MARKET_ROLES:
            entry = supplied["inputs"][role]
            source = (bundle.parent / entry["path"]).resolve(strict=True)
            observed = forward._timestamp(entry["observed_at"])
            if (source.suffix.lower() not in {".csv", ".parquet"} or not entry["source"].strip() or observed > now
                    or (role != "calendar" and (observed.date() != day or observed.time() < forward.CLOSE))):
                raise ValueError(f"{role} 缺少真实可用时间或可核验来源。")
            path = stage / f"{role}{source.suffix.lower()}"
            shutil.copyfile(source, path)
            if forward._sha(path) != entry["sha256"]:
                raise ValueError(f"{role} 不匹配生产者的原始证据哈希。")
            paths[role] = path
            evidence[role] = {"file": path.name, "sha256": entry["sha256"], "source": entry["source"],
                              "observed_at": observed.isoformat()}
        incoming = forward._calendar(paths["calendar"])
        forward._merge_calendar(calendar, incoming)
        if calendar_path is not None:
            official = forward._calendar(calendar_path)
            forward._merge_calendar(official, calendar)
        dates = forward._ordered_sessions(calendar, locked.policy.start)
        number = len(records)
        if number >= COUNT or len(dates) <= number + 1 or dates[number] != day or not incoming.get(day):
            raise ValueError("当天不是连续前瞻中的下一待封存交易日，或缺少下一交易日日历。")
        next_day = dates[number + 1]
        panel = forward._table(paths["panel"], tuple(dict.fromkeys((*PANEL_COLUMNS, *locked.policy.features))))
        predictions = _forecast(locked, panel, now)
        paths.update(predictions=stage / "predictions.parquet", orders=stage / "orders.parquet")
        predictions.to_parquet(paths["predictions"], index=False)
        pd.DataFrame(columns=forward.FIELDS["orders"]).to_parquet(paths["orders"], index=False)
        paths["baseline_roster"] = locked.policy.roster
        forward._verify_inputs(paths, locked.policy, day, next_day, now)
        prior = [_load_tables(_daily_paths(locked, date_, record)) for date_, record in records.items()]
        batch = _load_tables(paths)
        combined = _combine([*prior, batch])
        _, planned = _simulate(locked, combined, pd.DatetimeIndex(dates[:number + 2]), run_id, check=False)
        observed_orders = combined["orders"].sort_values(list(forward.FIELDS["orders"])).reset_index(drop=True)
        fixed_orders = planned.loc[planned["signal_date"].ne(str(day))].sort_values(list(forward.FIELDS["orders"])).reset_index(drop=True)
        for name in ("signal_date", "date"):
            fixed_orders[name] = pd.to_datetime(fixed_orders[name])
        pd.testing.assert_frame_equal(observed_orders, fixed_orders, check_dtype=False)
        planned.loc[planned["signal_date"].eq(str(day))].to_parquet(paths["orders"], index=False)
        forward._verify_inputs(paths, locked.policy, day, next_day, now)
        finished = forward._now()
        if (finished.date() != day or not forward.CLOSE <= finished.time() <= forward.DEFAULT_CUTOFF
                or forward._sha(bundle) != bundle_hash or _load_lock(root).record != locked.record):
            raise ValueError("封存输入已变化，或验证完成时间已超过当日截止。")
        lock_hash = forward._sha(locked.path)
        for role in ("predictions", "orders"):
            path = paths[role]
            evidence[role] = {"file": path.name, "sha256": forward._sha(path), "source": f"frozen_expert:{lock_hash}",
                              "observed_at": finished.isoformat()}
        previous = list(records)[-1] if records else None
        write_json(stage / "manifest.json", {
            "schema_version": 1, "trade_date": str(day), "next_trade_date": str(next_day),
            "captured_at": finished.isoformat(), "lock_sha256": lock_hash, "market_bundle_sha256": bundle_hash,
            "previous_sha256": forward._sha(destination.parent / str(previous) / "manifest.json") if previous else None,
            "files": evidence, "inference": predictions.attrs.get("inference", {}),
        })
        stage.rename(destination)
    return destination / "manifest.json"


def _version(locked: Lock, dates: list[date]) -> dict:
    return {"run_id": locked.source.name, "fold": "expert_forward", "valid_from": str(dates[0]), "valid_through": str(dates[-1]),
            "training_max_label_date": locked.model["training_max_label_date"], "available_at": locked.record["locked_at"]}


def verify_release_evidence(root: Path, report: dict, manifest: dict) -> None:
    """Revalidate the capture and ledger chain before returning any approved expert."""
    locked = _load_lock(root)
    records, _, dates = _captures(locked, None)
    if len(records) != COUNT or report["sessions"] != [str(day) for day in records]:
        raise ValueError("缺少可重验的连续 60 日专家证据。")
    expected = _version(locked, dates[:COUNT])
    if (report.get("schema_version") != 1 or report.get("arm") != locked.model["arm"]
            or manifest.get("model_versions") != [expected] or report.get("lock_sha256") != forward._sha(locked.path)
            or manifest["forward"].get("lock_sha256") != report["lock_sha256"]
            or report.get("pinball_gate") is not True or report.get("risk_gate") is not True
            or report.get("accepted_at") != manifest.get("certified_at")
            or not forward._timestamp(records[dates[COUNT - 1]]["captured_at"])
            <= forward._timestamp(report["accepted_at"]) <= forward._now()):
        raise ValueError("发布版本与冻结前瞻模型不一致。")
    lineage = {str(day): forward._sha(root / ROOT / "captures" / str(day) / "manifest.json") for day in records}
    if report["capture_manifests"] != lineage:
        raise ValueError("资格报告的每日封存证据已变化。")
    expected_files = {f"{arm}_predictions.parquet" for arm in locked.policy.arms}
    expected_files.update(f"{arm}_{cost}_{kind}.parquet" for arm in locked.policy.arms
                          for cost in locked.policy.costs for kind in ("daily", "orders", "trades", "positions"))
    if set(report["artifacts"]) != expected_files:
        raise ValueError("资格报告缺少全套已验证的预测和模拟账本。")
    for filename, digest in report["artifacts"].items():
        if forward._sha(expert_replay._file(root, f"{ROOT}/acceptance/{filename}")) != digest:
            raise ValueError("资格报告的预测或账本哈希不一致。")


def accept(run_id: str, *, calendar_path: Path | None = forward.CALENDAR) -> Path:
    """Recompute sealed forecasts, labels, costs and account entries before issuing eligibility evidence."""
    root, _, _ = expert_replay._source(run_id)
    locked = _load_lock(root)
    records, _, sessions = _captures(locked, calendar_path)
    if len(records) != COUNT:
        raise ValueError(f"前瞻封存有效 {len(records)}/{COUNT} 日，尚不能验收。")
    collected = []
    for day, record in records.items():
        paths = _daily_paths(locked, day, record)
        stamp = forward._timestamp(record["captured_at"])
        forward._verify_inputs({**paths, "baseline_roster": locked.policy.roster}, locked.policy, day,
                               date.fromisoformat(record["next_trade_date"]), stamp)
        data = _load_tables(paths)
        expected = _forecast(locked, data["panel"], stamp).set_index(["stock", "arm"])[Q].sort_index()
        observed = data["predictions"].set_index(["stock", "arm"])[Q].sort_index()
        pd.testing.assert_index_equal(expected.index, observed.index)
        np.testing.assert_allclose(expected.to_numpy(), observed.to_numpy(), atol=1e-7, rtol=0)
        collected.append(data)
    days = sessions[:COUNT]
    evaluated, _ = _simulate(locked, _combine(collected), pd.DatetimeIndex(days), run_id, check=True)
    arm = locked.model["arm"]
    rivals = locked.policy.arms - {arm}
    config = forward._json(locked.policy.config)
    summaries = {name: {kind: results[kind] for kind in ("prediction", "execution")} for name, results in evaluated.items()}
    returns = {cost: {name: equity_returns(results[cost]["daily"], config["initial_cash"]) for name, results in evaluated.items()}
               for cost in locked.policy.costs}
    contrasts = {
        rival: {cost: paired_block_interval([(returns[cost][arm], returns[cost][rival])],
                                            seed=config["seed"], block_size=expert_research.BOOTSTRAP_MAIN)
                for cost in sorted(locked.policy.costs)}
        for rival in sorted(rivals)
    }
    drawdown = expert_research.RISK_ACCEPTANCE["maximum_drawdown"]
    turnover = expert_research.RISK_ACCEPTANCE["maximum_mean_daily_turnover"]
    risk = True
    for cost, metrics in summaries[arm]["execution"].items():
        domestic = summaries["domestic_gpu"]["execution"][cost]
        floor = max(-drawdown, domestic["max_drawdown"]) if arm == "fusion" else -drawdown
        ceiling = min(turnover, domestic["mean_daily_turnover"]) if arm == "fusion" else turnover
        risk &= (metrics["order_count"] > 0 and metrics["max_drawdown"] + 1e-12 >= floor
                 and metrics["mean_daily_turnover"] <= ceiling + 1e-12 and all(
                     (metrics["reject_rate"] or 0) <= (summaries[rival]["execution"][cost]["reject_rate"] or 0) + 1e-12
                     for rival in rivals
                 ))
    cost_and_risk = risk and all(
        result["double"]["ci95_low"] > 0 and all(cost["net_return_difference"] > 0 for cost in result.values())
        for result in contrasts.values()
    )
    lower_pinball = all(summaries[arm]["prediction"]["pinball"] < summaries[rival]["prediction"]["pinball"] for rival in rivals)
    qualified = cost_and_risk and lower_pinball
    destination = root / ROOT / "acceptance"
    report_path = root / ACCEPTANCE
    lineage = {str(day): forward._sha(root / ROOT / "captures" / str(day) / "manifest.json") for day in records}
    run = RunContext(
        task="expert-forward-acceptance", run_id=f"{run_id[:80]}-forward-{stable_hash(run_id)[:12]}",
        as_of_date=str(days[-1]), git_commit=current_git_commit(), seed=config["seed"],
        config_hash=stable_hash({"candidate": run_id, "lock_sha256": forward._sha(locked.path)}),
        data_version=stable_hash(lineage), artifact_root=str(destination),
    )
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        stage = Path(staging)
        for name, result in evaluated.items():
            _write_table(stage / f"{name}_predictions.parquet", result["predictions"])
            for cost in locked.policy.costs:
                for table in ("daily", "orders", "trades", "positions"):
                    _write_table(stage / f"{name}_{cost}_{table}.parquet", result[cost][table])
        artifacts = {path.name: forward._sha(path) for path in sorted(stage.glob("*.parquet"))}
        if not destination.exists():
            stage.rename(destination)
        else:
            for name, sha in artifacts.items():
                if forward._sha(expert_replay._file(root, f"{ROOT}/acceptance/{name}")) != sha:
                    raise ValueError("重新验收产生不同的冻结账本工件。")
    report = {
        "schema_version": 1, "run_id": run_id, "arm": arm, "sessions": [str(day) for day in days],
        "accepted_at": forward._now().isoformat(),
        "qualified_forward_results": qualified, "double_cost_and_risk_gate": cost_and_risk,
        "capture_chain_verified": True, "pinball_gate": lower_pinball, "risk_gate": risk,
        "lock_sha256": forward._sha(locked.path), "capture_manifests": lineage, "arms": summaries, "contrasts": contrasts,
        "max_scoring_signal_sessions": COUNT - forward.LABEL_OFFSET, "artifacts": artifacts, "run": asdict(run),
    }
    if report_path.is_file():
        previous = forward._json(report_path)
        report.update(run=previous["run"], accepted_at=previous["accepted_at"])
    _immutable_json(report_path, report)
    if qualified:
        manifest = {
            "schema_version": 1, "run_id": run_id, "arm": arm, "research_report_sha256": locked.record["report_sha256"],
            "certified_at": report["accepted_at"],
            "forward": {"sha256": forward._sha(report_path), "lock_sha256": report["lock_sha256"]},
            "model_versions": [_version(locked, days)],
        }
        verify_release_evidence(root, report, manifest)
        _immutable_json(root / expert_release.MANIFEST, manifest)
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "lock", "capture", "accept", "promote"):
        command = commands.add_parser(name)
        command.add_argument("run_id")
        if name == "lock":
            command.add_argument("first_session", type=date.fromisoformat)
        elif name == "capture":
            command.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            result = status(args.run_id)
        elif args.command == "lock":
            result = str(lock(args.run_id, args.first_session))
        elif args.command == "capture":
            result = str(capture(args.run_id, args.bundle))
        elif args.command == "accept":
            result = str(accept(args.run_id))
        else:
            evidence = status(args.run_id)
            if evidence["state"] != "qualified":
                blockers = "".join(evidence["blockers"])
                raise ValueError(f"正式发布待验收，前瞻 {evidence['valid_captures']}/{COUNT} 日。{blockers}")
            entry = expert_release.promote_verified(args.run_id)
            result = {"run_id": entry.run_id, "model_id": entry.model_id, "stage": entry.stage}
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"专家前瞻：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
