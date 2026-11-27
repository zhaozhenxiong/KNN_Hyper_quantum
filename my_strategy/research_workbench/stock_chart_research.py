"""Derived, read-only layers for the stock research chart.

The adapter consumes the same raw bars and frozen replay evidence as the stock
page. It never downloads prices, runs inference, invents future sessions, or
promotes mechanical geometry into a trading confirmation.
"""

from __future__ import annotations

import copy
from datetime import date
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import latent_api

SCHEMA = "khquant_stock_research_v1"
PREDICTION_FILE = "reports/stock_research_prediction.json"
PRICES = ("open", "high", "low", "close")


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} 必须是有限数值。")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} 必须是有限数值。") from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} 必须是{'正' if positive else '有限'}数值。")
    return result


def _optional(value: Any) -> float | None:
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _day(value: Any, label: str = "日期") -> str:
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}无效。") from error


def _validate_bars(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    slots, closed, previous = [], [], None
    for index, original in enumerate(rows):
        day = _day(original.get("time"), f"第 {index} 根日线")
        if previous is not None and day <= previous:
            raise ValueError("日线日期必须严格递增且不能重复。")
        previous = day
        item = {"time": day, "status": original.get("status") or "tradable",
                "open": None, "high": None, "low": None, "close": None, "volume": None,
                "is_closed": False}
        values = [original.get(key) for key in PRICES]
        if all(value is None for value in values):
            slots.append(item)
            continue
        if original.get("is_closed", True) is not True:
            raise ValueError("历史图只接受真实闭合日线。")
        for key, value in zip(PRICES, values, strict=True):
            item[key] = _finite(value, f"{day} {key}", positive=True)
        if item["high"] < max(item["open"], item["low"], item["close"]) or item["low"] > min(item["open"], item["high"], item["close"]):
            raise ValueError(f"{day} 的 OHLC 几何不一致。")
        if original.get("volume") is not None:
            item["volume"] = _finite(original["volume"], f"{day} volume")
            if item["volume"] < 0:
                raise ValueError(f"{day} 成交量不能为负。")
        item["is_closed"] = True
        slots.append(item)
        closed.append(copy.deepcopy(item))
    return slots, closed


def _quantile(rows: list[dict], key: str, probability: float) -> float | None:
    values = sorted(row[key] for row in rows if row.get(key) is not None)
    if not values:
        return None
    position = (len(values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def _indicators(bars: list[dict], *, chips: bool) -> dict:
    rows, profiles = [], {}
    for index, bar in enumerate(bars):
        history = bars[:index + 1]
        row = {"time": bar["time"], "is_closed": bar.get("is_closed", True)}
        for window in (5, 10, 20, 60):
            sample = history[-window:]
            row[f"ma{window}"] = sum(item["close"] for item in sample) / window if len(sample) == window else None
        sample = history[-20:]
        if len(sample) == 20:
            values = np.asarray([item["close"] for item in sample], dtype=float)
            deviation = float(values.std(ddof=0))
            row.update(boll_mid=float(values.mean()), boll_upper=float(values.mean() + 2 * deviation),
                       boll_lower=float(values.mean() - 2 * deviation))
        else:
            row.update(boll_mid=None, boll_upper=None, boll_lower=None)
        previous = bars[max(0, index - 20):index]
        row["volume_mean20_prior"] = (sum(item["volume"] for item in previous) / 20
                                      if len(previous) == 20 and all(item.get("volume") is not None for item in previous) else None)
        row["relative_volume20"] = (bar["volume"] / row["volume_mean20_prior"]
                                    if bar.get("volume") is not None and row["volume_mean20_prior"] else None)
        if chips:
            sample = history[-120:]
            weighted = [((item["high"] + item["low"] + item["close"]) / 3, item.get("volume") or 0.0) for item in sample]
            total = sum(weight for _, weight in weighted)
            if total > 0:
                def price_quantile(probability):
                    cumulative = 0.0
                    for price, weight in sorted(weighted):
                        cumulative += weight
                        if cumulative / total >= probability:
                            return price
                    return max(price for price, _ in weighted)
                low, middle, high = (price_quantile(value) for value in (.15, .5, .85))
                row.update(chip70_low=low, chip50=middle, chip70_high=high)
                lower, upper = min(price for price, _ in weighted), max(price for price, _ in weighted)
                edges = np.linspace(lower, upper, 49) if upper > lower else np.asarray([lower - .005, upper + .005])
                masses = np.zeros(len(edges) - 1)
                for price, weight in weighted:
                    masses[min(len(masses) - 1, max(0, int(np.searchsorted(edges, price, side="right") - 1)))] += weight / total
                profiles[bar["time"]] = {
                    "status": "available", "as_of": bar["time"], "chip70_low": low, "chip50": middle, "chip70_high": high,
                    "bins": [{"price_low": float(edges[i]), "price_high": float(edges[i + 1]), "mass_fraction": float(mass)}
                             for i, mass in enumerate(masses)],
                }
            else:
                row.update(chip70_low=None, chip50=None, chip70_high=None)
        rows.append(row)
    return {"rows": rows, "chip_profiles": profiles,
            "chip_method": "rolling_120_closed_bars_hlc3_volume_weighted;not_shareholder_cost"}


def _structure(bars: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    fractals = []
    for index in range(1, len(bars) - 1):
        left, current, right = bars[index - 1:index + 2]
        top = current["high"] > left["high"] and current["high"] > right["high"] and current["low"] > left["low"] and current["low"] > right["low"]
        bottom = current["low"] < left["low"] and current["low"] < right["low"] and current["high"] < left["high"] and current["high"] < right["high"]
        if not (top or bottom) or right.get("is_closed", True) is not True:
            continue
        kind = "top" if top else "bottom"
        field = "high" if top else "low"
        if fractals and fractals[-1]["kind"] == kind:
            field, choose = ("high", max) if top else ("low", min)
            if current[field] != choose(current[field], bars[fractals[-1]["index"]][field]):
                fractals[-1] = {"index": index, "time": current["time"], "kind": kind,
                                "price": current[field], "confirmed_at": right["time"], "is_confirmed": True}
            continue
        fractals.append({"index": index, "time": current["time"], "kind": kind,
                         "price": current[field], "confirmed_at": right["time"], "is_confirmed": True})
    pens = []
    for left, right in zip(fractals, fractals[1:]):
        if right["index"] - left["index"] < 2:
            continue
        pens.append({"start_time": left["time"], "end_time": right["time"], "start_price": left["price"],
                     "end_price": right["price"], "direction": "up" if right["kind"] == "top" else "down",
                     "display_confirmed": True, "confirmation_time": right["confirmed_at"],
                     "start_index": left["index"], "end_index": right["index"]})
    if fractals and fractals[-1]["index"] < len(bars) - 1:
        left = fractals[-1]
        tail = bars[left["index"] + 1:]
        rising = left["kind"] == "bottom"
        end = max(tail, key=lambda row: row["high"]) if rising else min(tail, key=lambda row: row["low"])
        pens.append({"start_time": left["time"], "end_time": end["time"], "start_price": left["price"],
                     "end_price": end["high"] if rising else end["low"], "direction": "up" if rising else "down",
                     "display_confirmed": False, "confirmation_time": None,
                     "start_index": left["index"], "end_index": bars.index(end)})
    zones = []
    for group in zip(pens, pens[1:], pens[2:]):
        if not all(pen["display_confirmed"] for pen in group):
            continue
        ranges = [(min(pen["start_price"], pen["end_price"]), max(pen["start_price"], pen["end_price"])) for pen in group]
        low, high = max(item[0] for item in ranges), min(item[1] for item in ranges)
        if low <= high:
            zones.append({"start_time": group[0]["start_time"], "end_time": group[-1]["end_time"], "low": low, "high": high,
                          "direction": group[-1]["direction"], "display_confirmed": True,
                          "confirmation_time": group[-1]["confirmation_time"]})
    closes = pd.Series([bar["close"] for bar in bars], dtype=float)
    dif = closes.ewm(span=12, adjust=False).mean() - closes.ewm(span=26, adjust=False).mean()
    histogram = (dif - dif.ewm(span=9, adjust=False).mean()) * 2
    divergences = []
    for previous, current in zip(pens, pens[1:]):
        if not current["display_confirmed"] or previous["direction"] != current["direction"]:
            continue
        rising = current["direction"] == "up"
        old_price = max(previous["start_price"], previous["end_price"]) if rising else min(previous["start_price"], previous["end_price"])
        new_price = max(current["start_price"], current["end_price"]) if rising else min(current["start_price"], current["end_price"])
        old_force = float(histogram.iloc[previous["start_index"]:previous["end_index"] + 1].clip(lower=0).sum()) if rising else float(-histogram.iloc[previous["start_index"]:previous["end_index"] + 1].clip(upper=0).sum())
        new_force = float(histogram.iloc[current["start_index"]:current["end_index"] + 1].clip(lower=0).sum()) if rising else float(-histogram.iloc[current["start_index"]:current["end_index"] + 1].clip(upper=0).sum())
        if (new_price > old_price if rising else new_price < old_price) and new_force < old_force:
            divergences.append({"time": current["end_time"], "confirmed_at": current["confirmation_time"], "kind": "top" if rising else "bottom",
                                "label": "顶背驰参考" if rising else "底背驰参考", "previous_price": old_price, "price": new_price,
                                "previous_macd_area": old_force, "macd_area": new_force, "is_confirmed": True})
    for row in fractals:
        row.pop("index", None)
    for row in pens:
        row.pop("start_index", None)
        row.pop("end_index", None)
    return pens, zones, fractals, divergences


def _sources(bars: list[dict], indicators: dict, fractals: list[dict]) -> tuple[list[dict], list[dict]]:
    events = []
    for fractal in fractals:
        events.append({"source": "structure", "date": fractal["time"], "side": "SELL" if fractal["kind"] == "top" else "BUY",
                       "event_type": ("顶" if fractal["kind"] == "top" else "底") + "分型 · 回溯结构",
                       "first_available_at": fractal["confirmed_at"], "confirmed_at": fractal["confirmed_at"],
                       "reference_price": fractal["price"], "confirmed": True, "input_eligible": True,
                       "candidate_eligible": True, "evidence_visible": True, "excluded": False})
    volume_rows = []
    for index, bar in enumerate(bars):
        indicator = indicators["rows"][index]
        previous = bars[max(0, index - 20):index]
        row = {"date": bar["time"], "w_close_location": ((bar["close"] - bar["low"]) / (bar["high"] - bar["low"])
                                                         if bar["high"] > bar["low"] else .5),
               "w_relative_volume20": indicator["relative_volume20"], "wyckoff_event": "none"}
        volume_rows.append(row)
        if len(previous) < 20 or not indicator["relative_volume20"]:
            continue
        low, high = min(item["low"] for item in previous), max(item["high"] for item in previous)
        event = side = None
        if bar["low"] < low < bar["close"] and indicator["relative_volume20"] > 1:
            event, side = "Spring 量价参考", "BUY"
        elif bar["high"] > high > bar["close"] and indicator["relative_volume20"] > 1:
            event, side = "Upthrust 量价参考", "SELL"
        elif bar["close"] > high and indicator["relative_volume20"] >= 1.5:
            event, side = "放量突破参考", "BUY"
        elif bar["close"] < low and indicator["relative_volume20"] >= 1.5:
            event, side = "放量跌破参考", "SELL"
        if event:
            row["wyckoff_event"] = event
            events.append({"source": "wyckoff", "date": bar["time"], "side": side, "event_type": event,
                           "first_available_at": bar["time"], "confirmed_at": bar["time"], "reference_price": bar["close"],
                           "confirmed": True, "input_eligible": True, "candidate_eligible": True, "evidence_visible": True,
                           "excluded": False, "mechanical_rule": "closed_bar_prior_20_geometry_and_relative_volume"})
    return events, volume_rows


def _ict(bars: list[dict], fractals: list[dict]) -> dict:
    gaps, candidates = [], []
    for index in range(2, len(bars)):
        first, middle, third = bars[index - 2:index + 1]
        direction = "bullish" if third["low"] > first["high"] else "bearish" if third["high"] < first["low"] else None
        if direction:
            gaps.append({"direction": direction, "start_index": index - 2, "end_index": index,
                         "candle_dates": [first["time"], middle["time"], third["time"]], "confirmed_time": third["time"],
                         "zone": [first["high"], third["low"]] if direction == "bullish" else [third["high"], first["low"]]})
        for fractal in fractals:
            if fractal["confirmed_at"] >= third["time"]:
                continue
            if fractal["kind"] == "bottom" and third["low"] < fractal["price"] < third["close"]:
                candidates.append({"direction": "long", "sweep": {"time": third["time"], "extreme": third["low"], "reference_price": fractal["price"]},
                                   "mechanical_status": "developing", "reason": "日线几何扫荡候选；MSS、回撤与净盈亏比未确认。"})
            elif fractal["kind"] == "top" and third["high"] > fractal["price"] > third["close"]:
                candidates.append({"direction": "short", "sweep": {"time": third["time"], "extreme": third["high"], "reference_price": fractal["price"]},
                                   "mechanical_status": "developing", "reason": "日线几何扫荡候选；MSS、回撤与净盈亏比未确认。"})
    for gap in gaps:
        lower, upper = gap["zone"]
        bullish = gap["direction"] == "bullish"
        status = "active"
        close_date = None
        for bar in bars[gap["end_index"] + 1:]:
            if bar["low"] <= upper and bar["high"] >= lower:
                status = "touched"
            if (bar["low"] <= lower if bullish else bar["high"] >= upper):
                status = "filled"
            if (bar["close"] < lower if bullish else bar["close"] > upper):
                status, close_date = "closed_through", bar["time"]
                break
        gap["lifecycle"] = {"status": status, "is_active": status in {"active", "touched"}, "close_through_date": close_date,
                            "evaluated_through": bars[-1]["time"], "intrabar_sequence_inferred": False}
    return {"status": "completed" if bars else "unavailable", "final_status": "not_assessed", "recent_fvgs": gaps,
            "latest_fvg": gaps[-1] if gaps else None, "candidates": candidates, "latest_candidate": candidates[-1] if candidates else None,
            "notes": [], "warnings": ["仅使用闭合日线做机械几何扫描；不是分钟级 ICT 入场确认。"]}


def _period(key: str, bar: dict) -> str:
    stamp = pd.Timestamp(bar["time"])
    return f"{stamp.isocalendar().year}-W{stamp.isocalendar().week:02d}" if key == "周线" else bar["time"][:7]


def _aggregate(bars: list[dict], key: str) -> list[dict]:
    groups = {}
    for bar in bars:
        groups.setdefault(_period(key, bar), []).append(bar)
    result = []
    ordered = list(groups.values())
    for index, rows in enumerate(ordered):
        last = rows[-1]
        closed = index < len(ordered) - 1 or (key == "周线" and pd.Timestamp(last["time"]).weekday() == 4)
        result.append({"time": last["time"], "period": _period(key, last), "open": rows[0]["open"],
                       "high": max(row["high"] for row in rows), "low": min(row["low"] for row in rows), "close": last["close"],
                       "volume": sum(row["volume"] or 0 for row in rows) if all(row.get("volume") is not None for row in rows) else None,
                       "is_closed": closed, "status": "tradable" if closed else "forming"})
    return result


def _frozen_prediction(result: dict) -> dict | None:
    if result.get("research_prediction") is not None:
        return result["research_prediction"]
    run_id = result.get("context", {}).get("run_id")
    if not run_id:
        return None
    try:
        folder = latent_api._run_folder(run_id)
        path = latent_api._safe(folder, folder / PREDICTION_FILE)
    except latent_api.RouteError:
        return None
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _empty_prediction(status: str, reason: str) -> dict:
    return {"status": status, "reason": reason, "forecast": [], "dates": [], "samples": [], "quality": {},
            "validation": {"future_sessions_fabricated": False, "inference_triggered_by_chart": False}}


def _path_valid(path: list) -> bool:
    try:
        if len(path) != 5 or any(not isinstance(row, list) or len(row) != 6 for row in path):
            return False
        values = [[_finite(value, "预测样本") for value in row] for row in path]
    except ValueError:
        return False
    return all(row[0] > 0 and row[1] > 0 and row[2] > 0 and row[3] > 0 and row[4] >= 0 and row[5] >= 0
               and row[1] >= max(row[0], row[2], row[3]) and row[2] <= min(row[0], row[1], row[3]) for row in values)


def _prediction(payload: dict | None, symbol: str, bars: list[dict]) -> dict:
    if payload is None:
        return _empty_prediction("not_executed", "当前冻结回放没有已封存的股票预测工件；页面加载不会自动推理。")
    if not isinstance(payload, dict):
        raise ValueError("预测工件必须是 JSON 对象。")
    if payload.get("status") in {"unavailable", "not_executed", "not_requested"}:
        if payload.get("symbol") not in {None, symbol}:
            raise ValueError("预测标的与行情标的不一致。")
        return _empty_prediction("not_executed" if payload["status"] != "unavailable" else "unavailable",
                                 str(payload.get("reason") or "预测未执行。"))
    native = payload.get("schema") == "kcpredict_research_v1"
    forecast_root = payload.get("prediction", payload) if native else payload
    if forecast_root.get("schema") not in {"khquant_stock_prediction_v1", "kronos_forecast_v1"} or forecast_root.get("status") != "completed":
        raise ValueError("预测工件不符合已完成的冻结预测契约。")
    input_info = forecast_root.get("input") or {}
    quality = forecast_root.get("quality") or {}
    if forecast_root.get("symbol") != symbol or input_info.get("symbol", symbol) != symbol:
        raise ValueError("预测标的与行情标的不一致。")
    last = bars[-1]
    for stamp in (forecast_root.get("as_of"), input_info.get("history_end")):
        if stamp is not None and _day(stamp) != last["time"]:
            raise ValueError("预测截止日必须等于最后真实闭合日线。")
    if input_info.get("last_close") is not None and not math.isclose(_finite(input_info["last_close"], "预测末收盘"), last["close"], rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("预测末收盘与真实历史不一致。")
    if type(quality.get("prediction_usable")) is not bool:
        raise ValueError("预测质量必须明确给出 prediction_usable。")
    identity = forecast_root.get("identity") or {}
    horizon = input_info.get("horizon", (identity.get("parameters") or {}).get("horizon"))
    if type(horizon) is not int or horizon != 5:
        raise ValueError("个股图只展示冻结的五个交易日预测。")
    forecast = copy.deepcopy(forecast_root.get("forecast") or [])
    dates = [_day(row.get("date"), "预测日期") for row in forecast]
    if len(forecast) != horizon or dates != sorted(set(dates)) or any(day <= last["time"] for day in dates):
        raise ValueError("预测必须有五个严格位于历史之后的唯一交易日。")
    if identity.get("future_dates") != dates:
        raise ValueError("预测日期必须与冻结交易日历身份完全一致。")
    for row in forecast:
        row["date"] = _day(row["date"])
        for field in (*PRICES, "close_p10", "close_p90"):
            if field in row or field in {"close", "close_p10", "close_p90"}:
                row[field] = _finite(row.get(field), f"预测 {field}", positive=True)
        if row.get("volume") is not None:
            row["volume"] = _finite(row["volume"], "预测 volume")
        if not row["close_p10"] <= row["close"] <= row["close_p90"]:
            raise ValueError("预测收盘价分位数顺序无效。")
        if all(field in row for field in PRICES) and (row["high"] < max(row["open"], row["low"], row["close"]) or row["low"] > min(row["open"], row["high"], row["close"])):
            raise ValueError("预测中位 OHLC 几何无效。")
    mask = quality.get("valid_mask")
    if not isinstance(mask, list) or not mask or not all(type(value) is bool for value in mask):
        raise ValueError("预测质量缺少布尔 valid_mask。")
    count = sum(mask)
    if bool(count) != bool(forecast):
        raise ValueError("有效预测样本数与未来路径不一致。")
    if quality.get("valid_samples") != count or quality.get("requested_samples") != len(mask):
        raise ValueError("预测样本统计与 valid_mask 不一致。")
    if not math.isclose(_finite(quality.get("valid_fraction"), "预测有效率"), count / len(mask), rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("预测有效率与 valid_mask 不一致。")
    minimum = _finite(quality.get("minimum_valid_fraction", .8), "预测有效率门槛")
    if not 0 < minimum <= 1:
        raise ValueError("预测有效率门槛必须位于 0 到 1 之间。")
    numerical = count >= quality.get("minimum_valid_samples", 8) and count / len(mask) >= minimum
    if quality["prediction_usable"] and not numerical:
        raise ValueError("prediction_usable 与样本质量门槛矛盾。")
    samples = copy.deepcopy(forecast_root.get("samples") or [])
    if samples:
        if len(samples) != len(mask) or any(len(path) != horizon for path in samples):
            raise ValueError("预测样本形状与 horizon 不一致。")
        if [_path_valid(path) for path in samples] != mask:
            raise ValueError("预测样本有效性与 quality.valid_mask 不一致。")
        valid_paths = [path for path, valid in zip(samples, mask, strict=True) if valid]
        for index, row in enumerate(forecast):
            closes = [float(path[index][3]) for path in valid_paths]
            for field, probability in (("close", .5), ("close_p10", .1), ("close_p90", .9)):
                expected = float(np.quantile(closes, probability))
                if not math.isclose(row[field], expected, rel_tol=1e-7, abs_tol=1e-7):
                    raise ValueError("预测分位数不能由有效样本复算。")
        samples = valid_paths
    status = "pass" if quality["prediction_usable"] else "fail"
    return {"status": status, "reason": "预测质量通过；仍只是模型情景。" if status == "pass" else "预测质量未通过；只保留采样情景，不升级为可用预测。",
            "forecast": forecast, "dates": dates, "samples": samples, "quality": copy.deepcopy(quality),
            "summary": copy.deepcopy(forecast_root.get("summary")), "parameters": copy.deepcopy(identity.get("parameters") or {}),
            "resource": copy.deepcopy(forecast_root.get("resource") or {}),
            "validation": {"symbol_cutoff_last_close": True, "future_calendar_identity": True,
                           "samples_recomputed": bool(samples), "future_sessions_fabricated": False, "inference_triggered_by_chart": False}}


def build_research(result: dict, prediction_payload: dict | None = None) -> dict:
    symbol = result["symbol"]
    slots, closed = _validate_bars(result.get("bars") or [])
    if not closed:
        return {"schema": SCHEMA, "status": "unavailable", "reason": "所选区间没有真实闭合日线。",
                "meta": {"symbol": symbol, "months": 3, "data_end": None}, "bars": slots,
                "layers": {"frequencies": {}, "source_events": [], "volume_rows": []},
                "prediction": _empty_prediction("unavailable", "缺少真实闭合历史，不能评估预测。")}
    daily_indicators = _indicators(closed, chips=True)
    pens, zones, fractals, divergences = _structure(closed)
    events, volume_rows = _sources(closed, daily_indicators, fractals)
    frequencies = {"日线": {"bars": slots, "pens": pens, "zones": zones, "fractals": fractals,
                           "divergences": divergences, "indicators": daily_indicators}}
    for name in ("周线", "月线"):
        bars = _aggregate(closed, name)
        indicators = _indicators(bars, chips=False)
        p, z, f, d = _structure(bars) if len(bars) >= 3 else ([], [], [], [])
        frequencies[name] = {"bars": bars, "pens": p, "zones": z, "fractals": f, "divergences": d, "indicators": indicators}
    prediction = _prediction(prediction_payload if prediction_payload is not None else _frozen_prediction(result), symbol, closed)
    return {"schema": SCHEMA, "status": "available", "meta": {"symbol": symbol, "months": 3,
            "data_end": closed[-1]["time"], "price_basis": "unadjusted", "timezone": "Asia/Shanghai",
            "closed_bars": len(closed), "calendar_slots": len(slots), "derivation": "local_closed_daily_bars"},
            "bars": slots, "layers": {"frequencies": frequencies, "source_events": events, "volume_rows": volume_rows},
            "ict": _ict(closed, fractals), "prediction": prediction,
            "notices": ["结构与 ICT 为日线机械派生参考，非买卖确认。", "筹码为 120 根闭合日线 HLC/3 成交量加权估算，非真实持仓成本。",
                        "参考结果不代表实际成交；真实信号、模拟成交、部分成交与拒单单独分层。"]}


def attach(result: dict) -> dict:
    """Enrich a chart response without changing its original market/ledger facts."""
    enriched = dict(result)
    try:
        enriched["research"] = build_research(result)
        if enriched["research"].get("prediction", {}).get("status") == "unavailable":
            reason = enriched["research"]["prediction"]["reason"]
            enriched["warnings"] = [*result.get("warnings", []), reason]
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        reason = f"预测工件校验未通过，未展示预测：{error}"
        try:
            fallback = dict(result)
            fallback["research_prediction"] = {"status": "unavailable", "symbol": result.get("symbol"), "reason": reason}
            enriched["research"] = build_research(fallback)
            enriched["research"]["notices"].append(reason)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            enriched["research"] = {"schema": SCHEMA, "status": "unavailable", "reason": f"研究图层暂不可用：{error}",
                                    "meta": {"symbol": result.get("symbol"), "months": 3},
                                    "prediction": _empty_prediction("unavailable", "研究输入校验未通过，未展示预测。")}
        enriched["warnings"] = [*result.get("warnings", []), enriched["research"]["reason"] if enriched["research"]["status"] == "unavailable" else reason]
    return enriched
