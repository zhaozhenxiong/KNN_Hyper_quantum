#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared charting helpers for strategy and AI Agent backtests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def setup_chinese_fonts() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "sans-serif"


def apply_auto_date_axis(axes, dates: pd.Series) -> None:
    import matplotlib.dates as mdates

    dates = pd.to_datetime(dates).dropna()
    if dates.empty:
        return
    span_days = max((dates.max() - dates.min()).days, 1)
    if span_days <= 2:
        locator = mdates.HourLocator(interval=1)
        formatter = mdates.DateFormatter("%m-%d %H:%M")
    elif span_days <= 10:
        locator = mdates.HourLocator(interval=4)
        formatter = mdates.DateFormatter("%m-%d %H:%M")
    elif span_days <= 90:
        locator = mdates.DayLocator(interval=5)
        formatter = mdates.DateFormatter("%m-%d")
    elif span_days <= 180:
        locator = mdates.DayLocator(interval=10)
        formatter = mdates.DateFormatter("%m-%d")
    elif span_days <= 365:
        locator = mdates.DayLocator(interval=20)
        formatter = mdates.DateFormatter("%Y-%m-%d")
    elif span_days <= 730:
        locator = mdates.MonthLocator(interval=1)
        formatter = mdates.DateFormatter("%Y-%m")
    else:
        locator = mdates.MonthLocator(interval=3)
        formatter = mdates.DateFormatter("%Y-%m")

    left = dates.min() - pd.Timedelta(days=max(1, min(5, span_days // 30 + 1)))
    right = dates.max() + pd.Timedelta(days=max(1, min(5, span_days // 30 + 1)))
    for ax in axes:
        ax.set_xlim(left, right)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", labelrotation=0)


def plot_intraday_backtest_chart(
    title: str,
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    output_path: str | Path | None = None,
    score_col: str = "score",
):
    import matplotlib.pyplot as plt

    setup_chinese_fonts()
    chart = bars.copy()
    chart["date"] = pd.to_datetime(chart["date"])
    chart["_x"] = range(len(chart))
    trades = trades.copy() if trades is not None else pd.DataFrame()
    if not trades.empty and "date" in trades.columns:
        trades["date"] = pd.to_datetime(trades["date"])
        x_lookup = chart.set_index("date")["_x"].to_dict()
        trades["_x"] = trades["date"].map(x_lookup)
        trades = trades.dropna(subset=["_x"]).copy()

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15, 13),
        gridspec_kw={"height_ratios": [3.1, 1.0, 1.1, 1.0, 1.25]},
        sharex=False,
    )

    ax_price = axes[0]
    x = chart["_x"]
    ax_price.plot(x, chart["close"], color="#3498db", label="分时价格", linewidth=1.0)
    if "vwap" in chart.columns:
        ax_price.plot(x, chart["vwap"], color="#8e44ad", linestyle="--", label="VWAP", linewidth=0.9)
    cost_col = _first_existing_col(chart, ["holding_cost", "avg_cost"])
    if cost_col:
        ax_price.plot(x, pd.to_numeric(chart[cost_col], errors="coerce"), color="#7f8c8d", linestyle="--", label="持仓成本", linewidth=0.9)
    if not trades.empty:
        buys = trades[trades["action"] == "BUY"]
        sells = trades[trades["action"].isin(["SELL", "REDUCE"])]
        if not buys.empty:
            ax_price.scatter(buys["_x"], buys["price"], marker="^", color="#c0392b", s=48, label="买入", zorder=5)
        if not sells.empty:
            ax_price.scatter(sells["_x"], sells["price"], marker="v", color="#1e8449", s=48, label="卖出/减仓", zorder=5)

    norm_equity = chart["equity"] / chart["equity"].iloc[0]
    ax_equity = ax_price.twinx()
    ax_equity.plot(x, norm_equity, color="#e74c3c", label="策略净值", linewidth=1.15, alpha=0.95)
    ax_equity.axhline(1, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.35)
    ax_equity.set_ylabel("策略净值")
    ax_equity.tick_params(axis="y", colors="#c0392b")
    ax_price.set_title(title, fontsize=14, fontweight="bold")
    ax_price.set_ylabel("价格")
    price_handles, price_labels = ax_price.get_legend_handles_labels()
    equity_handles, equity_labels = ax_equity.get_legend_handles_labels()
    ax_price.legend(price_handles + equity_handles, price_labels + equity_labels, loc="upper left")
    ax_price.grid(True, alpha=0.25)

    drawdown = chart["equity"] / chart["equity"].cummax() - 1
    axes[1].fill_between(x, 0, drawdown * 100, color="#e74c3c", alpha=0.25)
    axes[1].plot(x, drawdown * 100, color="#e74c3c", linewidth=0.8)
    axes[1].set_ylabel("回撤(%)")
    axes[1].grid(True, alpha=0.25)

    if score_col in chart.columns:
        axes[2].plot(x, chart[score_col], color="#2c3e50", label="分时分数", linewidth=0.9)
    axes[2].axhline(0, color="#7f8c8d", linestyle="--", linewidth=0.8, alpha=0.45)
    axes[2].set_ylabel("分数")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.25)

    axes[3].bar(x, pd.to_numeric(chart.get("volume", 0), errors="coerce").fillna(0), color="#95a5a6", alpha=0.55, width=0.8)
    axes[3].set_ylabel("成交量")
    axes[3].grid(True, alpha=0.2)

    ax_pos = axes[4]
    position = pd.to_numeric(chart.get("position_weight", 0), errors="coerce").fillna(0)
    position_change = position.diff().fillna(position)
    ax_pos.step(x, position, where="post", color="#27ae60", label="瀹為檯浠撲綅", linewidth=1.1)
    if "target_weight" in chart.columns:
        ax_pos.plot(x, pd.to_numeric(chart["target_weight"], errors="coerce").fillna(0), color="#8e44ad", label="鐩爣浠撲綅", linewidth=0.8, alpha=0.75)
    if "effective_target_weight" in chart.columns:
        ax_pos.plot(x, pd.to_numeric(chart["effective_target_weight"], errors="coerce").fillna(0), color="#16a085", label="鎵ц鐩爣浠撲綅", linewidth=0.8, alpha=0.75)
    ax_pos.bar(x, position_change, color="#f39c12", alpha=0.35, width=0.8, label="浠撲綅鍙樺寲")
    ax_pos.axhline(0, color="#7f8c8d", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_pos.set_ylim(-0.10, 1.00)
    ax_pos.set_ylabel("浠撲綅")
    ax_pos.legend(loc="upper left")
    ax_pos.grid(True, alpha=0.25)

    _apply_intraday_bar_axis(axes, chart)
    fig.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def _apply_intraday_bar_axis(axes, chart: pd.DataFrame) -> None:
    if chart.empty:
        return
    max_ticks = 10
    if len(chart) <= max_ticks:
        tick_idx = list(range(len(chart)))
    else:
        import numpy as np

        tick_idx = sorted(set(np.linspace(0, len(chart) - 1, max_ticks, dtype=int).tolist()))
    dates = pd.to_datetime(chart.iloc[tick_idx]["date"])
    span_days = max((pd.to_datetime(chart["date"]).max() - pd.to_datetime(chart["date"]).min()).days, 0)
    fmt = "%m-%d %H:%M" if span_days <= 2 else "%m-%d\n%H:%M"
    labels = [dt.strftime(fmt) for dt in dates]
    for ax in axes:
        ax.set_xlim(-1, len(chart))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(labels, rotation=0)


def plot_backtest_chart(
    title: str,
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    output_path: str | Path | None = None,
    score_col: str = "score",
    mark_full_cycles_only: bool = True,
):
    import matplotlib.pyplot as plt

    setup_chinese_fonts()
    chart = daily.copy()
    chart["date"] = pd.to_datetime(chart["date"])
    trades = trades.copy() if trades is not None else pd.DataFrame()
    if not trades.empty and "date" in trades.columns:
        trades["date"] = pd.to_datetime(trades["date"])

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 12),
        gridspec_kw={"height_ratios": [3.2, 1.25, 1.1, 1.35]},
        sharex=False,
    )

    ax_price = axes[0]
    ax_price.plot(chart["date"], chart["close"], color="#3498db", label="股价", linewidth=1.15)
    cost_col = _first_existing_col(chart, ["holding_cost", "avg_cost", "entry_price"])
    if cost_col:
        cost = pd.to_numeric(chart[cost_col], errors="coerce")
        ax_price.plot(chart["date"], cost, color="#8e44ad", linestyle="--", label="持仓成本", linewidth=1.1)

    if not trades.empty and {"position_before", "position_after"}.issubset(trades.columns):
        if mark_full_cycles_only:
            buys = trades[(trades["action"] == "BUY") & (trades["position_before"] == 0) & (trades["position_after"] > 0)]
            sells = trades[(trades["action"].isin(["SELL", "REDUCE"])) & (trades["position_before"] > 0) & (trades["position_after"] == 0)]
            buy_label = "完整开仓"
            sell_label = "完整清仓"
        else:
            buys = trades[trades["action"] == "BUY"]
            sells = trades[trades["action"].isin(["SELL", "REDUCE"])]
            buy_label = "买入"
            sell_label = "卖出/减仓"
        if not buys.empty:
            ax_price.scatter(buys["date"], buys["price"], marker="^", color="#c0392b", s=62, label=buy_label, zorder=5)
        if not sells.empty:
            ax_price.scatter(sells["date"], sells["price"], marker="v", color="#1e8449", s=62, label=sell_label, zorder=5)

    norm_equity = chart["equity"] / chart["equity"].iloc[0]
    ax_equity = ax_price.twinx()
    ax_equity.plot(chart["date"], norm_equity, color="#e74c3c", label="策略净值", linewidth=1.3, alpha=0.95)
    ax_equity.axhline(1, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.35)
    ax_equity.set_ylabel("策略净值")
    ax_equity.tick_params(axis="y", colors="#c0392b")

    ax_price.set_title(title, fontsize=14, fontweight="bold")
    ax_price.set_ylabel("价格")
    price_handles, price_labels = ax_price.get_legend_handles_labels()
    equity_handles, equity_labels = ax_equity.get_legend_handles_labels()
    ax_price.legend(price_handles + equity_handles, price_labels + equity_labels, loc="upper left")
    ax_price.grid(True, alpha=0.25)

    drawdown = chart["equity"] / chart["equity"].cummax() - 1
    axes[1].fill_between(chart["date"], 0, drawdown * 100, color="#e74c3c", alpha=0.25)
    axes[1].plot(chart["date"], drawdown * 100, color="#e74c3c", linewidth=0.8)
    axes[1].set_ylabel("回撤(%)")
    axes[1].grid(True, alpha=0.25)

    if score_col in chart.columns:
        axes[2].plot(chart["date"], chart[score_col], color="#2c3e50", label="绛栫暐鍒嗘暟", linewidth=1)
    if "ml_score" in chart.columns:
        axes[2].plot(chart["date"], pd.to_numeric(chart["ml_score"], errors="coerce"), color="#d35400", label="ML鍒嗘暟", linewidth=0.9, alpha=0.9)
    if "ml_score_signal" in chart.columns:
        axes[2].plot(chart["date"], pd.to_numeric(chart["ml_score_signal"], errors="coerce"), color="#c0392b", label="ML Agent", linewidth=0.9, linestyle="--", alpha=0.85)
    axes[2].axhline(0, color="#7f8c8d", linestyle="--", linewidth=0.8, alpha=0.45)
    axes[2].set_ylabel("鍒嗘暟")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.25)

    ax_pos = axes[3]
    position = pd.to_numeric(chart.get("position_weight", 0), errors="coerce").fillna(0)
    position_change = position.diff().fillna(position)
    ax_pos.step(chart["date"], position, where="post", color="#27ae60", label="瀹為檯浠撲綅", linewidth=1.2)
    if "target_weight" in chart.columns:
        ax_pos.plot(chart["date"], pd.to_numeric(chart["target_weight"], errors="coerce").fillna(0), color="#8e44ad", label="鐩爣浠撲綅", linewidth=0.9, alpha=0.75)
    if "effective_target_weight" in chart.columns:
        ax_pos.plot(chart["date"], pd.to_numeric(chart["effective_target_weight"], errors="coerce").fillna(0), color="#16a085", label="鐢熸晥鐩爣浠撲綅", linewidth=0.9, alpha=0.75)
    ax_pos.bar(chart["date"], position_change, color="#f39c12", alpha=0.35, width=1.0, label="浠撲綅鍙樺寲")
    ax_pos.axhline(0, color="#7f8c8d", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_pos.set_ylim(-0.10, 1.00)
    ax_pos.set_ylabel("浠撲綅")
    ax_pos.legend(loc="upper left")
    ax_pos.grid(True, alpha=0.25)

    apply_auto_date_axis(axes, chart["date"])
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_backtest_chart(
    title: str,
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    output_path: str | Path | None = None,
    score_col: str = "score",
    mark_full_cycles_only: bool = True,
):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    setup_chinese_fonts()
    chart = daily.copy()
    chart["date"] = pd.to_datetime(chart["date"])
    trades = trades.copy() if trades is not None else pd.DataFrame()
    if not trades.empty and "date" in trades.columns:
        trades["date"] = pd.to_datetime(trades["date"])

    equity = pd.to_numeric(chart["equity"], errors="coerce")
    close = pd.to_numeric(chart["close"], errors="coerce")
    norm_equity = equity / equity.iloc[0]
    drawdown = equity / equity.cummax() - 1
    metrics = _calc_chart_metrics(chart, trades)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 12),
        gridspec_kw={"height_ratios": [3.5, 1.15, 1.15, 1.4]},
        sharex=False,
    )
    fig.patch.set_facecolor("#fbfcfd")

    ax_price = axes[0]
    ax_price.set_facecolor("#ffffff")
    ax_price.plot(chart["date"], close, color="#1f4e79", label="股价", linewidth=1.35)
    cost_col = _first_existing_col(chart, ["holding_cost", "avg_cost", "entry_price"])
    if cost_col:
        cost = pd.to_numeric(chart[cost_col], errors="coerce")
        ax_price.plot(chart["date"], cost, color="#6b7280", linestyle="--", label="持仓成本", linewidth=1.05, alpha=0.95)

    if not trades.empty and {"position_before", "position_after"}.issubset(trades.columns):
        if mark_full_cycles_only:
            buys = trades[(trades["action"] == "BUY") & (trades["position_before"] == 0) & (trades["position_after"] > 0)]
            sells = trades[(trades["action"].isin(["SELL", "REDUCE"])) & (trades["position_before"] > 0) & (trades["position_after"] == 0)]
            buy_label = "完整开仓"
            sell_label = "完整清仓"
        else:
            buys = trades[trades["action"] == "BUY"]
            sells = trades[trades["action"].isin(["SELL", "REDUCE"])]
            buy_label = "买入"
            sell_label = "卖出/减仓"
        if not buys.empty:
            ax_price.scatter(buys["date"], buys["price"], marker="^", color="#d62728", edgecolor="white", linewidth=0.55, s=76, label=buy_label, zorder=6)
        if not sells.empty:
            ax_price.scatter(sells["date"], sells["price"], marker="v", color="#2ca02c", edgecolor="white", linewidth=0.55, s=76, label=sell_label, zorder=6)

    ax_equity = ax_price.twinx()
    ax_equity.plot(chart["date"], norm_equity, color="#f39c12", label="策略净值", linewidth=1.55, alpha=0.98)
    ax_equity.axhline(1, color="#f39c12", linestyle="--", linewidth=0.8, alpha=0.35)
    ax_equity.set_ylabel("策略净值")
    ax_equity.tick_params(axis="y", colors="#b9770e")

    start_date = chart["date"].min().strftime("%Y-%m-%d")
    end_date = chart["date"].max().strftime("%Y-%m-%d")
    subtitle = (
        f"总收益 {metrics['total_return']:.2%}   年化 {metrics['annual_return']:.2%}   "
        f"最大回撤 {metrics['max_drawdown']:.2%}   Sharpe {metrics['sharpe']:.2f}"
    )
    ax_price.set_title(f"{title}  |  {start_date} ~ {end_date}\n{subtitle}", fontsize=14, fontweight="bold", pad=12)
    ax_price.set_ylabel("价格")
    price_handles, price_labels = ax_price.get_legend_handles_labels()
    equity_handles, equity_labels = ax_equity.get_legend_handles_labels()
    ax_price.legend(price_handles + equity_handles, price_labels + equity_labels, loc="upper left", frameon=True, framealpha=0.9)
    ax_price.grid(True, alpha=0.22, linestyle="-", linewidth=0.7)
    _add_summary_box(ax_price, chart, metrics)

    ax_dd = axes[1]
    ax_dd.set_facecolor("#ffffff")
    ax_dd.fill_between(chart["date"], 0, drawdown * 100, color="#e74c3c", alpha=0.22)
    ax_dd.plot(chart["date"], drawdown * 100, color="#c0392b", linewidth=0.85)
    ax_dd.set_ylabel("回撤(%)")
    ax_dd.grid(True, alpha=0.22, linestyle="-", linewidth=0.7)

    ax_score = axes[2]
    ax_score.set_facecolor("#ffffff")
    if score_col in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart[score_col], errors="coerce"), color="#263238", label="策略分数", linewidth=1.05)
    if "ml_score" in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart["ml_score"], errors="coerce"), color="#d35400", label="ML原始分", linewidth=0.9, alpha=0.86)
    if "ml_score_signal" in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart["ml_score_signal"], errors="coerce"), color="#8e44ad", label="ML有效分", linewidth=0.95, linestyle="--", alpha=0.9)
    if "dl_score" in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart["dl_score"], errors="coerce"), color="#00897b", label="DL原始分", linewidth=0.9, alpha=0.82)
    if "dl_score_signal" in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart["dl_score_signal"], errors="coerce"), color="#3949ab", label="DL有效分", linewidth=0.95, linestyle="--", alpha=0.86)
    if "prob_score" in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart["prob_score"], errors="coerce"), color="#0369a1", label="Prob原始分", linewidth=0.9, alpha=0.82)
    if "prob_score_signal" in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart["prob_score_signal"], errors="coerce"), color="#be123c", label="Prob有效分", linewidth=0.95, linestyle="-.", alpha=0.86)
    if "entry_probability" in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart["entry_probability"], errors="coerce"), color="#dc2626", label="入场概率", linewidth=0.95, alpha=0.78)
    if "exit_probability" in chart.columns:
        ax_score.plot(chart["date"], -pd.to_numeric(chart["exit_probability"], errors="coerce"), color="#16a34a", label="离场概率(-)", linewidth=0.95, alpha=0.78)
    ax_score.axhline(0, color="#7f8c8d", linestyle="--", linewidth=0.8, alpha=0.45)
    ax_score.axhline(0.35, color="#d62728", linestyle=":", linewidth=0.8, alpha=0.35)
    ax_score.axhline(-0.25, color="#2ca02c", linestyle=":", linewidth=0.8, alpha=0.35)
    ax_score.set_ylim(-1.05, 1.05)
    ax_score.set_ylabel("分数")
    ax_score.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax_score.grid(True, alpha=0.22, linestyle="-", linewidth=0.7)

    ax_pos = axes[3]
    ax_pos.set_facecolor("#ffffff")
    position = pd.to_numeric(chart.get("position_weight", 0), errors="coerce").fillna(0)
    position_change = position.diff().fillna(position)
    ax_pos.step(chart["date"], position, where="post", color="#1e8449", label="实际仓位", linewidth=1.25)
    if "target_weight" in chart.columns:
        ax_pos.plot(chart["date"], pd.to_numeric(chart["target_weight"], errors="coerce").fillna(0), color="#8e44ad", label="目标仓位", linewidth=0.9, alpha=0.75)
    if "effective_target_weight" in chart.columns:
        ax_pos.plot(chart["date"], pd.to_numeric(chart["effective_target_weight"], errors="coerce").fillna(0), color="#16a085", label="鐢熸晥鐩爣浠撲綅", linewidth=0.9, alpha=0.75)
    ax_pos.bar(chart["date"], position_change, color="#f39c12", alpha=0.30, width=1.0, label="浠撲綅鍙樺寲")
    ax_pos.axhline(0, color="#7f8c8d", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_pos.set_ylim(-0.10, 1.02)
    ax_pos.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_pos.set_ylabel("浠撲綅")
    ax_pos.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax_pos.grid(True, alpha=0.22, linestyle="-", linewidth=0.7)

    apply_auto_date_axis(axes, chart["date"])
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=30)
        for spine in ["top", "right"]:
            ax.spines[spine].set_alpha(0.18)
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    return fig


def _calc_chart_metrics(chart: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float]:
    equity = pd.to_numeric(chart["equity"], errors="coerce").dropna()
    if equity.empty:
        return {"total_return": 0.0, "annual_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "n_trades": 0.0, "win_rate": 0.0}
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    days = max((pd.to_datetime(chart["date"]).max() - pd.to_datetime(chart["date"]).min()).days, 1)
    annual_return = (1 + total_return) ** (365 / days) - 1
    drawdown = equity / equity.cummax() - 1
    daily_ret = equity.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * (252 ** 0.5)) if len(daily_ret) > 2 and daily_ret.std() else 0.0
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": float(drawdown.min()),
        "sharpe": float(sharpe),
        "n_trades": float(len(trades)) if trades is not None else 0.0,
        "win_rate": float(_estimate_trade_win_rate(trades)),
    }


def _estimate_trade_win_rate(trades: pd.DataFrame) -> float:
    if trades is None or trades.empty or "action" not in trades.columns:
        return 0.0
    wins = 0
    closed = 0
    entry_price = None
    for _, row in trades.sort_values("date").iterrows():
        action = str(row.get("action", ""))
        price = pd.to_numeric(pd.Series([row.get("price")]), errors="coerce").iloc[0]
        if pd.isna(price):
            continue
        if action == "BUY" and entry_price is None:
            entry_price = float(price)
        elif action in {"SELL", "REDUCE"} and entry_price is not None:
            closed += 1
            wins += int(float(price) > entry_price)
            if float(row.get("position_after", 0) or 0) == 0:
                entry_price = None
    return wins / closed if closed else 0.0


def _add_summary_box(ax, chart: pd.DataFrame, metrics: dict[str, float]) -> None:
    latest = chart.iloc[-1]
    final_equity = pd.to_numeric(pd.Series([latest.get("equity")]), errors="coerce").iloc[0]
    latest_action = str(latest.get("final_action", "-"))
    latest_position = float(pd.to_numeric(pd.Series([latest.get("position_weight", 0)]), errors="coerce").fillna(0).iloc[0])
    ml_signal = latest.get("ml_score_signal", latest.get("ml_score", ""))
    dl_signal = latest.get("dl_score_signal", latest.get("dl_score", ""))
    prob_signal = latest.get("prob_score_signal", latest.get("prob_score", ""))
    entry_probability = latest.get("entry_probability", "")
    try:
        ml_text = f"{float(ml_signal):.3f}"
    except Exception:
        ml_text = "-"
    try:
        dl_text = f"{float(dl_signal):.3f}"
    except Exception:
        dl_text = "-"
    try:
        prob_text = f"{float(prob_signal):.3f}"
    except Exception:
        prob_text = "-"
    try:
        entry_text = f"{float(entry_probability):.1%}"
    except Exception:
        entry_text = "-"
    text = (
        f"鏈€缁堣祫浜? {final_equity:,.0f}\n"
        f"浜ゆ槗娆℃暟: {int(metrics['n_trades'])}\n"
        f"鑳滅巼: {metrics['win_rate']:.1%}\n"
        f"鏈€鏂板姩浣? {latest_action}\n"
        f"鏈€鏂颁粨浣? {latest_position:.0%}\n"
        f"入场概率: {entry_text}\n"
        f"Prob鏈夋晥鍒? {prob_text}\n"
        f"DL鏈夋晥鍒? {dl_text}\n"
        f"ML鏈夋晥鍒? {ml_text}"
    )
    ax.text(
        0.012,
        0.68,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#ffffff", "edgecolor": "#d0d7de", "alpha": 0.92},
    )


def plot_strategy_result_from_files(output_root: str | Path, stock: str, strategy_name: str):
    output_root = Path(output_root)
    tag = f"{stock.replace('.', '_')}_{strategy_name}"
    daily_path = output_root / "signals" / f"{tag}_daily.csv"
    trades_path = output_root / "signals" / f"{tag}_trades.csv"
    if not daily_path.exists():
        raise FileNotFoundError(daily_path)
    daily = pd.read_csv(daily_path)
    trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()

    stock_name = ""
    ranking_path = output_root / "reports" / "strategy_ranking.csv"
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path)
        hit = ranking[(ranking["stock"].eq(stock)) & (ranking["strategy_name"].eq(strategy_name))]
        if not hit.empty and "stock_name" in hit.columns:
            stock_name = str(hit.iloc[0].get("stock_name", "") or "")
    title = f"{stock} {stock_name} - {strategy_name}".strip()
    return plot_backtest_chart(title, daily, trades, score_col="score")


def _first_existing_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            values = pd.to_numeric(df[name], errors="coerce")
            if values.notna().any():
                return name
    return None


def _has_ohlc(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in ["open", "high", "low", "close"])


def _candle_width_days(dates: pd.Series) -> float:
    import numpy as np
    import matplotlib.dates as mdates

    nums = mdates.date2num(pd.to_datetime(dates))
    if len(nums) <= 1:
        return 0.55
    diffs = np.diff(sorted(nums))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.55
    return float(max(0.25, min(0.85, np.median(diffs) * 0.65)))


def _draw_candlesticks(ax, chart: pd.DataFrame) -> None:
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

    width = _candle_width_days(chart["date"])
    up_color = "#d62728"      # A鑲′範鎯細绾㈡定
    down_color = "#1f9d55"    # 缁胯穼
    flat_color = "#64748b"
    dates = mdates.date2num(pd.to_datetime(chart["date"]))

    o = pd.to_numeric(chart["open"], errors="coerce")
    h = pd.to_numeric(chart["high"], errors="coerce")
    l = pd.to_numeric(chart["low"], errors="coerce")
    c = pd.to_numeric(chart["close"], errors="coerce")

    for x, open_, high_, low_, close_ in zip(dates, o, h, l, c):
        if pd.isna(open_) or pd.isna(high_) or pd.isna(low_) or pd.isna(close_):
            continue
        color = up_color if close_ > open_ else down_color if close_ < open_ else flat_color
        ax.vlines(x, low_, high_, color=color, linewidth=0.9, alpha=0.92, zorder=2)
        bottom = min(open_, close_)
        height = abs(close_ - open_)
        if height == 0:
            height = max(abs(close_) * 0.001, 0.01)
            bottom -= height / 2
        ax.add_patch(
            Rectangle(
                (x - width / 2, bottom),
                width,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.82,
                zorder=3,
            )
        )
    ax.plot([], [], color=up_color, linewidth=3, label="K绾?绾㈡定)")
    ax.plot([], [], color=down_color, linewidth=3, label="K绾?缁胯穼)")


def _add_summary_box_pretty(ax, chart: pd.DataFrame, metrics: dict[str, float]) -> None:
    latest = chart.iloc[-1]
    final_equity = pd.to_numeric(pd.Series([latest.get("equity")]), errors="coerce").iloc[0]
    latest_action = str(latest.get("final_action", "-"))
    latest_position = float(pd.to_numeric(pd.Series([latest.get("position_weight", 0)]), errors="coerce").fillna(0).iloc[0])
    ml_signal = latest.get("ml_score_signal", latest.get("ml_score", ""))
    dl_signal = latest.get("dl_score_signal", latest.get("dl_score", ""))
    prob_signal = latest.get("prob_score_signal", latest.get("prob_score", ""))
    entry_probability = latest.get("entry_probability", "")
    try:
        ml_text = f"{float(ml_signal):.3f}"
    except Exception:
        ml_text = "-"
    try:
        dl_text = f"{float(dl_signal):.3f}"
    except Exception:
        dl_text = "-"
    try:
        prob_text = f"{float(prob_signal):.3f}"
    except Exception:
        prob_text = "-"
    try:
        entry_text = f"{float(entry_probability):.1%}"
    except Exception:
        entry_text = "-"
    text = (
        f"鏈€缁堣祫浜? {final_equity:,.0f}\n"
        f"浜ゆ槗娆℃暟: {int(metrics['n_trades'])}\n"
        f"鑳滅巼: {metrics['win_rate']:.1%}\n"
        f"鏈€鏂板姩浣? {latest_action}\n"
        f"鏈€鏂颁粨浣? {latest_position:.0%}\n"
        f"入场概率: {entry_text}\n"
        f"Prob鏈夋晥鍒? {prob_text}\n"
        f"DL鏈夋晥鍒? {dl_text}\n"
        f"ML鏈夋晥鍒? {ml_text}"
    )
    ax.text(
        0.012,
        0.70,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.3,
        linespacing=1.32,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#ffffff", "edgecolor": "#d0d7de", "alpha": 0.92},
        zorder=8,
    )


def plot_backtest_chart(
    title: str,
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    output_path: str | Path | None = None,
    score_col: str = "score",
    mark_full_cycles_only: bool = True,
):
    """Render the final daily backtest chart with candlesticks when OHLC data exists."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    setup_chinese_fonts()
    chart = daily.copy()
    chart["date"] = pd.to_datetime(chart["date"])
    trades = trades.copy() if trades is not None else pd.DataFrame()
    if not trades.empty and "date" in trades.columns:
        trades["date"] = pd.to_datetime(trades["date"])

    equity = pd.to_numeric(chart["equity"], errors="coerce")
    close = pd.to_numeric(chart["close"], errors="coerce")
    norm_equity = equity / equity.iloc[0]
    drawdown = equity / equity.cummax() - 1
    metrics = _calc_chart_metrics(chart, trades)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16.5, 12.2),
        gridspec_kw={"height_ratios": [3.7, 1.05, 1.15, 1.35]},
        sharex=False,
    )
    fig.patch.set_facecolor("#f6f8fa")

    ax_price = axes[0]
    ax_price.set_facecolor("#ffffff")
    if _has_ohlc(chart):
        _draw_candlesticks(ax_price, chart)
    else:
        ax_price.plot(chart["date"], close, color="#1f4e79", label="收盘价", linewidth=1.35)
    if "ma20" in chart.columns:
        ax_price.plot(chart["date"], pd.to_numeric(chart["ma20"], errors="coerce"), color="#2563eb", label="MA20", linewidth=0.95, alpha=0.78)
    if "ma60" in chart.columns:
        ax_price.plot(chart["date"], pd.to_numeric(chart["ma60"], errors="coerce"), color="#7c3aed", label="MA60", linewidth=0.95, alpha=0.72)

    cost_col = _first_existing_col(chart, ["holding_cost", "avg_cost", "entry_price"])
    if cost_col:
        cost = pd.to_numeric(chart[cost_col], errors="coerce")
        ax_price.plot(chart["date"], cost, color="#6b7280", linestyle="--", label="持仓成本", linewidth=1.05, alpha=0.95)

    if not trades.empty and {"position_before", "position_after"}.issubset(trades.columns):
        full_opens, add_buys, full_closes, reduce_sells = _split_trade_markers(trades)
        all_buys = pd.concat([full_opens, add_buys], ignore_index=True, sort=False)
        all_sells = pd.concat([full_closes, reduce_sells], ignore_index=True, sort=False)
        _draw_trade_markers(ax_price, full_opens, add_buys, full_closes, reduce_sells)
        _annotate_latest_trade_price(ax_price, all_buys, all_sells)

    ax_equity = ax_price.twinx()
    ax_equity.plot(chart["date"], norm_equity, color="#f59e0b", label="策略净值", linewidth=1.55, alpha=0.98)
    ax_equity.axhline(1, color="#f59e0b", linestyle="--", linewidth=0.8, alpha=0.35)
    ax_equity.set_ylabel("策略净值")
    ax_equity.tick_params(axis="y", colors="#b45309")
    ax_equity.spines["right"].set_color("#f59e0b")

    start_date = chart["date"].min().strftime("%Y-%m-%d")
    end_date = chart["date"].max().strftime("%Y-%m-%d")
    subtitle = (
        f"总收益 {metrics['total_return']:.2%}   年化 {metrics['annual_return']:.2%}   "
        f"最大回撤 {metrics['max_drawdown']:.2%}   Sharpe {metrics['sharpe']:.2f}"
    )
    ax_price.set_title(f"{title}  |  {start_date} ~ {end_date}\n{subtitle}", fontsize=14, fontweight="bold", pad=12)
    ax_price.set_ylabel("价格")
    price_handles, price_labels = ax_price.get_legend_handles_labels()
    equity_handles, equity_labels = ax_equity.get_legend_handles_labels()
    ax_price.legend(price_handles + equity_handles, price_labels + equity_labels, loc="upper left", frameon=True, framealpha=0.92, ncol=3)
    ax_price.grid(True, alpha=0.18, linestyle="-", linewidth=0.7)
    _add_summary_box_pretty(ax_price, chart, metrics)

    ax_dd = axes[1]
    ax_dd.set_facecolor("#ffffff")
    ax_dd.fill_between(chart["date"], 0, drawdown * 100, color="#ef4444", alpha=0.20)
    ax_dd.plot(chart["date"], drawdown * 100, color="#b91c1c", linewidth=0.85)
    ax_dd.set_ylabel("回撤(%)")
    ax_dd.grid(True, alpha=0.18, linestyle="-", linewidth=0.7)

    ax_score = axes[2]
    ax_score.set_facecolor("#ffffff")
    if score_col in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart[score_col], errors="coerce"), color="#111827", label="策略分数", linewidth=1.05)
    if "ml_score" in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart["ml_score"], errors="coerce"), color="#d97706", label="ML原始分", linewidth=0.9, alpha=0.86)
    if "ml_score_signal" in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart["ml_score_signal"], errors="coerce"), color="#9333ea", label="ML有效分", linewidth=0.95, linestyle="--", alpha=0.9)
    if "dl_score" in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart["dl_score"], errors="coerce"), color="#0f766e", label="DL原始分", linewidth=0.9, alpha=0.82)
    if "dl_score_signal" in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart["dl_score_signal"], errors="coerce"), color="#4338ca", label="DL有效分", linewidth=0.95, linestyle="--", alpha=0.86)
    if "prob_score" in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart["prob_score"], errors="coerce"), color="#0369a1", label="Prob原始分", linewidth=0.9, alpha=0.82)
    if "prob_score_signal" in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart["prob_score_signal"], errors="coerce"), color="#be123c", label="Prob有效分", linewidth=0.95, linestyle="-.", alpha=0.86)
    if "entry_probability" in chart.columns:
        ax_score.plot(chart["date"], pd.to_numeric(chart["entry_probability"], errors="coerce"), color="#dc2626", label="入场概率", linewidth=0.95, alpha=0.78)
    if "exit_probability" in chart.columns:
        ax_score.plot(chart["date"], -pd.to_numeric(chart["exit_probability"], errors="coerce"), color="#16a34a", label="离场概率(-)", linewidth=0.95, alpha=0.78)
    ax_score.axhline(0, color="#64748b", linestyle="--", linewidth=0.8, alpha=0.45)
    ax_score.axhline(0.35, color="#dc2626", linestyle=":", linewidth=0.8, alpha=0.35)
    ax_score.axhline(-0.25, color="#16a34a", linestyle=":", linewidth=0.8, alpha=0.35)
    ax_score.set_ylim(-1.05, 1.05)
    ax_score.set_ylabel("分数")
    ax_score.legend(loc="upper left", frameon=True, framealpha=0.9, ncol=3)
    ax_score.grid(True, alpha=0.18, linestyle="-", linewidth=0.7)

    ax_pos = axes[3]
    ax_pos.set_facecolor("#ffffff")
    position = pd.to_numeric(chart.get("position_weight", 0), errors="coerce").fillna(0)
    position_change = position.diff().fillna(position)
    ax_pos.step(chart["date"], position, where="post", color="#15803d", label="实际仓位", linewidth=1.25)
    if "target_weight" in chart.columns:
        ax_pos.plot(chart["date"], pd.to_numeric(chart["target_weight"], errors="coerce").fillna(0), color="#9333ea", label="目标仓位", linewidth=0.9, alpha=0.75)
    if "effective_target_weight" in chart.columns:
        ax_pos.plot(chart["date"], pd.to_numeric(chart["effective_target_weight"], errors="coerce").fillna(0), color="#0f766e", label="生效目标仓位", linewidth=0.9, alpha=0.75)
    ax_pos.bar(chart["date"], position_change, color="#f59e0b", alpha=0.30, width=1.0, label="仓位变化")
    ax_pos.axhline(0, color="#64748b", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_pos.set_ylim(-0.10, 1.02)
    ax_pos.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_pos.set_ylabel("仓位")
    ax_pos.legend(loc="upper left", frameon=True, framealpha=0.9, ncol=4)
    ax_pos.grid(True, alpha=0.18, linestyle="-", linewidth=0.7)

    apply_auto_date_axis(axes, chart["date"])
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=30)
        for spine in ["top", "right"]:
            ax.spines[spine].set_alpha(0.18)
        ax.spines["left"].set_alpha(0.28)
        ax.spines["bottom"].set_alpha(0.28)
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
    return fig


def _split_trade_markers(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = trades.copy()
    data["position_before"] = pd.to_numeric(data["position_before"], errors="coerce").fillna(0)
    data["position_after"] = pd.to_numeric(data["position_after"], errors="coerce").fillna(0)
    buys = data[data["action"] == "BUY"].copy()
    sells = data[data["action"].isin(["SELL", "REDUCE"])].copy()
    full_opens = buys[(buys["position_before"] == 0) & (buys["position_after"] > 0)].copy()
    add_buys = buys[~buys.index.isin(full_opens.index)].copy()
    full_closes = sells[(sells["position_before"] > 0) & (sells["position_after"] == 0)].copy()
    reduce_sells = sells[~sells.index.isin(full_closes.index)].copy()
    return full_opens, add_buys, full_closes, reduce_sells


def _draw_trade_markers(
    ax,
    full_opens: pd.DataFrame,
    add_buys: pd.DataFrame,
    full_closes: pd.DataFrame,
    reduce_sells: pd.DataFrame,
) -> None:
    if not full_opens.empty:
        ax.scatter(
            full_opens["date"],
            full_opens["price"],
            marker="$B$",
            color="#d62728",
            edgecolor="white",
            linewidth=0.65,
            s=130,
            label="B 完整开仓",
            zorder=8,
        )
    if not add_buys.empty:
        ax.scatter(
            add_buys["date"],
            add_buys["price"],
            marker="^",
            color="#d62728",
            edgecolor="white",
            linewidth=0.65,
            s=78,
            label="加仓",
            zorder=7,
        )
    if not full_closes.empty:
        ax.scatter(
            full_closes["date"],
            full_closes["price"],
            marker="$S$",
            color="#1f9d55",
            edgecolor="white",
            linewidth=0.65,
            s=130,
            label="S 完整清仓",
            zorder=8,
        )
    if not reduce_sells.empty:
        ax.scatter(
            reduce_sells["date"],
            reduce_sells["price"],
            marker="v",
            color="#1f9d55",
            edgecolor="white",
            linewidth=0.65,
            s=78,
            label="减仓",
            zorder=7,
        )


def _annotate_latest_trade_price(ax, buys: pd.DataFrame, sells: pd.DataFrame) -> None:
    frames = []
    if buys is not None and not buys.empty:
        buy_rows = buys.copy()
        buy_rows["_trade_label"] = "买入"
        buy_rows["_trade_color"] = "#d62728"
        frames.append(buy_rows)
    if sells is not None and not sells.empty:
        sell_rows = sells.copy()
        sell_rows["_trade_label"] = "卖出"
        sell_rows["_trade_color"] = "#1f9d55"
        frames.append(sell_rows)
    if not frames:
        return

    trades = pd.concat(frames, ignore_index=True, sort=False)
    trades["date"] = pd.to_datetime(trades["date"], errors="coerce")
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades = trades.dropna(subset=["date", "price"]).sort_values("date")
    if trades.empty:
        return

    latest = trades.iloc[-1]
    label = f"最近{latest['_trade_label']}: {latest['price']:.2f}"
    color = latest["_trade_color"]
    y_offset = 22 if latest["_trade_label"] == "涔板叆" else -28
    ax.annotate(
        label,
        xy=(latest["date"], latest["price"]),
        xytext=(12, y_offset),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9.5,
        color=color,
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0, "alpha": 0.85},
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#ffffff", "edgecolor": color, "alpha": 0.93},
        zorder=9,
    )


