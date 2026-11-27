"""Render verified multiview daily and prediction ledgers without changing results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from my_strategy.core.paths import artifact_run_dir
from my_strategy.research_workbench.runner import _write_table


COST = "double"
ARMS = {
    "fusion_rank": "Current fusion · rank",
    "diverse_equal_rank": "Diverse equal · rank",
    "dynamic_fusion_rank": "Dynamic fusion · rank",
    "dynamic_fusion_defensive": "Dynamic + defensive",
    "best_single_rank": "Inner-OOF best single",
}
SCORE_VIEWS = {
    "fusion": "Current fusion",
    "fusion_equal": "Current ML/DL equal",
    "diverse_equal": "Diverse equal",
    "dynamic_fusion": "Dynamic fusion",
}
PALETTES = {
    "light": {
        "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "secondary": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    },
    "dark": {
        "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#f0efec", "secondary": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
    },
}


def _completed_verified(root: Path) -> dict:
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    verification = json.loads((root / "reports/artifact_verification.json").read_text(encoding="utf-8"))
    if (status.get("status") != "completed" or not verification.get("all_daily_dates_unique")
            or not verification.get("all_trade_counts_match")
            or verification.get("ledger_tables") != verification.get("matrix_rows")):
        raise ValueError("Figures require a completed study with verified daily and trade ledgers")
    return json.loads((root / "reports/expert_multiview_study.json").read_text(encoding="utf-8"))


def _daily_frame(root: Path, period: str) -> pd.DataFrame:
    result = None
    for arm in ARMS:
        path = root / "backtests" / period / f"{arm}_{COST}_daily.parquet"
        daily = pd.read_parquet(path, columns=["date", "equity", "market_value", "benchmark"])
        daily["date"] = pd.to_datetime(daily["date"], errors="raise")
        daily = daily.sort_values("date").reset_index(drop=True)
        if daily.empty or daily["date"].duplicated().any() or daily["equity"].le(0).any():
            raise ValueError(f"Invalid daily ledger for {period}/{arm}")
        if not np.isfinite(daily[["equity", "market_value", "benchmark"]].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite daily values for {period}/{arm}")
        frame = pd.DataFrame({"date": daily["date"]})
        nav = daily["equity"] / daily["equity"].iloc[0]
        frame[f"{arm}_nav"] = nav
        frame[f"{arm}_drawdown"] = nav / nav.cummax() - 1
        frame[f"{arm}_exposure"] = daily["market_value"] / daily["equity"]
        benchmark_nav = daily["benchmark"] / daily["benchmark"].iloc[0]
        frame["benchmark_nav"] = benchmark_nav
        if result is None:
            result = frame
        else:
            if not result["date"].equals(frame["date"]) or not np.allclose(
                result["benchmark_nav"], frame["benchmark_nav"], rtol=0, atol=1e-12,
            ):
                raise ValueError(f"Daily calendars or benchmarks differ for {period}")
            result = result.merge(frame.drop(columns="benchmark_nav"), on="date", validate="one_to_one")
    return result


def _score_frame(root: Path, period: str) -> pd.DataFrame:
    values = {}
    for view in SCORE_VIEWS:
        path = root / "backtests" / period / f"{view}_predictions.parquet"
        prediction = pd.read_parquet(path, columns=["date", "stock", "q50"])
        if prediction[["date", "stock"]].duplicated().any():
            raise ValueError(f"Duplicate prediction keys for {period}/{view}")
        score = pd.to_numeric(prediction["q50"], errors="raise")
        if not np.isfinite(score).all():
            raise ValueError(f"Non-finite q50 for {period}/{view}")
        values[view] = score.to_numpy(dtype=float)
    pooled = np.concatenate(list(values.values()))
    edges = np.histogram_bin_edges(pooled, bins=36)
    result = pd.DataFrame({
        "bin_lower": edges[:-1], "bin_upper": edges[1:], "bin_midpoint": (edges[:-1] + edges[1:]) / 2,
    })
    for view, scores in values.items():
        counts, _ = np.histogram(scores, bins=edges)
        result[f"{view}_count"] = counts
        result[f"{view}_density"] = counts / counts.sum()
    return result


def _style(axis, palette: dict) -> None:
    axis.set_facecolor(palette["surface"])
    axis.grid(axis="y", color=palette["grid"], linewidth=0.7, linestyle="-")
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(palette["axis"])
        axis.spines[side].set_linewidth(0.7)
    axis.tick_params(colors=palette["muted"], labelsize=9, length=0, pad=6)
    axis.title.set_color(palette["ink"])
    axis.xaxis.label.set_color(palette["secondary"])
    axis.yaxis.label.set_color(palette["secondary"])


def _render(period: str, daily: pd.DataFrame, scores: pd.DataFrame, destination: Path, mode: str) -> None:
    palette = PALETTES[mode]
    colors = dict(zip([*ARMS, "benchmark"], palette["series"], strict=True))
    with plt.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color": palette["ink"], "axes.labelcolor": palette["secondary"],
        "figure.facecolor": palette["page"], "savefig.facecolor": palette["page"],
    }):
        figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=False)
        figure.subplots_adjust(left=0.065, right=0.98, top=0.77, bottom=0.075, hspace=0.38, wspace=0.16)
        figure.suptitle(f"Multiview expert research · {period}", x=0.065, y=0.965, ha="left", fontsize=22, weight="bold")
        figure.text(0.065, 0.925, "Development/diagnostic evidence only · double transaction costs · no promotion claim",
                    ha="left", fontsize=11, color=palette["secondary"])
        figure.text(0.065, 0.895, "NAV starts at 1.0. q50 is T+1-to-T+6 open excess return versus CSI 300, not absolute return.",
                    ha="left", fontsize=10, color=palette["muted"])
        panels = (("nav", "Net asset value", "NAV"), ("drawdown", "Drawdown", "Drawdown"),
                  ("exposure", "Invested exposure", "Market value / equity"))
        for axis, (metric, title, ylabel) in zip(axes.flat[:3], panels, strict=True):
            _style(axis, palette)
            axis.set_title(title, loc="left", fontsize=14, weight="bold", pad=12)
            axis.set_ylabel(ylabel, fontsize=10)
            for arm, label in ARMS.items():
                axis.plot(daily["date"], daily[f"{arm}_{metric}"], label=label, color=colors[arm], linewidth=2,
                          solid_capstyle="round", solid_joinstyle="round")
            if metric == "nav":
                axis.plot(daily["date"], daily["benchmark_nav"], label="CSI 300 benchmark", color=colors["benchmark"],
                          linewidth=2, solid_capstyle="round")
            else:
                axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.margins(x=0.015)
        distribution = axes.flat[3]
        _style(distribution, palette)
        distribution.set_title("Calibrated q50 distribution", loc="left", fontsize=14, weight="bold", pad=12)
        distribution.set_xlabel("Predicted five-session excess return", fontsize=10)
        distribution.set_ylabel("Share of stock-days per bin", fontsize=10)
        for index, (view, label) in enumerate(SCORE_VIEWS.items()):
            # Four small multiples share bins; facets avoid treating score shape as strategy identity color.
            axis = distribution.inset_axes([0.02 + (index % 2) * 0.52, 0.54 - (index // 2) * 0.53, 0.45, 0.38])
            _style(axis, palette)
            axis.bar(scores["bin_midpoint"], scores[f"{view}_density"],
                     width=(scores["bin_upper"] - scores["bin_lower"]) * 0.88,
                     color=palette["series"][0], alpha=0.82)
            axis.axvline(0, color=palette["axis"], linewidth=1)
            axis.set_title(label, loc="left", fontsize=10, pad=5)
            axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            axis.tick_params(labelsize=7)
        distribution.axis("off")
        handles = [plt.Line2D([0], [0], color=colors[name], linewidth=2, label=label)
                   for name, label in [*ARMS.items(), ("benchmark", "CSI 300 benchmark")]]
        figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.875), ncol=3,
                      frameon=False, fontsize=10, labelcolor=palette["secondary"], handlelength=2.2,
                      columnspacing=1.8)
        figure.text(0.065, 0.025, "Source: verified double-cost daily ledgers and calibrated prediction ledgers. "
                    "Tables: matching *_daily_chart_data.csv and *_score_distribution.csv.",
                    fontsize=9, color=palette["muted"])
        for suffix in ("png", "svg"):
            figure.savefig(destination / f"{period}_research_overview_{mode}.{suffix}", dpi=160 if suffix == "png" else None,
                           bbox_inches="tight")
        plt.close(figure)


def _html(report: dict, periods: list[str]) -> str:
    cards = []
    for period in periods:
        fold = next(item for item in report["folds"] if item["period"] == period)
        start = pd.Timestamp(fold["evaluation_start"]).strftime("%Y-%m-%d")
        end = pd.Timestamp(fold["evaluation_end"]).strftime("%Y-%m-%d")
        cards.append(f"""<section id=\"{period}\"><h2>{period}</h2><p>{start} to {end} · {fold['evaluation_sessions']} sessions · double costs</p>
<picture><source media=\"(prefers-color-scheme: dark)\" srcset=\"{period}_research_overview_dark.png\"><img src=\"{period}_research_overview_light.png\" alt=\"NAV, drawdown, exposure and q50 score distributions for {period}\"></picture>
<p class=\"links\"><a href=\"{period}_daily_chart_data.csv\">Daily chart data</a><a href=\"{period}_score_distribution.csv\">Score distribution data</a><a href=\"{period}_research_overview_light.svg\">SVG</a></p></section>""")
    return """<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Multiview expert research figures</title>
<style>:root{color-scheme:light dark;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--secondary:#52514e;--border:#e1e0d9}@media(prefers-color-scheme:dark){:root{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--secondary:#c3c2b7;--border:#2c2c2a}}*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:40px 24px}h1{font-size:30px;line-height:1.2;margin-bottom:8px}section{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;margin-top:28px}h2{margin:0}p{color:var(--secondary)}img{display:block;width:100%;height:auto;margin-top:20px}.links{display:flex;gap:20px;flex-wrap:wrap}a{color:inherit;text-underline-offset:4px}.notice{border-left:3px solid #eda100;padding-left:14px;max-width:900px}@media(max-width:600px){main{padding:20px 10px}section{padding:14px}}</style></head><body><main><h1>Multiview expert research</h1><p>Verified report figures · research only · double transaction costs</p><p class=\"notice\">No independent prospective holdout or promotion evidence. The historical replay is diagnostic only. q50 forecasts excess return versus CSI 300, not absolute stock return.</p>""" + "".join(cards) + "</main></body></html>"


def generate(root: Path, *, overwrite: bool = False) -> Path:
    report = _completed_verified(root)
    destination = root / "reports" / "figures"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Figure directory already exists: {destination}")
    if overwrite and destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("source_run_id") != report["run"]["run_id"]:
            raise ValueError("Refusing to overwrite figures belonging to another study")
    destination.mkdir(parents=True, exist_ok=overwrite)
    periods = [fold["period"] for fold in report["folds"]]
    for period in periods:
        daily = _daily_frame(root, period)
        scores = _score_frame(root, period)
        _write_table(destination / f"{period}_daily_chart_data.csv", daily)
        _write_table(destination / f"{period}_score_distribution.csv", scores)
        for mode in PALETTES:
            _render(period, daily, scores, destination, mode)
    (destination / "index.html").write_text(_html(report, periods), encoding="utf-8")
    manifest = {
        "source_run_id": report["run"]["run_id"], "cost": COST, "periods": periods,
        "source_tables": ["daily", "predictions"], "modes": list(PALETTES),
        "score_semantics": report["interpretation"]["score"],
        "accessibility": "Every plotted value is available in the matching CSV; light and dark palettes were separately validated.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace only figures previously generated for this same run")
    args = parser.parse_args()
    print(generate(artifact_run_dir(args.run_id, create=False), overwrite=args.overwrite))


if __name__ == "__main__":
    main()
