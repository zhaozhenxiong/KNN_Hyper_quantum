"use strict";

(() => {
  const root = $("#strategy-sectors");
  if (!root) {
    window.SectorResonance = { show() {}, clear() {}, refresh() {} };
    return;
  }

  const statusNode = $("#sector-status");
  const dateInput = $("#sector-date");
  const sortInput = $("#sector-sort");
  const themeButton = $("#sector-theme");
  const summaryNode = $("#sector-summary");
  const overviewNode = $("#sector-overview");
  const detailNode = $("#sector-detail");
  const downloadsNode = $("#sector-downloads");
  const palettes = {
    light: { teal: "#1baf7a", amber: "#eb6834", slate: "#2a78d6", danger: "#e34948", muted: "#898781" },
    dark: { teal: "#199e70", amber: "#d95926", slate: "#3987e5", danger: "#e66767", muted: "#898781" },
  };
  const colors = { ...palettes.light };
  const isoDay = /^\d{4}-\d{2}-\d{2}$/;
  const themeKey = "khquant.strategy.sector-theme.v1";
  const downloadNames = {
    sector_daily: "板块日表",
    selection_candidates: "股票候选明细",
    sector_candidates: "行业候选汇总",
    report: "板块共振报告",
    chart_csv: "行业图表源数据",
  };
  const selectionReasons = {
    NOT_REBALANCE_DAY: "非调仓日：购买候选按 0 记录，不沿用上一调仓日。",
    END_OF_EVALUATION: "评估期末：没有下一执行交易日，不生成新订单意图。",
    NO_SESSION_ROWS: "当日没有可用会话行。",
    NO_POINT_IN_TIME_MEMBERSHIP: "此日期没有该行业的点时成分快照，行业候选数不可拆分；实际候选保留在未分类桶。",
    NO_FINITE_SCORES: "当日没有有限分数。",
    NO_ELIGIBLE_ROWS: "当日没有符合基础条件的股票行。",
    NO_QUALIFIED_CANDIDATES: "没有满足当前选择模式的合格候选。",
    DEFENSIVE_CASH: "防守暴露为 0，账户保持现金。",
    NO_TARGET_WEIGHTS: "现金、集中度和持仓约束后没有正目标权重。",
    NO_ORDER_INTENTS: "目标与当前持仓相比没有需要下单的差异。",
  };
  const local = {
    run: "",
    generation: 0,
    detailRequest: 0,
    controllers: new Set(),
    overview: null,
    dates: [],
    date: "",
    sort: "trend",
    selected: "",
    chartPayload: null,
    chartRows: [],
    charts: [],
    readoutDay: "",
    syncing: false,
    syncingRange: false,
    theme: readTheme(),
  };

  function readTheme() {
    try {
      const saved = JSON.parse(window.localStorage.getItem(themeKey));
      return saved === "dark" ? "dark" : "light";
    } catch { return "light"; }
  }

  function finite(value) { return Number.isFinite(num(value)); }
  function positive(value) { return finite(value) && num(value) > 0; }
  function nonNegative(value) { return finite(value) && num(value) >= 0; }
  function flag(value) {
    if (value === true || value === 1) return true;
    return typeof value === "string" && ["1", "true", "yes", "y"].includes(value.trim().toLowerCase());
  }

  function rawIndustry(row) {
    const value = row?.industry === null || row?.industry === undefined ? "" : String(row.industry).trim();
    return value || "unknown";
  }

  function unknownIndustry(row) {
    return ["", "unknown", "__unknown__", "unclassified", "none", "null", "n/a", "na", "未知", "未分类"]
      .includes(String(row?.industry ?? "").trim().toLowerCase());
  }

  function industryLabel(row) {
    if (unknownIndustry(row)) return "未分类";
    const name = String(row?.industry_name || "").trim();
    const code = rawIndustry(row);
    return name && name !== code ? `${name} · ${code}` : code;
  }

  function nativeThs() { return (local.chartPayload || local.overview)?.provider === "ths_fuyao"; }

  function basisLabel(value) {
    return ({ native_archive: "原生封存板块数据", recomputed_features: "由同一冻结输入确定性复算" })[value] || value || "未注明";
  }

  function resonanceKind(value) {
    const state = String(value ?? "").trim().toLowerCase();
    if (!state) return "";
    const strong = state.includes("strong") || state.includes("强");
    if (!strong) return "";
    if (state.includes("down") || state.includes("negative") || state.includes("下降") || state.includes("下跌")) return "down";
    if (state.includes("up") || state.includes("positive") || state.includes("上升") || state.includes("上涨")) return "up";
    return "";
  }

  function resonanceText(value) {
    const raw = String(value ?? "").trim();
    if (!raw) return "—";
    const key = raw.toLowerCase();
    const labels = {
      strong_up: "强上升市场共振", strong_uptrend: "强上升市场共振", strong_up_resonance: "强上升市场共振",
      strong_upward_resonance: "强上升市场共振", strong_positive: "强上升市场共振",
      strong_down: "强下降市场共振", strong_downtrend: "强下降市场共振", strong_down_resonance: "强下降市场共振",
      strong_downward_resonance: "强下降市场共振", strong_negative: "强下降市场共振",
      up: "上升趋势", uptrend: "上升趋势", up_trend: "上升趋势", rising: "上升趋势",
      down: "下降趋势", downtrend: "下降趋势", down_trend: "下降趋势", falling: "下降趋势",
      neutral: "中性／无强共振", none: "无强共振", insufficient: "证据不足", insufficient_evidence: "证据不足",
      unknown: "未知", not_rebalance_day: "非调仓日",
    };
    return labels[key] || raw;
  }

  function resonanceHtml(value) {
    const kind = resonanceKind(value);
    const key = String(value ?? "").toLowerCase();
    const tone = kind || (key.includes("up") || key.includes("上升") ? "up" : key.includes("down") || key.includes("下降") ? "down" : "neutral");
    return `<span class="sector-state ${tone}"><i aria-hidden="true"></i>${h(resonanceText(value))}</span>`;
  }

  function confirmationText(value, candidates) {
    if (value === true) return "有策略确认";
    if (value === false) return "无策略确认";
    const raw = String(value ?? "").trim();
    if (!raw) return "—";
    const key = raw.toLowerCase();
    const labels = {
      confirmed: "有策略确认", yes: "有策略确认", true: "有策略确认", "1": "有策略确认",
      not_confirmed: "无策略确认", unconfirmed: "无策略确认", no: "无策略确认", false: "无策略确认", "0": "无策略确认",
      insufficient: "证据不足", insufficient_evidence: "证据不足", unknown: "证据不足",
      not_rebalance_day: "非调仓日",
    };
    if (labels[key]) return labels[key];
    if (["有", "有策略确认", "无", "无策略确认", "证据不足", "非调仓日"].includes(raw)) return raw;
    return raw || (num(candidates) > 0 ? "有候选，确认证据未注明" : "—");
  }

  function signed(value, formatter) {
    if (!finite(value)) return "—";
    const number = num(value);
    return `<span class="sector-signed ${number > 0 ? "up" : number < 0 ? "down" : "flat"}">${h(`${number > 0 ? "+" : ""}${formatter(number)}`)}</span>`;
  }

  function signedPct(value) { return signed(value, pct); }
  function signedScore(value) { return signed(value, (number) => decimals(number, 0)); }

  function chartDate(time) {
    if (typeof time === "string") return isoDay.test(time) ? time : "";
    if (time && typeof time === "object") {
      const day = `${time.year}-${String(time.month).padStart(2, "0")}-${String(time.day).padStart(2, "0")}`;
      return isoDay.test(day) ? day : "";
    }
    return "";
  }

  function tableRegion(label, contents) {
    return `<div class="sector-table-region" role="region" aria-label="${h(label)}" tabindex="0">${contents}</div>`;
  }

  function safeDownload(path) {
    return typeof path === "string" && /^\/api\/(?!\/)[^\s\\]+$/.test(path) ? path : "";
  }

  function fileKind(path) {
    const clean = String(path).split("?", 1)[0].toLowerCase();
    if (clean.endsWith(".parquet")) return "Parquet";
    if (clean.endsWith(".csv")) return "CSV";
    if (clean.endsWith(".json")) return "JSON";
    if (clean.endsWith(".html")) return "HTML";
    return "文件";
  }

  function downloadLinks(downloads, allowed) {
    return allowed.map((key) => {
      const path = safeDownload(downloads?.[key]);
      if (!path) return "";
      return `<a class="table-button sector-download" href="${h(path)}" download>${h(downloadNames[key] || key)}<small>${h(fileKind(path))}</small></a>`;
    }).join("");
  }

  function invalidate() {
    local.generation += 1;
    local.detailRequest += 1;
    for (const controller of local.controllers) controller.abort();
    local.controllers.clear();
  }

  async function request(path, generation) {
    const controller = new AbortController();
    local.controllers.add(controller);
    try { return await api(path, { signal: controller.signal }); }
    finally { local.controllers.delete(controller); }
  }

  function current(generation, run = local.run) {
    return generation === local.generation && Boolean(run) && run === local.run;
  }

  function destroyCharts() {
    for (const record of local.charts) {
      try { record.chart.remove(); } catch { /* The chart host may already be detached. */ }
    }
    local.charts = [];
    local.syncing = false;
    local.syncingRange = false;
  }

  function applyTheme() {
    root.dataset.sectorTheme = local.theme;
    themeButton.setAttribute("aria-pressed", String(local.theme === "dark"));
    themeButton.setAttribute("aria-label", local.theme === "dark" ? "切换板块区域为浅色模式" : "切换板块区域为暗色模式");
    themeButton.textContent = local.theme === "dark" ? "切换浅色" : "切换暗色";
    Object.assign(colors, palettes[local.theme]);
    if (local.charts.length) renderCharts();
  }

  function chartThemeOptions() {
    const dark = local.theme === "dark";
    return {
      layout: {
        background: { color: dark ? "#1b2723" : "#fcfdfa" },
        textColor: dark ? "#b8c7bf" : "#566760",
        attributionLogo: false,
        fontSize: 11,
      },
      grid: {
        vertLines: { color: dark ? "#26362f" : "#edf1ee" },
        horzLines: { color: dark ? "#2b3b34" : "#e7ede8" },
      },
      rightPriceScale: { borderColor: dark ? "#43544c" : "#d5ded8" },
      timeScale: { borderColor: dark ? "#43544c" : "#d5ded8", timeVisible: false },
      crosshair: {
        vertLine: { color: colors.muted, labelBackgroundColor: colors.slate },
        horzLine: { color: colors.muted, labelBackgroundColor: colors.slate },
      },
    };
  }

  function populateDates(payload) {
    local.dates = [...new Set(arr(payload.dates).filter((day) => isoDay.test(day)))].sort();
    const responseDate = isoDay.test(payload.date) ? payload.date : "";
    const selectedDate = local.dates.includes(local.date) ? local.date : "";
    local.date = responseDate || selectedDate || local.dates.at(-1) || "";
    dateInput.value = local.date;
    if (local.dates.length) {
      dateInput.min = local.dates[0];
      dateInput.max = local.dates.at(-1);
      dateInput.setAttribute("list", "sector-date-options");
      let choices = $("#sector-date-options");
      if (!choices) {
        choices = document.createElement("datalist");
        choices.id = "sector-date-options";
        root.append(choices);
      }
      choices.replaceChildren(...local.dates.map((day) => option(day, day)));
    } else {
      dateInput.removeAttribute("min");
      dateInput.removeAttribute("max");
      dateInput.removeAttribute("list");
      $("#sector-date-options")?.remove();
    }
  }

  function nonRebalanceNote(payload) {
    const texts = [payload?.reason, payload?.notice, ...arr(payload?.sectors).flatMap((row) => [row?.strategy_confirmation, row?.selection_reason])];
    return texts.some((value) => String(value ?? "").toUpperCase().includes("NOT_REBALANCE_DAY"));
  }

  function renderSummary(payload) {
    const totals = payload.totals || {};
    const items = [
      ["合格分数股票", totals.qualified_count, "有限分数且满足当前选择模式；排名模式下不等于建议买入。"],
      ["Top-N 购买候选", totals.purchase_candidate_count, "全市场统一排序后的集中度前候选，尚未应用行业／单板仓位上限。"],
      ["实际目标", totals.target_count, "现金、防守暴露、集中度和持仓约束后，目标权重大于 0 的股票。"],
      ["买入订单意图", totals.buy_intent_count, "目标与当前持仓比较后形成的买入／加仓意图，不是成交。"],
      ["实际订单", totals.order_count, "执行链订单账本记录，可能与意图和成交不同。"],
      ["模拟成交", totals.fill_count, "实际成交账本记录，不回填为信号候选。"],
    ];
    summaryNode.innerHTML = `<div class="sector-basis"><strong>全市场 · ${h(payload.date || local.date || "未注明日期")}</strong><span>${h(basisLabel(payload.market_basis))}</span></div>
      <dl class="sector-totals">${items.map(([label, value, definition]) => `<div><dt>${h(label)}</dt><dd>${h(count(value))}</dd><dd class="sector-definition">${h(definition)}</dd></div>`).join("")}</dl>
      <p class="sector-day-reason" data-sector-day-reason ${nonRebalanceNote(payload) ? "" : "hidden"}><code>NOT_REBALANCE_DAY</code> · ${h(selectionReasons.NOT_REBALANCE_DAY)}</p>
      ${payload.notice ? `<p class="sector-notice">${h(payload.notice)}</p>` : ""}
      <details class="sector-definitions"><summary>核对数量口径</summary><p>合格分数 → 全市场 Top-N 购买候选（集中度前） → 实际目标（集中度后） → 买入意图 → 订单 → 成交。各层不能互相替代；非调仓日购买候选为 0，只有账本明确返回 <code>NOT_REBALANCE_DAY</code> 时才据此标注，不从零值猜测原因。</p></details>`;
  }

  function sparkline(row) {
    const values = arr(row.sparkline).filter((point) => isoDay.test(point?.date) && finite(point?.value));
    if (!values.length) return '<span class="sector-no-spark">—</span>';
    const width = 92;
    const height = 30;
    const numbers = values.map((point) => num(point.value));
    let minimum = Math.min(...numbers);
    let maximum = Math.max(...numbers);
    if (minimum === maximum) { minimum -= 0.5; maximum += 0.5; }
    const coordinates = values.map((point, index) => {
      const x = values.length === 1 ? width / 2 : 2 + index * (width - 4) / (values.length - 1);
      const y = height - 3 - (num(point.value) - minimum) * (height - 6) / (maximum - minimum);
      return [x, y];
    });
    const path = coordinates.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
    const last = values.at(-1);
    const title = `${industryLabel(row)}近期序列，截至 ${last.date}：${decimals(last.value, 3)}`;
    return `<svg class="sector-spark" viewBox="0 0 ${width} ${height}" role="img" tabindex="0" aria-label="${h(title)}"><title>${h(title)}</title><path d="${path}" fill="none" stroke="${colors.teal}" stroke-width="1.7" vector-effect="non-scaling-stroke"/><circle cx="${coordinates.at(-1)[0].toFixed(2)}" cy="${coordinates.at(-1)[1].toFixed(2)}" r="2.2" fill="${colors.teal}"/></svg>`;
  }

  function sortedSectors() {
    return arr(local.overview?.sectors).slice().sort((a, b) => {
      const unknown = Number(unknownIndustry(a)) - Number(unknownIndustry(b));
      if (unknown) return unknown;
      const primary = local.sort === "candidates" ? num(b.purchase_candidate_count) - num(a.purchase_candidate_count) : num(b.market_trend_score) - num(a.market_trend_score);
      if (Number.isFinite(primary) && primary) return primary;
      const secondary = local.sort === "candidates" ? num(b.market_trend_score) - num(a.market_trend_score) : num(b.purchase_candidate_count) - num(a.purchase_candidate_count);
      if (Number.isFinite(secondary) && secondary) return secondary;
      return industryLabel(a).localeCompare(industryLabel(b), "zh-CN");
    });
  }

  function renderOverview() {
    const payload = local.overview;
    if (!payload) return;
    if (payload.status !== "available") {
      overviewNode.innerHTML = empty("该归档没有可用的板块数据", `${payload.reason || "此归档未封存板块明细。"} 不会用最新数据伪造补齐。`);
      return;
    }
    const sectors = sortedSectors();
    if (!sectors.length) {
      overviewNode.innerHTML = empty("此日期没有行业记录", "全市场汇总仍按 API 返回值展示；可切换其它归档交易日。");
      return;
    }
    const rows = sectors.map((row) => {
      const raw = rawIndustry(row);
      const label = industryLabel(row);
      const participants = num(row.price_participant_count);
      const constituents = num(row.constituent_count);
      const coverage = finite(participants) && finite(constituents) && constituents > 0 ? participants / constituents : NaN;
      return [
        `<button type="button" class="table-button sector-industry-link" data-sector-industry="${h(raw)}" aria-pressed="${String(local.selected === raw)}" aria-label="查看 ${h(label)} 行业详情">${h(label)}</button>${unknownIndustry(row) ? '<small class="sector-unranked">不参与排名</small>' : ""}`,
        signedScore(row.market_trend_score), resonanceHtml(row.resonance_state), signedPct(row.relative_strength_20d),
        h(pct(row.breadth_above_ma20)), signedPct(row.amount_change_5d), h(decimals(row.activity_ratio, 3)),
        h(count(row.qualified_count)), h(count(row.purchase_candidate_count)), h(count(row.target_count)),
        `<span class="sector-coverage">${h(count(row.price_participant_count))}／${h(count(row.constituent_count))}<small>${h(pct(coverage))} 有价覆盖</small></span>`,
        h(confirmationText(row.strategy_confirmation, row.purchase_candidate_count)), sparkline(row),
      ];
    });
    overviewNode.innerHTML = tableRegion("行业共振总览表，可横向滚动", scrollTable([
      ["行业"], ["趋势分", "numeric"], ["市场共振状态"], ["20日相对强度", "numeric"], ["MA20上宽度", "numeric"],
      ["5日成交额变化", "numeric"], ["活跃度", "numeric"], ["合格分数", "numeric"], ["Top-N候选", "numeric"],
      ["实际目标", "numeric"], ["有价／成分覆盖", "numeric"], ["策略确认"], ["近期序列"],
    ], rows));
  }

  function renderOverviewDownloads(payload) {
    const links = downloadLinks(payload.downloads, ["sector_daily", "selection_candidates", "sector_candidates", "report"]);
    downloadsNode.innerHTML = links || '<span class="sector-note">此归档未提供板块下载文件。</span>';
  }

  function sectorFor(industry) {
    return arr(local.overview?.sectors).find((row) => rawIndustry(row) === industry) || null;
  }

  function updateDayReason(rows) {
    const reasons = [...new Set(arr(rows).map((row) => String(row.selection_reason || "").toUpperCase()))];
    if (!reasons.includes("NOT_REBALANCE_DAY")) return;
    const note = summaryNode.querySelector("[data-sector-day-reason]");
    if (note) note.hidden = false;
  }

  async function loadOverview({ preserveSelection = false } = {}) {
    const generation = local.generation;
    const run = local.run;
    const selected = preserveSelection ? local.selected : "";
    const params = new URLSearchParams({ account: "full_pool" });
    if (local.date) params.set("date", local.date);
    root.setAttribute("aria-busy", "true");
    overviewNode.setAttribute("aria-busy", "true");
    statusNode.textContent = "正在读取全市场板块总览…";
    if (!preserveSelection) overviewNode.innerHTML = empty("正在读取板块总览", "只读取当前回测的全池账户，不会重新运行回测。");
    else overviewNode.classList.add("sector-loading");
    try {
      const payload = await request(`/api/expert/replays/${encodeURIComponent(run)}/sectors?${params}`, generation);
      if (!current(generation, run)) return;
      local.overview = payload;
      const sourceNote = $("#sector-source-note");
      if (sourceNote) sourceNote.textContent = payload.provider === "ths_fuyao"
        ? "同花顺行业分类 · 原生行业指数 · 当前成分快照不回填历史"
        : "点时 CAPCO 行业 · 等权合成指数 · 候选与实际目标分层";
      populateDates(payload);
      renderSummary(payload);
      renderOverviewDownloads(payload);
      renderOverview();
      if (payload.status === "available") {
        statusNode.textContent = `${local.date || "未注明日期"} · ${count(arr(payload.sectors).length)} 条行业记录 · ${basisLabel(payload.market_basis)}`;
      } else statusNode.textContent = `板块数据不可用：${payload.reason || "旧归档未封存所需账本。"}`;
      if (selected && payload.status === "available" && sectorFor(selected)) {
        local.selected = selected;
        renderOverview();
        loadDetail();
      } else {
        local.selected = "";
        detailNode.replaceChildren();
      }
    } catch (error) {
      if (!current(generation, run) || error.name === "AbortError") return;
      local.overview = null;
      summaryNode.replaceChildren();
      downloadsNode.replaceChildren();
      detailNode.replaceChildren();
      overviewNode.innerHTML = empty("板块总览读取失败", error.message);
      statusNode.textContent = error.message;
    } finally {
      if (current(generation, run)) {
        root.removeAttribute("aria-busy");
        overviewNode.removeAttribute("aria-busy");
        overviewNode.classList.remove("sector-loading");
      }
    }
  }

  function detailSkeleton(industry) {
    const label = industryLabel(sectorFor(industry) || { industry });
    const basis = nativeThs() ? "同花顺原生行业指数；历史成分宽度不可用，不以当前成分回填。" : "等权后复权合成指数仅作研究，不是可交易价格。";
    detailNode.innerHTML = `<div class="sector-detail-head"><div><h3>${h(label)}</h3><p>行业下钻 · ${h(local.date || "当前日期")} · ${h(basis)}</p></div><button type="button" class="button button-quiet" data-sector-close>关闭行业详情</button></div>
      <p class="sector-detail-status" data-sector-chart-status role="status">正在读取行业图表…</p>
      <div class="sector-downloads sector-detail-downloads" data-sector-chart-downloads></div>
      <div class="sector-charts" data-sector-charts aria-busy="true">${empty("正在准备六个图表面板", "图表加载期间仍可查看下方真实账本。")}</div>
      <section class="sector-subsection" aria-labelledby="sector-candidate-heading"><div class="sector-subsection-head"><h4 id="sector-candidate-heading">候选股票与执行对账</h4><span>${h(local.date || "—")}</span></div>
        <div data-sector-candidate-summary></div><p class="sector-detail-status" data-sector-candidate-status role="status">正在读取候选股票…</p><div data-sector-candidate-table></div></section>
      <section class="sector-subsection" aria-labelledby="sector-daily-heading"><div class="sector-subsection-head"><h4 id="sector-daily-heading">行业逐日真实数据表</h4><span>所有图表值均可在此核对</span></div><div data-sector-daily-table>${empty("正在读取行业日表", "不会用插值填补 null 或 NaN。")}</div></section>
      <p class="sector-attribution"><a href="https://www.tradingview.com/" target="_blank" rel="noopener noreferrer">TradingView Lightweight Charts™</a> Copyright (c) 2023 TradingView, Inc.</p>`;
  }

  function candidateReasonHtml(rows) {
    const reasons = [...new Set(arr(rows).map((row) => String(row.selection_reason || "").trim()).filter(Boolean))];
    if (!reasons.length) return "";
    return `<div class="sector-selection-reasons"><strong>选择账本原因</strong>${reasons.slice(0, 4).map((reason) => `<p><code>${h(reason)}</code> · ${h(selectionReasons[reason] || reason)}</p>`).join("")}</div>`;
  }

  function sumField(rows, key) {
    let found = false;
    let total = 0;
    for (const row of rows) {
      if (!finite(row[key])) continue;
      found = true;
      total += num(row[key]);
    }
    return found ? total : null;
  }

  function reconciliation(rows, industry) {
    const expected = sectorFor(industry);
    if (!expected) return '<p class="sector-reconcile warn">总览中找不到此行业，无法执行候选数量对账。</p>';
    if (nativeThs() && expected.selection_reason === "NO_POINT_IN_TIME_MEMBERSHIP") {
      return '<div class="sector-reconcile unknown"><strong>点时成分不可用，不执行行业候选对账</strong><p>该日期没有此行业的历史成分快照，不能把行业候选数写成 0；实际候选与执行数量保留在未分类桶并对账。</p></div>';
    }
    const actual = {
      qualified_count: rows.filter((row) => flag(row.qualified)).length,
      purchase_candidate_count: rows.filter((row) => flag(row.purchase_candidate)).length,
      target_count: rows.filter((row) => finite(row.target_weight) && num(row.target_weight) > 0).length,
      buy_intent_count: rows.filter((row) => ["BUY", "ADD"].includes(String(row.intent_side || "").toUpperCase())).length,
      order_count: sumField(rows, "order_count"),
      fill_count: sumField(rows, "fill_count"),
    };
    const labels = {
      qualified_count: "合格分数", purchase_candidate_count: "Top-N 候选", target_count: "实际目标",
      buy_intent_count: "买入意图", order_count: "订单", fill_count: "成交",
    };
    const checks = Object.entries(labels).map(([key, label]) => {
      const want = finite(expected[key]) ? num(expected[key]) : null;
      const got = actual[key];
      const match = want !== null && got !== null && want === got;
      const comparable = want !== null && got !== null;
      return `<li class="${!comparable ? "unknown" : match ? "match" : "mismatch"}"><span>${h(label)}</span><strong>明细 ${h(got === null ? "—" : count(got))}／总览 ${h(want === null ? "—" : count(want))}</strong><small>${!comparable ? "无法比较" : match ? "一致" : "不一致"}</small></li>`;
    }).join("");
    const comparable = Object.keys(labels).filter((key) => finite(expected[key]) && actual[key] !== null);
    const passed = comparable.length && comparable.every((key) => num(expected[key]) === actual[key]);
    return `<div class="sector-reconcile ${passed ? "match" : "warn"}"><strong>${passed ? "候选与执行数量已对账" : "对账结果需核查"}</strong><ul>${checks}</ul><p>Top-N 按 <code>purchase_candidate</code> 标记计，实际目标按 <code>target_weight &gt; 0</code> 计；订单与成交为明细行合计。</p></div>`;
  }

  function intentLabel(value) {
    return ({ BUY: "买入", ADD: "加仓", REDUCE: "减仓", SELL: "卖出" })[String(value || "").toUpperCase()] || value || "—";
  }

  function renderCandidates(payload, industry) {
    const summary = detailNode.querySelector("[data-sector-candidate-summary]");
    const table = detailNode.querySelector("[data-sector-candidate-table]");
    const status = detailNode.querySelector("[data-sector-candidate-status]");
    if (!summary || !table || !status) return;
    if (payload.status !== "available") {
      summary.innerHTML = "";
      table.innerHTML = empty("此行业候选明细不可用", `${payload.reason || "旧归档没有封存股票候选明细。"} 不会伪造候选、订单或成交。`);
      status.textContent = payload.reason || "候选明细不可用。";
      return;
    }
    const rows = arr(payload.rows).slice().sort((a, b) => {
      const rankA = finite(a.global_rank) ? num(a.global_rank) : Number.POSITIVE_INFINITY;
      const rankB = finite(b.global_rank) ? num(b.global_rank) : Number.POSITIVE_INFINITY;
      return rankA - rankB || String(a.stock || "").localeCompare(String(b.stock || ""), "zh-CN");
    });
    updateDayReason(rows);
    summary.innerHTML = reconciliation(rows, industry) + candidateReasonHtml(rows);
    status.textContent = `${payload.date || local.date || "未注明日期"} · 返回 ${count(rows.length)} 行股票候选账本。`;
    if (!rows.length) {
      table.innerHTML = empty("此行业当日没有候选账本行", "零值原因以选择账本和总览为准，不从其他日期沿用候选。");
      return;
    }
    table.innerHTML = tableRegion(`${industryLabel({ industry })}候选股票表，可横向滚动`, scrollTable([
      ["股票", "code"], ["全市场排名", "numeric"], ["分数", "numeric"], ["合格"], ["Top-N候选"], ["当前持股", "numeric"],
      ["目标权重", "numeric"], ["意图"], ["选择原因"], ["订单", "numeric"], ["成交", "numeric"],
    ], rows.map((row) => [
      h(row.stock), h(finite(row.global_rank) ? count(row.global_rank) : "—"), h(decimals(row.score, 6)),
      h(row.qualified === null || row.qualified === undefined ? "—" : flag(row.qualified) ? "合格" : "未合格"),
      h(row.purchase_candidate === null || row.purchase_candidate === undefined ? "—" : flag(row.purchase_candidate) ? "Top-N" : "否"),
      h(count(row.current_shares)), h(pct(row.target_weight)), h(intentLabel(row.intent_side)),
      h(selectionReasons[row.selection_reason] || row.selection_reason || (flag(row.purchase_candidate) ? "进入全市场 Top-N（集中度前）" : "—")),
      h(count(row.order_count)), h(count(row.fill_count)),
    ])));
  }

  async function loadCandidates(generation, detailRequest, industry) {
    const run = local.run;
    const params = new URLSearchParams({ account: "full_pool" });
    if (local.date) params.set("date", local.date);
    try {
      const payload = await request(`/api/expert/replays/${encodeURIComponent(run)}/sectors/${encodeURIComponent(industry)}/candidates?${params}`, generation);
      if (!current(generation, run) || detailRequest !== local.detailRequest || industry !== local.selected) return;
      renderCandidates(payload, industry);
    } catch (error) {
      if (!current(generation, run) || detailRequest !== local.detailRequest || error.name === "AbortError") return;
      const status = detailNode.querySelector("[data-sector-candidate-status]");
      const table = detailNode.querySelector("[data-sector-candidate-table]");
      if (status) status.textContent = error.message;
      if (table) table.innerHTML = empty("候选股票读取失败", error.message);
    }
  }

  function cleanChartRows(payload) {
    const byDay = new Map();
    for (const row of arr(payload.series)) if (isoDay.test(row?.date)) byDay.set(row.date, row);
    return [...byDay.values()].sort((a, b) => a.date.localeCompare(b.date));
  }

  function renderDailyTable(rows) {
    const target = detailNode.querySelector("[data-sector-daily-table]");
    if (!target) return;
    if (!rows.length) {
      target.innerHTML = empty("没有可展示的行业日表", "API 没有返回有效日期行；缺失值不会插值补齐。");
      return;
    }
    target.innerHTML = tableRegion("行业逐日真实数据，可横向滚动", scrollTable([
      ["日期", "code"], ["开", "numeric"], ["高", "numeric"], ["低", "numeric"], ["收", "numeric"], ["MA20", "numeric"], ["MA60", "numeric"],
      ["沪深300同基期", "numeric"], ["成交额", "numeric"], ["成交额MA5", "numeric"], ["成交额MA20", "numeric"], ["活跃度", "numeric"],
      ["MA20上宽度", "numeric"], ["Top-N候选", "numeric"], ["实际目标", "numeric"], ["趋势分", "numeric"], ["市场共振状态"], ["策略确认"], ["选择原因"],
    ], rows.map((row) => [
      h(row.date), h(decimals(row.open, 3)), h(decimals(row.high, 3)), h(decimals(row.low, 3)), h(decimals(row.close, 3)),
      h(decimals(row.ma20, 3)), h(decimals(row.ma60, 3)), h(decimals(row.benchmark_normalized, 3)), h(count(row.sector_amount)),
      h(count(row.amount_ma5)), h(count(row.amount_ma20)), h(decimals(row.activity_ratio, 3)), h(pct(row.breadth_above_ma20)),
      h(count(row.purchase_candidate_count)), h(count(row.target_count)), h(decimals(row.market_trend_score, 0)),
      h(resonanceText(row.resonance_state)), h(confirmationText(row.strategy_confirmation, row.purchase_candidate_count)),
      h(selectionReasons[row.selection_reason] || row.selection_reason || "—"),
    ])));
  }

  function csvCell(value) {
    if (value === null || value === undefined) return "";
    const text = typeof value === "object" ? JSON.stringify(value) : String(value);
    return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  function exportChartCsv() {
    if (!local.chartRows.length || !local.selected) return;
    const columns = ["date", "open", "high", "low", "close", "ma20", "ma60", "benchmark_normalized", "sector_amount", "amount_ma5", "amount_ma20",
      "activity_ratio", "breadth_above_ma20", "purchase_candidate_count", "target_count", "market_trend_score", "resonance_state", "strategy_confirmation", "selection_reason"];
    const lines = [columns.join(","), ...local.chartRows.map((row) => columns.map((key) => csvCell(row[key])).join(","))];
    const blob = new Blob(["﻿", lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    const identity = `${local.run}-${local.selected}`.replace(/[^a-z0-9_.-]+/gi, "_").slice(0, 150);
    anchor.href = url;
    anchor.download = `${identity || "sector-chart"}-chart.csv`;
    anchor.hidden = true;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  function renderChartDownloads(payload) {
    const target = detailNode.querySelector("[data-sector-chart-downloads]");
    if (!target) return;
    const links = downloadLinks(payload.downloads, ["chart_csv", "selection_candidates", "report"]);
    const fallback = payload.status === "available" && !safeDownload(payload.downloads?.chart_csv) && local.chartRows.length
      ? '<button type="button" class="table-button sector-download" data-sector-export>导出当前图表 CSV<small>客户端</small></button>' : "";
    target.innerHTML = links + fallback;
  }

  function legendHtml(items) {
    return `<div class="sector-chart-legend" aria-label="图例">${items.map(([label, tone, mark = "line"]) => `<span><i class="${mark} ${tone}" aria-hidden="true"></i>${h(label)}</span>`).join("")}</div>`;
  }

  function lineData(rows, key, valid = finite) {
    return rows.map((row) => valid(row[key]) ? { time: row.date, value: num(row[key]) } : { time: row.date });
  }

  function constantData(rows, value) { return rows.map((row) => ({ time: row.date, value })); }

  function candleData(rows) {
    return rows.map((row) => {
      const values = ["open", "high", "low", "close"].map((key) => num(row[key]));
      const valid = values.every((value) => Number.isFinite(value) && value > 0) &&
        values[2] <= Math.min(values[0], values[3]) && values[1] >= Math.max(values[0], values[3]) && values[2] <= values[1];
      return valid ? { time: row.date, open: values[0], high: values[1], low: values[2], close: values[3] } : { time: row.date };
    });
  }

  function primaryValue(kind, row) {
    if (!row) return NaN;
    if (kind === "price") return candleData([row])[0].close;
    if (kind === "amount") return nonNegative(row.sector_amount) ? num(row.sector_amount) : NaN;
    if (kind === "activity") return nonNegative(row.activity_ratio) ? num(row.activity_ratio) : NaN;
    if (kind === "breadth") return nonNegative(row.breadth_above_ma20) ? num(row.breadth_above_ma20) : NaN;
    if (kind === "candidates") {
      if (!nonNegative(row.purchase_candidate_count) && !nonNegative(row.target_count)) return NaN;
      return Math.max(nonNegative(row.purchase_candidate_count) ? num(row.purchase_candidate_count) : 0, nonNegative(row.target_count) ? num(row.target_count) : 0);
    }
    if (kind === "trend") return finite(row.market_trend_score) ? num(row.market_trend_score) : NaN;
    return NaN;
  }

  function readoutFields(kind) {
    const amount = (value) => count(value);
    return ({
      price: [
        ["开", (row) => decimals(row.open, 3)], ["高", (row) => decimals(row.high, 3)], ["低", (row) => decimals(row.low, 3)], ["收", (row) => decimals(row.close, 3)],
        ["MA20", (row) => decimals(row.ma20, 3)], ["MA60", (row) => decimals(row.ma60, 3)], ["沪深300同基期", (row) => decimals(row.benchmark_normalized, 3)],
      ],
      amount: [["板块成交额", (row) => amount(row.sector_amount)], ["MA5", (row) => amount(row.amount_ma5)], ["MA20", (row) => amount(row.amount_ma20)]],
      activity: [["活跃度 MA5／MA20", (row) => decimals(row.activity_ratio, 3)], ["参考线", () => "1.000"]],
      breadth: [["MA20上成分比例", (row) => pct(row.breadth_above_ma20)], ["+1 阈值", () => "60.00%"], ["-1 阈值", () => "40.00%"]],
      candidates: [["Top-N购买候选", (row) => count(row.purchase_candidate_count)], ["实际目标", (row) => count(row.target_count)],
        ["选择原因", (row) => selectionReasons[row.selection_reason] || row.selection_reason || "—"],
        ["策略确认", (row) => confirmationText(row.strategy_confirmation, row.purchase_candidate_count)]],
      trend: [["市场趋势分", (row) => finite(row.market_trend_score) ? `${num(row.market_trend_score) > 0 ? "+" : ""}${decimals(row.market_trend_score, 0)}` : "—"],
        ["市场状态", (row) => resonanceText(row.resonance_state)], ["策略确认", (row) => confirmationText(row.strategy_confirmation, row.purchase_candidate_count)]],
    })[kind] || [];
  }

  function fillReadout(record, day) {
    const row = record.rows.get(day);
    record.readout.replaceChildren();
    const date = document.createElement("strong");
    date.textContent = day || "无日期";
    record.readout.append(date);
    if (!row) {
      const missing = document.createElement("span");
      missing.textContent = "该日无记录";
      record.readout.append(missing);
      return;
    }
    for (const [label, value] of readoutFields(record.kind)) {
      const item = document.createElement("span");
      item.className = "sector-readout-item";
      const name = document.createElement("small");
      name.textContent = label;
      const number = document.createElement("b");
      number.textContent = value(row);
      item.append(name, number);
      record.readout.append(item);
    }
  }

  function updateReadouts(day, synchronize = true, source = null) {
    if (!isoDay.test(day)) return;
    local.readoutDay = day;
    for (const record of local.charts) fillReadout(record, day);
    if (!synchronize || local.syncing) return;
    local.syncing = true;
    try {
      for (const record of local.charts) {
        if (record === source || !record.primary) continue;
        const row = record.rows.get(day);
        const value = primaryValue(record.kind, row);
        if (record.rows.has(day) && finite(value)) record.chart.setCrosshairPosition(value, day, record.primary);
      }
    } catch { /* A missing custom point must not break the shared readout. */ }
    finally { local.syncing = false; }
  }

  function fillTooltip(record, day) {
    const row = record.rows.get(day);
    record.tooltip.replaceChildren();
    const heading = document.createElement("strong");
    heading.textContent = day;
    record.tooltip.append(heading);
    if (!row) return;
    for (const [label, value] of readoutFields(record.kind)) {
      const line = document.createElement("div");
      const name = document.createElement("span");
      name.textContent = label;
      const number = document.createElement("b");
      number.textContent = value(row);
      line.append(name, number);
      record.tooltip.append(line);
    }
  }

  function showTooltip(record, day, event = null) {
    if (!day || !record.rows.has(day)) { record.tooltip.hidden = true; return; }
    fillTooltip(record, day);
    record.tooltip.hidden = false;
    const rect = record.frame.getBoundingClientRect();
    const hostRect = record.host.getBoundingClientRect();
    const x = event ? event.clientX - rect.left : hostRect.left - rect.left + hostRect.width / 2;
    const y = event ? event.clientY - rect.top : 18;
    const left = Math.max(8, Math.min(x + 14, rect.width - record.tooltip.offsetWidth - 8));
    const top = Math.max(8, Math.min(y + 14, rect.height - record.tooltip.offsetHeight - 8));
    record.tooltip.style.left = `${left}px`;
    record.tooltip.style.top = `${top}px`;
  }

  function wireChart(record) {
    record.chart.subscribeCrosshairMove((param) => {
      if (local.syncing) return;
      const day = chartDate(param.time);
      if (day) updateReadouts(day, true, record);
    });
    record.chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (!range || local.syncingRange) return;
      local.syncingRange = true;
      try {
        for (const other of local.charts) if (other !== record) other.chart.timeScale().setVisibleLogicalRange(range);
      } catch { /* Different missing-data patterns can limit synchronization. */ }
      finally { local.syncingRange = false; }
    });
    record.host.addEventListener("pointermove", (event) => {
      const rect = record.host.getBoundingClientRect();
      const day = chartDate(record.chart.timeScale().coordinateToTime(event.clientX - rect.left));
      if (!day) { record.tooltip.hidden = true; return; }
      updateReadouts(day, true, record);
      showTooltip(record, day, event);
    });
    record.host.addEventListener("pointerleave", () => { record.tooltip.hidden = true; });
    record.host.addEventListener("focus", () => {
      const day = local.readoutDay || local.chartRows.at(-1)?.date;
      if (day) { updateReadouts(day, true, record); showTooltip(record, day); }
    });
    record.host.addEventListener("blur", () => { record.tooltip.hidden = true; });
    record.host.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const days = local.chartRows.map((row) => row.date);
      const currentDay = local.readoutDay || days.at(-1);
      const index = Math.max(0, days.indexOf(currentDay));
      const next = event.key === "Home" ? 0 : event.key === "End" ? days.length - 1 : Math.max(0, Math.min(days.length - 1, index + (event.key === "ArrowRight" ? 1 : -1)));
      if (days[next]) { updateReadouts(days[next], true, record); showTooltip(record, days[next]); }
    });
  }

  function lineSeries(chart, data, color, options = {}) {
    const series = chart.addLineSeries({
      color, lineWidth: options.lineWidth || 2, lineStyle: options.lineStyle ?? window.LightweightCharts.LineStyle.Solid,
      lineType: options.lineType ?? window.LightweightCharts.LineType.Simple,
      priceLineVisible: false, lastValueVisible: options.lastValueVisible !== false, crosshairMarkerRadius: 4,
      ...(options.priceFormat ? { priceFormat: options.priceFormat } : {}),
      ...(options.autoscaleInfoProvider ? { autoscaleInfoProvider: options.autoscaleInfoProvider } : {}),
    });
    series.setData(data);
    return series;
  }

  function addResonanceMarkers(series, rows) {
    const markers = [];
    let previous = "";
    const validDates = new Set(candleData(rows).filter((point) => positive(point.close)).map((point) => point.time));
    for (const row of rows) {
      const kind = resonanceKind(row.resonance_state);
      if (kind && kind !== previous && validDates.has(row.date)) {
        markers.push({ time: row.date, position: kind === "up" ? "belowBar" : "aboveBar", color: kind === "up" ? colors.danger : colors.teal,
          shape: kind === "up" ? "arrowUp" : "arrowDown", text: kind === "up" ? "强上升" : "强下降", size: 1 });
      }
      previous = kind;
    }
    if (markers.length && typeof series.setMarkers === "function") series.setMarkers(markers);
  }

  function groupedBarsView() {
    let bars = [];
    let barSpacing = 8;
    return {
      defaultOptions() {
        return { ...window.LightweightCharts.customSeriesDefaultOptions, priceFormat: { type: "price", precision: 0, minMove: 1 }, lastValueVisible: false, priceLineVisible: false };
      },
      priceValueBuilder(data) {
        const candidate = nonNegative(data?.purchase) ? num(data.purchase) : 0;
        const target = nonNegative(data?.target) ? num(data.target) : 0;
        return [0, candidate, target];
      },
      isWhitespace(data) { return !nonNegative(data?.purchase) && !nonNegative(data?.target); },
      renderer() {
        return {
          draw(target, priceToCoordinate) {
            target.useBitmapCoordinateSpace(({ context, horizontalPixelRatio, verticalPixelRatio }) => {
              const zero = priceToCoordinate(0);
              if (zero === null || !Number.isFinite(zero)) return;
              const pairWidth = Math.max(2, Math.floor(barSpacing * horizontalPixelRatio * 0.78));
              const gap = Math.max(2, Math.round(2 * horizontalPixelRatio));
              const width = Math.max(1, Math.floor((pairWidth - gap) / 2));
              for (const bar of bars) {
                const candidate = nonNegative(bar.originalData?.purchase) ? num(bar.originalData.purchase) : null;
                const targetValue = nonNegative(bar.originalData?.target) ? num(bar.originalData.target) : null;
                const center = Math.round(bar.x * horizontalPixelRatio);
                const baseline = Math.round(zero * verticalPixelRatio);
                const entries = [[candidate, colors.teal, center - gap / 2 - width], [targetValue, colors.amber, center + gap / 2]];
                for (const [value, color, left] of entries) {
                  if (value === null || value <= 0) continue;
                  const coordinate = priceToCoordinate(value);
                  if (coordinate === null || !Number.isFinite(coordinate)) continue;
                  const top = Math.round(coordinate * verticalPixelRatio);
                  context.fillStyle = color;
                  context.fillRect(Math.round(left), Math.min(top, baseline), width, Math.max(1, Math.abs(baseline - top)));
                }
              }
            });
          },
        };
      },
      update(data) { bars = data.bars; barSpacing = data.barSpacing; },
    };
  }

  function buildPrice(record) {
    const chart = record.chart;
    const candle = chart.addCandlestickSeries({ upColor: colors.danger, downColor: colors.teal, borderVisible: false,
      wickUpColor: colors.danger, wickDownColor: colors.teal, priceLineVisible: false, lastValueVisible: true });
    candle.setData(candleData(local.chartRows));
    addResonanceMarkers(candle, local.chartRows);
    lineSeries(chart, lineData(local.chartRows, "ma20", positive), colors.amber, { lastValueVisible: false });
    lineSeries(chart, lineData(local.chartRows, "ma60", positive), colors.slate, { lineStyle: window.LightweightCharts.LineStyle.Dashed, lastValueVisible: false });
    lineSeries(chart, lineData(local.chartRows, "benchmark_normalized", positive), colors.muted, { lineStyle: window.LightweightCharts.LineStyle.Dotted, lastValueVisible: false });
    record.primary = candle;
  }

  function buildAmount(record) {
    const histogram = record.chart.addHistogramSeries({ color: colors.teal, priceLineVisible: false, lastValueVisible: true, priceFormat: { type: "volume" } });
    histogram.setData(lineData(local.chartRows, "sector_amount", nonNegative));
    lineSeries(record.chart, lineData(local.chartRows, "amount_ma5", nonNegative), colors.amber, { lastValueVisible: false, priceFormat: { type: "volume" } });
    lineSeries(record.chart, lineData(local.chartRows, "amount_ma20", nonNegative), colors.slate, { lineStyle: window.LightweightCharts.LineStyle.Dashed, lastValueVisible: false, priceFormat: { type: "volume" } });
    record.primary = histogram;
  }

  function buildActivity(record) {
    record.primary = lineSeries(record.chart, lineData(local.chartRows, "activity_ratio", nonNegative), colors.teal);
    lineSeries(record.chart, constantData(local.chartRows, 1), colors.amber, { lineStyle: window.LightweightCharts.LineStyle.Dashed, lineWidth: 1, lastValueVisible: false });
  }

  function buildBreadth(record) {
    const priceFormat = { type: "custom", minMove: 0.01, formatter: (value) => `${(value * 100).toFixed(0)}%` };
    record.primary = lineSeries(record.chart, lineData(local.chartRows, "breadth_above_ma20", nonNegative), colors.teal, { priceFormat });
    if (!nativeThs()) {
      lineSeries(record.chart, constantData(local.chartRows, 0.6), colors.amber, { lineStyle: window.LightweightCharts.LineStyle.Dashed, lineWidth: 1, lastValueVisible: false, priceFormat });
      lineSeries(record.chart, constantData(local.chartRows, 0.4), colors.danger, { lineStyle: window.LightweightCharts.LineStyle.Dashed, lineWidth: 1, lastValueVisible: false, priceFormat });
    }
    record.chart.priceScale().applyOptions({ scaleMargins: { top: 0.12, bottom: 0.12 } });
  }

  function buildCandidates(record) {
    const data = local.chartRows.map((row) => {
      const purchase = nonNegative(row.purchase_candidate_count) ? num(row.purchase_candidate_count) : null;
      const target = nonNegative(row.target_count) ? num(row.target_count) : null;
      if (purchase === null && target === null) return { time: row.date };
      return { time: row.date, value: Math.max(purchase || 0, target || 0), purchase, target };
    });
    const series = record.chart.addCustomSeries(groupedBarsView(), { lastValueVisible: false, priceLineVisible: false });
    series.setData(data);
    record.primary = series;
  }

  function buildTrend(record) {
    record.primary = lineSeries(record.chart, lineData(local.chartRows, "market_trend_score"), colors.teal,
      { lineType: window.LightweightCharts.LineType.WithSteps, priceFormat: { type: "price", precision: 0, minMove: 1 },
        autoscaleInfoProvider: () => ({ priceRange: { minValue: -3.5, maxValue: 3.5 } }) });
    lineSeries(record.chart, constantData(local.chartRows, 0), colors.muted, { lineStyle: window.LightweightCharts.LineStyle.Dashed, lineWidth: 1, lastValueVisible: false,
      priceFormat: { type: "price", precision: 0, minMove: 1 } });
  }

  function chartSpecs() {
    const native = nativeThs();
    return [
      { kind: "price", title: native ? "1 · 同花顺原生行业指数 K 线与沪深300（共同基期）" : "1 · 等权后复权合成 K 线与沪深300（共同基期）",
        note: native ? "真实行业指数 OHLC；MA20／MA60 与归一化沪深300共用同一价格轴。缺少历史成分宽度时不产生强共振箭头。" : "合成指数不可交易；MA20／MA60 与归一化沪深300共用同一价格轴。箭头明确标出强共振状态起点，逐日状态见下表。",
        legend: [[native ? "同花顺行业K线 · 上涨／下跌" : "合成K线 · 上涨／下跌", "teal", "candle"], ["MA20", "amber"], ["MA60", "slate", "dash"], ["沪深300同基期", "muted", "dot"]], build: buildPrice },
      { kind: "amount", title: "2 · 板块成交额与均线", note: "成交额放大只表示交易活跃，不代表真实资金净流入。柱、MA5 和 MA20 均为同一成交额单位。",
        legend: [["板块成交额", "teal", "bar"], ["成交额 MA5", "amber"], ["成交额 MA20", "slate", "dash"]], build: buildAmount },
      { kind: "activity", title: "3 · 交易活跃度", note: native ? "activity_ratio = 成交额 MA5／MA20；仅描述活跃程度，不单独构成强共振。" : "activity_ratio = 成交额 MA5／MA20；大于等于 1 才满足强共振的活跃度条件。",
        legend: [["活跃度", "teal"], ["1.0 参考线", "amber", "dash"]], build: buildActivity },
      { kind: "breadth", title: "4 · 板块宽度", note: native ? "同花顺接口未提供历史时点成分，历史 MA20 宽度保持缺失；不会用当前成分回填。" : "成分为个股位于各自 MA20 之上的比例；60% 为 +1 阈值，40% 为 -1 阈值。",
        legend: native ? [["历史成分宽度不可用", "muted"]] : [["MA20上成分比例", "teal"], ["+1 阈值 60%", "amber", "dash"], ["-1 阈值 40%", "danger", "dash"]], build: buildBreadth },
      { kind: "candidates", title: "5 · Top-N 购买候选与实际目标（分组柱）", note: "候选在集中度约束前，目标在约束后。零值可能来自非调仓日或其它约束，只有候选账本返回原因时才明确标注。",
        legend: [["Top-N购买候选", "teal", "bar"], ["实际目标", "amber", "bar"]], build: buildCandidates },
      { kind: "trend", title: "6 · 市场趋势分与策略确认", note: native ? "原生模式趋势分由价格与20日相对强度构成，范围为 -2 至 +2；宽度不记分，强共振停用。" : "趋势分固定为 -3 至 +3 的阶梯线；策略确认单独读取，不改变市场趋势分。",
        legend: [["市场趋势分", "teal"], ["0 参考线", "muted", "dash"]], build: buildTrend },
    ];
  }

  function createChartRecord(host, spec) {
    const frame = host.closest(".sector-chart-frame");
    const readout = frame.parentElement.querySelector(".sector-chart-readout");
    const tooltip = frame.querySelector(".sector-tooltip");
    const chart = window.LightweightCharts.createChart(host, {
      autoSize: true,
      ...chartThemeOptions(),
      rightPriceScale: { ...chartThemeOptions().rightPriceScale, scaleMargins: { top: 0.12, bottom: 0.14 } },
      timeScale: { ...chartThemeOptions().timeScale, rightOffset: 2, barSpacing: 8 },
      localization: { locale: "zh-CN" },
    });
    const record = { kind: spec.kind, chart, host, frame, readout, tooltip, primary: null, rows: new Map(local.chartRows.map((row) => [row.date, row])) };
    local.charts.push(record);
    spec.build(record);
    chart.timeScale().fitContent();
    wireChart(record);
    return record;
  }

  function renderCharts() {
    destroyCharts();
    const host = detailNode.querySelector("[data-sector-charts]");
    if (!host) return;
    host.removeAttribute("aria-busy");
    if (!local.chartRows.length) {
      host.innerHTML = empty("没有可绘制的行业序列", "null、NaN 和无效价格关系不会转换为零；请核对下方 API 原始行。");
      return;
    }
    if (!window.LightweightCharts) {
      host.innerHTML = empty("图表库加载失败", "六个面板无法绘制，但下方行业日表仍提供全部真实数值。");
      return;
    }
    const specs = chartSpecs();
    host.innerHTML = specs.map((spec) => `<section class="sector-chart-card" data-chart-card="${spec.kind}"><div class="sector-chart-card-head"><h5>${h(spec.title)}</h5><p>${h(spec.note)}</p></div>${legendHtml(spec.legend)}<div class="sector-chart-readout" id="sector-readout-${spec.kind}" aria-label="${h(spec.title)}当前日期读数"></div><div class="sector-chart-frame"><div class="sector-chart-host" data-chart-host="${spec.kind}" tabindex="0" role="group" aria-label="${h(spec.title)}。左右方向键逐日读取，Home 和 End 跳转首尾。" aria-describedby="sector-readout-${spec.kind}"></div><div class="sector-tooltip" role="tooltip" hidden></div></div></section>`).join("");
    for (const spec of specs) {
      const chartHost = host.querySelector(`[data-chart-host="${spec.kind}"]`);
      try { createChartRecord(chartHost, spec); }
      catch (error) {
        const record = local.charts.find((item) => item.host === chartHost);
        if (record) {
          try { record.chart.remove(); } catch { /* Ignore partial chart cleanup. */ }
          local.charts = local.charts.filter((item) => item !== record);
        }
        const card = chartHost.closest(".sector-chart-card");
        card.innerHTML = empty(`${spec.title}暂不可绘制`, `${error.message}；请使用下方行业日表核对数值。`);
      }
    }
    updateReadouts(local.chartRows.at(-1).date, false);
  }

  async function loadChart(generation, detailRequest, industry) {
    const run = local.run;
    const params = new URLSearchParams({ account: "full_pool" });
    const status = detailNode.querySelector("[data-sector-chart-status]");
    try {
      const payload = await request(`/api/expert/replays/${encodeURIComponent(run)}/sectors/${encodeURIComponent(industry)}/chart?${params}`, generation);
      if (!current(generation, run) || detailRequest !== local.detailRequest || industry !== local.selected) return;
      destroyCharts();
      local.chartPayload = payload;
      local.chartRows = cleanChartRows(payload);
      const heading = detailNode.querySelector(".sector-detail-head h3");
      if (heading) heading.textContent = industryLabel({ industry: payload.industry, industry_name: payload.industry_name });
      renderChartDownloads(payload);
      renderDailyTable(local.chartRows);
      if (status) status.textContent = payload.status === "available"
        ? `${payload.start || local.chartRows[0]?.date || "—"} 至 ${payload.end || local.chartRows.at(-1)?.date || "—"} · ${payload.provider === "ths_fuyao" ? "同花顺原生行业指数" : "等权后复权合成"} · ${count(local.chartRows.length)} 个交易日${payload.notice ? ` · ${payload.notice}` : ""}`
        : `行业图表不可用：${payload.reason || "旧归档未封存行业图表。"}`;
      if (payload.status !== "available") {
        local.chartRows = [];
        const charts = detailNode.querySelector("[data-sector-charts]");
        if (charts) { charts.removeAttribute("aria-busy"); charts.innerHTML = empty("此行业图表不可用", `${payload.reason || "旧归档未封存行业图表。"} 不会用最新行情伪造历史序列。`); }
        return;
      }
      renderCharts();
    } catch (error) {
      if (!current(generation, run) || detailRequest !== local.detailRequest || error.name === "AbortError") return;
      destroyCharts();
      if (status) status.textContent = error.message;
      const charts = detailNode.querySelector("[data-sector-charts]");
      if (charts) { charts.removeAttribute("aria-busy"); charts.innerHTML = empty("行业图表读取失败", error.message); }
      const daily = detailNode.querySelector("[data-sector-daily-table]");
      if (daily) daily.innerHTML = empty("行业日表读取失败", error.message);
    }
  }

  function loadDetail() {
    if (!local.selected || !local.run) return Promise.resolve();
    const industry = local.selected;
    const generation = local.generation;
    const detailRequest = ++local.detailRequest;
    destroyCharts();
    local.chartPayload = null;
    local.chartRows = [];
    detailSkeleton(industry);
    renderOverview();
    return Promise.all([loadChart(generation, detailRequest, industry), loadCandidates(generation, detailRequest, industry)]);
  }

  function clear() {
    invalidate();
    destroyCharts();
    local.run = "";
    local.overview = null;
    local.dates = [];
    local.date = "";
    local.selected = "";
    local.chartPayload = null;
    local.chartRows = [];
    local.readoutDay = "";
    root.hidden = true;
    root.removeAttribute("aria-busy");
    overviewNode.removeAttribute("aria-busy");
    overviewNode.classList.remove("sector-loading");
    statusNode.textContent = "";
    dateInput.value = "";
    dateInput.removeAttribute("min");
    dateInput.removeAttribute("max");
    dateInput.removeAttribute("list");
    $("#sector-date-options")?.remove();
    summaryNode.replaceChildren();
    overviewNode.replaceChildren();
    detailNode.replaceChildren();
    downloadsNode.replaceChildren();
  }

  function show(runId, accounts) {
    if (!runId || !accounts?.full_pool) { clear(); return Promise.resolve(); }
    clear();
    local.run = String(runId);
    local.sort = sortInput.value === "candidates" ? "candidates" : "trend";
    root.hidden = false;
    summaryNode.replaceChildren();
    detailNode.replaceChildren();
    downloadsNode.replaceChildren();
    return loadOverview();
  }

  function refresh() {
    if (!local.run || root.hidden) return Promise.resolve();
    const date = local.date;
    invalidate();
    destroyCharts();
    local.date = date;
    return loadOverview({ preserveSelection: true });
  }

  dateInput.addEventListener("change", () => {
    const value = dateInput.value;
    if (value && (!isoDay.test(value) || (local.dates.length && !local.dates.includes(value)))) {
      dateInput.value = local.date;
      statusNode.textContent = "请选择此归档提供的交易日；日期输入框已列出可用日期。";
      return;
    }
    invalidate();
    destroyCharts();
    local.date = value;
    loadOverview({ preserveSelection: true });
  });

  sortInput.addEventListener("change", () => {
    local.sort = sortInput.value === "candidates" ? "candidates" : "trend";
    renderOverview();
  });

  themeButton.addEventListener("click", () => {
    local.theme = local.theme === "dark" ? "light" : "dark";
    try { window.localStorage.setItem(themeKey, JSON.stringify(local.theme)); } catch { /* Theme persistence is optional. */ }
    applyTheme();
  });

  overviewNode.addEventListener("click", (event) => {
    const button = event.target.closest("[data-sector-industry]");
    if (!button || !local.overview || local.overview.status !== "available") return;
    invalidate();
    local.selected = button.dataset.sectorIndustry;
    loadDetail();
    detailNode.scrollIntoView({ block: "start" });
  });

  detailNode.addEventListener("click", (event) => {
    if (event.target.closest("[data-sector-export]")) { exportChartCsv(); return; }
    if (event.target.closest("[data-sector-close]")) {
      invalidate();
      destroyCharts();
      local.selected = "";
      local.chartPayload = null;
      local.chartRows = [];
      detailNode.replaceChildren();
      renderOverview();
    }
  });

  statusNode.setAttribute("role", "status");
  statusNode.setAttribute("aria-live", "polite");
  statusNode.setAttribute("aria-atomic", "true");
  dateInput.setAttribute("aria-label", "选择板块总览交易日");
  sortInput.setAttribute("aria-label", "行业总览排序方式");
  themeButton.type = "button";
  if (!sortInput.options.length) sortInput.replaceChildren(option("trend", "按市场趋势分"), option("candidates", "按 Top-N 购买候选"));
  applyTheme();

  window.SectorResonance = { show, clear, refresh };
})();
