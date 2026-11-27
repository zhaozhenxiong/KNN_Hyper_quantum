"use strict";

(() => {
  const page = $("#page-backtest");
  const form = $("#strategy-form");
  const submit = $("#strategy-run");
  const feedback = $("#strategy-status");
  const search = $("#strategy-symbol");
  const suggestions = $("#strategy-suggestions");
  const watchStatus = $("#strategy-watch-status");
  const chips = $("#strategy-watchlist");
  const clear = $("#strategy-clear");
  const bulk = $("#strategy-bulk");
  const bulkAdd = $("#strategy-bulk-add");
  const stage = $("#strategy-chart-stage");
  const placeholder = $("#strategy-chart-empty");
  const chartNode = $("#strategy-equity");
  const scrub = $("#strategy-scrub");
  const result = $("#strategy-result");
  const activity = $("#strategy-activity");
  const activityForm = $("#strategy-activity-form");
  const activityStatus = $("#strategy-activity-status");
  const activityRows = $("#strategy-activity-rows");
  const activityPager = $("#strategy-activity-pager");
  const chartMonth = $("#strategy-show-month");
  const holdingsPanel = $("#strategy-holdings");
  const holdingsAccount = $("#strategy-holdings-account");
  const holdingsStatus = $("#strategy-holdings-status");
  const holdingsRows = $("#strategy-holdings-rows");
  const colors = { full_pool: "#0d6251", watchlist: "#a15e21", benchmark: "#536b82" };
  const names = { full_pool: "全池", watchlist: "关注股", benchmark: "沪深 300" };
  const labels = { daily: "日账本", orders: "订单", trades: "成交", positions: "持仓" };
  const orderLabels = { filled: "成交", partial: "部分成交", rejected: "拒单", pretrade_rejected: "下单前拦截" };
  const actionLabels = { BUY: "买入", ADD: "加仓", REDUCE: "减仓", SELL: "卖出" };
  const reasons = { no_previous_session_liquidity: "缺少前一交易日流动性", liquidity_below_lot: "流动性不足一手",
    position_limit: "已达持仓数量上限", concentration: "已达集中度上限", cash_buffer: "可用现金不足",
    delisted: "已退市", suspended: "停牌", no_open: "缺少开盘价", limit_up: "涨停无法买入", limit_down: "跌停无法卖出" };
  const watchKey = "khquant.strategy.watchlist.v1";
  const jobKey = "khquant.strategy.job.v1";
  const code = /^\d{6}\.(?:SH|SZ|BJ)$/;
  const shortCode = /^\d{6}$/;
  const runKey = /^[a-z0-9][a-z0-9_.-]{0,127}$/i;
  const isoDay = /^\d{4}-\d{2}-\d{2}$/;
  const active = (job) => ["queued", "running"].includes(job?.status);
  const input = (name) => form.elements.namedItem(name);
  const requestOptions = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const local = { model: null, job: null, rendered: "", pending: false, importing: false, timer: null, request: 0, search: 0, searchTimer: null,
    matches: [], highlighted: -1, chart: null, series: [], dates: [], readoutDay: "", initialized: false,
    candidatePending: false, candidateRequest: 0, pinned: false };
  const activityView = { run: "", refresh: 0, page: 1, pages: 1, afterLast: false };
  const holdingsView = { run: "", refresh: 0 };
  const activityField = (name) => activityForm.elements.namedItem(name);

  function readStorage(storage, key, fallback) {
    try { return JSON.parse(window[storage].getItem(key)) ?? fallback; } catch { return fallback; }
  }

  function writeStorage(storage, key, value) {
    try { window[storage].setItem(key, JSON.stringify(value)); } catch { /* Storage is optional. */ }
  }

  const saved = arr(readStorage("localStorage", watchKey, []));
  const watchlist = new Map(saved.filter((item) => code.test(item?.symbol)).map((item) => [item.symbol, item]));

  function updateWatchlist(message) {
    writeStorage("localStorage", watchKey, [...watchlist.values()]);
    clear.disabled = !watchlist.size || local.importing;
    chips.hidden = !watchlist.size;
    chips.innerHTML = [...watchlist.values()].map(({ symbol, name }) => `<li><button type="button" class="strategy-chip" data-remove-symbol="${h(symbol)}" aria-label="移除 ${h(symbol)}"><strong>${h(symbol)}</strong>${name ? `<span>${h(name)}</span>` : ""}<span aria-hidden="true">×</span></button></li>`).join("");
    watchStatus.textContent = message || (watchlist.size ? `已选 ${count(watchlist.size)} 只。回测时同时生成关注股账户。` : "空名单时运行全池。");
  }

  function closeSuggestions() {
    suggestions.hidden = true;
    search.setAttribute("aria-expanded", "false");
    search.removeAttribute("aria-activedescendant");
    local.highlighted = -1;
  }

  function showMatches(items) {
    local.matches = items.filter((item) => code.test(item.symbol));
    suggestions.innerHTML = local.matches.map(({ symbol, name }, index) => `<li id="strategy-match-${index}" role="option" aria-selected="false" data-match="${index}"><strong>${h(symbol)}</strong><span>${h(name)}</span></li>`).join("");
    local.highlighted = -1;
    search.removeAttribute("aria-activedescendant");
    suggestions.hidden = !local.matches.length;
    search.setAttribute("aria-expanded", String(!suggestions.hidden));
  }

  async function find(query) {
    const selected = await api(`/api/stock-chart/search?${new URLSearchParams({ q: query })}`);
    return arr(selected.symbols);
  }

  function addMatch(match) {
    if (!match) return;
    watchlist.set(match.symbol, { symbol: match.symbol, name: match.name || "" });
    search.value = "";
    local.search += 1;
    closeSuggestions();
    updateWatchlist();
    search.focus();
  }

  search.addEventListener("input", () => {
    const q = search.value.trim();
    const request = ++local.search;
    clearTimeout(local.searchTimer);
    closeSuggestions();
    if (!q) { updateWatchlist(); return; }
    local.searchTimer = setTimeout(async () => {
      try {
        const matches = await find(q);
        if (request !== local.search || document.activeElement !== search) return;
        showMatches(matches);
        if (!local.matches.length) watchStatus.textContent = "未找到匹配股票。";
      } catch (error) {
        if (request === local.search) watchStatus.textContent = error.message;
      }
    }, 180);
  });

  search.addEventListener("keydown", (event) => {
    if (event.key === "Escape") { closeSuggestions(); return; }
    if (suggestions.hidden) { if (event.key === "Enter") event.preventDefault(); return; }
    if (event.key === "Enter") { event.preventDefault(); addMatch(local.matches[Math.max(0, local.highlighted)]); return; }
    if (!["ArrowDown", "ArrowUp"].includes(event.key)) return;
    event.preventDefault();
    const last = local.matches.length - 1;
    local.highlighted = event.key === "ArrowDown" ? Math.min(last, local.highlighted + 1) : Math.max(0, local.highlighted - 1);
    const selected = suggestions.children[local.highlighted];
    for (const item of suggestions.children) item.setAttribute("aria-selected", String(item === selected));
    search.setAttribute("aria-activedescendant", selected.id);
    selected.scrollIntoView({ block: "nearest" });
  });

  suggestions.addEventListener("pointerdown", (event) => event.preventDefault());
  suggestions.addEventListener("click", (event) => addMatch(local.matches[Number(event.target.closest("[data-match]")?.dataset.match)]));
  search.addEventListener("blur", closeSuggestions);
  chips.addEventListener("click", (event) => {
    const button = event.target.closest("[data-remove-symbol]");
    if (button && watchlist.delete(button.dataset.removeSymbol)) { updateWatchlist(); search.focus(); }
  });
  clear.addEventListener("click", () => { watchlist.clear(); updateWatchlist(); search.focus(); });

  bulkAdd.addEventListener("click", async () => {
    const tokens = [...new Set(bulk.value.toUpperCase().split(/[\s,，、;；]+/).filter(Boolean))];
    if (!tokens.length) { watchStatus.textContent = "请粘贴股票代码。"; return; }
    local.importing = true;
    bulkAdd.disabled = true;
    clear.disabled = true;
    controls();
    watchStatus.textContent = "正在核对股票代码…";
    const unresolved = [];
    let added = 0;
    try {
      for (let offset = 0; offset < tokens.length; offset += 8) {
        const batch = await Promise.all(tokens.slice(offset, offset + 8).map(async (token) => {
          if (!code.test(token) && !shortCode.test(token)) return { token };
          try {
            const matches = (await find(token)).filter((item) => code.test(item.symbol) &&
              (code.test(token) ? item.symbol === token : item.symbol.startsWith(`${token}.`)));
            return { token, found: matches.length === 1 ? matches[0] : null };
          } catch { return { token }; }
        }));
        for (const { token, found } of batch) {
          if (!found) { unresolved.push(token); continue; }
          if (!watchlist.has(found.symbol)) added += 1;
          watchlist.set(found.symbol, { symbol: found.symbol, name: found.name || "" });
        }
      }
      bulk.value = unresolved.join(" ");
      if (!unresolved.length) $("#strategy-paste").open = false;
      const note = unresolved.length ? `。未识别 ${unresolved.slice(0, 8).join("、")}${unresolved.length > 8 ? `等 ${unresolved.length} 项` : ""}，请核对交易所后缀。` : "。";
      updateWatchlist(`新增 ${count(added)} 只，已选 ${count(watchlist.size)} 只${note}`);
    } finally {
      local.importing = false;
      bulkAdd.disabled = false;
      clear.disabled = !watchlist.size;
      controls();
    }
  });

  function controls() {
    submit.disabled = !local.model?.run_id || local.model.status === "unavailable" || local.pending || local.importing || active(local.job);
  }

  function chartReadout(date) {
    if (!date) return;
    local.readoutDay = date;
    const indices = local.dates.indexOf(date);
    if (indices >= 0) scrub.value = String(indices);
    $("#strategy-readout").innerHTML = `<strong class="strategy-readout-date">${h(date)}</strong>` + local.series.map(({ id, label, byDay }) => {
      const point = byDay.get(date);
      return `<div class="strategy-readout-line" style="--series:${colors[id]}"><span>${h(label)}</span><strong>${h(decimals(point?.nav, 4))}</strong>
        <small>累计 ${h(pct(point?.cumulative_return))} · 回撤 ${h(pct(point?.drawdown))}</small></div>`;
    }).join("");
  }

  function chartDate(time) {
    if (typeof time === "string") return time;
    if (time && typeof time === "object") {
      return `${time.year}-${String(time.month).padStart(2, "0")}-${String(time.day).padStart(2, "0")}`;
    }
    return null;
  }

  function renderChart(performance) {
    local.chart?.remove();
    local.chart = null;
    local.series = arr(performance?.series).filter(({ id }) => Object.hasOwn(colors, id)).map(({ id, label, points }) => {
      const byDay = new Map(arr(points).filter((point) => isoDay.test(point.date) && Number.isFinite(num(point.nav))).map((point) => [point.date, point]));
      return { id, label: label || names[id], byDay };
    }).filter(({ byDay }) => byDay.size);
    local.dates = [...new Set(local.series.flatMap(({ byDay }) => [...byDay.keys()]))].sort();
    placeholder.hidden = Boolean(local.dates.length);
    stage.hidden = !local.dates.length;
    if (!local.dates.length) { placeholder.textContent = "此区间没有可绘制的净值。"; return; }
    scrub.max = String(local.dates.length - 1);
    if (window.LightweightCharts) {
      const chart = window.LightweightCharts.createChart(chartNode, {
        autoSize: true, layout: { background: { color: "#fcfdfa" }, textColor: "#566760", attributionLogo: false },
        grid: { vertLines: { color: "#edf1ee" }, horzLines: { color: "#e7ede8" } },
        rightPriceScale: { borderColor: "#d5ded8" }, timeScale: { borderColor: "#d5ded8" },
      });
      for (const item of local.series) {
        const { id, byDay } = item;
        const line = chart.addLineSeries({ color: colors[id], lineWidth: id === "benchmark" ? 1 : 2,
          lineStyle: id === "benchmark" ? window.LightweightCharts.LineStyle.Dashed : window.LightweightCharts.LineStyle.Solid,
          priceLineVisible: false, priceFormat: { type: "price", precision: 4, minMove: 0.0001 } });
        line.setData([...byDay].sort(([a], [b]) => a.localeCompare(b)).map(([time, point]) => ({ time, value: num(point.nav) })));
        item.line = line;
      }
      chart.subscribeCrosshairMove(({ time }) => chartReadout(chartDate(time)));
      chart.timeScale().fitContent();
      local.chart = chart;
    } else { placeholder.hidden = false; placeholder.textContent = "图表库加载失败，可用日期控件查询净值。"; }
    chartReadout(local.dates.at(-1));
  }

  scrub.addEventListener("input", () => chartReadout(local.dates[Number(scrub.value)]));

  function selectChartDay(day) {
    if (!local.dates.includes(day)) return;
    chartReadout(day);
    if (!local.chart) return;
    const item = local.series.find(({ id, byDay }) => id !== "benchmark" && byDay.has(day));
    if (!item?.line) return;
    const scale = local.chart.timeScale();
    const range = scale.getVisibleRange();
    if (range && (day < chartDate(range.from) || day > chartDate(range.to))) {
      const index = local.dates.indexOf(day);
      scale.setVisibleRange({ from: local.dates[Math.max(0, index - 12)], to: local.dates[Math.min(local.dates.length - 1, index + 12)] });
    }
    local.chart.setCrosshairPosition(num(item.byDay.get(day).nav), day, item.line);
  }

  function holdingState(row) {
    const states = [row.stranded_delisted ? "退市滞留" : (row.market_status === "suspended" ? "停牌" : "在市")];
    if (row.mark_is_stale) states.push("陈旧估值");
    if (row.status !== "verified") states.push("收益未核验");
    return states.join("；");
  }

  function holdingLink(payload, stock) {
    const context = payload.context || {};
    const query = new URLSearchParams({ symbol: stock, as_of: payload.as_of, backtest_run: context.run_id });
    if (context.source_run_id) query.set("source_run", context.source_run_id);
    const modelVersion = context.model_version || context.model_id;
    if (modelVersion) query.set("model_version", modelVersion);
    return `#stock-chart?${query}`;
  }

  function renderHoldings(payload) {
    const { summary = {}, reconciliation = {}, context = {} } = payload;
    const items = arr(payload.holdings);
    $("#strategy-holdings-state").textContent = summary.status === "verified" ? "账本已核验" : "未完全核验";
    $("#strategy-holdings-notice").textContent = payload.research_notice || "历史模拟持仓与收益，不构成当前投资建议。";
    const totals = [
      ["期末持仓", count(summary.positions)], ["在市持仓", count(summary.listed_positions)],
      ["退市滞留持仓", count(summary.stranded_delisted_positions)], ["剩余成本（元）", decimals(summary.cost_basis, 2)],
      ["期末市值（元）", decimals(summary.market_value, 2)], ["未实现收益（元）", decimals(summary.unrealized_pnl, 2)],
      ["未实现收益率", pct(summary.unrealized_return)], ["已实现收益（元）", decimals(summary.realized_pnl, 2)],
      ["累计收益贡献（元）", decimals(summary.total_pnl, 2)],
    ];
    if (summary.stranded_delisted_positions) totals.push(["退市零回收后累计贡献（元）", decimals(summary.zero_recovery_pnl, 2)]);
    $("#strategy-holdings-summary").innerHTML = items.length ? metrics(totals) : "";
    holdingsRows.innerHTML = items.length ? scrollTable([
      ["股票", "code"], ["状态"], ["持股数", "numeric"], ["每股成本", "numeric"], ["剩余成本（元）", "numeric"],
      ["估值价", "numeric"], ["估值日", "code"], ["市值（元）", "numeric"], ["未实现收益（元）", "numeric"],
      ["未实现收益率", "numeric"], ["已实现收益（元）", "numeric"], ["累计收益贡献（元）", "numeric"],
      ["历史目标权重", "numeric"], ["最后成交", "code"],
    ], items.map((row) => [
      `<a class="table-button" href="${h(holdingLink(payload, row.stock))}" title="查看同一回测上下文中的个股证据">${h(row.stock)}</a>`,
      h(holdingState(row)), count(row.shares), decimals(row.cost_per_share, 6), decimals(row.cost_basis, 2),
      decimals(row.mark_price, 5), h(row.mark_date), decimals(row.market_value, 2), decimals(row.unrealized_pnl, 2),
      pct(row.unrealized_return), decimals(row.realized_pnl, 2), decimals(row.total_pnl, 2),
      pct(row.last_target_weight), h(row.last_trade_date),
    ])) : empty(summary.status === "unavailable" ? "持仓收益不可用" : "期末无持仓",
      reconciliation.reason || "此账户在回测截止日没有非零持仓。");
    const advice = arr(payload.advice);
    $("#strategy-holdings-advice").innerHTML = `<strong>历史研究建议（截至 ${h(payload.as_of || "未知日期")}）</strong>` +
      (advice.length ? `<ul>${advice.map((item) => `<li><strong>${item.level === "warning" ? "风险核查" : "研究提示"}</strong> ${h(item.text)}</li>`).join("")}</ul>` : "<p>没有额外研究提示。</p>");
    const download = Object.entries(payload.downloads || {}).filter(([name, path]) => ["positions", "trades"].includes(name) &&
      /^\/api\/expert\/replays\/[a-z0-9_.-]+\/downloads\/[a-z0-9_.-]+$/i.test(path)).map(([name, path]) =>
      `<a class="table-button" href="${h(path)}" download>下载${h(labels[name])}</a>`).join("");
    $("#strategy-holdings-downloads").innerHTML = download;
    holdingsStatus.textContent = summary.status === "verified" && reconciliation.passed
      ? `已与 ${context.run_id || "此回测"} 的期末日账本对账；费用与滑点不重复扣减。`
      : (reconciliation.reason || "持仓收益未完全核验，请先检查账本。");
  }

  async function loadHoldings() {
    if (!holdingsView.run) return;
    const request = ++holdingsView.refresh;
    const params = new URLSearchParams({ account: holdingsAccount.value });
    holdingsStatus.textContent = "正在读取期末持仓账本…";
    holdingsRows.setAttribute("aria-busy", "true");
    try {
      const response = await api(`${resultFor(holdingsView.run)}/holdings?${params}`);
      if (request !== holdingsView.refresh) return;
      renderHoldings(response);
    } catch (error) {
      if (request !== holdingsView.refresh) return;
      $("#strategy-holdings-state").textContent = "读取失败";
      $("#strategy-holdings-summary").replaceChildren();
      $("#strategy-holdings-advice").replaceChildren();
      $("#strategy-holdings-downloads").replaceChildren();
      holdingsRows.innerHTML = empty("持仓收益暂不可用", error.message);
      holdingsStatus.textContent = error.message;
    } finally {
      if (request === holdingsView.refresh) holdingsRows.removeAttribute("aria-busy");
    }
  }

  function showHoldings(run, accounts) {
    holdingsView.run = run;
    holdingsView.refresh += 1;
    const namesInRun = ["full_pool", "watchlist"].filter((id) => accounts[id]);
    holdingsPanel.hidden = !namesInRun.length;
    if (holdingsPanel.hidden) return;
    holdingsAccount.replaceChildren(...namesInRun.map((id) => option(id, names[id])));
    $("#strategy-holdings-state").textContent = "历史研究";
    $("#strategy-holdings-summary").replaceChildren();
    $("#strategy-holdings-advice").replaceChildren();
    $("#strategy-holdings-downloads").replaceChildren();
    holdingsRows.replaceChildren();
    loadHoldings();
  }

  holdingsAccount.addEventListener("change", loadHoldings);

  function researchStatus(response) {
    const research = response.research_status || {};
    if (research.status !== "research_only") return "";
    const gate = research.release_gate || {};
    const status = "<p>研究状态：<code>research_only</code>（仅研究，未发布）。";
    if (gate.basis !== "pinned_archive") return `${status}发布门证据不可核验：${h(gate.reason || "缺少锁定研究决策")}。</p>`;
    if (gate.passed !== false) return `${status}已读取锁定研究决策，但不能据此判定发布门通过。</p>`;
    const reasons = arr(gate.reasons).filter((item) => typeof item === "string" && item);
    return `${status}发布门未通过：历史数值门 ${gate.historical_numerical_gate === true ? "通过" : "未通过"}，点时核验 ${gate.point_in_time_verified === true ? "通过" : "未通过"}，合格前瞻结果 ${gate.qualified_forward_results === true ? "有" : "无"}。</p>${reasons.length ? `<ul>${reasons.map((item) => `<li>${h(item)}</li>`).join("")}</ul>` : ""}`;
  }

  function emptyActivityDiagnosis(response, box) {
    const { summary = {}, evidence = {}, inactivity_diagnosis: diagnosis = {} } = response;
    const full = diagnosis.full_run || {};
    const orderFacts = `<p>全运行订单 ${count(full.orders ?? summary.orders)} 条，其中买入 ${count(full.buy_orders)} 条、卖出 ${count(full.sell_orders)} 条；拒单 ${count(full.rejected_orders)} 条、下单前拦截 ${count(full.pretrade_rejected_orders)} 条。全运行汇总不受月份筛选影响。</p>`;
    let scoreEvidence;
    if (diagnosis.basis !== "pinned_archive") {
      scoreEvidence = `<p>归档分数不可核验：${h(diagnosis.reason || "缺少锁定预测证据")}。这里只陈述订单账本事实，不推断分数原因。</p>`;
    } else {
      const coverage = `<p>全运行 ${count(full.sessions)} 个共同交易日、${count(full.score_days)} 个分数覆盖日，其中 ${count(full.rebalances)} 个计划调仓信号日；调仓日有效分数 ${count(full.eligible_on_rebalance)} 股日，达标 ${count(full.qualified_on_rebalance)} 股日；非调仓日达标 ${count(full.qualified_off_rebalance)} 股日不触发交易。</p>`;
      const threshold = pct(evidence.score_threshold);
      const reasons = {
        NO_SCHEDULED_REBALANCES: "<p>可核验配置原因：全运行没有计划调仓信号日。</p>",
        NO_FINITE_SCORES: "<p>可核验分数原因：计划调仓日没有有限分数，达标候选为 0。</p>",
        NO_QUALIFIED_CANDIDATES: `<p>可核验分数原因：计划调仓日没有分数严格大于 ${h(threshold)} 的候选，订单账本为 0。</p>`,
        QUALIFIED_CANDIDATES_WITHOUT_ORDERS: `<p>计划调仓日存在 ${count(full.qualified_on_rebalance)} 个达标股日，但订单账本为 0；旧归档没有目标仓位和订单意图明细，不能进一步推断未下单原因。</p>`,
        UNFILLED_ORDERS: `<p>订单账本已有 ${count(full.orders)} 条记录，但成交账本为空；以下原因来自原订单状态，不能改写成分数原因。</p>`,
      };
      let detail = reasons[full.reason_code] || "<p>分数与订单已分别汇总；现有归档不能进一步确定执行链原因。</p>";
      if (full.reason_code === "NO_QUALIFIED_CANDIDATES" && full.all_negative_on_rebalance) {
        detail = `<p>可核验分数原因：计划调仓日 ${count(full.negative_on_rebalance)}／${count(full.eligible_on_rebalance)} 股日分数均小于 0，严格条件分数大于 ${h(threshold)} 下没有达标候选。调仓日分数范围为 ${h(decimals(full.score_min_on_rebalance, 6))} 至 ${h(decimals(full.score_max_on_rebalance, 6))}。</p>`;
      }
      scoreEvidence = coverage + detail;
    }
    const reasonRows = arr(diagnosis.zero_fill_reasons).map((row) => `<li>${h(actionLabels[row.side] || row.side)} · ${h(orderLabels[row.status] || row.status)} · ${h(orderReason(row) || "未注明原因")}：${count(row.count)} 条（${h(row.first_date)} 至 ${h(row.last_date)}）</li>`).join("");
    box.innerHTML = `<strong>为什么整个区间没有成交？</strong>${orderFacts}${scoreEvidence}${reasonRows ? `<ul>${reasonRows}</ul>` : ""}${researchStatus(response)}
      <p>零成交是本次真实回放结果，策略现金净值保持不变；不得为制造交易而降低归档门槛或添加订单。后续研究须另建对照并继续通过原发布门。</p>`;
    box.hidden = false;
  }

  function activityDiagnosis(response) {
    const box = $("#strategy-activity-diagnosis");
    const { summary = {}, inactivity_diagnosis: diagnosis = {} } = response;
    const reasonRows = arr(diagnosis.zero_fill_reasons).map((row) => `<li>${h(actionLabels[row.side] || row.side)} · ${h(orderLabels[row.status] || row.status)} · ${h(orderReason(row) || "未注明原因")}：${count(row.count)} 条（${h(row.first_date)} 至 ${h(row.last_date)}）</li>`).join("");
    if (!summary.last_trade_date) {
      emptyActivityDiagnosis(response, box);
      return;
    }
    const scoreEvidence = diagnosis.basis === "pinned_archive"
      ? `<p>最后成交后有 ${count(diagnosis.rebalances_after_last)} 个计划调仓信号日，调仓日达标 ${count(diagnosis.positive_on_rebalance_after_last)} 股日；非调仓日达标 ${count(diagnosis.positive_off_rebalance_after_last)} 股日不触发交易。</p>`
      : `<p>归档分数不可核验：${h(diagnosis.reason || "缺少锁定预测证据")}。这里只陈述订单事实，不推断分数原因。</p>`;
    const afterLast = count(diagnosis.zero_fill_after_last);
    box.innerHTML = `<strong>为什么之后没有成交？</strong>${scoreEvidence}
      <p>后续买入订单 ${count(diagnosis.buy_orders_after_last)} 条、卖出订单 ${count(diagnosis.sell_orders_after_last)} 条。全运行汇总不受月份筛选影响。</p>
      ${reasonRows ? `<ul>${reasonRows}</ul>` : ""}
      <p>最后成交当天另有 ${count(diagnosis.zero_fill_on_last_trade_date)} 条零成交订单，不计入“其后 ${afterLast} 条”。若要评估更高交易频率，应另建对照研究；本归档门槛、调仓和成本保持不变。</p>
      ${num(diagnosis.zero_fill_after_last) > 0 ? `<button type="button" class="table-button" data-activity-after-last="1">查看这 ${afterLast} 条</button>` : ""}`;
    box.hidden = false;
  }

  function activityMetrics(response) {
    const { summary = {}, evidence = {}, inactivity_diagnosis: diagnosis = {} } = response;
    const full = diagnosis.full_run || {};
    const month = summary.last_trade_date?.slice(0, 7);
    $("#strategy-activity-summary").innerHTML = metrics([
      ["模拟成交", count(summary.trades)], ["订单记录", count(summary.orders)],
      ["全运行最后成交", summary.last_trade_date || "从未成交"],
      ...(summary.last_trade_date ? [["其后零成交订单（不含当日）", count(summary.unfilled_after_last)]] : [
        ["计划调仓", count(full.rebalances)], ["调仓日达标", count(full.qualified_on_rebalance)],
        ["拒单", count(full.rejected_orders)], ["下单前拦截", count(full.pretrade_rejected_orders)],
      ]),
    ]) + (month ? `<button type="button" class="table-button" data-activity-month="${h(month)}" data-activity-day="${h(summary.last_trade_date)}">查看最后成交月</button>` : "");
    activityDiagnosis(response);
    $("#strategy-activity-evidence").textContent = evidence.basis === "pinned_archive"
      ? (evidence.selection_mode === "cross_sectional_rank"
        ? `分数统计来自锁定预测。每 ${count(evidence.rebalance_days)} 个交易日调仓，有限分数均参与横截面排名，不设置绝对分数门槛。`
        : `分数统计来自锁定预测。每 ${count(evidence.rebalance_days)} 个交易日调仓，门槛为分数大于 ${pct(evidence.score_threshold)}。达标数按股票交易日计。`)
      : (evidence.reason || "此运行未提供可核验的账户分数。");
    const items = arr(response.months);
    const monthField = activityField("month");
    const selected = monthField.value;
    monthField.replaceChildren(option("", "所有月份"), ...items.slice().reverse().map(({ month: value }) => option(value, value)));
    monthField.value = items.some((item) => item.month === selected) ? selected : "";
    $("#strategy-activity-months").innerHTML = items.length ? scrollTable([
      ["月份", "code"], ["交易日", "numeric"], ["有效原价覆盖", "numeric"], ["计划调仓", "numeric"], ["分数覆盖日", "numeric"], ["有分数股日", "numeric"],
      ["调仓日达标", "numeric"], ["非调仓日达标", "numeric"],
      ["成交", "numeric"], ["部分成交", "numeric"], ["拒单", "numeric"],
      ["下单前拦截", "numeric"], ["总费用（元）", "numeric"], ["期末现金（元）", "numeric"],
      ["陈旧估值持仓", "numeric"], ["退市滞留持仓", "numeric"],
    ], items.slice().reverse().map((row) => [
      `<button type="button" class="table-button" data-activity-month="${h(row.month)}" data-activity-day="${h(row.last_date || "")}">${h(row.month)}</button>`,
      count(row.sessions), pct(num(row.valid_raw_stock_days) / num(row.active_stock_days)),
      count(row.rebalance_days), count(row.score_days), count(row.eligible_stock_days),
      count(row.positive_on_rebalance), count(row.positive_off_rebalance),
      count(row.trades), count(row.partial), count(row.rejected), count(row.pretrade_rejected),
      decimals(row.fees, 2), decimals(row.cash_at_end, 2), count(row.stale_positions_at_end), count(row.delisted_positions_at_end),
    ])) : empty("暂无月度记录", "此运行没有取得共同回放日期。");
  }

  function activityTable(response) {
    const { filtered = {}, downloads = {} } = response;
    const items = arr(response.entries);
    const download = downloads.orders && /^\/api\/expert\/replays\/[a-z0-9_.-]+\/downloads\/[a-z0-9_.-]+$/i.test(downloads.orders)
      ? ` <a class="table-button" href="${h(downloads.orders)}" download>下载原始订单</a>` : "";
    const month = activityField("month").value;
    const hasDays = !month || arr(response.months).some((item) => item.month === month && item.sessions > 0);
    activityRows.innerHTML = (items.length ? scrollTable([
      ["执行／尝试日", "code"], ["信号日", "code"], ["股票", "code"], ["方向"], ["委托股数", "numeric"],
      ["成交股数", "numeric"], ["成交价", "numeric"], ["费用（元）", "numeric"], ["状态"], ["原因"], ["原账本行", "numeric"],
    ], items.map((row) => [
      `<button type="button" class="table-button" data-activity-day="${h(row.date)}">${h(row.date)}</button>`, h(row.signal_date),
      h(row.symbol), h(actionLabels[row.action] || actionLabels[row.side] || row.side), count(row.requested_shares),
      count(row.filled_shares), decimals(row.price, 5), decimals(row.fee, 4),
      h(orderLabels[row.status] || row.status), h(orderReason(row)), count(row.source_row),
    ])) : empty(hasDays ? "没有符合筛选的订单" : "该月份没有回放覆盖", hasDays ? "可切换月份、股票代码或订单状态。" : "月度表保留了请求范围内缺少模型或行情证据的月份。")) + (download ? `<div class="strategy-downloads">${download}</div>` : "");
    activityView.page = filtered.page || 1;
    activityView.pages = filtered.pages || 1;
    activityPager.hidden = activityView.pages <= 1;
    $("[data-activity-page=prev]").disabled = activityView.page <= 1;
    $("[data-activity-page=next]").disabled = activityView.page >= activityView.pages;
    $("#strategy-activity-pages").textContent = `第 ${activityView.page}／${activityView.pages} 页`;
    activityStatus.textContent = activityView.afterLast
      ? `全运行最后成交后的零成交订单 ${count(filtered.total)} 条。`
      : `符合筛选的订单 ${count(filtered.total)} 条。`;
  }

  function orderReason(row) {
    if (["price_limit", "approximate_price_limit"].includes(row.reason)) {
      const limit = row.side === "BUY" ? "涨停" : "跌停";
      return row.reason === "price_limit" ? `开盘触及${limit}，未成交` : `开盘触及${limit}，按板块默认限幅估计`;
    }
    return reasons[row.reason] || row.reason;
  }

  async function loadActivity(pageNumber = 1) {
    if (!activityView.run) return;
    const request = ++activityView.refresh;
    const params = new URLSearchParams(new FormData(activityForm));
    if (params.get("side") === "all") params.delete("side");
    if (activityView.afterLast) params.set("after_last", "1");
    params.set("page", String(pageNumber));
    activityStatus.textContent = "正在读取回测账本…";
    activityRows.setAttribute("aria-busy", "true");
    try {
      const response = await api(`${resultFor(activityView.run)}/activity?${params}`);
      if (request !== activityView.refresh) return;
      activityMetrics(response);
      activityTable(response);
    } catch (error) {
      if (request !== activityView.refresh) return;
      activityRows.replaceChildren();
      activityPager.hidden = true;
      activityStatus.textContent = error.message;
    } finally {
      if (request === activityView.refresh) activityRows.removeAttribute("aria-busy");
    }
  }

  function showActivity(run, accounts) {
    activityView.run = run;
    activityView.refresh += 1;
    activityView.afterLast = false;
    const namesInRun = ["full_pool", "watchlist"].filter((id) => accounts[id]);
    activity.hidden = !namesInRun.length;
    chartMonth.hidden = activity.hidden || !local.dates.length;
    if (activity.hidden) return;
    activityForm.reset();
    activityField("account").replaceChildren(...namesInRun.map((id) => option(id, names[id])));
    activityField("month").replaceChildren(option("", "所有月份"));
    $("#strategy-activity-summary").replaceChildren();
    $("#strategy-activity-diagnosis").replaceChildren();
    $("#strategy-activity-diagnosis").hidden = true;
    $("#strategy-activity-months").replaceChildren();
    $("#strategy-activity-evidence").textContent = "";
    activityRows.replaceChildren();
    activityPager.hidden = true;
    loadActivity();
  }

  activityForm.addEventListener("submit", (event) => { event.preventDefault(); activityView.afterLast = false; loadActivity(); });
  activityField("account").addEventListener("change", () => { activityView.afterLast = false; loadActivity(); });
  activity.addEventListener("click", (event) => {
    const control = event.target.closest("[data-activity-day], [data-activity-page], [data-activity-after-last]");
    if (!control) return;
    if (control.dataset.activityAfterLast) {
      activityField("month").value = "";
      activityField("symbol").value = "";
      activityField("side").value = "all";
      activityField("status").value = "all";
      activityView.afterLast = true;
      loadActivity();
      return;
    }
    if (control.dataset.activityPage) {
      loadActivity(activityView.page + (control.dataset.activityPage === "next" ? 1 : -1));
      return;
    }
    activityView.afterLast = false;
    selectChartDay(control.dataset.activityDay);
    if (control.dataset.activityMonth) {
      activityField("month").value = control.dataset.activityMonth;
      loadActivity();
    }
  });
  chartMonth.addEventListener("click", () => {
    if (!local.readoutDay) return;
    activityView.afterLast = false;
    activityField("month").value = local.readoutDay.slice(0, 7);
    activity.scrollIntoView({ block: "start" });
    loadActivity();
  });

  function accountPanel(id, account) {
    const metricsReport = account.metrics || {};
    const downloads = Object.entries(account.downloads || {}).filter(([name, path]) => Object.hasOwn(labels, name) &&
      /^\/api\/expert\/replays\/[a-z0-9_.-]+\/downloads\/[a-z0-9_.-]+$/i.test(path)).map(([name, path]) =>
      `<a class="table-button" href="${h(path)}" download>下载${h(labels[name])}</a>`).join("");
    const stranded = metricsReport.stranded_delisted_value_at_end;
    const execution = metrics([
      ["扣费累计收益", pct(metricsReport.total_return)], ["最大回撤", pct(metricsReport.max_drawdown)],
      ["日均换手", pct(metricsReport.mean_daily_turnover)], ["拒单率", pct(metricsReport.reject_rate)],
      ["模拟成交", count(metricsReport.trade_count)], ["交易费用（元）", decimals(metricsReport.fees, 2)],
      ["模拟滑点金额（元）", decimals(metricsReport.modeled_slippage_paid, 2)],
      [metricsReport.listed_positions_at_end === undefined ? "期末持仓" : "期末在市持仓",
        count(metricsReport.listed_positions_at_end ?? metricsReport.positions_at_end)],
      ...(stranded === undefined || stranded === null ? [] : [
        ["退市滞留持仓", count(metricsReport.delisted_positions_at_end)],
        ["退市滞留估值（元）", decimals(stranded, 2)],
        ["退市资产零回收收益", pct(metricsReport.zero_recovery_return)],
      ]),
    ]);
    const note = num(stranded) > 0 ? '<p class="panel-inset strategy-note">退市持仓按最后核验价格估值。零回收收益将这部分估值置零。</p>' : "";
    return panel(`${names[id]}账户`, execution + note + (downloads ? `<div class="strategy-downloads">${downloads}</div>` : ""));
  }

  function range(start, end) { return start && end ? `${start} 至 ${end}` : "—"; }

  function candidateTable(payload) {
    const rows = arr(payload.candidates).map((item) => {
      const query = new URLSearchParams({ symbol: item.stock, source_run: payload.source_run_id,
        as_of: payload.signal_date, backtest_run: payload.candidate_of, candidate_run: payload.run_id });
      if (payload.model_version) query.set("model_version", payload.model_version);
      const reason = [item.market_status, item.limit_basis, item.risk_warning_status].filter(Boolean).join(" · ");
      return [`<a class="table-button" href="#stock-chart?${query}">${h(item.stock)}</a>`, h(item.side),
        decimals(item.q10, 6), decimals(item.q50, 6), decimals(item.q90, 6), decimals(item.score, 6),
        pct(item.target_weight), count(item.requested_shares), h(reason || item.reason || "下一交易日开盘预申报")];
    });
    return scrollTable([["股票"], ["动作"], ["P10", "numeric"], ["中位数", "numeric"], ["P90", "numeric"],
      ["策略分数", "numeric"], ["目标权重", "numeric"], ["预申报股数", "numeric"], ["交易证据"]], rows);
  }

  async function showCandidates(report, run) {
    const host = $("#strategy-candidates");
    if (!host) return;
    const source = report.source || {};
    const request = ++local.candidateRequest;
    if (!source.plan_next) {
      const next = source.next_trade_date;
      host.innerHTML = panel("次日研究观察名单", `<div class="panel-inset"><p>信号日 ${h(source.end)}；${next ? `目标交易日 ${h(next)}` : "未核验下一交易日"}。</p>
        <p>名单来自滚动回放和下一开盘预申报，不是最终拟合模型的正式前瞻推荐。</p>
        <button class="button" type="button" data-candidate-generate="${h(run)}" ${!next || local.candidatePending ? "disabled" : ""}>生成研究观察名单</button></div>`);
      return;
    }
    host.innerHTML = panel("次日研究观察名单", '<div class="empty"><strong>正在读取封存候选</strong><p>不会触发重新推理。</p></div>');
    try {
      const payload = await api(`/api/expert/replays/${encodeURIComponent(run)}/candidates`);
      if (request !== local.candidateRequest || local.rendered !== run) return;
      const warnings = arr(payload.blockers).map((message) => `<p class="warn-text">${h(message)}</p>`).join("");
      const body = `<div class="panel-inset">${metrics([["信号日", payload.signal_date], ["目标交易日", payload.trade_date],
        ["模型版本", payload.model_version || payload.model_id], ["数据状态", payload.freshness === "stale" ? "历史／已过期" : "目标日可用"]])}
        ${warnings}<p>${h(payload.empty_reason || "仅列出形成有效买入预申报的股票；订单未成交。")}</p>
        <div class="strategy-downloads"><a class="table-button" href="${h(payload.downloads?.candidates)}" download>下载候选 CSV</a>
        <a class="table-button" href="${h(payload.downloads?.report)}" download>下载候选证据 JSON</a></div></div>${candidateTable(payload)}`;
      host.innerHTML = panel("次日研究观察名单", body);
    } catch (error) {
      if (request === local.candidateRequest) host.innerHTML = panel("次日研究观察名单", empty("候选暂不可用", error.message));
    }
  }

  function render(report, run) {
    const { source = {}, window: dates = {}, coverage = {}, accounts = {} } = report;
    const fees = source.execution_costs || {};
    const requested = range(coverage.requested_start || source.start, coverage.requested_end || source.end);
    const actual = range(coverage.model_start ?? (dates.sessions ? dates.start : null),
      coverage.model_end ?? (dates.sessions ? dates.end : null));
    const missing = arr(coverage.unavailable_symbols);
    const notes = arr(report.warnings).filter((note) => typeof note === "string" && note);
    if (missing.length) notes.push(`缺少覆盖的股票 ${[...new Set(missing)].join("、")}`);
    const listed = arr(source.symbols || report.symbols);
    const scope = panel("回测口径", metrics([
      ["请求区间", requested], ["可运行区间", actual], ["交易日", count(dates.sessions)], ["模型版本", source.model_version || source.model_id],
      ["冻结研究编号", source.source_run_id], ["状态", source.approval === "candidate" ? "研究候选" : currentLabels[source.approval] || source.approval],
      ["成本档位", source.cost === "double" ? "双倍佣金与滑点" : "常规成本"], ["初始资金（元）", count(dates.initial_cash ?? source.initial_cash)],
      ["佣金（万分之）", Number.isFinite(num(fees.commission)) ? decimals(fees.commission * 10000, 4) : "—"],
      ["滑点（万分之）", Number.isFinite(num(fees.slippage)) ? decimals(fees.slippage * 10000, 4) : "—"],
      ["最低佣金（元）", fees.min_commission], ["印花税（仅卖出）", pct(fees.stamp_tax)],
      ["过户费（双边，万分之）", decimals(num(fees.transfer_fee) * 10000, 4)], ["基准", report.performance?.benchmark],
    ]) + `<div class="panel-inset"><p>关注股范围 ${listed.length ? h(listed.join("、")) : "空名单"}</p>` +
      (requested !== actual && actual !== "—" ? "<p>实际回测已按模型与行情的共同可用日期截取。</p>" : "") +
      notes.map((note) => `<p>${h(note)}</p>`).join("") +
      '<p>信号以盘后数据生成，模拟使用下一交易日开盘价。账户独立计算现金、成交和费用。</p></div>');
    renderChart(report.performance);
    result.innerHTML = scope + ["full_pool", "watchlist"].filter((id) => accounts[id]).map((id) => accountPanel(id, accounts[id])).join("") +
      '<div id="strategy-candidates"></div>';
    showActivity(run, accounts);
    showHoldings(run, accounts);
    local.rendered = run;
    showCandidates(report, run);
    window.SectorResonance?.show(run, accounts);
  }

  function resultFor(run) { return `/api/expert/replays/${encodeURIComponent(run)}`; }
  function currentRun() { return new URLSearchParams(location.hash.split("?")[1]).get("run"); }

  async function openResult(run, request) {
    if (!runKey.test(run) || local.rendered === run) return;
    const report = await api(resultFor(run));
    if (request === local.request && !page.hidden) { render(report, run); feedback.textContent = "回测结果已就绪。"; }
  }

  async function poll(request, delay = 0) {
    clearTimeout(local.timer);
    const job = local.job;
    if (page.hidden || !job || !/^\/api\/expert\/jobs\/[a-z0-9_.-]+$/i.test(job.status_url)) return;
    if (delay) { local.timer = setTimeout(() => poll(request), delay); return; }
    try {
      const snapshot = active(job) ? await api(job.status_url) : job;
      if (request !== local.request || page.hidden) return;
      local.job = snapshot;
      writeStorage("sessionStorage", jobKey, snapshot);
      feedback.textContent = snapshot.message || "正在回放模拟账户…";
      if (active(snapshot)) poll(request, 1200);
      else if (snapshot.status === "completed") await openResult(snapshot.run_id, request);
    } catch (error) { if (request === local.request) feedback.textContent = error.message; }
    finally { if (request === local.request) controls(); }
  }

  async function show() {
    clearTimeout(local.timer);
    const request = ++local.request;
    if (page.hidden) return;
    try {
      const context = new URLSearchParams(location.hash.split("?")[1]);
      const sourceRun = context.get("source_run") || "";
      const model = await readLinkedModel(context, { refresh: true });
      if (request !== local.request || page.hidden) return;
      local.model = model;
      local.pinned = Boolean(sourceRun);
      renderCurrentModel($("#strategy-model"), model);
      if (!local.initialized && isoDay.test(model.as_of)) {
        input("start").value = model.analysis_start || (model.as_of < "2025-03-03" ? model.as_of : "2025-03-03");
        input("end").value = model.as_of;
        local.initialized = true;
      }
      const selected = currentRun();
      if (!local.job) local.job = readStorage("sessionStorage", jobKey, null);
      if (local.job && (!selected || selected === local.job.run_id)) await poll(request);
      else if (selected && runKey.test(selected)) await openResult(selected, request);
      else if (!local.rendered) feedback.textContent = model.run_id ? "已就绪，可运行回测。" : (model.reason || "当前策略模型待发布。");
    } catch (error) { if (request === local.request) feedback.textContent = error.message; }
    finally { if (request === local.request) controls(); }
  }

  result.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-candidate-generate]");
    if (!button || local.candidatePending || button.disabled) return;
    const backtest = button.dataset.candidateGenerate;
    if (!runKey.test(backtest)) return;
    local.candidatePending = true;
    button.disabled = true;
    button.textContent = "正在生成预申报…";
    try {
      const submitted = await api("/api/expert/candidates", requestOptions({ backtest_run_id: backtest }));
      local.job = submitted;
      writeStorage("sessionStorage", jobKey, submitted);
      local.rendered = "";
      const previous = new URLSearchParams(location.hash.split("?")[1]);
      const query = new URLSearchParams({ run: submitted.run_id });
      for (const key of ["source_run", "model_version"]) {
        if (previous.get(key)) query.set(key, previous.get(key));
      }
      history.replaceState(null, "", `#backtest?${query}`);
      await poll(local.request);
    } catch (error) {
      feedback.textContent = error.message;
      button.disabled = false;
    } finally {
      local.candidatePending = false;
      button.textContent = "生成研究观察名单";
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submit.disabled || !form.reportValidity()) return;
    const data = new FormData(form);
    const body = Object.fromEntries(["start", "end", "cost", "initial_cash", "commission_bps", "slippage_bps"].map((name) => [name, data.get(name)]).filter(([, value]) => value !== ""));
    body.sector_provider = "ths_fuyao";
    if (body.end < body.start) { feedback.textContent = "结束日期应在开始日期当天或之后。"; return; }
    if (Object.hasOwn(body, "initial_cash")) {
      body.initial_cash = Number(body.initial_cash);
      if (!(body.initial_cash > 0)) { feedback.textContent = "初始资金必须大于 0。"; return; }
    }
    body.symbols = [...watchlist.keys()];
    if (local.pinned) {
      body.source_run_id = local.model.source_run_id;
      if (local.model.model_version) body.model_version = local.model.model_version;
    }
    local.pending = true;
    controls();
    const request = local.request;
    feedback.textContent = "正在锁定当前模型、股票池和回测设置…";
    try {
      const submitted = await api("/api/expert/portfolio-replays", requestOptions(body));
      local.job = submitted;
      writeStorage("sessionStorage", jobKey, submitted);
      if (request === local.request && !page.hidden) {
        if (local.rendered && local.rendered !== submitted.run_id) {
          local.rendered = "";
          renderChart(null);
          placeholder.textContent = "等待新的回测结果。";
          result.replaceChildren();
          showActivity("", {});
          showHoldings("", {});
          window.SectorResonance?.clear();
        }
        history.replaceState(null, "", `#backtest?${new URLSearchParams({ run: submitted.run_id })}`);
        await poll(request);
      }
    } catch (error) { if (request === local.request) feedback.textContent = error.message; }
    finally { local.pending = false; controls(); }
  });

  updateWatchlist();
  window.addEventListener("hashchange", show);
  $("#refresh").addEventListener("click", show);
  show();
})();
