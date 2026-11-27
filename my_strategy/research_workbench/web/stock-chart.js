"use strict";

(() => {
  const page = $("#page-stock-chart");
  const form = $("#stock-chart-form");
  const symbolInput = $("#stock-chart-symbol");
  const suggestions = $("#stock-chart-suggestions");
  const costSelect = $("#stock-chart-cost");
  const startInput = $("#stock-chart-start");
  const endInput = $("#stock-chart-end");
  const overlay = $("#stock-chart-overlay");
  const modelBox = $("#stock-chart-current-model");
  const latestBox = $("#stock-chart-latest");
  const rankBox = $("#stock-chart-rank");
  const assumptions = {
    initial_cash: $("#stock-chart-initial-cash"), commission_bps: $("#stock-chart-commission"),
    slippage_bps: $("#stock-chart-slippage"),
  };
  const stage = $("#stock-chart-stage");
  const markerLayer = $("#stock-chart-markers");
  const ledger = $("#stock-chart-ledger-rows");
  const defaultSymbol = "002436.SZ";
  const costNames = { normal: "常规成本", double: "双倍佣金／滑点" };
  const sideNames = { BUY: "买入", SELL: "卖出", REDUCE: "减仓" };
  const actionNames = { buy: "买入", add: "加仓", hold: "持有", reduce: "减仓", sell: "卖出", watch: "观察", unavailable: "暂无可用动作" };
  const stageNames = { signal: "盘后信号", fill: "模拟成交", partial: "部分成交", rejected: "拒单" };
  const stateNames = {
    available: "证据可用", covered: "证据可用", replayed: "已回放", replayed_without_orders: "已计算 · 无委托",
    no_events: "区间无事件", not_covered: "研究未覆盖",
    missing_evidence: "研究证据缺失", traded: "原始行情", tradable: "原始行情", suspended: "停牌", delisted: "退市",
    missing: "缺少行情", no_trade_price: "无有效成交价",
  };
  const local = {
    symbol: defaultSymbol, view: null, viewRequest: 0, model: null, accountWindow: null,
    suggestions: [], suggestionIndex: -1, searchTimer: null, searchRequest: 0,
    filter: "all", events: [], byId: new Map(), rows: new Map(), groups: [],
    activeId: "", activeGroup: "", chart: null, bars: new Map(), frame: 0, loading: false, pinned: false,
  };
  const numeric = (value) => Number.isFinite(num(value));
  const dateText = (value) => typeof value === "string" ? value.slice(0, 10) : "";
  const exactPrice = (value) => numeric(value) ? num(value).toFixed(5) : "—";
  const exactFee = (value) => numeric(value) ? num(value).toFixed(8) : "—";
  const variant = (event) => event.stage === "fill" && event.status === "partial" ? "partial" : event.stage;
  const eventName = (event) => stageNames[variant(event)] || text(event.stage);
  const sideName = (event) => sideNames[event.action] || sideNames[event.side] || text(event.side);
  const eventSide = (event) => event.action === "REDUCE" || event.side === "SELL" || event.action === "SELL" ? "sell"
    : event.side === "BUY" || event.action === "BUY" ? "buy" : "";
  const eventDay = (event) => dateText(event.time);
  const filledShares = (event) => event.stage === "signal" ? null
    : event.stage === "fill" ? event.filled_shares ?? event.shares : event.filled_shares;
  const availableBars = (bar) => [bar.open, bar.high, bar.low, bar.close].every(numeric);
  const urlHash = () => new URLSearchParams(location.hash.split("?")[1] || "");
  const openPage = () => !page.hidden;

  function endpoint(action, params = {}) {
    return `/api/stock-chart/${action}?${new URLSearchParams(params)}`;
  }

  function feedback(message = "") {
    const target = $("#stock-chart-error");
    target.textContent = message;
    target.hidden = !message;
  }

  function calculation(message = "") {
    const target = $("#stock-chart-calculation");
    target.textContent = message;
    target.hidden = !message;
  }

  function closeSuggestions() {
    suggestions.hidden = true;
    symbolInput.setAttribute("aria-expanded", "false");
    symbolInput.removeAttribute("aria-activedescendant");
  }

  function searchInfo(message = "") {
    const node = $("#stock-chart-search-info");
    node.textContent = message;
    node.hidden = !message;
  }

  function buildSuggestions(rows) {
    local.suggestions = rows;
    local.suggestionIndex = -1;
    symbolInput.removeAttribute("aria-activedescendant");
    suggestions.replaceChildren(...rows.map((stock, index) => {
      const item = document.createElement("li");
      item.id = `stock-chart-suggestion-${index}`;
      item.dataset.suggestion = index;
      item.setAttribute("role", "option");
      item.setAttribute("aria-selected", "false");
      item.innerHTML = `<strong>${h(stock.name || stock.symbol)}</strong><span class="code">${h(stock.symbol)}</span>`;
      return item;
    }));
    suggestions.hidden = !rows.length;
    symbolInput.setAttribute("aria-expanded", String(!!rows.length));
  }

  async function search(query, request, choose = false) {
    if (!query) return;
    try {
      const result = await api(endpoint("search", { q: query }));
      if (request !== local.searchRequest) return;
      const matches = arr(result.symbols);
      if (choose && matches.length === 1) {
        selectMatched(matches[0].symbol);
        return;
      }
      buildSuggestions(matches);
      searchInfo(matches.length ? "" : "本地行情仓未匹配该代码或名称。");
    } catch (error) {
      if (request === local.searchRequest) searchInfo(error.message);
    }
  }

  function controlsDisabled(disabled) {
    const apply = $("#stock-chart-apply");
    for (const field of [startInput, endInput, apply]) field.disabled = disabled;
    for (const field of [costSelect, ...Object.values(assumptions)]) field.disabled = disabled || !modelAvailable();
    overlay.disabled = disabled || !arr(local.view?.strategies).some((strategy) => strategy.id === "full_pool"
      && (arr(strategy.events).length || arr(strategy.predictions).length || arr(strategy.positions).length));
    apply.textContent = disabled ? "读取中…" : modelAvailable() ? "运行单股回放" : "查看区间";
  }

  function modelAvailable() {
    return ["approved", "research_only", "production_fallback"].includes(local.model?.status) && Boolean(local.model.model_id);
  }

  function renderModel() {
    const model = local.model;
    const context = local.view?.context;
    const pinnedId = context?.model_id;
    const status = pinnedId ? context.replay?.approval || model?.status : model?.status;
    const tag = { approved: "正式模型", candidate: "研究候选", research_only: "研究候选", production_fallback: "生产回退" }[status] || "待发布";
    const selected = !pinnedId || pinnedId === model?.model_id ? model : null;
    const cutoff = dateText(context?.model_as_of || selected?.as_of);
    const label = local.pinned || status === "production_fallback" ? model?.label : (status === "approved" || status === "unavailable" ? selected?.label : "");
    modelBox.innerHTML = `<strong>${h(tag)}</strong>${label ? ` <span>${h(label)}</span>` : ""}` +
      (cutoff ? ` <span>数据截至 ${h(cutoff)}</span>` : "") +
      ((pinnedId || model?.model_id) ? ` <code>${h(pinnedId || model.model_id)}</code>` : "") +
      (local.pinned ? ` <span>研究 ${h(model.source_run_id)}</span>` : "") +
      (urlHash().get("candidate_run") ? ` <span>候选回放 ${h(urlHash().get("candidate_run"))}</span>` : "");
  }

  function stateToURL() {
    const query = new URLSearchParams({ symbol: local.symbol, ...requestSettings() });
    const previous = urlHash();
    for (const key of ["source_run", "model_version", "backtest_run", "candidate_run"]) {
      if (previous.get(key)) query.set(key, previous.get(key));
    }
    if (overlay.checked) query.set("overlay", "1");
    history.replaceState(null, "", `#stock-chart?${query}`);
  }

  function chooseSymbol(symbol, settings = {}) {
    local.symbol = symbol.toUpperCase();
    local.filter = "all";
    ++local.searchRequest;
    closeSuggestions();
    searchInfo();
    symbolInput.value = local.symbol;
    startInput.value = dateText(settings.start);
    endInput.value = dateText(settings.end || settings.as_of);
    costSelect.value = Object.hasOwn(costNames, settings.cost) ? settings.cost : "normal";
    for (const [key, input] of Object.entries(assumptions)) input.value = settings[key] || "";
    overlay.checked = settings.overlay === "1";
    return loadView();
  }

  function requestSettings() {
    const fields = { start: startInput.value, end: endInput.value, cost: costSelect.value || "normal" };
    for (const [key, input] of Object.entries(assumptions)) {
      if (input.value !== "") fields[key] = key === "initial_cash" ? input.valueAsNumber : input.value;
    }
    return fields;
  }

  function resetView(message) {
    local.events = [];
    local.byId.clear();
    local.activeId = local.activeGroup = "";
    local.groups = [];
    markerLayer.replaceChildren();
    for (const box of [latestBox, rankBox]) {
      box.hidden = true;
      box.replaceChildren();
    }
    stage.hidden = true;
    $("#stock-chart-empty").hidden = false;
    $("#stock-chart-empty").textContent = message;
    $("#stock-chart-fit").disabled = true;
    $("#stock-chart-description").textContent = "";
    $("#stock-chart-readout").replaceChildren();
    $("#stock-chart-evidence-title").textContent = "区间证据";
    $("#stock-chart-coverage").textContent = "等待行情和回测账本。";
    $("#stock-chart-name").textContent = local.symbol || "日线图";
    renderSelection();
    renderLedger();
  }

  async function loadView(runReplay = false) {
    const request = ++local.viewRequest;
    const query = new URLSearchParams({ symbol: local.symbol });
    if (startInput.value) query.set("start", startInput.value);
    if (endInput.value) query.set("end", endInput.value);
    local.loading = true;
    local.view = null;
    local.accountWindow = null;
    feedback();
    calculation();
    resetView("正在读取本地行情…");
    controlsDisabled(true);
    stage.setAttribute("aria-busy", "true");
    try {
      const [prices, current] = await Promise.allSettled([
        api(`/api/stock-chart/view?${query}`),
        readLinkedModel(urlHash(), { refresh: true }),
      ]);
      if (request !== local.viewRequest) return;
      local.model = current.status === "fulfilled" ? current.value : null;
      local.pinned = Boolean(urlHash().get("source_run"));
      renderModel();
      if (prices.status === "fulfilled") {
        local.view = prices.value;
        const { symbol, name, context } = local.view;
        local.symbol = symbol || local.symbol;
        symbolInput.value = name ? `${local.symbol} ${name}` : local.symbol;
        const useDefaultStart = !startInput.value;
        if (useDefaultStart) startInput.value = dateText(context?.start);
        if (!endInput.value) endInput.value = dateText(context?.end);
        renderView();
      }
      if (prices.status === "rejected") {
        throw prices.reason;
      }
      stateToURL();
      if (!modelAvailable()) {
        const reason = current.status === "rejected" ? current.reason?.message : local.model?.reason;
        renderLatest(null, reason || "暂时没有可运行的当前模型。");
        return;
      }
      if (!runReplay) {
        renderLatest(null, "页面仅读取本地行情；点击“运行单股回放”后才会提交模型任务。");
        return;
      }
      try {
        await loadStockReplay(request);
      } catch (error) {
        if (request !== local.viewRequest) return;
        calculation(error.message);
        renderLatest(null, "当前股票或区间缺少可用的回放证据。");
      }
    } catch (error) {
      if (request !== local.viewRequest) return;
      resetView("当前股票在所选区间没有可展示的本地行情。");
      feedback(error.message);
    } finally {
      if (request === local.viewRequest) {
        local.loading = false;
        controlsDisabled(false);
        stage.removeAttribute("aria-busy");
      }
    }
  }

  async function loadStockReplay(request) {
    calculation("正在运行独立单股回放…");
    const payload = { symbol: local.symbol, ...requestSettings() };
    if (local.pinned) {
      payload.source_run_id = local.model.source_run_id;
      if (local.model.model_version) payload.model_version = local.model.model_version;
    }
    const job = await api("/api/expert/stock-replays", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    let state = job;
    while (request === local.viewRequest && ["queued", "running"].includes(state.status)) {
      calculation(state.message || "正在核对模型和回放账本…");
      await new Promise((resolve) => setTimeout(resolve, 950));
      if (request !== local.viewRequest) return;
      state = await api(job.status_url);
    }
    if (request !== local.viewRequest) return;
    if (state.status !== "completed") throw new Error(state.message || "单股回放未完成。");
    const result = await api(job.result_url);
    if (request !== local.viewRequest) return;
    if (result.chart?.symbol !== local.symbol) throw new Error("回放结果与当前股票不一致。");
    local.view = result.chart;
    local.accountWindow = result.window;
    renderModel();
    renderView();
    renderLatest(result.latest_action || result.chart.latest_action);
    calculation();
  }

  function renderLatest(action, reason) {
    latestBox.hidden = false;
    rankBox.hidden = !action?.original_pool;
    if (!action) {
      latestBox.innerHTML = '<h2>最新单股动作</h2><p>' + h(reason || "所选区间内暂无可用动作。") + "</p>";
      rankBox.replaceChildren();
      return;
    }
    const state = action.state;
    const direction = ["buy", "add"].includes(state) ? "buy" : ["sell", "reduce"].includes(state) ? "sell" : "";
    const fields = [
      ["信号日", dateText(action.signal_date)], ["下一计划执行日", dateText(action.next_trade_date)],
      ["当前持仓", numeric(action.current_shares) ? `${count(action.current_shares)} 股` : "—"],
      ["当前仓位", pct(action.current_weight)], ["目标仓位", pct(action.target_weight)],
      ["五日超额收益 q10／q50／q90", numeric(action.q50) ? `${pct(action.q10)} ／ ${pct(action.q50)} ／ ${pct(action.q90)}` : "—"],
      ["调仓状态", action.rebalance_due ? "当日调仓" : action.signal_date ? "未到调仓日" : "—"],
    ];
    const pending = action.pending_order;
    if (pending?.requested_shares > 0) {
      fields.push(["待模拟执行", `${sideNames[pending.side] || pending.side} ${count(pending.requested_shares)} 股`]);
    }
    latestBox.innerHTML = `<header><h2>最新单股动作</h2><strong class="stock-chart-side-${direction || "neutral"}">${h(action.label || actionNames[state] || "暂无可用动作")}</strong></header>` +
      `<dl>${metricItems(fields)}</dl>` + (action.reason ? `<p>${h(action.reason)}</p>` : "");
    const original = action.original_pool;
    if (!original) {
      rankBox.replaceChildren();
      return;
    }
    const inserted = original.rank_basis === "hypothetical_insertion";
    const total = numeric(original.total) ? num(original.total) + Number(inserted) : null;
    const rank = numeric(original.rank) && numeric(total) ? `${count(original.rank)}／${count(total)}` : "—";
    const status = actionNames[String(original.action || "").toLowerCase()] || original.action;
    rankBox.innerHTML = '<h2>全池参考</h2><dl>' + metricItems([
      [inserted ? "拟插入全池排名" : "同日分数排名", rank], ["全池动作", status], ["全池该股持仓", numeric(original.shares) ? `${count(original.shares)} 股` : "—"],
      ["全池该股仓位", pct(original.weight)],
    ]) + "</dl>";
  }

  function buildChart() {
    if (local.chart) return local.chart;
    if (!window.KHQuantStockResearch) return null;
    const chart = window.KHQuantStockResearch.create({
      host: $("#stock-chart-price"),
      onInspect: (bar, forecast, date) => renderReadout(bar, date, forecast),
      onRender: positionMarkers,
    });
    local.chart = chart;
    return local.chart;
  }

  function latestBar() {
    return [...local.bars.values()].findLast((bar) => availableBars(bar));
  }

  function renderReadout(bar, date = "", forecast = null) {
    const readout = $("#stock-chart-readout");
    if (forecast) {
      readout.innerHTML = metricItems([["预测交易日", date], ["P10 收盘", decimals(forecast.close_p10, 2)],
        ["中位收盘", decimals(forecast.close, 2)], ["P90 收盘", decimals(forecast.close_p90, 2)],
        ["模型中位量", count(forecast.volume)], ["性质", "冻结模型情景 · 非实际成交"]]);
      return;
    }
    const day = bar?.time || date;
    const fields = [["交易日", day], ["开", decimals(bar?.open, 2)], ["高", decimals(bar?.high, 2)],
      ["低", decimals(bar?.low, 2)], ["收", decimals(bar?.close, 2)], ["成交量", count(bar?.volume)]];
    if (bar && !availableBars(bar)) fields.push(["状态", stateNames[bar.status] || bar.status || "无有效价格"]);
    readout.innerHTML = metricItems(fields);
  }

  function renderView() {
    const result = local.view;
    const context = result.context || {};
    const held = new Map(arr(result.bars).map((bar) => [dateText(bar.time), bar]));
    for (const row of arr(result.nontrading)) {
      const day = dateText(row.time);
      if (day && !held.has(day)) held.set(day, { time: day, status: row.status });
    }
    // Event-only dates retain a slot without implying a trade-price observation.
    for (const strategy of arr(result.strategies)) {
      for (const event of arr(strategy.events)) {
        const day = eventDay(event);
        if (event.status !== "pending" && day && !held.has(day)) held.set(day, { time: day, status: "missing" });
      }
    }
    local.bars = new Map([...held.entries()].sort(([left], [right]) => left.localeCompare(right)));
    const observed = [...local.bars.values()].filter(availableBars);
    const title = [result.name, result.symbol || local.symbol].filter(Boolean).join("  ");
    $("#stock-chart-name").textContent = title;
    $("#stock-chart-description").textContent = ["未复权日线", context.model_id ? "独立单股回放" : "",
      context.cost && costNames[context.cost]].filter(Boolean).join(" · ");
    const emptyBox = $("#stock-chart-empty");
    const researchReady = result.research?.status === "available";
    emptyBox.hidden = !!observed.length && researchReady;
    stage.hidden = !observed.length || !researchReady;
    $("#stock-chart-fit").disabled = !observed.length || !researchReady;
    const plot = observed.length && researchReady ? buildChart() : null;
    if (plot) {
      plot.mount(result);
    } else {
      emptyBox.hidden = false;
      emptyBox.textContent = observed.length ? result.research?.reason || "研究图层暂不可用，仍可检查行情与账本证据。" : "所选区间没有可展示的原始日线。";
      stage.hidden = true;
      $("#stock-chart-fit").disabled = true;
    }
    renderReadout(latestBar());
    renderCoverage();
    renderEvents();
  }

  function renderCoverage() {
    const result = local.view;
    const context = result.context || {};
    const raw = [...local.bars.values()].filter(availableBars);
    const untraded = arr(result.nontrading).filter((row) => row.status === "suspended").length;
    const coverage = context.replay?.coverage;
    const items = [["有效行情", `${count(raw.length)} 日`],
      [coverage?.requested_start ? "冻结模型覆盖" : "研究日期",
        context.coverage_start && context.coverage_end ? `${dateText(context.coverage_start)} 至 ${dateText(context.coverage_end)}` : context.model_id ? "当前区间无覆盖" : "本地行情"],
      ["有证据停牌", `${count(untraded)} 日`]];
    if (coverage?.requested_start) items.push(
      ["选择日期", `${dateText(coverage.requested_start)} 至 ${dateText(coverage.requested_end)}`],
      ["行情实际到达", coverage.market_end ? dateText(coverage.market_end) : "区间无日线"],
    );
    if (coverage?.model_start && local.accountWindow?.initial_position === "cash") items.push(
      ["独立账户起点", dateText(coverage.model_start)],
      ["起始仓位", "全现金"],
      ["起始资金", `${count(local.accountWindow.initial_cash)} 元`],
    );
    if (coverage?.eligible_first_session) items.push(["特征通过预热", `${dateText(coverage.eligible_first_session)} 至 ${dateText(coverage.eligible_last_session)}`]);
    const effective = [...new Set(Object.values(context.replay?.devices?.effective_devices || {}).filter((item) => item !== "rule"))];
    if (effective.length) items.push(["模型推理", effective.join("／")]);
    const entries = visibleStrategies();
    const warnings = arr(result.warnings);
    const strategies = entries.length ? `<ul class="stock-chart-state-list">${entries.map((strategy) => {
      const status = !arr(strategy.events).length && strategy.status === "covered" ? "no_events"
        : !arr(strategy.events).length && strategy.status === "replayed" ? "replayed_without_orders" : strategy.status;
      return `<li><strong>${h(strategy.label || strategy.id)}</strong><span>${h(stateNames[status] || status || "—")}</span></li>`;
    }).join("")}</ul>` : "";
    const archive = arr(result.sources).find((source) => /^\/api\/expert\/replays\/[a-z0-9_.-]+$/i.test(source.report_url || ""));
    $("#stock-chart-coverage").innerHTML = `<dl class="stock-chart-evidence-list">${metricItems(items)}</dl>${strategies}` +
      (warnings.length ? `<ul class="stock-chart-warnings">${warnings.map((message) => `<li>${h(message)}</li>`).join("")}</ul>` : "") +
      (archive ? `<a class="stock-chart-source-link" href="${h(archive.report_url)}">查看回放报告</a>` : "");
  }

  function visibleStrategies() {
    return arr(local.view?.strategies).filter((strategy) => strategy.id === "single_stock"
      || (strategy.id === "full_pool" && overlay.checked));
  }

  function renderEvents() {
    local.events = visibleStrategies().flatMap((strategy) => {
      return arr(strategy.events).filter((event) => event.status !== "pending").map((event, row) => ({ ...event,
        key: `${strategy.id}|${event.id ?? row}`, arm: strategy.id, label: strategy.label || strategy.id,
      }));
    }).sort((left, right) => eventDay(left).localeCompare(eventDay(right)) || left.stage.localeCompare(right.stage));
    local.byId = new Map(local.events.map((item) => [item.key, item]));
    if (!local.byId.has(local.activeId)) local.activeId = local.activeGroup = "";
    renderGroups();
    renderSelection();
    renderLedger();
  }

  function groupKey(event) {
    return `${eventDay(event)}|${variant(event)}|${eventSide(event)}|${event.arm}`;
  }

  function renderGroups() {
    const grouped = new Map();
    for (const event of local.events) {
      const key = groupKey(event);
      if (!eventDay(event)) continue;
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(event);
    }
    const groups = [...grouped].map(([key, events]) => {
      const first = events[0];
      const day = eventDay(first);
      const stage = variant(first);
      const side = eventSide(first);
      const reference = first.arm === "full_pool";
      const sides = [...new Set(events.map(sideName))];
      const button = document.createElement("button");
      button.type = "button";
      button.className = `stock-chart-marker ${stage} ${side ? `stock-chart-side-${side}` : ""}`;
      button.dataset.stockGroup = key;
      button.dataset.stockSide = side;
      button.dataset.stockTrack = first.arm;
      button.hidden = true;
      button.title = `${day} ${first.label} ${eventName(first)} ${sides.join("／")}`;
      button.setAttribute("aria-label", button.title);
      const caption = events.length > 1 ? String(events.length) : sides[0]?.slice(0, 1) || "";
      button.innerHTML = `<i class="stock-chart-glyph ${stage}" aria-hidden="true"></i><span aria-hidden="true">${h(caption)}</span>`;
      const opposite = side === "buy" ? "sell" : "buy";
      const split = side && grouped.has(`${day}|${stage}|${opposite}|${first.arm}`);
      const peer = grouped.has(`${day}|${stage}|${side}|${reference ? "single_stock" : "full_pool"}`);
      return { key, day, events, button, stage, side, split, peer, reference };
    });
    local.groups = groups;
    markerLayer.replaceChildren(...groups.map((group) => group.button));
    updateSelected();
    positionMarkers();
  }

  function positionMarkers() {
    if (local.frame) return;
    local.frame = requestAnimationFrame(() => {
      local.frame = 0;
      const plot = local.chart;
      if (!plot || stage.hidden || !openPage()) return;
      const height = $("#stock-chart-price").clientHeight;
      markerLayer.classList.toggle("compact", plot.compact());
      for (const group of local.groups) {
        const item = group.events.find((event) => numeric(event.price));
        const point = plot.point(group.day, item ? num(item.price) : null);
        if (!point) {
          group.button.hidden = true;
          continue;
        }
        const offsets = { signal: 17, fill: -17, partial: -34, rejected: -49 };
        const level = group.stage === "signal" ? Math.max(point.priceY ?? 18, point.lowY ?? 18) : Math.min(point.priceY ?? 81, point.highY ?? 81);
        const y = Math.min(height - 62, Math.max(12, level + offsets[group.stage]));
        group.button.hidden = !point.visible || point.x < 4 || point.x > point.width - 4 || !Number.isFinite(y);
        if (!group.button.hidden) {
          const sideOffset = group.split ? (group.side === "buy" ? -8 : 8) : 0;
          const trackOffset = group.peer ? (group.reference ? 6 : -6) : 0;
          group.button.style.left = `${Math.min(point.width - 8, Math.max(8, point.x + sideOffset + trackOffset))}px`;
          group.button.style.top = `${y + (group.reference ? 8 : 0)}px`;
        }
      }
    });
  }

  function ledgerLink(event) {
    if (/^\/api\/(?:latent\/runs|stock-chart\/replays)\/[a-z0-9_-]+\/downloads\/[a-z0-9_.-]+$/i.test(event.source_url || "")) return event.source_url;
    if (/^\/api\/expert\/replays\/[a-z0-9_.-]+\/downloads\/[a-z0-9_.-]+$/i.test(event.source_url || "")) return event.source_url;
    if (event.source_url?.startsWith("/api/stock-chart/legacy-evidence?")) {
      const candidate = new URL(event.source_url, location.origin);
      const fields = [...candidate.searchParams.keys()].sort();
      if (candidate.origin === location.origin && candidate.pathname === "/api/stock-chart/legacy-evidence"
          && fields.join(",") === "event_id,source_id,symbol") return candidate.pathname + candidate.search;
    }
    const context = local.view?.context || {};
    const filename = String(event.source_file || "").split(/[\\/]/).at(-1);
    const runId = context.run_id;
    if (!/^[a-z0-9_-]+$/i.test(runId || "") || !/^[a-z0-9_]+$/i.test(context.fold || "")) return "";
    const valid = new Set([
      `${event.arm}_${context.cost}_orders.parquet`, `${event.arm}_${context.cost}_trades.parquet`,
      `${event.arm}_predictions.parquet`, "domestic_predictions.parquet",
    ]);
    return valid.has(filename) && /^[a-z0-9_.-]+$/i.test(filename)
      ? `/api/latent/runs/${encodeURIComponent(runId)}/downloads/${encodeURIComponent(`${context.fold}--${filename}`)}` : "";
  }

  function renderSelection() {
    const active = local.byId.get(local.activeId);
    const region = $("#stock-chart-events");
    $("#stock-chart-event-day").textContent = active ? eventDay(active) : "";
    if (!active) {
      region.textContent = local.view && !local.events.length ? "当前策略和区间没有可定位的事件。" : "选择图上标记或账本行，查看对应记录。";
      return;
    }
    const group = local.groups.find((item) => item.key === groupKey(active));
    const options = group && group.events.length > 1 ? `<div class="stock-chart-group-list" aria-label="同一交易日的事件">${group.events.map((entry) =>
      `<button type="button" class="table-button" data-stock-event="${h(entry.key)}"${entry.key === active.key ? ' aria-current="true"' : ""}>${h(entry.label)} · ${h(sideName(entry))}</button>`
    ).join("")}</div>` : "";
    const executionLabel = { signal: "计划执行日", fill: "成交日", rejected: "尝试执行日" };
    const details = [["信号日期", active.signal_date], [executionLabel[active.stage] || "执行日", active.execution_date],
      [active.stage === "signal" ? "信号参考价" : "模拟成交价", active.stage === "rejected" ? "—" : exactPrice(active.price)],
      ["委托股数", count(active.requested_shares)], ["成交股数", count(filledShares(active))], ["费用（元）", exactFee(active.fee)]];
    if (numeric(active.transfer_fee)) details.push(["其中过户费（元）", exactFee(active.transfer_fee)]);
    const link = ledgerLink(active);
    const downloadName = link.startsWith("/api/stock-chart/legacy-evidence?") ? "legacy-trade.csv" : "";
    const source = active.source_file ? `<p class="stock-chart-source">${h(active.source_file)}${numeric(active.source_row) ? ` · 源行 ${h(active.source_row)}` : ""}</p>` : "";
    region.innerHTML = `<div class="stock-chart-selection"><strong>${h(active.label)}</strong>${eventTag(active)}</div>${options}` +
      `<dl class="stock-chart-event-values">${metricItems(details)}</dl>` +
      (active.reason ? `<p class="stock-chart-reason">${h(active.reason)}</p>` : "") + source +
      (link ? `<a class="stock-chart-source-link" href="${h(link)}" download="${h(downloadName)}">下载对应证据表</a>` : "");
  }

  function eventTag(event) {
    const side = eventSide(event);
    return `<span class="stock-chart-event-tag ${h(variant(event))} ${side ? `stock-chart-side-${side}` : ""}" data-stock-side="${side}"><i class="stock-chart-glyph ${h(variant(event))}" aria-hidden="true"></i>${h(eventName(event))} · ${h(sideName(event))}</span>`;
  }

  function renderLedger() {
    for (const button of page.querySelectorAll("[data-stock-stage]")) {
      button.setAttribute("aria-pressed", String(button.dataset.stockStage === local.filter));
    }
    const shown = local.events.filter((event) => local.filter === "all" || event.stage === local.filter);
    local.rows = new Map();
    if (!shown.length) {
      const note = local.view ? "当前区间没有符合所选策略和事件类型的记录。" : "待载入回测证据。";
      ledger.innerHTML = empty("暂无账本记录", note);
      return;
    }
    ledger.innerHTML = table([
      ["事件日期", "nowrap"], ["账户来源"], ["事件", "nowrap"], ["信号日", "nowrap"], ["计划／执行日", "nowrap"],
      ["参考／成交价", "numeric"], ["委托股数", "numeric"], ["成交股数", "numeric"], ["费用（元）", "numeric"], ["原因", "wrap"],
    ], shown.toReversed().map((event) => [
      `<button type="button" class="table-button code" data-stock-event="${h(event.key)}">${h(eventDay(event))}</button>`,
      `<span class="stock-chart-track" data-stock-track="${h(event.arm)}">${h(event.label)}</span>`, eventTag(event), h(event.signal_date), h(event.execution_date),
      h(event.stage === "rejected" ? "—" : exactPrice(event.price)), h(count(event.requested_shares)),
      h(count(filledShares(event))), h(exactFee(event.fee)), h(event.reason),
    ]));
    local.rows = new Map([...ledger.querySelectorAll("[data-stock-event]")].map((node) => [node.dataset.stockEvent, node]));
    updateSelected();
  }

  function updateSelected() {
    for (const [id, node] of local.rows) {
      if (id === local.activeId) node.setAttribute("aria-current", "true");
      else node.removeAttribute("aria-current");
    }
    for (const group of local.groups) {
      group.button.setAttribute("aria-pressed", String(group.key === local.activeGroup));
    }
  }

  function locateCandle(event) {
    const plot = local.chart;
    const day = eventDay(event);
    if (!plot || stage.hidden || !local.bars.has(day)) return;
    plot.locate(day);
    renderReadout(local.bars.get(day), day);
  }

  function chooseEvent(id, fromChart = false) {
    const event = local.byId.get(id);
    if (!event) return;
    local.activeId = event.key;
    local.activeGroup = groupKey(event);
    if (local.filter !== "all" && event.stage !== local.filter) {
      local.filter = "all";
      renderLedger();
    }
    updateSelected();
    renderSelection();
    locateCandle(event);
    if (fromChart) {
      const row = local.rows.get(event.key)?.closest("tr");
      if (row) ledger.scrollTop = Math.max(0, row.offsetTop - ledger.offsetTop - 43);
    }
    if (!fromChart && !stage.hidden) stage.scrollIntoView({ block: "nearest" });
  }

  function selectMatched(symbol) {
    return chooseSymbol(symbol, Object.fromEntries(urlHash()));
  }

  function moveSuggestion(index) {
    const items = [...suggestions.children];
    if (!items.length) return;
    local.suggestionIndex = (index + items.length) % items.length;
    items.forEach((item, position) => item.setAttribute("aria-selected", String(position === local.suggestionIndex)));
    const current = items[local.suggestionIndex];
    symbolInput.setAttribute("aria-activedescendant", current.id);
    current.scrollIntoView({ block: "nearest" });
  }

  symbolInput.addEventListener("focus", () => symbolInput.select());
  symbolInput.addEventListener("input", () => {
    clearTimeout(local.searchTimer);
    closeSuggestions();
    searchInfo();
    const request = ++local.searchRequest;
    const query = symbolInput.value.trim();
    if (query) local.searchTimer = setTimeout(() => search(query, request), 180);
  });
  symbolInput.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeSuggestions();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      if (!suggestions.hidden) {
        event.preventDefault();
        const start = local.suggestionIndex < 0 && event.key === "ArrowUp" ? 0 : local.suggestionIndex;
        moveSuggestion(start + (event.key === "ArrowDown" ? 1 : -1));
      }
    } else if (event.key === "Enter") {
      event.preventDefault();
      clearTimeout(local.searchTimer);
      const query = symbolInput.value.trim();
      const chosen = !suggestions.hidden && local.suggestions[Math.max(0, local.suggestionIndex)];
      if (chosen) selectMatched(chosen.symbol);
      else if (/^\d{6}\.(?:SH|SZ|BJ)$/i.test(query)) selectMatched(query.toUpperCase());
      else search(query, ++local.searchRequest, true);
    }
  });
  suggestions.addEventListener("pointerdown", (event) => event.preventDefault());
  suggestions.addEventListener("click", (event) => {
    const item = event.target.closest("[data-suggestion]");
    if (item) selectMatched(local.suggestions[Number(item.dataset.suggestion)].symbol);
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".stock-chart-search")) closeSuggestions();
  });
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (local.loading || !form.reportValidity()) return;
    if (endInput.value < startInput.value) {
      feedback("结束日期必须在开始日期当天或之后。");
      endInput.focus();
      return;
    }
    loadView(true);
  });
  overlay.addEventListener("change", () => {
    if (local.view) {
      renderCoverage();
      renderEvents();
    }
    stateToURL();
  });
  page.addEventListener("click", (event) => {
    const group = event.target.closest("[data-stock-group]");
    if (group) {
      const entry = local.groups.find((item) => item.key === group.dataset.stockGroup);
      if (entry) chooseEvent(entry.events[0].key, true);
    }
    const row = event.target.closest("[data-stock-event]");
    if (row) chooseEvent(row.dataset.stockEvent, !ledger.contains(row));
    const filter = event.target.closest("[data-stock-stage]");
    if (filter) {
      local.filter = filter.dataset.stockStage;
      renderLedger();
    }
  });
  $("#stock-chart-fit").addEventListener("click", () => local.chart?.fit());
  $("#stock-research-export").addEventListener("click", () => {
    if (!local.view) return;
    const payload = { ...local.view, exported_at: new Date().toISOString(), export_scope: "actual_bars_derived_research_frozen_scenarios_and_ledger" };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `khquant-${local.symbol}-${dateText(local.view.context?.end) || "chart"}-evidence.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  $("#refresh").addEventListener("click", () => {
    if (!openPage()) return;
    refreshOverview({ selectDefault: false });
    chooseSymbol(local.symbol, Object.fromEntries(urlHash()));
  });

  function onViewChange() {
    if (!openPage() || requestedPage() === "workspace") {
      ++local.viewRequest;
      ++local.searchRequest;
      return;
    }
    const query = urlHash();
    const chosen = query.get("symbol") || defaultSymbol;
    chooseSymbol(chosen, Object.fromEntries(query));
  }
  window.addEventListener("hashchange", onViewChange);
  onViewChange();
})();
