"use strict";

(() => {
  const page = $("#page-latent");
  const presets = {
    baseline_replay: "快速回放",
    domestic_tune: "国内滚动调优",
  };
  const profiles = { baseline: "国内基线", domestic: "国内滚动调优", overseas: "海外对照" };
  const periods = {
    oos_1: "滚动窗口 1", oos_2: "滚动窗口 2", oos_3: "滚动窗口 3",
    historical_replay: "历史复核", prospective_holdout: "前瞻留出",
  };
  const arms = {
    historical_quantiles: "历史经验分位数", supertrend_atr: "SuperTrend + ATR",
    catboost_base: "CatBoost 基础特征", catboost_supertrend_atr: "CatBoost + SuperTrend / ATR",
    domestic: "国内特征", overseas_raw: "国内 + 海外原始特征", overseas_pca: "国内 + 海外 + PCA",
    domestic_fixed: "固定仓位规则", domestic_tuned: "调优仓位规则",
  };
  const comparisons = {
    tuned_vs_fixed: "调优仓位 − 固定仓位", raw_vs_domestic: "海外原始特征 − 国内特征",
    pca_vs_raw: "海外 + PCA − 海外原始特征", catboost_base_vs_historical: "CatBoost 基础 − 历史分位数",
    catboost_extended_vs_historical: "CatBoost + SuperTrend / ATR − 历史分位数",
    supertrend_vs_historical: "SuperTrend + ATR − 历史分位数",
    catboost_extended_vs_base: "CatBoost + SuperTrend / ATR − CatBoost 基础",
  };
  const costs = { normal: "常规成本", double: "双倍佣金／滑点" };
  const paperStatuses = {
    awaiting_market_close: "等候收盘", verified_calendar_unavailable: "缺少核验日历",
    awaiting_same_day_inputs: "等候当日资料", collecting: "收集中", ready_for_scoring: "可以评估",
    missed_or_invalid_capture: "记录缺失或无效", blocked: "暂不可操作", unavailable: "不可用",
  };
  const history = $("#latent-history");
  const detail = $("#latent-detail");
  const bundle = $("#latent-bundle");
  const prepare = $("#latent-prepare");
  const storageKey = "khquant.latent.run_id";
  const local = {
    overview: null, detail: null, selection: "", period: "", capture: null, captureDate: "",
    timer: null, refreshing: false, detailRequest: 0, submitting: "", preparing: false,
    runFeedback: "", paperFeedback: "", overviewError: "", detailError: "", bundleKey: "", capturesKey: "",
  };
  try { local.selection = sessionStorage.getItem(storageKey) || ""; } catch { /* storage may be disabled */ }

  const active = (record) => activeStatuses.has(record?.status || record?.job?.status);
  const showActive = () => !page.hidden;
  const named = (names, value) => names[value] || text(value);
  const reportCoverage = (report) => report?.input_coverage?.domestic || report?.input_coverage || {};
  const runStatus = (record) => [named(periods, record.phase === "completed" ? "" : record.phase), record.message]
    .filter((part) => part && part !== "—").join(" · ");
  const safePath = (value) => typeof value === "string" && /^\/api\/latent\/(?:runs|paper)\//.test(value) ? value : "";
  const numeric = (value) => Number.isFinite(num(value));

  function notify(message = "", error = false) {
    const target = $("#latent-message");
    target.textContent = message;
    target.hidden = !message;
    target.classList.toggle("error", error);
  }

  function renderDates() {
    const overview = local.overview;
    const coverage = reportCoverage(local.detail?.report);
    const latest = overview?.market?.latest_date || state.overview?.data?.latest_date;
    const end = coverage.last_session || local.detail?.report?.run?.as_of_date || overview?.historical_end;
    $("#latent-dates").innerHTML = metricItems([
      ["原始行情最新交易日", latest], ["所选实验行情截止", end],
      ["前瞻留出开始", overview?.paper?.state?.prospective_start],
    ]);
  }

  function renderHistory() {
    const runs = arr(local.overview?.runs);
    const buttons = [...page.querySelectorAll("[data-latent-preset]")];
    for (const button of buttons) {
      const id = button.dataset.latentPreset;
      const preset = arr(local.overview?.presets).find((entry) => entry.id === id);
      button.disabled = !preset?.enabled || !!preset.busy || !!local.submitting;
      button.textContent = local.submitting === id ? "正在提交…" : presets[id];
      button.title = preset?.blocker || (preset?.busy ? "正在运行实验" : "");
    }
    const feedback = $("#latent-run-feedback");
    feedback.textContent = local.runFeedback;
    feedback.hidden = !local.runFeedback;
    if (!runs.length) {
      history.innerHTML = local.overviewError ? empty("无法读取实验运行", local.overviewError)
        : local.overview ? empty("暂无 Latent-Quant 实验", "可以从快速回放开始。")
          : empty("正在读取实验运行", "历史运行和后台进度会显示在这里。");
      history.dataset.rendered = "";
      return;
    }
    const rendered = table([["运行"], ["状态"], ["更新时间", "nowrap"]], runs.map((run) => {
      const updated = shortTime(run.updated_at);
      const compactTime = /^\d{4}-\d{2}-\d{2} /.test(updated) ? updated.slice(5, 16) : updated;
      return [
        `<button class="table-button latent-run-link" type="button" title="${h(run.run_id)}" data-latent-run="${h(run.run_id)}"${run.run_id === local.selection ? ' aria-current="true"' : ""}>
          <strong>${h(named(profiles, run.profile))}</strong><span class="code">${h(run.run_id)}</span></button>`,
        statusBadge(run.status), `<span class="latent-full-time">${h(updated)}</span>
          <span class="latent-mobile-time" title="${h(updated)}">${h(compactTime)}</span>`,
      ];
    }));
    if (history.dataset.rendered !== rendered) {
      const focused = history.contains(document.activeElement) ? document.activeElement.dataset.latentRun : "";
      history.innerHTML = rendered;
      history.dataset.rendered = rendered;
      if (focused) [...history.querySelectorAll("[data-latent-run]")].find((node) => node.dataset.latentRun === focused)?.focus({ preventScroll: true });
    }
  }

  function paperReasons(paper) {
    const current = paper.state || {};
    const reasons = [];
    if (current.calendar_needed) reasons.push(`已核验 ${count(current.confirmed_calendar_sessions)} 个留出交易日，日历已确认至 ${text(current.calendar_known_through)}。`);
    if (arr(current.missing_past_captures).length) reasons.push(`缺少历史日记录 ${current.missing_past_captures.join("、")}。`);
    if (Array.isArray(current.invalid_captures) && current.invalid_captures.length) {
      reasons.push(`存在无效日记录 ${current.invalid_captures.join("、")}。`);
    } else if (current.invalid_captures && Object.keys(current.invalid_captures).length) {
      reasons.push(`存在无效日记录 ${Object.keys(current.invalid_captures).join("、")}。`);
    }
    if (paper.prepare_blocker) reasons.push(named(paperStatuses, paper.prepare_blocker));
    if (paper.input_status?.ready === false && paper.input_status.message && !reasons.includes(paper.input_status.message)) {
      reasons.push(paper.input_status.message);
    }
    return reasons;
  }

  function renderPaper() {
    const paper = local.overview?.paper || {};
    const current = paper.state || {};
    const badge = $("#latent-paper-status");
    const label = current.status ? named(paperStatuses, current.status) : local.overview ? "暂无状态" : "读取中";
    badge.textContent = label;
    badge.className = `pill ${current.ready_for_scoring ? "" : current.status === "missed_or_invalid_capture" ? "fail" : "neutral"}`;
    const completed = current.valid_daily_captures;
    const target = current.target_sessions;
    $("#latent-progress-count").textContent = numeric(completed) && numeric(target) ? `${count(completed)} / ${count(target)}` : "—";
    const progress = $("#latent-progress");
    progress.max = numeric(target) && num(target) > 0 ? num(target) : 60;
    progress.value = numeric(completed) ? num(completed) : 0;
    const readyNote = current.ready_for_scoring ? "已收齐有效纸面记录。" : "";
    const captureNote = current.capture_cutoff ? `每日有效封存须在上海时间 ${current.capture_cutoff} 前完成。` : "";
    $("#latent-paper-note").textContent = readyNote || captureNote;
    const reasons = paperReasons(paper);
    const reasonList = $("#latent-paper-blockers");
    reasonList.hidden = !reasons.length;
    reasonList.innerHTML = reasons.map((reason) => `<li>${h(reason)}</li>`).join("");

    const items = arr(paper.bundles);
    const bundleKey = JSON.stringify(items);
    if (bundleKey !== local.bundleKey) {
      const old = bundle.value;
      bundle.replaceChildren(...(items.length ? items.map((entry) => option(entry.id, entry.label || entry.date || entry.id))
        : [option("", "暂无可用数据包")]));
      if (items.some((entry) => entry.id === old)) bundle.value = old;
      local.bundleKey = bundleKey;
    }
    const isPreparing = local.preparing || active(paper.job);
    bundle.disabled = !items.length || isPreparing;
    prepare.disabled = !bundle.value || !paper.prepare_available || isPreparing;
    prepare.textContent = isPreparing ? "准备中…" : "准备纸面记录";
    $("#latent-paper-feedback").textContent = paper.job?.message || local.paperFeedback || (!items.length ? "数据包通过核验后可在这里选择。" : "");
    const template = $("#latent-paper-template");
    const templateUrl = safePath(paper.template_url);
    template.hidden = !templateUrl;
    if (templateUrl) template.href = templateUrl;

    const captures = arr(paper.captures);
    const captureRegion = $("#latent-captures");
    captureRegion.hidden = !captures.length;
    const rendered = captures.length ? `<h3>已封存记录</h3><div class="latent-capture-list">${captures.map((entry) =>
      `<button type="button" class="table-button code" data-latent-capture="${h(entry.date)}"${entry.date === local.captureDate ? ' aria-current="true"' : ""}>${h(entry.date)}</button>`
    ).join("")}</div>${captureDetails(local.capture)}` : "";
    if (rendered !== local.capturesKey) {
      captureRegion.innerHTML = rendered;
      local.capturesKey = rendered;
    }
  }

  function captureDetails(capture) {
    if (!capture) return "";
    const manifest = capture.manifest || {};
    return `<div class="latent-capture-detail"><p>${h(capture.date)} ${h(manifest.captured_at || manifest.created_at || "已封存")}</p>${downloadList(arr(capture.downloads))}</div>`;
  }

  function predictionTable(fold) {
    return scrollTable([["对照组", "nowrap"], ["样本数", "numeric"], ["Pinball loss", "numeric"], ["80% 区间覆盖", "numeric"], ["区间宽度", "numeric"]],
      Object.entries(fold.arms || {}).map(([arm, values]) => {
        const prediction = values.prediction || {};
        return [h(named(arms, arm)), h(count(prediction.rows)), h(decimals(prediction.pinball, 5)), h(pct(prediction.coverage_80)), h(pct(prediction.width_80))];
      }));
  }

  function executionTable(fold) {
    return scrollTable([["对照组", "nowrap"], ["成本口径", "nowrap"], ["净收益", "numeric"], ["超额收益", "numeric"], ["最大回撤", "numeric"], ["累计换手", "numeric"], ["拒单率", "numeric"]],
      Object.entries(fold.arms || {}).flatMap(([arm, values]) => Object.entries(values.execution || {}).map(([cost, execution]) => [
        h(named(arms, arm)), h(named(costs, cost)), returnCell(execution.total_return), returnCell(execution.excess_return),
        h(pct(execution.max_drawdown)), h(pct(execution.turnover)), h(pct(execution.reject_rate)),
      ])));
  }

  function returnCell(value) {
    const direction = num(value) > 0 ? "positive" : num(value) < 0 ? "negative" : "";
    return `<span class="${direction}">${h(pct(value))}</span>`;
  }

  function contrastTable(contrasts) {
    return scrollTable([["配对对照", "nowrap"], ["成本口径", "nowrap"], ["净收益差", "numeric"], ["95% 区间下沿", "numeric"], ["95% 区间上沿", "numeric"]],
      Object.entries(contrasts || {}).flatMap(([key, item]) => Object.entries(item).map(([cost, evidence]) => [
        h(named(comparisons, key)), h(named(costs, cost)), returnCell(evidence.net_return_difference),
        h(pct(evidence.ci95_low)), h(pct(evidence.ci95_high)),
      ])));
  }

  function portfolioData(fold) {
    const selection = fold.portfolio;
    if (!selection) return "";
    const winner = selection.selected_strategy || selection.winner || {};
    const fixed = fold.arms?.domestic_fixed?.strategy || {};
    const clean = (value) => value.params || value.strategy || value;
    const data = [["固定规则", clean(fixed)], ["胜出规则", clean(winner)]].filter(([, params]) => Object.keys(params).length);
    const fields = {
      top_n: "目标持仓数", rebalance_days: "调仓间隔", max_weight: "单股上限", max_industry_weight: "行业上限",
      max_board_weight: "板块上限", cash_buffer: "现金缓冲", invest_fraction: "投资比例", score_threshold: "评分门槛",
      turnover_limit: "单日换手上限", min_amount: "最低成交额", max_participation: "成交参与率",
      max_position_weight: "单股上限", min_cash_weight: "现金缓冲",
    };
    const ratios = new Set(["max_weight", "max_industry_weight", "max_board_weight", "cash_buffer", "invest_fraction", "turnover_limit", "max_participation", "max_position_weight", "min_cash_weight"]);
    const rows = data.flatMap(([name, params]) => Object.entries(params)
      .filter(([, value]) => typeof value === "number" || typeof value === "string")
      .map(([key, value]) => [h(name), h(fields[key] || key), h(ratios.has(key) ? pct(value) : value)]));
    return rows.length ? scrollTable([["规则"], ["参数"], ["数值", "numeric"]], rows)
      + (selection.rule ? `<p class="note">${h(selection.rule)}</p>` : "") : "";
  }

  function modelData(fold) {
    const modelEvidence = fold.models || {};
    const items = modelEvidence.arms || (modelEvidence.winner ? { domestic: modelEvidence } : {});
    const parameters = Object.entries(items).flatMap(([arm, item]) => {
      const winner = item.winner || item;
      const params = winner.params || item.fixed_params || {};
      if (!Object.keys(params).length && !numeric(winner.tree_count)) return [];
      return [[
        h(named(arms, arm)), h(count(params.depth)), h(decimals(params.learning_rate, 3)), h(count(winner.tree_count || params.iterations)),
        h(decimals(winner.validation_pinball, 5)),
      ]];
    });
    return parameters.length ? scrollTable([["模型", "nowrap"], ["树深", "numeric"], ["学习率", "numeric"], ["树数量", "numeric"], ["验证 Pinball", "numeric"]], parameters) : "";
  }

  function reportData(report) {
    const coverage = reportCoverage(report);
    const overseas = report.input_coverage?.overseas;
    const statement = coverage.point_in_time?.verified === false ? "复权档案的历史发布时间和版本修订尚未核实。" : "";
    const evidence = [
      ["历史股票数", count(coverage.loaded_stocks)], ["共同信号", count(coverage.signal_eligible_rows)],
      ["有五日标签", count(coverage.eligible_labeled_rows)], ["最晚成熟信号日", coverage.eligible_label_last_signal],
      ["无确证可交易状态", count(coverage.missing_trade_status_stock_days)],
    ];
    if (overseas) evidence.push(["海外指数共同覆盖交易日", count(overseas.all_markets_fresh_sessions)]);
    return panel("实验数据", metrics(evidence) + (statement ? `<p class="note">${h(statement)}</p>` : ""));
  }

  function foldSelector(folds) {
    return `<label class="latent-fold-select" for="latent-fold-select">评估窗口<select id="latent-fold-select">${folds.map((fold) =>
      `<option value="${h(fold.period)}"${fold.period === local.period ? " selected" : ""}>${h(named(periods, fold.period))}</option>`).join("")}</select></label>`;
  }

  function foldBody(fold, report) {
    if (!fold) return empty("暂无评估窗口", "实验完成后会显示滚动对照结果。");
    const model = modelData(fold);
    const portfolio = portfolioData(fold);
    const notes = [
      `评估 ${shortTime(fold.evaluation_start).slice(0, 10)} 至 ${shortTime(fold.evaluation_end).slice(0, 10)}`,
      `${count(fold.evaluation_sessions)} 个交易日`,
    ];
    if (fold.period === "historical_replay") notes.push("历史复核窗口，已参与历史研究");
    const parameters = (model || portfolio) ? panel("本窗参数",
      (model ? `<h3 class="latent-subtitle">模型参数</h3>${model}` : "") + (portfolio ? `<h3 class="latent-subtitle">组合参数</h3>${portfolio}` : "")) : "";
    return `<p class="note latent-fold-note">${h(notes.join(" · "))}<span class="latent-swipe-hint">表格可横向滚动</span></p>` +
      `<h3 class="latent-subtitle">预测质量</h3>${predictionTable(fold)}` +
      `<h3 class="latent-subtitle">成本与成交</h3>${executionTable(fold)}` +
      `<details class="group-breakdown"><summary>本窗净收益差及区间</summary>${contrastTable(fold.contrasts)}</details>` +
      `<details class="group-breakdown"><summary>滚动窗口合并净收益差及区间</summary>${contrastTable(report.oos_block_bootstrap)}</details>` +
      (parameters ? `<div class="latent-parameters">${parameters}</div>` : "");
  }

  function downloadList(entries) {
    return `<ul class="latent-download-list">${entries.filter((entry) => safePath(entry.url)).map((entry) =>
      `<li><a href="${h(safePath(entry.url))}" download="${h(entry.filename || "")}">${h(entry.label || entry.filename || entry.id)}</a></li>`).join("")}</ul>`;
  }

  function downloadsPanel(downloads) {
    const entries = downloads.filter((entry) => safePath(entry.url));
    if (!entries.length) return "";
    const runs = entries.filter((entry) => !entry.fold);
    const folds = [...new Set(entries.map((entry) => entry.fold).filter(Boolean))];
    return panel("下载研究证据", (runs.length ? downloadList(runs) : "") + folds.map((period) => {
      const items = entries.filter((entry) => entry.fold === period);
      return `<details class="group-breakdown"><summary>${h(named(periods, period))} · ${count(items.length)} 个文件</summary>${downloadList(items)}</details>`;
    }).join(""));
  }

  function renderDetail() {
    const record = local.detail;
    if (local.detailError) {
      detail.innerHTML = empty("无法读取所选实验", local.detailError);
      detail.dataset.rendered = "";
      return;
    }
    if (!record) {
      detail.innerHTML = local.selection ? empty("正在读取所选实验", local.selection) : "";
      detail.dataset.rendered = "";
      return;
    }
    const report = record.report;
    const caption = `<div class="latent-detail-head"><div><h2>${h(named(profiles, record.profile))}</h2><p class="code">${h(record.run_id)}</p></div>${statusBadge(record.status)}</div>`;
    const message = runStatus(record);
    if (!report) {
      detail.innerHTML = `<section class="panel">${caption}${message ? `<p class="panel-inset">${h(message)}</p>` : ""}
        ${downloadsPanel(arr(record.downloads))}</section>`;
      detail.dataset.rendered = "";
      return;
    }
    const folds = arr(report.folds);
    const selected = folds.find((fold) => fold.period === local.period) || folds.find((fold) => fold.period === "oos_3") || folds[0];
    local.period = selected?.period || "";
    const result = report.decision?.prospective_holdout_available === false || report.decision?.qualified_forward_results === false
      ? "前瞻结果尚未评估" : "";
    const markup = `<section class="panel">${caption}${result ? `<p class="note">${h(result)}</p>` : ""}</section>` + reportData(report) +
      panel("逐窗对照", foldBody(selected, report), `<div class="latent-actions">${foldSelector(folds)}${local.period ? stockChartLink(record, report, local.period) : ""}</div>`) + downloadsPanel(arr(record.downloads));
    if (detail.dataset.rendered !== markup) {
      const refocus = document.activeElement?.id === "latent-fold-select";
      detail.innerHTML = markup;
      detail.dataset.rendered = markup;
      if (refocus) $("#latent-fold-select")?.focus({ preventScroll: true });
    }
  }

  function stockChartLink(record, report, period) {
    const stocks = arr(reportCoverage(report).stocks);
    const fromChart = urlParams().get("symbol");
    let recent = "";
    try { recent = sessionStorage.getItem("khquant.stock-chart.symbol") || ""; } catch { /* optional browser storage */ }
    const symbol = fromChart || (recent && (!stocks.length || stocks.includes(recent)) ? recent
      : stocks.includes("002650.SZ") ? "002650.SZ" : stocks[0] || "002650.SZ");
    const params = new URLSearchParams({ run_id: record.run_id, fold: period, symbol, cost: "normal" });
    return `<a class="button button-quiet stock-chart-link" href="#stock-chart?${h(params)}">查看个股图</a>`;
  }

  function urlParams() {
    return new URLSearchParams(location.hash.split("?")[1] || "");
  }

  function select(id) {
    if (id === local.selection && local.detail) return;
    local.selection = id;
    local.period = "";
    local.detail = null;
    local.detailError = "";
    local.detailRequest += 1;
    detail.dataset.rendered = "";
    try { sessionStorage.setItem(storageKey, id); } catch { /* storage may be disabled */ }
    renderDates();
    renderHistory();
    renderDetail();
  }

  async function loadDetail() {
    if (!local.selection) return;
    const request = ++local.detailRequest;
    const id = local.selection;
    try {
      const record = await api(`/api/latent/runs/${encodeURIComponent(id)}`);
      if (request !== local.detailRequest) return;
      local.detail = record;
      local.detailError = "";
    } catch (error) {
      if (request !== local.detailRequest) return;
      local.detailError = error.message;
      detail.dataset.rendered = "";
    }
    renderDetail();
    renderDates();
  }

  function schedulePoll() {
    clearTimeout(local.timer);
    if (!showActive()) return;
    const running = arr(local.overview?.runs).some(active) || active(local.detail) || active(local.overview?.paper?.job);
    if (running || local.overviewError) local.timer = setTimeout(refresh, running ? 2000 : 5000);
  }

  async function refresh() {
    if (!showActive() || local.refreshing) return;
    clearTimeout(local.timer);
    local.refreshing = true;
    try {
      local.overview = await api("/api/latent/overview");
      if (local.overviewError) notify();
      local.overviewError = "";
      const runs = arr(local.overview.runs);
      if (!runs.some((run) => run.run_id === local.selection) && !active(local.detail) && !local.submitting) {
        select(runs[0]?.run_id || "");
      }
      renderDates();
      renderPaper();
      renderHistory();
      await loadDetail();
    } catch (error) {
      local.overviewError = error.message;
      if (local.overview) notify(`无法刷新 Latent-Quant 状态。${error.message}`, true);
      renderHistory();
    } finally {
      local.refreshing = false;
      schedulePoll();
    }
  }

  async function launch(preset) {
    if (local.submitting) return;
    local.submitting = preset;
    local.runFeedback = `正在提交${presets[preset]}…`;
    renderHistory();
    notify();
    try {
      const result = await api("/api/latent/runs", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ preset }),
      });
      if (!result.run_id) throw new Error("服务未返回实验编号。");
      select(result.run_id);
      local.detail = { ...result, profile: preset === "domestic_tune" ? "domestic" : "baseline" };
      local.runFeedback = result.message || "实验已提交，后台运行状态会自动更新。";
      renderDetail();
    } catch (error) {
      local.runFeedback = "";
      notify(error.message, true);
    } finally {
      local.submitting = "";
      renderHistory();
      refresh();
    }
  }

  async function preparePaper() {
    if (local.preparing || !bundle.value) return;
    local.preparing = true;
    local.paperFeedback = "正在核验数据包…";
    renderPaper();
    notify();
    try {
      const result = await api("/api/latent/paper/prepare", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ bundle_id: bundle.value }),
      });
      local.paperFeedback = result.message || "纸面记录已提交，状态会自动更新。";
      if (result.job) local.overview.paper.job = result.job;
    } catch (error) {
      local.paperFeedback = "";
      notify(error.message, true);
    } finally {
      local.preparing = false;
      renderPaper();
      refresh();
    }
  }

  page.addEventListener("click", async (event) => {
    const request = event.target.closest("[data-latent-preset]");
    if (request) return launch(request.dataset.latentPreset);
    const run = event.target.closest("[data-latent-run]");
    if (run) {
      select(run.dataset.latentRun);
      await loadDetail();
      schedulePoll();
      return;
    }
    const capture = event.target.closest("[data-latent-capture]");
    if (capture) {
      const date = capture.dataset.latentCapture;
      local.captureDate = date === local.captureDate ? "" : date;
      local.capture = null;
      if (local.captureDate) {
        try {
          const result = await api(`/api/latent/paper/captures/${encodeURIComponent(date)}`);
          if (local.captureDate === date) local.capture = result;
        } catch (error) { notify(error.message, true); }
      }
      renderPaper();
    }
  });
  page.addEventListener("change", (event) => {
    if (event.target.id !== "latent-fold-select") return;
    local.period = event.target.value;
    renderDetail();
  });
  bundle.addEventListener("change", renderPaper);
  prepare.addEventListener("click", preparePaper);
  $("#refresh").addEventListener("click", () => {
    if (!showActive()) return;
    notify();
    refreshOverview({ selectDefault: false });
    refresh();
  });
  window.addEventListener("workbench:overview", () => { if (showActive()) renderDates(); });
  function onViewChange() {
    if (!showActive()) {
      clearTimeout(local.timer);
      return;
    }
    if (requestedPage() === "latent") {
      const query = urlParams();
      if (query.get("run_id")) select(query.get("run_id"));
      if (query.get("fold")) local.period = query.get("fold");
    }
    refresh();
  }
  window.addEventListener("hashchange", onViewChange);
  onViewChange();
})();
