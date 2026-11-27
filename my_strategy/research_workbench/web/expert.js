"use strict";

(() => {
  const page = $("#page-expert");
  const form = $("#expert-form");
  const button = $("#expert-run");
  const previewButton = $("#expert-preview");
  const setup = $("#expert-setup");
  const output = $("#expert-result");
  const status = $("#expert-status");
  const states = {
    loading_inputs: "核对冻结输入", sector_artifacts: "封存板块与行业证据", historical_replay: "单独历史复核",
    oos_1: "KNN 60%＋多模态 40% · 窗口 1", oos_2: "KNN 60%＋多模态 40% · 窗口 2", oos_3: "KNN 60%＋多模态 40% · 窗口 3",
  };
  const labels = { historical_quantiles: "历史分位数", supertrend_atr: "SuperTrend＋ATR", domestic_fixed: "原参数国内 GPU",
    domestic_gpu: "国内量价 GPU", sector_gpu: "板块共振 GPU",
    overseas_gpu: "海外事件专家 GPU", fusion: "多模态融合", online_knn_volume_trend: "纯在线 KNN",
    knn60_multimodal40: "KNN 60%＋多模态 40%" };
  const actions = { BUY: "买入", ADD: "加仓", HOLD: "持有", REDUCE: "减仓", SELL: "卖出", OBSERVE: "观望" };
  let overview, job, displayed, timer, currentModel;
  let loading = false;
  let configured = false;
  let pending = false;
  let validated = "";
  let revision = 0;

  const field = (name) => form.elements.namedItem(name);

  function inputs() {
    const data = new FormData(form);
    const result = Object.fromEntries(["start", "end", "pool", "commission_bps", "knn_device"].map((name) => [name, data.get(name)]));
    if (data.get("sector_view") === "true") result.sector_view = true;
    result.sector_provider = "ths_fuyao";
    for (const name of ["seed", "rebalance_days", "top_n", "knn_query_chunk"]) result[name] = Number(data.get(name));
    for (const name of ["max_weight", "max_industry_weight"]) result[name] = Number(data.get(name)) / 100;
    if (result.pool === "board") {
      result.boards = data.getAll("board");
      result.stock_cap = Number(data.get("stock_cap"));
    } else if (result.pool === "manual") {
      result.symbols = [...new Set(String(data.get("symbols")).split(/[\s,，、;；]+/).filter(Boolean).map((item) => item.toUpperCase()))];
    }
    return result;
  }

  function showFields() {
    const pool = field("pool").value;
    field("start").readOnly = field("end").readOnly = pool === "full";
    const board = $("#expert-board");
    board.hidden = board.disabled = pool !== "board";
    field("stock_cap").required = pool === "board";
    $("#expert-manual").hidden = field("symbols").disabled = pool !== "manual";
    field("symbols").required = pool === "manual";
    const fee = Number(field("commission_bps").value);
    const base = overview?.study_defaults?.base_cost;
    if (base) {
      $("#expert-fee-hint").textContent = `常规 ${(fee / 100).toLocaleString("zh-CN", { maximumFractionDigits: 8 })}% · 压力 ${(fee / 50).toLocaleString("zh-CN", { maximumFractionDigits: 8 })}% · 最低佣金 ${base.min_commission}／${base.min_commission * 2} 元`;
    }
  }

  function initialize(settings) {
    for (const name of ["start", "end", "pool", "commission_bps", "seed", "stock_cap", "rebalance_days", "top_n", "knn_device", "knn_query_chunk"]) {
      field(name).value = settings[name];
    }
    field("sector_view").checked = settings.sector_view === true;
    for (const name of ["max_weight", "max_industry_weight"]) field(name).value = settings[name] * 100;
    configured = true;
    showFields();
  }

  function controls() {
    button.disabled = !configured || pending || validated !== JSON.stringify(inputs()) || ["queued", "running"].includes(job?.status);
    previewButton.disabled = !configured || pending;
  }

  const dateOnly = (value) => String(value || "").slice(0, 10);

  function renderPreview(preflight = null) {
    if (!preflight) {
      setup.replaceChildren();
      return;
    }
    const windows = arr(preflight.windows).map((win) => [h(win.id), h(`${dateOnly(win.evaluation_start)} 至 ${dateOnly(win.evaluation_end)}`),
      count(win.evaluation_sessions), count(win.maturity?.train_sessions), h(win.blocked_reason || "")]);
    const tableHtml = windows.length ? scrollTable([["窗口"], ["评估日期"], ["交易日", "numeric"], ["成熟训练日", "numeric"], ["阻塞"]], windows) : "";
    const notes = [...arr(preflight.blockers), ...Object.entries(preflight.rejections || {}).map(([stock, note]) => `${stock} · ${note}`),
      ...arr(preflight.warnings)];
    const symbols = arr(preflight.selected_symbols);
    const bySymbol = preflight.coverage?.by_symbol || {};
    const basis = preflight.coverage?.basis === "frozen_signal_eligible_days" ? "合格信号日" : "核验复权日（预热前）";
    const selected = symbols.length ? `<details><summary>入选名单 ${count(symbols.length)} 只</summary>${scrollTable(
      [["股票代码"], [basis, "numeric"]], symbols.slice(0, 60).map((stock) => [h(stock), count(bySymbol[stock])]),
    )}${symbols.length > 60 ? "<p>预览显示前 60 只，完整名单可在运行报告核对。</p>" : ""}</details>` : "";
    setup.innerHTML = `<dl class="coverage">${metricItems([["预检结果", {ready: "可提交", partial: "部分覆盖", blocked: "资料不足"}[preflight.status] || "—"],
      ["股票池口径", preflight.scope_label || "按研究配置"], ["实际交易日期", `${preflight.first_trade_session} 至 ${preflight.last_trade_session}`], ["样本数量", `${count(preflight.selected_stocks)} 只`],
      [preflight.membership ? "股票池按日更新" : "名单锚定日", preflight.membership ? "上市至退市前" : preflight.anchor_session]])}</dl>${tableHtml}${selected}` +
      (notes.length ? `<ul class="latent-blockers">${notes.map((note) => `<li>${h(note)}</li>`).join("")}</ul>` : "");
  }

  function foldTable(summary) {
    const rows = [];
    const arm = summary.final_portfolio?.arm || "fusion";
    const hybrid = arm === "knn60_multimodal40";
    for (const fold of [...arr(summary.folds), ...[summary.historical_review].filter(Boolean)]) {
      const name = fold.period === "historical_replay" ? "历史复核" : fold.fold || fold.period;
      for (const selected of hybrid ? [arm, "online_knn_volume_trend", "fusion"] : [arm]) {
        const model = fold.arms?.[selected];
        for (const code of ["normal", "double"]) {
          const metrics = model?.execution?.[code];
          if (metrics) rows.push([h(name), ...(hybrid ? [h(labels[selected] || selected)] : []),
            code === "normal" ? "常规" : "双倍佣金／滑点", pct(metrics.total_return),
            pct(metrics.max_drawdown), pct(metrics.turnover), pct(metrics.reject_rate), decimals(model.prediction?.pinball, 6)]);
        }
      }
    }
    return rows.length ? scrollTable([["滚动窗口"], ...(hybrid ? [["策略"]] : []), ["成本"], ["净收益", "numeric"], ["最大回撤", "numeric"],
      ["换手", "numeric"], ["拒单率", "numeric"], ["Pinball loss（混合分不适用）", "numeric"]], rows) : "";
  }

  function scoreHealthPanel(evidence) {
    const folds = arr(evidence?.folds);
    if (!folds.length) return "";
    if (evidence.mode === "knn60_multimodal40") {
      const rows = folds.map((fold) => [
        h(fold.period), decimals(fold.raw_q50?.median, 6), decimals(fold.calibrated_q50?.median, 6),
        decimals(fold.multimodal_q50?.median, 6), decimals(fold.hybrid_score?.median, 6),
        count(fold.coverage?.full_60_40_rows), count(fold.coverage?.knn_fallback_rows), count(fold.coverage?.invalid_knn_rows),
        h(`${fold.knn_backend?.actual_device || "—"} · ${fold.knn_backend?.backend || "—"}`),
      ]);
      const notes = folds.flatMap((fold) => arr(fold.concerns).map((note) => `<li>${h(fold.period)}：${h(note)}</li>`)).join("");
      return panel("混合分数、实际权重与 GPU 诊断", `<div class="panel-inset"><p>${h(evidence.summary)}</p>
        <p>原始与校准 q50 分属各模型，量纲不直接相加；混合列仅用于同日横截面选股。多模态无效股日明确按 KNN 100% 兜底。</p>
        ${notes ? `<ul class="latent-blockers">${notes}</ul>` : ""}</div>` + scrollTable([
        ["窗口"], ["KNN 原始 q50 中位数", "numeric"], ["KNN 校准 q50 中位数", "numeric"],
        ["多模态 q50 中位数", "numeric"], ["混合分中位数", "numeric"], ["完整 60/40 股日", "numeric"],
        ["KNN 兜底股日", "numeric"], ["KNN 无效股日", "numeric"], ["KNN 实际后端"],
      ], rows));
    }
    const threshold = folds[0]?.entry?.min_score;
    const ranked = folds[0]?.entry?.selection_mode === "cross_sectional_rank";
    const rows = folds.map((fold) => {
      const fitted = arr(fold.contributing_models).filter((item) => Number.isFinite(num(item.q50_tree_count)));
      const raw = fold.raw_q50;
      const calibrated = fold.calibrated_q50 || {};
      const model = ranked ? "纯在线 KNN" : fitted.map((item) => `${labels[item.arm] || item.arm} ${pct(item.weight)}／${item.q50_tree_count} 树`).join("；") || "—";
      return [
        h(fold.period), h(model),
        decimals(raw?.median, 6), decimals(calibrated.median, 6), count(raw?.unique), count(calibrated.unique),
        count(fold.raw_qualified_rows), count(fold.qualified_rows), h(arr(fold.concerns).join("；") || "未发现塌缩"),
      ];
    });
    const notes = folds.flatMap((fold) => arr(fold.concerns).map((note) => `<li>${h(fold.period)}：${h(note)}</li>`)).join("");
    const explanation = ranked
      ? "纯在线 KNN 按同一信号日的校准 q50 做全市场横截面排名，不设 q50 大于零的绝对门槛；原始分数仅用于历史诊断。"
      : `入场分数是校准后的 q50，必须严格大于 ${pct(threshold)}。原始分数对照只用于历史诊断，不能据此反调门槛或晋升模型。`;
    return panel("分数与空仓诊断", `<div class="panel-inset"><p>${explanation}</p>
      ${notes ? `<ul class="latent-blockers">${notes}</ul>` : ""}</div>` + scrollTable([
        ["窗口"], [ranked ? "在线 KNN 模型" : "参与融合的学习模型／权重／q50 树数"], ["原始 q50 中位数", "numeric"], ["校准 q50 中位数", "numeric"],
        ["原始不同值", "numeric"], ["校准不同值", "numeric"], ["原始达标股日", "numeric"], ["校准达标股日", "numeric"], ["诊断"],
      ], rows));
  }

  function decisionActions(result) {
    const recent = arr(result.latest_actions);
    if (!recent.length) return '<div class="panel-inset">该窗口暂无可展示的最终委托。</div>';
    const rows = recent.map((item) => {
      const side = String(item.action || item.side || "").toUpperCase();
      const color = ["BUY", "ADD"].includes(side) ? "expert-action-buy" : ["SELL", "REDUCE"].includes(side) ? "expert-action-sell" : "";
      return [h(item.symbol), h(item.signal_date), h(item.execution_date), `<strong class="${color}">${h(actions[side] || side)}</strong>`,
        pct(item.target_weight), h(item.reason)];
    });
    return scrollTable([["股票"], ["信号日"], ["模拟执行日"], ["最终动作"], ["目标权重", "numeric"], ["依据"]], rows);
  }

  function diagnostics(report) {
    const names = Object.entries(report.experts || {}).map(([id, value]) => [
      h(labels[id] || id), h({ active: "已参与", abstain: "弃权" }[value?.status] || value?.status || value?.state),
      h(value?.reason || value?.training_device || value?.device),
    ]);
    const coverage = report.input_coverage || {};
    const windows = [...arr(report.folds), ...[report.historical_review].filter(Boolean)];
    const knn = report.training_engine === "online_knn_volume_trend";
    const hybrid = report.training_engine === "knn60_multimodal40";
    const parameters = windows.flatMap((fold) => Object.entries(fold.models?.arms || {}).map(([id, entry]) => [
      h(fold.period), h(labels[id] || id),
      ...(knn ? [h(arr(entry.features).join("、")), h(entry.training_device)] : [
        h(entry.winner?.depth), h(entry.winner?.learning_rate), h(arr(entry.features).join("、")), h(arr(entry.candidates).length),
      ]),
    ]));
    const contrasts = windows.flatMap((fold) => Object.entries(fold.contrasts || {}).map(([id, value]) => {
      const name = { fusion_vs_historical: "融合 − 历史分位", fusion_vs_domestic: "融合 − 国内量价" }[id] || id;
      return [h(fold.period), h(name), pct(value.normal?.net_return_difference), pct(value.double?.net_return_difference)];
    }));
    const content = metrics([["行情行数", count(coverage.rows)], ["合格特征行", count(coverage.signal_eligible_rows)],
      ["成熟标签行", count(coverage.eligible_labeled_rows)], ["特征宽度口径", coverage.breadth_scope]]) +
      (hybrid ? `<dl class="metrics">${metricItems([["固定决策权重", "KNN 60%＋多模态 40%"],
        ["KNN 特征", arr(report.knn?.features || coverage.model_features).join("、")],
        ["KNN 请求设备", report.knn?.requested_device || report.experts?.online_knn_volume_trend?.requested_device]])}</dl>` : "") +
      (names.length ? scrollTable([["专家"], ["状态"], ["设备或原因"]], names) : "") +
      (parameters.length ? `<h3>${knn ? "分折 KNN 参数" : "分折调参"}</h3>${scrollTable(knn ? [["窗口"], ["模型"], ["特征"], ["训练设备"]] : [["窗口"], ["模型"], ["树深"], ["学习率"], ["特征"], ["候选数"]], parameters)}` : "") +
      (contrasts.length ? `<h3>专家净收益差</h3>${scrollTable([["窗口"], ["对照"], ["常规成本"], ["双倍成本"]], contrasts)}` : "");
    return `<details class="panel panel-inset"><summary>${knn ? "因子覆盖、KNN 参数与运行诊断" : "因子覆盖、模型调参与专家对照"}</summary>${content}</details>`;
  }

  function explanation(runId) {
    return `<details class="panel panel-inset"><summary>模型依据与 SHAP</summary>
      <p>按窗口解释各模型的原始中位数预测，并显示融合权重。</p>
      <button class="button button-quiet" type="button" data-expert-explain="${h(runId)}">计算归因</button>
      <div class="expert-explain-output" role="status"></div></details>`;
  }

  function championPanel(champion, runId) {
    if (!champion) return "";
    const rows = arr(champion.ranking).map((item) => [h(labels[item.arm] || item.arm), item.historical_gate ? "通过" : "未通过",
      item.risk_pass ? "通过" : "未通过", pct(item.minimum_double_cost_ci_lower), decimals(item.mean_pinball, 6)]);
    const label = labels[champion.selected_arm] || champion.selected_arm;
    return panel("唯一候选选择", `<div class="panel-inset"><p>当前选择 <strong>${h(label)}</strong> · ${champion.state === "historical_candidate" ? "等待前瞻" : "研究候选"}</p>
      <button class="button button-quiet" type="button" data-expert-forward="${h(runId)}">查看前瞻状态</button>
      <div class="expert-forward-output" role="status"></div></div>` +
      scrollTable([["候选"], ["历史验收"], ["风险验收"], ["双费收益差区间下界", "numeric"], ["平均 Pinball loss", "numeric"]], rows));
  }

  function render(report, runId) {
    const result = report.final_portfolio || {};
    const decision = report.decision || {};
    const aggregate = result.metrics || {};
    const normal = aggregate.normal || {};
    const double = aggregate.double || {};
    const knn = result.arm === "online_knn_volume_trend";
    const hybrid = result.arm === "knn60_multimodal40";
    const reasons = arr(decision.reasons).map((reason) => `<li>${h(reason)}</li>`).join("");
    const base = `/api/expert/runs/${encodeURIComponent(runId)}/downloads/`;
    const arm = ["fusion", "domestic_gpu", "online_knn_volume_trend", "knn60_multimodal40"].includes(result.arm) ? result.arm : "fusion";
    const ledger = typeof result.fold === "string" && /^(?:oos_[1-3]|historical_replay)$/.test(result.fold)
      ? `<a class="table-button" href="${base}${result.fold}--${arm}_normal_trades.parquet" download>下载最新折成交账本</a>` : "";
    const manifest = /^\d[0-9a-z_.-]{1,119}$/i.test(runId)
      ? `<a class="table-button" href="#expert?run=${encodeURIComponent(runId)}">此次研究链接</a>` : "";
    const symbol = arr(report.input_coverage?.selected_symbols)[0] || arr(report.model_spec?.symbols)[0] || "002650.SZ";
    const linked = new URLSearchParams({ source_run: runId });
    if (report.unified?.model_version) linked.set("model_version", report.unified.model_version);
    const stockQuery = new URLSearchParams(linked);
    stockQuery.set("symbol", symbol);
    const preview = `<a class="table-button" href="#stock-chart?${stockQuery}">查看此${hybrid ? "混合" : knn ? " KNN" : "统一"}候选的个股证据</a>`;
    const backtest = `<a class="button" href="#backtest?${linked}">回测此${hybrid ? "混合" : knn ? " KNN" : "统一"}候选</a>`;
    const spec = report.model_spec;
    const costs = report.execution_costs;
    const poolName = { full: "沪深主板＋科创板全池", frozen_120: "原 120 股", manual: "手选股票", board: "板块抽样" };
    const firstDay = dateOnly(report.folds?.[0]?.evaluation_start);
    const lastDay = dateOnly(report.folds?.at(-1)?.evaluation_end);
    const used = spec || costs ? `<dl class="metrics">${metricItems([
      ["样本外交易日", firstDay && lastDay ? `${firstDay} 至 ${lastDay}` : "—"], ["股票池", poolName[spec?.pool] || "—"],
      ["常规佣金（万分之）", report.commission_bps], ["压力佣金（万分之）", Number.isFinite(num(costs?.double?.commission)) ? decimals(costs.double.commission * 10000, 4) : "—"],
      ["预测计算", report.reused_predictions_from ? "复用冻结预测" : hybrid ? `KNN ${report.folds?.[0]?.hybrid?.knn_backend?.actual_device || "—"}＋多模态 GPU` : knn ? "CPU 在线 KNN 重训" : "GPU 训练"]])}</dl>` : "";
    const body = `<div class="panel-inset">
      <dl class="metrics">${metricItems([["模型类型", hybrid ? "KNN 60%＋多模态 40%" : knn ? "纯在线 KNN" : report.unified?.model_version || "多模态本运行候选"],
        ["封存状态", hybrid ? "双方原生分位数、实际权重与混合预测已封存" : knn ? "逐折 KNN 参数与预测已封存" : report.unified ? "统一模型已封存" : "旧研究无最终拟合包"],
        ["常规成本净收益", pct(normal.total_return)], ["双倍成本净收益", pct(double.total_return)],
        ["前瞻验收", decision.qualified_forward_results ? "通过" : "未完成"], ["运行编号", runId]])}</dl>
      ${used}
      ${reasons ? `<ul class="latent-blockers">${reasons}</ul>` : ""}
      <div class="expert-downloads"><a class="table-button" href="${base}report.json" download>下载报告</a>
      <a class="table-button" href="${base}costs.csv" download>下载分折成本表</a>
      <a class="table-button" href="${base}parameters.json" download>下载${hybrid ? "混合与原生参数" : knn ? " KNN 参数" : "调优记录"}</a>${ledger}${manifest}${preview}${backtest}</div></div>`;
    const title = hybrid ? "KNN 60%＋多模态 40% 研究结果" : knn ? "纯在线 KNN 研究结果" : decision.qualified_forward_results ? "多模态专家验收结果" : "多模态专家候选结果";
    output.innerHTML = panel(title, body) + (knn ? "" : championPanel(report.single_champion, runId)) + scoreHealthPanel(report.score_health) + panel("最新历史模拟委托", decisionActions(result)) +
      (report.folds?.length ? panel("分折验收", foldTable(report)) : "") + diagnostics(report) + (knn ? "" : explanation(runId));
  }

  output.addEventListener("click", async (event) => {
    const forward = event.target.closest("[data-expert-forward]");
    if (forward) {
      const result = forward.parentElement.querySelector(".expert-forward-output");
      forward.disabled = true;
      result.textContent = "正在读取前瞻状态…";
      try {
        const progress = await api(`/api/expert/runs/${encodeURIComponent(forward.dataset.expertForward)}/forward-status`);
        result.innerHTML = `<p>已核验记录 ${count(progress.valid_captures)}／${count(progress.required_captures)} 个交易日</p>` +
          arr(progress.blockers).map((reason) => `<p>${h(reason)}</p>`).join("");
      } catch (error) {
        result.textContent = error.message;
      } finally {
        forward.disabled = false;
      }
      return;
    }
    const target = event.target.closest("[data-expert-explain]");
    if (!target || !/^[a-z0-9][a-z0-9_.-]{0,119}$/i.test(target.dataset.expertExplain)) return;
    const result = target.parentElement.querySelector(".expert-explain-output");
    result.textContent = "正在计算归因…";
    target.disabled = true;
    try {
      const evidence = await api(`/api/expert/runs/${encodeURIComponent(target.dataset.expertExplain)}/explain`);
      const rows = arr(evidence.folds).flatMap((fold) => Object.entries(fold.experts || {}).map(([arm, expert]) => [
        h(fold.period), h(labels[arm] || arm), pct(fold.fusion?.weights?.[arm] || 0), count(expert.sample_rows || fold.sample_rows),
        h(arr(expert.top5).map((item) => item.feature).join("、") || expert.reason),
      ]));
      if (rows.length) result.innerHTML = scrollTable([["窗口"], ["模型"], ["融合权重", "numeric"], ["样本", "numeric"], ["主要贡献特征"]], rows);
      else result.textContent = evidence.reason || "当前模型缺少可解释样本。";
    } catch (error) {
      result.textContent = error.message;
    } finally {
      target.disabled = false;
    }
  });

  async function refreshStatus() {
    clearTimeout(timer);
    if (!job) return;
    const activeJob = job;
    try {
      const current = await api(activeJob.status_url);
      if (job !== activeJob) return;
      job = current;
      const running = ["running", "queued"].includes(current.status);
      controls();
      const stage = current.reused_predictions_from ? "复用冻结预测并重算模拟账本" : states[current.phase] || "正在执行滚动验收";
      status.textContent = running ? `${stage} · ${current.run_id}` : current.message;
      if (current.status === "completed" && displayed !== current.run_id) {
        const [result, model] = await Promise.all([api(current.result_url), readCurrentModel({ refresh: true }).catch(() => currentModel)]);
        if (job !== current) return;
        currentModel = model;
        renderCurrentModel($("#expert-current-model"), model);
        render(result, current.run_id);
        displayed = current.run_id;
        status.textContent = model.status === "production_fallback" && model.arm === "knn60_multimodal40"
          && model.run_id === current.run_id ? "混合模型已完成验证，并因同口径收益严格更高切换为当前生产回退。"
          : model.status === "production_fallback" && model.arm === "online_knn_volume_trend"
          && model.run_id === current.run_id ? "纯在线 KNN 已完成验证并切换为当前生产回退。"
          : model.status === "production_fallback" && model.production_source === "research_candidate"
          && model.run_id === current.run_id ? "测试收益高于前生产模型，已自动替换生产回退。" : "训练完成；当前生产回退保持不变。";
      }
      if (running && !page.hidden) timer = setTimeout(refreshStatus, 1500);
    } catch (error) {
      status.textContent = error.message;
      controls();
    }
  }

  async function show() {
    clearTimeout(timer);
    if (page.hidden || loading) return;
    loading = true;
    try {
      const [settings, model] = await Promise.all([api("/api/expert/overview"), readCurrentModel({ refresh: true })]);
      overview = settings;
      currentModel = model;
      renderCurrentModel($("#expert-current-model"), model);
      if (!configured) {
        initialize(overview.study_defaults);
        renderPreview();
      }
      const requested = new URLSearchParams(location.hash.split("?")[1]).get("run");
      job = overview.latest_job;
      controls();
      const currentRun = model.run_id;
      const selected = /^[a-z0-9][a-z0-9_.-]{0,127}$/i.test(requested || "") ? requested : currentRun;
      if (selected && displayed !== selected && !(selected === job?.run_id && ["queued", "running"].includes(job.status))) {
        const result = await api(`/api/expert/runs/${encodeURIComponent(selected)}`);
        render(result, selected);
        displayed = selected;
      }
      if (job && (requested === job.run_id || ["queued", "running"].includes(job.status))) {
        await refreshStatus();
      } else {
        status.textContent = selected ? (selected === currentRun ? "已载入当前模型报告。" : "已载入此运行的研究报告。")
          : model.status === "production_fallback" ? `当前使用已封存的生产回退：${model.label || model.arm}。` : "等待启动 KNN 60%＋多模态 40% 研究。";
      }
    } catch (error) {
      status.textContent = error.message;
    } finally {
      loading = false;
    }
  }

  form.addEventListener("input", () => {
    revision += 1;
    validated = "";
    showFields();
    renderPreview();
    controls();
  });
  field("pool").addEventListener("change", () => {
    const defaults = overview?.study_defaults;
    if (!defaults) return;
    const end = field("end");
    const pool = field("pool").value;
    if (pool === "full") {
      field("start").value = defaults.start;
      end.value = defaults.end;
    } else if (pool === "frozen_120" && end.value === defaults.end) end.value = defaults.frozen_end;
  });

  previewButton.addEventListener("click", async () => {
    if (!form.reportValidity()) return;
    const request = revision;
    pending = true;
    controls();
    status.textContent = "正在核对交易日与股票覆盖…";
    try {
      const result = await api("/api/expert/preflight", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(inputs()),
      });
      if (request === revision) {
        renderPreview(result);
        validated = result.status === "blocked" ? "" : JSON.stringify(inputs());
      }
      status.textContent = request !== revision ? "配置已变更，可重新预览。" : result.status === "blocked" ? "预检发现资料不足。" : "研究口径已核对，可启动 KNN 60%＋多模态 40% 训练。";
    } catch (error) {
      status.textContent = error.message;
    } finally {
      pending = false;
      controls();
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (button.disabled || !form.reportValidity()) return;
    pending = true;
    controls();
    status.textContent = "正在提交 KNN 60%＋多模态 40% 研究…";
    clearTimeout(timer);
    try {
      job = await api("/api/expert/runs", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(inputs()),
      });
      if (displayed !== job.run_id) {
        output.replaceChildren();
        displayed = null;
      }
      history.replaceState(null, "", `#expert?run=${encodeURIComponent(job.run_id)}`);
      await refreshStatus();
    } catch (error) {
      status.textContent = error.message;
    } finally {
      pending = false;
      controls();
    }
  });
  window.addEventListener("hashchange", show);
  $("#refresh").addEventListener("click", show);
  show();
})();
