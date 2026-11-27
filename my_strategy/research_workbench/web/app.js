"use strict";

const $ = (selector) => document.querySelector(selector);
const views = new Set(["expert", "backtest", "stock-chart", "data"]);
const retiredViews = { diagnostics: "expert", factors: "expert", models: "expert", tuning: "backtest", signals: "stock-chart", latent: "expert" };
const boardNames = { sh_main: "沪市主板", sz_main: "深市主板", star: "科创板", chinext: "创业板", bse: "北交所" };
const currentLabels = { approved: "已批准", research_only: "研究候选", production_fallback: "生产回退", unavailable: "待发布" };
const state = {
  overview: null,
  overviewError: "",
  refreshInFlight: null,
  currentRequest: null,
  currentPending: false,
  dataStatus: null,
  dataStatusError: "",
  dataStatusInFlight: null,
  dataPollInFlight: null,
  dataPollTimer: null,
  dataGeneration: 0,
  dataUpdatePending: false
};

const text = (value) => value === undefined || value === null || value === "" ? "—" : String(value);
const h = (value) => text(value).replace(/[&<>"']/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[character]);
const num = (value) => value === null || value === undefined || value === "" || typeof value === "boolean" ? NaN : Number(value);
const decimals = (value, digits = 3) => Number.isFinite(num(value)) ? num(value).toFixed(digits) : "—";
const pct = (value) => Number.isFinite(num(value)) ? `${(num(value) * 100).toFixed(2)}%` : "—";
const count = (value) => Number.isFinite(num(value)) ? num(value).toLocaleString("zh-CN") : "—";
const arr = (value) => Array.isArray(value) ? value : [];
const shortTime = (value) => text(value).replace("T", " ").slice(0, 19);
const memberCount = (value) => Array.isArray(value) ? count(value.length) : count(value);

function empty(title, message) {
  return `<div class="empty"><strong>${h(title)}</strong><p>${h(message)}</p></div>`;
}

function panel(title, body, extra = "") {
  return `<section class="panel"><div class="panel-head"><h2>${h(title)}</h2>${extra}</div>${body}</section>`;
}

function metricItems(items) {
  return items.map(([label, value]) => `<div><dt>${h(label)}</dt><dd>${h(value)}</dd></div>`).join("");
}

function metrics(items) {
  return `<dl class="metrics">${metricItems(items)}</dl>`;
}

function table(columns, rows) {
  if (!rows.length) return empty("暂无记录", "此范围内暂无记录。");
  const heading = columns.map(([label, className = ""]) => `<th scope="col" class="${className}">${h(label)}</th>`).join("");
  const body = rows.map((row) => `<tr>${row.map((value, index) => `<td class="${columns[index][1] || ""}">${value}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${heading}</tr></thead><tbody>${body}</tbody></table>`;
}

function scrollTable(columns, rows) {
  return `<div class="table-scroll">${table(columns, rows)}</div>`;
}

async function api(path, options) {
  const response = await fetch(path, { cache: "no-store", ...options });
  const result = await response.json().catch(() => null);
  if (!response.ok || !result) {
    const message = result?.message || result?.error || `请求失败（HTTP ${response.status}）`;
    throw new Error(typeof message === "string" ? message : message.message || "请求失败");
  }
  return result;
}

function option(value, label) {
  const result = document.createElement("option");
  result.value = value;
  result.textContent = label;
  return result;
}

function readCurrentModel({ refresh = false } = {}) {
  if (!state.currentPending && (refresh || !state.currentRequest)) {
    state.currentPending = true;
    const request = api("/api/expert/current").catch((error) => {
      if (state.currentRequest === request) state.currentRequest = null;
      throw error;
    }).finally(() => {
      state.currentPending = false;
    });
    state.currentRequest = request;
  }
  return state.currentRequest;
}

async function readLinkedModel(params, { refresh = false } = {}) {
  const sourceRun = params.get("source_run") || "";
  const modelVersion = params.get("model_version") || "";
  if (!sourceRun && !modelVersion) return readCurrentModel({ refresh });
  if (!/^[a-z0-9][a-z0-9_.-]{0,127}$/i.test(sourceRun)) throw new Error("来源研究编号无效。");
  const research = await api(`/api/expert/runs/${encodeURIComponent(sourceRun)}`);
  const unified = research.unified || {};
  if (research.run?.run_id !== sourceRun || research.decision?.state !== "research_only") {
    throw new Error("指定研究尚未形成可回测的统一候选。");
  }
  if (modelVersion && modelVersion !== unified.model_version) throw new Error("链接中的模型版本与统一研究不一致。");
  const coverage = research.input_coverage || {};
  return { ...research, run_id: sourceRun, source_run_id: sourceRun,
    model_id: unified.model_version || research.model_id || sourceRun, model_version: unified.model_version || "",
    analysis_start: coverage.first_evaluation_session, as_of: coverage.last_evaluation_session, status: "research_only",
    label: research.final_portfolio?.arm === "online_knn_volume_trend" ? "纯在线 KNN · 指定研究候选"
      : research.final_portfolio?.arm === "knn60_multimodal40" ? "KNN 60% + 多模态 40% · 指定研究候选" : "多模态专家 · 指定统一研究候选" };
}

function linkedModelParams(query, model) {
  if (model?.source_run_id) query.set("source_run", model.source_run_id);
  if (model?.model_version) query.set("model_version", model.model_version);
  return query;
}

function renderCurrentModel(target, model) {
  const label = currentLabels[model?.status] || currentLabels.unavailable;
  const health = model?.score_health;
  const fold = arr(health?.folds).find((item) => item.status === "attention");
  const warning = fold ? `<p class="current-model-health warn-text">分数诊断 · ${h(fold.period)}：${h(arr(fold.concerns)[0] || health.summary)} 仅作历史复核，不改变发布状态。</p>` : "";
  target.innerHTML = `<div><strong>${h(model?.label || "当前策略模型")}</strong><span class="pill ${model?.status === "approved" ? "" : "warn"}">${h(label)}</span></div>` +
    (model?.run_id ? `<dl class="current-model-identity">${metricItems([["模型版本", model.model_id], ["数据截止", model.as_of], ["研究编号", model.run_id]])}</dl>${warning}`
      : `<p>${h(model?.reason || "当前没有可用于回放的策略模型。")}</p>`);
}

function renderData() {
  const data = state.overview?.data || {};
  $("#market-status").textContent = state.overviewError ? "行情覆盖读取失败"
    : data.latest_date ? `行情截至 ${data.latest_date}` : state.overview ? "暂无行情覆盖" : "读取行情覆盖…";
  $("#market-dot").className = `status-dot ${state.overviewError ? "error" : data.latest_date ? "ready" : ""}`;
  $("#coverage").innerHTML = metricItems([["最早交易日", data.data_start], ["最新交易日", data.latest_date], ["最新日股票", count(data.latest_stocks)]]);
  $("#data-detail").textContent = state.overviewError || (state.overview ? `${count(data.total_stocks)} 只股票在行情源中有历史记录。` : "正在读取本地行情数据。");
  const classification = arr(state.overview?.classification_options || state.overview?.classifications);
  $("#classification-detail").textContent = classification.length
    ? `已配置${classification.map((item) => item.label).join("、")}。缺失分类保留为未知。`
    : "缺失分类保留为未知。";
  const universe = state.overview?.universe;
  const target = $("#universe-content");
  target.hidden = !universe;
  if (universe) {
    const rows = Object.entries(universe.active_by_board || {}).map(([board, total]) => [
      h(boardNames[board] || board), memberCount(total), memberCount(universe.active_trading_by_board?.[board]),
    ]);
    target.innerHTML = panel("已配置股票池覆盖", metrics([
      ["覆盖日期", universe.date], ["在册证券", memberCount(universe.active_stocks)],
      ["有交易行情", memberCount(universe.active_trading_stocks)], ["缺少交易行情", memberCount(universe.active_missing_stocks)],
    ]) + (rows.length ? scrollTable([["板块"], ["在册证券", "numeric"], ["有交易行情", "numeric"]], rows) : ""));
  }
}

function refreshOverview() {
  if (!state.refreshInFlight) {
    state.refreshInFlight = api("/api/overview").then((result) => {
      state.overview = result;
      state.overviewError = "";
    }).catch((error) => {
      state.overviewError = error.message;
    }).finally(() => {
      state.refreshInFlight = null;
      renderData();
      renderDataUpdate();
      window.dispatchEvent(new Event("workbench:overview"));
    });
  }
  return state.refreshInFlight;
}

function dataJobActive(job = state.dataStatus?.job) {
  return ["queued", "running"].includes(job?.status);
}

function renderDataUpdate() {
  const job = state.dataStatus?.job;
  const result = job?.result;
  const progress = job?.progress;
  const labels = { queued: "等待启动", running: "更新中", completed: "已完成", partial: "部分完成", failed: "失败" };
  const label = labels[job?.status] || "未开始";
  const statePill = $("#data-update-state");
  statePill.textContent = label;
  statePill.className = `pill ${["partial", "failed"].includes(job?.status) ? "warn" : ""}`;
  $("#data-update-metrics").innerHTML = metricItems([
    ["更新范围", state.dataStatus?.scope_label || job?.scope_label || "原始仓全部股票"],
    ["目标完整交易日", job?.target_date || state.dataStatus?.target_date],
    ["处理进度", progress ? `${count(progress.processed)}／${count(progress.total)}` : dataJobActive(job) ? "正在准备" : "—"],
    ["当前行情截止", state.overview?.data?.latest_date],
  ]);
  $("#data-update-status").textContent = state.dataStatusError || (state.dataUpdatePending ? "正在提交数据更新…" : "") || job?.message
    || (state.dataStatus ? "点击后才会联网更新；页面加载和导航不会启动任务。" : "正在读取更新状态…");
  $("#update-data").disabled = state.dataUpdatePending || dataJobActive(job) || !state.dataStatus;
  const summary = result?.summary;
  $("#data-update-result").innerHTML = result ? metrics([
    ["更新前截止", result.before?.latest_date],
    ["更新后截止", result.after?.latest_date],
    ["处理股票", count(summary?.total)],
    ["成功", count(summary?.success)],
    ["失败", count(summary?.failed)],
    ["达到目标日", count(summary?.target_success)],
  ]) + (arr(result.failure_examples).length ? `<ul class="latent-blockers">${result.failure_examples.map((item) => `<li>${h(item)}</li>`).join("")}</ul>` : "") : "";
}

function scheduleDataPoll() {
  clearTimeout(state.dataPollTimer);
  state.dataPollTimer = null;
  const job = state.dataStatus?.job;
  if (!dataJobActive(job) || document.hidden || $("#page-data").hidden) return;
  state.dataPollTimer = setTimeout(() => pollDataJob(job.job_id, state.dataGeneration), 2500);
}

async function pollDataJob(jobId, generation) {
  if (generation !== state.dataGeneration || state.dataPollInFlight) return state.dataPollInFlight;
  const request = api(`/api/data/jobs/${encodeURIComponent(jobId)}`).then(async (job) => {
    if (generation !== state.dataGeneration || state.dataStatus?.job?.job_id !== jobId) return;
    state.dataStatus = { ...state.dataStatus, job };
    state.dataStatusError = "";
    renderDataUpdate();
    if (dataJobActive(job)) scheduleDataPoll();
    else await refreshOverview();
  }).catch((error) => {
    if (generation !== state.dataGeneration) return;
    state.dataStatusError = error.message;
    renderDataUpdate();
    scheduleDataPoll();
  }).finally(() => {
    if (state.dataPollInFlight === request) state.dataPollInFlight = null;
  });
  state.dataPollInFlight = request;
  return request;
}

function refreshDataStatus() {
  clearTimeout(state.dataPollTimer);
  if (state.dataStatusInFlight) return state.dataStatusInFlight;
  const generation = ++state.dataGeneration;
  const request = api("/api/data/status").then((result) => {
    if (generation !== state.dataGeneration) return;
    state.dataStatus = result;
    state.dataStatusError = "";
    renderDataUpdate();
    scheduleDataPoll();
  }).catch((error) => {
    if (generation !== state.dataGeneration) return;
    state.dataStatusError = error.message;
    renderDataUpdate();
  }).finally(() => {
    if (state.dataStatusInFlight === request) state.dataStatusInFlight = null;
  });
  state.dataStatusInFlight = request;
  return request;
}

$("#update-data").addEventListener("click", async () => {
  if (state.dataUpdatePending || dataJobActive()) return;
  clearTimeout(state.dataPollTimer);
  const generation = ++state.dataGeneration;
  state.dataUpdatePending = true;
  state.dataStatusError = "";
  $("#data-update-status").textContent = "正在提交数据更新…";
  renderDataUpdate();
  try {
    const job = await api("/api/data/updates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (generation !== state.dataGeneration) return;
    state.dataStatus = { ...state.dataStatus, job, target_date: job.target_date };
    $("#data-update-status").textContent = "数据更新已提交。";
    renderDataUpdate();
    await pollDataJob(job.job_id, generation);
  } catch (error) {
    if (generation === state.dataGeneration) state.dataStatusError = error.message;
  } finally {
    if (generation === state.dataGeneration) {
      state.dataUpdatePending = false;
      renderDataUpdate();
      scheduleDataPoll();
    }
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) clearTimeout(state.dataPollTimer);
  else scheduleDataPoll();
});

function requestedPage() {
  return location.hash.slice(1).split("?", 1)[0];
}

function showPage() {
  const requested = requestedPage();
  if (requested === "workspace") return;
  const retired = Object.hasOwn(retiredViews, requested) && retiredViews[requested];
  const page = views.has(requested) ? requested : retired || "expert";
  if (retired) {
    const params = new URLSearchParams(location.hash.split("?")[1]);
    const symbol = params.get("symbol");
    const query = retired === "stock-chart" && symbol ? `?${new URLSearchParams({ symbol })}` : "";
    history.replaceState(null, "", `#${page}${query}`);
  }
  $("#workspace").dataset.workbenchPage = page;
  for (const name of views) $(`#page-${name}`).hidden = name !== page;
  const nav = $(".navigation");
  const link = nav.querySelector(`[data-page="${page}"]`);
  nav.querySelector("[aria-current]")?.removeAttribute("aria-current");
  link.setAttribute("aria-current", "page");
  if (nav.scrollWidth > nav.clientWidth) {
    const visible = nav.getBoundingClientRect();
    const tab = link.getBoundingClientRect();
    if (tab.right > visible.right) nav.scrollLeft += tab.right - visible.right + 4;
    if (tab.left < visible.left) nav.scrollLeft -= visible.left - tab.left + 4;
  }
  scheduleDataPoll();
}

$("#refresh").addEventListener("click", () => {
  const refresh = $("#refresh");
  refresh.disabled = true;
  const requests = [refreshOverview(), readCurrentModel({ refresh: true }), refreshDataStatus()];
  Promise.allSettled(requests).then(() => { refresh.disabled = false; });
});
window.addEventListener("hashchange", showPage);
showPage();
refreshOverview();
refreshDataStatus();
