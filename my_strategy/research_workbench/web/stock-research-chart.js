"use strict";

/* SVG renderer adapted from KCPredict's MIT-licensed composite chart.
 * Historical facts, derived research marks and frozen scenarios remain separate. */
(() => {
  const $ = (id) => document.getElementById(id);
  const escape = (value) => String(value ?? "—").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const short = (value) => String(value || "").slice(0, 10);
  const fmt = (value, digits = 2) => finite(value) ? value.toFixed(digits) : "—";
  const percent = (value, digits = 2) => finite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
  const volume = (value) => finite(value) ? value >= 1e8 ? `${fmt(value / 1e8)}亿股` : value >= 1e4 ? `${fmt(value / 1e4)}万股` : `${fmt(value, 0)}股` : "—";
  const palette = {
    surface: "#fcfdfa", ink: "#182522", muted: "#566760", grid: "#e5ece7", rule: "#d5ded8",
    up: "#b43b42", down: "#12775d", blue: "#2a78d6", gold: "#eb6834", purple: "#6d28d9",
    amber: "#82530d", band: "#8aaae3", sample: "#7f98ba", future: "#edf2fa", chips: "#eb6834",
    ictBuy: "#0f766e", ictSell: "#6d28d9", wyckoffBuy: "#b36a13", wyckoffSell: "#2a78d6",
    ma5: "#1baf7a", ma10: "#eda100", ma20: "#e87ba4", ma60: "#008300",
  };
  const statusNames = { pass: "预测质量通过", fail: "预测质量未通过", unavailable: "预测不可用", not_executed: "预测未执行" };
  const table = (headers, rows) => `<table><thead><tr>${headers.map((item) => `<th>${escape(item)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${escape(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;

  function create(options) {
    const host = options.host;
    host.innerHTML = '<div class="stock-research-svg-wrap"><svg id="stock-research-svg" role="img" tabindex="0" aria-label="真实历史、结构参考与冻结预测连续时间轴；左右方向键平移，加减键缩放，Home恢复"></svg><div id="stock-research-tooltip" class="stock-research-tooltip" hidden></div></div>';
    const svg = $("stock-research-svg");
    const tooltip = $("stock-research-tooltip");
    const state = {
      data: null, view: null, B: [], F: [], N: 0, H: 0, T: 0, lastIndex: -1, last: null,
      dates: [], dateIndex: new Map(), frequency: "日线", start: 0, end: 0,
      layers: { structure: true, zones: true, fractals: false, sources: true, wyckoff: true,
        divergences: false, ict: true, prediction: true, band: true, samples: false,
        volume: true, ma: false, boll: false, chips: false },
      hover: null, pinned: false, drag: null, geometry: null,
    };
    let renderFrame = 0;

    function scheduleMarkers() {
      if (renderFrame) return;
      renderFrame = requestAnimationFrame(() => {
        renderFrame = 0;
        options.onRender?.();
      });
    }

    function available(bar) {
      return bar && ["open", "high", "low", "close"].every((key) => finite(bar[key]));
    }

    function currentView() {
      return state.data?.layers?.frequencies?.[state.frequency] || null;
    }

    function periodKey(value) {
      const day = short(value);
      if (state.frequency === "月线") return day.slice(0, 7);
      if (state.frequency === "周线") {
        const target = new Date(`${day}T00:00:00Z`);
        target.setUTCDate(target.getUTCDate() + 4 - (target.getUTCDay() || 7));
        const yearStart = new Date(`${target.getUTCFullYear()}-01-01T00:00:00Z`);
        const week = Math.ceil(((target - yearStart) / 86400000 + 1) / 7);
        return `${target.getUTCFullYear()}-W${String(week).padStart(2, "0")}`;
      }
      return day;
    }

    function indexForDate(value) {
      if (state.frequency === "日线") return state.dateIndex.get(short(value));
      const key = periodKey(value);
      return state.B.findIndex((bar) => periodKey(bar.time) === key);
    }

    function setFrequency(name) {
      if (!state.data || name === state.frequency) return;
      state.frequency = name;
      resetDataWindow();
      for (const button of document.querySelectorAll("[data-stock-frequency]")) {
        button.setAttribute("aria-pressed", String(button.dataset.stockFrequency === name));
      }
      $("stock-research-period-note").textContent = name === "日线" ? "真实历史与冻结情景共用时间轴" : `${name}仅展示历史；日线预测不改造成${name}预测，形成中末柱不作闭合确认`;
      inspect(state.lastIndex);
      draw();
    }

    function resetDataWindow() {
      const view = currentView();
      state.B = view?.bars || [];
      state.F = state.frequency === "日线" ? state.data?.prediction?.forecast || [] : [];
      state.N = state.B.length;
      state.H = state.F.length;
      state.T = state.N + state.H;
      state.lastIndex = state.B.findLastIndex(available);
      state.last = state.lastIndex >= 0 ? state.B[state.lastIndex] : null;
      state.dates = state.B.map((bar) => short(bar.time)).concat(state.F.map((row) => short(row.date)));
      state.dateIndex = new Map(state.dates.map((day, index) => [day, index]));
      state.start = 0;
      state.end = Math.max(0, state.T - 1);
      state.hover = state.pinned = null;
    }

    function mount(view) {
      state.view = view;
      state.data = view.research;
      state.frequency = "日线";
      resetDataWindow();
      buildDetails();
      updateQuality();
      if (state.lastIndex >= 0) inspect(state.lastIndex);
      draw();
      for (const button of document.querySelectorAll("[data-stock-frequency]")) {
        const rows = state.data.layers?.frequencies?.[button.dataset.stockFrequency]?.bars || [];
        button.disabled = !rows.length;
        button.setAttribute("aria-pressed", String(button.dataset.stockFrequency === "日线"));
      }
    }

    function updateQuality() {
      const prediction = state.data?.prediction || { status: "unavailable" };
      const quality = prediction.quality || {};
      const badge = $("stock-research-quality");
      badge.className = `stock-research-quality ${prediction.status}`;
      const sample = Number.isFinite(quality.valid_fraction)
        ? `有效 ${quality.valid_samples ?? "—"}/${quality.requested_samples ?? "—"} · 有效率 ${percent(quality.valid_fraction, 1)} / 门槛 ${percent(quality.minimum_valid_fraction ?? .8, 0)}`
        : "没有可评估的样本统计";
      badge.innerHTML = `<strong>${escape(statusNames[prediction.status] || "预测不可用")}</strong><span>${escape(prediction.reason || "未提供预测工件。")}</span><small>${escape(sample)} · 页面未触发推理</small>`;
      $("stock-research-scenario-note").textContent = state.H ? `末真实收盘连接 ${state.H} 日冻结情景；不是实际成交` : "没有未来路径；未推算或补造交易日期";
    }

    function buildDetails() {
      const data = state.data;
      const layers = data.layers || {};
      const daily = layers.frequencies?.["日线"] || {};
      const lastIndicators = daily.indicators?.rows?.at(-1) || {};
      const volumeRow = layers.volume_rows?.at(-1) || {};
      const prediction = data.prediction || {};
      const cells = [
        ["CZSC 结构", `${daily.pens?.length || 0} 笔 / ${daily.zones?.length || 0} 中枢`, "锚点与确认时钟分开；形成中笔会变化。"],
        ["威科夫 / 量价", `相对此前20日量 ${fmt(lastIndicators.relative_volume20)} 倍`, `收盘位置 ${percent(volumeRow.w_close_location, 1)}；旧状态不视为新确认。`],
        ["ICT / SMC", `${data.ict?.recent_fvgs?.length || 0} 个严格 FVG`, "扫荡只是机械候选；MSS、有效回撤与净盈亏比未确认。"],
      ];
      $("stock-research-summary").innerHTML = cells.map(([name, headline, body]) => `<section><h3>${escape(name)}</h3><strong>${escape(headline)}</strong><p>${escape(body)}</p></section>`).join("");
      const history = [];
      for (const zone of daily.zones || []) history.push(["CZSC 中枢", `${short(zone.start_time)} → ${short(zone.end_time)}`, `${fmt(zone.low)}–${fmt(zone.high)}`, short(zone.confirmation_time), zone.display_confirmed ? "已确认 / 回溯几何" : "形成中"]);
      for (const pen of daily.pens || []) history.push(["CZSC 笔", `${short(pen.start_time)} → ${short(pen.end_time)}`, `${fmt(pen.start_price)} → ${fmt(pen.end_price)}`, short(pen.confirmation_time), pen.display_confirmed ? "已确认 / 回溯锚点" : "形成中"]);
      for (const gap of data.ict?.recent_fvgs || []) history.push([`${gap.direction === "bearish" ? "空头" : "多头"} FVG`, (gap.candle_dates || []).join(" / "), `${fmt(gap.zone[0])}–${fmt(gap.zone[1])}`, short(gap.confirmed_time), gap.lifecycle?.status || "未报告"]);
      $("stock-research-history-table").innerHTML = history.length ? table(["方法", "锚点 / 三根K线", "价格", "确认时钟", "状态"], history) : '<p class="stock-research-note">本窗口没有可展示的确认结构。</p>';
      const forecast = prediction.forecast || [];
      $("stock-research-forecast-table").innerHTML = forecast.length ? table(["交易日期", "P10 收盘", "中位收盘", "P90 收盘", "中位变化 / 末收盘", "中位量（模型）"], forecast.map((row) => [row.date, fmt(row.close_p10), fmt(row.close), fmt(row.close_p90), percent(row.close / state.last.close - 1), volume(row.volume)])) : `<p class="stock-research-note">${escape(statusNames[prediction.status] || "预测不可用")}：${escape(prediction.reason || "没有已评估的未来路径")}</p>`;
      $("stock-research-audit").textContent = JSON.stringify({ meta: data.meta, prediction: { ...prediction, samples: `${prediction.samples?.length || 0} 条有效路径` }, notices: data.notices }, null, 2);
      renderSourceEvents();
    }

    function renderSourceEvents() {
      const family = $("stock-research-event-family")?.value || "";
      const selected = state.pinned && state.hover !== null && state.hover < state.N ? periodKey(state.dates[state.hover]) : null;
      const source = (state.data?.layers?.source_events || []).filter((row) => (!family || row.source === family) && (!selected || periodKey(row.date) === selected));
      const candidates = (state.data?.ict?.candidates || []).filter((row) => (!family || family === "ict") && (!selected || periodKey(row.sweep?.time) === selected)).map((row) => ({ source: "ict", date: row.sweep.time, side: row.direction === "long" ? "BUY" : "SELL", event_type: `${row.direction === "long" ? "SSL" : "BSL"} 机械候选`, first_available_at: row.sweep.time, reference_price: row.sweep.extreme, evidence: row }));
      const rows = [...source, ...candidates].sort((a, b) => short(b.date).localeCompare(short(a.date)));
      $("stock-research-source-events").innerHTML = rows.length ? rows.map((row) => `<article><header><span>${escape({ structure: "CZSC 结构", wyckoff: "威科夫量价", ict: "ICT" }[row.source] || row.source)}</span><time>${escape(short(row.date))}</time></header><strong class="${row.side === "BUY" ? "stock-chart-side-buy" : "stock-chart-side-sell"}">${escape(row.event_type)} · ${escape(row.side)}</strong><p>首次可用 ${escape(row.first_available_at || "未报告")}<br>确认 ${escape(row.confirmed_at || "未确认")}<br>来源参考价 ${fmt(row.reference_price)} · 非成交价</p></article>`).join("") : '<p class="stock-research-note">没有匹配的来源事件；未返回不代表历史从未发生。</p>';
    }

    function inspect(index) {
      if (!Number.isInteger(index) || index < 0 || index >= state.T) return;
      state.hover = index;
      const historical = index < state.N;
      const bar = historical ? state.B[index] : state.F[index - state.N];
      options.onInspect?.(historical ? bar : null, historical ? null : bar, state.dates[index]);
      const rows = currentView()?.indicators?.rows || [];
      const indicator = historical ? rows.find((row) => short(row.time) === short(bar.time)) : null;
      $("stock-research-indicator-readout").textContent = indicator ? [
        `MA5 ${fmt(indicator.ma5)}`, `MA10 ${fmt(indicator.ma10)}`, `MA20 ${fmt(indicator.ma20)}`, `MA60 ${fmt(indicator.ma60)}`,
        `BOLL ${fmt(indicator.boll_lower)} / ${fmt(indicator.boll_mid)} / ${fmt(indicator.boll_upper)}`,
        `70%估算 ${fmt(indicator.chip70_low)}–${fmt(indicator.chip70_high)}`, indicator.is_closed === false ? "形成中指标" : "已闭合指标",
      ].join("　") : historical ? "此日期没有已提供的历史指标" : "未来为模型情景；没有历史指标或筹码分布";
      renderSourceEvents();
    }

    function drawIndicators(context) {
      const specs = [];
      if (state.layers.ma) for (const n of [5, 10, 20, 60]) specs.push([`ma${n}`, palette[`ma${n}`], ""]);
      if (state.layers.boll) for (const key of ["boll_upper", "boll_mid", "boll_lower"]) specs.push([key, palette.blue, "4 3"]);
      if (state.layers.chips && state.frequency === "日线") for (const key of ["chip70_low", "chip50", "chip70_high"]) specs.push([key, palette.chips, "3 3"]);
      const rows = currentView()?.indicators?.rows || [];
      const output = [];
      for (const [key, color, dash] of specs) {
        let points = [];
        const flush = () => {
          if (points.length && !context.valuesOnly) output.push(line(points, color, 1.2, dash));
          points = [];
        };
        for (let index = state.start; index <= Math.min(state.N - 1, state.end); index++) {
          const row = rows.find((item) => short(item.time) === short(state.B[index].time));
          const value = row?.[key];
          if (finite(value)) {
            context.values.push(value);
            if (!context.valuesOnly) points.push([context.x(index), context.y(value)]);
          } else flush();
        }
        flush();
      }
      return output.join("");
    }

    function sourceMarkerRows() {
      const view = currentView();
      const rows = [];
      if (state.layers.fractals) for (const row of view?.fractals || []) rows.push({ family: "fractals", time: row.time, side: row.kind === "top" ? "SELL" : "BUY", label: `${row.is_confirmed ? "" : "形成"}${row.kind === "top" ? "顶" : "底"}`, detail: `锚点 ${short(row.time)}；确认 ${short(row.confirmed_at)}` });
      if (state.layers.divergences) for (const row of view?.divergences || []) rows.push({ family: "divergences", time: row.time, side: row.kind === "top" ? "SELL" : "BUY", label: row.label, detail: `MACD面积 ${fmt(row.previous_macd_area, 4)} → ${fmt(row.macd_area, 4)}；仅作背驰参考` });
      const sources = state.data.layers.source_events || [];
      if (state.layers.sources) for (const row of sources) rows.push({ family: "sources", time: row.date, side: row.side, label: `${row.source === "structure" ? "结" : "量"}${row.side === "BUY" ? " B" : " S"}`, detail: `${row.event_type}；参考价 ${fmt(row.reference_price)}；非成交价` });
      if (state.layers.wyckoff) for (const row of sources.filter((item) => item.source === "wyckoff")) rows.push({ family: "wyckoff", time: row.date, side: row.side, label: row.event_type, color: row.side === "BUY" ? palette.wyckoffBuy : palette.wyckoffSell, shape: row.side === "BUY" ? "circle" : "square", detail: "威科夫机械量价参考；不等于 Wyckoff 全流程确认" });
      if (state.layers.ict && state.frequency === "日线") for (const row of state.data.ict?.candidates || []) rows.push({ family: "ict", time: row.sweep.time, side: row.direction === "long" ? "BUY" : "SELL", label: row.direction === "long" ? "SSL候选" : "BSL候选", color: row.direction === "long" ? palette.ictBuy : palette.ictSell, shape: row.direction === "long" ? "circle" : "square", detail: `${row.mechanical_status}；${row.reason}` });
      return rows;
    }

    function drawMarkers(context) {
      const grouped = new Map();
      for (const row of sourceMarkerRows()) {
        const index = indexForDate(row.time);
        if (!Number.isInteger(index) || index < state.start || index > state.end || !available(state.B[index])) continue;
        const key = `${index}:${row.side}:${row.family}`;
        if (!grouped.has(key)) grouped.set(key, []);
        grouped.get(key).push(row);
      }
      let output = "";
      const lanes = new Map();
      for (const [key, rows] of grouped) {
        const [indexText, side] = key.split(":");
        const index = Number(indexText);
        const laneKey = `${index}:${side}`;
        const lane = lanes.get(laneKey) || 0;
        lanes.set(laneKey, lane + 1);
        const buy = side === "BUY";
        const row = rows[0];
        const x = context.x(index);
        const y = context.y(buy ? state.B[index].low : state.B[index].high) + (buy ? 1 : -1) * (13 + lane * 19);
        const color = row.color || (buy ? palette.up : palette.down);
        if (row.shape === "circle") output += `<circle cx="${x}" cy="${y}" r="3.5" fill="${color}"/>`;
        else if (row.shape === "square") output += `<rect x="${x - 3.5}" y="${y - 3.5}" width="7" height="7" fill="${color}"/>`;
        else output += `<path d="M${x} ${y + (buy ? -5 : 5)}l-4 ${buy ? 7 : -7}h8Z" fill="${color}"/>`;
        if (context.step > 7 || rows.some((item) => / [BS]$/.test(item.label))) output += `<text x="${x}" y="${y + (buy ? 15 : -8)}" text-anchor="middle" style="fill:${color};font-size:10px">${escape(rows.length > 1 ? `${row.label} +${rows.length - 1}` : row.label)}</text>`;
        context.interaction.push({ x0: x - 12, x1: x + 12, y0: y - 18, y1: y + 22, text: rows.map((item) => item.label).join(" / "), detail: rows.map((item) => item.detail).join("；") });
      }
      return output;
    }

    function line(points, color, width = 1, dash = "") {
      return points.length ? `<polyline points="${points.map((point) => point.join(",")).join(" ")}" fill="none" stroke="${color}" stroke-width="${width}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>` : "";
    }

    function draw() {
      if (!state.data || !state.T || state.lastIndex < 0) {
        svg.replaceChildren();
        scheduleMarkers();
        return;
      }
      const W = Math.max(1, svg.clientWidth), height = svg.clientHeight || 570;
      const left = 14, right = W < 600 ? 59 : 76, top = 29;
      const priceBottom = state.layers.volume ? height - 159 : height - 48;
      const volumeTop = height - 119, volumeBottom = height - 37, bottom = height - 12;
      const pw = Math.max(1, W - left - right), count = state.end - state.start + 1, step = pw / count;
      const x = (index) => left + (index - state.start + .5) * step;
      const clipX = (index) => Math.max(left, Math.min(W - right, x(index)));
      const context = { x, values: [], interaction: [] };
      const values = [];
      for (let index = state.start; index <= Math.min(state.end, state.N - 1); index++) if (available(state.B[index])) values.push(state.B[index].low, state.B[index].high);
      if (state.layers.prediction) {
        for (let index = Math.max(state.N, state.start); index <= state.end; index++) {
          const row = state.F[index - state.N];
          if (row) {
            values.push(row.close);
            if (state.layers.band) values.push(row.close_p10, row.close_p90);
          }
        }
        if (state.layers.samples) for (const path of state.data.prediction.samples || []) for (let index = Math.max(state.N, state.start); index <= state.end; index++) if (finite(path[index - state.N]?.[3])) values.push(path[index - state.N][3]);
      }
      const indicatorContext = { valuesOnly: true, values, x, y: null };
      drawIndicators(indicatorContext);
      if (!values.length) values.push(state.last.close * .95, state.last.close * 1.05);
      let min = Math.min(...values), max = Math.max(...values);
      const span = Math.max(max - min, max * .04, .01), low = Math.max(0, min - span * .1), high = max + span * .14;
      const y = (value) => priceBottom - (value - low) / (high - low) * (priceBottom - top);
      Object.assign(context, { y, low, high, step, priceBottom, volumeBottom, W, height, left, right });
      let output = `<defs><clipPath id="stock-research-price-clip"><rect x="${left}" y="${top}" width="${pw}" height="${priceBottom - top}"/></clipPath></defs>`;
      if (state.H && state.end >= state.N) output += `<rect x="${Math.max(left, x(state.N - .5))}" y="${top}" width="${Math.max(0, W - right - Math.max(left, x(state.N - .5)))}" height="${volumeBottom - top}" fill="${palette.future}"/>`;
      for (let index = 0; index <= 5; index++) {
        const value = low + (high - low) * index / 5, yy = y(value);
        output += `<line x1="${left}" y1="${yy}" x2="${W - right}" y2="${yy}" stroke="${palette.grid}"/><text x="${W - right + 9}" y="${yy + 3}">${fmt(value)}</text>`;
      }
      output += `<text x="${W - right + 9}" y="17">价格 / 元</text><g clip-path="url(#stock-research-price-clip)">`;
      if (state.layers.zones) for (const [index, zone] of (currentView()?.zones || []).entries()) {
        const start = indexForDate(zone.start_time), end = indexForDate(zone.end_time);
        if (!Number.isInteger(start) || !Number.isInteger(end) || end < state.start || start > state.end) continue;
        const xx = clipX(start - .4), width = Math.max(0, clipX(end + .4) - xx), color = zone.direction === "up" ? palette.up : palette.down;
        output += `<rect x="${xx}" y="${y(zone.high)}" width="${width}" height="${Math.max(1, y(zone.low) - y(zone.high))}" rx="2" fill="${color}" fill-opacity=".08" stroke="${color}" stroke-opacity=".65"${zone.display_confirmed ? "" : ' stroke-dasharray="5 4"'}/>`;
        if (width > 110) output += `<text class="chart-label" x="${xx + 8}" y="${y(zone.low) - 8}">${fmt(zone.low)}–${fmt(zone.high)} · ${zone.display_confirmed ? `确认 ${escape(short(zone.confirmation_time))}` : "形成中"}</text>`;
        context.interaction.push({ x0: xx, x1: xx + width, y0: y(zone.high), y1: y(zone.low), text: `回溯中枢 ${fmt(zone.low)}–${fmt(zone.high)}`, detail: `锚点 ${short(zone.start_time)} → ${short(zone.end_time)}；确认 ${short(zone.confirmation_time)}` });
      }
      const fvg = state.data.ict?.latest_fvg;
      if (state.frequency === "日线" && state.layers.ict && fvg?.zone) {
        const index = indexForDate(fvg.confirmed_time), life = fvg.lifecycle || {}, end = life.is_active ? state.N - 1 : indexForDate(life.close_through_date) ?? state.N - 1;
        if (Number.isInteger(index) && end >= state.start) {
          const xx = clipX(index - .35), width = Math.max(1, clipX(end + .45) - xx);
          output += `<rect x="${xx}" y="${y(fvg.zone[1])}" width="${width}" height="${Math.max(2, y(fvg.zone[0]) - y(fvg.zone[1]))}" fill="${palette.ictBuy}" fill-opacity=".1" stroke="${palette.ictBuy}" stroke-opacity=".65"/>`;
          context.interaction.push({ x0: xx, x1: xx + width, y0: y(fvg.zone[1]) - 7, y1: y(fvg.zone[0]) + 7, text: `严格三K线 FVG ${fmt(fvg.zone[0])}–${fmt(fvg.zone[1])}`, detail: `确认 ${short(fvg.confirmed_time)}；${life.status || "生命周期未报告"}；非买卖确认` });
        }
      }
      if (state.layers.prediction && state.H) {
        const points = (key) => [[state.lastIndex, state.last.close], ...state.F.map((row, index) => [state.N + index, row[key]])].map(([index, value]) => [x(index), y(value)]);
        if (state.layers.samples) for (const path of state.data.prediction.samples || []) output += `<g opacity=".22">${line([[x(state.lastIndex), y(state.last.close)], ...path.map((row, index) => [x(state.N + index), y(row[3])])], palette.sample, .8)}</g>`;
        if (state.layers.band) output += `<polygon points="${points("close_p90").concat(points("close_p10").reverse()).map((point) => point.join(",")).join(" ")}" fill="${palette.band}" fill-opacity=".25"/>${line(points("close_p90"), palette.band, .8)}${line(points("close_p10"), palette.band, .8)}`;
        output += line(points("close"), palette.blue, 2.4, "6 4");
        state.F.forEach((row, index) => { if (state.N + index >= state.start && state.N + index <= state.end) output += `<circle cx="${x(state.N + index)}" cy="${y(row.close)}" r="2.7" fill="${palette.surface}" stroke="${palette.blue}" stroke-width="1.7"/>`; });
      }
      for (let index = Math.max(0, state.start); index <= Math.min(state.N - 1, state.end); index++) {
        const bar = state.B[index];
        if (!available(bar)) continue;
        const color = bar.is_closed === false ? palette.muted : bar.close >= bar.open ? palette.up : palette.down;
        const xx = x(index), width = Math.max(1, Math.min(17, step * .56));
        output += `<line x1="${xx}" y1="${y(bar.high)}" x2="${xx}" y2="${y(bar.low)}" stroke="${color}"/><rect x="${xx - width / 2}" y="${Math.min(y(bar.open), y(bar.close))}" width="${width}" height="${Math.max(1, Math.abs(y(bar.open) - y(bar.close)))}" fill="${color}"/>`;
      }
      output += drawIndicators(context);
      if (state.layers.structure) for (const pen of currentView()?.pens || []) {
        const start = indexForDate(pen.start_time), end = indexForDate(pen.end_time);
        if (Number.isInteger(start) && Number.isInteger(end)) output += line([[x(start), y(pen.start_price)], [x(end), y(pen.end_price)]], palette.blue, pen.display_confirmed ? 2 : 1.4, pen.display_confirmed ? "" : "5 4");
      }
      output += drawMarkers(context);
      if (state.lastIndex >= state.start && state.lastIndex <= state.end) output += line([[left, y(state.last.close)], [W - right, y(state.last.close)]], palette.muted, .7, "2 4");
      output += "</g>";
      if (state.lastIndex >= state.start && state.lastIndex <= state.end) output += `<line x1="${x(state.lastIndex)}" y1="${top}" x2="${x(state.lastIndex)}" y2="${volumeBottom}" stroke="${palette.muted}" stroke-dasharray="3 4"/><text x="${left + 3}" y="17">实测截止 ${escape(short(state.last.time))}</text>`;
      if (state.H && state.end >= state.N) output += `<text x="${W - right - 3}" y="17" text-anchor="end" style="fill:${palette.blue}">未来 ${state.H} 日${state.data.prediction.status === "fail" ? " · 质量未通过" : " · 冻结情景"}</text>`;
      if (!state.H) output += `<text x="${W - right - 9}" y="${top + 20}" text-anchor="end" style="fill:${palette.amber}">${escape(statusNames[state.data.prediction.status])}；未生成未来路径</text>`;
      if (state.layers.volume) {
        const volumes = [];
        for (let index = state.start; index <= Math.min(state.N - 1, state.end); index++) if (finite(state.B[index].volume)) volumes.push(state.B[index].volume);
        if (state.layers.prediction) for (let index = Math.max(state.N, state.start); index <= state.end; index++) if (finite(state.F[index - state.N]?.volume)) volumes.push(state.F[index - state.N].volume);
        const maximum = Math.max(1, ...volumes), vy = (value) => volumeBottom - value / maximum * (volumeBottom - volumeTop - 15);
        output += `<line x1="${left}" y1="${volumeTop}" x2="${W - right}" y2="${volumeTop}" stroke="${palette.grid}"/><text x="${left}" y="${volumeTop + 12}">成交量 · 金线为此前20日均量</text>`;
        const means = [];
        for (let index = state.start; index <= state.end; index++) {
          const historical = index < state.N, bar = historical ? state.B[index] : state.F[index - state.N];
          if (!bar || (!historical && !state.layers.prediction)) continue;
          if (finite(bar.volume)) {
            const width = Math.max(1, Math.min(17, step * .56)), color = historical && available(bar) ? bar.close >= bar.open ? palette.up : palette.down : palette.band;
            output += `<rect x="${x(index) - width / 2}" y="${vy(bar.volume)}" width="${width}" height="${Math.max(0, volumeBottom - vy(bar.volume))}" fill="${color}" opacity=".72"/>`;
          }
          if (historical) {
            const indicator = currentView()?.indicators?.rows?.find((row) => short(row.time) === short(bar.time));
            if (finite(indicator?.volume_mean20_prior)) means.push([x(index), vy(indicator.volume_mean20_prior)]);
          }
        }
        output += `${line(means, palette.gold, 1.4)}<text x="${W - right + 9}" y="${volumeTop + 22}">${fmt(maximum / 1e8, 1)}</text><text x="${W - right + 9}" y="${volumeBottom}">0</text>`;
      }
      const ticks = new Set([state.start, state.end, state.lastIndex]);
      const stride = Math.max(1, Math.ceil(count / (W < 600 ? 4 : 9)));
      for (let index = state.start; index <= state.end; index += stride) ticks.add(index);
      const accepted = [];
      for (const index of [state.end, state.start, state.lastIndex, ...ticks]) if (index >= state.start && index <= state.end && accepted.every((other) => Math.abs(x(index) - x(other)) >= 57)) accepted.push(index);
      for (const index of accepted.sort((a, b) => a - b)) output += `<text x="${x(index)}" y="${bottom}" text-anchor="middle">${escape(state.dates[index].slice(5))}</text>`;
      if (state.hover !== null && state.hover >= state.start && state.hover <= state.end) {
        const xx = x(state.hover), row = state.hover < state.N ? state.B[state.hover] : state.F[state.hover - state.N];
        output += `<line x1="${xx}" y1="${top}" x2="${xx}" y2="${volumeBottom}" stroke="${palette.muted}" stroke-opacity=".65" stroke-dasharray="3 3"/>`;
        if (finite(row?.close)) output += `<circle cx="${xx}" cy="${y(row.close)}" r="4" fill="${palette.surface}" stroke="${state.hover < state.N ? palette.muted : palette.blue}" stroke-width="1.6"/>`;
      }
      svg.setAttribute("viewBox", `0 0 ${W} ${height}`);
      svg.innerHTML = output;
      state.geometry = context;
      $("stock-research-range-label").textContent = `${state.dates[state.start]} — ${state.dates[state.end]}`;
      $("stock-research-visible-range").textContent = `显示 ${count} / ${state.T} 根`;
      $("stock-research-pan").max = Math.max(0, state.T - count);
      $("stock-research-pan").value = state.start;
      $("stock-research-pan").disabled = count === state.T;
      $("stock-research-selection").hidden = !state.pinned;
      $("stock-research-selection-caption").textContent = state.pinned ? `固定日期 ${state.dates[state.hover]}${state.hover < state.N ? " · 历史实测" : " · 模型情景"}` : "";
      drawChips(context);
      scheduleMarkers();
    }

    function drawChips(context) {
      const host = $("stock-research-chip-profile");
      host.hidden = !state.layers.chips || state.frequency !== "日线";
      if (host.hidden) return;
      const index = state.hover ?? state.lastIndex;
      const bar = index < state.N ? state.B[index] : null;
      const profile = bar ? currentView()?.indicators?.chip_profiles?.[short(bar.time)] : null;
      if (!profile) {
        host.innerHTML = `<p>筹码估算</p><small>${bar ? "缺少该日有效分布" : "未来情景无筹码分布"}<br>非真实持仓成本</small>`;
        return;
      }
      const bins = profile.bins || [], maximum = Math.max(...bins.map((row) => row.mass_fraction), .001), height = svg.clientHeight;
      let content = `<svg viewBox="0 0 116 ${height}" aria-label="历史成交量加权筹码估算"><text x="4" y="17">筹码估算 · ${escape(short(profile.as_of).slice(5))}</text>`;
      for (const bin of bins) {
        const hi = Math.max(29, context.y(bin.price_high)), lo = Math.min(context.priceBottom, context.y(bin.price_low));
        if (lo <= hi) continue;
        const inside = bin.price_high >= profile.chip70_low && bin.price_low <= profile.chip70_high;
        content += `<rect x="4" y="${hi}" width="${108 * bin.mass_fraction / maximum}" height="${Math.max(1, lo - hi)}" fill="${inside ? palette.chips : palette.muted}" opacity="${inside ? .8 : .3}"/>`;
      }
      if (finite(profile.chip50)) content += `<line x1="0" x2="116" y1="${context.y(profile.chip50)}" y2="${context.y(profile.chip50)}" stroke="${palette.chips}" stroke-dasharray="3 3"/>`;
      host.innerHTML = `${content}</svg><small>120日 HLC/3 量权分布<br>非真实持仓成本</small>`;
    }

    function setWindow(count) {
      count = Math.max(1, Math.min(state.T, count));
      state.end = state.T - 1;
      state.start = state.T - count;
      state.hover = state.pinned = null;
      inspect(state.lastIndex);
      draw();
    }

    function zoom(factor, anchor = .7) {
      const old = state.end - state.start + 1;
      const count = Math.max(Math.min(state.T, 8), Math.min(state.T, Math.round(old * factor)));
      const center = state.start + old * anchor;
      state.start = Math.max(0, Math.min(state.T - count, Math.round(center - count * anchor)));
      state.end = state.start + count - 1;
      state.pinned = false;
      draw();
    }

    function indexAt(event) {
      const geometry = state.geometry, rect = svg.getBoundingClientRect();
      const xx = (event.clientX - rect.left) * geometry.W / rect.width;
      return Math.max(state.start, Math.min(state.end, Math.round((xx - geometry.left) / geometry.step + state.start - .5)));
    }

    function resetSelection() {
      state.pinned = false;
      state.hover = state.lastIndex;
      tooltip.hidden = true;
      inspect(state.lastIndex);
      draw();
    }

    function point(day, price = null) {
      const index = indexForDate(day), geometry = state.geometry;
      if (!Number.isInteger(index) || !geometry) return null;
      const bar = index < state.N ? state.B[index] : null;
      return { index, x: geometry.x(index), visible: index >= state.start && index <= state.end,
        priceY: finite(price) ? geometry.y(price) : null,
        highY: available(bar) ? geometry.y(bar.high) : null, lowY: available(bar) ? geometry.y(bar.low) : null,
        width: geometry.W - geometry.right, height: geometry.priceBottom, step: geometry.step };
    }

    function locate(day) {
      const index = indexForDate(day);
      if (!Number.isInteger(index)) return;
      const count = state.end - state.start + 1;
      if (index < state.start + 2 || index > state.end - 2) {
        state.start = Math.max(0, Math.min(state.T - count, Math.round(index - count / 2)));
        state.end = state.start + count - 1;
      }
      state.pinned = true;
      inspect(index);
      draw();
    }

    for (const input of document.querySelectorAll("[data-research-layer]")) input.addEventListener("change", () => {
      state.layers[input.dataset.researchLayer] = input.checked;
      draw();
    });
    for (const button of document.querySelectorAll("[data-stock-frequency]")) button.addEventListener("click", () => setFrequency(button.dataset.stockFrequency));
    for (const button of document.querySelectorAll("[data-stock-window]")) button.addEventListener("click", () => {
      for (const other of document.querySelectorAll("[data-stock-window]")) other.classList.toggle("active", other === button);
      setWindow(button.dataset.stockWindow === "all" ? state.T : Number(button.dataset.stockWindow) + state.H);
    });
    for (const button of document.querySelectorAll("[data-research-tab]")) button.addEventListener("click", () => {
      for (const other of document.querySelectorAll("[data-research-tab]")) {
        const selected = other === button;
        other.classList.toggle("active", selected);
        other.setAttribute("aria-selected", String(selected));
        $(other.dataset.researchTab).hidden = !selected;
      }
    });
    $("stock-research-event-family").addEventListener("change", renderSourceEvents);
    $("stock-research-pan").addEventListener("input", () => {
      const count = state.end - state.start + 1;
      state.start = Number($("stock-research-pan").value);
      state.end = state.start + count - 1;
      state.pinned = false;
      draw();
    });
    svg.addEventListener("wheel", (event) => {
      event.preventDefault();
      const geometry = state.geometry, rect = svg.getBoundingClientRect();
      zoom(event.deltaY > 0 ? 1.18 : .83, Math.max(0, Math.min(1, (event.clientX - rect.left - geometry.left) / (geometry.W - geometry.left - geometry.right))));
    }, { passive: false });
    svg.addEventListener("pointerdown", (event) => {
      state.drag = { x: event.clientX, start: state.start, end: state.end, moved: false };
      svg.setPointerCapture(event.pointerId);
    });
    svg.addEventListener("pointermove", (event) => {
      if (state.drag) {
        const offset = Math.round((state.drag.x - event.clientX) / state.geometry.step), count = state.drag.end - state.drag.start + 1;
        if (Math.abs(event.clientX - state.drag.x) > 5) state.drag.moved = true;
        state.start = Math.max(0, Math.min(state.T - count, state.drag.start + offset));
        state.end = state.start + count - 1;
        draw();
        return;
      }
      if (state.pinned) return;
      inspect(indexAt(event));
      draw();
      const geometry = state.geometry, rect = svg.getBoundingClientRect(), xx = event.clientX - rect.left, yy = event.clientY - rect.top;
      const hit = [...geometry.interaction].reverse().find((area) => xx >= area.x0 && xx <= area.x1 && yy >= area.y0 && yy <= area.y1);
      tooltip.hidden = !hit;
      if (hit) {
        tooltip.innerHTML = `${escape(hit.text)}<br><small>${escape(hit.detail)}</small>`;
        tooltip.style.left = `${Math.max(4, Math.min(xx + 12, geometry.W - 280))}px`;
        tooltip.style.top = `${Math.max(0, yy - 68)}px`;
      }
    });
    svg.addEventListener("pointerup", (event) => {
      if (state.drag && !state.drag.moved) {
        inspect(indexAt(event));
        state.pinned = !state.pinned;
      }
      state.drag = null;
      draw();
    });
    svg.addEventListener("pointercancel", () => { state.drag = null; });
    svg.addEventListener("pointerleave", () => {
      tooltip.hidden = true;
      if (!state.pinned && !state.drag) {
        state.hover = state.lastIndex;
        inspect(state.lastIndex);
        draw();
      }
    });
    svg.addEventListener("keydown", (event) => {
      if (["+", "=", "-", "Home", "ArrowLeft", "ArrowRight"].includes(event.key)) event.preventDefault();
      if (["+", "="].includes(event.key)) zoom(.8);
      else if (event.key === "-") zoom(1.25);
      else if (event.key === "Home") setWindow(state.T);
      else if (["ArrowLeft", "ArrowRight"].includes(event.key)) {
        const count = state.end - state.start + 1;
        state.start = Math.max(0, Math.min(state.T - count, state.start + (event.key === "ArrowLeft" ? -2 : 2)));
        state.end = state.start + count - 1;
        draw();
      }
    });
    function updateFullscreenControl() {
      const expanded = Boolean(document.fullscreenElement) || $("stock-chart-stage-workspace").classList.contains("stock-research-expanded");
      $("stock-research-fullscreen").textContent = expanded ? "退出全屏" : "全屏";
      $("stock-research-fullscreen").setAttribute("aria-pressed", String(expanded));
      document.body.classList.toggle("stock-research-expanded-open", expanded);
      draw();
    }

    function exitExpanded() {
      $("stock-chart-stage-workspace").classList.remove("stock-research-expanded");
      updateFullscreenControl();
    }

    $("stock-research-reset-selection").addEventListener("click", resetSelection);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !$("page-stock-chart").hidden) {
        resetSelection();
        if ($("stock-chart-stage-workspace").classList.contains("stock-research-expanded")) exitExpanded();
      }
    });
    $("stock-research-fullscreen").addEventListener("click", async () => {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if ($("stock-chart-stage-workspace").classList.contains("stock-research-expanded")) exitExpanded();
      else {
        try {
          if (!$("stock-chart-stage-workspace").requestFullscreen) throw new Error("Fullscreen API unavailable");
          await $("stock-chart-stage-workspace").requestFullscreen();
        } catch {
          $("stock-chart-stage-workspace").classList.add("stock-research-expanded");
          updateFullscreenControl();
        }
      }
    });
    document.addEventListener("fullscreenchange", updateFullscreenControl);
    const resizeObserver = new ResizeObserver(() => { if (state.data) draw(); });
    resizeObserver.observe(host);
    resizeObserver.observe(host.parentElement);

    return {
      mount,
      point,
      locate,
      fit: () => setWindow(state.T),
      resize: draw,
      compact: () => state.geometry ? state.geometry.step < 13 : false,
      exportData: () => state.view,
      state,
    };
  }

  window.KHQuantStockResearch = { create };
})();
