# KHQuant 研究工作台

本目录提供本地收盘后研究工作台，包含“多模态专家”“策略回测”“个股与策略”“数据与分类”四页。服务只监听 `127.0.0.1`，前端为本地原生 HTML/CSS/JavaScript，无构建步骤、CDN 或外部行情依赖。

## 启动

从仓库根目录运行，使用项目虚拟环境。当前预览配置是仓库根目录的 `.claude/launch.json`，名称为 `research-workbench`，端口为 **8126**。8124/8125 可作为其他检出或对照实例，不应互相覆盖进程。

独立启动时，先设置运行数据路径，再执行同一入口：

```powershell
$env:KHQUANT_DATA_ROOT = "E:\quant_strategy\my_strategy\data"
$env:KHQUANT_ARTIFACT_ROOT = "E:\quant_strategy\my_strategy\artifacts"
$env:KHQUANT_OUTPUT_ROOT = "E:\quant_strategy\my_strategy\data\processed\ai_agent_screen"
$env:KHQUANT_RAW_DB = "E:\quant_strategy\my_strategy\data\raw\khquant_raw.db"
$env:KHQUANT_PROCESSED_DB = "E:\quant_strategy\my_strategy\data\processed\warehouse\khquant.db"
$env:KHQUANT_MODEL_REGISTRY = "E:\quant_strategy\my_strategy\data\processed\model_registry\unified_model_registry.sqlite"
$env:KHQUANT_MODEL_REGISTRY_SNAPSHOT = "E:\quant_strategy\my_strategy\data\processed\model_registry\unified_model_registry.json"
E:\quant_strategy\.venv\Scripts\python.exe -B -m my_strategy.research_workbench.cli serve `
  --port 8126 --start 2024-01-02 --mode baseline --device auto `
  --price-basis adjusted --universe-policy main_star `
  --universe-path E:\quant_strategy\my_strategy\data\raw\workbench_universe\20261120\roster.csv `
  --classification-path E:\quant_strategy\my_strategy\data\raw\workbench_universe\20261120\industries\capco-industries.csv `
  --benchmark-path E:\quant_strategy\my_strategy\data\raw\workbench_universe\20261120\csi300.csv `
  --adjusted-path E:\quant_strategy\my_strategy\data\raw\workbench_universe\20261120\hfq-market\verified_market_hfq.parquet `
  --execution-reference-path E:\quant_strategy\my_strategy\data\raw\workbench_universe\20261120\hfq-market\sina_klc_market_reference.parquet
```

打开 `http://127.0.0.1:8126/`。hash 路由为 `#expert`、`#backtest`、`#stock-chart`、`#data`。

### 路径和参数口径

| 配置 | 作用 |
| --- | --- |
| `KHQUANT_DATA_ROOT` | 本地行情和处理数据根目录；缺省为当前检出的 `my_strategy/data`。 |
| `KHQUANT_ARTIFACT_ROOT` | 冻结研究、回放和状态文件根目录；运行记录位于其 `runs/` 下。 |
| `KHQUANT_OUTPUT_ROOT` | AI Agent 日频输出目录，保持 `data/processed/ai_agent_screen`。 |
| `KHQUANT_RAW_DB` / `KHQUANT_PROCESSED_DB` | 原始行情库和处理仓库的显式路径。 |
| `KHQUANT_MODEL_REGISTRY` / `..._SNAPSHOT` | 统一模型注册表及兼容快照，避免 worktree 误读另一检出的注册表。 |
| `--universe-policy main_star` | 使用交易所名录的沪深主板、科创板口径；此时必须提供 `--universe-path`。 |
| `--price-basis adjusted` | 研究回放使用复权研究输入；执行参考文件仍单独注入。 |
| `--start` / `--end` | 默认研究区间；省略时从本地最新交易日推导，个股图另有自己的三个月窗口。 |

路径可按机器调整，但不要把某台机器的绝对路径写进业务代码。页面加载、切换导航和查看图表不会创建研究或回放任务；只有显式提交“启动试验”“运行回测”或个股回放表单才会创建任务和运行工件。旧版启动 API 已退役，不能用于绕过当前研究流程。

## 页面与模型状态

- “多模态专家”保留研究候选、分折验收、下载和前瞻状态。`score_health` 是附加的只读历史诊断，不能自动晋升或降级候选，也不能代替历史数值、时点或前瞻验收。
- 入场分数是校准后的 q50，规则为严格 `score > min_score`；当前复核的 `min_score=0`。原始分数、正分股日数和校准偏移只用于诊断，不能拿评估期结果反调门槛。
- “策略回测”在显式提交后展示同一交易日的扣费净值；合理现金期不是页面故障。
- “数据与分类”显示本地行情覆盖、配置股票池和缺失分类。“更新数据”按钮只在显式点击后启动**原始仓已有全部股票**的日线更新，目标日按已验证日历和 Asia/Shanghai 收盘时刻确定；页面加载、导航和 GET 不联网或写入。任务显示处理进度、成功／失败／跳过和更新前后覆盖，部分失败标为“部分完成”。
- 原始行情更新不会扩展 `workbench_universe/20261120` 冻结研究快照或 `HISTORICAL_END`，也不会自动训练、回测、生成候选或更新模型。数据页的原始行情截止可以晚于统一研究截止，两者不能混称为同一数据版本。
- 日更主源使用腾讯配对的 `qfqday`／`day` 前复权与未复权日线，适配思路参考 [Lake_equity](https://github.com/zhaozhenxiong/Lake_equity) 的腾讯数据源；沪深主板和创业板成交量从手换算为股，科创板保留股单位，成交额从万元换算为元。AkShare 备用调用使用当前 `stock_zh_a_hist_em` 接口并避开本机系统代理；所有来源仍经过同一清洗、合并、逐股票事务和目标日验收。

### 2026-11-24 原始行情更新验收

- 页面显式启动的主任务 `20261124-152552-184772-20261123-data-update-19f8ef2` 固定目标日为 2026-11-23，范围是原始仓已有全部 4,036 只股票。主任务结果为 `partial`：3,857 只成功、179 只失败，全部 3,857 只成功股票均到达目标日。
- 对腾讯前复权响应只返回合法 `day` 的 80 只股票做同目标日补跑，update id 为 `20261124-171741-769611-20261123-data-update-retry-19f8ef2`；63 只成功、17 只失败。补跑复用同一 `DailyUpdater.update_pool`、质量门和事务，没有通过 GET 或页面导航触发。
- 合并补跑后的独立 SQLite 审计确认 **3,920／4,036** 只到达 2026-11-23；冻结名录 3,813 只中覆盖 **3,786** 只，缺 27 只。未达目标日的 116 只包括 99 只各来源无新行情、17 只未通过既有最少 100 行历史质量门；没有降低质量门或把部分覆盖称为全量成功。
- 目标日无效交易行、重复股票／日期及晚于 2026-11-23 的行均为 0。主任务结果面板仍保留该次 3,857／179 的历史结果，最终 3,920 只覆盖来自补跑后的独立审计，两者不能混为一次任务结果。详细证据见 `.claude/qa/data-update-20261123-audit.json` 和 `data-update-20261123-retry.json`。
- 更新后重新核验统一模型 `d722da6d5ac4824c54401b9fc9d5aba621fa8c7e67b155577b3aae4956125f72` 和空候选证据，SHA 均与更新前一致；没有自动训练、回测、生成新候选或晋升模型。

## 统一研究、回测与次日观察名单

一次研究运行对应一个统一候选版本；折模型、分位数头和专家臂是内部验证组件，不要求使用者逐折选择模型。操作顺序如下：

1. 在“多模态专家”选择范围并点击“预览口径”。预检会显示真实股票池、交易日、成熟训练样本和阻塞原因；`blocked` 或配置已变更时不能启动研究。`main_star` 明确指**沪深主板＋科创板**，不是全 A。`shsz` 还要求创业板名录及对齐行情证据，缺失时不会创建冻结快照。
2. 预检通过后点击“启动试验”。同一运行依次完成滚动样本外研究、最终拟合和统一模型封存；刷新页面只恢复原任务，不重复提交。原生专家的验证集候选先排除“单树且中位分数塌缩”的退化项，再按 pinball 选择。国内臂全部退化时任务失败；海外臂全部退化时只在最终未来信号模型中弃权，并用同一 train 内 OOF 重选剩余权重，历史折门不改写。任何参与最终融合的塌缩臂都不能封存。最终拟合使用独立 train／validation／calibration 时段，输出 `models/forward/model.json`、`reports/expert_final_fit.json` 和 `reports/unified_expert.json`。研究报告先封存并计算 SHA，模型和统一清单绑定该 SHA，避免循环身份。仅在最终拟合阶段失败且研究报告、全部预测及哈希闭包完整的运行，可供新运行复用历史预测；其他失败不能复用。
3. 在研究结果点击“回测此统一候选”。链接携带 `source_run` 和可用时的 `model_version`，回测提交固定该版本，不在提交后换成另一“当前模型”。历史净值来自 walk-forward 折预测，最终拟合只用于未来信号，不能在其训练区间冒充样本外结果。
4. 已完成且包含全池账户的回测可显式生成“次日研究观察名单”。服务端创建带 `candidate_of` 的新回放，复用同一冻结预测和严格组合／执行约束，写出下一交易日开盘预申报，再投影为候选表；不会重新排名、制造成交或自动下单。
5. 候选行进入“个股与策略”时继续携带同一研究、信号日和回放上下文。个股页默认只读真实行情与派生结构，只有点击“运行单股回放”才提交任务。

候选证据位于回放目录的 `reports/expert_candidates.json`、`backtests/custom/candidates.csv`、`candidates.parquet`、`full_pool_pending.parquet` 和 `full_pool_predictions.parquet`。报告保留信号日、目标交易日、模型版本、父回测、全部订单、排除原因及来源 SHA。`filled_shares` 必须为零、`execution_price` 必须为空。空名单是合法结果，可能来自非调仓、排名、现金、持仓或交易约束。

这类名单固定标记为 `research_only`、`evidence_mode=research_observation`、`score_basis=walk_forward_replay`，**不是最终拟合模型的正式前瞻推荐**。目标交易日已过或本地行情未更新时另标 `stale`。点时核验为 `False` 的最终拟合可保留为研究工件，但现有 forward lock 会拒绝；正式 champion 仍需历史数值门、点时核验、60 个唯一前瞻交易日及双倍成本、风险和 capture-chain 验收。

### 2026-11-24 真实统一研究验收

- 研究运行 `20261124-011833-455085-20261120-expert-research-19f8ef2` 已完成，范围为沪深主板＋科创板 3,902 只，输入截至 2026-11-20。统一版本为 `d722da6d5ac4824c54401b9fc9d5aba621fa8c7e67b155577b3aae4956125f72`。`verify_final_fit()` 已重验报告、配置、模型和原生文件哈希闭包；64 行精确回放通过，国内 GPU 与 CPU 最大绝对误差为 `2.735760062932968e-09`。
- 最终验证中的海外 GPU q50 为单树、8 个不同取值，因此只在最终未来信号包中弃权；同一 train 内 OOF 重新选择后的权重为国内 GPU 75%＋规则专家 25%。最终模型健康检查和有序有限分位数检查通过，但历史数值门、历史分数健康门和点时核验仍未通过，状态保持 `research_only`，不是 champion。
- 指定版本全池回放 `20261124-013317-729826-20261120-expert-replay-19f8ef2` 已完成，覆盖 420 个交易日。全池常规成本累计收益 5.78%，最大回撤 -14.78%，日均换手 8.91%，拒单率 6.76%，469 笔模拟成交，费用 21,853.30 元；期末有 3 个退市滞留持仓，最后核验估值 105,650 元，将其置零后的收益为 -4.78%。这些是历史模拟结果，不是收益承诺。
- 候选运行 `20261124-013457-648564-20261120-expert-replay-19f8ef2` 已生成信号日 2026-11-20、目标交易日 2026-11-23 的封存证据，状态为 `research_only / stale`。3,790 只有效预测股票的校准 q50 全部不大于 0，最高为 -0.00322322078505706，因此没有形成有效买入预申报；候选、订单和 pending 均为空，没有成交或执行价。该结果不是当前推荐。
- 候选日期解析复用 `latent_api._calendar_path()`，可读取共享的已验证前瞻日历，不再因 worktree 缺少另一份默认日历而误判没有下一交易日。修正前后历史账本指标一致。
- 使用上述真实研究和候选运行执行 `KHQUANT_QA_UNIFIED=1` 浏览器验收，34 项检查通过；1920／1100／760／650／375px 均无横向页面溢出。真实空名单、来源绑定、过期状态和零成交约束已验收；由于名单没有股票行，本轮没有真实候选行进入个股页的端到端样本。

## 2025-10-15 后无成交与期末持仓收益

当前页面复核的运行是 `20261124-212836-758317-20261124-expert-replay-19f8ef2`，不是下文最后成交为 2025-11-05 的旧问题回放。该运行有 420 个共同交易日、469 笔成交和 524 条订单；`expert_backtest_activity.py` 直接取可见成交账本的最大日期，**2025-10-15 是真实最后成交日**。

- 最后成交之后有 268 个回放交易日、268 个锁定分数覆盖日和 53 个计划调仓信号日。调仓日严格 `score > 0` 的达标数合计为 0，因此后续买入订单为 0；非调仓日另有 64 个正分股日，但原策略不在这些日子交易。
- 其后 7 条零成交订单全部是 `603388.SH` 卖出：3 条因开盘跌停拒单，4 条因停牌在下单前拦截。最后成交当天另有 1 条跌停卖出拒单，不计入严格 `date > last_trade_date` 的 7 条。“查看这 7 条”通过 `after_last=1` 清除旧月份等筛选，只读取得精确记录，全运行汇总不随月份筛选改变。
- 诊断统计来自原研究锁定的配置、预测及哈希证明；证据不可核验时明确降级，不能从“没有订单”反推分数原因。展示修复不改变 5 日调仓、分数门槛、成本、风险约束或旧净值。

`GET /api/expert/replays/{run_id}/holdings?account=full_pool|watchlist` 只读该回测授权的 daily、positions、trades 账本，不读取更新后的行情覆盖历史估值。`holding_cost_after` 是每股平均成本，剩余成本为它乘期末股数；`cash_flow` 已含费用，滑点已体现在成交价，收益计算不再次扣费或扣滑点。逐股已实现收益为累计现金流加剩余成本，未实现收益为期末市值减剩余成本，累计贡献等于累计现金流加期末市值。最后成交股数、成本或现金流不能对齐时字段留空并降级，不填零。

当前运行期末没有在市持仓，只有 `600070.SH`、`600811.SH`、`603388.SH` 三只退市滞留持仓：剩余成本 **306,275.42 元**，历史滞留估值 **105,650.00 元**，未实现收益 **-200,625.42 元（-65.50%）**，此前完整交易产生的已实现收益 **-574.33 元**，累计收益贡献 **-201,199.75 元**；退市估值置零后的累计贡献为 **-306,849.75 元**。期末市值、在市市值和退市滞留市值均与日账本零差异对账。页面保留真实估值日 2025-04-10、2025-04-14、2025-11-10，并明确提示滞留估值不等于可变现资产；期末无在市持仓时不生成买卖建议，历史研究提示不构成当前投资建议。

聚焦测试（含上述真实账本回归）为 **41 passed**。真实鼠标浏览器验收为 **47 项通过**，覆盖精确 7 条筛选、持仓收益、上下文链接、授权下载和 1920／1100／760／650／375px；全部请求均为 GET，无控制台或 HTTP 错误，研究、预测、回放和账本哈希验收前后不变。证据见 `.claude/qa/backtest-activity-holdings-qa.json` 与 `backtest-holdings-*-full.png`。

## 个股连续研究图

实现位于 `stock_chart_research.py`、`stock_chart_api.py`、`web/stock-research-chart.js` 和 `web/stock-chart.js`。页面壳、导航、卡片、表格和表单继续沿用工作台设计；综合图参考 KCPredict 的 SVG 连续时间轴，来源和许可见 `web/NOTICE-KCPredict.txt`。

### 历史事实

- 默认从实际行情截止日向前取 **三个日历月**，不是固定 60 或 90 根。
- 只接受日期严格递增且唯一、OHLC 为正且几何有效的本地**未复权**日线；显式 `is_closed=false` 会被拒绝，省略该字段时按已闭合输入处理。成交量若提供，必须为有限非负数值。
- 停牌、退市等无价格日期可以保留 calendar slot，但不能生成蜡烛、指标或“真实价格”。
- 日／周／月、分型、确认／形成中笔、中枢、背驰、来源 B/S、Wyckoff、ICT/FVG、MA5/10/20/60、BOLL、成交量和此前 20 日均量均由同一份历史派生。
- 筹码是最近 120 根闭合日线的 HLC/3 成交量加权估算，70% 区间对应 15%–85% 分位；**不是股东真实持仓成本**。
- 结构、Wyckoff 和 ICT 是机械派生研究参考，**不是买卖确认，也不代表实际成交**。

### 操作与证据

支持图层开关、滚轮缩放、拖动和 range 平移、悬浮读数、点击固定、Esc 取消固定、全屏、历史／未来／审计表、来源筛选和 JSON 证据导出。窄屏下表格允许在自身容器内滚动，页面本身不应横向溢出。

真实信号、模拟成交、部分成交和拒单来自原回放账本，不由研究图层推断。`single_stock` 与 `full_pool` 保留独立轨道；全池轨道只在显式开启叠加时显示。账本定位只移动可见时间窗，不修改账本或行情。

**页面加载不会自动推理。** 当前回放没有预测工件时显示“预测未执行”，不补未来日期、不画路径。导出文件同时保留真实行情、派生研究、冻结预测状态和账本，重新导出不会重新采样。

## 冻结预测工件契约

图表优先读取结果中的 `research_prediction`；否则在对应回放运行目录下只读查找：

```text
reports/stock_research_prediction.json
```

**当前模块只提供读取与校验，不提供预测生成命令。** 预测应由独立研究任务在页面外执行，将符合下述契约的结果封存到对应运行目录后再打开图表；不要通过手写未经复算的 JSON 冒充冻结预测。因此，没有独立工件时，`not_executed` 是正常且完整的状态。

接受的预测 schema 为 `khquant_stock_prediction_v1`、`kronos_forecast_v1`，以及带 `prediction` 根的 `kcpredict_research_v1`。完成预测必须为 `status="completed"`。适配器输出 `khquant_stock_research_v1`，不覆盖原 `bars`、`strategies` 或账本字段。

必要字段及校验：

| 字段 | 契约 |
| --- | --- |
| `symbol`、`input.symbol` | 必须与图表标的一致。 |
| `as_of`、`input.history_end` | 若提供，必须等于最后真实闭合日线日期。 |
| `input.last_close` | 若提供，必须等于最后真实收盘价。 |
| `input.horizon` 或 `identity.parameters.horizon` | 必须为整数 5。 |
| `forecast` | 恰好五条，日期唯一、递增且全部在历史之后；P10 ≤ 中位数 ≤ P90，OHLC 几何有效。 |
| `identity.future_dates` | 必须与 `forecast` 的五个日期逐项一致；页面不自行生成交易日。 |
| `samples` | 若提供，每条路径为 5 日，每日依次是 `[open, high, low, close, volume, amount]` 六个有限数值。 |
| `quality.valid_mask` | 非空布尔数组；有原始样本时，必须等于按 OHLC/成交量/成交额重新计算的有效性。 |
| `quality.requested_samples` / `valid_samples` | 分别等于掩码长度和有效样本数。 |
| `quality.valid_fraction` | 必须等于有效样本数除以请求样本数；`minimum_valid_fraction` 省略时为 0.8，提供时必须位于 `(0, 1]`。 |
| `quality.minimum_valid_samples` | 最少有效样本数，省略时为 8。 |
| `quality.prediction_usable` | 必须是布尔值；为 `true` 时必须满足最小样本数和有效率门槛，为 `false` 时即使数值门槛通过也仍显示 `fail`。 |

有原始样本时，只用有效路径重新计算收盘 P10／中位数／P90；与 `forecast` 不一致即拒绝。直接构建研究数据时，坏工件会明确抛出校验错误；API 附加层则将预测降级为 `unavailable`，保留真实历史和派生结构，并附加警告，不会把坏数据静默变成“未执行”。

### 四种页面状态

| 状态 | 含义 | 路径显示 |
| --- | --- | --- |
| `pass` | 冻结工件通过已声明的数值质量门。 | 显示五个冻结交易日、采样中位数和 P10–P90；仍不是收益认证。 |
| `fail` | 有预测结果但未通过质量门。 | 仍显示五日中位数、P10–P90、未来表和可选有效样本路径，但醒目标为质量未通过，不升级为可用预测。 |
| `unavailable` | 模型、输入或校验不可用。 | 不显示未来路径，保留真实历史、派生结构与原因。 |
| `not_executed` | 当前回放没有已封存预测，或明确未请求。 | 不补日期、不显示路径，也不触发推理。 |

输入状态 `not_requested` 归一为 `not_executed`。质量通过只说明工件满足预测质量门，不代表策略收益、交易资格或 ICT 确认。

## 2025-11-05 后无成交与净值平线

固定问题回放 `20261123-143024-091800-20261120-expert-replay-9920c41` 的只读回归结论：

- 2025 年 10 月有 8 笔 SELL、3 笔 REDUCE；最后成交日为 **2025-11-05**，不是 10 月后立即断更。
- daily 持续到 **2026-11-20**，日期严格递增且唯一；最后净值变化仍是 2025-11-05，之后 253 个交易日无成交、无费用，现金、市值和净值不变，基准继续变化。
- 期末净值为现金 **1,171,853.488111641** 加退市滞留估值 **840**，合计 **1,172,693.488111641**。滞留项为 `600070.SH` 2000 股，以 2025-04-10 最后可得价 0.42 估值。
- 预测文件覆盖到 2026-11-20。最后成交后仍有 64 个严格正 q50 **股日**，这不是 64 个合格调仓信号，也不等于应该成交；调仓日、排名、组合约束和交易状态仍生效。
- 分折诊断发现 oos_2 的海外 GPU q50 仅 12 个不同取值、融合权重 25%；oos_3 还存在单树及融合分数塌缩警告。这些证据用于历史复核，不改写原归档或发布状态。

因此，平线来自清仓后的现金加固定退市估值，不是行情截止或前端曲线停滞。不得通过降低门槛、伪造订单、删除合理空仓或修改净值曲线来“修复”。若后续研究产生新候选，仍须独立通过既有验收和发布流程。

## 验证

在仓库根目录运行。项目虚拟环境包含 pytest；系统默认 Python 不一定包含测试依赖。

```powershell
$tests = Get-ChildItem my_strategy/tests/test_research_workbench_*.py, my_strategy/tests/test_latent_quant_*.py, my_strategy/tests/test_expert_*.py
E:\quant_strategy\.venv\Scripts\python.exe -m pytest $tests.FullName -q
E:\quant_strategy\.venv\Scripts\python.exe -m pytest my_strategy/tests/test_research_workbench_stock_chart_research.py my_strategy/tests/test_research_workbench_stock_chart_api.py -q
E:\quant_strategy\.venv\Scripts\python.exe -m compileall -q my_strategy/research_workbench my_strategy/model_registry
node --check my_strategy/research_workbench/web/app.js
node --check my_strategy/research_workbench/web/stock-research-chart.js
node --check my_strategy/research_workbench/web/stock-chart.js
node --check my_strategy/research_workbench/web/strategy-backtest.js
```

加入活跃度诊断、期末持仓收益、显式原始仓数据更新、当前 AkShare 接口兼容和腾讯主源后的完整 workbench/latent/final-fit/candidate/artifact 目标回归为 **214 passed**；使用共享工件路径时，真实历史回放回归也全部执行，没有跳过。该命令只收集目标测试文件，不能据此宣称全仓测试通过。真实浏览器 QA 使用 Node 24 原生 WebSocket/CDP，不要求额外安装 Playwright。运行前必须先启动 8126 预览，并安装 Google Chrome；脚本当前使用 `C:/Program Files/Google/Chrome/Application/chrome.exe`，其他安装位置需调整 QA 脚本中的浏览器路径。验证命令如下：

```powershell
node .claude/qa/stock_chart_qa.mjs
$env:KHQUANT_QA_REPLAY = "1"; node .claude/qa/stock_chart_qa.mjs
$env:KHQUANT_QA_PAGES = "1"; node .claude/qa/stock_chart_qa.mjs
node .claude/qa/backtest_activity_holdings_qa.mjs
E:\quant_strategy\.venv\Scripts\python.exe .claude/qa/verify_replay_regression.py
```

默认浏览器验收为 22 项，显式真实回放为 34 项，四页视觉导航为 28 项，回测活跃度与持仓收益为 47 项。覆盖 1920、1100、760、650、375px，并将横向页面溢出、控制台／HTTP 错误和外部网络依赖列为失败条件。显式回放模式会创建真实研究运行，不属于只读导航；活跃度与持仓收益脚本只读现有运行。

`pass/fail/unavailable` 由适配器契约覆盖：`fail` 也断言保留五日预测和有效样本，坏预测在 API 边界降级时保留历史；这些路径没有真实冻结工件的浏览器端到端覆盖。当前真实页面端到端验收的预测状态是 `not_executed`，不能把契约测试描述成真实预测工件或浏览器渲染验收。
