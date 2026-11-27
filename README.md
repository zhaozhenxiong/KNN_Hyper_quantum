# KNN Hyper Quantum

**v0.0.1** — GPU 加速的在线 KNN 与多模态专家融合，用于 A 股收盘后研究、滚动样本外回测和可核验组合账本。

## 核心策略

```text
hybrid_score =
    0.60 × KNN 校准 q50 的同日横截面百分位
    + 0.40 × 多模态综合校准 q50 的同日横截面百分位
```

- KNN 使用 `ADX`、`MFI`、成交量变化三维特征，按交叉状态选择因果成熟样本。
- T 日收盘产生信号，T+1 开盘模拟进入，T+6 开盘退出；标签扣除沪深300同期开盘收益。
- KNN 近邻距离、稳定平局排序和分位数热点支持 NumPy CPU 与 PyTorch CUDA float64。
- 多模态预测无效时由 KNN 100% 兜底，并保存实际权重和原因；KNN 无效时不生成混合候选。
- 混合分只用于横截面决策，不冒充新的收益分位数；双方原生 q10/q50/q90 分别保留。
- 生产回退只有在相同窗口、股票池、资金和费用口径下，三折双倍成本复合收益严格更高时才替换。

## 仓库内容

- `my_strategy/latent_quant/`：因果数据、在线 KNN、GPU 后端、多模态模型与组合执行。
- `my_strategy/research_workbench/`：混合研究、回放、健康诊断和本地 Web 工作台。
- `my_strategy/model_registry/`：封存证据、生产回退选择和发布读取。
- `my_strategy/tests/`：CPU/CUDA、融合、生产替换、回放和页面 API 回归测试。
- `docs/v0.0.1-validation.md`：首个版本的真实运行结果与限制。

本仓库不包含本地行情数据库、训练模型、生产注册表、缓存或回测工件。运行真实研究前需要按 `my_strategy/research_workbench/README.md` 准备并显式配置本地数据路径。

## 环境与安装

推荐 Python 3.12。CUDA 是可选运行后端；没有兼容 GPU 时可使用 CPU，显式请求 `cuda` 时初始化失败会报错。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\my_strategy\requirements.txt
pip install -r .\my_strategy\requirements-latent-quant.txt
```

检查混合研究命令和测试环境：

```powershell
python -m my_strategy.research_workbench.expert_hybrid_study --help
```

```powershell
python -m pytest -q .\my_strategy\tests\test_latent_quant_knn_backend.py .\my_strategy\tests\test_research_workbench_expert_hybrid_study.py .\my_strategy\tests\test_expert_production_fallback.py .\my_strategy\tests\test_research_workbench_expert_replay.py
```

## 本地工作台

服务入口：

```powershell
python -m my_strategy.research_workbench.cli serve --port 8124 --mode baseline --device auto
```

完整的本地数据、股票池、分类和执行参考参数见 [研究工作台说明](my_strategy/research_workbench/README.md)。服务默认只监听本机；页面加载不会自动训练、更新行情或创建回放。

## v0.0.1 验证摘要

- 真实 CUDA 全量运行覆盖 1,309,930 个预测股日，全部使用完整 60/40。
- 三个开发折的双倍成本复合收益为 +10.4721%；其中 oos_3 为 -14.5351%，历史复核为 -9.2066%。
- 连续历史模拟收益 +10.6017%，最大回撤 -27.7496%，现金、权益和下一交易日账本检查通过。
- CPU/CUDA 数值一致，但本次真实整折测试没有加速。
- 隔离聚焦回归 49 项通过；完整历史测试集仍有 1 项既有失败和 11 个收集错误，详见验证文档。

所有结果均为历史研究模拟，不是独立前瞻证明、收益承诺或投资建议。
