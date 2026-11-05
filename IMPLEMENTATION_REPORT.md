# v4 实现与本机验证记录

日期：2026 年 11 月 5 日。代码仓库：`F:\MoDe_ECI`。

v4.1 的可训练专家主线及单轮预测图扩展已经实现。本机完成软件验证，未执行完整 ESC 专家训练、五折测试或正式研究评测。

## 实现范围

- 固定 ESC 扩展标签和文档级划分，兼容 v3 的解析数据；三类标签与方向评测保持一致。
- 无 gold 的目标相关规则分库和词表对照；D1 至 D4 各有独立专家参数。
- MPNet 四库、统一库、R0 至 R3、2/3-demo 配额、同文档排除、难负例审核和缺失记录。
- BERT 事件表示 MLP、BERT/RoBERTa MLM、BART 关系 token 和角色槽位；base/large 配置。
- C/S/M 视角、确定性共同预算裁剪，以及训练和推理一致的 D 示例输入。
- 文档级折外专家预测、query-only BERT 门控、G1/G2/G3、全错样本处理、门控内部选轮与全量重训。
- 单轮预测图 G；种子图的嵌套交叉拟合；删除查询自身双向边；图预算裁剪、无证据屏蔽及扰乱图诊断。
- 概率与完整输入日志、错误案例、完整覆盖检查、分层评测、互补性诊断、校准工具、原论文判决、bootstrap 和多种子汇总。
- 配置/代码/数据/索引/模型资产/训练 checkpoint 指纹；完成阶段复用；服务器逐阶段和五折运行脚本。

## 本机环境

使用已有的 `F:\MoDE_ECI_research\.venv\Scripts\python.exe`，通过本次命令进程的 `PYTHONPATH=F:\MoDe_ECI\src` 指向新代码。没有将新项目安装进旧项目环境。

| 依赖 | 验证版本 |
| --- | --- |
| Python | 3.12.14 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.57.6 |
| Sentence Transformers | 3.4.1 |
| spaCy | 3.8.16，en_core_web_sm |
| NumPy | 2.5.4 |
| scikit-learn | 1.9.1 |
| pytest | 9.1.1 |

微型模型流程在 CPU 上运行。服务器需按驱动安装 CUDA wheel，并保存实际依赖版本。

## 已执行的检查

| 检查 | 结果与证据 |
| --- | --- |
| pytest | **32 项通过**；`runs/pytest_delivery.log` |
| 无图端到端 | BERT MLM 与 BART 槽位七专家，内部折外预测，G1/G2/G3 选择及重训，保存加载、完成阶段重复执行、完整预测覆盖；`runs/verification_delivery/verification.json` |
| 基础输出头 | BERT 事件 MLP 和 BART 关系 token 训练、保存加载、前向与预测通过；同一验收摘要 |
| 预测图端到端 | 微型 S+G 系统完成嵌套种子交叉拟合、G 训练、G3 和预测，G 覆盖非零且总体预测覆盖 100%；同一验收摘要 |
| 后处理与诊断 | no_demo、标签扰乱、top-2、去 M、图扰乱、原论文判决、独立温度校准、汇总与 bootstrap 均通过；`runs/verification_delivery/reporting_verification.json` |
| 真实本地资产 | 真实 MPNet 编码与归一化；三篇真实训练文档的 696 对小索引；三条查询的检索与真实 BERT 前向，输出形状 3×3；`runs/local_assets/verification.json` |
| 真实数据与 CLI | 导入 258 篇文档、70,579 对；fold 0 计划生成成功；`runs/esc_plan.json` |
| 打包 | 成功构建 `mode_eci_v4-0.4.1-py3-none-any.whl`，位于 `runs/wheels/` |
| 语法和差异 | compileall 与 `git diff --check` 通过 |

单元测试覆盖 gold 查询隔离、原始/扩展方向解析、句法作用范围、否定、同文档检索排除、缺失 demo 配额、R3 独立结构召回、事件不连续/重叠标记、共同预算一致性、固定无 gold decoder、梯度、三类归一化、全错门控、方向反转计数、缺失预测拒绝、折外泄漏拒绝、图边排除和 checkpoint 修改检测。

最终端到端验收对应源码指纹：

`6c48191083e7b40500bad1cd94acc03559175cf02c4292a4a65018242a2e6c8d`

所有合成数据、微型模型和验收指标均标记 `test_only`。真实资产脚本也标记软件验证，不将其输出解释为研究成绩。

## 服务器阶段仍需完成

1. 审核训练侧规则与难负例，冻结词表、解析器版本、检索策略和 CSV。未提供 approved 审核记录时，R2 使用普通近邻负例并记录原因。
2. 在真实开发数据上完成单专家、等权融合、门控目标和检索消融，检查专家共同错误及分数分布迁移。
3. 验证实际 GPU 上的 batch、混合精度、训练耗时和 BART large 峰值显存；本机未加载 BART base/large 或 RoBERTa 的完整预训练权重训练。
4. 冻结方案后运行五折三种子与完整统计分析。图阈值、门控参数和解码阈值只在训练侧或开发阶段选择。

当前阶段恢复以完整训练任务为单位，不恢复半轮优化器；不要并发写同一训练任务。独立温度工具未自动并入主流程的每个内部组。共享 backbone、自由解释、自然词 verbalizer、多轮图更新、GNN 及额外主题外协议不在默认实现中。
