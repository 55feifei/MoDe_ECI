# MoDE ECI v4 实验工程

本仓库实现 v4.1 的可训练多专家方案：规则选择四类 demo 分支之一，MPNet 检索训练示例，BERT 或 BART 专家输出三类分布，独立 BERT 门控融合概率。正式训练和评测在服务器执行；本机验收使用微型随机模型及少量真实资产前向，不提供研究成绩。

## 已实现的方法

| 模块 | 实现 |
| --- | --- |
| 数据 | 固定版本 ESC 与扩展关系解析；文档级五折和 topic 37、41 开发集；可导入 v3 数据 |
| 分库 | D1 有线索句内、D2 无线索句内、D3 有线索句间、D4 无线索句间；目标依存范围规则与词表对照 |
| 检索 | 四库或统一库；R0 随机、R1 MPNet、R2 审核难负例、R3 加跨主题结构召回；2 或 3 个 demo |
| 专家 | BERT 事件表示加 MLP、BERT MLM、BART 关系 token、BART base/large 角色槽位、RoBERTa MLM |
| 融合 | 等权、单专家、query-only BERT 门控；G1 融合损失、G2 正确专家监督、G3 联合损失 |
| 图扩展 | 单轮预测图文本专家；嵌套交叉拟合种子图；移除查询自身双向边；无证据时屏蔽 G |
| 评测 | 方向/存在性指标、NLL、Brier、可靠性、专家共同错误、配对文档 bootstrap、五折多种子汇总 |

无图系统有 7 份独立专家参数，每条查询激活一个 D 及 C、S、M，共融合 4 个专家。新字段叫 `demo_cell`；无线索池包含 NONE，与 v3 限定因果正例的 `Implicit` 路由标签不同。方法与产物说明见 [EXPERIMENTS.md](EXPERIMENTS.md)，本机验证记录见 [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md)。

## 安装

Python 3.11+。先根据服务器驱动安装合适的 PyTorch CUDA wheel，再安装项目。Linux 示例：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m spacy download en_core_web_sm
python -m mode_eci --config configs/bert_mlm.yaml doctor
python scripts/download_models.py --config configs/bert_mlm.yaml
```

Windows 解释器为 `.venv\Scripts\python.exe`。命令从仓库根目录执行，也可使用安装后的 `mode-eci`。不需要 LLM API 或密钥。模型 revision 已固定；`name` 可改为本地目录，运行时会记录实际模型和 tokenizer 指纹。

`training.precision` 支持 fp32/fp16/bf16。默认微批量 8、累积 4，有效 batch 32；large 配置使用 2 × 16 并开启梯度检查点。按服务器显存调整微批量；本机微型模型验收不代表 large 显存验收。MPNet 默认在 CPU 编码，可将 `retrieval.device` 改为 `cuda`。

## 数据

导入已有解析结果，保留划分：

```powershell
python -m mode_eci --config configs/bert_mlm.yaml import-data --source F:\MoDE_ECI_research\data\esc
```

或从固定 commit 重新解析：

```bash
python -m mode_eci --config configs/bert_mlm.yaml prepare-data --download
```

输出 `documents.jsonl/pairs.jsonl/splits.json/statistics.json`。标签顺序固定为 `NONE / E1_TO_E2 / E2_TO_E1`；`Corpus.query()` 不携带 gold。正式开发和测试保留全部候选及自然类别比例。

## 单专家比较

四种输出头的配置为 `bert_cls.yaml`、`bert_mlm.yaml`、`bart_relation.yaml`、`bart_slots.yaml`；另有 `bart_large.yaml` 和 `roberta_mlm.yaml`。先用 S 视角比较 B0 至 B3：

```bash
python scripts/create_config.py --base configs/bart_slots.yaml --output configs/bart_slots_s.yaml --branches S
python -m mode_eci --config configs/bart_slots_s.yaml --fold 0 train-experts
python -m mode_eci --config configs/bart_slots_s.yaml --fold 0 predict --split dev --fusion S
```

仅训练 S 时，应使用仅含 S 的配置，避免完整预测要求其他专家的 checkpoint。BERT/BART 默认共同用固定 BERT 与 BART tokenizer 检查 512 token，并按同一陈述式预算裁剪原文。相同数据、种子与检索设置产生同一 demo 清单。

## 多专家与门控

一次完成外层折的开发流程：

```bash
python -m mode_eci --config configs/bert_mlm.yaml --fold 0 run --split dev
python -m mode_eci --config configs/bart_slots.yaml --fold 0 run --split dev
```

也可逐阶段运行：

```bash
python -m mode_eci --config configs/bart_slots.yaml --fold 0 plan
python -m mode_eci --config configs/bart_slots.yaml --fold 0 audit-rules --sample 200
python -m mode_eci --config configs/bart_slots.yaml --fold 0 build-index --scope inner-0
python -m mode_eci --config configs/bart_slots.yaml --fold 0 train-experts --scope inner-0
python -m mode_eci --config configs/bart_slots.yaml --fold 0 oof
python -m mode_eci --config configs/bart_slots.yaml --fold 0 train-experts --scope final
python -m mode_eci --config configs/bart_slots.yaml --fold 0 train-gate --objective G1
python -m mode_eci --config configs/bart_slots.yaml --fold 0 train-gate --objective G2
python -m mode_eci --config configs/bart_slots.yaml --fold 0 train-gate --objective G3
python -m mode_eci --config configs/bart_slots.yaml --fold 0 predict --split dev --fusion equal
python -m mode_eci --config configs/bart_slots.yaml --fold 0 predict --split dev --fusion learned --objective G3
```

`oof` 自动完成三个内部组及缺少的专家任务。内部留出文档不参与对应专家、S 预热或 demo 索引。专家用固定开发集的方向 F1、再按 NLL 选择 checkpoint；门控在折外记录的独立文档组选择轮次，再用全部折外记录重训。开发集已用于模型选择，应如实报告。

每个无图系统每折每种子有 `3 × 7 + 7 = 28` 个专家训练任务；三种门控共享它们。完成阶段自动复用；中断任务从固定种子重跑，尚不恢复半轮优化器状态。代码、数据、模型或专家配置变动时需新实验 ID，避免混用。不要让两个进程同时写同一 scope/branch。

## 审核与消融

各 scope 的 `index/negative_review.csv` 可填写 approved/rejected/ambiguous/unreviewed 和依据，再通过 `retrieval.review_file` 创建新配置、重建索引。R2 只优先选择语义相近且 approved 的负例。无审核表时会记录使用普通近邻负例，不能据此宣称难负例收益。

```bash
python scripts/create_config.py --base configs/bart_slots.yaml --output configs/bart_r3.yaml --strategy R3 --review-file artifacts/reviews/approved.csv
python scripts/create_config.py --base configs/bart_slots.yaml --output configs/bart_single_pool.yaml --pool single
python scripts/create_config.py --base configs/bart_slots.yaml --output configs/bart_random.yaml --strategy R0
python scripts/create_config.py --base configs/bart_slots.yaml --output configs/bart_lexical.yaml --rule lexical
python scripts/create_config.py --base configs/bart_slots.yaml --output configs/bart_no_m.yaml --branches D1 D2 D3 D4 C S
```

检索策略、分库规则或专家集合改变后，应重训相关专家和门控。`predict --demo-variant no_demo/shuffle_labels`、`--drop M`、`--top-k 1/2` 是冻结模型的一致性诊断；`--drop` 只支持等权融合。它们不能代替重新训练的消融。

门控温度和辅助损失系数可修改配置的 `gate` 节。各候选按参数指纹保存，可复用同一组专家和折外预测。

## 五折多种子

完成开发配置选择后，冻结配置：

```bash
python scripts/run_server.py --config configs/bart_slots.yaml --seeds 209 42 3407 --folds 0 1 2 3 4 --stage development
python scripts/run_server.py --config configs/bart_slots.yaml --seeds 209 42 3407 --folds 0 1 2 3 4 --stage test
```

测试阶段只使用已完成产物预测。`summarize --predictions <各折文件列表> --output runs/summary.json` 按种子拼接五折，检查每条非开发样本恰好一次；保留逐折成绩和种子均值/标准差。`compare --first <预测文件> --second <预测文件> --output runs/bootstrap.json` 做配对文档 bootstrap；`--by-topic` 补充主题重采样。

## 解码、校准与图

BART 角色专家保存两个槽位上 None/E1/E2 的词表概率。原论文判决作为独立结果：

```bash
python -m mode_eci --config configs/bart_slots.yaml decode --predictions <已有预测文件> --output runs/paper_decoding --mode paper --branch S --threshold <开发阶段固定的阈值>
```

`--mode existence` 对融合分布使用存在性阈值。主表使用三类 argmax；阈值判决不回流为门控的专家正确性定义。

独立温度对照先用 `cache-experts --scope inner-0 --split held` 和同一冻结专家的 `--split test` 生成原始记录，再执行 `calibrate --calibration-records <留出记录> --prediction-records <评测记录> --output <目录>`。程序检查校准文档未被专家训练，且与评测文档隔离。这是减少训练量的独立对照；主流程保留 T=1，不自动将校准分布混入门控或完整训练专家。

```bash
python scripts/create_config.py --base configs/bart_slots.yaml --output configs/bart_graph.yaml --graph
```

G 的训练图来自无图种子系统的文档级折外预测，G 的留出/测试图来自未训练过这些文档的种子专家。嵌套成本较高，先验证无图系统。只用预测边，不使用 gold，不做传递闭包；删除查询自身双向边，无图证据时屏蔽 G。`--demo-variant shuffle_graph` 是边数匹配的扰乱图诊断。

## 本机验证

```bash
python -m pytest
python -m mode_eci --config configs/bert_mlm.yaml verify-local --output runs/verification
python scripts/verify_reporting.py --verification-dir runs/verification
```

`verify-local` 在本地生成微型随机 BERT/BART，覆盖四种输出头、七专家、G1/G2/G3、保存加载、阶段复用和预测。全部标记 `test_only`；代码变动后使用新验收目录。

真实本地资产的少量前向验证：

```powershell
python scripts/verify_local_assets.py --data-dir F:\MoDE_ECI_research\data\esc --bert F:\MoDE_ECI_research\models\bert-base-uncased --mpnet F:\MoDE_ECI_research\models\all-mpnet-base-v2
```

该脚本只建立三篇训练文档的小索引并对三条查询前向，不做正式训练或研究评测。
