# v4 实验定义与代码对应

本工程实现 v4.1 的独立参数、无图主线，并提供单轮预测图扩展。研究结论需要服务器上的完整开发和五折多种子结果。

## 概率接口

所有专家的输出顺序为 `[NONE, E1_TO_E2, E2_TO_E1]`。

`bert_mlm` 在唯一查询 MASK 位置读取三个原子关系 token 的 logits，只在这三个 logits 上归一化和计算交叉熵。`bert_cls` 对两个事件正文的 hidden states 分别取均值，拼接后经 MLP 三分类，支持不连续和重叠 span。

`bart_relation` 对固定单槽位预测关系 token。`bart_slots` 使用固定 decoder 模板 `In this context, <SLOT> causes <SLOT>.`，手工按 BART 约定右移。不同 gold 使用完全相同的 decoder 输入。

| 标签 | 原因槽位 | 结果槽位 |
| --- | --- | --- |
| NONE | `<None>` | `<None>` |
| E1_TO_E2 | `<E1>` | `<E2>` |
| E2_TO_E1 | `<E2>` | `<E1>` |

槽位主损失是两个位置完整词表交叉熵的平均。三类兼容分数是两个目标 token 的词表 log probability 之和，再在三个合法组合上 softmax。这是固定模板的组合评分，不称为自回归联合概率。`slot_relation_weight` 可单列实验加入组合交叉熵，默认 0；专家温度默认 1。

## 规则和检索

`features.py` 使用 ESC 原始句界。主规则检查因果谓词的主语/补语、原因从句或介词范围，以及紧邻句的结果标记。as/since/so 仅记录歧义命中；范围未解决时路由到无线索分支并保存 unresolved。规则是可审核的保守代理，否定不取消显式线索。

`retrieval.py` 保存训练对的标签、demo_cell、主题、句距、依存路径、提及相似度、审核状态和单位向量。查询编码没有 gold。检索排除同文档，在固定库和各标签子池中选择。配额不足时减少数量并记录缺失，不跨四库借例。

R3 的结构路从正标签全池召回跨主题候选，与语义路合并后按结构等级、MPNet 相似度、稳定 ID 排序。等级由连接词组别、句距桶和非空依存路径匹配组成，不用查询方向或 gold 拓扑。topic_id 不同不自动证明语义领域不同。

默认三类各选一个 demo，由样本 ID 和固定种子打乱位置；2-demo 的正方向也由无 gold 种子规则决定。只有 D 在训练和推理中都使用 demo；C 增加文档句子，S 使用目标句，M 仅用事件表面形式。

## 通用 demo 专家预训练

`model.demo_pretraining: true` 在每个 scope 中增加一个训练阶段 `experts/D_all`。它使用该 scope 的全部训练查询；每条查询仍按自身的 D1–D4 类型从相应训练库检索，模板、demo 清单及监督标签与专门化阶段一致。D_all 按全部开发查询的方向 F1、再按 NLL 选择 checkpoint。随后 D1–D4 分别加载同一 scope 的最佳 D_all 参数，只在所属类别查询上继续微调；优化器和学习率调度重新初始化。

D_all 仅用于初始化，不加入门控或最终融合。C/S/M 的训练方式由原有设置决定。未开启此选项的配置沿用原流程；`model.warmup` 是独立的 S 预热开关，同时开启时 D 分支优先使用 D_all。D_all 自身此时可从 S 初始化。

若某个 D 分支在当前 scope 无训练查询，则保留该 scope 的 D_all 参数，训练记录明确标注 `training_mode: initialized_only`、`training_samples: 0` 和 `epochs_run: 0`，不宣称完成该分支微调。其余有样本分支正常训练。当前 scope 规则在真实 ESC 数据上未产生 D3 查询，因此本机实验会采用此处理。

每个内部折独立训练 D_all，且不使用内部留出或测试文档；最终 scope 使用完整外层训练集。三内部折、七专家的系统增加四个 D_all 阶段，共 32 个专家训练任务。各阶段记录训练 pair IDs、类别计数及初始化 checkpoint 指纹，支持完成阶段复用。

本机完整运行配置为 `configs/roberta_mlm_dall_seed3407_gpu.yaml`，参考 seed3407 GPU 配置，仅增加 D_all、独立实验 ID，并为 8GB GPU 使用 BF16 和 `4 × 8` 有效 batch 32。数据不抽样，最多 10 轮、patience 2。`scripts/run_demo_pretraining.py` 强制离线复用缓存权重，先完成最终 D_all→D1–D4 和 dev/test 独立评测，写入 `demo_results.json`；随后完成其余最终专家、三组 OOF、G3 门控、dev/test 的 D/equal/learned 评测及训练来源审核。独立评测也可用 `scripts/evaluate_demo_specialists.py` 运行。一个外层折的结果不能当作五折结果。

## 门控和交叉拟合

门控只输入目标句、事件标记、demo_cell 和句距。分支 logits 经活跃 mask 和 softmax 后，线性加权专家三类分布。默认不截取 top-2。

正确专家集合按同一三类分布的 argmax 定义，包含 gold 为 NONE 的样本。教师目标在正确集合中对真实类别概率计算 `softmax(q_gold / teacher_temperature)`；错误专家目标为零。全错样本无教师目标：G2 训练前排除，G3 只计算融合损失。覆盖率按真实类别报告。

G1 优化融合 NLL；G2 只优化教师交叉熵；G3 优化融合 NLL 加逐样本教师辅助项。专家输出 detach，训练门控时不加载专家。未激活分支权重为零。

外层训练文档尽量平衡正例数、候选数和主题，分三个内部组。每个组的专家、S 预热和索引只能使用其他组。专家 checkpoint 用固定开发集选择；内部留出 gold 仅在预测生成后用于门控监督。门控在折外记录的文档级内部验证中选择轮次，再用全部折外记录重训。最终专家另外用完整外层训练文档训练。

## 产物

```text
artifacts/<experiment_id>/seed_<seed>/fold_<fold>/
  manifest.json                 配置、代码、数据、模型和划分指纹
  features.jsonl                无 gold 的规则结果
  rule_audit.csv                训练侧规则审核
  inner-0/                      其他内部组结构相同
    index/                      当前训练文档的索引和审核 CSV
    retrieval/                  固定 demo 清单
    experts/<branch>/           history、来源、训练状态、checkpoint
    predictions/held/           无 gold 的折外专家记录
  oof/                          拼接记录和来源文档边界
  final/                        完整外层训练专家与索引
  gates/<objective>_<hash>/      门控选择、重训、教师目标和覆盖率
  results/                      逐条预测和评测
  graph_seed/                   G 扩展的嵌套无图种子实验
```

评测按全部（`overall`）、句内（`intra`）和跨句（`cross`）分别统计 F1。`direction` 对两个因果类做 micro P/R/F1，方向反转同时计 1 个 FP 和 1 个 FN。`direction_fp_only` 中方向反转只计 1 个 FP，只有预测成 `NONE` 的正例计 FN。`existence` 将两个因果方向合并为同一个正类。无正例子集 F1 为 `null`。跨种子汇总在 `pooled_f1` 中给出两个方向口径的均值和标准差。概率诊断含 NLL、三类 Brier、存在性 AP 和可靠性分箱。日志保存原始关系 logits 或槽位概率、组合分数、实际输入、裁剪、示例和权重。

## 可选实验边界

已提供 BART large、独立温度校准、原论文判决和单轮预测图。共享 backbone、自由解释生成、自然词组 verbalizer、多轮图更新、GNN 编码和额外主题外划分仍是后续研究项，不参与默认主线。

温度工具使用独立校准组和同一冻结专家，检查文档隔离；未自动并入主流程每个内部组。不要把校准分布与 T=1 训练的门控混用。若增加 RoBERTa 公平对照，应把其 tokenizer 加入所有被比较配置的共同 tokenizer 列表。

接口依据：[BART](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/bart)、[BERT](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/bert)、[SentenceTransformer](https://sbert.net/docs/package_reference/sentence_transformer/model.html)。新增的组合归一化和教师监督按方案书单列，不称为参考论文原式。
