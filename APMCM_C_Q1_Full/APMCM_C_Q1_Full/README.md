# 2026 APMCM C 题第一问完整研究包

研究对象为非洲企鹅在海面发出的三类鸣声，即平直型 `flat`、调制型 `mod` 和双声型 `tv`。报告按题面四项要求组织。各节保留方法选择依据、结果、解释和可追溯的图表。

| 题目要求 | Markdown 报告 | 复现代码 |
|---|---|---|
| 描述性统计、可视化和三类鸣声比较 | [01 描述性统计与可视化](01_descriptive/report.md) | [descriptive.py](code/descriptive.py) |
| 识别主要判别特征，构建三类分类器 | [02 分类模型与对照实验](02_classification/report.md) | [classification.py](code/classification.py) |
| 非参数检验声学特征差异 | [03 单指标和多指标非参数检验](03_nonparametric/report.md) | [nonparametric.py](code/nonparametric.py) |
| 信息功能、传播稳健性和后续选指标 | [04 传播与回放证据](04_function_and_transmission/report.md) | [evidence.py](code/evidence.py) |
| 数据来源、文件说明和研究边界 | [数据说明](data/README.md) · [尚未完成或无法从现有数据判定的部分](coverage_and_limits.md) | [复现与预测说明](code/README.md) |

## 核心结论

- 在 108 条声学样本中，Fpeak 是七项单指标检验经 Holm 校正后唯一仍显著的指标。F01 和 Q50 的原始检验达到 0.05 阈值。六项完整指标的联合 PERMANOVA 也显著（置换 \(p=0.0079\)，\(R^2=4.87\%\)）。
- 分层分类器以 F02 识别双声型、脉冲率有效测值分流调制型、六指标堆叠逻辑回归判别余下样本。按录音批次留出时，固定阈值版判对 102/108 条，与规则基线持平。空脉冲率调制型找回 3 条，平直型误报 3 条。
- 三类回放均引发鸣叫回应。调制型与双声型的靠近观察率高于平直型，支持两者可能兼具招募功能。平直型的时频结构相关较早降至 0.5 以下。振幅包络的相对排序随距离而变。
- 后续以时频图相似度、振幅频谱相似度和振幅包络相似度评价结构衰减，以刺激类型、行为情境、距离和群体大小建立行为响应模型。Fpeak、Q50、包络调制、脉冲率与双基频间距的逐轨道效应需要对应音轨支持。

## 开始复现

从解压后的根目录执行。已在 Python 3.12.14 上验证，具体依赖版本见 [requirements.txt](code/requirements.txt)。

```bash
python -m pip install -r code/requirements.txt
python code/run_all.py
```

图和结果表已随包提供。上面命令从清洗后的 CSV 重新生成复现版表图和分类模型。历次雨云图及两版分类器的原始交付保存在各自文件夹。此前只在对话中计算的分析已整理为本包脚本。

数据来源为 Thiebault 等的非洲企鹅海面鸣声与回放数据，原数据采用 CC BY 4.0 许可。[数据集与引用信息](https://doi.org/10.5281/zenodo.21376648)
