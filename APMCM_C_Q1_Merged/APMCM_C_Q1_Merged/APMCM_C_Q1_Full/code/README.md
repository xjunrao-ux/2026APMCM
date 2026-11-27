# 复现和预测

## 执行分析

在解压目录根部运行。已用 Python 3.12.14 和 [requirements.txt](requirements.txt) 中的版本验证。

```bash
python -m pip install -r code/requirements.txt
python code/run_all.py
```

各节也可以单独运行。

| 命令 | 生成内容 |
|---|---|
| `python code/descriptive.py` | 分组描述统计和新版分布总图 |
| `python code/classification.py` | 原分类结构、所有已比较特征组合的折外结果、混淆矩阵和重训模型 |
| `python code/nonparametric.py` | KWH、MWU、PERMANOVA、PERMDISP 与已做的敏感性结果 |
| `python code/evidence.py` | 配对回放计数、传播结构相关性汇总与图 |

复现入口读取 `data/` 中的清洗后 CSV，覆写各节中对应的复现结果。原版雨云图和早期分类器单独保留。历次分类器脚本和存档结果见 [原 50 条第三层方案](previous_versions/q1_three_stage/README.md) 和 [84 条扩充方案](previous_versions/q1_stage3_expanded/README.md)。复现结果使用固定随机种子，置换次数和统计口径见各节报告。

## 应用三层分类器

```bash
python code/predict_cascade.py data/Data01_clean.csv predictions_demo.csv
```

自有 CSV 需要标题列 `F02`、`PulseRate`、`Duration`、`F01`、`Fpeak`、`Q25`、`Q50`、`Q75`。F02 和脉冲率缺失用空单元格表示，其他六项应为第三层样本提供实测值。输出表包含走过的分类层、预测的鸣声类型和可用的第三层调制型模型分数。这个示例使用重训模型预测输入表，性能评估见 [分组折外预测](../02_classification/tables/classification_oof.csv)。
