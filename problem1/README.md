# 第一问 · 海面叫声的声学差异与分类

先看本目录的 **建模决策.md**，再看 **results/problem1_conclusion.md**。results 已包含使用随题数据实际运行的结果。

## 在本机 VS Code 运行

1. 从 VS Code 的「文件 → 打开文件夹」选择 `D:\2026亚太\2026APMCM\problem1`。
2. 在「Python: Select Interpreter」中选择 `D:\Python\python.exe`（Python 3.13.5）。本目录已提供默认解释器设置。系统终端的裸 `python` 当前指向另一套 Anaconda 环境，其中 NumPy 与 pandas 二进制版本不兼容。
3. 打开 `train_classifier.py`，点击右上角「运行 Python 文件」。也可在该文件夹的 PowerShell 终端运行以下命令。

```powershell
& "D:\Python\python.exe" ".\train_classifier.py"
```

终端依次显示 A 至 E 五个阶段，最后打印两种验证方式下的六行模型结果和输出路径。重复运行会更新本目录下的同名结果。输入目录由脚本自身的位置解析。

## 在另一台电脑运行

保持 `problem1` 和 `题目与要求` 处于同一级目录，并保留后者的 `date` 子目录及原始 CSV 文件名。使用 Python 3.13 建立虚拟环境并安装已验证版本。在 VS Code 中选择新环境的解释器。

```powershell
py -3.13 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" train_classifier.py
```

## 输入与执行顺序

| 阶段 | 模块 | 输入与计算 | 查阅位置（均位于 results） |
|---|---|---|---|
| A | acoustics.py | 读取 Data01，核实字段、日期、类别、缺失、重复、物理约束与极端观测 | data_quality_report.md、clean_calls.csv、audit_flags.csv、date_counts.csv |
| B | explore.py | 分组描述、分布图、相关系数、VIF 与散点图 | eda_report.md、descriptive.csv、pearson.csv、spearman.csv、vif.csv、figures |
| C | explore.py | 两类秩检验、Holm 校正、效应量、按日期限制的置换 | nonparametric_tests.csv、within_date_permutation.csv |
| D | classifiers.py | 固定六项共同测量；依次建立 L2、RBF-SVM、随机森林；在相同外折与内折比较 | model_comparison.csv、fold_scores.csv、confusion_matrices.csv、class_scores.csv、oof_predictions.csv |
| E | integration.py | 解释系数和外折预测，结合 Data02 及 Data03–11 的分类别、分距离汇总 | coefficients.csv、rf_permutation.csv、weight_sensitivity.csv、propagation_by_distance.csv、playback_by_type.csv、problem1_conclusion.md |

## 读表与复用

- **主评估**使用 model_comparison.csv 中 `scheme=date` 的三行。accuracy、macro_f1、balanced_accuracy 合并每条叫声的唯一折外预测后计算；带 fold_mean、fold_sd 后缀的列是五折分数的算术均值与样本标准差。class_scores.csv 的 n 是真实类别的支持数。混淆矩阵的行为真实类别、列为预测类别。
- **日期分组**在内外层分别寻找同时满足三类覆盖的划分，按固定种子顺序采用第一个满足条件的划分。选择划分只用日期和类别。相同实验设置的三模型共用同一套折。常规分层对照允许同日样本分落训练折和测试折。
- **检验表**每行保留参与组的有效数、中位数与 IQR。family 表示参与同一次 Holm 校正的假设集合。置换表中的 p 值用包含原始排列的加一估计。
- **模型文件**中 call_classifier.joblib 是含标准化器的全样本 L2 模型，random_forest.joblib 是全样本随机森林。调参配置在 model_config.json。向其 predict 方法传入含 Duration、F01、Fpeak、Q25、Q50、Q75 的 DataFrame，按该顺序排列，使用原数据单位。两份模型均已拟合，可以直接在同版本 scikit-learn 中加载。
- **图像**覆盖样本数、八项指标的箱线图和直方图、六项指标的相关矩阵、散点图、折外混淆矩阵、模型指标和系数对比。图中空类别表示该类该项没有观测。
