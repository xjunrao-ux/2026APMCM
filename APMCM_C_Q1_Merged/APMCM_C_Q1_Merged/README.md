# 2026 APMCM C 题第一问 · 两版合并包

从[第一问整合报告](integrated/report.md)开始。合并稿按题目四项要求给出统一口径、可直接引用的数字和建议图表。两套原始目录完整保留，报告中涉及有出入的内容，以[对照与勘误](integrated/comparison.md)为准。

| 内容 | 入口 |
|---|---|
| 第一问合并稿，含四项结论和后续变量选择 | [integrated/report.md](integrated/report.md) |
| 两版共同点、不同点、可借鉴内容及勘误 | [integrated/comparison.md](integrated/comparison.md) |
| 论文主图、补图和各自用途 | [integrated/figure_index.md](integrated/figure_index.md) |
| 前一版，对应清洗数据、分批交叉验证、可运行分类器和统计代码 | [APMCM_C_Q1_Full/README.md](APMCM_C_Q1_Full/README.md) |
| 用户提供版，对应 11 张题目数据表、题面、原 R 代码、音频示例、图表和研究记录 | [2026APMCM_C_P1/README.md](2026APMCM_C_P1/README.md) |

## 如何复核

从解压后的根目录运行。以下两个入口使用包内相对路径。

```bash
python -m pip install -r APMCM_C_Q1_Full/code/requirements.txt
python integrated/audit.py
python APMCM_C_Q1_Full/code/run_all.py
```

第一个 Python 脚本核对两包的数据和几处数字口径，结果放在 `integrated/tables/`。第二个脚本复现前一版四部分分析、图表与三层分类器。用户提供版的旧脚本仍采用其原电脑上的 Windows 路径，部分打包脚本还依赖历史 Git 目录。选用这些脚本时，可参考[原运行说明](2026APMCM_C_P1/02_代码/运行说明.md)。包内同时保留了它们生成的成图。

已有图像和结论的来源均在整合报告和图表索引中标注。正式传播衰减拟合和行为响应模型属于后续题目，前一版的缺口说明仍可参阅。[研究覆盖与数据缺口](APMCM_C_Q1_Full/coverage_and_limits.md)
