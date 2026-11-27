# 论文图表取用索引

| 论文用途 | 建议用图 | 说明 |
|---|---|---|
| 展示各指标分布 | [A 版雨云图合集](../APMCM_C_Q1_Full/01_descriptive/figures/nature_style/APMCM_C_Q1_Nature.pdf) 或 [分布总图](../APMCM_C_Q1_Full/01_descriptive/figures/descriptive_reproduced.png) | 正文可选 Fpeak、F01、脉冲率；完整八项作附录 |
| 展示 Fpeak 的低频峰与高频峰分布 | [B 版峰位图](../2026APMCM_C_P1/03_图表/fig3_数据怎么读.png) | 直方图适合引用。涉及峰位比例的图注需与[定义核对表](tables/peak_definition_counts.csv)对齐 |
| 显著性和效应量 | [A 版 Holm 检验图](../APMCM_C_Q1_Full/03_nonparametric/figures/univariate_p_values.png) · [B 版效应量森林图](../2026APMCM_C_P1/03_图表/fig4_效应量森林图.png) | 主图采用 Holm。森林图使用 B 版单条样本 bootstrap，区间作为补充 |
| 多元结构重叠 | [B 版 PCoA 图](../2026APMCM_C_P1/03_图表/fig6_多元空间与显著指标.png) | PCoA 子图可说明组间和组内变异。标注单指标面板使用 BH |
| 三层分类与第三层代价 | [A 版混淆矩阵](../APMCM_C_Q1_Full/02_classification/figures/cascade_confusion.png) · [第三层找回及误报](../APMCM_C_Q1_Full/02_classification/figures/stage3_tradeoff.png) | A 版按日期和地点整批验证 |
| 多项 logit 与 AUC 补充对照 | [B 版 AUC 图](../2026APMCM_C_P1/03_图表/fig8_AUC对比.png) | 保留 B 版逐条留一和原始特征打分口径。脉冲率测值指示的列应标注清楚 |
| 展示三类鸣声结构 | [B 版包络与时频谱](../2026APMCM_C_P1/03_图表/fig7_三类呼叫包络与时频谱.png) | 每类一条例示，配合传播和回放的定量证据使用 |
| 传播和行为 | [A 版三通道传播曲线](../APMCM_C_Q1_Full/04_function_and_transmission/figures/propagation_profiles.png) · [分类型回放反应](../APMCM_C_Q1_Full/04_function_and_transmission/figures/playback_response.png) | 可直接支持稳健性和潜在联系／招募功能的讨论 |

B 版[原图清单](../2026APMCM_C_P1/03_图表/图表清单.md)也保留在原目录中。正式论文可统一图例、类型译名和配色；采用 SVG 或 PDF 的图可见 A 版各小节的 figures 目录。
