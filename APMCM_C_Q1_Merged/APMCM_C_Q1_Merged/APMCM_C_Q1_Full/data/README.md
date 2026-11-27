# 数据口径与来源

| 包内文件 | 用途 |
|---|---|
| [Data01_clean.csv](Data01_clean.csv) | 108 条声学测量，包含时长、第一和第二基频、峰值频率、频谱四分位频率及脉冲率 |
| [Data02_clean.csv](Data02_clean.csv) | 278 条回放记录，按 `expID` 形成 139 组自然鸣声与海浪声对照 |
| [Data02_latency_review.csv](Data02_latency_review.csv) | 标志为未发生但记有行为潜伏期的原值复核表 |
| [Data03–11 传播长表](Data03-11_propagation_long.csv) | 三类鸣声、三种相关测度、七个传播距离 |
| [Data03–11 有效样本覆盖](Data03-11_coverage.csv) | 各类型、测度和距离的有效数、缺失数及范围 |
| [数据预处理说明](preprocessing.md) | 缺失、字段标准化、异常记录与传播长表口径 |
| [附件清单](attachment_inventory.csv) · [源表审计](input_audit.csv) · [缺失清单](missing_by_column.csv) | 溯源与质量核查 |

CSV 使用 UTF-8。声学空值保留为缺失，Data02 的 `behavior_combo` 读取时应指定字符串以保留前导零。传播实验中 `reference_m=1` 表示基准录音位置，`distance_m` 是实际比较距离。长表多行相关值可能共享同一录音，汇总中的有效数是可用相关值数量。Data02 的重要分组键为 `expID`，Data01 的分类分批键为日期与地点。

数据来源为 Thiebault 等的 African penguins sea-surface vocalisations and playback experiment at sea，原数据采用 CC BY 4.0 许可。[原始数据与引用](https://doi.org/10.5281/zenodo.21376648) · [研究论文](https://ore.eu/article/view/1538/version/2510)
