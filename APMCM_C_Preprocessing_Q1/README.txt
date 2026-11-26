2026 APMCM C 题｜预处理数据与第 1 问描述性统计

attachment_inventory.csv 是附件清单，含题目译文。input_audit.csv 和 missing_by_column.csv 是 11 张源数据表的行列数、重复行和缺失值检查记录。
Data01_clean.csv 保留 108 条鸣声。Date 统一为 YYYY-MM-DD。保留 F02、PulseRate 的原始缺失，并添加是否有观测的字段。spectral_IQR_Hz 是单条鸣声的 Q75 减 Q25。
Data02_clean.csv 保留 278 条回放记录。Date 统一为 YYYY-MM-DD。expID 是配对单位，每组各一次自然鸣声和对照。behavior_combo 按抬头、鸣叫、靠近排列，读取时指定为字符串以保留前导零。
Data02 的 Latency_flagged 字段标记行为标志为 0 但记录了潜伏期的条目。Latency_if_event_s 只在行为标志为 1 时保留潜伏期。原值保存在原字段，9 条待复核记录另见 Data02_latency_review.csv。
Data03-11_propagation_long.csv 将 9 张传播矩阵转成长表，保留距离、1 米参照、类型、测量方法、源文件、原始数据行号与相关系数。缺失值保留为空。source_row 仅用来回查原表，不能当作录音编号。各距离的实际有效数见 Data03-11_coverage.csv。
上述清洗过程未删除原始样本、未补填缺失，也未做标准化，以保留声学指标的原始单位。source_row 从 1 开始，指去除标题行后的数据行。
Q1_descriptive_statistics.csv 按鸣声类型和指标列出有效数、缺失数、均值、样本标准差、最小值、四分位数和最大值。组内四分位数采用线性插值。Q1_acoustic_distributions.png 展示 8 个题面指标的箱线图和实测散点。
CSV 采用带 BOM 的 UTF-8 编码。PDF、脚本和音视频保留作实验说明或复查。
