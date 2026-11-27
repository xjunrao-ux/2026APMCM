# C题补充材料使用方案

第一问用三段录音展示叫声的可辨特征，第二问用原研究的传播曲线检验保真度模型，第三问用回放脚本、模型报告和现场视频解释行为观测，第四问用实测群体规模设定保护情景。下列对应关系依据赛题要求和附件内容整理；同一份材料可以支持多道题。[C 题英文原文，第 2–4 页为四问要求，第 6 页为附件说明](<题目与要求/2026 APMCM Problem C.pdf>)

## 九个文件分别用在哪里

| 附件 | 主要题号 | 可以回答的问题 | 论文中的用法 |
|---|---|---|---|
| [Supplementary 5 · flat.wav](题目与要求/date/Supplementary5_Audio_recordings_of_African_penguins’_sea-surfacecalls_flat.wav) | 第一问 | 平直型的谐波轨迹在一次鸣叫中如何变化？ | 在三类叫声对照图中展示较平稳的轨迹，解释“平直型”的声学含义。 |
| [Supplementary 5 · modulated.wav](题目与要求/date/Supplementary5_Audio_recordings_of_African_penguins’_sea-surfacecalls_modulated.wav) | 第一问，辅助第二问 | 频率随时间变化和振幅随时间起伏，在图上分别是什么样？ | 对照同一片段的时频谱和振幅包络，分别指出扫频和强弱起伏，引出第二问的不同声学线索。 |
| [Supplementary 5 · two-voice.wav](题目与要求/date/Supplementary5_Audio_recordings_of_African_penguins’_sea-surfacecalls_two-voice.wav) | 第一问 | 为什么双声型需要记录第二基频 F02？ | 在时频谱上标示两套谐波列，把分类变量 F02 与可观察结构联系起来。 |
| [Code2 · 传播退化分析](题目与要求/date/Code2_AfricanPenguins_signal_degradation.R) | 第二问 | 换用原作者的平滑方法，三类叫声的距离排序是否一致？ | 按脚本的米制距离和 GAM 公式拟合九条通道曲线。团队再使用相同权重与阈值求交点，与主模型对照。 |
| [Supplementary 4 · 传播退化图](题目与要求/date/Supplementary4_Signal_degradation_estimation.pdf) | 第二问，辅助第一问的稳健性讨论 | 不同距离上，频谱和节律哪一种线索保留得更好？ | 核对三类叫声在 4–256 m 相对于 1 m 参考录音的声谱图、幅度谱和包络相关性。原图的点是均值，误差棒是标准差，灰带是 GAM 曲线的 90% 置信区间。 |
| [Code1 · 海上回放分析](题目与要求/date/Code1_AfricanPenguins_playbacks_analyses.R) | 第三问 | 同一场回放中，抬头、鸣叫、接近如何计数和组合？ | 对照原脚本核查七种有响应组合，再计入“三项均无”得到八种结果。按实验编号关联自然声和对照声，并核对各行为发生后的潜伏期描述量。 |
| [Supplementary 3 · 回放模型与诊断](题目与要求/date/Supplementary3_Modelling_analyses_on_playback_experiments.pdf) | 第三问 | 为什么要分别处理行为是否发生、发生后的强度？ | 与原研究的三种模型比较：抬头时长用两部分负二项模型，鸣叫次数用两部分泊松模型，是否接近用二元模型。参考其先验比较、后验预测检查和残差诊断，说明团队的建模取舍。 |
| [Supplementary 2 · 回放实验视频](题目与要求/date/Supplementary2_Video_recordings_of_a_playback_experiment.mp4) | 第三问，辅助第四问的实验改进 | 现场从哪个事件开始计时，观察者能看到什么？ | 用有“playback”标记的画面说明计时起点和观测条件；第四问可据此提出同步记录播放时刻与个体位置。该视频展示一次实验，整体响应比例由 Data02 统计。 |
| [Supplementary 1 · 群体规模分布](题目与要求/date/Supplementary1_Distribution_of_African_penguin_group_sizes.pdf) | 第三问、第四问 | 第三问为何固定为 8 只，第四问典型、小型和大型群体如何取值？ | 用图说明 139 次回放实验覆盖的群体规模，再从 Data02 逐实验计算分位数。第三问固定中位数，第四问可比较三个群体规模情景。 |

## 四道题的落地安排

### 第一问　把三段录音用于解释分类特征

**已有成果。** 合并稿的图 7 已展示三段录音的归一化包络和时频谱。正文可在解释 F02 和脉冲率的段落引用该图。图注写明“每类一个代表性录音”。三张示例图用来解释个体声音的结构；分类准确率、组间差异和特征重要性沿用 Data01 的样本分析。[现有图 7](APMCM_C_Q1_Merged/APMCM_C_Q1_Merged/2026APMCM_C_P1/03_图表/fig7_三类呼叫包络与时频谱.png) · [第一问合并报告](APMCM_C_Q1_Merged/APMCM_C_Q1_Merged/integrated/report.md)

### 第二问　用 Code2 和 Supplementary 4 核对传播模型

**已有成果。** 团队已经按 Code2 的设定运行 GAM，并让它与保序主模型预测同一批留出距离。以三通道等权、阈值 0.5 为例，主模型对平直型、调制型、双声型给出约 85、145、170 m，GAM 对照给出约 87、162、188 m。两者的距离排序一致，估计值有差异。正文“模型检验”可用这一组数字说明为什么引入原作者方法。[团队的 GAM 实现](problem2/gam_reference.R) · [第二问结果](problem2/第二问运行结果.md)

**定稿时补充。** 在九条通道曲线的图注或结果段引用 Supplementary 4，说明测量以 1 m 录音为参照。它与 Code2 使用同一批传播观测，只能作为复核与方法对照。本文的综合保真度、连续阈值和通信距离来自团队计算；通信距离的不确定性沿用团队的删距分析。

### 第三问　用 Code1 和 Supplementary 3 核对模型，用视频交代观测

**已有成果。** 第三问诊断报告已对照 Code1 的分组频数和行为描述量，并说明团队模型与 Supplementary 3 的差异。写作时把“播放后多久开始抬头”归入**潜伏期**，把“抬头后持续多久”归入**强度**。题面要求的预测表使用团队在群体规模固定为 8 只、距离为 50、100、150 m 时的结果。[第三问诊断](problem3/results/q3_model_diagnostics.md) · [潜伏期预测表](problem3/results/q3_latency_predictions.csv)

**定稿时补充。** 可在实验说明中用一句话引用 Supplementary 2，说明回放起点和行为观察窗口的含义。未观测到响应时应保留“无响应”，不能据此推断个体没有听见声音。

### 第四问　用 Supplementary 1 确定群体规模情景

**建议采用。** 同一场实验在 Data02 中有自然声和对照声两行。为使每场实验只贡献一次观测，取每场的自然声记录，共 139 条。群体规模范围为 1–40 只，第 25、50、75 百分位数分别为 **3、8、15 只**。将 8 只设为典型群体，3 只和 15 只作为小群与大群的敏感性情景。[群体规模图](题目与要求/date/Supplementary1_Distribution_of_African_penguin_group_sizes.pdf) · [Data02 原表](题目与要求/date/Data02_AfricanPenguins_playbacks_data.csv)

第四问再按叫声类型、距离和行为情境接入第二问的保真度与第三问的响应组合概率。若报告“预计有多少只企鹅响应”，须额外设定**群体已出现响应时，组内响应个体所占比例**，并改变该比例做敏感性分析。8 只是典型群体的观测规模；群体出现响应的概率与个体响应比例分别描述两个层次。
