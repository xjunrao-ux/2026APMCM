# 翻译任务说明（APMCM 论文 英译中）

你在为一个本地 PDF 翻译任务（pdf-translate 引擎 v2.5.0）翻译一个切片。不要运行 `next`/`apply`/`build`，只产出回复 JSON 文件。

## 输入
切片 JSON 文件形如：
{"packet":"<64位hash>","pages":["p0003",...],"items":[[块ID, 源文本], [块ID, 源文本, {"role":"table_cell"}], ...]}

## 输出
把全部条目翻译后写入指定的回复 JSON 文件，格式与 packet 相同、顺序一致：
{"packet":"<原样复制输入里的 packet>","items":[[块ID,"中文译文"], ...]}
- 必须包含切片中**每一个**块，顺序与输入一致，块 ID 原样复制。
- 纯标识符（如 "C"、"apmcm26XXXXXXX"、"F02"）没有可译内容时写成 ["块ID","",{"keep":"identifier"}]（该写法只用于不含空格的短标识符）。
- 文件用 UTF-8 编码、合法 JSON（不要 markdown 代码围栏、不要注释、字符串内不能有裸换行）。
- 写完后自检：`python -X utf8 -c "import json;json.load(open(r'输出路径',encoding='utf-8'))"`。

## 硬性规则（校验会拒绝）
1. `<s0>…</s0>`、`<s1>…</s1>` 这类强调标签：按源文顺序各保留一次，包住对应译文片段，不能增删、不能调换顺序。源文没有标签就不要加。
2. `⟦M0001⟧` 这类公式占位符：原样各保留一次、顺序不变；不要改写、不要翻译、不要漏掉。公式两侧的文字要译成通顺中文（源文在公式附近可能因 PDF 抽取而词序错乱，按语义重建即可）。
3. 数字、单位、百分数、p 值、置信区间、页码、引用编号（如 [12]、Table 3、Figure 1）、变量符号一律不改。数字个数与顺序保持与源文一致。
4. 不删句、不摘要、不增补原文没有的内容；专名与关键术语首次出现用「中文（原英文）」，之后只用中文。
5. 源文中的断词（如 "mea-sures"、"funda-mental"）要还原成完整单词后理解，再译成中文。

## 术语表（全篇统一，必须使用）
- African penguin 非洲企鹅；call 鸣叫；call type 鸣叫类型；sea-surface call 海面鸣叫
- flat 平直型；modulated 调制型；two-voice 双声型；second fundamental 第二基频（F02）；pulse 脉冲
- fidelity 保真度；fidelity index 保真度指数；propagation correlation 传播相关性
- playback experiment 回放实验；recognition probability 识别概率；effectiveness index 有效性指数
- effective distance 有效距离；contact radius 接触半径；recruitment radius 招募半径；recruitment 招募
- mixture cure model 混合治愈模型；gated response model 门控响应模型；penalised multinomial model 惩罚多项模型
- Kruskal–Wallis test Kruskal–Wallis 检验；Dunn comparison Dunn 两两比较；Mann–Whitney test Mann–Whitney 检验
- PERMANOVA 置换多元方差分析（PERMANOVA）；Holm correction Holm 校正；block bootstrap 分块自举
- signal-to-noise ratio 信噪比（SNR）；signal degradation 信号退化；acoustic communication 声学交流
- group size 群体规模；latency 潜伏期；vessel noise 船舶噪声；foraging 觅食；context 情境
- vocal reply 声音回应；approach 接近；tiered threshold 分级阈值；sensitivity analysis 敏感性分析
- uncertainty 不确定性；robustness 稳健性；conservation 保护；model assumption 模型假设

## 文体
- 数学建模竞赛论文的学术中文：准确、简洁、书面化；标题用短语，正文避免口语。
- 目录行（Contents 页）：只译标题文字，保留行首编号（如 "1 . 1" 可规范为 "1.1"）、原有的点线 ". . . ." 和行尾页码数字。
- 摘要页（p0001）：整段学术摘要，术语按术语表。
- 表格单元格（带 "role":"table_cell"）：短标签，如 "Symbol"→"符号"、"Meaning"→"含义"。
- 页眉页脚（如 "Team # apmcm26XXXXXXX"、"Page 2 of 25"）：译自然语言部分，编号保留，如 "队伍 # apmcm26XXXXXXX"、"第 2 页 / 共 25 页"。
- 图注/表注（Figure 1 …, Table 1 …）：保留 Figure/Table 编号，如 "图 1 数据集、问题及其结果流向"；"Table 1 主要符号"。
- 若某块源文是两三个碎片拼在一行（抽取造成），按行的视觉顺序给出对应中文即可，不合并、不重排块。

## 步骤
1. 读取切片文件；
2. 逐条翻译（可先在脑中按页理解上下文）；
3. 用 write 工具写出回复 JSON；
4. 用 python 校验 JSON 合法，并检查每个块的强调标签与 ⟦M⟧ 占位符数量、顺序与源文一致；
5. 最后回复：输出文件路径、条目数、遇到的疑问（1-3 条即可）。
