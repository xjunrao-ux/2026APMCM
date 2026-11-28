# 翻译审阅报告

状态：complete
引擎：2.5.0；协议：2
源文件：YL02-2026APMCM-C.pdf
布局：paired
已写入 314 个译文排版段；独立保留 75 块；阻塞 0 项。
源块共 1015；可追踪分组覆盖 637 个源块（分组段与源块并非一对一，不与排版段数相加）。
保留统计：{'numeric_or_symbol': 351, 'equation': 35, 'code': 29, 'identifier': 2}

## 当前输出

- [translated](translated.paired.pdf)；SHA-256：dd6e2e4267f87bc32fa234148a0a6e98c3894718c6ad08b131c003d036b42fef
- [bilingual](bilingual.paired.pdf)；SHA-256：d49eaafe79f3ed78430b4fdd93beb6838f65806a09bd35d27ec4accf08fc4994

## 核验

文字回提取：314 块；保留区域：75 块。
视觉核验：已逐页核对全部 28 页中英文对照版：目录点线、表格区域中文标签、公式与参考文献处理均已确认。

弹性段落对照：保留字形、栏目和图文顺序，较短一侧补段后留白；页高可能增加。

## 问题与裁决


## 字体

- sans｜SimHei Regular｜9b1959db3b3abeb7efdaec26edf7dfe871a6039de8d614af7248575207be629e｜explicit
- serif｜SimSun Regular｜1526ac24375f51f6eb73bc2d3f8072dbe4a80a3a65217677c9d9a84f67dab2ab｜explicit

## 全文统一样式

方案状态：confirmed；确认：用户确认：正文宋体10.5pt/1.35倍行距，标题黑体，图注9.5pt，表格8.5pt；交付中英对照与纯中文PDF
- body：serif，10.5 pt；行距 14.18 pt；段后 6.3 pt。
- title：sans，19.0 pt；行距 22.8 pt；段后 8.4 pt。
- heading：sans，12.5 pt；行距 15.62 pt；段后 5.78 pt。
- subheading：sans，11.5 pt；行距 14.38 pt；段后 5.25 pt。
- caption：serif，9.5 pt；行距 12.35 pt；段后 4.73 pt。
- table：serif，8.5 pt；行距 9.77 pt；段后 2.62 pt。
- table_compact：serif，5 pt；行距 5.5 pt；段后 2.1 pt。
- label_compact：sans，6.0 pt；行距 6.6 pt；段后 2.1 pt。
- running：serif，8.0 pt；行距 9.6 pt；段后 2.62 pt。
- footnote：serif，8.0 pt；行距 9.6 pt；段后 2.1 pt。
- footer：sans，8.0 pt；行距 9.6 pt；段后 2.1 pt。
- author：serif，9.5 pt；行距 11.88 pt；段后 3.15 pt。
- affiliation：serif，9.5 pt；行距 11.88 pt；段后 2.1 pt。

## 代码与补充说明

### p0027-b0012
（代码保留原文，不加改动）本段先按列统计非缺失样本数 n 与各距离上的平均保真度 mu；再计算组内平方和 ss_within；最后用退化模型 m3 对距离 DIST 与均值 mu 做非线性最小二乘拟合，初值取 a=0.85、d0=120.0 m、指数 1.5。

## 术语


本报告由当前 qa 自动生成。修改内容、布局或输出后，必须重新构建和核验；不以追加完成声明代替验收。
