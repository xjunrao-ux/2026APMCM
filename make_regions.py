# -*- coding: utf-8 -*-
"""Build regions.json: merge indent-split paragraphs, register the two tables."""
import json
import pymupdf as fitz

doc = json.load(open('apmcm-c/document.json', encoding='utf-8'))
idx = {b['id']: b for p in doc['pages'] for b in p['blocks']}
REASON_P = ('源页该自然段的首行缩进与其余行齐左被打字机式提取拆成多个块；'
            '按源页整段视觉边界合并为一段，并以完整译文绑定')

PARAS = [
    # page, ids, style_role, translation
    (1, ['p0001-b0003', 'p0001-b0004'], 'title', '非洲企鹅在海上的声学通信与回放反应'),
    (1, ['p0001-b0005', 'p0001-b0006'], None,
     '非洲企鹅（Spheniscus demersus）是一种濒危海鸟，栖息于南部非洲沿海地区。南部非洲海岸鸟类保护基金会（SAFCCB）的研究表明，'
     '非洲企鹅种群正以年均 8% 的速度下降。一些研究预测，其野生种群可能在 2035 年前灭绝。因此，国际自然保护联盟（IUCN）已将'
     '该物种评估为极危（Critically Endangered）。非洲企鹅也被称为黑脚企鹅、开普企鹅或公驴企鹅。它会发出响亮的、类似驴叫的叫声；'
     '当许多企鹅同时鸣叫时，会产生相当大的环境噪声。它们在陆地上繁殖、换羽和休息，并在附近的沿岸海域觅食。海洋环境开阔，'
     '个体之间的距离变化迅速。波浪、光照和能见度限制了视觉信号的使用。因此，叫声很可能是非洲企鹅在海上保持联系、识别同伴'
     '并聚集为觅食群体的重要手段。'),
    (1, ['p0001-b0008', 'p0001-b0009'], None,
     '研究人员对非洲企鹅在海面的自然叫声及其对回放的反应开展了野外实验。首先，他们记录并分析了不同类型海面叫声的声学特征。'
     '他们还在海上播放企鹅叫声和对照声音，观察附近的企鹅是否'),
    (2, ['p0002-b0002', 'p0002-b0003'], None,
     '利用声学特征、声音传播衰减和海上回放反应数据建立数学模型，探究不同叫声类型在海上通信中所起的作用，并评估研究结果对'
     '企鹅觅食群体研究和保护管理的意义。'),
    (2, ['p0002-b0005', 'p0002-b0006'], None,
     '利用所提供叫声数据，考察三种叫声类型（CallType）：平调（flat）、调制（mod）和双声（tv）。对叫声时长、特征频率、'
     '峰值频率、频谱四分位数和脉冲率等指标进行描述性统计与可视化，并比较三种叫声类型之间的主要差异。利用所提供数据中的'
     '叫声样本识别主要影响因素，并为三种叫声类型构建分类器。此外，至少选择两种非参数检验，判断各声学特征指标在叫声类型'
     '之间是否存在显著差异。'),
    (2, ['p0002-b0007', 'p0002-b0008'], None,
     '将分类结果与传播和回放数据结合，讨论三种叫声类型在信息功能与传输稳健性方面可能存在的差异，并为后续的信号衰减模型'
     '和行为反应模型选取关键声学指标。'),
    (2, ['p0002-b0009', 'p0002-b0010'], None, '问题 2 声信号传播衰减与有效通信距离'),
    (2, ['p0002-b0011', 'p0002-b0013', 'p0002-b0012', 'p0002-b0014', 'p0002-b0015', 'p0002-b0016'], None,
     '利用所提供的声学与回放实验数据，综合考虑频谱图相关性、幅度谱相关性和幅度包络相关性，构造随传播距离变化的声学保真度'
     '指标。为平调（flat）、调制（mod）和双声（tv）叫声建立信号传播衰减模型，分析模型并给出明确结论。'),
    (3, ['p0003-b0003', 'p0003-b0004', 'p0003-b0006', 'p0003-b0005', 'p0003-b0007', 'p0003-b0008'], None,
     '利用所提供的回放实验数据，比较对自然叫声回放与对照声音回放的总体反应率。分析反应差异与情境（Context）、叫声类型'
     '（CallType）、回放距离（Distance）以及观测企鹅数（Npenguins）之间的关系。将抬头、鸣叫和接近视为三种反应行为。'
     '分析单次回放刺激之后反应行为组合的概率分布，并比较自然叫声与对照声音是否改变这些组合的分布。在同一建模框架内，'
     '比较对自然叫声和对照声音的反应，评估叫声类型是否会提高特定反应组合出现的概率。'),
    (3, ['p0003-b0009', 'p0003-b0010'], None,
     '针对发声次数（VocalNcalls）、抬头时长（HeadsUpDuration）和接近时长（ApproachDuration）等反应强度指标，建立反应'
     '潜伏期模型，比较不同回放条件下企鹅开始作出反应所需的时间。给出代表性条件下预测结果的数值表。代表性条件为：将观测'
     '企鹅数（Npenguins）设为其中位数，并报告各叫声类型在回放距离（Distance）为 50 m、100 m 和 150 m 时的反应潜伏期。'
     '若某条件下未观察到反应，则将该结果视为无反应。需注意，“无反应”并不意味着受试个体没有感知到声音刺激。'),
    (4, ['p0004-b0002', 'p0004-b0001', 'p0004-b0003'], None, '问题 4 评估声学通信并分析保护情景'),
    (4, ['p0004-b0004', 'p0004-b0005'], None,
     '整合前述模型得到的信号保真度估计、反应组合概率和企鹅所处情境，构建海上声学通信有效性指数。明确给出指数函数及其'
     '权重的生态学解释。利用该指数估计典型群体规模下各叫声类型的有效通信半径，或单次回放刺激下预期作出反应的个体数。'
     '在诸如观测企鹅中作出反应的个体比例（Npenguins）等假设下开展敏感性分析。'),
    (4, ['p0004-b0006', 'p0004-b0007'], None,
     '考虑海上环境噪声升高或海况恶化削弱信号识别能力的情景，分析由此引起的通信半径和反应概率的变化。例如，考虑将识别'
     '相关阈值从 0.5 提高到 0.6 或 0.7，或假设所有距离上的信号相关性总体下降 5%–20%。'),
    (4, ['p0004-b0008', 'p0004-b0009'], None,
     '考虑到现有样本的局限，提出对未来回放实验的改进建议。从控制声学干扰的角度，为非洲企鹅觅食生境提出具体的保护建议'
     '（例如保护觅食生境、规范海上船只活动），并说明这些建议的适用条件与局限。'),
    (5, ['p0005-b0002', 'p0005-b0003'], None,
     '数据文件 01。《Data01_AfricanPenguins_seasurface_calls_acoustics.csv》（包含非洲企鹅海面叫声声学测量数据的数据文件）。'),
    (5, ['p0005-b0004', 'p0005-b0005'], None,
     '数据文件 02。《Data02_AfricanPenguins_playbacks_data.csv》（关于海面叫声回放实验的数据文件）。'),
    (5, ['p0005-b0006', 'p0005-b0007'], None,
     '数据文件 03。《Data03_correlation1m_flat_call_spectrogram.csv》（记录平调叫声在 1 m 处与更远距离之间的频谱图相关度'
     '指标的数据文件）。'),
    (5, ['p0005-b0008', 'p0005-b0009'], None,
     '数据文件 04。《Data04_correlation1m_modulated_call_spectrogram.csv》（记录调制叫声在 1 m 处与更远距离之间的频谱图'
     '相关度指标的数据文件）。'),
    (5, ['p0005-b0010', 'p0005-b0011'], None,
     '数据文件 05。《Data05_correlation1m_two_voice_call_spectrogram.csv》（记录叫声在 1 m 处与更远距离之间的频谱图相关度'
     '指标的数据文件）。'),
    (5, ['p0005-b0012', 'p0005-b0013'], None,
     '数据文件 06。《Data06_correlation1m_flat_call_amplitude_spectrum.csv》（记录平调叫声在 1 m 处与更远距离之间的幅度谱'
     '相关度指标的数据文件）。'),
    (5, ['p0005-b0014', 'p0005-b0015', 'p0005-b0016', 'p0005-b0017'], 'body',
     '数据文件 07。《Data07_correlation1m_modulated_call_amplitude_spectrum.csv》（记录调制叫声在 1 m 处与更远距离之间的'
     '幅度谱相关度指标的数据文件）。'),
    (5, ['p0005-b0018', 'p0005-b0019', 'p0005-b0020', 'p0005-b0021'], 'body',
     '数据文件 08。《Data08_correlation1m_two_voice_call_amplitude_spectrum.csv》（记录双声叫声在 1 m 处与更远距离之间的'
     '幅度谱相关度指标的数据文件）。'),
    (5, ['p0005-b0022', 'p0005-b0023'], None,
     '数据文件 09。《Data09_correlation1m_flat_call_amplitude_envelope.csv》（记录平调叫声在 1 m 处与更远距离之间的包络'
     '相关度指标的数据文件）。）。'),
    (5, ['p0005-b0024', 'p0005-b0025', 'p0005-b0026', 'p0005-b0027'], 'body',
     '数据文件 10。《Data10_correlation1m_modulated_call_amplitude_envelope.csv》（记录调制叫声在 1 m 处及'),
    (6, ['p0006-b0002', 'p0006-b0003'], None,
     '数据文件 11。《Data11_correlation1m_two_voice_call_amplitude_envelope.csv》（记录双声叫声在 1 m 处与更远距离之间的'
     '包络相关度指标的数据文件）。'),
    (6, ['p0006-b0004', 'p0006-b0005'], None,
     '代码文件 1。《Code1_AfricanPenguins_playbacks_analyses.R》（用于回放实验统计分析的代码）。'),
    (6, ['p0006-b0006', 'p0006-b0007'], None,
     '代码文件 2。《Code2_AfricanPenguins_signal_degradations.R》（用于信号衰减统计分析的代码）。'),
    (7, ['p0007-b0060', 'p0007-b0061'], None,
     '记录不同叫声类型在 1 米处与更远距离处录制的频谱图、幅度波形和幅度包络的相关系数。'),
]

rows = []
# 跨行标题的首行缩进（源页首行 x=114，续行 x=90）：中文按首行锚点排，避免与相邻单行标题缩进不一致
HEAD_X0 = {'p0002-b0009': 114.0, 'p0004-b0002': 114.0}
for page, ids, role, text in PARAS:
    rect = fitz.Rect(idx[ids[0]]['bbox'])
    for bid in ids[1:]:
        rect |= fitz.Rect(idx[bid]['bbox'])
    rect = fitz.Rect(HEAD_X0.get(ids[0], 90.0), rect.y0, 522.0, rect.y1)
    row = dict(id='para-p%d-%s' % (page, ids[0].split('-')[-1]), page=page,
               bbox=[round(v, 2) for v in rect], kind='paragraph',
               source_ids=ids, translation=text, reason=REASON_P)
    if role:
        row['style_role'] = role
    rows.append(row)

# ---- tables on page 7 -------------------------------------------------------
pdf = fitz.open('apmcm-c/source.pdf')
page7 = pdf[6]
page_blocks = [b for p in doc['pages'] if p['number'] == 7 for b in p['blocks']]
tables = page7.find_tables().tables

T1_LABELS = ['第一基频', '第二基频', '峰值频率', '第25百分位', '中位频率', '第75百分位']
NOTE = ('VocalRelNCalls：每次反应叫声序号所对应的叫声类型。例如，VR1 是第一次反应叫声的叫声类型，'
        'VR19 是第十九次反应叫声的叫声类型。')

for n, table in enumerate(tables):
    tb = fitz.Rect(table.bbox)
    box = fitz.Rect(tb.x0 - .1, tb.y0 - .2, tb.x1 + .1, tb.y1 + .2)
    inside = [b for b in page_blocks if fitz.Rect(b['bbox']).intersects(tb)]
    box |= fitz.Rect(inside[0]['bbox'])
    for b in inside[1:]:
        box |= fitz.Rect(b['bbox'])
    row = dict(id='table-p7-%d' % (n + 1), page=7, bbox=[round(v, 2) for v in box], kind='table',
               source_ids=[b['id'] for b in inside], reviewed=True, labels=[],
               reason='原页为带框线的统计表，整表保留行列、数值与表线，仅替换经过核对的自然语言表头／说明文字')
    if n == 0:
        for cell, text in zip(table.rows[1].cells, T1_LABELS):
            c = fitz.Rect(cell)
            row['labels'].append(dict(source='', text=text, font_size=7.5,
                                      bbox=[round(v, 2) for v in (c + (0.3, 0.3, -0.3, -0.3))],
                                      background=[1, 1, 1]))
    else:
        c = fitz.Rect(table.rows[1].cells[0])
        row['labels'].append(dict(source='VocalRelNCalls', text=NOTE, font_size=8.0,
                                  bbox=[round(v, 2) for v in (c + (1.0, 1.2, -1.0, -1.2))],
                                  background=[1, 1, 1], align='left'))
    rows.append(row)

json.dump(dict(regions=rows), open('regions.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('regions', len(rows), 'paragraph', sum(1 for r in rows if r['kind'] == 'paragraph'),
      'table', sum(1 for r in rows if r['kind'] == 'table'))
for r in rows:
    if r['kind'] == 'table':
        print(r['id'], r['bbox'], len(r['source_ids']), 'labels', len(r['labels']))
        print('   labels:', [(l['text'][:20], l['bbox']) for l in r['labels']][:3])
