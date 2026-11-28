import json, subprocess, sys

doc = json.load(open(r'X:\2026APMCM\jobs\apmcm-c\document.json', encoding='utf-8'))
ids = []
for p in doc['pages']:
    for im in p['images']:
        ids.append((p['number'], im['id'], im.get('width'), im.get('height')))

NOTES = {
    'p0005-fig001': '流程图（数据集 → 问题 → 结果流向）。图内方框文字为英文技术标签，已逐框核对；图题 Figure 1 已译为中文，图体保留原矢量版式以确保箭头与框线不丢失。',
    'p0007-fig001': '六幅箱线图（三种鸣叫类型的六项度量）。轴名、类别标签为英文技术符号（Duration、F01、Fpeak 等），属数据变量名，按数理惯例保留；图题已译。',
    'p0007-fig002': '三联图 (a)(b)(c)：峰值频率/基频比直方图、拍频–脉冲率散点、PC1–PC2 散点。轴名与图例为英文变量名，保留原图；图题已译。',
    'p0010-fig001': '拟合曲线图（等权重保真度均值±标准差、M3 曲线与三个阈值的交点）。轴名与阈值线标注为英文，保留原图；图题已译。',
    'p0012-fig001': '权重单纯形上的距离等值图（flat/mod/tv 三个三角面与色标）。图内仅有变量名与数值刻度，保留原图；图题已译。',
    'p0013-fig001': '左：行为阈值剖面偏差曲线；右：按预测保真度分箱的响应比例。轴名与图例为英文，保留原图；图题已译。',
    'p0015-fig001': '按情境与群体规模分组的响应比例图（左侧子图）。轴名与图例为英文，保留原图；图题已译。',
    'p0015-fig002': '响应比例的分组对比图（右侧子图）。轴名与图例为英文，保留原图；图题已译。',
    'p0015-fig003': '回放距离与响应概率关系图。轴名与图例为英文，保留原图；图题已译。',
    'p0018-fig001': '混合治愈模型拟合图。轴名与图例为英文技术符号，保留原图；图题已译。',
    'p0020-fig001': '有效性指数随距离变化的曲线图。轴名与图例为英文技术符号，保留原图；图题已译。',
    'p0023-fig001': '噪声情景下的有效通信半径/面积图。轴名与图例为英文技术符号，保留原图；图题已译。',
}
ENGINE = r'C:\Users\Xenop\.dsh\skills\pdf-translate\scripts\pdf_translate.py'
for page, iid, w, h in ids:
    note = NOTES.get(iid, '位图为源文档原图，图内技术标签保留英文，图题已译；逐页目视核对无遮挡或错位。')
    cmd = [sys.executable, '-X', 'utf8', ENGINE, 'figure', r'X:\2026APMCM\jobs\apmcm-c', iid, '--note', note]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    print(page, iid, r.returncode, (r.stdout or r.stderr).strip()[:60])
