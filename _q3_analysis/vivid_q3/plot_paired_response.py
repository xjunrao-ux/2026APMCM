from figure_data import OUT, inputs

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS, _lighten


def draw_sankey(ax, source_nodes, target_nodes, flows,
                source_colors=None, target_colors=None,
                left_x=0.10, right_x=0.90,
                node_width=0.025, node_gap=0.03,
                title_top=None):
    """D3 风格 Sankey 图（矩形节点 + 贝塞尔流带）。

    Args:
        source_nodes: list of (name, total_value)，例如 [('国产', 1473), ('俄罗斯', 1500)]
        target_nodes: list of (name, total_value)，例如 [('总供给', 3000)]
        flows: list of dict {source_idx: int, target_idx: int, value: float}
        source_colors / target_colors: 可选，None 时自动按 PALETTE 分色
        node_width: 矩形宽度（相对坐标 0-1）
        node_gap: 同侧节点间垂直间距
        title_top: 顶部副标题（如"流量守恒 (万 m³)"）
    """
    total_src = sum(v for _, v in source_nodes)
    total_tgt = sum(v for _, v in target_nodes)
    if abs(total_src - total_tgt) > 1e-3:
        raise ValueError(f'流量不守恒：source 合计 {total_src} ≠ target 合计 {total_tgt}')

    Y_MARGIN = 0.08
    avail_h_src = (1 - 2 * Y_MARGIN) - node_gap * max(0, len(source_nodes) - 1)
    avail_h_tgt = (1 - 2 * Y_MARGIN) - node_gap * max(0, len(target_nodes) - 1)

    src_ranges = []
    y = 1 - Y_MARGIN
    for _, v in source_nodes:
        h = v / total_src * avail_h_src
        src_ranges.append((y - h, y))
        y = y - h - node_gap

    tgt_ranges = []
    y = 1 - Y_MARGIN
    for _, v in target_nodes:
        h = v / total_tgt * avail_h_tgt
        tgt_ranges.append((y - h, y))
        y = y - h - node_gap

    if source_colors is None:
        source_colors = [PALETTE[i % len(PALETTE)] for i in range(len(source_nodes))]
    if target_colors is None:
        target_colors = [COLORS.get('text', '#444444')] * len(target_nodes)

    # 左侧节点矩形 + 标签
    for i, ((name, v), (y0, y1)) in enumerate(zip(source_nodes, src_ranges)):
        rect = Rectangle((left_x - node_width / 2, y0), node_width, y1 - y0,
                         facecolor=source_colors[i], edgecolor='none', alpha=0.92, zorder=3)
        ax.add_patch(rect)
        ax.text(left_x - node_width / 2 - 0.022, (y0 + y1) / 2,
                f'{name}\n{v}/{total_src}  ({v / total_src:.2%})',
                ha='right', va='center', fontsize=9,
                color=COLORS.get('text', '#222'), fontweight='bold')

    # 右侧节点矩形 + 标签
    for i, ((name, v), (y0, y1)) in enumerate(zip(target_nodes, tgt_ranges)):
        rect = Rectangle((right_x - node_width / 2, y0), node_width, y1 - y0,
                         facecolor=target_colors[i], edgecolor='none', alpha=0.92, zorder=3)
        ax.add_patch(rect)
        ax.text(right_x + node_width / 2 + 0.022, (y0 + y1) / 2,
                f'{name}\n{v}/{total_tgt}  ({v / total_tgt:.2%})',
                ha='left', va='center', fontsize=9,
                color=COLORS.get('text', '#222'), fontweight='bold')

    # 贝塞尔流带
    src_top_used = [r[1] for r in src_ranges]
    tgt_top_used = [r[1] for r in tgt_ranges]

    for flow in flows:
        si, ti, val = flow['source_idx'], flow['target_idx'], flow['value']
        s_h = val / total_src * avail_h_src
        t_h = val / total_tgt * avail_h_tgt
        s_y_top = src_top_used[si]; s_y_bot = s_y_top - s_h; src_top_used[si] = s_y_bot
        t_y_top = tgt_top_used[ti]; t_y_bot = t_y_top - t_h; tgt_top_used[ti] = t_y_bot

        mid_x = (left_x + right_x) / 2
        path_data = [
            (Path.MOVETO,    (left_x + node_width / 2, s_y_top)),
            (Path.CURVE4,    (mid_x, s_y_top)),
            (Path.CURVE4,    (mid_x, t_y_top)),
            (Path.CURVE4,    (right_x - node_width / 2, t_y_top)),
            (Path.LINETO,    (right_x - node_width / 2, t_y_bot)),
            (Path.CURVE4,    (mid_x, t_y_bot)),
            (Path.CURVE4,    (mid_x, s_y_bot)),
            (Path.CURVE4,    (left_x + node_width / 2, s_y_bot)),
            (Path.CLOSEPOLY, (left_x + node_width / 2, s_y_top)),
        ]
        codes, verts = zip(*path_data)
        flow_color = flow.get('color', source_colors[si])
        patch = PathPatch(Path(verts, codes),
                          facecolor=flow_color, edgecolor='none', alpha=0.42, zorder=2)
        ax.add_patch(patch)
        if 'label' in flow:
            midpoint = (s_y_top + s_y_bot + t_y_top + t_y_bot) / 4
            ax.text(flow.get('label_x', mid_x), midpoint, flow['label'], ha='center', va='center',
                    fontsize=9, color=COLORS['dark'], zorder=4)

    if title_top:
        ax.text(0.5, 0.98, title_top, ha='center', va='top',
                fontsize=10, color=COLORS.get('text', '#444'), fontweight='bold')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('auto'); ax.axis('off')


def main():
    setup_style()
    counts, combo, _ = inputs()
    natural_only, control_only = counts[1, 0], counts[0, 1]
    both, neither = counts[1, 1], counts[0, 0]
    left = [('有响应', both + control_only), ('无响应', natural_only + neither)]
    right = [('有响应', both + natural_only), ('无响应', control_only + neither)]
    flows = [
        {'source_idx': 0, 'target_idx': 0, 'value': both,
         'color': _lighten(PALETTE[0], .27), 'label': f'{both} 对均响应'},
        {'source_idx': 1, 'target_idx': 0, 'value': natural_only,
         'color': PALETTE[0], 'label': f'{natural_only} 对仅自然声有响应', 'label_x': .60},
        {'source_idx': 1, 'target_idx': 1, 'value': neither,
         'color': COLORS['gray'], 'label': f'{neither} 对均无响应'},
        {'source_idx': 0, 'target_idx': 1, 'value': control_only, 'color': PALETTE[2]},
    ]
    assert sum(flow['value'] for flow in flows) == 139
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    fig.subplots_adjust(left=.018, right=.982, bottom=.145, top=.863)
    draw_sankey(ax, left, right, flows,
                source_colors=[PALETTE[0], COLORS['gray']],
                target_colors=[PALETTE[0], COLORS['gray']],
                left_x=.235, right_x=.765, node_width=.018, node_gap=.055)
    ax.text(.235, .985, '对照声', ha='center', va='bottom', fontsize=11, weight='bold')
    ax.text(.765, .985, '自然鸣声', ha='center', va='bottom', fontsize=11, weight='bold')
    ax.plot([.31, .34], [.011, .011], color=PALETTE[2], lw=1.8)
    ax.text(.352, .011, f'{control_only} 对仅对照声有响应', fontsize=8, va='center')
    no_response, no_response_low, no_response_high = combo['natural_equal', 'No response'][2:]
    fig.text(.06, .067, '139 对实验，带宽表示配对数。', fontsize=8)
    fig.text(.43, .067,
             f'调整后差值 +{-no_response:.2f} 个百分点  [95% 同时区间 {-no_response_high:.2f}, {-no_response_low:.2f}]',
             fontsize=8, color=COLORS['dark'])
    save_fig(fig, str(OUT / '01_paired_response.pdf'), formats=('pdf', 'png', 'svg'))
    plt.close(fig)


if __name__ == '__main__':
    main()
