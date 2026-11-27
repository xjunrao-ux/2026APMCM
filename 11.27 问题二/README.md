# 11.27 问题二（今天的产出）

本文件夹汇总今天第二问「声保真度指标 + 退化曲线」的全部代码与结果，按递进顺序排。

## 文件索引

| 文件 / 文件夹 | 是什么 | 说明 |
|---|---|---|
| `fidelity_measure.py` | 第一版保真度 | 0.5/0.3/0.2 主观加权，几何+算术合成，含权重敏感性 |
| `plot_fidelity.py` | 早期画图脚本 | Pillow 版（当时没装 matplotlib），已废弃 |
| `step1_geometric.py` | 几何平均对照 | 复刻团队保序拟合，验证算术 vs 几何（结论排序不变） |
| `results/` | 第一版结果 | `fidelity_by_distance.csv`（通道均值+合成值）、`fidelity_weight_sensitivity.csv`、`fidelity_decay.png` |
| `critic_fidelity/` | **最终版（重点）** | CRITIC 客观赋权 + 信噪比退化曲线 + 三次样条对比，含完整代码、README、结果 |

## 最终结论（critic_fidelity 里）

- **CRITIC 客观权重**：包络 0.4612 > 声谱图 0.2752 > 幅度谱 0.2636（包络最独立、信息量最大）。
- **有效通信距离 D(T)**（信噪比模型 × CRITIC 权重）：

| 叫声 | D(0.5) | D(0.6) | D(0.7) 可靠识别 |
|---|---|---|---|
| flat | 71.9 m | 49.9 m | 30.2 m |
| mod | 126.4 m | 88.6 m | **54.9 m** |
| tv | 144.9 m | 93.9 m | 40.8 m |

- 潜在识别(0.5)下 tv > mod > flat；可靠识别(0.7)下 **mod 反超 tv**（因为 CRITIC 把 0.46 权重给了包络，而 tv 包络最弱）。
- 三次样条插值在 128–256 m 会欠冲（flat 掉到 0.082），信噪比模型单调平滑、更可取。

## 运行

```bash
pip install numpy pandas scipy matplotlib
python critic_fidelity/critic_fidelity.py
```
