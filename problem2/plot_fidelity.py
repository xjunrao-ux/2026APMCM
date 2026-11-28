# -*- coding: utf-8 -*-
"""用 Pillow 画保真度衰减曲线（无 matplotlib 依赖）。"""
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CSV = r"G:\亚太数模\2026APMCM\problem2\results\fidelity_by_distance.csv"
OUT = r"G:\亚太数模\2026APMCM\problem2\results\fidelity_decay.png"

df = pd.read_csv(CSV)

W, H = 1000, 640
LEFT, RIGHT, TOP, BOT = 90, 960, 60, 560   # 绘图区

img = Image.new("RGB", (W, H), "white")
dr = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("arial.ttf", 20)
    font_s = ImageFont.truetype("arial.ttf", 16)
    font_t = ImageFont.truetype("arial.ttf", 22)
except Exception:
    font = font_s = font_t = ImageFont.load_default()

# log2 映射：距离 4..256 -> x 均匀
xs = [4, 8, 16, 32, 64, 128, 256]
import math
def X(d):
    return LEFT + (math.log2(d) - 2) / 6 * (RIGHT - LEFT)
def Y(f):
    return BOT - f * (BOT - TOP)

# 网格 + 刻度
for d in xs:
    x = X(d)
    dr.line([(x, TOP), (x, BOT)], fill=(225, 225, 225), width=1)
    dr.text((x - 10, BOT + 8), str(d), fill=(60, 60, 60), font=font_s)
for fy in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    y = Y(fy)
    dr.line([(LEFT, y), (RIGHT, y)], fill=(225, 225, 225), width=1)
    dr.text((LEFT - 42, y - 9), f"{fy:.1f}", fill=(60, 60, 60), font=font_s)

dr.rectangle([LEFT, TOP, RIGHT, BOT], outline=(120, 120, 120), width=1)
dr.text((LEFT, TOP - 34), "acoustic fidelity F(d) (geometric mean)", fill=(0, 0, 0), font=font_t)
# y 轴标签（旋转文字用 transposed 图实现略复杂，此处放左侧简化）
dr.text((22, (TOP + BOT) // 2 - 40), "F(d)", fill=(0, 0, 0), font=font_t)
dr.text((LEFT + (RIGHT - LEFT) // 2 - 60, BOT + 40), "propagation distance (m)", fill=(0, 0, 0), font=font_t)

colors = {"flat": (124, 205, 124), "mod": (139, 71, 38), "tv": (39, 64, 139)}
labels = {"flat": "flat", "mod": "mod", "tv": "tv"}

for ct in ["flat", "mod", "tv"]:
    sub = df[df["call_type"] == ct].sort_values("distance")
    pts = [(X(d), Y(f)) for d, f in zip(sub["distance"], sub["F_geo_mean"])]
    col = colors[ct]
    # 折线
    dr.line(pts, fill=col, width=3)
    # 误差棒 + 点
    for d, f, sd in zip(sub["distance"], sub["F_geo_mean"], sub["F_geo_sd"]):
        x, y = X(d), Y(f)
        ylo, yhi = Y(f - sd), Y(f + sd)
        dr.line([(x, ylo), (x, yhi)], fill=col, width=2)
        dr.line([(x - 5, ylo), (x + 5, ylo)], fill=col, width=2)
        dr.line([(x - 5, yhi), (x + 5, yhi)], fill=col, width=2)
        dr.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col, outline="white")

# 图例
lx, ly = RIGHT - 190, TOP + 10
for i, ct in enumerate(["flat", "mod", "tv"]):
    y = ly + i * 30
    dr.ellipse([lx, y, lx + 14, y + 14], fill=colors[ct])
    dr.text((lx + 22, y - 4), labels[ct], fill=(0, 0, 0), font=font_s)

img.save(OUT)
print("saved", OUT)
