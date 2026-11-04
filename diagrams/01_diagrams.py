"""本書で使う概念図（分析ワークフロー・タスク整理・交絡DAG・運用ループ）を描く。

実行: python3 diagrams/01_diagrams.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import C_BLUE, C_GRAY, C_ORANGE, C_PURPLE, C_RED, C_TEAL, savefig

INK = "#131A24"


def box(ax, x, y, w, h, text, edge, face="#FFFFFF", size=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.2, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK)


def arrow(ax, p1, p2, color, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                                 color=color, lw=1.1, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# --- 図1-1 分析ワークフロー -------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 2.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2.2)
ax.axis("off")
steps = ["可視化", "前処理", "特徴量", "モデル", "評価", "運用"]
colors = [C_BLUE, C_TEAL, C_TEAL, C_PURPLE, C_ORANGE, C_GRAY]
for i, (s_, c) in enumerate(zip(steps, colors)):
    box(ax, 0.15 + i * 1.62, 1.15, 1.35, 0.55, s_, c, size=9.5)
    if i < 5:
        arrow(ax, (0.15 + i * 1.62 + 1.35, 1.42), (0.15 + (i + 1) * 1.62, 1.42), C_GRAY)
arrow(ax, (7.9, 1.15), (1.5, 1.15), C_ORANGE, ls="--", rad=-0.22)
ax.text(5.0, 1.97, "評価から前処理へ戻る反復が実務の本体（1周して終わりではない）",
        ha="center", fontsize=8.5, color=C_ORANGE)
savefig(fig, "ch01_workflow")

# --- 図1-2 タスク×手法 ------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 3.2))
ax.set_xlim(0, 8.2)
ax.set_ylim(0, 3.4)
ax.axis("off")
cols = ["統計", "機械学習", "深層学習"]
rows = ["予測", "異常検知", "分類・クラスタ", "因果推論"]
cell = {(0, 0): "6・7・11章", (0, 1): "8・11章", (0, 2): "9・11章",
        (1, 0): "12章", (1, 1): "12章", (1, 2): "12章",
        (2, 0): "13章", (2, 1): "13章", (2, 2): "13章",
        (3, 0): "14章", (3, 1): "—", (3, 2): "—"}
for j, c in enumerate(cols):
    ax.text(2.3 + j * 1.9, 3.05, c, ha="center", fontsize=9.5, color=C_GRAY)
for i, r in enumerate(rows):
    y = 2.35 - i * 0.62
    ax.text(1.35, y, r, ha="right", va="center", fontsize=9.5, color=INK)
    for j in range(3):
        x = 1.6 + j * 1.9
        val = cell[(i, j)]
        face = "#F3F6FA" if val != "—" else "#FFFFFF"
        edge = C_BLUE if val != "—" else "#D2D9E2"
        box(ax, x, y - 0.22, 1.65, 0.44, val, edge, face, size=9)
savefig(fig, "ch01_tasks")

# --- 図14-1 交絡のDAG ------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 2.9))
ax.set_xlim(0, 5.6)
ax.set_ylim(0, 2.9)
ax.axis("off")
box(ax, 2.15, 1.95, 1.3, 0.55, "気温", C_ORANGE, "#FDF1E3")
box(ax, 0.35, 0.35, 1.55, 0.55, "アイス売上", C_BLUE, "#EAF2FB")
box(ax, 3.7, 0.35, 1.55, 0.55, "水難事故", C_BLUE, "#EAF2FB")
arrow(ax, (2.45, 1.95), (1.3, 0.92), C_GRAY)
arrow(ax, (3.15, 1.95), (4.35, 0.92), C_GRAY)
arrow(ax, (1.9, 0.62), (3.7, 0.62), C_RED, ls="--")
ax.text(2.8, 0.72, "見かけの相関", ha="center", fontsize=8, color=C_RED)
ax.text(2.8, 0.06, "気温を統制しないと、因果と誤解する", ha="center", fontsize=8, color=C_GRAY)
savefig(fig, "ch14_dag")

# --- 図15-1 運用ループ -----------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 2.1))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2.1)
ax.axis("off")
steps = ["監視", "劣化検知", "再学習", "再評価", "デプロイ"]
for i, s_ in enumerate(steps):
    box(ax, 0.2 + i * 1.95, 1.05, 1.55, 0.5, s_, C_BLUE, "#EAF2FB", size=9)
    if i < 4:
        arrow(ax, (0.2 + i * 1.95 + 1.55, 1.3), (0.2 + (i + 1) * 1.95, 1.3), C_GRAY)
arrow(ax, (9.0, 1.05), (0.9, 1.05), C_ORANGE, ls="--", rad=-0.2)
ax.text(5.0, 1.9, "性能指標と入力分布の両方を見て、再学習の時期を決める", ha="center", fontsize=8.5, color=C_ORANGE)
savefig(fig, "ch17_loop")
print("done")

# --- 図2-1 ライブラリの層 ---------------------------------------------------
fig, ax = plt.subplots(figsize=(8.4, 3.6))
ax.set_xlim(0, 8.4)
ax.set_ylim(0, 3.6)
ax.axis("off")
layers = [
    ("アプリケーション層", ["Prophet", "sktime", "Darts", "statsforecast"], C_ORANGE, 2.75),
    ("モデル層", ["statsmodels", "scikit-learn", "LightGBM", "PyTorch"], C_PURPLE, 1.85),
    ("データ層", ["pandas", "NumPy", "SciPy", "matplotlib"], C_BLUE, 0.95),
]
for title, items, color, y in layers:
    ax.text(0.25, y + 0.22, title, ha="left", va="center", fontsize=9, color=C_GRAY)
    for i, item in enumerate(items):
        box(ax, 2.35 + i * 1.45, y, 1.3, 0.45, item, color, "#FFFFFF", size=8.5)
    ax.add_patch(FancyBboxPatch((2.2, y - 0.12), 6.0, 0.69, boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=0.8, edgecolor="#D2D9E2", facecolor="none", linestyle="--"))
ax.text(4.2, 0.35, "上位ほど用途が限定され、下位ほど土台になる", ha="center", fontsize=8.5, color=C_GRAY)
savefig(fig, "ch02_layers")
