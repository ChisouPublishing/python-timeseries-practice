"""第11章：分位点予測を、被覆率の表・CRPS・PITで評価する。

実行: python3 11_prob/05_evaluate.py（先に 11_prob/01_quantile.py を実行）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import C_GRAY, C_ORANGE, OUT_DIR, mae, pinball_loss, savefig
from prob_data import load_splits

q = pd.read_csv(OUT_DIR / "11_prob_quantiles.csv", index_col="date", parse_dates=True)
taus = np.array([float(c) for c in q.columns])
X, y = load_splits()["test"]
assert q.index.equals(X.index)
y = y.to_numpy()

# 名目の水準ごとに、中心区間の被覆率を比べる（信頼性の表）
for level in (0.5, 0.8, 0.9):
    lo_tau, hi_tau = round((1 - level) / 2, 2), round(1 - (1 - level) / 2, 2)
    lo, hi = q[str(lo_tau)].to_numpy(), q[str(hi_tau)].to_numpy()
    print(f"名目{level:.0%}区間（{lo_tau}〜{hi_tau}）: 実際の被覆率={np.mean((y >= lo) & (y <= hi)):.1%}")

# CRPSは全分位点のPinball lossの平均の2倍で近似する。点予測ならCRPSはMAEと同じ
crps = 2 * np.mean([pinball_loss(y, q[c], t) for c, t in zip(q.columns, taus)])
print(f"CRPS（19分位点で近似）: {crps:.1f}")
print(f"参考 前週同曜日の点予測のMAE（＝CRPS）: {mae(y, X['lag7']):.1f}")

# PIT: 実績が何%点の位置に来たか。較正されていれば0〜1にほぼ一様に散らばる
pit = (q.to_numpy() <= y[:, None]).mean(axis=1)
counts, _ = np.histogram(pit, bins=np.linspace(0, 1, 6))
print("PITの5区間の度数:", counts.tolist(), f"（一様なら各{len(y) / 5:.0f}前後）")

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(np.arange(5) * 0.2 + 0.1, counts, width=0.18, color=C_ORANGE)
ax.axhline(len(y) / 5, color=C_GRAY, ls="--", lw=0.8, label="一様な場合の度数")
ax.set_xticks(np.linspace(0, 1, 6))
ax.set_xlabel("実績が位置した分位点（PIT）")
ax.set_ylabel("日数")
ax.legend(fontsize=8, frameon=False)
ax.set_title("PITのヒストグラム（最終テスト151日）", fontsize=10)
print("図:", savefig(fig, "ch11_pit"))
