"""第11章：LightGBMの分位点回帰で、翌日売上の分位点と区間を作る。

実行: python3 11_prob/01_quantile.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import C_BLUE, C_ORANGE, OUT_DIR, mae, pinball_loss, savefig
from prob_data import load_splits

splits = load_splits()
Xtr, ytr = splits["train"]
Xte, yte = splits["test"]
print("学習:", Xtr.index.min().date(), "〜", Xtr.index.max().date(), f"/ {len(ytr)}日")
print("テスト:", Xte.index.min().date(), "〜", Xte.index.max().date(), f"/ {len(yte)}日")

# 5%〜95%の19分位点を、分位点ごとに別のモデルで学習する
taus = np.round(np.arange(0.05, 0.96, 0.05), 2)
raw = {}
for tau in taus:
    model = lgb.LGBMRegressor(objective="quantile", alpha=tau, n_estimators=300, learning_rate=0.05,
                              num_leaves=15, min_child_samples=20, random_state=42, verbose=-1)
    raw[tau] = model.fit(Xtr, ytr).predict(Xte)
raw = pd.DataFrame(raw, index=Xte.index)

# 分位点の交差を数え、各日で並べ替えて単調にする
crossed_days = int((raw.diff(axis=1).iloc[:, 1:] < 0).any(axis=1).sum())
pred = pd.DataFrame(np.sort(raw.to_numpy(), axis=1), index=raw.index, columns=raw.columns)
print(f"隣り合う分位点が1か所でも交差した日: {crossed_days} / {len(pred)}日（並べ替えて単調化）")
print(f"単調化の前に10%点が90%点を上回った日: {int((raw[0.1] > raw[0.9]).sum())}日")

y = yte.to_numpy()
for tau in (0.1, 0.5, 0.9):
    print(f"Pinball τ={tau}: {pinball_loss(y, pred[tau], tau):.1f}")
print(f"中央値のMAE: {mae(y, pred[0.5]):.1f}")
lo, hi = pred[0.1].to_numpy(), pred[0.9].to_numpy()
print(f"80%区間（10〜90%点）: 被覆率={np.mean((y >= lo) & (y <= hi)):.1%} / 平均幅={np.mean(hi - lo):.0f}")
print(f"実績が10%点を下回った日={np.mean(y < lo):.1%} / 90%点を上回った日={np.mean(y > hi):.1%}")

pred.to_csv(OUT_DIR / "11_prob_quantiles.csv", index_label="date")

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.fill_between(pred.index, lo, hi, color=C_ORANGE, alpha=0.2, label="80%区間（10〜90%点）")
ax.plot(pred.index, pred[0.5], color=C_ORANGE, lw=1, label="中央値（50%点）")
ax.plot(yte.index, y, color=C_BLUE, lw=1, label="実績")
ax.set_ylabel("売上（円）")
ax.legend(ncol=3, fontsize=8, frameon=False)
ax.set_title("分位点回帰による翌日売上の区間（最終テスト期間）", fontsize=10)
print("図:", savefig(fig, "ch11_quantile"))
