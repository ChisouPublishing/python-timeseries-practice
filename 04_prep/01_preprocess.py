"""第4章：前処理（リサンプリング・補間・外れ値・定常化・分割・スケーリング）。

実行: python3 04_prep/01_preprocess.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

from common import (
    C_BLUE,
    C_GRAY,
    C_ORANGE,
    C_PURPLE,
    C_TEAL,
    load_airpassengers,
    load_retail,
    savefig,
)

# --- 4.1 リサンプリング ----------------------------------------------------
ts = load_retail(store_id=1).set_index("date").sort_index().asfreq("D")
weekly = ts.resample("W-SUN").agg({"sales": "sum", "promo": "sum"})
print("日次:", len(ts), "行 → 週次:", len(weekly), "行")
print(weekly.head(3).round(0).to_string())

# 区間の境界は label / closed で決まる。既定は右端ラベル・右閉区間
left = ts.resample("W-SUN", label="left", closed="left").agg({"sales": "sum"})
print("既定の最初の週:", weekly["sales"].iloc[0], "／ label=left の最初の週:", left["sales"].iloc[0])

# --- 4.2 欠損補間 ----------------------------------------------------------
gaps = ts.copy()
# 補間の違いを見るため、連続 5 日の欠損を人為的に作る
gaps.loc["2024-02-05":"2024-02-09", "sales"] = np.nan
filled = pd.DataFrame({
    "ffill": gaps["sales"].ffill(),
    "linear": gaps["sales"].interpolate(method="linear"),
    "time": gaps["sales"].interpolate(method="time"),
})
print(filled.loc["2024-02-03":"2024-02-12"].round(0).to_string())

fig, ax = plt.subplots(figsize=(9, 3))
w = slice("2024-01-25", "2024-02-20")
ax.plot(gaps.loc[w].index, gaps.loc[w, "sales"], "o", ms=3, color=C_GRAY, label="観測値")
for col, color, ls in [("ffill", C_ORANGE, "--"), ("linear", C_BLUE, "-"), ("time", C_TEAL, ":")]:
    ax.plot(filled.loc[w].index, filled.loc[w, col], color=color, linestyle=ls, label=col)
ax.set_ylabel("売上（円）")
ax.legend(ncol=4, fontsize=8, frameon=False)
ax.set_title("補間方法で埋まる値は変わる（5日間の欠損）", fontsize=10)
print("図:", savefig(fig, "ch04_interpolation"))

# 補間した事実を残す（補間フラグ）
gaps["is_imputed"] = gaps["sales"].isna().astype(int)
print("補間フラグの合計:", int(gaps["is_imputed"].sum()))

# --- 4.3 外れ値 ------------------------------------------------------------
s = ts["sales"]
z = (s - s.mean()) / s.std()
print("Zスコア>3 の日数:", int((z.abs() > 3).sum()))
q1, q3 = s.quantile([0.25, 0.75])
iqr = q3 - q1
print("IQR基準の外れ値:", int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()), "件")
print("最大の日:", s.idxmax().date(), int(s.max()), "円（年末商戦）")

# --- 4.4 定常化 ------------------------------------------------------------
y = load_airpassengers()
logy = np.log(y)
d1 = logy.diff().dropna()
d12 = logy.diff().diff(12).dropna()
with warnings.catch_warnings():
    # statsmodels 0.15 の移行期警告（戻り値の型変更予定）を抑える
    warnings.simplefilter("ignore", FutureWarning)
    for name, series in [("水準", y), ("対数", logy), ("対数差分", d1), ("対数差分＋季節差分", d12)]:
        p = adfuller(series.dropna())[1]
        print(f"ADF p値 {name:8s}: {p:.4f}")

fig, axes = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True)
axes[0].plot(y.index, y, color=C_BLUE)
axes[0].set_title("元系列（トレンドと振幅の拡大）", fontsize=10)
axes[1].plot(d1.index, d1, color=C_ORANGE)
axes[1].axhline(d1.mean(), color=C_GRAY, lw=0.8, ls="--")
axes[1].set_title("対数1階差分（平均がほぼ一定）", fontsize=10)
print("図:", savefig(fig, "ch04_diff"))

# --- 4.5 分割 --------------------------------------------------------------
X = np.arange(len(ts)).reshape(-1, 1)
tscv = TimeSeriesSplit(n_splits=5)
for i, (tr, te) in enumerate(tscv.split(X), start=1):
    print(f"fold{i}: train {tr[0]:4d}-{tr[-1]:4d} / test {te[0]:4d}-{te[-1]:4d}")

# --- 4.6 スケーリングとリーク ---------------------------------------------
data = ts.dropna(subset=["sales"])
split = int(len(data) * 0.8)
train, test = data.iloc[:split], data.iloc[split:]

good = StandardScaler().fit(train[["sales"]])
leak = StandardScaler().fit(data[["sales"]])  # 全期間で fit＝未来情報を使っている
print("訓練のみでfit  テスト平均:", round(float(good.transform(test[["sales"]]).mean()), 4))
print("全期間でfit    テスト平均:", round(float(leak.transform(test[["sales"]]).mean()), 4))
print("（後者はテストの平均が0に寄る＝未来の情報が混ざっている）")
