"""第7章：状態空間モデル（構造時系列モデル）をカルマンフィルタで推定し、フィルタ値・平滑値・欠損の扱いを確かめる。

実行: python3 07_ets/03_state_space.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

from common import C_BLUE, C_GRAY, C_ORANGE, load_airpassengers, mae, savefig

warnings.filterwarnings("ignore")
y = load_airpassengers()
train, test = y[:-24], y[-24:]  # 01_ets.py と同じ分割

# 観測 = 水準 + 季節 + 誤差（対数）。水準は傾きを持ち、傾きと季節もゆっくり変わってよい
model = UnobservedComponents(np.log(train), level="local linear trend", seasonal=12)
res = model.fit(disp=False, maxiter=500)
print("推定した分散:", " / ".join(f"{n}={v:.2e}" for n, v in zip(model.param_names, res.params)))

fc = res.get_forecast(24)
pred = np.exp(fc.predicted_mean)
lo, hi = np.exp(fc.conf_int(alpha=0.10)).T.to_numpy()
covered = np.mean((test.to_numpy() >= lo) & (test.to_numpy() <= hi))
print(f"24か月先 MAE={mae(test, pred):.1f} / 90%区間 被覆率={covered:.1%} / 平均幅={np.mean(hi - lo):.1f}")

# フィルタ値はその月までの観測だけ、平滑値は学習期間全体の観測を使った水準の推定
level_f = np.exp(res.filtered_state[0])
level_s = np.exp(res.smoothed_state[0])
for month in ("1950-01", "1955-01", "1958-12"):
    i = train.index.get_loc(train.loc[month].index[0])
    print(f"{month}: 観測={train.iloc[i]:.0f} / 水準のフィルタ値={level_f[i]:.1f} / 平滑値={level_s[i]:.1f}")

# 学習期間の6か月を欠損にしても、補完せずにそのまま推定できる
gap = train.copy()
gap.loc["1955-01":"1955-06"] = np.nan
res_gap = UnobservedComponents(np.log(gap), level="local linear trend", seasonal=12).fit(disp=False, maxiter=500)
filled = np.exp(res_gap.smoothed_state[0] + res_gap.smoothed_state[2])  # 欠損月の水準＋季節の平滑値
missing = gap.isna().to_numpy()
print(f"欠損6か月: 実際の観測とのMAE={mae(train[missing], filled[missing]):.1f}"
      f" / 欠損ありで学習した24か月先 MAE={mae(test, np.exp(res_gap.forecast(24))):.1f}")

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(train.index, train, color=C_GRAY, lw=1, label="観測（学習期間）")
ax.plot(train.index[13:], level_f[13:], color=C_ORANGE, lw=1.2, label="水準のフィルタ値")
ax.plot(train.index, level_s, color=C_BLUE, lw=1.2, label="水準の平滑値")
ax.set_ylabel("旅客数（千人）")
ax.legend(fontsize=8, frameon=False)
ax.set_title("状態空間モデルで推定した水準（季節を除いた成分）", fontsize=10)
print("図:", savefig(fig, "ch07_state_space"))
