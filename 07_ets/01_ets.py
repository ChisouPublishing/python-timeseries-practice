"""第7章：指数平滑法とETS（加法/乗法・減衰トレンド・ARIMAとの比較）。

実行: python3 07_ets/01_ets.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from common import C_BLUE, C_GRAY, C_ORANGE, C_TEAL, load_airpassengers, mae, rmse, savefig

y = load_airpassengers()
train, test = y[:-24], y[-24:]
naive_pred = np.tile(train[-12:].to_numpy(), 2)
print("季節ナイーブ MAE:", round(mae(test, naive_pred), 1))

# --- 7.1 Holt-Winters（加法・乗法） ---------------------------------------
add = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12,
                           initialization_method="estimated").fit()
mul = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12,
                           initialization_method="estimated").fit()
pred_add, pred_mul = add.forecast(24), mul.forecast(24)
print("加法  MAE:", round(mae(test, pred_add), 1), "/ AIC:", round(add.aic, 1))
print("乗法  MAE:", round(mae(test, pred_mul), 1), "/ AIC:", round(mul.aic, 1))
print("乗法の季節指数（12か月）:", np.round(mul.season[-12:], 3).tolist())

# --- 7.3 ETS（誤差も明示的に選ぶ） ---------------------------------------
ets_add = ETSModel(train, error="add", trend="add", seasonal="add", seasonal_periods=12).fit(disp=False)
ets_mul = ETSModel(train, error="mul", trend="add", seasonal="mul", seasonal_periods=12).fit(disp=False)
ets_damped = ETSModel(train, error="mul", trend="add", damped_trend=True,
                      seasonal="mul", seasonal_periods=12).fit(disp=False)
rows = []
for name, m in [("ETS(A,A,A)", ets_add), ("ETS(M,A,M)", ets_mul), ("ETS(M,Ad,M)", ets_damped)]:
    p = m.forecast(24)
    rows.append((name, round(m.aic, 1), round(mae(test, p), 1), round(rmse(test, p), 1)))
print("モデル        AIC     MAE    RMSE")
for r in rows:
    print(f"{r[0]:12s} {r[1]:7.1f} {r[2]:7.1f} {r[3]:7.1f}")

# --- 図 -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
ax = axes[0]
ax.plot(test.index, test, color=C_BLUE, label="実測")
ax.plot(test.index, pred_add, color=C_ORANGE, ls="--", label="加法")
ax.plot(test.index, pred_mul, color=C_TEAL, label="乗法")
ax.set_title("加法と乗法の予測（24か月）", fontsize=10)
ax.legend(fontsize=8, frameon=False)

# 学習期間の当てはめ：乗法は振幅の拡大を追う
ax = axes[1]
ax.plot(train.index[-48:], train[-48:], color=C_GRAY, lw=1, label="実測")
ax.plot(train.index[-48:], mul.fittedvalues[-48:], color=C_TEAL, lw=1, label="乗法の当てはめ")
ax.set_title("乗法モデルは振幅の拡大を捉える", fontsize=10)
ax.legend(fontsize=8, frameon=False)
print("図:", savefig(fig, "ch07_ets"))
