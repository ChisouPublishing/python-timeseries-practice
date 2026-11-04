"""第6章：ARIMA系モデル（ACF/PACF・検定・SARIMA・診断・予測区間）。

実行: python3 06_arima/01_sarima.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

from common import C_BLUE, C_GRAY, C_ORANGE, load_airpassengers, mae, rmse, savefig

warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings("ignore", message="The test statistic is outside")
warnings.filterwarnings("ignore", message="p-value is smaller than")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
y = load_airpassengers()

# --- 6.1 ACF / PACF --------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
plot_acf(y, lags=36, ax=axes[0])
plot_pacf(y, lags=36, ax=axes[1], method="ywm")
axes[0].set_title("自己相関（ACF）", fontsize=10)
axes[1].set_title("偏自己相関（PACF）", fontsize=10)
print("図:", savefig(fig, "ch06_acf_pacf"))

# --- 6.2 単位根・ホワイトノイズ -------------------------------------------
for name, series in [("水準", y), ("対数", np.log(y))]:
    p_adf = adfuller(series, autolag="AIC")[1]
    p_kpss = kpss(series, regression="c", nlags="auto")[1]
    print(f"{name}: ADF p={p_adf:.4f} / KPSS p={p_kpss:.4f}")

# --- 6.3 SARIMA ------------------------------------------------------------
train, test = y[:-24], y[-24:]
model = SARIMAX(
    np.log(train),
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)
print("AIC:", round(model.aic, 2))

# 季節ナイーブ（前年同月）をベースラインにする
naive = train[-12:].to_numpy()
naive_pred = np.tile(naive, 2)

pred = np.exp(model.get_forecast(steps=24).predicted_mean)
ci = np.exp(model.get_forecast(steps=24).conf_int())
print("季節ナイーブ MAE:", round(mae(test, naive_pred), 1), "/ RMSE:", round(rmse(test, naive_pred), 1))
print("SARIMA       MAE:", round(mae(test, pred), 1), "/ RMSE:", round(rmse(test, pred), 1))
print("改善率（MAE）:", round((1 - mae(test, pred) / mae(test, naive_pred)) * 100, 1), "%")

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(train.index[-48:], train[-48:], color=C_GRAY, label="訓練")
ax.plot(test.index, test, color=C_BLUE, label="実測（テスト）")
ax.plot(test.index, pred, color=C_ORANGE, label="SARIMA")
ax.fill_between(test.index, ci.iloc[:, 0], ci.iloc[:, 1], color=C_ORANGE, alpha=0.15, label="95%区間")
ax.set_ylabel("旅客数（千人）")
ax.legend(ncol=4, fontsize=8, frameon=False)
ax.set_title("SARIMA(1,1,1)(1,1,1,12) の24か月先予測", fontsize=10)
print("図:", savefig(fig, "ch06_forecast"))

# --- 6.4 次数選択（auto_arima） ------------------------------------------
import pmdarima as pm

auto = pm.auto_arima(
    np.log(train), seasonal=True, m=12,
    start_p=0, max_p=2, start_q=0, max_q=2,
    start_P=0, max_P=1, start_Q=0, max_Q=1,
    d=None, D=None, stepwise=True, trace=False, error_action="ignore", suppress_warnings=True,
)
print("auto_arima の次数:", auto.order, auto.seasonal_order, "/ AIC:", round(auto.aic(), 2))
auto_pred = np.exp(auto.predict(n_periods=24))
print("auto_arima MAE:", round(mae(test, auto_pred), 1))

# --- 6.5 残差診断 ----------------------------------------------------------
lb = acorr_ljungbox(model.resid, lags=[12, 24], return_df=True)
print(lb.round(3).to_string())

fig = model.plot_diagnostics(figsize=(9, 6))
fig.suptitle("SARIMA の残差診断（左上:残差、右上:分布、左下:QQ、右下:ACF）", fontsize=10)
print("図:", savefig(fig, "ch06_diagnostics"))
