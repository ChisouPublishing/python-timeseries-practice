"""第16章 ケース3：金融データ（株価リターンは予測できるか）。

実行: python3 16_cases/03_finance.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

from common import C_BLUE, C_GRAY, C_ORANGE, load_stock, mae, savefig

warnings.filterwarnings("ignore")
price = load_stock()
ret = np.log(price).diff().dropna()
print("期間:", price.index.min().date(), "〜", price.index.max().date(), "/", len(price), "営業日")
print("日次リターン: 平均", round(ret.mean() * 100, 3), "% / 標準偏差", round(ret.std() * 100, 2), "%")
print("尖度:", round(stats.kurtosis(ret), 2), "（正規分布は0）")
print("最大下落:", round(ret.min() * 100, 2), "% / 最大上昇:", round(ret.max() * 100, 2), "%")

# --- ベースラインと2つのモデル --------------------------------------------
train, test = ret[:-60], ret[-60:]
naive = np.zeros(len(test))  # 明日のリターンは0（前日と同じ値段）と予測

arima = ARIMA(train, order=(1, 0, 0)).fit()
arima_pred = arima.forecast(len(test))

df = pd.DataFrame({"ret": ret})
for lag in (1, 2, 3, 5, 10):
    df[f"lag{lag}"] = df["ret"].shift(lag)
df["vol10"] = df["ret"].shift(1).rolling(10).std()
df["vol20"] = df["ret"].shift(1).rolling(20).std()
df = df.dropna()
Xtr, Xte = df.drop(columns="ret").iloc[:-60], df.drop(columns="ret").iloc[-60:]
ytr, yte = df["ret"].iloc[:-60], df["ret"].iloc[-60:]
gbm = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7,
                        random_state=42, verbose=-1).fit(Xtr, ytr)
gbm_pred = gbm.predict(Xte)

print("\nモデル        MAE(リターン)   前日値比")
for name, p in [("前日値（0）", naive), ("ARIMA(1,0,0)", arima_pred), ("LightGBM", gbm_pred)]:
    print(f"{name:12s} {mae(yte, p):14.5f} {mae(yte, p) / mae(yte, naive):9.2f}")

# 当て方向（符号）の一致率も見る
for name, p in [("ARIMA(1,0,0)", arima_pred), ("LightGBM", gbm_pred)]:
    hit = float(np.mean(np.sign(p) == np.sign(yte)))
    print(f"{name} の方向一致率: {hit:.3f}")

# --- 図 -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].hist(ret * 100, bins=60, color=C_BLUE, alpha=0.75, density=True)
xs = np.linspace(ret.min() * 100, ret.max() * 100, 200)
axes[0].plot(xs, stats.norm.pdf(xs, ret.mean() * 100, ret.std() * 100), color=C_ORANGE, lw=1.2, label="正規分布")
axes[0].set_xlabel("日次リターン（%）")
axes[0].set_title("裾が厚い（ファットテール）", fontsize=10)
axes[0].legend(fontsize=8, frameon=False)

axes[1].plot(test.index, test * 100, color=C_GRAY, lw=1, label="実測")
axes[1].plot(test.index, gbm_pred * 100, color=C_ORANGE, lw=1, label="LightGBM")
axes[1].axhline(0, color="black", lw=0.6)
axes[1].set_ylabel("リターン（%）")
axes[1].set_title("予測はほぼ0の周りを動く", fontsize=10)
axes[1].legend(fontsize=8, frameon=False)
print("図:", savefig(fig, "ch16_finance"))
