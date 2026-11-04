"""第16章 ケース4：Webトラフィック予測（Prophet と LightGBM）。

実行: python3 16_cases/04_web.py
"""

import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet

from common import C_BLUE, C_GRAY, C_ORANGE, load_web_traffic, mae, mape, savefig

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").disabled = True

pv = load_web_traffic()
train, test = pv[:-30], pv[-30:]
print("テスト期間:", test.index.min().date(), "〜", test.index.max().date(), "/ 30日")

# --- ベースライン（7日移動平均） ------------------------------------------
ma7 = train.rolling(7).mean().iloc[-1]
ma7_pred = pd.Series(ma7, index=test.index)

# --- Prophet（祝日を明示） -------------------------------------------------
df_p = pd.DataFrame({"ds": train.index, "y": train.to_numpy()})
m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
m.add_country_holidays(country_name="JP")
m.fit(df_p)
future = m.make_future_dataframe(periods=30)
fc = m.predict(future)
prophet_pred = pd.Series(fc["yhat"].to_numpy()[-30:], index=test.index)

# --- LightGBM --------------------------------------------------------------
d = pd.DataFrame({"y": pv})
d["dow"] = d.index.dayofweek
d["month"] = d.index.month
d["day"] = d.index.day
d["is_weekend"] = (d["dow"] >= 5).astype(int)
for lag in (1, 7, 14, 21, 28):
    d[f"lag{lag}"] = d["y"].shift(lag)
d["ma7"] = d["y"].shift(1).rolling(7).mean()
d["ma28"] = d["y"].shift(1).rolling(28).mean()
d["dow_sin"] = np.sin(2 * np.pi * d["dow"] / 7)
d["dow_cos"] = np.cos(2 * np.pi * d["dow"] / 7)
d = d.dropna()
cols = [c for c in d.columns if c != "y"]
Xtr = d.loc[d.index <= train.index.max(), cols]
ytr = d["y"].iloc[:-30]
gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=15,
                        random_state=42, verbose=-1).fit(Xtr, ytr)
# 固定起点の30日予測。テスト中の実測を入力しない。
history = train.copy()
recursive = []
for date in test.index:
    row = {"dow":date.dayofweek,"month":date.month,"day":date.day,
           "is_weekend":int(date.dayofweek >= 5)}
    for lag in (1, 7, 14, 21, 28): row[f"lag{lag}"] = history.iloc[-lag]
    row["ma7"], row["ma28"] = history.iloc[-7:].mean(), history.iloc[-28:].mean()
    row["dow_sin"] = np.sin(2*np.pi*date.dayofweek/7)
    row["dow_cos"] = np.cos(2*np.pi*date.dayofweek/7)
    assert history.index.max() < date
    value = float(gbm.predict(pd.DataFrame([row],index=[date])[cols])[0])
    recursive.append(value)
    history.loc[date] = value
assert history.loc[test.index].equals(pd.Series(recursive,index=test.index))
gbm_pred = pd.Series(recursive, index=test.index)
print("評価方式: 2024-12-01固定起点の30日先まで。全モデルでテスト実測を入力しない")

print("モデル        MAE       MAPE")
for name, p in [("7日移動平均", ma7_pred), ("Prophet", prophet_pred), ("LightGBM", gbm_pred)]:
    print(f"{name:12s} {mae(test, p):8.0f} {mape(test, p):7.1f}%")

# イベント日（2024-10-01）はテスト外なので、年末年始の誤差を見る
err = (test - gbm_pred).abs().sort_values(ascending=False)
print("LightGBM の誤差上位3日:")
print(err.head(3).round(0).to_string())

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(train.index[-60:], train[-60:], color=C_GRAY, lw=0.9, label="訓練（直近60日）")
ax.plot(test.index, test, color="black", lw=1.4, label="実測")
ax.plot(test.index, prophet_pred, color=C_ORANGE, lw=1.2, label="Prophet")
ax.plot(test.index, gbm_pred, color=C_BLUE, lw=1.2, label="LightGBM")
ax.set_ylabel("PV")
ax.legend(ncol=4, fontsize=8, frameon=False)
ax.set_title("テスト30日の予測比較", fontsize=10)
print("図:", savefig(fig, "ch16_web"))
