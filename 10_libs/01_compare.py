"""第10章：時系列ライブラリの使い分け（Prophet / statsforecast / sktime / Darts）。

実行: python3 10_libs/01_compare.py
"""

import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import C_BLUE, C_GRAY, C_ORANGE, C_PURPLE, C_TEAL, load_airpassengers, mae, savefig

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled = True

y = load_airpassengers()
train, test = y[:-24], y[-24:]
results = {}
preds = {}

# --- 10.1 Prophet ---------------------------------------------------------
from prophet import Prophet

df_p = pd.DataFrame({"ds": train.index, "y": train.to_numpy()})
m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
m.fit(df_p)
future = m.make_future_dataframe(periods=24, freq="MS")
fc = m.predict(future)
preds["Prophet"] = pd.Series(fc["yhat"].to_numpy()[-24:], index=test.index)
results["Prophet"] = mae(test, preds["Prophet"])

# --- 10.2 statsforecast ---------------------------------------------------
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS

df_sf = pd.DataFrame({"unique_id": "air", "ds": train.index, "y": train.to_numpy()})
sf = StatsForecast(models=[AutoARIMA(season_length=12), AutoETS(season_length=12)], freq="MS")
sf.fit(df_sf)
out = sf.predict(h=24).reset_index()
preds["statsforecast"] = pd.Series(out["AutoARIMA"].to_numpy(), index=test.index)
results["statsforecast"] = mae(test, preds["statsforecast"])
print(out.head(3).round(1).to_string(index=False))

# --- 10.3 sktime ----------------------------------------------------------
from sktime.forecasting.arima import AutoARIMA as SktimeAutoARIMA

y_train = pd.Series(train.to_numpy(), index=pd.PeriodIndex(train.index, freq="M"))
sk = SktimeAutoARIMA(sp=12, suppress_warnings=True)
sk.fit(y_train)
fh = np.arange(1, 25)
preds["sktime"] = pd.Series(sk.predict(fh=fh).to_numpy(), index=test.index)
results["sktime"] = mae(test, preds["sktime"])

# --- 10.4 Darts -----------------------------------------------------------
from darts import TimeSeries
from darts.models import AutoARIMA as DartsAutoARIMA

series = TimeSeries.from_series(train)
dmodel = DartsAutoARIMA(seasonal=True, season_length=12)
dmodel.fit(series)
preds["Darts"] = pd.Series(dmodel.predict(24).values().ravel(), index=test.index)
results["Darts"] = mae(test, preds["Darts"])

# --- 10.5 比較 ------------------------------------------------------------
naive = np.tile(train[-12:].to_numpy(), 2)
print("\nモデル            MAE")
print(f"{'季節ナイーブ':14s} {mae(test, naive):6.1f}")
for k, v in sorted(results.items(), key=lambda kv: kv[1]):
    print(f"{k:14s} {v:6.1f}")

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(test.index, test, color="black", lw=1.6, label="実測")
colors = {"Prophet": C_ORANGE, "statsforecast": C_TEAL, "sktime": C_PURPLE, "Darts": C_BLUE}
for k, p in preds.items():
    ax.plot(test.index, p, color=colors[k], lw=1, label=k)
ax.set_ylabel("旅客数（千人）")
ax.legend(ncol=5, fontsize=8, frameon=False)
ax.set_title("同じデータ・同じ24か月を4ライブラリで予測", fontsize=10)
print("図:", savefig(fig, "ch10_compare"))
