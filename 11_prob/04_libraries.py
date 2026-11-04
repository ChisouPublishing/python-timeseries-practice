"""第11章：ライブラリが出す90%予測区間を、航空旅客数の24か月で評価する。

実行: python3 11_prob/04_libraries.py
"""

import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import C_BLUE, C_ORANGE, C_PURPLE, C_TEAL, load_airpassengers, savefig, seed_everything

warnings.filterwarnings("ignore")
seed_everything(42)  # Prophetの区間はシミュレーションで作るため乱数を固定する
logging.getLogger("cmdstanpy").disabled = True
y = load_airpassengers()
train, test = y[:-24], y[-24:]
intervals = {}

# 第6章のauto_arimaと同じ次数・定数項。対数で学習し、区間の端を指数で戻す
fc = SARIMAX(np.log(train), order=(2, 0, 0), seasonal_order=(0, 1, 1, 12), trend="c").fit(disp=False).get_forecast(24)
ci = np.exp(fc.conf_int(alpha=0.10))
intervals["SARIMA（対数）"] = (ci.iloc[:, 0].to_numpy(), ci.iloc[:, 1].to_numpy())

sf = StatsForecast(models=[AutoARIMA(season_length=12)], freq="MS")
out = sf.forecast(df=pd.DataFrame({"unique_id": "air", "ds": train.index, "y": np.log(train.to_numpy())}), h=24, level=[90])
intervals["statsforecast（対数）"] = (np.exp(out["AutoARIMA-lo-90"].to_numpy()), np.exp(out["AutoARIMA-hi-90"].to_numpy()))

m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.90)
m.fit(pd.DataFrame({"ds": train.index, "y": train.to_numpy()}))
fp = m.predict(m.make_future_dataframe(periods=24, freq="MS")).tail(24)
intervals["Prophet（原系列）"] = (fp["yhat_lower"].to_numpy(), fp["yhat_upper"].to_numpy())


def interval_score(y_true, lo, hi, alpha=0.10):
    """区間の幅に、外れた量の 2/α 倍を罰則として足す（小さいほど良い）。"""
    return np.mean((hi - lo) + 2 / alpha * (lo - y_true) * (y_true < lo) + 2 / alpha * (y_true - hi) * (y_true > hi))


actual = test.to_numpy()
for name, (lo, hi) in intervals.items():
    inside = (actual >= lo) & (actual <= hi)
    print(f"{name}: 被覆率={inside.mean():.1%}（{inside.sum()}/24か月） 平均幅={np.mean(hi - lo):.1f}"
          f" 区間スコア={interval_score(actual, lo, hi):.1f} 24か月すべて区間内={'はい' if inside.all() else 'いいえ'}")

fig, ax = plt.subplots(figsize=(9, 3.4))
for (name, (lo, hi)), color in zip(intervals.items(), [C_ORANGE, C_TEAL, C_PURPLE]):
    ax.fill_between(test.index, lo, hi, color=color, alpha=0.15, label=name)
ax.plot(test.index, actual, color=C_BLUE, lw=1.4, label="実測")
ax.set_ylabel("旅客数（千人）")
ax.legend(ncol=4, fontsize=8, frameon=False)
ax.set_title("ライブラリが出す90%予測区間（24か月先まで）", fontsize=10)
print("図:", savefig(fig, "ch11_libraries"))
