"""Matplotlib：系列・予測区間・軸・保存。区間はSARIMAXで計算する。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import plt, C_BLUE, C_ORANGE, savefig
from _data import series
from statsmodels.tsa.statespace.sarimax import SARIMAX

y = series()
train, test = y.iloc[:-28], y.iloc[-28:]
fitted = SARIMAX(train, order=(1, 0, 0), seasonal_order=(1, 0, 0, 7),
                 trend="c").fit(disp=False, maxiter=200)
forecast = fitted.get_forecast(steps=len(test))
mean = forecast.predicted_mean
interval = forecast.conf_int(alpha=0.1)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(train.index[-42:], train.iloc[-42:], label="学習", color=C_BLUE)
ax.plot(test.index, test, label="実測", color="black")
ax.plot(mean.index, mean, label="予測", color=C_ORANGE)
ax.fill_between(mean.index, interval.iloc[:, 0], interval.iloc[:, 1],
                color=C_ORANGE, alpha=0.2, label="モデルの90%予測区間")
ax.set(xlabel="日付", ylabel="販売数量（合成値）")
ax.tick_params(labelsize=11)
ax.xaxis.label.set_size(12)
ax.yaxis.label.set_size(12)
ax.legend(frameon=False, fontsize=10.5, ncol=2)
fig.autofmt_xdate()
print("保存:", savefig(fig, "appendix_basics_forecast"))
print("実測・予測・区間の対象日一致:", mean.index.equals(test.index))
assert fitted.mle_retvals["converged"]
assert mean.index.equals(test.index)
