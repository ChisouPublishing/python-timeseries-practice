"""statsmodels：SARIMAXの推定、予測、区間、残差診断。"""
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from _data import series

y = series()
train, test = y.iloc[:-28], y.iloc[-28:]
model = SARIMAX(train, order=(1, 0, 0), seasonal_order=(1, 0, 0, 7), trend="c")
fitted = model.fit(disp=False, maxiter=200)
forecast = fitted.get_forecast(steps=len(test))
mean = forecast.predicted_mean
interval = forecast.conf_int(alpha=0.1)
mae = np.mean(np.abs(test - mean))
covered = ((test >= interval.iloc[:, 0]) & (test <= interval.iloc[:, 1])).mean()
# 初期状態の影響を避ける教材上の除外。モデルごとに必要な範囲は異なる。
diagnostic = acorr_ljungbox(fitted.resid.iloc[14:], lags=[14], model_df=2, return_df=True)
print("収束:", bool(fitted.mle_retvals["converged"]))
print("予測開始:", mean.index[0].date(), "/ 予測行数:", len(mean))
print(f"最終28日 MAE={mae:.3f} / 90%区間の実測被覆={covered:.1%}")
print(f"Ljung-Box（14ラグ・自由度補正2）p値={diagnostic.lb_pvalue.iloc[0]:.3f}")
inside = (interval.iloc[:, 0] <= mean) & (mean <= interval.iloc[:, 1])
print("区間の下限<=予測<=上限:", bool(inside.all()))
assert fitted.mle_retvals["converged"]
assert mean.index.equals(test.index)
