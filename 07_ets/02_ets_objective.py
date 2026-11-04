"""第7章：Holt-Wintersの乗法モデルとETS(M,A,M)の差を、推定されたパラメータで確かめる。

実行: python3 07_ets/02_ets_objective.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from common import load_airpassengers, mae

warnings.filterwarnings("ignore")
y = load_airpassengers()
train, test = y[:-24], y[-24:]

# 01_ets.py と同じ設定。Holt-Wintersは二乗誤差の和、ETSModelは尤度を最大にする
models = {
    "Holt-Winters乗法": ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12,
                                         initialization_method="estimated").fit(),
    "ETS(A,A,M)": ETSModel(train, error="add", trend="add", seasonal="mul", seasonal_periods=12).fit(disp=False),
    "ETS(M,A,M)": ETSModel(train, error="mul", trend="add", seasonal="mul", seasonal_periods=12).fit(disp=False),
}
for name, model in models.items():
    params = dict(zip(model.model.param_names, model.params)) if name.startswith("ETS") else model.params
    alpha, beta, gamma = (params[k] for k in ("smoothing_level", "smoothing_trend", "smoothing_seasonal"))
    print(f"{name}: alpha={alpha:.3f} beta={beta:.3f} gamma={gamma:.3f} MAE={mae(test, model.forecast(24)):.1f}")
