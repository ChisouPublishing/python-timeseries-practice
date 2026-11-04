"""第10章：同じ「AutoARIMA」の結果が違う理由を、入力の変換と選ばれた次数に分けて確かめる。

実行: python3 10_libs/02_autoarima_conditions.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pmdarima as pm
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from common import load_airpassengers, mae

warnings.filterwarnings("ignore")
y = load_airpassengers()
train, test = y[:-24], y[-24:]


def run_pmdarima(series):
    # 第6章 01_sarima.py と同じ探索範囲
    model = pm.auto_arima(
        series, seasonal=True, m=12,
        start_p=0, max_p=2, start_q=0, max_q=2,
        start_P=0, max_P=1, start_Q=0, max_Q=1,
        d=None, D=None, stepwise=True, error_action="ignore", suppress_warnings=True,
    )
    order = f"{model.order}{model.seasonal_order}"
    return order, model.with_intercept, model.predict(n_periods=24)


def run_statsforecast(series):
    frame = pd.DataFrame({"unique_id": "air", "ds": train.index, "y": np.asarray(series)})
    sf = StatsForecast(models=[AutoARIMA(season_length=12)], freq="MS")
    sf.fit(frame)
    fitted = sf.fitted_[0, 0].model_
    p, q, P, Q, m, d, D = fitted["arma"]
    has_constant = any(name in fitted["coef"] for name in ("intercept", "drift"))
    pred = sf.predict(h=24)["AutoARIMA"].to_numpy()
    return f"({p}, {d}, {q})({P}, {D}, {Q}, {m})", has_constant, pred


for lib, runner in [("pmdarima", run_pmdarima), ("statsforecast", run_statsforecast)]:
    for scale in ["対数", "原系列"]:
        series = np.log(train) if scale == "対数" else train
        order, constant, pred = runner(series)
        pred = np.exp(pred) if scale == "対数" else pred
        print(f"{lib} / {scale}: {order} 定数={'あり' if constant else 'なし'} MAE={mae(test, pred):.1f}")
