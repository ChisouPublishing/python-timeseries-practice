"""第6章：手動SARIMAとauto_arimaの差を、次数・差分・定数項に分けて確かめる。

実行: python3 06_arima/02_order_check.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import load_airpassengers, mae

warnings.filterwarnings("ignore")
y = load_airpassengers()
train, test = y[:-24], y[-24:]

# 01_sarima.py と同じ24か月・対数変換で、次数と定数項だけを変える
# 1行目だけは 01_sarima.py と同じく定常性・反転可能性の制約を外す
candidates = [
    ("本文の手動設定（制約なし）", (1, 1, 1), (1, 1, 1, 12), "n"),
    ("本文の手動設定（制約あり）", (1, 1, 1), (1, 1, 1, 12), "n"),
    ("航空旅客モデル", (0, 1, 1), (0, 1, 1, 12), "n"),
    ("auto_arimaと同じ次数・定数なし", (2, 0, 0), (0, 1, 1, 12), "n"),
    ("auto_arimaと同じ次数・定数あり", (2, 0, 0), (0, 1, 1, 12), "c"),
]
for i, (label, order, seasonal, trend) in enumerate(candidates):
    free = {"enforce_stationarity": False, "enforce_invertibility": False} if i == 0 else {}
    model = SARIMAX(np.log(train), order=order, seasonal_order=seasonal, trend=trend, **free).fit(disp=False)
    pred = np.exp(model.get_forecast(steps=24).predicted_mean)
    total_diff = order[1] + seasonal[1]
    print(f"{label}: {order}{seasonal} 定数={'あり' if trend == 'c' else 'なし'} "
          f"差分の合計={total_diff} AIC={model.aic:.2f} MAE={mae(test, pred):.1f}")
