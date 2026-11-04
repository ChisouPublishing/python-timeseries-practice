"""付録用の合成販売数量。公開小売データの結果とは区別する。"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def series():
    rng = np.random.default_rng(42)
    t = np.arange(180)
    return pd.Series(100 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 7)
                     + rng.normal(0, 2, len(t)),
                     index=pd.date_range("2024-01-01", periods=len(t), freq="D"), name="units")


def features():
    y = series()
    frame = pd.DataFrame({"target": y, "lag1": y.shift(1), "lag7": y.shift(7),
                          "mean7": y.shift(1).rolling(7).mean(),
                          "dow": y.index.dayofweek}).dropna()
    return frame[["lag1", "lag7", "mean7", "dow"]], frame["target"]
