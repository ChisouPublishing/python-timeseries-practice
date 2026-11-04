"""第11章で共通に使う小売データの特徴量と期間（第8章 02_optuna_search.py と同じ特徴量）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import load_retail

# 学習・較正・テストを時系列順に分ける。テストは第8章の最終テストと同じ期間
TRAIN_END = pd.Timestamp("2024-04-30")
CAL_START, CAL_END = pd.Timestamp("2024-05-01"), pd.Timestamp("2024-07-31")
TEST_START = pd.Timestamp("2024-08-01")


def features(frame):
    history = frame["sales"].ffill()  # 入力の補完は過去の値だけ
    out = pd.DataFrame(index=frame.index)
    for k in (1, 7, 14, 28):
        out[f"lag{k}"] = history.shift(k)
    for k in (7, 28):
        out[f"ma{k}"] = history.shift(1).rolling(k).mean()
        out[f"sd{k}"] = history.shift(1).rolling(k).std()
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month
    out["promo"] = frame["promo"]  # 当日の販促計画が前日までに確定するという教材上の前提
    return out


def load_splits():
    """(X, y) を学習・較正・テストの辞書で返す。正解が欠損した日は除く。"""
    raw = load_retail(store_id=1).set_index("date").asfreq("D")
    X = features(raw).dropna()
    y = raw["sales"].reindex(X.index)
    keep = y.notna()
    X, y = X.loc[keep], y.loc[keep]
    parts = {"train": X.index <= TRAIN_END,
             "cal": (X.index >= CAL_START) & (X.index <= CAL_END),
             "test": X.index >= TEST_START}
    return {name: (X.loc[mask], y.loc[mask]) for name, mask in parts.items()}
