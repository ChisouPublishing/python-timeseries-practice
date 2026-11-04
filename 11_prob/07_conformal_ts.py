"""第11章：時系列向けのConformal（EnbPI・NexCP）を、04節の Split Conformal と同じ151日で比べる。

実行: python3 11_prob/07_conformal_ts.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap

from prob_data import load_splits

warnings.filterwarnings("ignore")
ALPHA = 0.10
splits = load_splits()
(Xtr, ytr), (Xcal, ycal), (Xte, yte) = splits["train"], splits["cal"], splits["test"]
y = yte.to_numpy()


def lgbm():
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=42, verbose=-1)


def report(name, lo, hi):
    covered = (y >= lo) & (y <= hi)
    months = pd.Series(covered, index=yte.index).groupby(yte.index.month).mean()
    print(f"{name}: 被覆率={covered.mean():.1%} / 平均幅={np.mean(hi - lo):.0f} / 月別の最小={months.min():.1%}")


# 基準: 04節の Split Conformal（学習期間で学習し、較正期間の絶対誤差の分位点を固定で使う）
point = lgbm().fit(Xtr, ytr)
pred, cal_err = point.predict(Xte), np.abs(ycal - point.predict(Xcal)).to_numpy()
q = np.sort(cal_err)[int(np.ceil((len(cal_err) + 1) * (1 - ALPHA))) - 1]
report("Split Conformal（固定）", pred - q, pred + q)

# EnbPI: 学習＋較正の544日を7日単位のブロックでブートストラップし、外れた日の誤差を使う
Xfit, yfit = pd.concat([Xtr, Xcal]), pd.concat([ytr, ycal])
cv = BlockBootstrap(n_resamplings=20, length=7, overlapping=False, random_state=42)
enbpi = TimeSeriesRegressor(lgbm(), method="enbpi", cv=cv, agg_function="mean").fit(Xfit, yfit)
_, pis = enbpi.predict(Xte, ensemble=True, confidence_level=1 - ALPHA, allow_infinite_bounds=True)
report("EnbPI（誤差を更新しない）", pis[:, 0, 0], pis[:, 1, 0])

# EnbPI（逐次更新）: 実績が届くたびに最も古い誤差を捨て、新しい誤差を入れてから翌日を予測する
lo, hi = np.empty(len(y)), np.empty(len(y))
for t in range(len(y)):
    _, pi = enbpi.predict(Xte.iloc[[t]], ensemble=True, confidence_level=1 - ALPHA, allow_infinite_bounds=True)
    lo[t], hi[t] = pi[0, 0, 0], pi[0, 1, 0]
    enbpi.update(Xte.iloc[[t]], yte.iloc[[t]], ensemble=True)
report("EnbPI（逐次更新）", lo, hi)


def weighted_quantile(scores, weights, level):
    """重み付きの分位点。新しい点の重み1は無限大の誤差に置く（NexCPの定義）。"""
    order = np.argsort(scores)
    cum = np.cumsum(weights[order]) / (weights.sum() + 1.0)
    idx = np.searchsorted(cum, level)
    return np.inf if idx >= len(scores) else scores[order][idx]


# NexCP: 過去の誤差に、新しいほど大きい重み rho^(経過日数) を付ける。rho はテストを見る前に0.99と決めた
rho, history = 0.99, list(cal_err)
lo, hi = np.empty(len(y)), np.empty(len(y))
for t in range(len(y)):
    scores = np.array(history)
    weights = rho ** np.arange(len(scores))[::-1]
    q_t = weighted_quantile(scores, weights, 1 - ALPHA)
    lo[t], hi[t] = pred[t] - q_t, pred[t] + q_t
    history.append(abs(y[t] - pred[t]))  # 実績が確定してから誤差を加える
report(f"NexCP（rho={rho}、逐次更新）", lo, hi)
