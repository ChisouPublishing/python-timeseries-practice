"""第12章：事後の変化点検出（PELT・二分割）と、オンラインの変化点検出（BOCPD）を同じ振動データで比べる。

実行: python3 12_anomaly/03_changepoint.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.special import gammaln

from common import load_sensor

s = load_sensor()
v = s["vibration"].ffill()  # 入力の補完は過去の値だけ
onset, end = s.index[s.label == 1].min(), s.index[s.label == 1].max() + pd.Timedelta(minutes=1)
print(f"{len(v)}点（1分ごと）/ 既知の異常区間 {onset} 〜 {end}")


def fmt(times):
    return ", ".join(t.strftime("%m-%d %H:%M") for t in times)


# 学習期間（6月3〜5日）だけから、1時間以内のばらつきを求めて標準化する（水準のゆっくりした変化を含めない）
train = v[v.index < "2024-06-06"]
sd0 = np.sqrt(train.rolling(60).var().median())
z = ((v - train.mean()) / sd0).to_numpy()

# 事後: 7日分すべてを見てから区切る。ペナルティは事前に 3・log(点数) と決めた
pen = 3 * np.log(len(z))
for name, algo in [("PELT", rpt.Pelt(model="l2", min_size=30)), ("二分割（Binseg）", rpt.Binseg(model="l2", min_size=30))]:
    cps = [v.index[c] for c in algo.fit(z).predict(pen=pen)[:-1]]
    inside = [c for c in cps if onset - pd.Timedelta(minutes=30) <= c <= end + pd.Timedelta(minutes=30)]
    print(f"{name}（事後）: {len(cps)}個 / 異常区間の前後30分に{len(inside)}個（{fmt(inside)}）")


def bocpd(x, hazard, mu0, kappa0, alpha0, beta0, r_max=1500):
    """Adams & MacKay のBOCPD。平均・分散が未知の正規分布（正規-ガンマ事前分布）。
    各時刻で、その時刻までの観測だけから「直前の変化からの経過時間（run length）」の分布を更新する。"""
    log_r = np.array([0.0])  # run length の対数確率
    mu, kappa, alpha, beta = (np.array([p], float) for p in (mu0, kappa0, alpha0, beta0))
    short = np.empty(len(x))
    for t, xt in enumerate(x):
        # スチューデントのt分布による予測確率
        scale2 = beta * (kappa + 1) / (alpha * kappa)
        logp = (gammaln(alpha + 0.5) - gammaln(alpha) - 0.5 * np.log(2 * np.pi * alpha * scale2)
                - (alpha + 0.5) * np.log1p((xt - mu) ** 2 / (2 * alpha * scale2)))
        grow = log_r + logp + np.log(1 - hazard)
        cp = np.logaddexp.reduce(log_r + logp + np.log(hazard))
        log_r = np.append(cp, grow)[:r_max]
        log_r -= np.logaddexp.reduce(log_r)
        short[t] = np.exp(np.logaddexp.reduce(log_r[:10]))  # 直近10分以内に変化した確率
        # 事後分布の更新（run length ごと）。先頭は変化直後の事前分布
        mu_n = (kappa * mu + xt) / (kappa + 1)
        beta_n = beta + kappa * (xt - mu) ** 2 / (2 * (kappa + 1))
        mu = np.append(mu0, mu_n)[:r_max]
        kappa = np.append(kappa0, kappa + 1)[:r_max]
        alpha = np.append(alpha0, alpha + 0.5)[:r_max]
        beta = np.append(beta0, beta_n)[:r_max]
    return short


# 変化は平均12時間に1回と仮定する。事前分布の平均・分散も学習期間の標準化に合わせる
short = bocpd(z, hazard=1 / 720, mu0=0.0, kappa0=0.01, alpha0=10.0, beta0=10.0)
alarm = short > 0.5
alarm[:60] = False  # 開始直後は経過時間0から始まるので、最初の60分は判定しない
starts = v.index[alarm & ~np.roll(alarm, 1)]  # 連続する警報は最初の時刻だけ数える
near = (starts >= onset) & (starts <= end + pd.Timedelta(minutes=30))
print(f"BOCPD（オンライン）: 警報{len(starts)}回 / 異常区間とその後30分に{near.sum()}回 / それ以外{(~near).sum()}回（{fmt(starts[~near])}）")
for name, t0 in [("異常の開始", onset), ("異常の終了", end)]:
    hit = starts[(starts >= t0) & (starts < t0 + pd.Timedelta(minutes=30))]
    print(f"BOCPD {name}の検出遅れ: {'検出なし' if hit.empty else f'{int((hit[0] - t0).total_seconds() // 60)}分'}")
