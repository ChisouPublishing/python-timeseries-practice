"""第11章：CRPSの定義どおりの値と、分位点による近似の差を確かめる。CRPSが適切なスコアであることも数値で見る。

実行: python3 11_prob/06_crps_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import norm

from common import pinball_loss
from prob_data import load_splits


def crps_normal(y, mu, sd):
    """正規分布 N(mu, sd^2) のCRPSの閉形式（付録A.5）。"""
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_quantiles(y, mu, sd, taus):
    """分位点のPinball lossを平均して2倍する近似。taus は0〜1を等間隔に区切った中点を想定する。"""
    return 2 * np.mean([pinball_loss(y, norm.ppf(t, mu, sd), t) for t in taus])


def crps_energy(y, mu, sd, rng, n=4000):
    """エネルギー形式 E|X-y| - E|X-X'|/2 をサンプルで近似する。"""
    x1 = rng.normal(mu[:, None], sd, size=(len(y), n))
    x2 = rng.normal(mu[:, None], sd, size=(len(y), n))
    return np.mean(np.abs(x1 - y[:, None]).mean(axis=1) - 0.5 * np.abs(x1 - x2).mean(axis=1))


splits = load_splits()
(Xtr, ytr), (Xcal, ycal), (Xte, yte) = splits["train"], splits["cal"], splits["test"]
# 前週同曜日の値を中心に、較正期間の誤差の標準偏差を幅にした正規分布を予測分布とする
sd = float(np.std(ycal - Xcal["lag7"], ddof=1))
mu, y = Xte["lag7"].to_numpy(), yte.to_numpy()
print(f"予測分布: N(前週同曜日, {sd:.1f}^2) / 最終テスト{len(y)}日")

exact = np.mean(crps_normal(y, mu, sd))
print(f"閉形式のCRPS: {exact:.1f}")
print(f"エネルギー形式（4000サンプル）: {crps_energy(y, mu, sd, np.random.default_rng(0)):.1f}")
for name, taus in [("19分位点 0.05〜0.95", np.round(np.arange(0.05, 0.96, 0.05), 2)),
                   ("20分位点 中点0.025〜0.975", (np.arange(20) + 0.5) / 20),
                   ("99分位点 0.01〜0.99", np.round(np.arange(0.01, 1.0, 0.01), 2))]:
    approx = crps_quantiles(y, mu, sd, taus)
    print(f"{name}: {approx:.1f}（閉形式との差 {approx - exact:+.1f}）")
print(f"参考 中心だけを使うMAE: {np.mean(np.abs(y - mu)):.1f}")

# 適切なスコア: 実績が N(0,1) から出るとき、平均CRPSが最小になるのは同じ N(0,1) を予測したとき
truth = np.random.default_rng(1).normal(0, 1, 100_000)
scores = {s: np.mean(crps_normal(truth, 0.0, s)) for s in (0.5, 0.8, 1.0, 1.25, 2.0)}
print("真の分布 N(0,1) に対し、予測の標準偏差を変えたときの平均CRPS:",
      " / ".join(f"sd={s}: {v:.4f}" for s, v in scores.items()))
