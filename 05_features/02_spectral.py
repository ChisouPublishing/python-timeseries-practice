"""第5章：ピリオドグラム（素朴な推定とWelch法）と、2系列のクロススペクトル（コヒーレンス・位相）。

実行: python3 05_features/02_spectral.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from common import C_BLUE, C_GRAY, C_ORANGE, load_retail, load_web_traffic, savefig

pv = load_web_traffic()
x = pv.to_numpy(dtype=float)
print(f"Webトラフィック: {len(x)}日")

# 素朴なピリオドグラム（直線のトレンドを除く）と、Welch法（区間を分けて平均し、ばらつきを抑える）
f_raw, p_raw = signal.periodogram(x, fs=1.0, detrend="linear")
f_w, p_w = signal.welch(x, fs=1.0, nperseg=182, detrend="linear")


def peaks(f, p, k):
    """ピリオドグラムの局所的な山を、パワーの大きい順に k 個返す（周期の日数）。"""
    idx = signal.find_peaks(p)[0]
    top = idx[np.argsort(p[idx])[::-1][:k]]
    return [round(float(1 / f[i]), 1) for i in top]


print("素朴なピリオドグラムの山（周期・日）:", peaks(f_raw, p_raw, 5))
print("Welch法（182日ずつ）の山（周期・日）:", peaks(f_w, p_w, 3))
print(f"周波数の刻み: 素朴={f_raw[1]:.5f}（1/{1 / f_raw[1]:.0f}日）/ Welch={f_w[1]:.5f}（1/{1 / f_w[1]:.0f}日）")

# クロススペクトル: 店舗1と店舗2の日次売上。欠損は過去の値で補う
wide = load_retail().pivot(index="date", columns="store_id", values="sales").ffill()
a, b = wide[1].to_numpy(), wide[2].to_numpy()
f_c, coh = signal.coherence(a, b, fs=1.0, nperseg=56, detrend="linear")
_, pxy = signal.csd(a, b, fs=1.0, nperseg=56, detrend="linear")
for period in (7.0, 3.5):
    i = np.argmin(np.abs(f_c - 1 / period))
    print(f"店舗1×店舗2 周期{1 / f_c[i]:.1f}日: コヒーレンス={coh[i]:.2f} / 位相差={np.degrees(np.angle(pxy[i])):+.0f}度")
low = f_c < 1 / 14
print(f"店舗1×店舗2 周期14日より長い成分のコヒーレンス（平均）={coh[low & (f_c > 0)].mean():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].semilogy(f_raw[1:], p_raw[1:], color=C_GRAY, lw=0.6, label="素朴なピリオドグラム")
axes[0].semilogy(f_w[1:], p_w[1:], color=C_BLUE, lw=1.4, label="Welch法")
for period in (7, 3.5):
    axes[0].axvline(1 / period, color=C_ORANGE, ls="--", lw=0.8)
axes[0].set_xlabel("周波数（1/日）")
axes[0].set_ylabel("パワー（対数）")
axes[0].set_title("Webトラフィック", fontsize=10)
axes[0].legend(fontsize=8, frameon=False)
axes[1].plot(f_c, coh, color=C_BLUE, lw=1.4)
for period in (7, 3.5):
    axes[1].axvline(1 / period, color=C_ORANGE, ls="--", lw=0.8)
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("周波数（1/日）")
axes[1].set_ylabel("コヒーレンス")
axes[1].set_title("店舗1と店舗2の売上", fontsize=10)
print("図:", savefig(fig, "ch05_spectral"))
