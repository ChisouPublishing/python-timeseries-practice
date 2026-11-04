"""第13章：分類・クラスタリング（ROCKET・DTW・k-means・モチーフ発見）。

実行: python3 13_classify/01_classify_cluster.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stumpy
from sktime.classification.kernel_based import RocketClassifier
from sktime.datasets import load_unit_test
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw

from common import C_BLUE, C_GRAY, C_ORANGE, C_PURPLE, C_TEAL, load_web_traffic, savefig, seed_everything

warnings.filterwarnings("ignore")
seed_everything(42)

# --- 12.1 ROCKET による時系列分類 -----------------------------------------
X_train, y_train = load_unit_test(split="train")
X_test, y_test = load_unit_test(split="test")
print("学習:", X_train.shape, "/ テスト:", X_test.shape, "/ クラス:", sorted(set(y_train)))
clf = RocketClassifier(num_kernels=1000, random_state=42)
clf.fit(X_train, y_train)
print("ROCKET 正解率:", round(clf.score(X_test, y_test), 3))

# --- 12.2 距離（ユークリッド vs DTW） -------------------------------------
base = np.sin(np.linspace(0, 2 * np.pi, 60))
shifted = np.sin(np.linspace(0.8, 2 * np.pi + 0.8, 60))
noisy = base + np.random.default_rng(0).normal(0, 0.1, 60)
eu_shift = np.linalg.norm(base - shifted)
dtw_shift = dtw(base, shifted)
eu_noise = np.linalg.norm(base - noisy)
dtw_noise = dtw(base, noisy)
print(f"位相ずれ: ユークリッド={eu_shift:.2f} / DTW={dtw_shift:.2f}")
print(f"同一位相＋ノイズ: ユークリッド={eu_noise:.2f} / DTW={dtw_noise:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].plot(base, color=C_BLUE, label="系列A")
axes[0].plot(shifted, color=C_ORANGE, ls="--", label="系列B（位相ずれ）")
axes[0].set_title("形は同じだが時間軸がずれている", fontsize=10)
axes[0].legend(fontsize=8, frameon=False)
axes[1].bar(["ユークリッド", "DTW"], [eu_shift, dtw_shift], color=[C_GRAY, C_TEAL])
axes[1].set_ylabel("距離")
axes[1].set_title("位相ずれに強いのはDTW", fontsize=10)
print("図:", savefig(fig, "ch13_dtw"))

# --- 12.3 クラスタリング（DTW k-means） -----------------------------------
rng = np.random.default_rng(7)
t = np.linspace(0, 1, 48)
shapes = [
    np.sin(2 * np.pi * t),
    np.sign(np.sin(2 * np.pi * t)),
    2 * t - 1,
]
series, truth = [], []
for k, shape in enumerate(shapes):
    for _ in range(20):
        series.append(shape + rng.normal(0, 0.12, len(t)))
        truth.append(k)
series = np.array(series)[..., None]
km = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=42, n_init=3)
labels = km.fit_predict(series)
print("クラスタごとの件数:", np.bincount(labels).tolist())

fig, ax = plt.subplots(figsize=(7, 3.2))
for k, color in enumerate([C_BLUE, C_ORANGE, C_TEAL]):
    for i in np.where(labels == k)[0][:8]:
        ax.plot(series[i, :, 0], color=color, alpha=0.35, lw=0.9)
    ax.plot(km.cluster_centers_[k, :, 0], color=color, lw=2.4, label=f"クラスタ{k}")
ax.legend(fontsize=8, frameon=False)
ax.set_title("DTW k-means のクラスタとセントロイド", fontsize=10)
print("図:", savefig(fig, "ch13_clusters"))

# --- 12.4 モチーフ発見（行列プロファイル） --------------------------------
pv = load_web_traffic()
x = pv.to_numpy(dtype=float)
m = 7  # 1週間の形
mp = stumpy.stump(x, m=m)
motif_idx = int(np.argmin(mp[:, 0].astype(float)))
nn_idx = int(mp[motif_idx, 1])
print("モチーフ開始:", pv.index[motif_idx].date(), "／ 類似区間:", pv.index[nn_idx].date(),
      "／ 距離:", round(float(mp[motif_idx, 0]), 1))

fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(pv.index, pv.values, color=C_GRAY, lw=0.8)
ax.plot(pv.index[motif_idx:motif_idx + m], pv.values[motif_idx:motif_idx + m], color=C_ORANGE, lw=2.4, label="モチーフ")
ax.plot(pv.index[nn_idx:nn_idx + m], pv.values[nn_idx:nn_idx + m], color=C_BLUE, lw=2.4, label="最も類似した区間")
ax.legend(fontsize=8, frameon=False)
ax.set_title("行列プロファイルで見つけた1週間モチーフ", fontsize=10)
print("図:", savefig(fig, "ch13_motif"))
