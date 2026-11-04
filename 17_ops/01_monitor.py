"""第17章：運用（性能監視・ドリフト検出・パイプライン）。

実行: python3 17_ops/01_monitor.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import C_BLUE, C_GRAY, C_ORANGE, C_RED, load_web_traffic, mae, savefig

warnings.filterwarnings("ignore")
pv = load_web_traffic()

# --- 15.1 性能監視 ---------------------------------------------------------
# 暦月ごとの実績でMAEを計算し、閾値超えを警告にする
pred = pv.shift(1)  # 前日値予測（監視の例）
err = (pv - pred).abs().dropna()
monthly = err.resample("MS").mean()
threshold = float(err.iloc[:365].mean() * 1.2)
print("監視の閾値（基準365観測MAE×1.2）:", round(threshold, 0))
print(monthly.round(0).to_string())
alerts = monthly[monthly > threshold]
print("警告した月:", [d.strftime("%Y-%m") for d in alerts.index])

# --- 15.2 ドリフト検出 -----------------------------------------------------
# 「先頭600点の後の131日」を人為的に1.25倍へずらし、入力分布の変化を検出する
train_feature = pv.iloc[:600].to_numpy()
current = pv.iloc[600:].to_numpy() * 1.25
stat, p = ks_2samp(train_feature, current)
print(f"KS検定: statistic={stat:.3f} / p={p:.2e}")

def psi(base, cur, bins=10):
    base, cur = np.asarray(base, float), np.asarray(cur, float)
    if not len(base) or not len(cur) or not np.isfinite(base).all() or not np.isfinite(cur).all():
        raise ValueError("PSIは空でない有限値の配列を受け取る")
    inner = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.r_[-np.inf, inner, np.inf]
    b = np.histogram(base, bins=edges)[0].astype(float) + 1e-6
    c = np.histogram(cur, bins=edges)[0].astype(float) + 1e-6
    b, c = b / b.sum(), c / c.sum()
    assert np.isclose(b.sum(), 1) and np.isclose(c.sum(), 1)
    return float(np.sum((c - b) * np.log(c / b)))

assert np.isclose(psi(train_feature, train_feature), 0)
assert np.isfinite(psi(train_feature, np.array([-1e9, 1e9])))

value = psi(train_feature, current)
print("PSI:", round(value, 3), "（0.1超で注意、0.25超で要対応が目安）")

# --- 15.3 パイプライン -----------------------------------------------------
d = pd.DataFrame({"y": pv})
for lag in (1, 7, 14):
    d[f"lag{lag}"] = d["y"].shift(lag)
d["ma7"] = d["y"].shift(1).rolling(7).mean()
d = d.dropna()
X, y = d.drop(columns="y"), d["y"]
split = int(len(d) * 0.8)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)),
])
pipe.fit(X.iloc[:split], y.iloc[:split])
pred_pipe = pipe.predict(X.iloc[split:])
print("パイプライン MAE:", round(mae(y.iloc[split:], pred_pipe), 0))
print("スケーラーの平均（訓練のみでfit）:", round(float(pipe.named_steps["scaler"].mean_[0]), 0))

# --- 図 -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].hist(train_feature, bins=40, alpha=0.6, color=C_GRAY, label="訓練時", density=True)
axes[0].hist(current, bins=40, alpha=0.55, color=C_ORANGE, label="現在（+25%）", density=True)
axes[0].set_title("入力分布のずれ（データドリフト）", fontsize=10)
axes[0].legend(fontsize=8, frameon=False)

axes[1].plot(monthly.index, monthly.values, marker="o", ms=4, color=C_BLUE, label="月次MAE")
axes[1].axhline(threshold, color=C_RED, ls="--", lw=1, label="警告の閾値")
axes[1].set_ylabel("MAE")
axes[1].set_title("性能の監視と警告", fontsize=10)
axes[1].legend(fontsize=8, frameon=False)
print("図:", savefig(fig, "ch17_monitor"))
