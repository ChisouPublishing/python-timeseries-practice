"""第12章：異常検知（Zスコア・変化点・Isolation Forest・オートエンコーダ）。

実行: python3 12_anomaly/01_detect.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve

from common import C_BLUE, C_GRAY, C_ORANGE, C_RED, C_TEAL, load_sensor, savefig, seed_everything

warnings.filterwarnings("ignore")
seed_everything(42)
torch.manual_seed(42)

s = load_sensor()
s["vibration"] = s["vibration"].interpolate(method="time")
print("期間:", s.index.min(), "〜", s.index.max(), "/", len(s), "点")
print("異常ラベルの点数:", int(s["label"].sum()))

# --- 11.1 統計的手法 -------------------------------------------------------
# 全体の平均・標準偏差を使うZスコア（トレンド・周期に弱い）
z_global = (s["vibration"] - s["vibration"].mean()) / s["vibration"].std()
flag_global = z_global.abs() > 3

# 直近30分の移動統計を使うZスコア
ma = s["vibration"].rolling(30, min_periods=5).mean()
sd = s["vibration"].rolling(30, min_periods=5).std()
z_local = (s["vibration"] - ma) / sd

def score(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1

for name, flag in [("全体Zスコア", flag_global.fillna(False)), ("移動Zスコア", (z_local.abs() > 4).fillna(False))]:
    p, r, f1 = score(s["label"], flag.astype(int))
    print(f"{name:10s} Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")

# 変化点検出（振動の平均が変わる時点）
algo = rpt.Pelt(model="l2").fit(s["vibration"].to_numpy())
cps = algo.predict(pen=5)
print("検出した変化点:", [str(s.index[min(c, len(s) - 1)]) for c in cps[:3]], "… 計", len(cps))

# --- 11.2 Isolation Forest -------------------------------------------------
feat = s[["vibration", "temperature"]].copy()
feat["vib_ma30"] = s["vibration"].rolling(30).mean()
feat["vib_sd30"] = s["vibration"].rolling(30).std()
feat = feat.bfill()
iso = IsolationForest(n_estimators=200, contamination=0.012, random_state=42).fit(feat)
s["iso_score"] = -iso.decision_function(feat)
s["iso_flag"] = (iso.predict(feat) == -1).astype(int)
p, r, f1 = score(s["label"], s["iso_flag"])
print(f"IsolationForest Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")
print("AP:", round(average_precision_score(s["label"], s["iso_score"]), 3))

# --- 11.3 オートエンコーダ -------------------------------------------------
WIN = 30
v = s["vibration"].to_numpy(dtype="float32")
windows = np.stack([v[i:i + WIN] for i in range(len(v) - WIN + 1)])
normal_end = np.searchsorted(s.index, np.datetime64("2024-06-07 00:00"))
X = torch.tensor(windows, dtype=torch.float32)
X_tr = X[: normal_end - WIN]

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=WIN, latent=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, latent), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, input_dim))

    def forward(self, x):
        return self.dec(self.enc(x))

ae = AutoEncoder()
opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()
for epoch in range(300):
    ae.train()
    opt.zero_grad()
    loss = loss_fn(ae(X_tr), X_tr)
    loss.backward()
    opt.step()
print("AEの最終訓練損失:", round(loss.item(), 5))

ae.eval()
with torch.no_grad():
    err = ((ae(X) - X) ** 2).mean(dim=1).numpy()
err_series = np.full(len(v), np.nan)
err_series[WIN - 1:] = err  # 窓の末尾の時刻に割り当てる
thr = np.percentile(err[: len(X_tr)], 99.9)
s["ae_error"] = err_series
s["ae_flag"] = (s["ae_error"] > thr).astype(int)
p, r, f1 = score(s["label"], s["ae_flag"])
print(f"AutoEncoder   Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")
print("AEの閾値（正常期間の99.9%点）:", round(float(thr), 5))

# 検出遅延（最初の異常点から、最初に検知するまでの分数）
first_ab = s.index[s["label"] == 1][0]
def delay_min(flag):
    hit = s.index[(s["label"] == 1) & (flag == 1)]
    return int((hit[0] - first_ab).total_seconds() / 60) if len(hit) else None
print("検出遅延（分） IF:", delay_min(s["iso_flag"]), "/ AE:", delay_min(s["ae_flag"]))

# --- 11.4 評価（PR曲線） ---------------------------------------------------
prec, rec, _ = precision_recall_curve(s["label"], s["iso_score"])
ap = average_precision_score(s["label"], s["iso_score"])
print("正解率（すべて正常と予測）:", round(1 - s["label"].mean(), 4), "← 役に立たない指標")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
ax = axes[0]
w = slice("2024-06-07 09:00", "2024-06-07 17:00")
ax.plot(s.loc[w].index, s.loc[w, "vibration"], color=C_BLUE, lw=0.9, label="振動")
anom = s.loc[w][s.loc[w, "label"] == 1]
ax.scatter(anom.index, anom["vibration"], s=6, color=C_RED, label="既知の異常")
ax.set_title("異常区間の振動（5日目 12:00〜14:00）", fontsize=10)
ax.tick_params(axis="x", labelrotation=20)
ax.legend(fontsize=8, frameon=False)

ax = axes[1]
ax.plot(rec, prec, color=C_ORANGE)
ax.set_xlabel("再現率")
ax.set_ylabel("適合率")
ax.set_title(f"PR曲線（AP={ap:.3f}）", fontsize=10)
print("図:", savefig(fig, "ch12_anomaly"))
