"""第16章 ケース2：センサー異常検知（製造業）。

実行: python3 16_cases/02_sensor.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

from common import C_BLUE, C_GRAY, C_ORANGE, C_RED, load_sensor, savefig, seed_everything

warnings.filterwarnings("ignore")
seed_everything(42)
torch.manual_seed(42)

s = load_sensor().interpolate(method="time")
y = s["label"].to_numpy()

def evaluate(flag):
    tp = int(((y == 1) & (flag == 1)).sum())
    fp = int(((y == 0) & (flag == 1)).sum())
    fn = int(((y == 1) & (flag == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    first = s.index[y == 1][0]
    hit = s.index[(y == 1) & (flag == 1)]
    delay = int((hit[0] - first).total_seconds() / 60) if len(hit) else -1
    return p, r, f1, delay

# --- Isolation Forest ------------------------------------------------------
feat = s[["vibration", "temperature"]].copy()
feat["vib_ma30"] = s["vibration"].rolling(30).mean()
feat["vib_sd30"] = s["vibration"].rolling(30).std()
feat = feat.bfill()
iso = IsolationForest(n_estimators=200, contamination=0.012, random_state=42).fit(feat)
s["iso_score"] = -iso.decision_function(feat)
iso_flag = (s["iso_score"] > s["iso_score"].quantile(0.988)).astype(int)

# --- AutoEncoder（正常期間のみで学習） ------------------------------------
WIN = 30
v = s["vibration"].to_numpy(dtype="float32")
W = np.stack([v[i:i + WIN] for i in range(len(v) - WIN + 1)])
normal_end = int(np.searchsorted(s.index, np.datetime64("2024-06-07 00:00")))
X = torch.tensor(W, dtype=torch.float32)
Xtr = X[: normal_end - WIN]

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(WIN, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, WIN))

    def forward(self, x):
        return self.dec(self.enc(x))

ae = AE()
opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
for _ in range(300):
    opt.zero_grad()
    loss = ((ae(Xtr) - Xtr) ** 2).mean()
    loss.backward()
    opt.step()
with torch.no_grad():
    err = ((ae(X) - X) ** 2).mean(dim=1).numpy()
ae_err = np.full(len(v), np.nan)
ae_err[WIN - 1:] = err
s["ae_score"] = ae_err
thr = np.percentile(err[: len(Xtr)], 99.9)
ae_flag = (s["ae_score"] > thr).astype(int)

print("手法             Precision Recall   F1   遅延(分)")
for name, flag in [("IsolationForest", iso_flag), ("AutoEncoder", ae_flag)]:
    p, r, f1, d = evaluate(flag.to_numpy())
    print(f"{name:16s} {p:8.3f} {r:6.3f} {f1:6.3f} {d:7d}")

# アンサンブル：和集合（どちらか）と積集合（両方）
for label, ens in [("和集合(OR)", (iso_flag + ae_flag > 0).astype(int)),
                   ("積集合(AND)", (iso_flag * ae_flag > 0).astype(int))]:
    p, r, f1, d = evaluate(ens.to_numpy())
    print(f"{label:16s} {p:8.3f} {r:6.3f} {f1:6.3f} {d:7d}")

# --- 図 -------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True)
w = slice("2024-06-07 08:00", "2024-06-07 18:00")
axes[0].plot(s.loc[w].index, s.loc[w, "vibration"], color=C_BLUE, lw=0.9)
axes[0].fill_between(s.loc[w].index, 0, 1, where=s.loc[w, "label"] == 1,
                     transform=axes[0].get_xaxis_transform(), color=C_RED, alpha=0.12, label="既知の異常")
axes[0].set_ylabel("振動")
axes[0].legend(fontsize=8, frameon=False)
axes[1].plot(s.loc[w].index, s.loc[w, "iso_score"], color=C_ORANGE, label="Isolation Forest")
axes[1].plot(s.loc[w].index, s.loc[w, "ae_score"] * 1000, color=C_GRAY, label="AE誤差×1000")
axes[1].set_ylabel("異常スコア")
axes[1].legend(fontsize=8, frameon=False)
for ax in axes:
    ax.tick_params(axis="x", labelrotation=20)
print("図:", savefig(fig, "ch16_sensor"))
