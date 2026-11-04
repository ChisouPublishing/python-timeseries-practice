"""第9章：深層学習（window化・LSTM・多ステップ・学習の安定化）。

実行: python3 09_dl/01_lstm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import C_BLUE, C_GRAY, C_ORANGE, load_retail, mae, savefig, seed_everything

seed_everything(42)
torch.manual_seed(42)

ts = load_retail(store_id=1).set_index("date").sort_index().asfreq("D")
raw = ts["sales"].to_numpy(dtype="float32")
y = ts["sales"].ffill().to_numpy(dtype="float32")
CUTOFF = int(len(y) * 0.8) - 1
print("学習目的変数の最終許容日:", ts.index[CUTOFF].date())

# --- 9.1 window 化 ---------------------------------------------------------
def make_windows(series, targets, lookback=28, horizon=1):
    X, Y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i:i + lookback])
        Y.append(targets[i + lookback:i + lookback + horizon])
    return np.array(X), np.array(Y)

def split_windows(X, Y, lookback, horizon):
    first = np.arange(len(Y)) + lookback
    last = first + horizon - 1
    observed = np.isfinite(Y).all(axis=1)
    train = (last <= CUTOFF) & observed
    test = (first > CUTOFF) & observed
    # 入力の過去部分の共有は許可。学習の目的日がテスト起点を越えない。
    assert last[train].max() < first[test].min()
    assert np.isfinite(X).all()
    print(f"{horizon}期: 学習{train.sum()}窓 / テスト{test.sum()}窓 / 境界除外{((first <= CUTOFF) & (last > CUTOFF)).sum()}窓")
    print(f"{horizon}期: 学習最終目的日 {ts.index[last[train].max()].date()} / テスト最初の予測起点 {ts.index[first[test].min()-1].date()}")
    # 7日周期のベースライン。h<=7なので全て予測起点までに観測可能。
    naive = np.array([y[f-7:f-7+horizon] for f in first[test]])
    assert all(f - 7 + horizon - 1 <= f - 1 for f in first[test])
    return train, test, naive

LOOKBACK, HORIZON = 28, 1
X, Y = make_windows(y, raw, LOOKBACK, HORIZON)
X = X[..., None]
print("X:", X.shape, "/ Y:", Y.shape)
tr, te, naive = split_windows(X, Y, LOOKBACK, HORIZON)
mu, sd = X[tr].mean(), X[tr].std()
Ymu1, Ysd1 = Y[tr].mean(), Y[tr].std()
Xtr = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
Ytr = torch.tensor((Y[tr] - Ymu1) / Ysd1, dtype=torch.float32)
Xte = torch.tensor((X[te] - mu) / sd, dtype=torch.float32)

# --- 9.2 LSTM -------------------------------------------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMRegressor()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()
history = []
for epoch in range(120):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(Xtr), Ytr)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    history.append(loss.item())

model.eval()
with torch.no_grad():
    pred_z = model(Xte).numpy().ravel()
pred = pred_z * Ysd1 + Ymu1
actual = Y[te].ravel()
print("最終の訓練損失:", round(history[-1], 4))
print("LSTM MAE:", round(mae(actual, pred), 1))
print("同一対象日の季節ナイーブ MAE:", round(mae(actual, naive.ravel()), 1))

# --- 9.3 多ステップ（直接7期） --------------------------------------------
X7, Y7 = make_windows(y, raw, LOOKBACK, 7)
tr7, te7, naive7 = split_windows(X7, Y7, LOOKBACK, 7)
mu7, sd7 = X7[tr7].mean(), X7[tr7].std()
Ymu, Ysd = Y7[tr7].mean(), Y7[tr7].std()

class MultiStepLSTM(nn.Module):
    def __init__(self, hidden=32, horizon=7):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

m7 = MultiStepLSTM()
opt7 = torch.optim.Adam(m7.parameters(), lr=1e-2)
Xtr7 = torch.tensor(((X7[tr7] - mu7) / sd7)[..., None], dtype=torch.float32)
Ytr7 = torch.tensor((Y7[tr7] - Ymu) / Ysd, dtype=torch.float32)
Xte7 = torch.tensor(((X7[te7] - mu7) / sd7)[..., None], dtype=torch.float32)
for epoch in range(120):
    m7.train()
    opt7.zero_grad()
    loss = loss_fn(m7(Xtr7), Ytr7)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m7.parameters(), 1.0)
    opt7.step()

m7.eval()
with torch.no_grad():
    p7 = m7(Xte7).numpy() * Ysd + Ymu
print("7期先モデル MAE（全ホライズン平均）:", round(mae(Y7[te7], p7), 1))
print("ホライズン別 MAE:", [round(mae(Y7[te7][:, h], p7[:, h]), 0) for h in range(7)])

print("7期: 同一窓の季節ナイーブ MAE:", round(mae(Y7[te7], naive7), 1))

# --- 図 -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].plot(history, color=C_BLUE)
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("訓練MSE（標準化スケール）")
axes[0].set_title("学習曲線（単調に下がれば学習は安定）", fontsize=10)

ax = axes[1]
ax.plot(actual[:60], color=C_BLUE, label="実測")
ax.plot(pred[:60], color=C_ORANGE, label="LSTM")
ax.set_ylabel("売上（円）")
ax.legend(fontsize=8, frameon=False)
ax.set_title("テスト期間の先頭60日", fontsize=10)
print("図:", savefig(fig, "ch09_lstm"))
