"""第11章：ニューラルネットで分位点と正規分布を出力し、90%区間を比べる。

実行: python3 11_prob/03_nn.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn

from common import pinball_loss, seed_everything
from prob_data import load_splits

seed_everything(42)
torch.manual_seed(42)
splits = load_splits()
(Xtr, ytr), (Xte, yte) = splits["train"], splits["test"]

# 標準化の平均・標準偏差は学習期間だけから求める
x_mu, x_sd = Xtr.mean(), Xtr.std().replace(0, 1)
y_mu, y_sd = ytr.mean(), ytr.std()
to_t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
xtr, xte = to_t((Xtr - x_mu) / x_sd), to_t((Xte - x_mu) / x_sd)
ytr_t = to_t((ytr - y_mu) / y_sd).unsqueeze(1)
TAUS = torch.tensor([0.05, 0.5, 0.95])


def mlp(n_out):
    return nn.Sequential(nn.Linear(xtr.shape[1], 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, n_out))


def quantile_loss(out, y):
    diff = y - out
    return torch.mean(torch.maximum(TAUS * diff, (TAUS - 1) * diff))


def gaussian_nll(out, y):
    mean, log_sd = out[:, :1], out[:, 1:]
    return torch.mean(log_sd + 0.5 * ((y - mean) / torch.exp(log_sd)) ** 2)


results = {}
for name, n_out, loss_fn in [("分位点NN", 3, quantile_loss), ("正規分布NN", 2, gaussian_nll)]:
    net = mlp(n_out)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(600):  # 学習データだけで固定回数学習し、テストで止める時点を選ばない
        opt.zero_grad()
        loss = loss_fn(net(xtr), ytr_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = net(xte).numpy()
    if name == "分位点NN":
        lo, mid, hi = (np.sort(out, axis=1) * y_sd + y_mu).T
    else:
        mid, sd = out[:, 0] * y_sd + y_mu, np.exp(out[:, 1]) * y_sd
        lo, hi = mid - 1.645 * sd, mid + 1.645 * sd  # 正規分布の5%点・95%点
    results[name] = (lo, mid, hi)

y = yte.to_numpy()
for name, (lo, mid, hi) in results.items():
    covered = np.mean((y >= lo) & (y <= hi))
    print(f"{name}: 中央のMAE={np.mean(np.abs(y - mid)):.1f} / 90%区間 被覆率={covered:.1%} / 平均幅={np.mean(hi - lo):.0f}"
          f" / Pinball τ=0.95={pinball_loss(y, hi, 0.95):.1f}")
