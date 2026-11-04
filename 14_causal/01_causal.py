"""第14章：因果推論（Granger因果性・DiD・平行トレンドの確認）。

実行: python3 14_causal/01_causal.py
"""

import contextlib
import io
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import grangercausalitytests

from common import C_BLUE, C_GRAY, C_ORANGE, C_TEAL, savefig, seed_everything

warnings.filterwarnings("ignore")
seed_everything(42)

# --- 13.2 Granger因果性 ----------------------------------------------------
# 広告費と売上の2系列（広告が売上に先行する形で作る）
rng = np.random.default_rng(5)
n = 200
ad = np.zeros(n)
sales = np.zeros(n)
for t in range(1, n):
    ad[t] = 0.7 * ad[t - 1] + rng.normal(0, 1)
    sales[t] = 0.8 * sales[t - 1] + 2.0 * ad[t - 1] + rng.normal(0, 1)
with contextlib.redirect_stdout(io.StringIO()):
    g = grangercausalitytests(pd.DataFrame({"sales": sales, "ad": ad}), maxlag=3)
print("Granger（ad → sales）")
for lag in (1, 2, 3):
    print(f"  lag={lag}: p={g[lag][0]['ssr_ftest'][1]:.4f}")
with contextlib.redirect_stdout(io.StringIO()):
    g_rev = grangercausalitytests(pd.DataFrame({"ad": ad, "sales": sales}), maxlag=3)
print("Granger（sales → ad）")
for lag in (1, 2, 3):
    print(f"  lag={lag}: p={g_rev[lag][0]['ssr_ftest'][1]:.4f}")

# --- 13.3 DiD --------------------------------------------------------------
# 40店舗 × 60週。20店舗だけ30週目からキャンペーン（真の効果 +12%）
rng = np.random.default_rng(11)
weeks = np.arange(60)
rows = []
for store in range(40):
    treat = int(store < 20)
    base = rng.normal(1000, 80)
    trend = rng.normal(2.0, 0.5)
    for w in weeks:
        post = int(w >= 30)
        effect = 0.12 * base if (treat and post) else 0.0
        season = 40 * np.sin(2 * np.pi * w / 52)
        y = base + trend * w + season + effect + rng.normal(0, 30)
        rows.append((store, w, treat, post, y, effect))
df = pd.DataFrame(rows, columns=["store", "week", "treat", "post", "sales", "true_effect"])

# 平行トレンドの確認：介入前の傾きを群ごとに比べる
pre = df[df["post"] == 0]
slopes = pre.groupby("treat").apply(lambda d: np.polyfit(d["week"], d["sales"], 1)[0])
print("介入前の傾き（0=対照, 1=介入）:", slopes.round(2).to_dict())

model = smf.ols("sales ~ treat * post", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["store"]}, use_t=True
)
print("DiD推定値:", round(model.params["treat:post"], 1),
      "/ 95%CI:", np.round(model.conf_int().loc["treat:post"].to_numpy(), 1))
print("真の効果（介入群・介入後の生成値平均）:", round(df.loc[(df["treat"] == 1) & (df["post"] == 1), "true_effect"].mean(), 1))

# 見かけの前後差だけを見ると、季節要因で過大・過小になる
naive = df[df["treat"] == 1].groupby("post")["sales"].mean()
print("介入群の前後差（DiDなし）:", round(naive[1] - naive[0], 1))

fig, ax = plt.subplots(figsize=(8, 3.6))
for treat, color, label in [(0, C_GRAY, "対照群"), (1, C_ORANGE, "介入群")]:
    m = df[df["treat"] == treat].groupby("week")["sales"].mean()
    ax.plot(m.index, m.values, color=color, lw=1.6, label=label)
ax.axvline(30, color=C_BLUE, ls="--", lw=1)
ax.text(30.5, ax.get_ylim()[1] * 0.98, "介入", color=C_BLUE, fontsize=9, va="top")
ax.set_xlabel("週")
ax.set_ylabel("平均売上")
ax.legend(fontsize=8, frameon=False)
ax.set_title("DiD：介入前は平行、介入後に差が開く", fontsize=10)
print("図:", savefig(fig, "ch14_did"))
