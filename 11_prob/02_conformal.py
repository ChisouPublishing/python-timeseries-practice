"""第11章：Split Conformal・CQR・ACIで区間を較正し、被覆率を逐次監視する。

実行: python3 11_prob/02_conformal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import C_BLUE, C_GRAY, C_ORANGE, C_TEAL, OUT_DIR, savefig
from prob_data import load_splits

ALPHA = 0.10  # 目標の被覆率は90%
splits = load_splits()
(Xtr, ytr), (Xcal, ycal), (Xte, yte) = splits["train"], splits["cal"], splits["test"]
print(f"学習={len(ytr)}日 / 較正={len(ycal)}日（{Xcal.index.min().date()}〜{Xcal.index.max().date()}）/ テスト={len(yte)}日")


def lgbm(**kw):
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=42, verbose=-1, **kw)


def conformal_quantile(scores, alpha):
    """較正スコアの小さい方から ceil((n+1)(1-α)) 番目の値。n が小さく番号が n を超えるなら無限大。"""
    scores = np.sort(np.asarray(scores))
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return np.inf if rank > len(scores) else scores[rank - 1]


def report(name, lo, hi):
    y = yte.to_numpy()
    covered = (y >= lo) & (y <= hi)
    print(f"{name}: 被覆率={covered.mean():.1%} / 平均幅={np.mean(hi - lo):.0f}")
    return covered


# (1) Split Conformal: 点予測モデル＋較正期間の絶対誤差
point = lgbm().fit(Xtr, ytr)
q_split = conformal_quantile(np.abs(ycal - point.predict(Xcal)), ALPHA)
pred = point.predict(Xte)
print(f"Split Conformal: 較正{len(ycal)}日の絶対誤差の{int(np.ceil((len(ycal) + 1) * (1 - ALPHA)))}番目 = {q_split:.0f}")
cov_split = report("Split Conformal", pred - q_split, pred + q_split)

# (2) CQR: 分位点回帰の区間を、較正期間のはみ出し量で広げる（または狭める）
lo_m, hi_m = lgbm(objective="quantile", alpha=ALPHA / 2).fit(Xtr, ytr), lgbm(objective="quantile", alpha=1 - ALPHA / 2).fit(Xtr, ytr)
lo_raw, hi_raw = lo_m.predict(Xte), hi_m.predict(Xte)
report("分位点回帰（較正なし）", np.minimum(lo_raw, hi_raw), np.maximum(lo_raw, hi_raw))
scores = np.maximum(lo_m.predict(Xcal) - ycal, ycal - hi_m.predict(Xcal))
q_cqr = conformal_quantile(scores, ALPHA)
print(f"CQR: 較正期間のはみ出し量の補正 = {q_cqr:+.0f}")
report("CQR", np.minimum(lo_raw, hi_raw) - q_cqr, np.maximum(lo_raw, hi_raw) + q_cqr)

# (3) ACI: 実績が届くたびに、外れたら次の区間を広げ、当たったら少し狭める
gamma, alpha_t, cal_err = 0.01, ALPHA, np.abs(ycal - point.predict(Xcal)).to_numpy()
lo_aci, hi_aci = np.empty(len(yte)), np.empty(len(yte))
for t, y_t in enumerate(yte.to_numpy()):
    q_t = conformal_quantile(cal_err, alpha_t) if alpha_t > 0 else np.inf
    q_t = min(q_t, cal_err.max() * 3)  # 無限大の区間は表示できないため上限を置く
    lo_aci[t], hi_aci[t] = pred[t] - q_t, pred[t] + q_t
    alpha_t += gamma * (ALPHA - float(not lo_aci[t] <= y_t <= hi_aci[t]))
report("ACI", lo_aci, hi_aci)

# (4) 被覆率の逐次監視: 直近30日の被覆率が80%を下回った日を数える
rolling = pd.Series(cov_split.astype(float), index=yte.index).rolling(30).mean()
alerts = rolling[rolling < 0.80]
print(f"Split Conformalの直近30日被覆率: 最小={rolling.min():.1%} / 80%未満の日={len(alerts)}日"
      + (f"（最初={alerts.index[0].date()}）" if len(alerts) else ""))
for month, part in pd.Series(cov_split, index=yte.index).groupby(yte.index.month):
    print(f"  {month}月: 被覆率={part.mean():.1%}")

pd.DataFrame({"actual": yte, "point": pred, "split_lo": pred - q_split, "split_hi": pred + q_split,
              "aci_lo": lo_aci, "aci_hi": hi_aci}).to_csv(OUT_DIR / "11_prob_conformal.csv", index_label="date")

fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
ax = axes[0]
ax.fill_between(yte.index, pred - q_split, pred + q_split, color=C_ORANGE, alpha=0.2, label="Split Conformal 90%")
ax.plot(yte.index, lo_aci, color=C_TEAL, lw=0.8, label="ACI 90%")
ax.plot(yte.index, hi_aci, color=C_TEAL, lw=0.8)
ax.plot(yte.index, yte, color=C_BLUE, lw=1, label="実績")
ax.set_ylabel("売上（円）")
ax.legend(ncol=3, fontsize=8, frameon=False)
ax.set_title("較正した90%区間", fontsize=10)
ax = axes[1]
ax.plot(rolling.index, rolling, color=C_ORANGE, label="Split Conformal 直近30日の被覆率")
ax.axhline(0.9, color=C_GRAY, ls="--", lw=0.8, label="目標90%")
ax.axhline(0.8, color=C_GRAY, ls=":", lw=0.8, label="警告80%")
ax.set_ylim(0.5, 1.02)
ax.legend(ncol=3, fontsize=8, frameon=False)
ax.set_title("被覆率の逐次監視", fontsize=10)
print("図:", savefig(fig, "ch11_conformal"))
