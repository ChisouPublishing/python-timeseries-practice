"""第8章：機械学習による予測（特徴量・LightGBM・時系列CV・評価・リークの実測）。

実行: python3 08_ml/01_lightgbm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit

from common import C_BLUE, C_GRAY, C_ORANGE, C_TEAL, load_retail, mae, mape, rmse, savefig

ts = load_retail(store_id=1).set_index("date").sort_index().asfreq("D")
ts["y"] = ts["sales"].ffill()
ts["promo"] = ts["promo"].fillna(0)
ts["dow"] = ts.index.dayofweek
ts["month"] = ts.index.month
ts["is_weekend"] = (ts["dow"] >= 5).astype(int)

# --- 8.1 教師あり学習への変換 ---------------------------------------------
def make_features(d: pd.DataFrame, target: str = "y") -> pd.DataFrame:
    out = d.copy()
    for lag in (1, 2, 7, 14, 28):
        out[f"lag{lag}"] = out[target].shift(lag)
    for w in (7, 28):
        out[f"ma{w}"] = out[target].shift(1).rolling(w).mean()
        out[f"sd{w}"] = out[target].shift(1).rolling(w).std()
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    return out.dropna()

feat = make_features(ts)
cols = [c for c in feat.columns if c not in ("y", "sales", "is_imputed")]
X, y = feat[cols], feat["y"]
split = int(len(feat) * 0.8)
X_tr, X_te, y_tr, y_te = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
print("学習:", len(X_tr), "日 / テスト:", len(X_te), "日（テストは未来側）")

# --- 8.2 ベースライン ------------------------------------------------------
seasonal_naive = feat["lag7"].iloc[split:]
ma7_base = feat["ma7"].iloc[split:]
print("季節ナイーブ MAE:", round(mae(y_te, seasonal_naive), 1))
print("7日移動平均   MAE:", round(mae(y_te, ma7_base), 1))

# --- 8.3 LightGBM ---------------------------------------------------------
model = lgb.LGBMRegressor(
    n_estimators=400, learning_rate=0.05, num_leaves=15,
    min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
    random_state=42, verbose=-1,
).fit(X_tr, y_tr)
pred = model.predict(X_te)
print("LightGBM      MAE:", round(mae(y_te, pred), 1),
      "/ RMSE:", round(rmse(y_te, pred), 1),
      "/ MAPE:", round(mape(y_te, pred), 1), "%")
print("改善率（対 季節ナイーブ）:", round((1 - mae(y_te, pred) / mae(y_te, seasonal_naive)) * 100, 1), "%")

# --- 8.4 時系列CVと、ランダムCVの過大評価 ---------------------------------
def cv_score(cv, X, y):
    scores = []
    for tr, te in cv.split(X):
        m = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, random_state=42, verbose=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        scores.append(mae(y.iloc[te], m.predict(X.iloc[te])))
    return float(np.mean(scores)), float(np.std(scores))

ts_mean, ts_std = cv_score(TimeSeriesSplit(n_splits=5), X_tr, y_tr)
kf_mean, kf_std = cv_score(KFold(n_splits=5, shuffle=True, random_state=42), X_tr, y_tr)
print(f"TimeSeriesSplit MAE: {ts_mean:.1f} ± {ts_std:.1f}")
print(f"ランダムKfold   MAE: {kf_mean:.1f} ± {kf_std:.1f}  ← 未来が学習に混ざる")

# --- 8.5 特徴量重要度 -----------------------------------------------------
imp = pd.Series(model.feature_importances_, index=cols).sort_values()
print(imp.tail(6).to_string())

# --- 図 -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
ax = axes[0]
ax.plot(y_te.index, y_te, color=C_BLUE, lw=1, label="実測")
ax.plot(y_te.index, pred, color=C_ORANGE, lw=1, label="LightGBM")
ax.plot(y_te.index, seasonal_naive, color=C_GRAY, lw=0.8, ls="--", label="季節ナイーブ")
ax.set_ylabel("売上（円）")
ax.legend(fontsize=8, frameon=False)
ax.set_title("テスト期間（未来側）の予測", fontsize=10)

ax = axes[1]
bars = [("季節\nナイーブ", mae(y_te, seasonal_naive), C_GRAY),
        ("7日\n移動平均", mae(y_te, ma7_base), C_TEAL),
        ("LightGBM", mae(y_te, pred), C_ORANGE)]
ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars])
for i, b in enumerate(bars):
    ax.text(i, b[1], f"{b[1]:.0f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("MAE（円）")
ax.set_title("モデル比較（テスト）", fontsize=10)
print("図:", savefig(fig, "ch08_forecast"))

# ランダムCVのリークを図でも示す
fig, ax = plt.subplots(figsize=(7, 2.6))
ax.barh(["ランダムKfold", "TimeSeriesSplit"], [kf_mean, ts_mean], color=[C_GRAY, C_TEAL])
for i, v in enumerate([kf_mean, ts_mean]):
    ax.text(v, i, f" {v:.0f}", va="center", fontsize=9)
ax.set_xlabel("CVのMAE（円）")
ax.set_title("同じモデル・同じデータでも、分割でスコアが変わる", fontsize=10)
print("図:", savefig(fig, "ch08_leak"))
