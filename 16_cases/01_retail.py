"""第16章 ケース1：小売需要予測（前日終了時点から翌日1期先）。

実行: python3 16_cases/01_retail.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import holidays
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import C_BLUE, C_GRAY, C_ORANGE, load_retail, mae, mape, rmse, savefig

store = load_retail(store_id=1).set_index("date").sort_index().asfreq("D")
print("期間:", store.index.min().date(), "〜", store.index.max().date(), "/ 欠損:", int(store["sales"].isna().sum()))
store["y"] = store["sales"].ffill()  # 補間した事実はフラグで残す
store["is_imputed"] = store["sales"].isna().astype(int)
store["promo"] = store["promo"].fillna(0).astype(int)

jp = holidays.Japan(years=range(2023, 2025))
store["dow"] = store.index.dayofweek
store["month"] = store.index.month
store["is_holiday"] = store.index.normalize().isin(pd.to_datetime(list(jp.keys()))).astype(int)
store["is_month_end"] = store.index.is_month_end.astype(int)

def make_features(d):
    out = d.copy()
    for lag in (1, 7, 14, 28):
        out[f"lag{lag}"] = out["y"].shift(lag)
    out["ma7"] = out["y"].shift(1).rolling(7).mean()
    out["sd7"] = out["y"].shift(1).rolling(7).std()
    out["ma28"] = out["y"].shift(1).rolling(28).mean()
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["target"] = out["sales"]  # この行の日付が予測対象日
    return out.dropna()

feat = make_features(store)
cols = [c for c in feat.columns if c not in ("y", "sales", "target")]
cutoff = store.index[int(len(store) * 0.8) - 1]
split = int((feat.index <= cutoff).sum())
Xtr, Xte = feat[cols].iloc[:split], feat[cols].iloc[split:]
ytr, yte = feat["target"].iloc[:split], feat["target"].iloc[split:]

assert Xtr.index.max() < Xte.index.min()
assert yte.notna().all()
print("学習最終目的日:", Xtr.index.max().date(), "/ テスト最初の目的日:", Xte.index.min().date())
print("評価日数:", len(yte), "（正解欠損は除外、モデルは再学習せず毎日前日の実測を更新）")
baseline = feat.loc[Xte.index, "lag7"]
ma7 = feat.loc[Xte.index, "ma7"]
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=15,
                          random_state=42, verbose=-1).fit(Xtr, ytr)
pred = model.predict(Xte)

rows = [("前週同曜日", baseline), ("7日移動平均", ma7), ("LightGBM", pred)]
print("モデル          MAE      RMSE    MAPE")
for name, p in rows:
    print(f"{name:12s} {mae(yte, p):8.0f} {rmse(yte, p):8.0f} {mape(yte, p):6.1f}%")
print("改善率（対 前週同曜日）:", round((1 - mae(yte, pred) / mae(yte, baseline)) * 100, 1), "%")

imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
print(imp.head(6).to_string())

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
ax = axes[0]
ax.plot(yte.index, yte, color=C_BLUE, lw=1, label="実測")
ax.plot(yte.index, pred, color=C_ORANGE, lw=1, label="LightGBM")
ax.plot(yte.index, baseline, color=C_GRAY, lw=0.8, ls="--", label="前週同曜日")
ax.set_ylabel("売上（円）")
ax.legend(fontsize=8, frameon=False)
ax.set_title("テスト期間（直近2割）", fontsize=10)

ax = axes[1]
err = pd.Series(yte.to_numpy() - pred, index=yte.index)
ax.bar(err.index, err.values, color=C_GRAY, width=1.0)
ax.axhline(0, color="black", lw=0.6)
ax.set_ylabel("誤差（実測−予測）")
ax.set_title("誤差に残る癖（年末・曜日）", fontsize=10)
print("図:", savefig(fig, "ch16_retail"))
print("誤差が大きい上位3日:")
print(err.abs().sort_values(ascending=False).head(3).round(0).to_string())
