"""第5章：特徴量エンジニアリング（ラグ・ローリング・カレンダー・周期・重要度）。

実行: python3 05_features/01_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import holidays
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import C_BLUE, C_GRAY, C_ORANGE, C_TEAL, load_retail, load_web_traffic, savefig

ts = load_retail(store_id=1).set_index("date").sort_index().asfreq("D")
ts["y"] = ts["sales"].ffill()  # 入力のみ過去値で補う。正解salesは変更しない

# --- 5.1 ラグ特徴量 --------------------------------------------------------
df = ts.copy()
for lag in (1, 7, 14, 28):
    df[f"lag{lag}"] = df["y"].shift(lag)
print(df[["y", "lag1", "lag7", "lag28"]].iloc[28:32].round(0).to_string())
print("lag1 と y の相関:", round(df["y"].corr(df["lag1"]), 3))
print("shift(-1) と y の相関:", round(df["y"].corr(df["y"].shift(-1)), 3), "← 未来の値。特徴量に使ってはいけない")

# --- 5.2 ローリング統計量 --------------------------------------------------
df["ma7"] = df["y"].shift(1).rolling(7).mean()      # 当期を含めない
df["sd7"] = df["y"].shift(1).rolling(7).std()
df["max28"] = df["y"].shift(1).rolling(28).max()
print(df[["y", "ma7", "sd7", "max28"]].iloc[30:33].round(1).to_string())

# --- 5.3 カレンダー・祝日 --------------------------------------------------
jp = holidays.Japan(years=range(2023, 2025))
df["dow"] = df.index.dayofweek
df["month"] = df.index.month
df["is_holiday"] = df.index.normalize().isin(pd.to_datetime(list(jp.keys()))).astype(int)
df["is_month_end"] = df.index.is_month_end.astype(int)
print("祝日数:", int(df["is_holiday"].sum()))
print("曜日別の平均売上:")
print(df.groupby("dow")["y"].mean().round(0).to_string())

# --- 5.4 周期特徴量（sin/cos） --------------------------------------------
df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
ax = axes[0]
ax.scatter(df["dow_sin"].iloc[:7], df["dow_cos"].iloc[:7], color=C_BLUE, s=40, zorder=3)
for i, (x, yv) in enumerate(zip(df["dow_sin"].iloc[:7], df["dow_cos"].iloc[:7])):
    ax.annotate("月火水木金土日"[df.index[i].dayofweek], (x, yv), textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_aspect("equal")
ax.set_title("曜日の sin/cos 表現（円周上に等間隔）", fontsize=10)

# FFT で主要な周期を見る（Webトラフィックの日次系列）
pv = load_web_traffic()
x = pv.to_numpy() - pv.mean()
power = np.abs(np.fft.rfft(x)) ** 2
freq = np.fft.rfftfreq(len(x), d=1.0)
period = 1 / np.where(freq > 0, freq, np.nan)
top = np.argsort(power[1:])[::-1][:6] + 1
ax = axes[1]
ax.plot(period[1:40], power[1:40], color=C_ORANGE)
for i in top[:4]:
    ax.axvline(period[i], color=C_GRAY, lw=0.6, ls="--")
    ax.annotate(f"{period[i]:.0f}日", (period[i], power[i]), textcoords="offset points",
                xytext=(4, 2), fontsize=8, color=C_GRAY)
ax.set_xscale("log")
ax.set_xlabel("周期（日）")
ax.set_ylabel("パワー")
ax.set_title("FFTで見る主要な周期", fontsize=10)
print("図:", savefig(fig, "ch05_cyclic"))
print("上位の周期:", [round(float(period[i]), 1) for i in top[:4]])

# --- 5.5 ターゲットエンコーディングのリーク --------------------------------
wide = load_retail()
wide["y"] = wide["sales"]
# 200 種類のカテゴリ（商品カテゴリのつもり）を乱数で割り当てる
rng = np.random.default_rng(0)
wide["cat"] = rng.integers(0, 200, len(wide))
cut = wide["date"] < "2024-01-01"
test = ~cut

# 全期間の平均で置換：テスト行の y も平均に入っている
naive = wide.groupby("cat")["y"].transform("mean")
# 訓練期間だけで作った平均を、テストへ写す
train_mean = wide[cut].groupby("cat")["y"].mean()
encoded = wide["cat"].map(train_mean).fillna(wide.loc[cut, "y"].mean())

# テスト期間だけで相関を見る（モデルが未知データでどれだけ当てられるかの代用）
print("全期間平均で置換  :", round(wide.loc[test, "y"].corr(naive[test]), 3), "← リークで過大")
print("訓練期間のみで置換:", round(wide.loc[test, "y"].corr(encoded[test]), 3), "← 正しい作り方")

# --- 5.6 特徴量重要度 ------------------------------------------------------
feat = df.copy()
feat["target"] = feat["sales"]
feat = feat.dropna()
cols = ["lag1", "lag7", "lag14", "lag28", "ma7", "sd7", "max28",
        "dow", "month", "is_holiday", "is_month_end", "dow_sin", "dow_cos"]
X, y = feat[cols], feat["target"]
split = int(len(feat) * 0.8)
model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                          random_state=42, verbose=-1).fit(X.iloc[:split], y.iloc[:split])
imp = pd.Series(model.feature_importances_, index=cols).sort_values()
print(imp.tail(8).to_string())

fig, ax = plt.subplots(figsize=(7, 3.6))
colors = [C_ORANGE if c.startswith("lag") else C_TEAL if c.startswith(("ma", "sd", "max")) else C_BLUE for c in imp.index]
ax.barh(imp.index, imp.values, color=colors)
ax.set_xlabel("重要度（split回数）")
ax.set_title("特徴量重要度（ラグ・ローリング・カレンダー）", fontsize=10)
print("図:", savefig(fig, "ch05_importance"))
