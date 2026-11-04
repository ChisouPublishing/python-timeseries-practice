"""第3章：読み込み・頻度・タイムゾーン・欠損・可視化。

実行: python3 03_io/01_read_visualize.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from common import C_BLUE, C_GRAY, C_ORANGE, C_TEAL, load_airpassengers, load_retail, savefig

# --- 3.1 datetime とインデックス -------------------------------------------
raw = load_retail()
print(raw.dtypes.to_string())
# format を明示すると速く、誤変換も防げる（ここでは既に Timestamp なので文字列化して再変換）
s = pd.to_datetime(raw["date"].astype(str), format="%Y-%m-%d")
print("変換後の型:", s.dtype)
ts = raw[raw["store_id"] == 1].set_index("date").sort_index()
print("インデックス:", type(ts.index).__name__, "/", ts.index.freq)

# 欠損日を補って、はじめて「毎日1行」が保証される
full = ts.asfreq("D")
print("asfreq後の行数:", len(full), "/ 頻度:", full.index.freq)

# --- 3.2 頻度・タイムゾーン・DST -------------------------------------------
ny = pd.Series(
    [1, 2, 3],
    index=pd.to_datetime(["2024-03-09", "2024-03-10", "2024-03-11"]),
).tz_localize("America/New_York")
print("DST切替日を含む日次:", list(ny.index))
print("UTCへ変換:", list(ny.index.tz_convert("UTC")))

# 2024-03-10 は現地時間で 23 時間しかない
night = pd.date_range("2024-03-10", periods=2, freq="D", tz="America/New_York")
print("現地の1日:", night[0], "〜", night[1],
      "/ 実時間:", (night[1] - night[0]).total_seconds() / 3600, "時間")

# --- 3.3 欠損・重複・不規則 -----------------------------------------------
print("欠損:", int(ts["sales"].isna().sum()), "件")
print("重複した日付:", int(ts.index.duplicated().sum()), "件")

fig, ax = plt.subplots(figsize=(9, 1.8))
mask = ts["sales"].isna().to_numpy().reshape(1, -1)
ax.imshow(mask, aspect="auto", cmap="Greys", interpolation="nearest")
ax.set_yticks([])
ax.set_xlabel("日付（2023-01-01〜2024-12-31）")
ax.set_title("欠損の位置（黒＝欠損）", fontsize=10)
print("図:", savefig(fig, "ch03_missing"))

# --- 3.4 可視化：折れ線・STL ----------------------------------------------
y = load_airpassengers()
stl = STL(y, period=12, robust=True).fit()
fig = stl.plot()
fig.set_size_inches(9, 6)
fig.suptitle("AirPassengers の STL 分解（period=12）", fontsize=11)
print("図:", savefig(fig, "ch03_stl"))
print("季節成分の振幅:", round(float(stl.seasonal.max() - stl.seasonal.min()), 1))

# --- 3.5 複数系列の重ね方 --------------------------------------------------
wide = load_retail().pivot(index="date", columns="store_id", values="sales")
z = (wide - wide.mean()) / wide.std()
fig, ax = plt.subplots(figsize=(9, 3))
styles = {1: (C_BLUE, "-"), 2: (C_ORANGE, "--"), 3: (C_TEAL, ":")}
for store, (color, ls) in styles.items():
    ax.plot(z.index, z[store], color=color, linestyle=ls, linewidth=0.8, label=f"店舗{store}")
ax.set_ylabel("標準化した売上")
ax.legend(ncol=3, fontsize=9, frameon=False, loc="upper left")
ax.set_title("水準の違う3系列を標準化して重ねる", fontsize=11)
print("図:", savefig(fig, "ch03_overlay"))
