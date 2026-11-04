"""第1章：時系列データの全体像を、1本の系列で確かめる。

実行: python3 01_overview/01_series_basics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from common import C_BLUE, C_GRAY, load_retail, savefig

# 店舗1の日次売上を読む（long形式：date, store_id, sales, promo）
df = load_retail(store_id=1)
print(df.shape)
print(df.head(3).to_string(index=False))

# 日付を datetime にし、インデックスへ。並べ替えまでが1組
df["date"] = pd.to_datetime(df["date"])
ts = df.set_index("date").sort_index()
print(ts.index[:3])
print("期間:", ts.index.min().date(), "〜", ts.index.max().date(), "/", len(ts), "日")

# 行の並びそのものに意味がある＝シャッフルすると壊れる
shuffled = ts.sample(frac=1, random_state=42)
print("自己相関（1日ラグ）  元の順序:", round(ts["sales"].autocorr(1), 3),
      "／ シャッフル後:", round(shuffled["sales"].autocorr(1), 3))

# 欠損と基本統計
print("欠損:", int(ts["sales"].isna().sum()), "件")
print(ts["sales"].describe().round(1).to_string())

# 図1-1：水準・週次・年末スパイクが1枚で見える
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(ts.index, ts["sales"], color=C_BLUE, linewidth=0.8)
ax.set_ylabel("日次売上（円）")
ax.set_title("店舗1の日次売上（2023-01〜2024-12）", fontsize=11)
ax.axvspan(pd.Timestamp("2023-12-29"), pd.Timestamp("2024-01-01"), color=C_GRAY, alpha=0.15)
ax.annotate("年末スパイク", xy=(pd.Timestamp("2023-12-30"), ts["sales"].max()),
            xytext=(-70, -28), textcoords="offset points", fontsize=9, color=C_GRAY,
            arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=0.8))
print("図:", savefig(fig, "ch01_retail_series"))
