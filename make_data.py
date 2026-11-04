"""書籍で使うデータを data/ に作る。

公開データ（月次航空旅客数）は同梱済み。小売・センサー・Web・株価は、
乱数シードを固定した合成データとしてここで生成する。
実行: python3 make_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA_DIR, seed_everything

seed_everything(42)


def make_retail() -> pd.DataFrame:
    """3店舗 × 2年（2023-01-01〜2024-12-31）の日次売上。"""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    base = {1: 12_000.0, 2: 8_500.0, 3: 15_000.0}
    dow_factor = np.array([0.88, 0.85, 0.90, 0.95, 1.10, 1.25, 1.15])  # 月〜日
    rows = []
    for store_id, b in base.items():
        t = np.arange(len(dates))
        trend = 1.0 + 0.00018 * t
        for i, d in enumerate(dates):
            f = dow_factor[d.dayofweek] * trend[i]
            # 年末商戦・お盆・給料日
            if (d.month, d.day) in [(12, 29), (12, 30), (12, 31)]:
                f *= 1.75
            if d.month == 8 and 13 <= d.day <= 16:
                f *= 1.20
            if d.day == 25:
                f *= 1.12
            promo = int(rng.random() < 0.03)
            if promo:
                f *= 1.35
            y = b * f * np.exp(rng.normal(0, 0.045))
            rows.append((d, store_id, round(y), promo))
    df = pd.DataFrame(rows, columns=["date", "store_id", "sales", "promo"])
    # 実データらしく 1% の欠損を入れる（補間の演習用）
    miss = rng.random(len(df)) < 0.01
    df.loc[miss, "sales"] = np.nan
    return df.sort_values(["store_id", "date"]).reset_index(drop=True)


def make_sensor() -> pd.DataFrame:
    """7日間・1分間隔の振動／温度。5日目の12:00〜14:00に異常区間を入れる。"""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-06-03 00:00", periods=7 * 24 * 60, freq="min")
    minutes = np.arange(len(idx))
    daily = np.sin(2 * np.pi * (minutes % 1440) / 1440)
    # 設備の稼働サイクル（10分周期）＋日周変動＋ノイズ
    cycle = np.sin(2 * np.pi * minutes / 10)
    vibration = 0.40 + 0.12 * daily + 0.05 * cycle + rng.normal(0, 0.02, len(idx))
    temperature = 62.0 + 4.0 * daily + 0.8 * cycle + rng.normal(0, 0.25, len(idx))

    abnormal = (idx >= pd.Timestamp("2024-06-07 12:00")) & (idx < pd.Timestamp("2024-06-07 14:00"))
    n_ab = int(abnormal.sum())
    vibration[abnormal] += 0.45 + 0.25 * np.sin(np.linspace(0, 8 * np.pi, n_ab))
    temperature[abnormal] += 6.0 + 2.0 * np.sin(np.linspace(0, 6 * np.pi, n_ab))
    # センサーの一時欠損（10点）
    miss_at = rng.choice(np.where(~abnormal)[0], 10, replace=False)
    vibration[miss_at] = np.nan

    return pd.DataFrame(
        {"vibration": vibration, "temperature": temperature, "label": abnormal.astype(int)},
        index=idx,
    ).rename_axis("timestamp")


def make_web() -> pd.DataFrame:
    """2年分の日次PV。週次・年次季節性、キャンペーン、障害による落ち込み。"""
    rng = np.random.default_rng(11)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    t = np.arange(len(dates))
    dow = np.array([1.05, 1.10, 1.12, 1.08, 0.95, 0.80, 0.78])[dates.dayofweek]
    yearly = 1 + 0.12 * np.sin(2 * np.pi * (t - 80) / 365.25)
    trend = 1 + 0.00035 * t
    spike = np.zeros(len(dates))
    for day, rate in [("2023-03-15", 1.9), ("2023-11-24", 1.6), ("2024-03-20", 2.1), ("2024-10-01", 1.7)]:
        spike[dates == pd.Timestamp(day)] = rate
    drop = np.zeros(len(dates))
    drop[(dates >= "2024-05-10") & (dates <= "2024-05-11")] = -0.75  # 障害
    pv = 32_000 * dow * yearly * trend * (1 + spike + drop) * np.exp(rng.normal(0, 0.06, len(dates)))
    return pd.DataFrame({"date": dates, "pv": pv.round().astype(int)})


def make_stock() -> pd.DataFrame:
    """500営業日分の終値。裾の厚いリターンを持つランダムウォーク。"""
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2023-01-04", periods=500)
    ret = 0.0004 + 0.011 * rng.standard_t(df=4, size=len(dates)) / np.sqrt(4 / 2)
    close = 2_400 * np.exp(np.cumsum(ret))
    return pd.DataFrame({"date": dates, "close": close.round(2)})


def main() -> None:
    retail = make_retail()
    retail.to_csv(DATA_DIR / "retail_sales.csv", index=False)
    sensor = make_sensor()
    sensor.to_csv(DATA_DIR / "sensor.csv")
    web = make_web()
    web.to_csv(DATA_DIR / "web_traffic.csv", index=False)
    stock = make_stock()
    stock.to_csv(DATA_DIR / "stock.csv", index=False)

    print("retail_sales.csv", retail.shape, "欠損", int(retail["sales"].isna().sum()))
    print("sensor.csv", sensor.shape, "異常点", int(sensor["label"].sum()))
    print("web_traffic.csv", web.shape)
    print("stock.csv", stock.shape)
    print("airpassengers.csv", pd.read_csv(DATA_DIR / "airpassengers.csv").shape)


if __name__ == "__main__":
    main()
