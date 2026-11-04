"""同梱CSVの粒度・キー・期間・欠損を検査する。実行: python3 01_overview/02_data_contract.py"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from common import DATA_DIR, OUT_DIR

SPECS = {
    "airpassengers.csv": ("month", ["month"], "MS"),
    "retail_sales.csv": ("date", ["store_id", "date"], "D"),
    "sensor.csv": ("timestamp", ["timestamp"], "min"),
    "web_traffic.csv": ("date", ["date"], "D"),
    "stock.csv": ("date", ["date"], "B"),
}
report = {}
for name, (time_col, key, freq) in SPECS.items():
    df = pd.read_csv(DATA_DIR / name, parse_dates=[time_col])
    assert df[key].notna().all().all(), f"{name}: キーの欠損"
    assert not df.duplicated(key).any(), f"{name}: キーの重複"
    groups = df.groupby("store_id") if "store_id" in df else [(None, df)]
    for _, group in groups:
        actual = pd.DatetimeIndex(group[time_col].sort_values())
        expected = pd.date_range(actual.min(), actual.max(), freq=freq)
        assert actual.equals(expected), f"{name}: 時間軸に行の欠落"
    if "promo" in df: assert df["promo"].isin([0, 1]).all()
    if "label" in df: assert df["label"].isin([0, 1]).all()
    stats = {"rows":len(df), "key":key, "start":str(df[time_col].min()),
             "end":str(df[time_col].max()), "frequency":freq,
             "nulls":{c:int(n) for c,n in df.isna().sum().items()},
             "dtypes":{c:str(t) for c,t in df.dtypes.items()}}
    if "store_id" in df:
        stats["stores"] = {str(k):{"rows":len(g),"missing_sales":int(g.sales.isna().sum())} for k,g in df.groupby("store_id")}
    if "label" in df: stats["anomaly_points"]=int(df.label.sum())
    report[name] = stats
    print(name, f"行数={len(df)}", f"期間={df[time_col].min()}〜{df[time_col].max()}")
    print("  キー:", ", ".join(key), "/ 欠損:", stats["nulls"])
print("検査: キー欠損0・重複0・期待頻度の行欠落0")
(OUT_DIR / "data_contract.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
