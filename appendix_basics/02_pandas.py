"""pandas：CSV、日時、欠損、ラグ、過去だけの移動平均。"""
from io import StringIO
import pandas as pd

csv = StringIO(
    "date,units\n2024-01-01,10\n2024-01-02,\n"
    "2024-01-04,16\n2024-01-05,20\n2024-01-06,18\n"
)
raw = pd.read_csv(csv)
raw["date"] = pd.to_datetime(raw["date"], format="%Y-%m-%d")
daily = raw.set_index("date").sort_index().asfreq("D")
daily["missing"] = daily["units"].isna()
daily["filled"] = daily["units"].ffill()
daily["lag1"] = daily["filled"].shift(1)
daily["mean2_past"] = daily["filled"].shift(1).rolling(2).mean()
print(daily.to_string())
print("日付抽出:", daily.loc["2024-01-04":"2024-01-05", "units"].tolist())
print("観測行数:", daily["units"].resample("3D").count().tolist())
print("3日集計（観測値だけ）:", daily["units"].resample("3D").sum(min_count=1).tolist())
assert daily.loc["2024-01-04", "mean2_past"] == 10
assert daily["missing"].sum() == 2
