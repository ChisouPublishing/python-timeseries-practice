"""第8章：最良CV MAEと最終テストMAEの差を、期間ごとの条件に分けて確かめる。

実行: python3 08_ml/03_gap_check.py（先に 08_ml/02_optuna_search.py を実行）
保存済みの分割・予測台帳を読むだけで、モデルの学習や探索はやり直さない。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import OUT_DIR, load_retail, mae

raw = load_retail(store_id=1).set_index("date").asfreq("D")
study = json.loads((OUT_DIR / "08_ml_optuna_study.json").read_text())
ledger = pd.read_csv(OUT_DIR / "08_ml_optuna_predictions.csv", parse_dates=["target_date"]).set_index("target_date")
best = min((t for t in study["trials"] if t["value"] is not None), key=lambda t: t["value"])

# 同じ日の前週同曜日（lag7）を、どの期間でも同じ難しさの物差しとして使う
naive = raw["sales"].ffill().shift(7)
periods = [(f"CV{i + 1}", f["valid_start"], f["valid_end"], best["fold_mae"][i]) for i, f in enumerate(study["folds"])]
for month in range(8, 13):
    start = pd.Timestamp(2024, month, 1)
    end = start + pd.offsets.MonthEnd(0)
    part = ledger.loc[start:end].dropna(subset=["actual"])
    periods.append((f"テスト{month}月", str(start.date()), str(end.date()), mae(part["actual"], part["探索後モデル"])))

for name, start, end, model_mae in periods:
    sales = raw.loc[start:end, "sales"].dropna()
    naive_mae = mae(sales, naive.loc[sales.index])
    print(f"{name}: 平均売上={sales.mean():.0f} 販促日={raw.loc[start:end, 'promo'].sum():.0f}日 "
          f"前週同曜日MAE={naive_mae:.1f} 選択モデルMAE={model_mae:.1f} 比={model_mae / naive_mae:.2f}")
