"""第10章：店舗→大分類→商品の階層で予測し、hierarchicalforecast で合計をそろえる（BottomUp・TopDown・MinT）。

実行: python3 10_libs/03_hierarchy.py（先に python3 16_cases/05_public_download.py でデータを取得）
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace, TopDown
from hierarchicalforecast.utils import aggregate
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from retail_public import DATA, load_split

warnings.filterwarnings("ignore")
# 店舗0の全商品。商品の選び方に売上や評価期間の成績を使わない
products = sorted(pd.read_parquet(DATA / "train.parquet", columns=["product_id"],
                                  filters=[("store_id", "==", 0)]).product_id.unique().tolist())
frames = []
for split in ("train", "eval"):
    df, _ = load_split(split, products)
    frames.append(df.assign(split=split))
raw = pd.concat(frames)
raw = raw.assign(store="店舗0", cat="大分類" + raw.first_category_id.astype(str),
                 item="商品" + raw.product_id.astype(str), y=raw.sale_amount, ds=raw.dt)

spec = [["store"], ["store", "cat"], ["store", "cat", "item"]]
Y, S, tags = aggregate(raw[["store", "cat", "item", "ds", "y"]], spec)
cut = raw.loc[raw.split == "train", "ds"].max()
train, test = Y[Y.ds <= cut], Y[Y.ds > cut]
print(f"系列数: 店舗1 / 大分類{len(tags['store/cat'])} / 商品{len(tags['store/cat/item'])}"
      f" / 学習{train.ds.nunique()}日（〜{cut.date()}）/ 評価{test.ds.nunique()}日")

# すべての系列を別々に予測する（ベース予測）。MinTは学習期間の当てはめ残差も使う
sf = StatsForecast(models=[AutoETS(season_length=7)], freq="D", n_jobs=1)
base = sf.forecast(df=train, h=test.ds.nunique(), fitted=True)
fitted = sf.forecast_fitted_values()

top = base[base.unique_id == "店舗0"].AutoETS.to_numpy()
items = base[base.unique_id.isin(tags["store/cat/item"])].groupby("ds").AutoETS.sum().to_numpy()
print(f"ベース予測の不整合: 店舗合計の予測 − 商品の予測の和 = {np.round(top - items, 2).tolist()}")

hrec = HierarchicalReconciliation(reconcilers=[BottomUp(), TopDown(method="forecast_proportions"),
                                               MinTrace(method="ols"), MinTrace(method="wls_struct"),
                                               MinTrace(method="mint_shrink")])
rec = hrec.reconcile(Y_hat_df=base, Y_df=fitted, S_df=S, tags=tags)
merged = rec.merge(test, on=["unique_id", "ds"])

LABELS = {"AutoETS": "ベース予測（そろえない）", "AutoETS/BottomUp": "BottomUp",
          "AutoETS/TopDown_method-forecast_proportions": "TopDown（予測の比率）",
          "AutoETS/MinTrace_method-ols": "MinT（OLS）", "AutoETS/MinTrace_method-wls_struct": "MinT（WLS・構造）",
          "AutoETS/MinTrace_method-mint_shrink": "MinT（shrink）"}
levels = {"店舗": "store", "大分類": "store/cat", "商品": "store/cat/item"}
print("評価期間の実績の平均: " + " / ".join(f"{k}={test[test.unique_id.isin(tags[v])].y.mean():.2f}" for k, v in levels.items()))
for c, label in LABELS.items():
    maes = [np.mean(np.abs(m[c] - m.y)) for m in (merged[merged.unique_id.isin(tags[v])] for v in levels.values())]
    print(f"{label}: " + " / ".join(f"{k}MAE={v:.3f}" for k, v in zip(levels, maes)))

mint = rec[rec.unique_id == "店舗0"]["AutoETS/MinTrace_method-mint_shrink"].to_numpy()
mint_items = rec[rec.unique_id.isin(tags["store/cat/item"])].groupby("ds")["AutoETS/MinTrace_method-mint_shrink"].sum().to_numpy()
print(f"MinT（shrink）後の不整合の最大: {np.max(np.abs(mint - mint_items)):.2e}")
