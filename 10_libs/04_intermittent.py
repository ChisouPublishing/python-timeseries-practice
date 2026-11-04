"""第10章：時間単位の商品販売（ゼロが多い系列）で、間欠需要の手法と指標の選び方を確かめる。

実行: python3 10_libs/04_intermittent.py（先に python3 16_cases/05_public_download.py でデータを取得）
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ADIDA, IMAPA, TSB, CrostonClassic, CrostonSBA, SeasonalNaive

from retail_public import DATA, load_split

warnings.filterwarnings("ignore")
products = sorted(pd.read_parquet(DATA / "train.parquet", columns=["product_id"],
                                  filters=[("store_id", "==", 0)]).product_id.unique().tolist())


def hourly(split):
    """商品×時間の販売数量と欠品フラグを縦長にする（24時間）。"""
    df, _ = load_split(split, products)
    sales, stock = np.stack(df.hours_sale), np.stack(df.hours_stock_status)
    ds = (df.dt.to_numpy()[:, None] + pd.to_timedelta(np.arange(24), unit="h").to_numpy()).ravel()
    return pd.DataFrame({"unique_id": np.repeat(df.product_id.to_numpy(), 24), "ds": ds,
                         "y": sales.ravel(), "stockout": stock.ravel()})


train, test = hourly("train"), hourly("eval")
H = test.ds.nunique()


def classify(y):
    """Syntetos-Boylanの分類: ADI（需要が出る平均間隔）とCV²（ゼロ以外の大きさのばらつき）。"""
    nz = y[y > 0]
    adi, cv2 = len(y) / len(nz), (nz.std() / nz.mean()) ** 2
    return ("間欠" if cv2 < 0.49 else "塊状") if adi >= 1.32 else ("平滑" if cv2 < 0.49 else "不規則")


kinds = train.groupby("unique_id").y.apply(lambda s: classify(s.to_numpy()))
print(f"{len(products)}商品 / 学習{len(train) // len(products)}時間 / 評価{H}時間 / 時間単位でゼロの割合={np.mean(train.y == 0):.1%}")
print("系列の分類（学習期間）:", kinds.value_counts().to_dict())

models = [SeasonalNaive(season_length=24), CrostonClassic(), CrostonSBA(), TSB(alpha_d=0.1, alpha_p=0.1), ADIDA(), IMAPA()]
fc = StatsForecast(models=models, freq="h", n_jobs=1).forecast(df=train[["unique_id", "ds", "y"]], h=H)
fc["ゼロ予測"] = 0.0
scored = test.merge(fc, on=["unique_id", "ds"])
scored = scored[scored.stockout == 0]  # 欠品の時間は需要が観測できないので採点しない
print(f"採点する時間: {len(scored)}（欠品の時間を除く）")

for col in ["SeasonalNaive", "CrostonClassic", "CrostonSBA", "TSB", "ADIDA", "IMAPA", "ゼロ予測"]:
    err = scored[col] - scored.y
    total = scored.groupby("unique_id")[[col, "y"]].sum()  # 商品ごとの7日間の合計（補充量の判断）
    total_err = np.mean(np.abs(total[col] - total.y)) / total.y.mean()
    print(f"{col}: 時間MAE={np.mean(np.abs(err)):.3f} / 時間RMSE={np.sqrt(np.mean(err ** 2)):.3f}"
          f" / 7日合計の誤差率={total_err:.1%} / 偏り={err.sum() / scored.y.sum():+.1%}")
