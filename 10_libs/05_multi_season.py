"""第10章：時間単位の店舗販売（1日と1週間の2つの周期）を、MSTL・TBATS・Prophetで予測し、単一周期のモデルと比べる。

実行: python3 10_libs/05_multi_season.py（先に python3 16_cases/05_public_download.py でデータを取得）
"""

import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoETS, AutoTBATS, SeasonalNaive

from common import C_BLUE, C_GRAY, C_ORANGE, mae, savefig
from retail_public import DATA, load_split

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").disabled = True
products = sorted(pd.read_parquet(DATA / "train.parquet", columns=["product_id"],
                                  filters=[("store_id", "==", 0)]).product_id.unique().tolist())


def store_hourly(split):
    """店舗0の全商品の販売数量を時間ごとに合計する（欠品で売れなかった分は含まれない）。"""
    df, _ = load_split(split, products)
    sales = pd.DataFrame(np.stack(df.hours_sale), index=df.dt).groupby(level=0).sum()
    ds = (sales.index.to_numpy()[:, None] + pd.to_timedelta(np.arange(24), unit="h").to_numpy()).ravel()
    return pd.DataFrame({"unique_id": "店舗0", "ds": ds, "y": sales.to_numpy().ravel()})


train, test = store_hourly("train"), store_hourly("eval")
H = len(test)
print(f"学習{len(train)}時間（{train.ds.min().date()}〜{train.ds.max().date()}）/ 評価{H}時間")

# 周期の強さ: 24時間前・168時間前との相関
y = train.y.to_numpy()
print("自己相関: " + " / ".join(f"ラグ{k}={np.corrcoef(y[k:], y[:-k])[0, 1]:.3f}" for k in (24, 168)))

models = [SeasonalNaive(season_length=24, alias="季節ナイーブ（24時間）"),
          SeasonalNaive(season_length=168, alias="季節ナイーブ（168時間）"),
          AutoETS(season_length=24, alias="ETS（24時間の周期だけ）"),
          MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"), alias="MSTL（24・168時間）"),
          AutoTBATS(season_length=[24, 168], alias="TBATS（24・168時間）")]
fc = StatsForecast(models=models, freq="h", n_jobs=1).forecast(df=train, h=H).merge(test, on=["unique_id", "ds"])

m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
m.fit(train[["ds", "y"]])
fc["Prophet（日・週）"] = m.predict(test[["ds"]]).yhat.to_numpy()

for col in [c for c in fc.columns if c not in ("unique_id", "ds", "y")]:
    print(f"{col}: MAE={mae(fc.y, fc[col]):.2f}")

# MSTLの分解: 24時間と168時間の季節成分の振れ幅
dec = MSTL(season_length=[24, 168]).fit(y).model_
print("MSTLの成分の標準偏差: " + " / ".join(f"{c}={dec[c].std():.1f}" for c in ("trend", "seasonal24", "seasonal168", "remainder")))

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(fc.ds, fc.y, color=C_GRAY, lw=1, label="実績")
ax.plot(fc.ds, fc["季節ナイーブ（24時間）"], color=C_ORANGE, lw=1, ls="--", label="季節ナイーブ（24時間）")
ax.plot(fc.ds, fc["MSTL（24・168時間）"], color=C_BLUE, lw=1.2, label="MSTL（24・168時間）")
ax.set_ylabel("販売数量（正規化値の合計）")
ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
ax.set_title("店舗0の時間別販売と予測（評価期間7日）", fontsize=10)
print("図:", savefig(fig, "ch10_multi_season"))
