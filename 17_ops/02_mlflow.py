"""第17章：パイプラインを joblib で保存し、MLflow に実行条件・評価値・モデルの版を記録する。

実行: python3 17_ops/02_mlflow.py
記録先は outputs/mlflow/（実行のたびに作り直す）。サーバーは立てず、ローカルのSQLiteファイルを使う。
"""

import hashlib
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT_DIR, data_path, load_web_traffic, mae

warnings.filterwarnings("ignore")

# 01_monitor.py のパイプラインと同じデータ・分割・設定
pv = load_web_traffic()
d = pd.DataFrame({"y": pv})
for lag in (1, 7, 14):
    d[f"lag{lag}"] = d["y"].shift(lag)
d["ma7"] = d["y"].shift(1).rolling(7).mean()
d = d.dropna()
X, y = d.drop(columns="y"), d["y"]
split = int(len(d) * 0.8)
params = {"n_estimators": 300, "learning_rate": 0.05, "random_state": 42}
pipe = Pipeline([("scaler", StandardScaler()), ("model", lgb.LGBMRegressor(verbose=-1, **params))])
pipe.fit(X.iloc[:split], y.iloc[:split])
pred = pipe.predict(X.iloc[split:])

# (1) joblib: 前処理込みのパイプラインを1ファイルに保存し、読み戻して同じ予測になるか確かめる
model_file = OUT_DIR / "17_ops_pipeline.joblib"
joblib.dump(pipe, model_file)
same_joblib = np.array_equal(joblib.load(model_file).predict(X.iloc[split:]), pred)
print("joblibで読み戻した予測が一致:", same_joblib)

# (2) MLflow: 実行ごとに、データの版・設定・評価値・モデルを1か所に残す
store = OUT_DIR / "mlflow"
shutil.rmtree(store, ignore_errors=True)
store.mkdir()
mlflow.set_tracking_uri(f"sqlite:///{store / 'mlflow.db'}")
mlflow.set_experiment("web_pv_forecast")
with mlflow.start_run(run_name="lightgbm_pipeline"):
    mlflow.log_params(params)
    mlflow.log_param("train_end", str(X.index[split - 1].date()))
    mlflow.log_param("data_sha256", hashlib.sha256(data_path("web_traffic.csv").read_bytes()).hexdigest()[:12])
    mlflow.log_metric("test_mae", mae(y.iloc[split:], pred))
    # MLflow 3はskops形式で保存し、読み込みを許す型の明示を求める。ここでは自分で学習したLightGBMの型だけを許す
    trusted = ["collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMRegressor"]
    mlflow.sklearn.log_model(pipe, name="model", registered_model_name="web_pv_forecast",
                             input_example=X.iloc[:3], skops_trusted_types=trusted)

# (3) 記録を読み返す: 実行条件と評価値、登録したモデルの版
found = mlflow.search_runs(experiment_names=["web_pv_forecast"])
row = found.iloc[0]
print("記録した実行数:", len(found))
print("記録したパラメータ:", sorted(k.removeprefix("params.") for k in found.columns if k.startswith("params.")))
print("記録した評価値 test_mae:", round(row["metrics.test_mae"], 0))
version = mlflow.MlflowClient().get_latest_versions("web_pv_forecast")[0].version
print("登録したモデルの版:", version)
loaded = mlflow.sklearn.load_model(f"models:/web_pv_forecast/{version}")
print("登録モデルを読み戻した予測が一致:", np.array_equal(loaded.predict(X.iloc[split:]), pred))
