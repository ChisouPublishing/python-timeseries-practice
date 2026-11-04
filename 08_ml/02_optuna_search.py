"""時系列CV内でRidge/LightGBMを探索し、未使用の最終期間で一度比較する。
実行: python3 08_ml/02_optuna_search.py
探索予算・期間・評価基準はコードで固定。最終テストの結果を見て再探索しない。
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import C_BLUE, C_GRAY, C_ORANGE, OUT_DIR, load_retail, mae, savefig

optuna.logging.set_verbosity(optuna.logging.WARNING)
raw = load_retail(store_id=1).set_index("date").asfreq("D")

def features(frame):
    history = frame["sales"].ffill()
    out = pd.DataFrame(index=frame.index)
    for k in (1, 7, 14, 28): out[f"lag{k}"] = history.shift(k)
    for k in (7, 28):
        out[f"ma{k}"] = history.shift(1).rolling(k).mean()
        out[f"sd{k}"] = history.shift(1).rolling(k).std()
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month
    out["promo"] = frame["promo"]  # 当日の販促計画が前日までに確定するという教材上の前提
    return out

X = features(raw).dropna()
y = raw["sales"].reindex(X.index)
cutoff = pd.Timestamp("2024-08-01")
dev, holdout = X.index < cutoff, X.index >= cutoff
Xd, yd, Xt, yt = X.loc[dev], y.loc[dev], X.loc[holdout], y.loc[holdout]
assert Xd.index.max() < Xt.index.min()
# 将来の実測値を改変しても、変更日以前の特徴量が変わらないことを検査。
perturbed = raw.copy()
perturbed.loc[cutoff:, "sales"] = 1e9
pd.testing.assert_frame_equal(features(raw).loc[:cutoff], features(perturbed).loc[:cutoff])
folds = list(TimeSeriesSplit(n_splits=3, test_size=60).split(Xd))
split_log = []
for tr, va in folds:
    assert Xd.index[tr].max() < Xd.index[va].min()
    split_log.append({"train_end":str(Xd.index[tr].max().date()),"valid_start":str(Xd.index[va].min().date()),"valid_end":str(Xd.index[va].max().date()),"valid_observed":int(yd.iloc[va].notna().sum())})

def build_model(params):
    p = dict(params)
    family = p.pop("family")
    if family == "ridge": return make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"]))
    return lgb.LGBMRegressor(**p, random_state=42, n_jobs=1, deterministic=True, force_col_wise=True, verbose=-1)

def objective(trial):
    family = trial.suggest_categorical("family", ["ridge", "lightgbm"])
    if family == "ridge":
        params = {"family":family,"alpha":trial.suggest_float("alpha", 0.1, 1000, log=True)}
    else:
        params = {"family":family,
                  "n_estimators":trial.suggest_int("n_estimators",100,600,step=100),
                  "learning_rate":trial.suggest_float("learning_rate",0.02,0.2,log=True),
                  "num_leaves":trial.suggest_int("num_leaves",8,64),
                  "min_child_samples":trial.suggest_int("min_child_samples",10,60),
                  "reg_lambda":trial.suggest_float("reg_lambda",1e-3,10,log=True)}
    total_error, total_n = 0.0, 0
    fold_scores = []
    for step,(tr,va) in enumerate(folds):
        observed_tr = tr[yd.iloc[tr].notna().to_numpy()]
        observed_va = va[yd.iloc[va].notna().to_numpy()]
        model = build_model(params)
        model.fit(Xd.iloc[observed_tr],yd.iloc[observed_tr])
        error = np.abs(yd.iloc[observed_va].to_numpy()-model.predict(Xd.iloc[observed_va]))
        total_error += error.sum(); total_n += len(error)
        fold_scores.append(float(error.mean()))
        trial.report(total_error/total_n,step)
        trial.set_user_attr("fold_mae",fold_scores)
        if trial.should_prune(): raise optuna.TrialPruned()
    return total_error/total_n

study = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=6),
                           pruner=optuna.pruners.MedianPruner(n_startup_trials=8,n_warmup_steps=1))
study.optimize(objective,n_trials=24,n_jobs=1)
print("Optuna:",optuna.__version__,"/ 探索予算: 24試行 / 時系列3分割（各60日）")
print("開発期間:",Xd.index.min().date(),"〜",Xd.index.max().date())
print("最終テスト:",Xt.index.min().date(),"〜",Xt.index.max().date(),"/ 正解観測日:",int(yt.notna().sum()))
print("試行状態:",pd.Series([t.state.name for t in study.trials]).value_counts().sort_index().to_dict())
print("最良CV MAE:",round(study.best_value,1))
print("選択パラメータ:",json.dumps(study.best_params,ensure_ascii=False,sort_keys=True))
# 探索完了後にだけ最終テストを採点。モデルの再学習は開発期間だけ。
models = {"既定Ridge":{"family":"ridge","alpha":1.0},
          "既定LightGBM":{"family":"lightgbm","n_estimators":300,"learning_rate":0.05,"num_leaves":15},
          "探索後モデル":study.best_params}
observed_d, observed_t = yd.notna(), yt.notna()
ledger = pd.DataFrame({"origin":Xt.index-pd.Timedelta(days=1),"target_date":Xt.index,"actual":yt,
                       "季節ナイーブ":Xt["lag7"]})
for name,params in models.items():
    model=build_model(params)
    model.fit(Xd.loc[observed_d],yd.loc[observed_d])
    ledger[name]=model.predict(Xt)
metrics = {name:mae(yt.loc[observed_t],ledger.loc[observed_t,name]) for name in ["季節ナイーブ",*models]}
for name,value in metrics.items(): print(f"{name}: 最終テストMAE={value:.1f}")
print("探索後モデルの改善率（対既定LightGBM）:",round((1-metrics["探索後モデル"]/metrics["既定LightGBM"])*100,1),"%")
ledger.to_csv(OUT_DIR/'08_ml_optuna_predictions.csv',index=False)
study.trials_dataframe().to_csv(OUT_DIR/'08_ml_optuna_trials.csv',index=False)
artifact={"budget":24,"seed":42,"test_start":str(cutoff.date()),"folds":split_log,"best_params":study.best_params,
          "best_cv_mae":study.best_value,"test_mae":metrics,
          "trials":[{"number":t.number,"state":t.state.name,"params":t.params,"value":t.value,"fold_mae":t.user_attrs.get("fold_mae",[])} for t in study.trials]}
(OUT_DIR/'08_ml_optuna_study.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(8,3.6))
ax.bar(list(metrics),list(metrics.values()),color=[C_GRAY,C_BLUE,C_BLUE,C_ORANGE])
ax.set_ylabel('最終テストMAE（円）');ax.set_title('探索に使わない同一の最終期間で比較')
for i,v in enumerate(metrics.values()):ax.text(i,v,f'{v:.0f}',ha='center',va='bottom')
print('図:',savefig(fig,'ch08_optuna'))
