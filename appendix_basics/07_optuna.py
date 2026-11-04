"""Optuna：開発データ内で探索し、固定後に最終テストを一度採点。"""
import numpy as np
import optuna
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from _data import features, ROOT

X, y = features()
Xdev, Xtest = X.iloc[:-28], X.iloc[-28:]
ydev, ytest = y.iloc[:-28], y.iloc[-28:]
cv = TimeSeriesSplit(n_splits=3, test_size=14)

def objective(trial):
    alpha = trial.suggest_categorical("alpha", [0.1, 1.0, 10.0, 100.0])
    scores = []
    for tr, va in cv.split(Xdev):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(Xdev.iloc[tr], ydev.iloc[tr])
        scores.append(mean_absolute_error(ydev.iloc[va], model.predict(Xdev.iloc[va])))
    return float(np.mean(scores))

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(
    {"alpha": [0.1, 1.0, 10.0, 100.0]}, seed=42))
study.optimize(objective, n_trials=4)
out = ROOT / "outputs"
out.mkdir(exist_ok=True)
study.trials_dataframe().to_csv(out / "appendix_basics_optuna_trials.csv", index=False)
print("完了した試行数:", len(study.trials))
print("選択設定:", study.best_params)
print(f"最良CV平均MAE={study.best_value:.3f}")
model = make_pipeline(StandardScaler(), Ridge(**study.best_params)).fit(Xdev, ydev)
print(f"固定後の最終28日 MAE={mean_absolute_error(ytest, model.predict(Xtest)):.3f}")
print("履歴: outputs/appendix_basics_optuna_trials.csv")
