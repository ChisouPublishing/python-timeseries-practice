"""LightGBM：説明変数を指定してfit/predictする。"""
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from _data import features

X, y = features()
Xdev, Xtest = X.iloc[:-28], X.iloc[-28:]
ydev, ytest = y.iloc[:-28], y.iloc[-28:]
model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05,
                          num_leaves=7, min_child_samples=10,
                          random_state=42, n_jobs=1, verbosity=-1)
model.fit(Xdev, ydev)
pred = model.predict(Xtest)
print("説明変数:", X.columns.tolist())
print(f"LightGBM MAE={mean_absolute_error(ytest, pred):.3f}")
print(f"前週同曜日 MAE={mean_absolute_error(ytest, Xtest['lag7']):.3f}")
print("分岐回数:", dict(zip(X.columns, model.feature_importances_.tolist())))
assert len(pred) == 28
