"""scikit-learn：時系列分割、Pipeline、学習・予測・評価。"""
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from _data import features

X, y = features()
Xdev, Xtest = X.iloc[:-28], X.iloc[-28:]
ydev, ytest = y.iloc[:-28], y.iloc[-28:]
pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
cv = TimeSeriesSplit(n_splits=3, test_size=14)
scores = []
for i, (tr, va) in enumerate(cv.split(Xdev), 1):
    model = clone(pipeline).fit(Xdev.iloc[tr], ydev.iloc[tr])
    pred = model.predict(Xdev.iloc[va])
    score = mean_absolute_error(ydev.iloc[va], pred)
    scores.append(score)
    print(f"fold{i}: 学習{len(tr)}行・検証{len(va)}行 / MAE={score:.3f}")
    assert Xdev.index[tr].max() < Xdev.index[va].min()
    assert np.allclose(model[0].mean_, Xdev.iloc[tr].mean())
model = pipeline.fit(Xdev, ydev)
pred = model.predict(Xtest)
print(f"CV平均MAE={np.mean(scores):.3f}")
print(f"最終28日 MAE={mean_absolute_error(ytest, pred):.3f}")
print(f"最終28日 RMSE={root_mean_squared_error(ytest, pred):.3f}")
print(f"前週同曜日 MAE={mean_absolute_error(ytest, Xtest['lag7']):.3f}")
