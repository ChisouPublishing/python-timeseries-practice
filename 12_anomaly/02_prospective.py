"""未来の異常検知：学習・正常較正・テストを時刻で分離する。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from common import load_sensor, OUT_DIR

raw = load_sensor()
s = raw[["vibration","temperature"]].ffill()
# 現在のセンサー値が到着した直後に判定する。未来の観測は不要。
feat = s.copy()
feat["ma30"] = s.vibration.shift(1).rolling(30,min_periods=30).mean()
feat["sd30"] = s.vibration.shift(1).rolling(30,min_periods=30).std()
feat = feat.dropna()
train = feat.index < "2024-06-06"
cal = (feat.index >= "2024-06-06") & (feat.index < "2024-06-07")
test = feat.index >= "2024-06-07"
assert feat.index[train].max() < feat.index[cal].min() < feat.index[test].min()
assert raw.loc[feat.index[train|cal],"label"].eq(0).all()
scaler = StandardScaler().fit(feat.loc[train])
model = IsolationForest(n_estimators=200,contamination="auto",random_state=42,n_jobs=1)
model.fit(scaler.transform(feat.loc[train]))
cal_score = -model.score_samples(scaler.transform(feat.loc[cal]))
threshold = float(np.quantile(cal_score,0.999))
score = -model.score_samples(scaler.transform(feat.loc[test]))
flag = score > threshold
truth = raw.loc[feat.index[test],"label"]
p,r,f,_ = precision_recall_fscore_support(truth,flag,average='binary',zero_division=0)
fp=int(((truth==0)&flag).sum());tp=int(((truth==1)&flag).sum())
# 正常の暦日数ではなく、正常ラベルの観測時間へ換算した誤報件数。
normal_days=float((truth==0).sum()/1440)
onset=truth[truth==1].index.min()
detected=truth.index[(truth==1)&flag]
delay=None if not len(detected) else int((detected.min()-onset).total_seconds()/60)
print("学習:",int(train.sum()),"点 / 正常較正:",int(cal.sum()),"点 / テスト:",int(test.sum()),"点")
print("固定閾値（較正期間99.9%点）:",round(threshold,5))
print(f"未来テスト: Precision={p:.3f} Recall={r:.3f} F1={f:.3f} AP={average_precision_score(truth,score):.3f}")
print("検出点:",tp,"/ 異常点:",int(truth.sum()),"/ 誤報点:",fp)
print("正常24時間換算の誤報点数:",round(fp/normal_days,2))
print("既知の異常区間の検出遅延（分）:",delay)
ledger=pd.DataFrame({'timestamp':feat.index[test],'label':truth.to_numpy(),'score':score,'flag':flag})
ledger.to_csv(OUT_DIR/'12_anomaly_prospective_predictions.csv',index=False)
