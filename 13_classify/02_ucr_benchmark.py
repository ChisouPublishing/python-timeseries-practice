"""第13章：UCRアーカイブの10データセットで、1-NN（ユークリッド）・1-NN（DTW）・ROCKETを比べる。

実行: python3 13_classify/02_ucr_benchmark.py（初回は data/ucr/ へ取得する）
"""

import hashlib
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.datasets import load_UCR_UEA_dataset
from sktime.datasets.tsc_dataset_names import univariate_equal_length

from common import DATA_DIR

warnings.filterwarnings("ignore")
ROOT = DATA_DIR / "ucr"
ROOT.mkdir(parents=True, exist_ok=True)


def load(name, split):
    X, y = load_UCR_UEA_dataset(name, split=split, return_type="numpy3D", extract_path=str(ROOT))
    return X.astype(float), np.asarray(y)


# 選び方は結果を見る前に決めた: 名前の順に、学習200件以下・テスト500件以下・長さ500以下の先頭10件
chosen = []
for name in sorted(univariate_equal_length):
    Xtr, ytr = load(name, "train")
    Xte, yte = load(name, "test")
    if len(Xtr) <= 200 and len(Xte) <= 500 and Xtr.shape[2] <= 500:
        chosen.append((name, Xtr, ytr, Xte, yte))
    if len(chosen) == 10:
        break

models = {"1-NN ユークリッド": lambda: KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="euclidean"),
          "1-NN DTW（窓10%）": lambda: KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw",
                                                                     distance_params={"window": 0.1}),
          "ROCKET": lambda: RocketClassifier(num_kernels=2000, random_state=42)}
rows = []
for name, Xtr, ytr, Xte, yte in chosen:
    row = {"データ": name, "学習": len(Xtr), "テスト": len(Xte), "長さ": Xtr.shape[2], "クラス": len(set(ytr))}
    for label, make in models.items():
        row[label] = make().fit(Xtr, ytr).score(Xte, yte)
    rows.append(row)
table = pd.DataFrame(rows).set_index("データ")
pd.set_option("display.width", 200)
print(table.round(3).to_string())

acc = table[list(models)]
ranks = acc.rank(axis=1, ascending=False).mean()
print("平均正解率: " + " / ".join(f"{k}={v:.3f}" for k, v in acc.mean().items()))
print("平均順位（1が最良）: " + " / ".join(f"{k}={v:.2f}" for k, v in ranks.items()))
wins = (acc["ROCKET"] > acc["1-NN DTW（窓10%）"]).sum(), (acc["ROCKET"] < acc["1-NN DTW（窓10%）"]).sum()
print(f"ROCKET対DTW: 勝ち{wins[0]} / 負け{wins[1]} / 引き分け{len(acc) - sum(wins)}")
# テスト件数が少ないデータでは、正解率の標準誤差が大きい
se = np.sqrt(acc["ROCKET"] * (1 - acc["ROCKET"]) / table["テスト"])
print("ROCKETの正解率の標準誤差: 最大=" f"{se.max():.3f}（{se.idxmax()}）")

files = sorted(ROOT.glob("*/*.ts"))
digest = hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest() for p in files if p.parent.name in table.index))
print(f"使った.tsファイル: {sum(p.parent.name in table.index for p in files)}個 / まとめたSHA-256の先頭: {digest.hexdigest()[:16]}")
