"""第12章：UCR Anomaly Archive の10系列で、単純な方法と行列プロファイル（ディスコード）を比べる。

実行: python3 12_anomaly/04_ucr_anomaly.py（初回は data/ucr_anomaly/ へ約184MBを取得する）
"""

import hashlib
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import stumpy

from common import DATA_DIR

URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip"
SHA256 = "ac4b991c701e620ae9cc5ebd57ae45593a36cc9c0b6ed5e3c4b7e466cf4783d4"  # 2026-11-04 取得
ROOT = DATA_DIR / "ucr_anomaly"
ZIP = ROOT / "archive.zip"
M = 100  # 部分列の長さ。全系列で同じ値を事前に決めた

ROOT.mkdir(parents=True, exist_ok=True)
if not ZIP.exists():
    urllib.request.urlretrieve(URL, ZIP)
if hashlib.sha256(ZIP.read_bytes()).hexdigest() != SHA256:
    raise ValueError("アーカイブのハッシュが一致しない")
with zipfile.ZipFile(ZIP) as z:
    names = sorted(n for n in z.namelist() if "/UCR_Anomaly_FullData/" in n and n.endswith(".txt"))
    print(f"アーカイブの系列数: {len(names)}")
    # 選び方は結果を見る前に決めた: 番号順に25件おき（001, 026, …, 226）
    picked = [(Path(n).name, np.loadtxt(z.open(n)).ravel()) for n in names[::25]]


def locate(x, train_end):
    """3つの方法で、テスト部分の中で最も異常な位置を1つずつ返す。"""
    test = x[train_end:]
    diff = np.abs(np.diff(x))[train_end - 1:]  # 1時点前との差の絶対値
    s = pd.Series(x)
    z = ((s - s.shift(1).rolling(M).mean()) / s.shift(1).rolling(M).std()).abs().to_numpy()[train_end:]
    # 行列プロファイル: テストの各部分列と、学習部分の最も近い部分列との距離（学習部分に似たものが無いほど大きい）
    mp = stumpy.stump(test, M, x[:train_end], ignore_trivial=False)[:, 0].astype(float)
    return {"差分の最大": int(np.nanargmax(diff)), "移動Zの最大": int(np.nanargmax(z)),
            "ディスコード": int(np.nanargmax(mp)) + M // 2}


rows = []
for name, x in picked:
    num, train_end, begin, end = (int(v) for v in re.match(r"(\d+)_UCR_Anomaly_.*_(\d+)_(\d+)_(\d+)\.txt", name).groups())
    row = {"系列": f"{num:03d}", "長さ": len(x), "異常の長さ": end - begin + 1}
    for method, pos in locate(x, train_end).items():
        loc = train_end + pos
        row[method] = "○" if begin - 100 <= loc <= end + 100 else "×"  # 異常区間の前後100点以内なら正解
    rows.append(row)

table = pd.DataFrame(rows).set_index("系列")
print(table.to_string())
print("正解数（10系列）: " + " / ".join(f"{m}={int((table[m] == '○').sum())}" for m in ("差分の最大", "移動Zの最大", "ディスコード")))
