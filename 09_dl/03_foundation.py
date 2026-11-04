"""第9章：時系列基盤モデル（Chronos-Bolt・TimesFM 2.5）を学習せずに使い、01・02と同じ128窓の7日先で比べる。

実行: python3 09_dl/03_foundation.py（初回はHugging Faceから重み約1.1GBを取得する）
"""

import logging
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import timesfm
import torch
from chronos import BaseChronosPipeline

from common import load_retail, mae

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
torch.set_num_threads(4)

# 重みは版を固定する（ライセンスはどちらもApache-2.0。2026-11-04確認）
CHRONOS = ("amazon/chronos-bolt-small", "772f3d25d38aec6d914c8949dab4462e2d46f5d8")
TIMESFM = ("google/timesfm-2.5-200m-pytorch", "1d952420fba87f3c6dee4f240de0f1a0fbc790e3")

ts = load_retail(store_id=1).set_index("date").sort_index().asfreq("D")
raw = ts["sales"].to_numpy(dtype="float64")
y = ts["sales"].ffill().to_numpy(dtype="float64")  # 入力の補完は過去の値だけ（01・02と同じ）
CUTOFF = int(len(y) * 0.8) - 1
HORIZON = 7
firsts = [f for f in range(CUTOFF + 1, len(y) - HORIZON + 1) if np.isfinite(raw[f:f + HORIZON]).all()]
actual = np.array([raw[f:f + HORIZON] for f in firsts])
naive = np.array([y[f - 7:f - 7 + HORIZON] for f in firsts])
print(f"テスト{len(firsts)}窓 / 最初の予測起点 {ts.index[firsts[0] - 1].date()} / 学習はしない（ゼロショット）")

chronos = BaseChronosPipeline.from_pretrained(CHRONOS[0], revision=CHRONOS[1], device_map="cpu", torch_dtype=torch.float32)
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM[0], revision=TIMESFM[1], torch_compile=False)
tfm.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=HORIZON, normalize_inputs=True,
                                   use_continuous_quantile_head=True, fix_quantile_crossing=True))


def run_chronos(contexts):
    q, _ = chronos.predict_quantiles([torch.tensor(c) for c in contexts], prediction_length=HORIZON,
                                     quantile_levels=[0.1, 0.5, 0.9])
    q = q.numpy()
    return q[..., 1], q[..., 0], q[..., 2]


def run_timesfm(contexts):
    point, quant = tfm.forecast(horizon=HORIZON, inputs=[c.astype(np.float32) for c in contexts])
    return quant[..., 5], quant[..., 1], quant[..., 9]  # 列0は平均、1〜9が10%点〜90%点


for context_name, start in [("起点までの全履歴", lambda f: 0), ("直近28日だけ", lambda f: f - 28)]:
    contexts = [y[start(f):f] for f in firsts]
    for model_name, run in [("Chronos-Bolt small", run_chronos), ("TimesFM 2.5", run_timesfm)]:
        started = time.perf_counter()
        med, lo, hi = run(contexts)
        seconds = time.perf_counter() - started
        by_h = np.abs(med - actual).mean(axis=0)
        cover = np.mean((actual >= lo) & (actual <= hi))
        print(f"{model_name}・{context_name}: 7期先MAE（全ホライズン平均）={by_h.mean():.1f} / ホライズン別={np.round(by_h).astype(int).tolist()}"
              f" / 80%区間の被覆率={cover:.1%} / 予測時間={'1分未満' if seconds < 60 else f'{seconds / 60:.0f}分'}")

print(f"同一窓の季節ナイーブ MAE: {mae(actual, naive):.1f}")
