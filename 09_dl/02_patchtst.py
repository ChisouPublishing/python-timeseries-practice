"""第9章：neuralforecastのPatchTSTを、01_lstm.py の7期先と同じ予測起点・同じ対象日で比べる。

実行: python3 09_dl/02_patchtst.py
"""

import logging
import datetime
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import neuralforecast
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

from common import load_retail, mae

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

ts = load_retail(store_id=1).set_index("date").sort_index().asfreq("D")
raw = ts["sales"].to_numpy(dtype="float64")
y = ts["sales"].ffill().to_numpy(dtype="float64")  # 入力の補完は過去の値だけ
CUTOFF = int(len(y) * 0.8) - 1  # 01_lstm.py と同じ。学習する正解はこの日まで
LOOKBACK, HORIZON = 28, 7

# 01_lstm.py の7期先と同じテスト窓: 最初の対象日が CUTOFF より後で、7日分の正解がそろう窓
firsts = [f for f in range(CUTOFF + 1, len(y) - HORIZON + 1) if np.isfinite(raw[f:f + HORIZON]).all()]
actual = np.array([raw[f:f + HORIZON] for f in firsts])
naive = np.array([y[f - 7:f - 7 + HORIZON] for f in firsts])
print(f"テスト{len(firsts)}窓 / 最初の予測起点 {ts.index[firsts[0] - 1].date()} / 学習する正解の最終日 {ts.index[CUTOFF].date()}")

frame = pd.DataFrame({"unique_id": "store1", "ds": ts.index, "y": y,
                      "available_mask": np.isfinite(raw).astype(int)})  # 欠損日は学習の正解から外す
train = frame.iloc[:CUTOFF + 1]

maes = []
timings = []
for seed in (42, 43, 44):
    model = PatchTST(h=HORIZON, input_size=LOOKBACK, patch_len=7, stride=7, hidden_size=32, n_heads=4,
                     encoder_layers=2, scaler_type="standard", max_steps=300, learning_rate=1e-3,
                     random_seed=seed, accelerator="cpu", enable_progress_bar=False, logger=False,
                     enable_model_summary=False)
    nf = NeuralForecast(models=[model], freq="D")
    started = time.perf_counter()
    nf.fit(df=train)
    fit_seconds = time.perf_counter() - started
    assert all(p.device.type == "cpu" for p in nf.models[0].parameters())
    timings.append({"seed": seed, "fit_seconds": fit_seconds})
    # 各予測起点までの履歴だけを渡して7日先を出す。学習はやり直さない
    pred = np.array([nf.predict(df=frame.iloc[:f])["PatchTST"].to_numpy() for f in firsts])
    by_h = np.abs(pred - actual).mean(axis=0)
    maes.append(by_h.mean())
    print(f"seed={seed}: 学習時間=約{fit_seconds:.0f}秒（CPU） / 7期先MAE（全ホライズン平均）={by_h.mean():.1f} "
          f"/ ホライズン別={np.round(by_h).astype(int).tolist()}")

print(f"PatchTST 3シードの平均={np.mean(maes):.1f} / 最小={np.min(maes):.1f} / 最大={np.max(maes):.1f}")
print(f"同一窓の季節ナイーブ MAE: {mae(actual, naive):.1f}")

metadata = {
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "python": platform.python_version(), "platform": platform.platform(),
    "machine": platform.machine(), "accelerator": "cpu", "gpu_used": False,
    "torch": torch.__version__, "neuralforecast": neuralforecast.__version__,
    "torch_cpu_threads": torch.get_num_threads(),
    "timing_scope": "nf.fit(df=train) only; excludes construction and prediction",
    "timings": timings,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "retail_data_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "data/retail_sales.csv").read_bytes()).hexdigest(),
}
metadata_file = Path(__file__).resolve().parents[1] / "outputs/09_dl_02_patchtst_metadata.json"
metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
