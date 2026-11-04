"""第2章：環境とライブラリのバージョンを確認し、再現性を確かめる。

実行: python3 02_env/01_versions.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import statsmodels
import torch

print("Python", sys.version.split()[0])
for name, mod in [
    ("pandas", pd),
    ("numpy", np),
    ("matplotlib", matplotlib),
    ("statsmodels", statsmodels),
    ("scikit-learn", sklearn),
    ("lightgbm", lightgbm),
    ("torch", torch),
]:
    print(f"{name:12s} {mod.__version__}")

# 乱数は「シード固定 → 同じ結果」を確認する
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
a = np.random.rand(3)
np.random.seed(SEED)
b = np.random.rand(3)
print("1回目:", a.round(8))
print("2回目:", b.round(8))
print("一致:", np.allclose(a, b))

# ライブラリをまたぐときは、それぞれの random_state も固定する
rng = np.random.default_rng(SEED)
print("default_rng(42):", rng.normal(size=3).round(4))
