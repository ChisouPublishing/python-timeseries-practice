"""書籍『Python実践 時系列データ分析』共通ヘルパー。

- データの読み込み（data/ 配下。無ければ合成データをその場で作る）
- 図の保存（figures/ に SVG。紙面へは assets/images/svg/ に複製）
- 乱数シードの固定

各章のスクリプトは `from common import ...` で読み込む。
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
OUT_DIR = BASE_DIR / "outputs"
for _d in (DATA_DIR, FIG_DIR, OUT_DIR):
    _d.mkdir(exist_ok=True)

# 日本語が豆腐にならないよう、環境にある日本語フォントを1つ選ぶ
import matplotlib.font_manager as _fm

_available = {f.name for f in _fm.fontManager.ttflist}
for _candidate in ("Hiragino Sans", "Noto Sans JP", "Yu Gothic", "Hiragino Kaku Gothic ProN"):
    if _candidate in _available:
        plt.rcParams["font.family"] = _candidate
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#D2D9E2"
plt.rcParams["grid.linewidth"] = 0.6
plt.rcParams["axes.edgecolor"] = "#9AA7B8"
plt.rcParams["text.color"] = "#131A24"
plt.rcParams["axes.labelcolor"] = "#131A24"
plt.rcParams["xtick.color"] = "#48566C"
plt.rcParams["ytick.color"] = "#48566C"

# 色覚配慮（赤緑に頼らず、青・橙・紫・灰で区別する）
C_BLUE = "#1F6FB2"
C_ORANGE = "#D9822B"
C_PURPLE = "#6B4C9A"
C_GRAY = "#6E7C90"
C_RED = "#9E2A3D"
C_TEAL = "#0E6E6E"


def seed_everything(seed: int = SEED) -> None:
    """Python・NumPy の乱数を固定する（scikit-learn 等は random_state を別途渡す）。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def savefig(fig, name: str) -> Path:
    """図を figures/<name>.svg に保存する。"""
    path = FIG_DIR / f"{name}.svg"
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(path, format="svg", facecolor="white")
    plt.close(fig)
    return path.relative_to(BASE_DIR)


def data_path(name: str) -> Path:
    return DATA_DIR / name


def load_airpassengers() -> pd.Series:
    """月次航空旅客数（1949-01〜1960-12、144か月）。"""
    df = pd.read_csv(data_path("airpassengers.csv"), parse_dates=["month"])
    return df.set_index("month")["passengers"].asfreq("MS")


def load_retail(store_id: int | None = None) -> pd.DataFrame:
    """店舗別の日次売上（long形式：date, store_id, sales, promo）。"""
    df = pd.read_csv(data_path("retail_sales.csv"), parse_dates=["date"])
    if store_id is not None:
        df = df[df["store_id"] == store_id].copy()
    return df.sort_values(["store_id", "date"]).reset_index(drop=True)


def load_sensor() -> pd.DataFrame:
    """1分間隔の設備センサー（timestamp, vibration, temperature, label）。"""
    return pd.read_csv(data_path("sensor.csv"), parse_dates=["timestamp"]).set_index("timestamp")


def load_web_traffic() -> pd.Series:
    df = pd.read_csv(data_path("web_traffic.csv"), parse_dates=["date"])
    return df.set_index("date")["pv"].asfreq("D")


def load_stock() -> pd.Series:
    df = pd.read_csv(data_path("stock.csv"), parse_dates=["date"])
    return df.set_index("date")["close"].asfreq("B")


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def pinball_loss(y_true, y_pred, tau: float) -> float:
    """τ分位点の予測に対する平均Pinball loss（分位点損失）。"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))
