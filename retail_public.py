"""FreshRetailNet-50Kの固定版を扱う。数量は正規化済み、1は欠品。"""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/freshretail"
REVISION = "c374e85975c82816c495acfa904cc55ab14a47d3"
BASE = "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K"
HASHES = {
    "train.parquet": "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d",
    "eval.parquet": "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e",
}


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        path = DATA / name
        if not path.exists():
            tmp = path.with_suffix(".download")
            urllib.request.urlretrieve(f"{BASE}/resolve/{REVISION}/data/{name}", tmp)
            if hashlib.sha256(tmp.read_bytes()).hexdigest() != expected:
                raise ValueError(f"取得版のハッシュ不一致: {name}")
            tmp.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"保存版のハッシュ不一致: {name}")


def load_split(split, products=None):
    import pandas as pd
    path = DATA / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError("先に python3 16_cases/05_public_download.py を実行してください")
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[path.name]:
        raise ValueError("固定版のハッシュ不一致")
    df = pd.read_parquet(path, filters=[("store_id", "==", 0)])
    if products is None:
        # 売上・欠品率・最終評価の成績では選ばない。
        products = sorted(df.product_id.unique().tolist())[:20]
    df = df[df.product_id.isin(products)].copy()
    df["dt"] = pd.to_datetime(df.dt)
    df = df.sort_values(["product_id", "dt"]).reset_index(drop=True)
    if df.duplicated(["store_id", "product_id", "dt"]).any():
        raise ValueError("店舗×商品×日の重複")
    return df, products
