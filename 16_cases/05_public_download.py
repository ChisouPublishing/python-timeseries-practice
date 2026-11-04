"""公式固定版をダウンロードし、SHA-256を照合する（約115MB）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retail_public import download, REVISION

if __name__ == "__main__":
    download()
    print("FreshRetailNet-50K 固定版:", REVISION)
    print("train/evalのSHA-256照合: OK")
