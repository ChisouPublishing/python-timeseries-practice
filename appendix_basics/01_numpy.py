"""NumPy：配列の形、軸、条件抽出、乱数。"""
import numpy as np

sales = np.array([[10, 12, 14], [20, 18, 22]], dtype=float)
print("形（店舗, 日）:", sales.shape)
print("日別の店舗平均:", sales.mean(axis=0).tolist())
print("店舗別の期間平均:", sales.mean(axis=1).tolist())
print("15以上だけ抽出:", sales[sales >= 15].tolist())
rates = np.array([1.1, 0.9]).reshape(2, 1)
print("店舗別係数を掛ける:", np.round(sales * rates, 1).tolist())
rng = np.random.default_rng(42)
print("固定シードの整数:", rng.integers(0, 10, size=4).tolist())
assert sales.shape == (2, 3)
assert np.allclose(sales.mean(axis=0), [15, 15, 18])
