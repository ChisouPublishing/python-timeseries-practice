"""第7章：03_state_space.py に近い構造時系列モデルを PyMC でベイズ推定し、パラメータの不確かさを含めた予測区間を作る。

実行: python3 07_ets/04_bayes.py
"""

import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pymc as pm

from common import load_airpassengers, mae

warnings.filterwarnings("ignore")
logging.getLogger("pymc").setLevel(logging.ERROR)
y = load_airpassengers()
train, test = y[:-24], y[-24:]  # 01_ets.py と同じ分割
obs = np.log(train.to_numpy())
n, H = len(obs), len(test)
month = np.arange(n + H) % 12

# 水準 = 前月の水準 + 傾き（一定）+ 誤差、観測 = 水準 + 月の季節（12か月で和が0）+ 誤差（対数）
with pm.Model() as model:
    # 誤差の大きさの合計と、そのうち水準の誤差が占める割合に分けて置く（2つの分散を直接置くより推定が安定する）
    sigma_total = pm.HalfNormal("sigma_total", 0.05)
    share = pm.Beta("share", 2.0, 2.0)
    sigma_level = pm.Deterministic("sigma_level", sigma_total * pm.math.sqrt(share))
    sigma_obs = pm.Deterministic("sigma_obs", sigma_total * pm.math.sqrt(1 - share))
    slope = pm.Normal("slope", 0.0, 0.05)
    season = pm.ZeroSumNormal("season", sigma=0.3, shape=12)
    level = pm.GaussianRandomWalk("level", mu=slope, sigma=sigma_level, init_dist=pm.Normal.dist(obs[0], 0.5), steps=n - 1)
    pm.Normal("y", level + season[month[:n]], sigma_obs, observed=obs)
    idata = pm.sample(2000, tune=2000, chains=4, cores=1, random_seed=42, progressbar=False, target_accept=0.99)

post = idata.posterior.stack(draw_all=("chain", "draw"))
# 収束の診断: R-hat（1に近いほどよい）、有効サンプル数、発散（サンプラーが曲率の急な所を探索できなかった回数）
rhat = float(pm.stats.rhat(idata).to_array().max())
ess = float(pm.stats.ess(idata).to_array().min())
divergent = int(idata.sample_stats["diverging"].sum())
print(f"事後分布のサンプル: {post.sizes['draw_all']}個（4チェーン×2000）/ R-hatの最大={rhat:.3f} / 有効サンプル数の最小={ess:.0f} / 発散={divergent}回")
for name in ("sigma_level", "sigma_obs", "slope"):
    v = post[name].to_numpy()
    print(f"{name}: 事後平均={v.mean():.4f} / 90%信用区間=[{np.quantile(v, 0.05):.4f}, {np.quantile(v, 0.95):.4f}]")

# 将来24か月: 事後分布の各サンプルで、水準の誤差と観測の誤差を引いて系列を作る
rng = np.random.default_rng(0)
last = post["level"].isel(level_dim_0=-1).to_numpy()
slope_s, sl, so = post["slope"].to_numpy(), post["sigma_level"].to_numpy(), post["sigma_obs"].to_numpy()
season_s = post["season"].to_numpy()  # (12, サンプル数)
paths = last + np.cumsum(slope_s + sl * rng.standard_normal((H, len(sl))), axis=0)
sims = np.exp(paths + season_s[month[n:]] + so * rng.standard_normal((H, len(so))))
pred, lo, hi = np.median(sims, axis=1), np.quantile(sims, 0.05, axis=1), np.quantile(sims, 0.95, axis=1)
covered = np.mean((test.to_numpy() >= lo) & (test.to_numpy() <= hi))
print(f"24か月先 MAE={mae(test, pred):.1f} / 90%区間 被覆率={covered:.1%} / 平均幅={np.mean(hi - lo):.1f}")
