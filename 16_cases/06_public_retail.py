"""実販売と欠品を分けた翌日予測。探索後に公式evalを読み、一度採点する。"""
import argparse
import json
import platform
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import optuna
import pandas as pd
from retail_public import load_split, REVISION, HASHES

HOURS = slice(6, 22)  # 教材の対象時間を06:00〜21:59と定義。


def unpack(frame):
    if not frame.hours_sale.map(len).eq(24).all() or not frame.hours_stock_status.map(len).eq(24).all():
        raise ValueError("時間配列は24要素が必要")
    sales = np.stack(frame.hours_sale).astype(float)[:, HOURS]
    stock = np.stack(frame.hours_stock_status).astype(int)[:, HOURS]
    if sales.shape[1] != 16 or not np.isin(stock, [0, 1]).all():
        raise ValueError("時間ベクトルが不正")
    if not np.isfinite(sales).all() or (sales < 0).any():
        raise ValueError("販売数量が不正")
    return sales, stock


def predict_day(history, date, lookback, method):
    """date未満だけで、同曜日・同時刻の在庫あり販売を集計する。"""
    past = history[(history.dt < date) & (history.dt >= date - pd.Timedelta(days=lookback))]
    if past.empty:
        raise ValueError("過去の販売が未到着。予測を保留してください")
    y, s = unpack(past)
    # 欠品履歴まで欠けた場合の代替は、過去の観測販売の平均。
    fallback = y.mean(axis=0)
    same = past.dt.dt.dayofweek.to_numpy() == date.dayofweek
    result = []
    for hour in range(16):
        values = y[same & (s[:, hour] == 0), hour]
        if not len(values):
            values = y[s[:, hour] == 0, hour]
        result.append((np.mean(values) if method == "mean" else np.median(values))
                      if len(values) else fallback[hour])
    return float(np.sum(result))


def forecasts(history, targets, lookback, method):
    combined = pd.concat([history, targets], ignore_index=True)
    rows = []
    for pid, group in combined.groupby("product_id"):
        target = targets[targets.product_id == pid]
        y, stock = unpack(target)
        for i, row in enumerate(target.itertuples()):
            prior7 = group[group.dt == row.dt - pd.Timedelta(days=7)]
            if len(prior7) != 1:
                raise ValueError("前週同曜日が欠落")
            baseline = float(unpack(prior7)[0].sum())
            latest = group[group.dt == row.dt - pd.Timedelta(days=1)]
            if len(latest) != 1:
                raise ValueError("前日の実績が未到着。自動処理を保留")
            alert = bool(unpack(latest)[1].sum() > 0)
            rows.append({"product_id": int(pid), "dt": row.dt.date().isoformat(),
                         "actual": float(y[i].sum()), "clean": bool(stock[i].sum() == 0),
                         "baseline": baseline,
                         "stock_aware": predict_day(group, row.dt, lookback, method),
                         "review_previous_stockout": alert})
    return pd.DataFrame(rows)


def metrics(frame):
    result = {}
    for label, mask in [("all_observed", np.ones(len(frame), dtype=bool)),
                        ("no_stockout", frame.clean.to_numpy())]:
        part = frame[mask]
        if part.empty:
            raise ValueError("欠品なし評価対象がない")
        result[label] = {"n": len(part), **{
            model: float(np.mean(np.abs(part.actual - part[model])))
            for model in ("baseline", "stock_aware")}}
    return result


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    train, products = load_split("train")
    boundary = train.dt.max() - pd.Timedelta(days=13)
    fit, valid = train[train.dt < boundary], train[train.dt >= boundary]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(
        {"lookback": [7, 14, 21, 28], "method": ["mean", "median"]}, seed=42))
    def objective(trial):
        lookback = trial.suggest_categorical("lookback", [7, 14, 21, 28])
        method = trial.suggest_categorical("method", ["mean", "median"])
        return metrics(forecasts(fit, valid, lookback, method))["no_stockout"]["stock_aware"]
    study.optimize(objective, n_trials=8)
    frozen = {"revision": REVISION, "hashes": HASHES, "store": 0, "products": products,
              "validation_start": str(boundary.date()), "params": study.best_params,
              "selection_metric": "MAE on no-stockout hours 06:00-21:59 days",
              "validation_mae": study.best_value}
    # evalを開く前に設定を保存。最終順位に合わせて探索し直さない。
    (output / "selection.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    study.trials_dataframe().to_csv(output / "trials.csv", index=False)
    evaluation, _ = load_split("eval", products)
    assert train.dt.max() < evaluation.dt.min()
    predictions = forecasts(train, evaluation, **study.best_params)
    predictions.to_csv(output / "predictions.csv", index=False)
    scores = metrics(predictions)
    sales, status = unpack(train)
    summary = {"python": platform.python_version(), "pandas": pd.__version__,
               "numpy": np.__version__, "optuna": optuna.__version__,
               "train_rows": len(train), "eval_rows": len(evaluation),
               "train_days": train.dt.nunique(), "eval_days": evaluation.dt.nunique(),
               "stockout_hours": int(status.sum()), "observed_hours": int(status.size),
               "zero_hours": int((sales == 0).sum()),
               "zero_and_stockout_hours": int(((sales == 0) & (status == 1)).sum()),
               "previous_stockout_alerts": int(predictions.review_previous_stockout.sum()),
               "metrics": scores}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"対象: 店舗0・商品{len(products)}系列 / 学習{len(train)}行・{train.dt.nunique()}日 / 最終{len(evaluation)}行・{evaluation.dt.nunique()}日")
    print(f"対象時間: 06:00〜21:59 / 学習中の欠品時間: {status.sum()}/{status.size}")
    print(f"販売0の時間: {(sales == 0).sum()} / うち欠品: {((sales == 0) & (status == 1)).sum()}")
    print(f"開発末尾14日・欠品なし日のMAEで8候補を選択: {study.best_params}")
    print("最終評価（数量は正規化値）  件数  前週同曜日MAE  欠品考慮MAE")
    for label, score in scores.items():
        print(f"{label:14s} {score['n']:4d} {score['baseline']:12.3f} {score['stock_aware']:12.3f}")
    print(f"前日の欠品による人手確認候補: {summary['previous_stockout_alerts']}件")
    print("欠品中の真の需要・利益・廃棄削減は、このデータだけでは評価できない。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/freshretail"))
    run(parser.parse_args().output_dir)
