"""日本の家計支出を、情報到着の遅れを置いた回顧的評価で比較する。
実行: python3 15_kakei/02_forecast.py --stage develop
      python3 15_kakei/02_forecast.py --stage evaluate
最終テストを含む成果物のある出力先へは上書きしない。
"""
import argparse
import hashlib
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import sklearn
import statsmodels
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from japan_data import (ROOT, DATA, ITEMS, information_at, load_plan, make_features,
                        read_official_files, series_for, sha256)

optuna.logging.set_verbosity(optuna.logging.WARNING)
PLAN = load_plan()
MODELS = ['seasonal', 'ets', 'sarima', 'ridge', 'lightgbm', 'optuna']
WARNINGS = []


def signature():
    files = [Path(__file__), ROOT/'japan_data.py', DATA/'experiment-plan.json', DATA/'source-manifest.json']
    return {str(f.relative_to(ROOT)): sha256(f) for f in files}


def ml_model(params):
    if params['family'] == 'ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=params['alpha']))
    kwargs = {k:v for k,v in params.items() if k != 'family'}
    return lgb.LGBMRegressor(**kwargs, random_state=42, n_jobs=1, deterministic=True,
                            force_col_wise=True, verbosity=-1)


def predict(y, x, target, family, params=None):
    history, cutoff, origin = information_at(y, target)
    assert cutoff == target - 3 and history.index.max() == cutoff
    assert target - 12 in history.index
    baseline = float(history.loc[target-12])
    status = 'ok'
    start = time.perf_counter()
    if family == 'seasonal':
        value = baseline
    elif family in ('ets', 'sarima'):
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                if family == 'ets':
                    fitted = ExponentialSmoothing(history, **PLAN['ets']).fit(optimized=True)
                    value = float(fitted.forecast(3).iloc[-1])
                    converged = fitted.mle_retvals.get('success', True)
                else:
                    setting = PLAN['sarima']
                    fitted = SARIMAX(history, order=tuple(setting['order']),
                                     seasonal_order=tuple(setting['seasonal_order']),
                                     enforce_stationarity=False, enforce_invertibility=False).fit(
                                         disp=False, maxiter=setting['maxiter'])
                    value = float(fitted.forecast(3).iloc[-1])
                    converged = fitted.mle_retvals.get('converged', True)
                if captured:
                    WARNINGS.extend({'item': y.name, 'target': str(target), 'model': family,
                                     'message': str(w.message)} for w in captured)
                if not converged or not np.isfinite(value):
                    raise RuntimeError('収束条件を満たさない、または非有限の予測')
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            value = baseline
            status = 'seasonal_fallback'
            WARNINGS.append({'item':y.name, 'target':str(target), 'model':family, 'message':str(error)})
    else:
        if params is None:
            params = ({'family':'ridge', 'alpha':PLAN['default_ridge_alpha']} if family == 'ridge'
                      else {'family':'lightgbm', **PLAN['default_lightgbm']})
        train = x.loc[:cutoff].dropna()
        assert train.index.max() <= cutoff
        assert not x.loc[target].isna().any()
        model = ml_model(params)
        model.fit(train.to_numpy(), y.reindex(train.index).to_numpy())
        value = float(model.predict(x.loc[[target]].to_numpy())[0])
    # 支出額の非負制約は全モデルに同じように適用する。
    value = max(0.0, value)
    return {'item_code':y.name, 'target_month':str(target), 'origin':origin.isoformat(),
            'history_end':str(cutoff), 'model':family, 'prediction':value,
            'fit_seconds':time.perf_counter()-start, 'status':status,
            'availability_basis':'teaching_arrival_scenario', 'baseline_month':str(target-12)}


def forecast_months(y, x, months, family, params=None):
    return pd.DataFrame([predict(y,x,m,family,params) for m in months])


def choose_params(trial):
    family = trial.suggest_categorical('family', ['ridge', 'lightgbm'])
    if family == 'ridge':
        return {'family':family, 'alpha':trial.suggest_float('alpha',0.1,1000,log=True)}
    return {'family':family, 'n_estimators':trial.suggest_int('n_estimators',50,200,step=50),
            'learning_rate':trial.suggest_float('learning_rate',0.02,0.2,log=True),
            'num_leaves':trial.suggest_int('num_leaves',3,15),
            'min_child_samples':trial.suggest_int('min_child_samples',5,20),
            'reg_lambda':trial.suggest_float('reg_lambda',0.001,10,log=True)}


def errors_for(pred, y):
    actual = np.array([y.loc[pd.Period(m,'M')] for m in pred.target_month])
    return pred.prediction.to_numpy() - actual


def summary(pred, y):
    errors = errors_for(pred,y)
    actual = np.array([y.loc[pd.Period(m,'M')] for m in pred.target_month])
    peak = np.array([pd.Period(m,'M').month in PLAN['business_peak_months'][y.name] for m in pred.target_month])
    return {'mae':float(np.abs(errors).mean()),
            'wape_percent':float(100*np.abs(errors).sum()/np.abs(actual).sum()),
            'bias':float(errors.mean()), 'peak_mae':float(np.abs(errors[peak]).mean()),
            'mean_fit_seconds':float(pred.fit_seconds.mean()),
            'fallbacks':int((pred.status != 'ok').sum())}


def check_causality(y, x):
    for target in [pd.Period('2020-01','M'), pd.Period('2024-01','M')]:
        changed = y.copy(); changed.loc[target-2:] = 1e8
        pd.testing.assert_frame_equal(x.loc[:target],make_features(changed).loc[:target])
        # 学習行も特徴量も、未到着の正解で変化しない。
        history,cutoff,_ = information_at(y,target)
        other,_,_ = information_at(changed,target)
        pd.testing.assert_series_equal(history,other)


def develop(out):
    if (out/'selection.json').exists():
        raise FileExistsError('既存の選択記録を上書きしません。別の出力先を指定してください。')
    frame = read_official_files()
    all_selection, all_calibration, decisions = [], [], {}
    for code in PLAN['main_items']:
        y = series_for(frame,code); x = make_features(y)
        check_causality(y,x)
        folds = [pd.period_range(a,b,freq='M') for a,b in PLAN['fold_months']]
        def objective(trial):
            params = choose_params(trial)
            errors = []
            for step,months in enumerate(folds):
                pred = forecast_months(y,x,months,'optuna',params)
                errors.extend(np.abs(errors_for(pred,y)).tolist())
                trial.report(float(np.mean(errors)),step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(errors))
        study = optuna.create_study(direction='minimize',
                    sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=6),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=8,n_warmup_steps=1))
        study.optimize(objective,n_trials=PLAN['trials_per_item'],n_jobs=1)
        study.trials_dataframe().to_csv(out/f'trials_{code}.csv',index=False)
        best = study.best_params
        stats = {}; candidates = {}
        for family in MODELS:
            pred = forecast_months(y,x,pd.period_range(*PLAN['selection_months'],freq='M'),family,
                                   best if family == 'optuna' else None)
            stats[family] = summary(pred,y); candidates[family] = pred
            all_selection.append(pred)
        base = stats['seasonal']
        eligible = [m for m in MODELS if m != 'seasonal'
                    and stats[m]['mae'] <= (1-PLAN['adoption_min_cv_improvement'])*base['mae']
                    and stats[m]['peak_mae'] <= PLAN['adoption_max_peak_mae_ratio']*base['peak_mae']]
        selected = min(eligible,key=lambda m:stats[m]['mae']) if eligible else 'seasonal'
        calibration = {}
        for family in MODELS:
            pred = forecast_months(y,x,pd.period_range(*PLAN['calibration_months'],freq='M'),family,
                                   best if family == 'optuna' else None)
            residuals = np.sort(np.abs(errors_for(pred,y)))
            rank = math.ceil((len(residuals)+1)*PLAN['interval_nominal'])
            assert rank <= len(residuals)
            calibration[family] = {'absolute_error_radius':float(residuals[rank-1]),
                                   'n':len(residuals), 'order_statistic':rank}
            all_calibration.append(pred)
        decisions[code] = {'item_name':ITEMS[code], 'selected_model':selected,
                           'best_params':best, 'selection_metrics':stats,
                           'calibration':calibration,
                           'trials':{s:int(sum(t.state.name==s for t in study.trials))
                                     for s in ('COMPLETE','PRUNED','FAIL')}}
        print(f'{ITEMS[code]}: 選択={selected} / 探索={best}',flush=True)
    pd.concat(all_selection).to_csv(out/'selection_predictions.csv',index=False)
    pd.concat(all_calibration).to_csv(out/'calibration_predictions.csv',index=False)
    record = {'plan':PLAN,'signature':signature(),'items':decisions,
              'versions':{'python':platform.python_version(),'pandas':pd.__version__,
                          'numpy':np.__version__,'statsmodels':statsmodels.__version__,
                          'lightgbm':lgb.__version__,'optuna':optuna.__version__,'sklearn':sklearn.__version__}}
    (out/'selection.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    (out/'development_warnings.json').write_text(json.dumps(WARNINGS,ensure_ascii=False,indent=2)+'\n')


def evaluate(out):
    if (out/'test_metrics.json').exists():
        raise FileExistsError('最終テストは既に採点済みです。既存記録は変更しません。')
    frozen = json.loads((out/'selection.json').read_text())
    assert frozen['signature'] == signature(), '選択後にコード・設定・入力台帳が変わっています'
    frame = read_official_files(); all_pred=[]; metrics={}
    for code in PLAN['main_items']:
        y = series_for(frame,code); x = make_features(y)
        decision = frozen['items'][code]; metrics[code]={}
        actual_rows = frame[frame.item_code==code].set_index('month')
        for family in MODELS:
            pred = forecast_months(y,x,pd.period_range(*PLAN['test_months'],freq='M'),family,
                                  decision['best_params'] if family=='optuna' else None)
            radius = decision['calibration'][family]['absolute_error_radius']
            pred['lower'] = np.maximum(0,pred.prediction-radius)
            pred['upper'] = pred.prediction+radius
            pred['actual'] = [y.loc[pd.Period(m,'M')] for m in pred.target_month]
            pred['actual_snapshot_id'] = [actual_rows.loc[m,'snapshot_id'] for m in pred.target_month]
            pred['selected_before_test'] = family == decision['selected_model']
            stat = summary(pred,y)
            stat.update({'coverage_percent':float(100*((pred.actual>=pred.lower)&(pred.actual<=pred.upper)).mean()),
                         'mean_width':float((pred.upper-pred.lower).mean())})
            metrics[code][family] = stat; all_pred.append(pred)
        print(ITEMS[code]+' / 選択期間で固定した候補: '+decision['selected_model'])
        for family in MODELS:
            m=metrics[code][family]
            print(f"  {family:10s} MAE={m['mae']:.1f} WAPE={m['wape_percent']:.1f}% "
                  f"Bias={m['bias']:.1f} 被覆={m['coverage_percent']:.1f}% 幅={m['mean_width']:.1f}")
    pd.concat(all_pred).to_csv(out/'test_predictions.csv',index=False)
    (out/'test_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    (out/'test_warnings.json').write_text(json.dumps(WARNINGS,ensure_ascii=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['develop','evaluate'],required=True)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/japan')
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.stage=='develop': develop(args.output_dir)
    else: evaluate(args.output_dir)

if __name__=='__main__':main()
