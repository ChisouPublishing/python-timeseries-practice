"""未来値の混入、実績の多重結合、最終結果と区間の再照合を検査する。"""
from pathlib import Path
import importlib.util
import json
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from japan_data import ROOT, DATA, read_official_files, series_for, make_features, score_once, assumed_available_at
spec=importlib.util.spec_from_file_location('japan_forecast',ROOT/'15_kakei/02_forecast.py')
forecast=importlib.util.module_from_spec(spec);spec.loader.exec_module(forecast)

class JapanCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame=read_official_files()
        cls.out=ROOT/'outputs/japan'
        cls.pred=pd.read_csv(cls.out/'test_predictions.csv',dtype={'item_code':str})

    def test_unavailable_values_cannot_change_ridge_prediction(self):
        y=series_for(self.frame,'356');target=pd.Period('2024-06','M')
        a=forecast.predict(y,make_features(y),target,'ridge')
        altered=y.copy();altered.loc[target-2:]=1e8
        b=forecast.predict(altered,make_features(altered),target,'ridge')
        self.assertAlmostEqual(a['prediction'],b['prediction'],places=10)
        self.assertEqual(a['history_end'],'2024-03')

    def test_duplicate_or_missing_actuals_are_rejected(self):
        p=self.pred
        actual=p[['item_code','target_month','actual_snapshot_id','actual']].drop_duplicates()
        clean=score_once(p.drop(columns='actual'),actual)
        self.assertEqual(len(clean),144)
        for bad in [pd.concat([actual,actual.iloc[:1]]),actual.iloc[1:]]:
            with self.assertRaises(AssertionError):score_once(p.drop(columns='actual'),bad)

    def test_final_metrics_and_calibration_are_traceable(self):
        reported=json.loads((self.out/'test_metrics.json').read_text())
        selected=json.loads((self.out/'selection.json').read_text())
        cal=pd.read_csv(self.out/'calibration_predictions.csv',dtype={'item_code':str})
        for (code,model),rows in self.pred.groupby(['item_code','model']):
            self.assertEqual(len(rows),12)
            y=series_for(self.frame,code)
            actual=np.array([y.loc[pd.Period(m,'M')] for m in rows.target_month])
            np.testing.assert_allclose(actual,rows.actual)
            self.assertAlmostEqual(np.abs(rows.prediction-actual).mean(),reported[code][model]['mae'])
            past=cal[(cal.item_code==code)&(cal.model==model)]
            self.assertEqual(len(past),24)
            errs=sorted(abs(r.prediction-y.loc[pd.Period(r.target_month,'M')]) for r in past.itertuples())
            radius=errs[22]
            self.assertAlmostEqual(radius,selected['items'][code]['calibration'][model]['absolute_error_radius'])
            np.testing.assert_allclose(rows.upper-rows.prediction,radius)
            self.assertTrue(all(pd.Period(m,'M')-3==pd.Period(h,'M') for m,h in zip(rows.target_month,rows.history_end)))

    def test_known_release_dates_are_no_later_than_scenario(self):
        rows=pd.read_csv(DATA/'release_dates_known.csv')
        observed=rows[rows.observation_month.str.startswith('2024-')]
        self.assertEqual(len(observed),12)
        for row in observed.itertuples():
            date=pd.Timestamp(row.released_on).tz_localize('Asia/Tokyo')
            self.assertLessEqual(date,assumed_available_at(pd.Period(row.observation_month,'M')))

if __name__=='__main__':unittest.main()
