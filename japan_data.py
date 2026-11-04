"""家計調査の保存版を読む。公開統計と教材用の情報到着シナリオを分離する。"""
from pathlib import Path
import csv
import hashlib
import json
import re
import numpy as np
import pandas as pd
import holidays

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/japan'
ITEMS = {'356': 'アイスクリーム・シャーベット', '352': 'チョコレート',
         '242': 'はくさい', '301': 'みかん'}
JP_HOLIDAYS = holidays.JP(years=range(2014, 2027))

def load_plan():
    return json.loads((DATA / 'experiment-plan.json').read_text())

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_official_files():
    manifest = json.loads((DATA / 'source-manifest.json').read_text())
    records = []
    for source in manifest:
        file = DATA / source['file']
        if sha256(file) != source['sha256']:
            raise ValueError(f'元ファイルのハッシュが違います: {file}')
        rows = list(csv.reader(file.read_bytes().decode('cp932').splitlines()))
        assert rows[3][6:8] == ['符号', '品目分類']
        assert rows[0][0] == '二人以上の世帯_支出金額[円]'
        dates, year = [], None
        for col in range(8, len(rows[3])):
            if rows[1][col].strip():
                year = int(re.fullmatch(r'(\d{4})年', rows[1][col].strip()).group(1))
            month = int(re.fullmatch(r'(\d{1,2})月', rows[3][col].strip()).group(1))
            dates.append((col, f'{year}-{month:02d}'))
        found = set()
        for row in rows[4:]:
            if len(row) < 8 or row[6] not in ITEMS:
                continue
            code = row[6]
            assert code not in found and row[7] == ITEMS[code]
            found.add(code)
            for col, month in dates:
                cell = row[col].strip()
                if not re.fullmatch(r'\d+(?:\.\d+)?', cell):
                    raise ValueError(f'数値でない値: {code} {month} {cell!r}')
                records.append({'month': month, 'item_code': code, 'item_name': row[7],
                                'region': '全国', 'household': '二人以上の世帯',
                                'yen_per_household': float(cell),
                                'classification_version': '2015' if '2015_2019' in file.name else '2020',
                                'snapshot_id': source['sha256'], 'source_file': file.name})
        assert found == set(ITEMS)
    frame = pd.DataFrame(records).sort_values(['item_code', 'month']).reset_index(drop=True)
    assert not frame.duplicated(['item_code', 'month', 'region', 'household']).any()
    expected = pd.period_range('2015-01', '2024-12', freq='M').astype(str).tolist()
    for _, group in frame.groupby('item_code'):
        assert group['month'].tolist() == expected
    return frame

def series_for(frame, code):
    rows = frame[frame.item_code == code]
    return pd.Series(rows.yen_per_household.to_numpy(),
                     index=pd.PeriodIndex(rows.month, freq='M'), name=code)

def origin_for(target):
    """会議日が土日・日本の祝日なら次の営業日。"""
    date = (target - 1).start_time + pd.Timedelta(days=14)
    while date.weekday() >= 5 or date.date() in JP_HOLIDAYS:
        date += pd.Timedelta(days=1)
    return date.tz_localize('Asia/Tokyo') + pd.Timedelta(hours=9)

def assumed_available_at(month):
    # 実際の公表日を捏造せず、教材上の到着条件として独立した名前で保存する。
    return ((month + 2).start_time + pd.Timedelta(days=14, hours=8, minutes=30)).tz_localize('Asia/Tokyo')

def make_features(y):
    x = pd.DataFrame(index=y.index)
    for lag in load_plan()['lag_months']:
        x[f'lag{lag}'] = y.shift(lag)
    for width in (3, 12):
        x[f'mean{width}'] = y.shift(3).rolling(width).mean()
        x[f'std{width}'] = y.shift(3).rolling(width).std()
    x['month_sin'] = np.sin(2*np.pi*x.index.month/12)
    x['month_cos'] = np.cos(2*np.pi*x.index.month/12)
    x['trend'] = np.arange(len(x))
    return x

def information_at(y, target):
    cutoff = target - load_plan()['available_history_lag']
    history = y.loc[:cutoff]
    origin = origin_for(target)
    assert all(assumed_available_at(m) <= origin for m in history.index)
    return history, cutoff, origin

def safe_prediction(row, available_months):
    """必要な最新月が未到着なら、公開済みの前年同月で代替。なければ保留。"""
    if row['required_month'] in available_months:
        return {'status': 'model', 'prediction': row['prediction']}
    if row['baseline_month'] in available_months:
        return {'status': 'seasonal_fallback', 'prediction': row['seasonal_prediction']}
    return {'status': 'hold', 'prediction': None}

def score_once(predictions, actuals):
    keys = ['item_code', 'target_month', 'actual_snapshot_id']
    assert not actuals.duplicated(keys).any(), '採点用実績版が一意ではありません'
    result = predictions.merge(actuals, on=keys, how='left', validate='many_to_one', indicator=True)
    assert (result['_merge'] == 'both').all(), '未公表・未到着の正解を0として扱わない'
    result['error'] = result['prediction'] - result['actual']
    return result.drop(columns='_merge')
