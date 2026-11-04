"""公式CSVの保存版を検査し、縦持ちへ変換。--downloadで公式サイトから取得。"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from japan_data import DATA, ITEMS, read_official_files, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((DATA / 'source-manifest.json').read_text())
    if args.download:
        for source in manifest:
            request = urllib.request.Request(source['url'], headers={'User-Agent': 'Python-timeseries-book/1.0'})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            import hashlib
            if hashlib.sha256(data).hexdigest() != source['sha256']:
                raise ValueError('公開版が変わっています。既存版を上書きせず、新版として確認してください。')
            file = DATA / source['file']; file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(data)
    frame = read_official_files()
    frame.to_csv(DATA / 'selected_monthly.csv', index=False)
    quality = {'rows': len(frame), 'items': ITEMS, 'months_per_item': 120,
               'missing': int(frame.yen_per_household.isna().sum()),
               'duplicates': int(frame.duplicated(['month','item_code','region','household']).sum()),
               'derived_sha256': sha256(DATA / 'selected_monthly.csv')}
    (DATA / 'quality.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2)+'\n')
    print('家計調査: 全国・二人以上の世帯・1世帯当たり月間支出（円）')
    print(f'期間: {frame.month.min()}〜{frame.month.max()} / {len(ITEMS)}品目 / {len(frame)}行')
    print(f"欠損: {quality['missing']} / 重複: {quality['duplicates']} / 符号・名称: 一致")
    print('保存: data/japan/selected_monthly.csv')

if __name__ == '__main__':
    main()
