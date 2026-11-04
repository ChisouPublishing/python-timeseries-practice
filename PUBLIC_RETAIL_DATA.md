# 生鮮小売の公開データ

データ作成者：Dingdong-Inc。FreshRetailNet-50K, version 1.0 (2025)。

- 配布元：https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
- 本書の固定版：`c374e85975c82816c495acfa904cc55ab14a47d3`
- ライセンス：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 論文：Yangyang Wang et al., *FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail*, 2025, https://arxiv.org/abs/2505.16319

商用・非商用利用可能との公式記載を確認（2026-11-04）。本書は店舗0の商品ID昇順20系列を抽出し、06:00〜21:59を集計、欠品フラグを用いた予測・検査を追加する。これらは本書による加工であり、データ提供者による推奨・成果ではない。元データの数量は共通係数を掛けた正規化値で、円や実個数ではない。海外の生鮮小売データを日本企業の実績と扱わない。

```bash
python3 16_cases/05_public_download.py
python3 16_cases/06_public_retail.py --output-dir outputs/freshretail-reader
python3 -m unittest discover -s tests -p 'test_public_retail.py' -v
```

`pyarrow==25.0.1`を含むrequirementsが必要。初回取得は約115MB、保存先は`data/freshretail/`（Git対象外）。固定版のSHA-256は`retail_public.py`に保存し、取得時・読込時に照合する。2回目の実行で結果を上書きしないよう、別の出力先を指定する。

公式trainの末尾14日で8候補を探索し、選択記録を保存してから公式evalを採点する。毎日前日までの販売・欠品実績を更新する翌日1期先の評価で、7日先一括予測ではない。天気実測、対象日の実欠品、対象日の販促実績を予測特徴へ使わない。前日実績が未到着なら自動処理を止める。

真の潜在需要、在庫個数、入荷・発注・廃棄、売価・仕入原価、データの実到着時刻は公開されていない。本実験は前日分が翌朝までに確定する教材上の運用条件。欠品なし日のMAEもその部分集合での評価であり、全日の潜在需要に対する精度ではない。
