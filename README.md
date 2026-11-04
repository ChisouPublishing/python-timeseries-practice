# Python実践 時系列データ分析：実行コード

公開先： https://github.com/ChisouPublishing/python-timeseries-practice

このフォルダの内容を公開リポジトリのルートとして使用する構成です。本文の `15_kakei/01_prepare.py` などは、このフォルダからの相対パスです。

## 環境

Python 3.12.11（macOS、Apple Silicon）で全スクリプトの実行を確認。初稿の実行環境Python 3.10.17の保存出力と、処理時間とPythonの版の表示を除いて一致しました。`requirements.txt` の依存関係を仮想環境へインストールします。掲載図の日本語表示にはHiragino SansまたはNoto Sans JPなどの日本語フォントが必要です。OSの違うクリーン環境での全スクリプト再検証は、書籍の制作台帳で別途管理しています。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 日本の公開データの主軸ケース

家計調査の保存済み公式CSVを使う通常実行はネットワーク不要です。データの意味、出典、利用条件、保存版、公表の遅れの仮定は [data/japan/README.md](data/japan/README.md) を読んでください。

```bash
python3 15_kakei/01_prepare.py
python3 15_kakei/02_forecast.py --stage develop --output-dir outputs/japan-reader
python3 15_kakei/02_forecast.py --stage evaluate --output-dir outputs/japan-reader
python3 15_kakei/03_report.py --output-dir outputs/japan-reader
python3 -m unittest discover -s tests -p 'test_japan_case.py' -v
```

掲載時の試行表、選択記録、較正・最終予測、警告、照合台帳、会議資料は `outputs/japan/`。既存の選択記録・最終テストを上書きしないため、再実行には別の出力先を指定します。コード・設定・入力台帳を選択後に変えた状態でevaluateへ進むことも拒否します。

公式から再取得するときは `python3 15_kakei/01_prepare.py --download`。保存版とハッシュが違えば上書きを止めるので、新版として確認してください。主教材はe-Stat APIキーを要求しません。

## 補助ケース

生鮮小売の実データによる追加実験は [PUBLIC_RETAIL_DATA.md](PUBLIC_RETAIL_DATA.md) を参照してください。FreshRetailNet-50Kの公式固定版（初回約115MB）を取得し、販売と欠品を区別して翌日予測を検証します。取得コードは `16_cases/05_public_download.py`、分析コードは `16_cases/06_public_retail.py`。数量は正規化値であり、日本のスーパーの円売上・実個数ではありません。

`make_data.py` が小売・センサー・株価・Webなどの教材用合成CSVを生成します。合成データの結果を、実在企業や市場での効果として扱いません。`data/airpassengers.csv` は別の既存時系列教材です。

共通処理は `common.py`、各章のコードは `02_env/`〜`17_ops/`、図は `figures/`、実行結果は `outputs/`。本書のコードは著者による実装教材であり、実行可能であることをAI生成応答の証明として扱いません。


## 付録F：ライブラリの基本操作

`appendix_basics/01_numpy.py`〜`07_optuna.py`に、NumPy・pandas・Matplotlib・scikit-learn・statsmodels・LightGBM・Optunaの独立した入門スクリプトを収録しています。依存関係は既存のrequirementsと同じです。外部データ取得は不要です。

```bash
python3 appendix_basics/01_numpy.py
python3 appendix_basics/02_pandas.py
python3 appendix_basics/03_matplotlib.py
python3 appendix_basics/04_sklearn.py
python3 appendix_basics/05_statsmodels.py
python3 appendix_basics/06_lightgbm.py
python3 appendix_basics/07_optuna.py
```

共通の合成データと特徴量は`appendix_basics/_data.py`。図は`figures/appendix_basics_forecast.svg`、探索履歴は`outputs/appendix_basics_optuna_trials.csv`、掲載時の標準出力は`outputs/appendix_basics_*.txt`、標準エラーは対応する`.txt.err`です。図と探索履歴は再実行時に更新される練習用成果物です。statsmodelsは初期値の警告を出すため、収束と残差診断も確認してください。

Matplotlib・statsmodelsは固定起点から28日先まで、scikit-learn・LightGBM・Optunaは毎日前日の実績を更新する翌日予測です。条件を混ぜてMAEで順位を付けません。付録の合成データによる精度を実在企業の業務効果として扱いません。
