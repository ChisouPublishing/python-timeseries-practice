"""実データ構成と、同じ小売CSVから取り出す店舗キーのER図。実行: python3 diagrams/02_data_schema.py"""
from html import escape
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'figures'
OUT.mkdir(exist_ok=True)

def start(h, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="850" height="{h}" viewBox="0 0 850 {h}" role="img" aria-label="{escape(title)}">',
    '<style>text{font-family:"Hiragino Sans","Noto Sans JP",sans-serif;font-size:18px;fill:#172338}.title{font-size:20px;font-weight:700}.small{font-size:17px;fill:#4b5f75}</style>']
def card(s,x,y,w,h,title,lines):
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#f4f8fc" stroke="#487daa" stroke-width="1.5"/>')
    s.append(f'<text x="{x+16}" y="{y+30}" class="title">{escape(title)}</text>')
    for i,line in enumerate(lines):s.append(f'<text x="{x+16}" y="{y+64+29*i}">{escape(line)}</text>')
s=start(480,'同梱CSVは独立した5つのデータ。小売のみ店舗別の複数系列。')
card(s,4,4,406,130,'航空旅客数',['月 × 旅客数（千人）','公開系列／月をキーにする'])
card(s,440,4,406,130,'小売売上',['店舗 × 日 × 売上・販促','店舗IDと日付の複合キー'])
card(s,4,164,406,130,'設備センサー',['分 × 振動・温度・正解ラベル','時刻がキー／単一設備'])
card(s,440,164,406,130,'Webトラフィック',['日 × PV','日付がキー／単一サイト'])
card(s,4,324,406,130,'合成株価',['平日 × 終値','日付がキー／単一系列'])
s += ['<text x="440" y="360" class="small">CSV間に外部キーはない。</text>','<text x="440" y="392" class="small">日付一致だけでは結合しない。</text>','<text x="440" y="424" class="small">単位・粒度もそれぞれ異なる。</text>','</svg>']
(OUT/'ch01_data_schema.svg').write_text('\n'.join(s))
s=start(310,'小売CSVから重複を除いて得た店舗キー集合と日次売上の1対多の関係。売上の主キーは店舗IDと日付。')
card(s,4,50,310,160,'店舗キー集合（導出）',['PK  store_id','小売CSVから重複を除く','配布マスタCSVは存在しない'])
card(s,414,4,432,276,'日次売上（retail_sales.csv）',['PK・FK  store_id','PK  date','sales：日次売上（欠損あり）','promo：販促予定の0 / 1','1店舗・1日につき1行'])
s += ['<path d="M314 130 H414 M328 119 V141 M400 130 L414 119 M400 130 L414 141" fill="none" stroke="#487daa" stroke-width="2"/>','<text x="336" y="105">1：多</text>','<text x="4" y="266" class="small">同じdateは3店舗に現れる。</text>','<text x="4" y="297" class="small">salesの欠損でも行は残る。</text>','</svg>']
(OUT/'ch01_retail_er.svg').write_text('\n'.join(s))
print('図: figures/ch01_data_schema.svg')
print('図: figures/ch01_retail_er.svg')
s=start(350,'運用時に提案する予測実行・予測明細・実績の台帳。実行と明細は1対多。実績は系列と対象日、確定版で後から照合する。実装済みの配布CSVとは異なる設計例。')
card(s,4,4,250,244,'予測実行',['PK  run_id','origin','model_version','data_version','学習ラベルの確定日'])
card(s,300,4,250,244,'予測明細',['PK・FK  run_id','PK  series_id','PK  target_date','prediction','fallback_reason'])
card(s,596,4,250,244,'実績の各版',['PK  series_id','PK  target_date','PK  actual_version','actual / status','available_at'])
s += ['<path d="M254 130 H300 M264 119 V141 M286 130 L300 119 M286 130 L300 141" fill="none" stroke="#487daa" stroke-width="2"/>','<text x="260" y="105">1:N</text>','<path d="M550 130 H596" fill="none" stroke="#d9822b" stroke-width="2" stroke-dasharray="5 4"/>','<text x="556" y="105">照合</text>','<text x="4" y="294" class="small">設計例。実績は対象日だけでなく、系列・確定時刻・版を選んで結合する。</text>','<text x="4" y="328" class="small">本書の配布CSVにこの3テーブルが実装済み、という意味ではない。</text>','</svg>']
(OUT/'ch17_prediction_ledger.svg').write_text('\n'.join(s))
print('図: figures/ch17_prediction_ledger.svg')
