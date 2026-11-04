"""固定済みの比較結果から図・会議資料・運用検査を出力する。再探索はしない。"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from common import C_BLUE,C_ORANGE,C_GRAY,C_TEAL,savefig
from japan_data import ROOT,DATA,ITEMS,read_official_files,series_for,safe_prediction,score_once

OUT = ROOT/'outputs/japan'
LABELS = {'seasonal':'前年同月','ets':'ETS','sarima':'SARIMA','ridge':'Ridge',
          'lightgbm':'LightGBM','optuna':'Optuna選択'}

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=OUT)
    OUT=parser.parse_args().output_dir
    selection=json.loads((OUT/'selection.json').read_text())
    metrics=json.loads((OUT/'test_metrics.json').read_text())
    frame=read_official_files()
    pred=pd.read_csv(OUT/'test_predictions.csv',dtype={'item_code':str})
    codes=selection['plan']['main_items']
    fig,axes=plt.subplots(2,1,figsize=(6.2,6.0))
    for ax,code in zip(axes,codes):
        y=series_for(frame,code).loc[:'2019-12']
        grouped=y.groupby(y.index.month)
        ax.plot(range(1,13),grouped.mean(),marker='o',color=C_BLUE)
        ax.fill_between(range(1,13),grouped.min(),grouped.max(),color=C_BLUE,alpha=.12,
                        label='2015〜2019年の最小〜最大')
        ax.set(title=ITEMS[code],xlabel='月',ylabel='支出額（円/世帯/月）',xticks=range(1,13))
        ax.legend(fontsize=8)
    print('図:',savefig(fig,'ch15_kakei_seasons'))
    fig,axes=plt.subplots(2,1,figsize=(6.2,6.4))
    for ax,code in zip(axes,codes):
        chosen=selection['items'][code]['selected_model']
        rows=pred[(pred.item_code==code)&(pred.model==chosen)].sort_values('target_month')
        base=pred[(pred.item_code==code)&(pred.model=='seasonal')].sort_values('target_month')
        months=np.arange(1,13)
        ax.plot(months,rows.actual,color=C_BLUE,marker='o',label='公表系列の実績')
        ax.plot(months,rows.prediction,color=C_ORANGE,marker='s',label='事前選択: '+LABELS[chosen])
        if chosen!='seasonal': ax.plot(months,base.prediction,color=C_GRAY,linestyle='--',label='前年同月')
        ax.fill_between(months,rows.lower.to_numpy(),rows.upper.to_numpy(),color=C_ORANGE,alpha=.15,
                        label='較正期間で固定した90%経験的区間')
        ax.set(title=ITEMS[code],xlabel='2024年の対象月',ylabel='支出額（円/世帯/月）',xticks=months)
        ax.legend(fontsize=8,ncol=2)
    print('図:',savefig(fig,'ch15_kakei_forecasts'))
    fig,axes=plt.subplots(2,1,figsize=(6.2,6.0))
    for ax,code in zip(axes,codes):
        families=list(LABELS)
        chosen=selection['items'][code]['selected_model']
        ax.barh([LABELS[m] for m in families],[metrics[code][m]['mae'] for m in families],
                color=[C_ORANGE if m==chosen else C_BLUE for m in families])
        ax.invert_yaxis();ax.set(title=ITEMS[code],xlabel='2024年 MAE（円/世帯/月）')
    print('図:',savefig(fig,'ch15_kakei_comparison'))
    # 集計統計から作る分析用テーブル。元の政府DB構造を推測する図ではない。
    fig,ax=plt.subplots(figsize=(6.2,4.6));ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
    def box(x,y,title,detail):
        ax.add_patch(FancyBboxPatch((x,y),2.8,1.25,boxstyle='round,pad=0.04',
                    facecolor='#F0F5FA',edgecolor=C_BLUE))
        ax.text(x+1.4,y+.91,title,ha='center',va='center',fontsize=10,fontweight='bold')
        ax.text(x+1.4,y+.39,detail,ha='center',va='center',fontsize=9.5)
    box(.15,4.3,'元ファイルの保存版','表ID・取得日\nハッシュ')
    box(3.6,4.3,'月次支出の分析表','保存版・世帯・地域\n品目・対象月')
    box(7.05,4.3,'品目と分類版','符号・名称\n定義・分類版')
    box(.15,.8,'実行・予測台帳','起点・入力版\n設定・対象月')
    box(3.6,.8,'実績との照合','系列・対象月\n採点用の保存版')
    box(7.05,.8,'会議向け報告','誤差・区間\n採否・限界')
    for start,end in [((3.0,4.9),(3.5,4.9)),((7,4.9),(6.5,4.9)),
                      ((5,4.25),(5,2.1)),((3,1.4),(3.5,1.4)),((6.5,1.4),(7,1.4))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='->',mutation_scale=13,color=C_TEAL))
    ax.text(5,3.1,'実績は1つの版を選ぶ\n改訂版との多重結合を防ぐ',ha='center',fontsize=9)
    print('図:',savefig(fig,'ch15_kakei_schema'))
    assert not pred.duplicated(['item_code','target_month','model']).any()
    actuals=pred[['item_code','target_month','actual_snapshot_id','actual']].drop_duplicates()
    scored=score_once(pred.drop(columns='actual'),actuals)
    assert len(scored)==144 and np.isfinite(scored.error).all()
    scored.to_csv(OUT/'score_ledger.csv',index=False)
    first=pred[pred.selected_before_test].iloc[0]
    baseline=pred[(pred.item_code==first.item_code)&(pred.target_month==first.target_month)&(pred.model=='seasonal')].iloc[0]
    request={'required_month':first.history_end,'baseline_month':first.baseline_month,
             'prediction':float(first.prediction),'seasonal_prediction':float(baseline.prediction)}
    normal=safe_prediction(request,{first.history_end,first.baseline_month})
    delayed=safe_prediction(request,{first.baseline_month})
    hold=safe_prediction(request,set())
    assert normal['status']=='model' and delayed['status']=='seasonal_fallback' and hold['status']=='hold'
    rejected={}
    for name,table in [('duplicate_actual',pd.concat([actuals,actuals.iloc[:1]])),
                       ('missing_actual',actuals.iloc[1:])]:
        try: score_once(pred.drop(columns='actual'),table)
        except (AssertionError,pd.errors.MergeError):rejected[name]=True
        else:raise AssertionError(name+' was not rejected')
    check={'normal':normal,'delayed_latest_month':delayed,'all_missing':hold,
           'rejected':rejected,'scored_rows':len(scored),
           'scenario_note':'Controlled arrival-delay drill, not an observed government outage.'}
    (OUT/'operations_check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
    print('運用検査: 通常=model / 最新月未到着=seasonal_fallback / 全履歴未到着=hold')
    print('照合検査: 実績重複と実績欠落を拒否 / 採点台帳144行')
    reports=[]
    for code in codes:
        chosen=selection['items'][code]['selected_model']; m=metrics[code][chosen]
        rows=pred[(pred.item_code==code)&pred.selected_before_test]
        reports.append({'item_code':code,'item_name':ITEMS[code],'selected_before_test':chosen,
                        'test_mae':m['mae'],'test_wape_percent':m['wape_percent'],
                        'coverage_percent':m['coverage_percent'],'interval_mean_width':m['mean_width'],
                        'baseline_test_mae':metrics[code]['seasonal']['mae'],
                        'needs_future_recheck':True,
                        'decision_after_test':('新候補の導入を保留し前年同月を維持' if m['mae']>metrics[code]['seasonal']['mae'] else '事前選択の基準予測を維持し別期間で再確認'),
                        'use':'全国の二人以上の世帯の支出動向の参考',
                        'not_supported':'店舗の発注数量・在庫削減・利益改善の推計'})
    (OUT/'meeting_report.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n')
    lines=['# 季節商品企画のための支出見通し：2024年回顧的評価','',
           '取得時点の公表系列と教材用到着シナリオを使用。当時の初回公表値による実運用再現ではない。','',
           '| 品目 | 事前選択 | MAE（円/世帯/月） | 前年同月MAE | 90%経験的区間の被覆 |',
           '| --- | --- | --- | --- | --- |']
    for r in reports:
        lines.append(f"| {r['item_name']} | {LABELS[r['selected_before_test']]} | {r['test_mae']:.1f} | {r['baseline_test_mae']:.1f} | {r['coverage_percent']:.1f}% |")
    lines.extend(['','アイスクリームのETSは最終期間で前年同月に劣ったため、導入を保留する。最終期間で良かった別モデルへ、その場で差し替えない。', 'チョコレートは選択時の基準に沿って前年同月を維持する。','翌月以降の別期間で再確認する。12か月の被覆率だけで90%の保証を主張しない。',
                  '店舗POS・単価・在庫・納期・欠品・販促計画はこのデータにない。全国の家計支出から店舗発注数を直接算出しない。',
                  '出典：総務省統計局「家計調査」 https://www.stat.go.jp/data/kakei/longtime/index.html を加工して作成。'])
    (OUT/'meeting_report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
