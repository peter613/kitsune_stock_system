import matplotlib
matplotlib.use('TkAgg') # 確保在終端機執行時能彈出視窗
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from catboost import CatBoostRanker
import os
import yfinance as yf
import config

# ==========================================
# 升級版：匯出帶有「買入原因 (SHAP)」的交易日誌
# ==========================================
def export_full_trade_log(test_df, filename, select_rank, weights=None, model=None):
    import shap  # 確保有安裝 pip install shap
    all_trade_records = []
    
    unique_dates = sorted(test_df['date'].unique())
    feature_names = model.feature_names_
    
    # 預先計算全樣本的 SHAP 貢獻度（這需要一點時間，但很值得）
    print(f"正在分析選股原因 (SHAP Analysis)...")
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(test_df[feature_names])

    for date in unique_dates:
        # 取得該日期的資料索引
        date_mask = test_df['date'] == date
        group = test_df[date_mask].copy()
        shap_group = shap_values_all[test_df[date_mask].index]
        
        # 加上排名
        group['rank'] = group['predict_score'].rank(ascending=False, method='first').astype(int)
        top_picks = group[group['rank'].isin(select_rank)].copy()
        
        if top_picks.empty: continue

        # 計算當月權重平均報酬 (與主引擎同步)
        if weights is None:
            monthly_avg = top_picks['return_20d'].mean() * 100
        else:
            current_weights = top_picks['rank'].map(weights)
            normalized_weights = current_weights / current_weights.sum()
            monthly_avg = np.dot(top_picks['return_20d'], normalized_weights) * 100

        for i, (idx_in_df, row) in enumerate(top_picks.iterrows()):
            # --- 核心：找出該股票前三大的貢獻特徵 ---
            # 取得該行在總表中的相對位置
            abs_idx = test_df.index.get_loc(idx_in_df)
            stock_shap = shap_values_all[abs_idx]
            
            # 找出貢獻度最高的前 6 個特徵名稱
            top_indices = np.argsort(stock_shap)[-6:][::-1]
            reasons = [f"{feature_names[j]}({stock_shap[j]:.2f})" for j in top_indices]
            reason_str = " | ".join(reasons)
            # ---------------------------------------

            all_trade_records.append({
                '換股日期': date.date(),
                '當月組合平均報酬%': round(monthly_avg, 2),
                '排名': row['rank'],
                '股票代碼': row['id'],
                '股票名稱': row['name'],
                '預測分數': round(row['predict_score'], 4),
                '選股主因 (Feature:Contribution)': reason_str, # 🌟 新增這一欄
                '該股當月報酬%': round(row['return_20d'] * 100, 2),
                '成交量/值(Log)': round(row['amount60'], 2) if 'amount60' in row else 'N/A',
                'MOM': round(row['mom'], 2) if 'mom' in row else 'N/A'
            })
            
    full_log_df = pd.DataFrame(all_trade_records)
    full_log_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ 帶原因分析的日誌已存至: {filename}")

def apply_limit_up_filter(test_df):
    print("正在載入 market.parquet 執行漲停濾網分析...")
    # 讀取 market 資料庫 (建議使用 parquet 速度較快)
    market_df = pd.read_parquet(config.DATABASE_DIR / "market.parquet") # 確保路徑正確
    market_df['date'] = pd.to_datetime(market_df['date'])
    market_df['id'] = market_df['id'].astype(str)
    
    # 計算漲停標記 (台股 10%)
    # 這裡假設 close / prev_close >= 1.098 作為漲停判斷
    market_df = market_df.sort_values(['id', 'date'])
    market_df['is_limit_up'] = (market_df['close'] / market_df.groupby('id')['close'].shift(1) >= 1.09).astype(int)
    
    # 計算 5 日內漲停累計次數
    market_df['limit_up_5d'] = market_df.groupby('id')['is_limit_up'].transform(lambda x: x.rolling(window=5).sum())
    
    # 將濾網合併回 test_df
    # 注意：我們需要的是「換股日當天」看「過去 5 天」的資料
    filter_data = market_df[['id', 'date', 'limit_up_5d']]
    test_df = pd.merge(test_df, filter_data, on=['id', 'date'], how='left')
    
    # 執行過濾：若 5 日內漲停次數 >= 2，強制將預測分數歸零
    # 這樣在後續 rank 排序時，這些股票會被排到最後面
    original_count = len(test_df)
    exclude_mask = test_df['limit_up_5d'] >= 2
    test_df.loc[exclude_mask, 'predict_score'] = -999 
    
    excluded_stocks = test_df[exclude_mask]['name'].unique()
    print(f"漲停濾網執行完畢，共過濾 {exclude_mask.sum()} 筆交易紀錄。")
    if len(excluded_stocks) > 0:
        print(f"過濾標的範例: {excluded_stocks[:5]}...")
        
    return test_df
# ==========================================
# 主回測函式
# ==========================================
def backtester(weight,version='v7', select_rank=[3, 4, 5], benchmark='0050.TW'):
    weights = weight
    """
    select_rank: list, 指定要選取的排名 (例如 [3, 4, 5] 代表只買第 3, 4, 5 名)
    """
    
    # 路徑設定
    MODEL_FILE = config.CBM_DIR / f"strategy_{version}.cbm"
    DATA_FILE = config.FEATURES_DIR / f"train_{version}.csv"
    
    INITIAL_CAPITAL = 1000000
    TRANSACTION_COST = 0.005

    rank_str = ",".join(map(str, select_rank))
    print(f"\n啟動回測系統 | 版本: {version} | 策略: 只買第 [{rank_str}] 名 | 對照組: {benchmark}")
    # print(f"資料來源: {DATA_FILE}")

    if not os.path.exists(DATA_FILE): raise FileNotFoundError(f"❌ 找不到資料檔: {DATA_FILE}")
    if not os.path.exists(MODEL_FILE): raise FileNotFoundError(f"❌ 找不到模型檔: {MODEL_FILE}")

    # 1. 讀取資料
    df = pd.read_csv(DATA_FILE, dtype={'id': str})
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. 切分回測資料
    unique_dates = sorted(df['date'].unique())
    split_idx = int(len(unique_dates) * 0.8)
    start_test_date = unique_dates[split_idx]
    
    test_df = df[df['date'] >= start_test_date].copy()
    test_df = test_df.sort_values(['date', 'id']).reset_index(drop=True)

    print(f"回測期間: {test_df['date'].min().date()} 到 {test_df['date'].max().date()}")
    # ==========================================
    # 🔥 核心新增：實盤 -15% 個股停損模擬
    # 邏輯：只要個股當月真實報酬率低於 -15%，就強制視為 -15% 出場
    # ==========================================
    test_df['return_20d'] = test_df['return_20d'].clip(lower=-0.15)

    # 3. 載入模型
    # print(f"載入模型: {MODEL_FILE}...")
    model = CatBoostRanker()
    model.load_model(MODEL_FILE)

    feature_cols = model.feature_names_
    missing_cols = [c for c in feature_cols if c not in test_df.columns]
    if missing_cols: raise ValueError(f"🚨 資料缺失以下欄位: {missing_cols}")

    test_df_features = test_df[feature_cols]

    # 4. 執行預測
    print("計算預測分數...")
    test_df['predict_score'] = model.predict(test_df_features)
    
    # 在排序前加入漲停濾網
    # test_df = apply_limit_up_filter(test_df)

    # 5. 模擬交易
    monthly_results = []
    capital = INITIAL_CAPITAL
    equity_curve = [capital] 

    for date, group in test_df.groupby('date'):
        # 1. 先排序
        sorted_group = group.sort_values('predict_score', ascending=False)
        
        # 2. 加上排名
        sorted_group['rank'] = range(1, len(sorted_group) + 1)
        
        # 3. [關鍵] 只選出指定排名的股票
        top_picks = sorted_group[sorted_group['rank'].isin(select_rank)]
        
        # 防呆：如果該月股票太少，選不到第 3,4,5 名怎麼辦？
        if top_picks.empty:
            print(f"⚠️ {date.date()}: 選不到指定排名的股票 (跳過交易)")
            equity_curve.append(capital) # 資產持平
            continue

        if top_picks['return_20d'].isna().any():
            print(f"⚠️ {date.date()}: 尚未結算 (跳過)")
            continue

        # 計算損益
        avg_return = top_picks['return_20d'].mean() - TRANSACTION_COST


        # 1. 優先處理「等權重」情況：如果傳入的是 None，直接計算平均報酬
        if weights is None:
            avg_return = top_picks['return_20d'].mean() - TRANSACTION_COST
            
        else:
            # 2. 處理「手動權重」情況：進行 map 映射
            current_weights = top_picks['rank'].map(weights)
            
            # 防呆：如果有排名沒對應到權重，或是權重總合為 0
            if current_weights.isna().any() or current_weights.sum() == 0:
                avg_return = top_picks['return_20d'].mean() - TRANSACTION_COST
            else:
                # 重新歸一化並計算點積 (dot product)
                normalized_weights = current_weights / current_weights.sum()
                avg_return = np.dot(top_picks['return_20d'], normalized_weights) - TRANSACTION_COST
        # ----------------------------------------------------

        profit = capital * avg_return
        capital += profit
        equity_curve.append(capital)
        
        monthly_results.append({
            'date': date,
            'avg_return': avg_return,
            'total_capital': capital
        })

    # 6. 績效結算
    results_df = pd.DataFrame(monthly_results)

    if not results_df.empty:
        total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
        annual_return = ((1 + total_return) ** (12 / len(results_df))) - 1
        winrate = (results_df['avg_return'] > 0).mean() * 100
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        print("\n" + "="*80)
        print(f"回測績效報告{version} (Strategy: Rank {rank_str} | Weights: {weights})")
        print("="*80)
        print(f"總報酬率: {total_return*100:.2f}%")
        print(f"勝率: {winrate:.2f}%")
        print(f"年化報酬: {annual_return*100:.2f}%")
        print(f'期望值: {(results_df['avg_return'].mean() * 100):.2f}%')
        print(f"最大回撤: {max_drawdown*100:.2f}%")
        print(f"最終資產: {int(capital):,}")
        print("="*80)
        
        # 下載 Benchmark
        print(f"📥 下載對照組 ({benchmark})...")
        benchmark_curve = None
        bench_total_return = 0.0
        bench_annual_return = 0.0

        try:
            start_d = results_df['date'].min()
            end_d = results_df['date'].max()
            benchmark_df = yf.download(benchmark, start=start_d, end=end_d, progress=False)
            
            if not benchmark_df.empty:
                if isinstance(benchmark_df.columns, pd.MultiIndex):
                    try:
                        close = benchmark_df['Close'][benchmark]
                    except KeyError:
                        close = benchmark_df.iloc[:, 0]
                else:
                    close = benchmark_df['Close']

                b_start = close.iloc[0]
                benchmark_curve = (close / b_start) * INITIAL_CAPITAL
                
                bench_total_return = (benchmark_curve.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
                bench_annual_return = ((1 + bench_total_return) ** (12 / len(results_df))) - 1
                
        except Exception as e:
            print(f"⚠️ 對照組下載失敗: {e}")

        # 繪圖
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        label_ai = f'Rank {rank_str} Strategy ({total_return*100:.0f}% | Year: {annual_return*100:.1f}%)'
        plt.plot(results_df['date'], equity_curve[1:], label=label_ai, color='red', linewidth=2)
        
        if benchmark_curve is not None:
            label_bench = f'{benchmark} ({bench_total_return*100:.0f}% | Year: {bench_annual_return*100:.1f}%)'
            plt.plot(benchmark_df.index, benchmark_curve, label=label_bench, color='gray', linestyle='--', alpha=0.7)
        
        plt.title(f'Equity Curve: Rank {rank_str} vs {benchmark}')
        plt.ylabel('Capital')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.fill_between(results_df['date'], drawdown[1:], 0, color='green', alpha=0.3)
        plt.title('Drawdown (Risk)')
        plt.ylabel('% Drawdown')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        print("分析圖表已產生...")
        plt.show()

        #====產生期望值分布
        test_df['rank'] = test_df.groupby('date')['predict_score'].rank(ascending=False, method='first').astype(int)
        
        stats_df = test_df[test_df['rank'].isin([c for c in range(1,11)])]
        
        # 1. 在這裡新增 max_return 與 min_return 的計算
        rank_stats = stats_df.groupby('rank')['return_20d'].agg(
            avg_return=lambda x: (x.mean() - TRANSACTION_COST) * 100,
            winrate=lambda x: ((x - TRANSACTION_COST) > 0).mean() * 100,
            max_return=lambda x: (x.max() - TRANSACTION_COST) * 100,
            min_return=lambda x: (x.min() - TRANSACTION_COST) * 100
        ).reset_index()

        plt.figure(figsize=(10, 5))
        bars = plt.bar(rank_stats['rank'], rank_stats['avg_return'], color='steelblue', alpha=0.8)
        plt.title('Average Return and Win Rate by Rank')
        plt.xlabel('Rank')
        plt.ylabel('Average Return (%)')
        plt.xticks(rank_stats['rank'])
        plt.grid(axis='y', alpha=0.3)

        # 2. 在迴圈中把 max_r 與 min_r 拉出來，並加進字串中 (字體稍微縮小為 9，避免太擠)
        for bar, w, r, max_r, min_r in zip(bars, rank_stats['winrate'], rank_stats['avg_return'], rank_stats['max_return'], rank_stats['min_return']):
            yval = bar.get_height()
            va_align = 'bottom' if yval >= 0 else 'top'
            
            plt.text(bar.get_x() + bar.get_width()/2, yval, 
                     f'{r:.2f}%\n(W:{w:.1f}%)\nMax:{max_r:.1f}%\nMin:{min_r:.1f}%', 
                     ha='center', va=va_align, fontsize=8)
                     
        plt.tight_layout()
        print("分析圖表已產生...")
        plt.show()

    else:
        print("無交易紀錄。")

    # 輸出日誌
    LOG_DIR = config.BACKTEST_TRADELOG_DIR
    csv_filename = LOG_DIR / f"trade_log_{version}_rank_{rank_str.replace(',', '_')}.csv"
    export_full_trade_log(
        test_df=test_df, 
        filename=csv_filename, 
        select_rank=select_rank, 
        weights=weights, 
        model=model
    )

    return results_df  # 回傳結果供後續分析

if __name__ == "__main__":
    
    backtester(None,version='v16', select_rank=[c for c in range(1,6)], benchmark='00631L.TW')#c for c in range(1,11)