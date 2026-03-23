import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from catboost import CatBoostRanker
import os
import yfinance as yf
import config

def export_full_trade_log(test_df, filename, select_rank, weights=None, model=None):
    import shap
    all_trade_records = []
    
    unique_dates = sorted(test_df['date'].unique())
    feature_names = model.feature_names_
    
    print(f"正在分析選股原因 (SHAP Analysis)...")
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(test_df[feature_names])

    for date in unique_dates:
        date_mask = test_df['date'] == date
        group = test_df[date_mask].copy()
        shap_group = shap_values_all[test_df[date_mask].index]
        
        top_picks = group[group['rank'].isin(select_rank)].copy()
        
        if top_picks.empty: continue

        if weights is None:
            monthly_avg = top_picks['return_20d'].mean() * 100
        else:
            current_weights = top_picks['rank'].map(weights)
            normalized_weights = current_weights / current_weights.sum()
            monthly_avg = np.dot(top_picks['return_20d'], normalized_weights) * 100

        for i, (idx_in_df, row) in enumerate(top_picks.iterrows()):
            abs_idx = test_df.index.get_loc(idx_in_df)
            stock_shap = shap_values_all[abs_idx]
            
            top_indices = np.argsort(stock_shap)[-6:][::-1]
            reasons = [f"{feature_names[j]}({stock_shap[j]:.2f})" for j in top_indices]
            reason_str = " | ".join(reasons)

            all_trade_records.append({
                '換股日期': date.date(),
                '當月組合平均報酬%': round(monthly_avg, 2),
                '排名': row['rank'],
                '股票代碼': row['id'],
                '股票名稱': row['name'],
                '預測分數': round(row['predict_score'], 4),
                '選股主因 (Feature:Contribution)': reason_str, 
                '該股當月報酬%': round(row['return_20d'] * 100, 2),
                '成交量/值(Log)': round(row['amount60'], 2) if 'amount60' in row else 'N/A',
                'MOM': round(row['mom'], 2) if 'mom' in row else 'N/A'
            })
            
    full_log_df = pd.DataFrame(all_trade_records)
    full_log_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"帶原因分析的日誌已存至: {filename}")

def backtester(version='v16', select_rank=[c for c in range(1,6)], benchmark='00631L.TW'):
    MODEL_FILE = config.CBM_DIR / f"strategy_{version}.cbm"
    DATA_FILE = config.FEATURES_DIR / f"train_{version}.csv"
    
    INITIAL_CAPITAL = 1000000
    TRANSACTION_COST = 0.005
    FIXED_STOP_LOSS = 0.99
    BALANCE = 0.25
    VOL_THRESHOLD = 0.30

    print(f"\n啟動回測系統 V16 | 策略: 月換股 + 日監控 (Vol & Balance) | 對照組: {benchmark}")

    # 1. 資料預處理
    df = pd.read_csv(DATA_FILE, dtype={'id': str})
    df['date'] = pd.to_datetime(df['date'])
    test_df = df[df['date'] >= sorted(df['date'].unique())[int(len(df['date'].unique()) * 0.8)]].copy()
    test_df = test_df.sort_values(['date', 'id']).reset_index(drop=True)

    # 2. 載入模型與評分
    model = CatBoostRanker()
    model.load_model(MODEL_FILE)
    test_df['predict_score'] = model.predict(test_df[model.feature_names_])
    test_df['rank'] = test_df.groupby('date')['predict_score'].rank(ascending=False, method='first').astype(int)

    # 3. 載入大盤波動率雷達 (0050)
    print("正在初始化大盤日監控雷達...")
    try:
        vol_df = pd.read_csv(config.DATABASE_DIR / "0050.csv")
        vol_df.columns = [c.lower() for c in vol_df.columns]
        p_col = 'price' if 'price' in vol_df.columns else 'close'
        vol_df['date'] = pd.to_datetime(vol_df['date']).dt.tz_localize(None)
        vol_df = vol_df.sort_values('date').set_index('date')
        vol_df['realized_vol'] = vol_df[p_col].pct_change().rolling(20).std() * (252 ** 0.5)
    except Exception as e:
        print(f"波動率模組失效: {e}")
        vol_df = pd.DataFrame()

    # 4. 載入 Market Data 執行日頻率回測
    market_df = pd.read_parquet(config.DATABASE_DIR / "market.parquet")
    market_df['date'] = pd.to_datetime(market_df['date'])
    market_df['id'] = market_df['id'].astype(str)
    market_df = market_df.sort_values(['id', 'date']).set_index(['id', 'date'])

    vol_cut_count = 0
    total_rebalance_count = 0
    monthly_results = []
    capital = INITIAL_CAPITAL
    equity_curve = [capital]

    print("進入完全體防禦迴圈 (日檢查模式)...")
    for date, group in test_df.groupby('date'):
        top_picks = group[group['rank'].isin(select_rank)]
        if top_picks.empty: continue
        
        current_month_capital = capital
        initial_per_stock = current_month_capital / len(top_picks)
        portfolio = {}
        has_scaled_down = False
        
        for idx in top_picks.index:
            s_id = str(test_df.loc[idx, 'id']).replace('.0', '')
            try:
                f_data = market_df.loc[s_id].loc[date:].head(20)
                if not f_data.empty:
                    portfolio[idx] = {
                        'entry_p': f_data.iloc[0]['close'],
                        'shares': initial_per_stock / f_data.iloc[0]['close'],
                        'dates': f_data.index.get_level_values('date'),
                        'prices': f_data['close'].values,
                        'cash_out': 0.0,
                        'alive': True
                    }
            except KeyError: continue

        for d_idx in range(20):
            day_total_val = 0.0
            stocks_to_check = []
            today_date = None
            
            for p in portfolio.values():
                if p['alive'] and d_idx < len(p['dates']):
                    today_date = p['dates'][d_idx]
                    break
            
            # 【日檢查 1】黑天鵝大盤波動偵測
            if today_date is not None and not has_scaled_down and not vol_df.empty:
                try:
                    current_vol = vol_df.loc[:today_date, 'realized_vol'].iloc[-1]
                    if current_vol > VOL_THRESHOLD:
                        has_scaled_down = True
                        vol_cut_count += 1
                        print(f"🚨 [VOL ALERT] {today_date.date()} 波動 {current_vol:.2f} 觸發砍半倉")
                        for idx, p in portfolio.items():
                            if p['alive'] and d_idx < len(p['prices']):
                                cur_p = p['prices'][d_idx]
                                sell_shares = p['shares'] * 0.5
                                p['cash_out'] += sell_shares * cur_p
                                p['shares'] -= sell_shares
                except: pass

            # 【日檢查 2】個股停損與淨值計算
            for idx, p in portfolio.items():
                if p['alive'] and d_idx < len(p['prices']):
                    cur_p = p['prices'][d_idx]
                    if cur_p <= p['entry_p'] * (1 - FIXED_STOP_LOSS):
                        p['cash_out'] += p['shares'] * cur_p
                        p['shares'] = 0
                        p['alive'] = False
                    else:
                        v = p['shares'] * cur_p
                        day_total_val += v
                        stocks_to_check.append((idx, v, cur_p))
            
            if day_total_val <= 0: continue
            
            # 【日檢查 3】25% 權重再平衡
            for idx, v, cur_p in stocks_to_check:
                if (v / day_total_val) > BALANCE:
                    take_cash = (v - 0.20 * day_total_val) / 0.8
                    portfolio[idx]['cash_out'] += take_cash
                    portfolio[idx]['shares'] -= (take_cash / cur_p)
                    day_total_val -= take_cash
                    total_rebalance_count += 1

        # 月底結算
        total_monthly_return = 0
        for idx, p in portfolio.items():
            if len(p.get('prices', [])) > 0:
                final_v = p['shares'] * p['prices'][-1]
                stock_ret = (final_v + p['cash_out']) / initial_per_stock - 1
                test_df.at[idx, 'return_20d'] = stock_ret
                total_monthly_return += (stock_ret / len(portfolio))

        capital += (capital * (total_monthly_return - TRANSACTION_COST))
        equity_curve.append(capital)
        monthly_results.append({'date': date, 'avg_return': total_monthly_return, 'capital': capital})  # ← 移到最後

    # 5. 迴圈結束後統一計算報表統計
    results_df = pd.DataFrame(monthly_results)
    total_months = len(results_df)
    rank_str = ','.join(map(str, select_rank))
    total_return = (capital / INITIAL_CAPITAL) - 1
    annual_return = (1 + total_return) ** (12 / max(total_months, 1)) - 1
    winrate = (results_df['avg_return'] > 0).mean() * 100
    eq = pd.Series(equity_curve)
    peak = eq.cummax()
    drawdown = ((eq - peak) / peak).tolist()
    max_drawdown = min(drawdown)

    print("\n" + "="*80)
    print(f"回測績效報告 {version} (Strategy: Rank {rank_str})")
    print("="*80)
    print(f"總報酬率: {total_return*100:.2f}%")
    print(f"勝率: {winrate:.2f}%")
    print(f"年化報酬: {annual_return*100:.2f}%")
    print(f"期望值: {(results_df['avg_return'].mean() * 100):.2f}%")
    print(f"月度最大回撤: {max_drawdown*100:.2f}%")
    print(f"最終資產: {int(capital):,}")
    print("-" * 80)
    print(f"系統換手追蹤 (2年期間):")
    print(f"總再平衡次數: {total_rebalance_count} 次")
    print(f"平均每月再平衡: {total_rebalance_count / total_months:.2f} 次")
    print(f"波動率砍倉統計:")
    print(f"觸發砍半倉次數: {vol_cut_count} 次 / 共 {len(results_df)} 個月")
    print(f"砍倉頻率: {vol_cut_count / len(results_df) * 100:.1f}%")
    print("="*80)

    print(f"下載對照組 ({benchmark})...")
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
        print(f"對照組下載失敗: {e}")

    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    label_ai = f'Rank {rank_str} Strategy ({total_return*100:.0f}% | Year: {annual_return*100:.1f}%)'
    plt.plot(results_df['date'], equity_curve[1:], label=label_ai, color='red', linewidth=2)
    
    if benchmark_curve is not None:
        label_bench = f'{benchmark} ({bench_total_return*100:.0f}% | Year: {bench_annual_return*100:.1f}%)'
        plt.plot(benchmark_df.index, benchmark_curve, label=label_bench, color='gray', linestyle='--', alpha=0.7)
    
    plt.title(f'Equity Curve (30% Rebalancing + {FIXED_STOP_LOSS*100}% Fixed Stop): Rank {rank_str} vs {benchmark}')
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

    stats_df = test_df[test_df['rank'].isin([c for c in range(1, 11)])]
    rank_stats = stats_df.groupby('rank')['return_20d'].agg(
        avg_return=lambda x: (x.mean() - TRANSACTION_COST) * 100,
        winrate=lambda x: ((x - TRANSACTION_COST) > 0).mean() * 100,
        max_return=lambda x: (x.max() - TRANSACTION_COST) * 100,
        min_return=lambda x: (x.min() - TRANSACTION_COST) * 100
    ).reset_index()

    plt.figure(figsize=(10, 5))
    bars = plt.bar(rank_stats['rank'], rank_stats['avg_return'], color='steelblue', alpha=0.8)
    plt.title('Average Return and Win Rate by Rank (Hybrid Defense)')
    plt.xlabel('Rank')
    plt.ylabel('Average Return (%)')
    plt.xticks(rank_stats['rank'])
    plt.grid(axis='y', alpha=0.3)

    for bar, w, r, max_r, min_r in zip(bars, rank_stats['winrate'], rank_stats['avg_return'], rank_stats['max_return'], rank_stats['min_return']):
        yval = bar.get_height()
        va_align = 'bottom' if yval >= 0 else 'top'
        plt.text(bar.get_x() + bar.get_width()/2, yval,
                 f'{r:.2f}%\n(W:{w:.1f}%)\nMax:{max_r:.1f}%\nMin:{min_r:.1f}%',
                 ha='center', va=va_align, fontsize=8)

    plt.tight_layout()
    print("期望值分布圖表已產生...")
    plt.show()

    LOG_DIR = config.BACKTEST_TRADELOG_DIR
    csv_filename = LOG_DIR / f"trade_log_{version}_rank_{rank_str.replace(',', '_')}.csv"
    export_full_trade_log(
        test_df=test_df,
        filename=csv_filename,
        select_rank=select_rank,
        model=model
    )

    return results_df

if __name__ == "__main__":
    backtester(version='v16', select_rank=[c for c in range(1,6)], benchmark='00631L.TW')