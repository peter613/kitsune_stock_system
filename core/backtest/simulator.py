"""
Backtester V2 — 單位淨值法 (Unit NAV Method)
=============================================
修正項目：
  BUG1: cash_out 現金完全納入每日總資產，不再遺漏
  BUG2: 月報酬改為「期末總淨值 / 期初總資產」直接計算，不用平均報酬率近似
  BUG3: 再平衡賣出現金保留在投組現金池，當日仍計入總資產
  BUG4: test_df['return_20d'] 只覆寫被選股票，rank_stats 分兩組計算
  WARN1: 每次實際交易（停損、砍倉、再平衡）皆扣手續費
  WARN2: 砍倉後的閒置現金以「現金池」方式每日計入總資產
  WARN3: prices[-1] 邊界確認與 return_20d 標籤窗口說明一致

核心概念：單位淨值法
  - 初始發行 1,000,000 單位，單位淨值 = 1.0
  - 每個月底：新單位淨值 = 當月期末總資產（含現金池）/ 期初總資產
  - 累積淨值 = 所有月份單位淨值連乘，等同基金 NAV 走勢
  - 報酬計算完全基於資產流量，消除「平均報酬套全倉」的近似誤差
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
from catboost import CatBoostRanker
import os
import yfinance as yf
import config


# ── 中文字體自動設定（解決 CJK glyph 警告）──────────────
def _set_cjk_font():
    candidates = [
        'Microsoft JhengHei',  # Windows 正黑體（繁體台灣）
        'Microsoft YaHei',     # Windows 微軟雅黑
        'PingFang TC',         # macOS 繁體
        'PingFang SC',         # macOS 簡體
        'Noto Sans CJK TC',    # Linux
        'Noto Sans TC',
        'WenQuanYi Micro Hei',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False

_set_cjk_font()


# ──────────────────────────────────────────────
# 輔助函式
# ──────────────────────────────────────────────

def _calc_tx_cost(value: float, cost_rate: float) -> float:
    """計算單筆交易手續費（買或賣皆適用）"""
    return value * cost_rate


def export_full_trade_log(test_df, filename, select_rank, model=None):
    import shap
    all_trade_records = []

    unique_dates = sorted(test_df['date'].unique())
    feature_names = model.feature_names_

    print("正在分析選股原因 (SHAP Analysis)...")
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(test_df[feature_names])

    for date in unique_dates:
        date_mask = test_df['date'] == date
        group = test_df[date_mask].copy()

        top_picks = group[group['rank'].isin(select_rank)].copy()
        if top_picks.empty:
            continue

        # 使用回測後已覆寫的 simulated_return（若存在），否則用原始 return_20d
        ret_col = 'simulated_return' if 'simulated_return' in top_picks.columns else 'return_20d'
        monthly_avg = top_picks[ret_col].mean() * 100

        for idx_in_df, row in top_picks.iterrows():
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
                '該股當月報酬%': round(row[ret_col] * 100, 2),
                '成交量/值(Log)': round(row['amount60'], 2) if 'amount60' in row else 'N/A',
                'MOM': round(row['mom'], 2) if 'mom' in row else 'N/A',
            })

    full_log_df = pd.DataFrame(all_trade_records)
    full_log_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"帶原因分析的日誌已存至: {filename}")


# ──────────────────────────────────────────────
# 主回測函式
# ──────────────────────────────────────────────

def backtester(version='v16', select_rank=None, benchmark='00631L.TW'):
    if select_rank is None:
        select_rank = list(range(1, 6))

    MODEL_FILE   = config.CBM_DIR      / f"strategy_{version}.cbm"
    DATA_FILE    = config.FEATURES_DIR / f"train_{version}.csv"

    INITIAL_CAPITAL  = 1_000_000
    TRANSACTION_COST = 0.005   # 0.5% per trade（買進 + 賣出各算一次）
    FIXED_STOP_LOSS  = 0.20    # 固定停損：入場價跌 20%
    BALANCE_HIGH     = 0.25    # 再平衡上限：單股超過 25% 時修剪
    BALANCE_TARGET   = 0.20    # 修剪至 20%
    VOL_THRESHOLD    = 0.99    # 大盤波動率超過 30% 時砍半倉

    rank_str = ','.join(map(str, select_rank))
    print(f"\n啟動回測系統 V2 (單位淨值法) | 策略: Rank {rank_str} | 對照組: {benchmark}")

    # ── 1. 資料預處理 ──────────────────────────────────
    df = pd.read_csv(DATA_FILE, dtype={'id': str})
    df['date'] = pd.to_datetime(df['date'])
    cutoff_idx = int(len(df['date'].unique()) * 0.8)
    cutoff_date = sorted(df['date'].unique())[cutoff_idx]
    test_df = df[df['date'] >= cutoff_date].copy()
    test_df = test_df.sort_values(['date', 'id']).reset_index(drop=True)

    # ── 2. 模型評分 ──────────────────────────────────
    model = CatBoostRanker()
    model.load_model(MODEL_FILE)
    test_df['predict_score'] = model.predict(test_df[model.feature_names_])
    test_df['rank'] = (
        test_df.groupby('date')['predict_score']
        .rank(ascending=False, method='first')
        .astype(int)
    )
    # 記錄原始 return_20d（標籤），回測結果另存 simulated_return
    test_df['simulated_return'] = np.nan

    # ── 3. 大盤波動率序列 ──────────────────────────────
    print("初始化大盤波動率雷達 (0050.TW)...")
    vol_series = pd.Series(dtype=float)
    try:
        raw = yf.download('0050.TW', start='2020-01-01', progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw['Close'].iloc[:, 0]
        else:
            close = raw['Close']
        close.index = pd.to_datetime(close.index).tz_localize(None)
        vol_series = close.pct_change().rolling(20).std() * (252 ** 0.5)
        vol_series = vol_series.dropna()
    except Exception as e:
        print(f"波動率模組失效: {e}")

    # ── 4. 市場日頻資料 ──────────────────────────────
    market_df = pd.read_parquet(config.DATABASE_DIR / "market.parquet")
    market_df['date'] = pd.to_datetime(market_df['date'])
    market_df['id']   = market_df['id'].astype(str)
    market_df = market_df.sort_values(['id', 'date']).set_index(['id', 'date'])

    # ── 5. 單位淨值法初始化 ──────────────────────────
    # NAV = 每單位淨值，初始為 1.0
    # capital = 總資產（隨 NAV 等比縮放）
    nav            = 1.0           # 累積單位淨值
    capital        = float(INITIAL_CAPITAL)
    nav_curve      = [nav]         # 每月月底 NAV
    equity_curve   = [capital]     # 每月月底總資產（用於繪圖）
    monthly_results = []

    vol_cut_count       = 0
    total_rebalance_cnt = 0

    print("進入單位淨值法回測迴圈 (日監控模式)...")

    for date, group in test_df.groupby('date'):
        top_picks = group[group['rank'].isin(select_rank)]
        if top_picks.empty:
            continue

        n_stocks = len(top_picks)

        # ── 月初：買進，扣入場手續費 ──────────────────
        # 每檔分配等額資金；買進成本扣在分配資金內
        alloc_per_stock = capital / n_stocks
        buy_cost_total  = _calc_tx_cost(capital, TRANSACTION_COST)  # 全倉買入成本
        # 扣除買入成本後才是實際投入各股的資金
        net_alloc_per_stock = (capital - buy_cost_total) / n_stocks

        # 現金池：用來放停損/砍倉/再平衡回收的現金
        # 月初現金池 = 0（全倉投入）
        cash_pool = 0.0

        portfolio = {}
        has_scaled_down = False

        for idx in top_picks.index:
            s_id = str(test_df.loc[idx, 'id']).replace('.0', '')
            try:
                f_data = market_df.loc[s_id].loc[date:].head(20)
                if not f_data.empty:
                    entry_p = f_data.iloc[0]['close']
                    portfolio[idx] = {
                        's_id'      : s_id,
                        'entry_p'   : entry_p,
                        'shares'    : net_alloc_per_stock / entry_p,
                        'dates'     : f_data.index.get_level_values('date'),
                        'prices'    : f_data['close'].values,
                        'alive'     : True,
                        # 追蹤該股自己回收的現金（停損/砍倉/再平衡）
                        # 用於計算個股 simulated_return
                        'own_cash'  : 0.0,
                    }
            except KeyError:
                cash_pool += net_alloc_per_stock

        if not portfolio:
            continue

        # ── 日內監控迴圈 ───────────────────────────────
        for d_idx in range(20):

            today_date = None
            for p in portfolio.values():
                if p['alive'] and d_idx < len(p['dates']):
                    today_date = p['dates'][d_idx]
                    break

            # 【日檢查 1】大盤波動砍半倉
            if (today_date is not None
                    and not has_scaled_down
                    and len(vol_series) > 0):
                check_ts = pd.to_datetime(today_date).tz_localize(None)
                try:
                    cur_vol = vol_series.asof(check_ts)
                    if pd.notna(cur_vol) and cur_vol > VOL_THRESHOLD:
                        has_scaled_down = True
                        vol_cut_count += 1
                        print(f"  [VOL ALERT] {check_ts.date()} 波動 {cur_vol:.2f} → 砍半倉")
                        for _idx, p in portfolio.items():
                            if p['alive'] and d_idx < len(p['prices']):
                                cur_p      = p['prices'][d_idx]
                                sell_sh    = p['shares'] * 0.5
                                sell_val   = sell_sh * cur_p
                                fee        = _calc_tx_cost(sell_val, TRANSACTION_COST)
                                net_proc   = sell_val - fee
                                cash_pool     += net_proc
                                p['own_cash'] += net_proc   # 各股砍倉現金各自歸屬
                                p['shares']   -= sell_sh
                except Exception as e:
                    print(f"  波動率砍倉異常: {e}")

            # 【日檢查 2】個股停損
            for idx, p in portfolio.items():
                if p['alive'] and d_idx < len(p['prices']):
                    cur_p = p['prices'][d_idx]
                    if cur_p <= p['entry_p'] * (1 - FIXED_STOP_LOSS):
                        sell_val      = p['shares'] * cur_p
                        fee           = _calc_tx_cost(sell_val, TRANSACTION_COST)
                        net_proceeds  = sell_val - fee
                        cash_pool    += net_proceeds
                        p['own_cash'] += net_proceeds   # 記錄屬於該股的回收現金
                        p['shares']   = 0.0
                        p['alive']    = False

            # 【日檢查 3】計算活倉總值（不含現金池）
            stock_vals = {}
            day_stock_total = 0.0
            for idx, p in portfolio.items():
                if p['alive'] and d_idx < len(p['prices']):
                    v = p['shares'] * p['prices'][d_idx]
                    stock_vals[idx] = v
                    day_stock_total += v

            if day_stock_total <= 0:
                continue

            # 【日檢查 4】再平衡：單股超過 BALANCE_HIGH
            # 現金池也納入總資產計算（現金不參與權重再平衡）
            day_total_assets = day_stock_total + cash_pool

            for idx, v in stock_vals.items():
                weight = v / day_total_assets
                if weight > BALANCE_HIGH:
                    target_v   = BALANCE_TARGET * day_total_assets
                    trim_val   = v - target_v
                    fee        = _calc_tx_cost(trim_val, TRANSACTION_COST)
                    net_trim   = trim_val - fee
                    cash_pool += net_trim
                    portfolio[idx]['own_cash'] += net_trim   # 再平衡現金歸屬該股
                    portfolio[idx]['shares']   -= trim_val / portfolio[idx]['prices'][d_idx]
                    total_rebalance_cnt += 1

        # ── 月底結算：單位淨值法 ──────────────────────
        # 期末總資產 = 所有存活股票市值 + 現金池
        end_stock_val = 0.0
        for idx, p in portfolio.items():
            if len(p.get('prices', [])) > 0:
                end_stock_val += p['shares'] * p['prices'][-1]

        # 月底賣出手續費（整批賣出）
        sell_cost = _calc_tx_cost(end_stock_val, TRANSACTION_COST)
        end_total = end_stock_val - sell_cost + cash_pool

        # 本月報酬率（淨值增長率）
        month_return = (end_total / capital) - 1.0

        # 覆寫 simulated_return（僅被選中的股票）
        # 分母：alloc_per_stock（月初原始等額分配，未扣費）→ 與原始 return_20d 基準一致
        # 分子：月底持股市值（扣賣出費）+ 該股自己回收的現金(own_cash)
        for idx, p in portfolio.items():
            if len(p.get('prices', [])) > 0:
                final_mkt   = p['shares'] * p['prices'][-1]
                sell_fee    = _calc_tx_cost(final_mkt, TRANSACTION_COST)
                # own_cash 已在回收時扣過費，不重複扣
                total_recv  = (final_mkt - sell_fee) + p['own_cash']
                sim_ret     = total_recv / alloc_per_stock - 1.0
                test_df.at[idx, 'simulated_return'] = sim_ret

        # 更新 NAV 與資本
        nav    = nav * (1 + month_return)
        capital = end_total
        nav_curve.append(nav)
        equity_curve.append(capital)

        monthly_results.append({
            'date'        : date,
            'month_return': month_return,
            'nav'         : nav,
            'capital'     : capital,
        })

    # ──────────────────────────────────────────────
    # 6. 績效報表
    # ──────────────────────────────────────────────
    results_df  = pd.DataFrame(monthly_results)
    total_months = len(results_df)
    total_return = nav - 1.0
    annual_return = (nav ** (12 / max(total_months, 1))) - 1.0
    winrate = (results_df['month_return'] > 0).mean() * 100
    expected_val = results_df['month_return'].mean() * 100

    nav_series = pd.Series(nav_curve)
    peak       = nav_series.cummax()
    drawdown   = ((nav_series - peak) / peak).tolist()
    max_dd     = min(drawdown)

    print("\n" + "=" * 80)
    print(f"回測績效報告 {version}  (Strategy: Rank {rank_str})  [單位淨值法]")
    print("=" * 80)
    print(f"初始資金:     {INITIAL_CAPITAL:>12,.0f}")
    print(f"最終資產:     {int(capital):>12,}")
    print(f"期末單位淨值: {nav:>12.4f}")
    print(f"總報酬率:     {total_return*100:>10.2f}%")
    print(f"年化報酬:     {annual_return*100:>10.2f}%")
    print(f"月均期望值:   {expected_val:>10.2f}%")
    print(f"勝率:         {winrate:>10.2f}%")
    print(f"最大回撤:     {max_dd*100:>10.2f}%")
    print("-" * 80)
    print(f"月數:              {total_months}")
    print(f"總再平衡次數:      {total_rebalance_cnt}")
    print(f"平均每月再平衡:    {total_rebalance_cnt / max(total_months,1):.2f} 次")
    print(f"波動砍倉觸發:      {vol_cut_count} / {total_months} 個月  "
          f"({vol_cut_count/max(total_months,1)*100:.1f}%)")
    print("=" * 80)

    # ──────────────────────────────────────────────
    # 7. 對照組
    # ──────────────────────────────────────────────
    print(f"下載對照組 ({benchmark})...")
    benchmark_nav   = None
    bench_total_ret = 0.0
    bench_annual_ret = 0.0

    try:
        start_d = results_df['date'].min()
        end_d   = results_df['date'].max()
        bdf = yf.download(benchmark, start=start_d, end=end_d, progress=False)
        if not bdf.empty:
            if isinstance(bdf.columns, pd.MultiIndex):
                close = bdf['Close'].iloc[:, 0]
            else:
                close = bdf['Close']
            # 對照組也以 NAV 方式呈現（初始 = 1.0）
            benchmark_nav   = close / close.iloc[0]
            bench_total_ret = benchmark_nav.iloc[-1] - 1.0
            bench_annual_ret = (benchmark_nav.iloc[-1] ** (12 / total_months)) - 1.0
    except Exception as e:
        print(f"對照組下載失敗: {e}")

    # ──────────────────────────────────────────────
    # 8+9. 所有圖表一次畫完，統一 plt.show()
    #      避免多次 show() 造成 tkinter mainloop 衝突
    # ──────────────────────────────────────────────

    # 先把 rank_stats 算好（純資料，不依賴圖表）
    stats_df = test_df[test_df['rank'].isin(range(1, 11))].copy()
    stats_df['ret_for_stat'] = stats_df['simulated_return'].fillna(stats_df['return_20d'])

    rank_stats = (
        stats_df.groupby('rank')['ret_for_stat']
        .agg(
            avg_return = lambda x: (x.mean() - TRANSACTION_COST) * 100,
            winrate    = lambda x: ((x - TRANSACTION_COST) > 0).mean() * 100,
            max_return = lambda x: (x.max()  - TRANSACTION_COST) * 100,
            min_return = lambda x: (x.min()  - TRANSACTION_COST) * 100,
        )
        .reset_index()
    )

    # 三個子圖合成一張 figure（高度比 3:1:3）
    fig = plt.figure(figsize=(14, 14))
    gs  = fig.add_gridspec(3, 1, height_ratios=[3, 1, 3], hspace=0.4)

    # 子圖 1：NAV 走勢
    ax1 = fig.add_subplot(gs[0])
    label_ai = (f'Rank {rank_str}  |  NAV {nav:.3f}  '
                f'({total_return*100:.0f}%  /  Annual {annual_return*100:.1f}%)')
    ax1.plot(results_df['date'], nav_curve[1:],
             label=label_ai, color='crimson', linewidth=2)

    if benchmark_nav is not None:
        label_b = (f'{benchmark}  |  NAV {benchmark_nav.iloc[-1]:.3f}  '
                   f'({bench_total_ret*100:.0f}%  /  Annual {bench_annual_ret*100:.1f}%)')
        ax1.plot(bdf.index, benchmark_nav,
                 label=label_b, color='steelblue', linestyle='--', alpha=0.75)

    ax1.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
    ax1.set_title(f'Cumulative NAV  —  Rank {rank_str} vs {benchmark}')
    ax1.set_ylabel('NAV (base=1.0)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 子圖 2：回撤
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(results_df['date'], drawdown[1:], 0,
                     color='tomato', alpha=0.35, label='Drawdown')
    ax2.set_title('Drawdown from Peak NAV')
    ax2.set_ylabel('Drawdown')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    # 子圖 3：Rank 期望值長條圖
    ax3 = fig.add_subplot(gs[2])
    colors = ['crimson' if r in select_rank else 'steelblue'
              for r in rank_stats['rank']]
    bars = ax3.bar(rank_stats['rank'], rank_stats['avg_return'],
                   color=colors, alpha=0.8)

    ax3.set_title(f'Avg Return & Win Rate by Rank  (red = selected ranks: {rank_str})')
    ax3.set_xlabel('Rank')
    ax3.set_ylabel('Avg Return (%)')
    ax3.set_xticks(rank_stats['rank'])
    ax3.axhline(0, color='gray', linewidth=0.8)
    ax3.grid(axis='y', alpha=0.3)

    for bar, row in zip(bars, rank_stats.itertuples()):
        yval   = bar.get_height()
        va     = 'bottom' if yval >= 0 else 'top'
        offset = 0.08 if yval >= 0 else -0.08
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            yval + offset,
            f'{row.avg_return:.2f}%\n(W:{row.winrate:.1f}%)\n'
            f'Max:{row.max_return:.1f}%\nMin:{row.min_return:.1f}%',
            ha='center', va=va, fontsize=8,
        )

    print("圖表產生完畢，關閉視窗後程式繼續執行...")
    plt.show()   # ← 唯一一次 show，避免 tkinter mainloop 衝突

    # ──────────────────────────────────────────────
    # 10. 匯出交易日誌
    # ──────────────────────────────────────────────
    LOG_DIR      = config.BACKTEST_TRADELOG_DIR
    csv_filename = LOG_DIR / f"trade_log_{version}_rank_{rank_str.replace(',', '_')}.csv"
    export_full_trade_log(
        test_df    = test_df,
        filename   = csv_filename,
        select_rank= select_rank,
        model      = model,
    )

    return results_df, nav_curve


if __name__ == "__main__":
    backtester(version='v16', select_rank=list(range(1, 6)), benchmark='00631L.TW')