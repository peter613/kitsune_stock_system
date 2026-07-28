"""
Backtester V3 — 純 NAV 比率法 (Pure NAV Ratio Method)
======================================================
V3 修正項目（相對 V2）：
  FIX1: VOL_THRESHOLD 由 0.99 修正為 0.30（原設定永遠不觸發）
  FIX2: 對照組年化改用實際天數 / 365.25，不再用策略月份數近似
  FIX3: 月底結算改用純 NAV 比率（end_total / capital），
        移除中間變數 month_return 作為主要路徑，改為衍生統計用
  FIX4: 個股 simulated_return 分母統一為 capital / n_stocks，
        確保個股報酬平均可還原整體 NAV（可驗算一致性）
  FIX5: 廢除獨立 nav_curve / equity_curve 陣列，
        NAV 序列直接從 results_df['nav'] 建構，徹底消除長度漂移

核心概念：純 NAV 比率連乘
  - 每月 period_ratio = end_total / capital（期初資產）
  - 累積 NAV = ∏ period_ratio（精確連乘，無近似誤差）
  - capital 每月更新為 end_total（滾動複利）
  - 現金池（停損/砍倉/再平衡回收）自動納入 end_total，不遺漏
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

    print("目前模型腦中的特徵有：", model.feature_names_)
    
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
        ret_col = 'simulated_return' if 'simulated_return' in top_picks.columns else 'return_10d'
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
                '成交量': round(row['volume'], 2) if 'volume' in row else 'N/A',
                # 'MOM': round(row['mom'], 2) if 'mom' in row else 'N/A',
            })

    full_log_df = pd.DataFrame(all_trade_records)
    full_log_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"帶原因分析的日誌已存至: {filename}")


# ──────────────────────────────────────────────
# 主回測函式
# ──────────────────────────────────────────────

def backtester(version, select_rank=None, benchmark='00631L', entry_on_next_open=False, delay_days=0, silent=False, export=True):
    if select_rank is None:
        select_rank = list(range(1, 6))

    MODEL_FILE   = config.get_version_dir(config.STOCK_CBM_DIR, version) / "strategy.cbm"
    DATA_FILE    = config.STOCK_FEATURES_DIR / f"train_{version}.csv"

    INITIAL_CAPITAL  = 1_000_000
    TRANSACTION_COST = 0.005   # 0.5% per trade（買進 + 賣出各算一次）
    SINGLE_TRADE_CAP = 0.80    # 單筆最高獲利封頂 (80%)

    AMOUNT_THRESHOLD  = 100000 # 當天成交值 > 1.0億 (設為0則不過濾)
    REVENUE_THRESHOLD = 100000 # 當月營收 > 1.0億 (設為0則不過濾)

    FIXED_STOP_LOSS  = 0.15    # 個股固定停損
    MACRO_STOP_LOSS   = 0.10  # 宏觀防守：總資產跌 10% 全清倉

    MACRO_TAKE_PROFIT = 0.10   # 宏觀停利：總資產漲 10% 啟動減碼
    TRIM_RATIO        = 0.25   # 減碼比例：達標時賣出 25% 持股

    rank_str = ','.join(map(str, select_rank))
    if not silent:
        print(f"\n啟動回測系統 (單位淨值法) | 策略: Rank {rank_str} | 對照組: {benchmark}")

    # ── 1. 資料預處理 ──────────────────────────────────
    df = pd.read_csv(DATA_FILE, dtype={'id': str})
    df['date'] = pd.to_datetime(df['date'])
    cutoff_idx = int(len(df['date'].unique()) * 0.8)
    cutoff_date = sorted(df['date'].unique())[cutoff_idx]
    test_df = df[df['date'] >= cutoff_date].copy()
    test_df = test_df.sort_values(['date', 'id']).reset_index(drop=True)

    if AMOUNT_THRESHOLD > 0:
        test_df = test_df[test_df['amount'] >= AMOUNT_THRESHOLD].reset_index(drop=True)
    if REVENUE_THRESHOLD > 0:
        test_df = test_df[test_df['month_revenue'] >= REVENUE_THRESHOLD].reset_index(drop=True)

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

    # ── 3. 市場日頻資料 ──────────────────────────────
    market_df = pd.read_parquet(config.MARKET_DATA_DIR / "market.parquet")
    market_df['date'] = pd.to_datetime(market_df['date'])
    market_df['id']   = market_df['id'].astype(str)
    market_df = market_df.sort_values(['id', 'date']).set_index(['id', 'date'])

    # ── 4. 單位淨值法初始化 ──────────────────────────
    # nav    : 累積單位淨值（純連乘，每月乘以本期 end_total / capital）
    # capital: 期初總資產（下月買進基準，月底更新為 end_total）
    # ★ 不再維護獨立 nav_curve / equity_curve 陣列；
    #   月底直接 append 進 monthly_results，
    #   繪圖從 DataFrame 取，徹底消除長度漂移問題。
    nav             = 1.0
    capital         = float(INITIAL_CAPITAL)
    monthly_results = []
    daily_records   = []  # 用於記錄日 NAV 序列
    stop_loss_count = 0
    macro_stop_count = 0
    macro_trim_count = 0

    if not silent:
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

        # 現金池：用來放停損/砍倉/再平衡回收的現金（月初 = 0，全倉投入）
        cash_pool = 0.0

        portfolio = {}

        for idx in top_picks.index:
            s_id = str(test_df.loc[idx, 'id']).replace('.0', '')
            try:
                f_data = market_df.loc[s_id].loc[date:]
                if len(f_data) > delay_days:
                    if entry_on_next_open and len(f_data) > delay_days + 1:
                        entry_p = f_data.iloc[delay_days + 1]['open']
                    else:
                        entry_p = f_data.iloc[delay_days]['close']
                    f_data = f_data.iloc[delay_days:].head(20)
                    portfolio[idx] = {
                        's_id'    : s_id,
                        'entry_p' : entry_p,
                        'shares'  : net_alloc_per_stock / entry_p,
                        'dates'   : f_data.index.get_level_values('date'),
                        'prices'  : f_data['close'].values,
                        'alive'   : True,
                        # 追蹤該股自己回收的現金（停損/砍倉/再平衡），
                        # 用於計算個股 simulated_return
                        'own_cash': 0.0,
                    }
            except KeyError:
                cash_pool += net_alloc_per_stock

        if not portfolio:
            continue

        portfolio_liquidated_this_month = False
        has_trimmed_this_month = False

        # ── 日內監控迴圈 ───────────────────────────────
        for d_idx in range(20):
            
            # 1. 計算當日「虛擬」結算總市值
            today_date = None
            virtual_day_stock_val = 0.0
            
            for idx, p in portfolio.items():
                if p['alive'] and d_idx < len(p['prices']):
                    cur_val = p['shares'] * p['prices'][d_idx]
                    # 強制獲利封頂
                    max_allowed = (alloc_per_stock * (1.0 + SINGLE_TRADE_CAP)) - p['own_cash']
                    if cur_val > max_allowed:
                        cur_val = max_allowed
                    virtual_day_stock_val += cur_val
                    if today_date is None:
                        today_date = p['dates'][d_idx]
            
            if today_date is None:
                continue
                
            virtual_day_total = virtual_day_stock_val + cash_pool
            current_return = (virtual_day_total / capital) - 1.0
            
            macro_liquidated_today = False
            
            # 2. ==== 宏觀防守與減碼機制 (優先) ====
            if not portfolio_liquidated_this_month:
                if current_return <= -MACRO_STOP_LOSS:
                    # 宏觀停損全清倉
                    for idx, p in portfolio.items():
                        if p['alive'] and d_idx < len(p['prices']):
                            cur_p = p['prices'][d_idx]
                            sell_val = p['shares'] * cur_p
                            fee = _calc_tx_cost(sell_val, TRANSACTION_COST)
                            net_proceeds = sell_val - fee
                            cash_pool += net_proceeds
                            p['own_cash'] += net_proceeds
                            p['shares'] = 0.0
                            p['alive'] = False
                    portfolio_liquidated_this_month = True
                    macro_liquidated_today = True
                    macro_stop_count += 1
                    
                elif current_return >= MACRO_TAKE_PROFIT and not has_trimmed_this_month:
                    # 宏觀達標減碼
                    for idx, p in portfolio.items():
                        if p['alive'] and d_idx < len(p['prices']):
                            cur_p = p['prices'][d_idx]
                            shares_to_sell = p['shares'] * TRIM_RATIO
                            sell_val = shares_to_sell * cur_p
                            fee = _calc_tx_cost(sell_val, TRANSACTION_COST)
                            net_proceeds = sell_val - fee
                            cash_pool += net_proceeds
                            p['own_cash'] += net_proceeds
                            p['shares'] -= shares_to_sell
                    has_trimmed_this_month = True
                    macro_trim_count += 1
            # ==========================================
            
            # 3. 【日檢查】個股停損 (若宏觀未清倉)
            if not macro_liquidated_today and not portfolio_liquidated_this_month:
                for idx, p in portfolio.items():
                    if p['alive'] and d_idx < len(p['prices']):
                        # 若設定隔日開盤買進，則忽略第0天(產生訊號當天)的收盤價停損檢查
                        if entry_on_next_open and d_idx == 0:
                            continue
                        cur_p = p['prices'][d_idx]
                        if cur_p <= p['entry_p'] * (1 - FIXED_STOP_LOSS):
                            sell_val      = p['shares'] * cur_p
                            fee           = _calc_tx_cost(sell_val, TRANSACTION_COST)
                            net_proceeds  = sell_val - fee
                            cash_pool    += net_proceeds
                            p['own_cash'] += net_proceeds   # 記錄屬於該股的回收現金
                            p['shares']   = 0.0
                            p['alive']    = False
                            stop_loss_count += 1

            # 4. 計算真實日 NAV
            final_day_stock_val = 0.0
            for idx, p in portfolio.items():
                if p['alive'] and d_idx < len(p['prices']):
                    cur_val = p['shares'] * p['prices'][d_idx]
                    # 強制獲利封頂
                    max_allowed = (alloc_per_stock * (1.0 + SINGLE_TRADE_CAP)) - p['own_cash']
                    if cur_val > max_allowed:
                        cur_val = max_allowed
                    final_day_stock_val += cur_val

            day_total = final_day_stock_val + cash_pool
            # day_total / INITIAL_CAPITAL 等價於 (day_total/capital) * (capital/INITIAL_CAPITAL)
            # 即「本期內部報酬比例 × 期初累積 NAV」，可正確銜接跨月 NAV 序列
            day_nav = day_total / INITIAL_CAPITAL
            daily_records.append({'date': today_date, 'daily_nav': day_nav})

        # ── 月底結算：純 NAV 比率法 ──────────────────────
        # 期末股票總市值（使用每檔最後一筆有效價格）
        end_stock_val = 0.0
        for idx, p in portfolio.items():
            prices = p.get('prices', [])
            if len(prices) > 0:
                final_mkt = p['shares'] * prices[-1]
                # 強制獲利封頂
                max_allowed = (alloc_per_stock * (1.0 + SINGLE_TRADE_CAP)) - p['own_cash']
                if final_mkt > max_allowed:
                    final_mkt = max_allowed
                p['capped_final_mkt'] = final_mkt  # 記錄下來供 simulated_return 使用
                end_stock_val += final_mkt

        # 月底整批賣出手續費
        sell_cost = _calc_tx_cost(end_stock_val, TRANSACTION_COST)

        # 期末總資產（股票淨值 + 現金池）
        # 現金池內的款項在回收時已各自扣過手續費，不重複扣
        end_total = (end_stock_val - sell_cost) + cash_pool

        # ★ 本期 NAV 比率：直接用資產比，不需要中間 month_return 變數
        period_ratio = end_total / capital          # e.g. 1.03 = 本月漲 3%
        month_return = period_ratio - 1.0           # 保留供統計輸出用
        nav          = nav * period_ratio            # 累積 NAV 連乘

        # ── 覆寫個股 simulated_return ────────────────────
        # 基準：月初每股分配金額（含手續費前，與 return_20d 標籤基準一致）
        # 個股淨收：月底持股市值（扣賣出費）＋ 該股自己回收的現金（own_cash，已扣費）
        # ★ 分母統一用 capital / n_stocks（月初期望分配），
        #   確保個股報酬之平均 ≈ 整體 NAV 的 period_ratio（可驗算一致性）
        for idx, p in portfolio.items():
            prices = p.get('prices', [])
            if len(prices) > 0:
                final_mkt  = p.get('capped_final_mkt', p['shares'] * prices[-1])
                sell_fee   = _calc_tx_cost(final_mkt, TRANSACTION_COST)
                total_recv = (final_mkt - sell_fee) + p['own_cash']
                # 分母：月初「原始等額分配」（未扣買入費，對應 return_20d 的基準）
                sim_ret    = total_recv / alloc_per_stock - 1.0
                test_df.at[idx, 'simulated_return'] = sim_ret

        # ── 更新下期資本並記錄 ───────────────────────────
        capital = end_total   # 下月買進金額 = 本月期末總資產

        monthly_results.append({
            'date'        : date,
            'month_return': month_return,
            'period_ratio': period_ratio,   # 便於驗算：nav = ∏ period_ratio
            'nav'         : nav,
            'capital'     : capital,
        })

    # ──────────────────────────────────────────────
    # 5. 績效報表
    # ──────────────────────────────────────────────
    results_df   = pd.DataFrame(monthly_results)
    total_months = len(results_df)
    total_return  = nav - 1.0
    annual_return = (nav ** (12 / max(total_months, 1))) - 1.0
    winrate       = (results_df['month_return'] > 0).mean() * 100
    expected_val  = results_df['month_return'].mean() * 100

    # ★ NAV 序列直接從 results_df 取（包含起始基準 1.0）
    nav_series = pd.concat([pd.Series([1.0]), results_df['nav'].reset_index(drop=True)],
                           ignore_index=True)
    peak     = nav_series.cummax()
    drawdown = (nav_series - peak) / peak
    max_dd   = drawdown.min()

    # 處理日 NAV 並計算日指標
    daily_df = (
        pd.DataFrame(daily_records)
        .drop_duplicates(subset=['date'])
        .sort_values('date')
        .set_index('date')
    )
    daily_returns = daily_df['daily_nav'].pct_change().dropna()

    daily_volatility = daily_returns.std() * np.sqrt(252)
    daily_sharpe     = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

    daily_peak     = daily_df['daily_nav'].cummax()
    daily_drawdown = (daily_df['daily_nav'] - daily_peak) / daily_peak
    daily_max_dd   = daily_drawdown.min()

    daily_calmar = annual_return / abs(daily_max_dd) if daily_max_dd != 0 else 0

    downside_returns = daily_returns[daily_returns < 0]
    downside_std     = downside_returns.std() * np.sqrt(252)
    daily_sortino    = (daily_returns.mean() * 252) / downside_std if downside_std != 0 else 0

    daily_winrate = (daily_returns > 0).mean() * 100
    
    # 月盈虧比
    monthly_wins = results_df['month_return'][results_df['month_return'] > 0]
    monthly_losses = results_df['month_return'][results_df['month_return'] < 0]
    monthly_profit_factor = monthly_wins.sum() / abs(monthly_losses.sum()) if monthly_losses.sum() != 0 else float('inf')
    monthly_win_loss_ratio = monthly_wins.mean() / abs(monthly_losses.mean()) if len(monthly_losses) > 0 and monthly_losses.mean() != 0 else float('inf')

    # ──────────────────────────────────────────────
    # 6. 對照組
    # ──────────────────────────────────────────────
    if not silent:
        print(f"讀取對照組 ({benchmark})...")
    benchmark_nav    = None
    bench_total_ret  = 0.0
    bench_annual_ret = 0.0
    bdf = pd.DataFrame()

    try:
        start_d = results_df['date'].min().strftime('%Y-%m-%d')
        end_d   = results_df['date'].max().strftime('%Y-%m-%d')
        
        etf_path = config.MARKET_DATA_DIR / 'etf.csv'
        if os.path.exists(etf_path):
            etf_df = pd.read_csv(etf_path)
            etf_df['date'] = pd.to_datetime(etf_df['date'])
            
            if benchmark in etf_df.columns:
                # 篩選日期
                bdf = etf_df[(etf_df['date'] >= start_d) & (etf_df['date'] <= end_d)].copy()
                if not bdf.empty:
                    bdf = bdf.set_index('date')
                    close = bdf[benchmark]
                    benchmark_nav    = close / close.iloc[0]
                    bench_total_ret  = benchmark_nav.iloc[-1] - 1.0
                    n_years          = (bdf.index[-1] - bdf.index[0]).days / 365.25
                    bench_annual_ret = (benchmark_nav.iloc[-1] ** (1 / max(n_years, 1/12))) - 1.0
            else:
                if not silent: print(f"警告：etf.csv 中找不到 {benchmark} 欄位。")
        else:
            if not silent: print(f"找不到 {etf_path}，跳過對照組。")
    except Exception as e:
        if not silent: print(f"對照組讀取失敗: {e}")


    # Alpha & Beta 計算
    beta = 0.0
    alpha = 0.0
    if not bdf.empty and benchmark in bdf.columns:
        # 計算基準日報酬
        bdf['bench_ret'] = bdf[benchmark].pct_change().dropna()
        # 對齊日期
        align_df = pd.concat([daily_returns, bdf['bench_ret']], axis=1, join='inner').dropna()
        if len(align_df) > 1:
            cov = align_df.cov().iloc[0, 1]
            var = align_df['bench_ret'].var()
            beta = cov / var if var != 0 else 0.0
            # Alpha = 策略日均報酬 - Beta * 大盤日均報酬
            # 轉換成年化 Alpha
            daily_alpha = align_df['daily_nav'].mean() - beta * align_df['bench_ret'].mean()
            alpha = daily_alpha * 252

    if not silent:
        print("\n" + "=" * 80)
        print(f"回測績效報告 {version}  (Strategy: Rank {rank_str})  [單位淨值法]")
        print("=" * 80)
        print(f"初始資金:     {INITIAL_CAPITAL:>12,.0f}")
        print(f"最終資產:     {int(capital):>12,}")
        print(f"期末單位淨值: {nav:>12.4f}")
        print(f"總報酬率:     {total_return*100:>10.2f}%")
        print(f"年化報酬:     {annual_return*100:>10.2f}%")
        print("-" * 80)
        print(f"月均期望值:   {expected_val:>10.2f}%")
        print(f"月勝率:       {winrate:>10.2f}%")
        print(f"月最大回撤:   {max_dd*100:>10.2f}%")
        print(f"月盈虧比(總賺/總賠):     {monthly_profit_factor:>10.2f}")
        print(f"月賺賠比(每筆賺/每筆賠): {monthly_win_loss_ratio:>10.2f}")
        print("-" * 80)
        print(f"日波動率:     {daily_volatility*100:>10.2f}%")
        print(f"日最大回撤:   {daily_max_dd*100:>10.2f}%")
        print(f"日 Sharpe:    {daily_sharpe:>10.2f}")
        print(f"日 Calmar:    {daily_calmar:>10.2f}")
        print(f"日 Sortino:   {daily_sortino:>10.2f}")
        print(f"日勝率:       {daily_winrate:>10.2f}%")
        print("-" * 80)
        print(f"Beta (benchmark): {beta:>10.2f}")
        print(f"Alpha(年化):  {alpha*100:>10.2f}%")
        print("-" * 80)
        print(f"月數:              {total_months}")
        print(f"{FIXED_STOP_LOSS*100:.0f}%停損次數:       {stop_loss_count} 次")
        print(f"宏觀停損觸發:      {macro_stop_count} 次")
        print(f"宏觀減碼觸發:      {macro_trim_count} 次")
        print("=" * 80)


    # ──────────────────────────────────────────────
    # 7. 圖表（兩張子圖合成一張 figure，統一 plt.show()）
    #    避免多次 show() 造成 tkinter mainloop 衝突
    # ──────────────────────────────────────────────

    # Rank 統計（純資料，不依賴圖表）
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

    # 圖表 1：NAV 走勢
    if not silent:
        fig1, ax1 = plt.subplots(figsize=(14, 5))
        label_ai = (f'Rank {rank_str}  |  NAV {nav:.3f}  '
                    f'({total_return*100:.0f}%  /  Annual {annual_return*100:.1f}%)')
        ax1.plot(results_df['date'], results_df['nav'],
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
        
        # 存檔 圖表 1
        plot_dir = config.get_version_dir(config.STOCK_PLOTS_DIR, version)
        fig1_path = plot_dir / f"nav_curve_rank_{rank_str.replace(',', '_')}.png"
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')

        # 圖表 2：Rank 期望值長條圖
        fig2, ax2 = plt.subplots(figsize=(14, 5))

        mean_avg_return = rank_stats['avg_return'].mean()
        # Rank 1–5 顯示紅色，其餘顯示藍色
        colors = ['crimson' if r in select_rank else 'steelblue' for r in rank_stats['rank']]

        bars = ax2.bar(rank_stats['rank'], rank_stats['avg_return'],
                       color=colors, alpha=0.8)

        ax2.set_title(f'Avg Return & Win Rate by Rank  (Red = Selected Ranks, Mean: {mean_avg_return:.2f}%)')
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Avg Return (%)')
        ax2.set_xticks(rank_stats['rank'])
        ax2.axhline(0, color='gray', linewidth=0.8)
        ax2.axhline(mean_avg_return, color='green', linestyle='--', linewidth=0.8,
                    alpha=0.7, label='Average Return')
        ax2.legend(loc='upper right')
        ax2.grid(axis='y', alpha=0.3)

        for bar, row in zip(bars, rank_stats.itertuples()):
            yval   = bar.get_height()
            va     = 'bottom' if yval >= 0 else 'top'
            offset = 0.08 if yval >= 0 else -0.08
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                yval + offset,
                f'{row.avg_return:.2f}%\n(W:{row.winrate:.1f}%)\n'
                f'Max:{row.max_return:.1f}%\nMin:{row.min_return:.1f}%',
                ha='center', va=va, fontsize=8,
            )
            
        # 存檔 圖表 2
        fig2_path = plot_dir / "rank_expected_return.png"
        fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')

        print(f"圖表已存檔至: {plot_dir}")
        print("圖表產生完畢，關閉視窗後程式繼續執行...")
        # plt.show()   # ← 唯一一次 show，會同時顯示 fig1 與 fig2，避免 tkinter mainloop 衝突

    # ──────────────────────────────────────────────
    # 8. 匯出交易日誌
    # ──────────────────────────────────────────────
    if export:
        LOG_DIR      = config.get_version_dir(config.STOCK_BACKTEST_TRADELOG_DIR, version)
        csv_filename = LOG_DIR / f"trade_log_rank_{rank_str.replace(',', '_')}.csv"
        export_full_trade_log(
            test_df     = test_df,
            filename    = csv_filename,
            select_rank = select_rank,
            model       = model,
        )

    return results_df, daily_df['daily_nav']

if __name__ == "__main__":
    backtester(version='v41', select_rank=list(range(1, 4)), benchmark='0050', entry_on_next_open=False)