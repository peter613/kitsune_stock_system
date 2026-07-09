import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
from catboost import CatBoostRanker
import yfinance as yf
import config


# ── 中文字體自動設定（解決 CJK glyph 警告）──────────────
def _set_cjk_font():
    candidates = [
        'Microsoft JhengHei',
        'Microsoft YaHei',
        'PingFang TC',
        'PingFang SC',
        'Noto Sans CJK TC',
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
    return value * cost_rate


# ──────────────────────────────────────────────
# 繪圖函式
# ──────────────────────────────────────────────

def plot_delay_comparison(results: list[dict], version: str = ''):
    """
    results: list of dicts returned by backtester(), one per delay_days value.
    Each dict must contain keys:
        delay, expected_val, winrate,
        daily_sharpe, daily_sortino, daily_calmar,
        daily_max_dd, stop_loss_count
    """
    delays       = [r['delay']           for r in results]
    expected     = [r['expected_val']    for r in results]
    winrate      = [r['winrate']         for r in results]
    sharpe       = [r['daily_sharpe']    for r in results]
    sortino      = [r['daily_sortino']   for r in results]
    calmar       = [r['daily_calmar']    for r in results]
    maxdd        = [r['daily_max_dd']    for r in results]   # already in %
    stoploss     = [r['stop_loss_count'] for r in results]

    x = np.arange(len(delays))
    xticks = [f'delay {d}' for d in delays]

    # ── 圖 1：獲利指標 ───────────────────────────────────
    fig1, ax1l = plt.subplots(figsize=(8, 4))
    fig1.suptitle(f'獲利指標  {version}', fontsize=13, fontweight='bold')
    ax1r = ax1l.twinx()

    bar_w = 0.45
    bars = ax1l.bar(x, expected, width=bar_w, color='#185FA5', alpha=0.85, label='月均期望值 %', zorder=3)
    ax1l.axhline(0, color='#185FA5', linewidth=0.8, linestyle='--', alpha=0.4)
    ax1l.set_ylabel('月均期望值 (%)', color='#185FA5', fontsize=10)
    ax1l.tick_params(axis='y', labelcolor='#185FA5')
    ax1l.set_xticks(x); ax1l.set_xticklabels(xticks, fontsize=9)
    ax1l.grid(axis='y', color='#cccccc', linewidth=0.5, zorder=0)
    ax1l.set_axisbelow(True)

    line_wr, = ax1r.plot(x, winrate, color='#D85A30', marker='o', linewidth=2,
                         markersize=5, label='月勝率 %', zorder=4)
    ax1r.axhline(50, color='#D85A30', linewidth=0.8, linestyle='--', alpha=0.4)
    ax1r.set_ylabel('月勝率 (%)', color='#D85A30', fontsize=10)
    ax1r.tick_params(axis='y', labelcolor='#D85A30')
    ax1r.set_ylim(40, 90)

    for bar, val in zip(bars, expected):
        ax1l.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                  f'{val:.1f}%', ha='center', va='bottom', fontsize=8, color='#185FA5')
    for xi, val in zip(x, winrate):
        ax1r.text(xi, val + 0.8, f'{val:.1f}%', ha='center', va='bottom', fontsize=8, color='#D85A30')

    ax1l.legend([bars, line_wr], ['月均期望值 %', '月勝率 %'], loc='upper right', fontsize=8, framealpha=0.7)
    fig1.tight_layout()
    plot_dir = config.get_version_dir(config.STOCK_PLOTS_DIR, version)
    path1 = plot_dir / 'delay_profit.png'
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"獲利指標圖已儲存：{path1}")

    # ── 圖 2：品質指標 ───────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    fig2.suptitle(f'基金品質指標  {version}', fontsize=13, fontweight='bold')

    ax2.plot(x, sharpe,  color='#534AB7', marker='o', linewidth=2, markersize=5,
             linestyle='-',  label='Sharpe')
    ax2.plot(x, sortino, color='#0F6E56', marker='s', linewidth=2, markersize=5,
             linestyle='--', label='Sortino')
    ax2.plot(x, calmar,  color='#BA7517', marker='^', linewidth=2, markersize=5,
             linestyle=':',  label='Calmar')
    ax2.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5, label='基準線 = 1.0')

    ax2.set_ylabel('比率', fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels(xticks, fontsize=9)
    ax2.grid(color='#cccccc', linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.7)
    ax2.set_ylim(bottom=0)
    fig2.tight_layout()
    path2 = plot_dir / 'delay_quality.png'
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"品質指標圖已儲存：{path2}")

    # ── 圖 3：風險指標（1×2 橫排）────────────────────────
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4))
    fig3.suptitle(f'風險指標  {version}', fontsize=13, fontweight='bold')
 
    # 左：最大回撤（折線 + 填色，負數，越低越糟）
    ax3a.plot(x, maxdd, color='#A32D2D', marker='o', linewidth=2, markersize=5, zorder=4)
    ax3a.fill_between(x, maxdd, 0, color='#A32D2D', alpha=0.10)
    ax3a.axhline(0, color='#A32D2D', linewidth=0.6, linestyle='--', alpha=0.3)
    ax3a.set_ylabel('最大回撤 (%)', color='#A32D2D', fontsize=10)
    ax3a.tick_params(axis='y', labelcolor='#A32D2D')
    ax3a.set_xticks(x); ax3a.set_xticklabels(xticks, fontsize=9)
    ax3a.set_title('日最大回撤', fontsize=10, pad=6, loc='left')
    ax3a.grid(color='#cccccc', linewidth=0.5, zorder=0)
    ax3a.set_axisbelow(True)
    # 數值標籤（負數往下貼）
    for xi, val in enumerate(maxdd):
        ax3a.text(xi, val - 0.2, f'{val:.1f}%', ha='center', va='top', fontsize=8, color='#A32D2D')
 
    # 右：停損次數（柱狀圖，正數離散計數，越高越糟）
    bars3 = ax3b.bar(x, stoploss, width=0.45, color='#5F5E5A', alpha=0.80, zorder=3)
    ax3b.set_ylabel('20% 停損次數（次）', color='#5F5E5A', fontsize=10)
    ax3b.tick_params(axis='y', labelcolor='#5F5E5A')
    ax3b.set_xticks(x); ax3b.set_xticklabels(xticks, fontsize=9)
    ax3b.set_title('20% 停損觸發次數', fontsize=10, pad=6, loc='left')
    ax3b.grid(axis='y', color='#cccccc', linewidth=0.5, zorder=0)
    ax3b.set_axisbelow(True)
    ax3b.set_ylim(bottom=0)
    # 數值標籤
    for bar, val in zip(bars3, stoploss):
        ax3b.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                  f'{val}', ha='center', va='bottom', fontsize=8, color='#5F5E5A')
 
    fig3.tight_layout()
    path3 = plot_dir / 'delay_risk.png'
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"風險指標圖已儲存：{path3}")


# ──────────────────────────────────────────────
# 主回測函式
# ──────────────────────────────────────────────

def backtester(version, select_rank=None, entry_on_next_open=False, delay_days=0):
    if select_rank is None:
        select_rank = list(range(1, 6))

    MODEL_FILE   = config.get_version_dir(config.STOCK_CBM_DIR, version) / "strategy.cbm"
    DATA_FILE    = config.STOCK_FEATURES_DIR / f"train_{version}.csv"

    INITIAL_CAPITAL  = 1_000_000
    TRANSACTION_COST = 0.005
    FIXED_STOP_LOSS  = 0.20
    AMOUNT_THRESHOLD  = 100000 # 當天成交值 > 1.0億 (設為0則不過濾)
    REVENUE_THRESHOLD = 100000 # 當月營收 > 1.0億 (設為0則不過濾)

    rank_str = ','.join(map(str, select_rank))

    # ── 1. 資料預處理 ──────────────────────────────────
    df = pd.read_csv(DATA_FILE, dtype={'id': str})
    df['date'] = pd.to_datetime(df['date'])
    cutoff_idx = int(len(df['date'].unique()) * 0.8)
    cutoff_date = sorted(df['date'].unique())[cutoff_idx]
    test_df = df[df['date'] >= cutoff_date].copy()
    test_df = test_df.sort_values(['date', 'id']).reset_index(drop=True)

    if AMOUNT_THRESHOLD > 0:
        test_df = test_df[test_df['amount'] >= AMOUNT_THRESHOLD].reset_index(drop=True)
    if REVENUE_THRESHOLD > 0 and 'month_revenue' in test_df.columns:
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
    test_df['simulated_return'] = np.nan

    # ── 3. 市場日頻資料 ──────────────────────────────
    market_df = pd.read_parquet(config.MARKET_DATA_DIR / "market.parquet")
    market_df['date'] = pd.to_datetime(market_df['date'])
    market_df['id']   = market_df['id'].astype(str)
    market_df = market_df.sort_values(['id', 'date']).set_index(['id', 'date'])

    # ── 4. 單位淨值法初始化 ──────────────────────────
    nav             = 1.0
    capital         = float(INITIAL_CAPITAL)
    monthly_results = []
    daily_records   = []
    stop_loss_count = 0

    for date, group in test_df.groupby('date'):
        top_picks = group[group['rank'].isin(select_rank)]
        if top_picks.empty:
            continue

        n_stocks = len(top_picks)
        alloc_per_stock     = capital / n_stocks
        buy_cost_total      = _calc_tx_cost(capital, TRANSACTION_COST)
        net_alloc_per_stock = (capital - buy_cost_total) / n_stocks
        cash_pool = 0.0
        portfolio = {}

        for idx in top_picks.index:
            s_id = str(test_df.loc[idx, 'id']).replace('.0', '')
            try:
                f_data = market_df.loc[s_id].loc[date:]
                if len(f_data) > delay_days:
                    f_data = f_data.iloc[delay_days:].head(20)
                else:
                    f_data = pd.DataFrame()
                if not f_data.empty:
                    if entry_on_next_open and len(f_data) > 1:
                        entry_p = f_data.iloc[1]['open']
                    else:
                        entry_p = f_data.iloc[0]['close']
                    portfolio[idx] = {
                        's_id'    : s_id,
                        'entry_p' : entry_p,
                        'shares'  : net_alloc_per_stock / entry_p,
                        'dates'   : f_data.index.get_level_values('date'),
                        'prices'  : f_data['close'].values,
                        'alive'   : True,
                        'own_cash': 0.0,
                    }
            except KeyError:
                cash_pool += net_alloc_per_stock

        if not portfolio:
            continue

        for d_idx in range(20):
            for idx, p in portfolio.items():
                if p['alive'] and d_idx < len(p['prices']):
                    if entry_on_next_open and d_idx == 0:
                        continue
                    cur_p = p['prices'][d_idx]
                    if cur_p <= p['entry_p'] * (1 - FIXED_STOP_LOSS):
                        sell_val      = p['shares'] * cur_p
                        fee           = _calc_tx_cost(sell_val, TRANSACTION_COST)
                        net_proceeds  = sell_val - fee
                        cash_pool    += net_proceeds
                        p['own_cash'] += net_proceeds
                        p['shares']   = 0.0
                        p['alive']    = False
                        stop_loss_count += 1

            today_date    = None
            day_stock_val = 0.0
            for idx, p in portfolio.items():
                if p['alive'] and d_idx < len(p['prices']):
                    day_stock_val += p['shares'] * p['prices'][d_idx]
                    if today_date is None:
                        today_date = p['dates'][d_idx]

            if today_date is not None:
                day_total = day_stock_val + cash_pool
                day_nav = day_total / INITIAL_CAPITAL
                daily_records.append({'date': today_date, 'daily_nav': day_nav})

        end_stock_val = 0.0
        for idx, p in portfolio.items():
            prices = p.get('prices', [])
            if len(prices) > 0:
                end_stock_val += p['shares'] * prices[-1]

        sell_cost = _calc_tx_cost(end_stock_val, TRANSACTION_COST)
        end_total = (end_stock_val - sell_cost) + cash_pool
        period_ratio = end_total / capital
        month_return = period_ratio - 1.0
        nav          = nav * period_ratio

        for idx, p in portfolio.items():
            prices = p.get('prices', [])
            if len(prices) > 0:
                final_mkt  = p['shares'] * prices[-1]
                sell_fee   = _calc_tx_cost(final_mkt, TRANSACTION_COST)
                total_recv = (final_mkt - sell_fee) + p['own_cash']
                sim_ret    = total_recv / alloc_per_stock - 1.0
                test_df.at[idx, 'simulated_return'] = sim_ret

        capital = end_total
        monthly_results.append({
            'date'        : date,
            'month_return': month_return,
            'period_ratio': period_ratio,
            'nav'         : nav,
            'capital'     : capital,
        })

    # ── 5. 績效計算 ──────────────────────────────────
    results_df    = pd.DataFrame(monthly_results)
    total_months  = len(results_df)
    total_return  = nav - 1.0
    annual_return = (nav ** (12 / max(total_months, 1))) - 1.0
    winrate       = (results_df['month_return'] > 0).mean() * 100
    expected_val  = results_df['month_return'].mean() * 100

    nav_series = pd.concat([pd.Series([1.0]), results_df['nav'].reset_index(drop=True)],
                           ignore_index=True)
    peak     = nav_series.cummax()
    drawdown = (nav_series - peak) / peak
    max_dd   = drawdown.min()

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

    # ── 6. 回傳 dict（不再 print）────────────────────
    return {
        'delay'          : delay_days,
        'version'        : version,
        'rank_str'       : rank_str,
        # 獲利
        'expected_val'   : expected_val,          # 月均期望值 (%)
        'winrate'        : winrate,               # 月勝率 (%)
        # 品質
        'daily_sharpe'   : daily_sharpe,
        'daily_sortino'  : daily_sortino,
        'daily_calmar'   : daily_calmar,
        # 風險
        'daily_max_dd'   : daily_max_dd * 100,    # 轉成 %
        'stop_loss_count': stop_loss_count,
        # 額外保留
        'total_return'   : total_return,
        'annual_return'  : annual_return,
        'daily_volatility': daily_volatility,
        'daily_winrate'  : daily_winrate,
        'nav'            : nav,
        'capital'        : capital,
        'total_months'   : total_months,
        'monthly_max_dd' : max_dd * 100,
    }


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    VERSION     = 'v38'
    SELECT_RANK = list(range(1, 4))

    results = []
    for i in range(10):#延遲10天
        print(f'[回測中] time delay: {i}')
        r = backtester(version=VERSION, select_rank=SELECT_RANK,
                       entry_on_next_open=False, delay_days=i)
        results.append(r)

    plot_delay_comparison(results, version=VERSION)