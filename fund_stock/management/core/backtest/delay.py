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
# 主回測與評估
# ──────────────────────────────────────────────
from simulator import backtester

def run_delay_eval(version, select_rank, delay_days):
    results_df, daily_nav = backtester(version=version, select_rank=select_rank, 
                                       benchmark='00631L', entry_on_next_open=False, 
                                       delay_days=delay_days, silent=True, export=False)
    
    total_months = len(results_df)
    nav = daily_nav.iloc[-1]
    annual_return = (nav ** (12 / max(total_months, 1))) - 1.0
    winrate       = (results_df['month_return'] > 0).mean() * 100
    expected_val  = results_df['month_return'].mean() * 100

    daily_returns = daily_nav.pct_change().dropna()
    daily_sharpe     = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    
    daily_peak     = daily_nav.cummax()
    daily_drawdown = (daily_nav - daily_peak) / daily_peak
    daily_max_dd   = daily_drawdown.min()
    
    daily_calmar = annual_return / abs(daily_max_dd) if daily_max_dd != 0 else 0
    
    downside_returns = daily_returns[daily_returns < 0]
    downside_std     = downside_returns.std() * np.sqrt(252)
    daily_sortino    = (daily_returns.mean() * 252) / downside_std if downside_std != 0 else 0

    return {
        'delay': delay_days,
        'expected_val': expected_val,
        'winrate': winrate,
        'daily_sharpe': daily_sharpe,
        'daily_sortino': daily_sortino,
        'daily_calmar': daily_calmar,
        'daily_max_dd': daily_max_dd * 100,
        'stop_loss_count': 0  # Not strictly tracked in delay loop anymore
    }

if __name__ == "__main__":
    VERSION     = 'v41'
    SELECT_RANK = list(range(1, 4))

    results = []
    for i in range(10):#延遲10天
        print(f'[回測中] time delay: {i}')
        r = run_delay_eval(version=VERSION, select_rank=SELECT_RANK, delay_days=i)
        results.append(r)

    plot_delay_comparison(results, version=VERSION)