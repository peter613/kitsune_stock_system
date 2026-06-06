"""
drawdown_distribution.py
台股加權指數（^TWII）單日漲跌幅分布分析
─────────────────────────────────────────
目的：找出單日跌幅的極端值門檻，供期貨夜盤風控參考
資料：yfinance ^TWII（1997 年至今）
"""

import pandas as pd
import numpy as np
import yfinance as yf

# ── 1. 下載資料 ──
print("下載加權指數 (^TWII)...")
df = yf.download('^TWII', period='max', progress=False)

if isinstance(df.columns, pd.MultiIndex):
    close = df['Close'].iloc[:, 0]
else:
    close = df['Close']

close = close.dropna()
daily_returns = close.pct_change().dropna()
print(f"資料區間: {close.index[0].date()} ~ {close.index[-1].date()}，共 {len(daily_returns)} 筆交易日")


# ── 2. 基本統計 ──
print("\n" + "=" * 60)
print("日報酬率基本統計")
print("=" * 60)
print(f"  均值:     {daily_returns.mean()*100:>8.4f}%")
print(f"  標準差:   {daily_returns.std()*100:>8.4f}%")
print(f"  1σ 區間:  {daily_returns.mean()*100 - daily_returns.std()*100:.2f}% ~ {daily_returns.mean()*100 + daily_returns.std()*100:.2f}%")
print(f"  2σ 區間:  {daily_returns.mean()*100 - 2*daily_returns.std()*100:.2f}% ~ {daily_returns.mean()*100 + 2*daily_returns.std()*100:.2f}%")
print(f"  3σ 區間:  {daily_returns.mean()*100 - 3*daily_returns.std()*100:.2f}% ~ {daily_returns.mean()*100 + 3*daily_returns.std()*100:.2f}%")


# ── 3. 跌幅分位數 (只看下跌的日子) ──
down_days = daily_returns[daily_returns < 0]
print(f"\n" + "=" * 60)
print(f"跌幅分位數 (共 {len(down_days)} 個下跌日 / {len(daily_returns)} 個交易日)")
print("=" * 60)

for p in [50, 25, 10, 5, 2.5, 1, 0.5]:
    val = np.percentile(down_days, p)
    # 在所有交易日中的排名
    rank_in_all = (daily_returns < val).mean() * 100
    print(f"  下跌日 {p:5.1f}th pctl:  {val*100:>7.2f}%  (全部交易日中約 {rank_in_all:.1f}% 的日子比這更慘)")


# ── 4. 各跌幅門檻的歷史發生次數 ──
print("\n" + "=" * 60)
print("各跌幅門檻歷史發生次數")
print("=" * 60)

thresholds = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
for t in thresholds:
    count = (daily_returns * 100 <= t).sum()
    freq = count / len(daily_returns) * 100
    years = len(daily_returns) / 252
    per_year = count / years
    print(f"  跌幅 ≤ {t:>3}%:  {count:>4} 次  ({freq:>5.2f}%)  平均每年 {per_year:.1f} 次")


# ── 5. 歷史最慘的 20 天 ──
print("\n" + "=" * 60)
print("歷史最慘的 20 個交易日")
print("=" * 60)

worst = daily_returns.nsmallest(20)
for i, (date, ret) in enumerate(worst.items(), 1):
    prev_close = close.loc[:date].iloc[-2] if len(close.loc[:date]) >= 2 else 0
    curr_close = close.loc[date]
    print(f"  {i:>2}. {date.date()}  {ret*100:>7.2f}%  (收盤 {curr_close:,.0f} ← {prev_close:,.0f})")


# ── 6. 結論建議 ──
p1 = np.percentile(daily_returns, 1) * 100
p5 = np.percentile(daily_returns, 5) * 100
sigma2 = (daily_returns.mean() - 2 * daily_returns.std()) * 100
sigma3 = (daily_returns.mean() - 3 * daily_returns.std()) * 100

print("\n" + "=" * 60)
print("極端值門檻建議")
print("=" * 60)
print(f"  溫和異常 (5th pctl):   {p5:.2f}%  ← 每 20 個交易日約遇到 1 次")
print(f"  2σ 事件:               {sigma2:.2f}%  ← 理論上每 22 個交易日 1 次")
print(f"  極端異常 (1st pctl):   {p1:.2f}%  ← 每 100 個交易日約遇到 1 次")
print(f"  3σ 事件:               {sigma3:.2f}%  ← 理論上每 370 個交易日 1 次")
print("=" * 60)
