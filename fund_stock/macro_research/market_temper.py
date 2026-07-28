"""
market_temper.py
市場水溫觀測器 (Market Temperature - 奧卡姆剃刀・雙座標軸單圖)
─────────────────────────────────────────
一圖到底、雙座標軸疊加觀測：
左 Y 軸 (紅線): KAMA 均值 (多空趨勢與溫度)
右 Y 軸 (藍線): ADX  均值 (全市場趨勢動能強度)
零干擾虛線，60 日平均值收斂於圖表主標題。
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 讓本地執行時彈出互動視窗
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from talib import KAMA, ADX

# 設定 Windows 繁體中文微軟正黑體，防止方塊字
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Taipei Sans TC Beta', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 確保可匯入根目錄 config
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
import config

def compute_tech(group):
    close = group['close'].values.astype(float)
    high  = group['high'].values.astype(float)
    low   = group['low'].values.astype(float)

    # ── 計算個股 KAMA(30) 比率 ──
    try:
        kama = close / (KAMA(close, timeperiod=30) + 1e-9)
    except Exception:
        kama = np.full(len(close), np.nan)

    # ── 計算個股 ADX(14) 趨勢強度 ──
    try:
        adx = ADX(high, low, close, timeperiod=14)
    except Exception:
        adx = np.full(len(close), np.nan)

    group['kama'] = kama
    group['adx'] = adx
    return group

def analyze_market_temper(lookback_days=60):
    print("=" * 66)
    print(" [Market Temper] 台股白名單市場水溫儀表板 (奧卡姆剃刀極簡版)")
    print("=" * 66)

    # 1. 讀取市場資料
    market_path = config.MARKET_DATA_DIR / 'market.parquet'
    if not market_path.exists():
        print(f"找不到檔案: {market_path}")
        return None

    df = pd.read_parquet(market_path)
    df['date'] = pd.to_datetime(df['date'])
    df['id'] = df['id'].astype(str).str.zfill(4)

    # 2. 白名單與黑名單篩選
    white_list_path = config.STOCK_WHITE_LIST_DIR / 'white_list.csv'
    if white_list_path.exists():
        white_df = pd.read_csv(white_list_path, encoding='utf-8-sig')
        white_ids = white_df['id'].astype(str).str.zfill(4)
        df = df[df['id'].isin(white_ids)]

    black_ids = ['2412', '3045', '4904']
    df = df[~df['id'].isin(black_ids)]

    # 3. 取近 2 年資料做指標運算 (確保均線與 ADX 充足預熱)
    start_date = df['date'].max() - pd.DateOffset(years=2)
    df = df[df['date'] >= start_date].copy().reset_index(drop=True)

    print("正在計算白名單 KAMA 與 ADX 均值...")
    df = df.sort_values(['id', 'date']).reset_index(drop=True)
    df = df.groupby('id', group_keys=False).apply(compute_tech)
    df = df.dropna(subset=['kama', 'adx']).reset_index(drop=True)

    # 4. 每日聚合運算 (市場均值：KAMA Ratio 與 ADX)
    daily_df = df.groupby('date').agg(
        kama_val=('kama', 'mean'),
        adx_val=('adx', 'mean')
    ).reset_index().sort_values('date')

    # 5. 截取最近 lookback_days 個交易日 (單季 60 日)
    df_lookback = daily_df.tail(lookback_days).copy().reset_index(drop=True)

    kama_avg = df_lookback['kama_val'].mean()
    adx_avg = df_lookback['adx_val'].mean()

    print(f"\n時間區間: {df_lookback['date'].min().strftime('%Y-%m-%d')} ~ {df_lookback['date'].max().strftime('%Y-%m-%d')}")
    print(f"{lookback_days}日平均 -> KAMA: {kama_avg:.4f} | ADX: {adx_avg:.2f}")
    print("-" * 54)
    print(f"{'日期':<12} | {'KAMA 均值 (多空)':>16} | {'ADX 均值 (動能)':>16}")
    print("-" * 54)
    for _, row in df_lookback.tail(10).iterrows():
        print(f"{row['date'].strftime('%Y-%m-%d'):<12} | {row['kama_val']:>16.4f} | {row['adx_val']:>16.2f}")
    print("-" * 54)

    # 6. 繪製奧卡姆剃刀・上下分格對照圖 (上圖 KAMA / 下圖 ADX)
    fig, (ax_kama, ax_adx) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={'height_ratios': [1.1, 1]})

    # 標題清楚標示 60 日平均
    fig.suptitle(
        f'台股白名單市場水溫觀測圖 ({df_lookback['date'].min().strftime('%Y-%m-%d')} ~ {df_lookback['date'].max().strftime('%Y-%m-%d')})\n'
        f'KAMA 均值: {kama_avg:.4f}  |  ADX 均值: {adx_avg:.2f}\n',
        fontsize=15, fontweight='bold', y=0.96
    )

    # ── 上圖: KAMA 均值 (多空趨勢方向) ──
    ax_kama.plot(
        df_lookback['date'], df_lookback['kama_val'],
        color='#d62728', linewidth=2, marker='o', markersize=4,
    )
    ax_kama.set_ylabel('KAMA Ratio', fontsize=11)
    ax_kama.grid(True, linestyle='--', alpha=0.35)
    ax_kama.legend(loc='upper left', framealpha=0.9, fontsize=11)

    # ── 下圖: ADX 均值 (趨勢動能浪頭) ──
    ax_adx.plot(
        df_lookback['date'], df_lookback['adx_val'],
        color='#1f77b4', linewidth=2, marker='^', markersize=4,
    )
    ax_adx.set_ylabel('ADX', fontsize=11)
    ax_adx.grid(True, linestyle='--', alpha=0.35)
    ax_adx.legend(loc='upper left', framealpha=0.9, fontsize=11)

    ax_adx.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax_adx.tick_params(axis='x', rotation=25)

    plt.tight_layout()
    fig.subplots_adjust(top=0.86)

    # 不呼叫 plt.savefig，僅互動呈現
    print(f"\n[顯示圖表] 請在彈出的視窗中檢視 {lookback_days} 日上下對照圖 (關閉視窗結束)...")
    plt.show()

    return df_lookback

if __name__ == "__main__":
    analyze_market_temper(lookback_days=60)
