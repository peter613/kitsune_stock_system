"""
market_temper.py
市場水溫觀測器 (Market Temperature - Plotly 互動式雙軸分格觀測圖)
─────────────────────────────────────────
Plotly 互動圖表特性：
1. 滑鼠直接 Hover：隨時觸碰任何點位，即時顯示當日 KAMA 與 ADX 數值。
2. 零遮擋純淨圖表：移除所有圖例 (showlegend=False)，不受方塊遮擋線條。
3. 標出 y=1.00 黑色虛線：一目瞭然掌握多空強弱分水嶺。
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from talib import KAMA, ADX, EMA

# 確保可匯入根目錄 config
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
import config

def compute_tech(group):
    close = group['close'].values.astype(float)
    high  = group['high'].values.astype(float)
    low   = group['low'].values.astype(float)
    amount = group['amount'].values.astype(float)

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

    # ── 計算個股 20日 EMA 成交值 (與預測模型統一) ──
    try:
        amount_ln = np.log1p(amount)
        amount_ema_20 = EMA(amount_ln, timeperiod=20)
    except Exception:
        amount_ema_20 = np.full(len(close), np.nan)

    group['kama'] = kama
    group['adx'] = adx
    group['amount_ema_20'] = amount_ema_20
    return group

def analyze_market_temper(lookback_days=60, filter_top_liquidity=True):
    print("=" * 66)
    print(" [Market Temper] 台股白名單市場水溫儀表板")
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
    df['safe_id'] = df['id'].astype(str)
    df = df.groupby('id', group_keys=False).apply(compute_tech)
    df = df.reset_index(drop=True)
    df['id'] = df['safe_id']
    df = df.dropna(subset=['kama', 'adx', 'amount_ema_20']).reset_index(drop=True)

    if filter_top_liquidity:
        print("執行 IPO 上市未滿 240 天過濾 (與 pred.py 統一)...")
        id_counts = df['id'].value_counts()
        valid_ids = id_counts[id_counts >= 240].index
        df = df[df['id'].isin(valid_ids)].copy().reset_index(drop=True)

        print("執行 EMA 成交值流動性過濾 (保留 Top 1/3，與預測集 market_kama_mean 統一)...")
        df['liq_rank'] = df.groupby('date')['amount_ema_20'].rank(ascending=False, method='min')
        df['day_count'] = df.groupby('date')['id'].transform('count')
        df = df[df['liq_rank'] <= (df['day_count'] / 3)].copy()

    # 4. 每日聚合運算 (市場均值：KAMA Ratio 與 ADX)
    daily_df = df.groupby('date').agg(
        kama_val=('kama', 'mean'),
        adx_val=('adx', 'mean')
    ).reset_index().sort_values('date')

    # 5. 截取最近 lookback_days 個交易日 (單季 60 日)
    df_lookback = daily_df.tail(lookback_days).copy().reset_index(drop=True)

    kama_avg = df_lookback['kama_val'].mean()
    adx_avg = df_lookback['adx_val'].mean()
    kama_now = df_lookback['kama_val'].iloc[-1]
    adx_now = df_lookback['adx_val'].iloc[-1]
    latest_date_str = df_lookback['date'].iloc[-1].strftime('%Y-%m-%d')

    print(f"\n時間區間: {df_lookback['date'].min().strftime('%Y-%m-%d')} ~ {df_lookback['date'].max().strftime('%Y-%m-%d')}")
    print(f"{lookback_days}日平均 -> KAMA: {kama_avg:.4f} | ADX: {adx_avg:.2f}")
    print(f"最新現值({latest_date_str}) -> KAMA: {kama_now:.4f} | ADX: {adx_now:.2f}")
    print("-" * 54)
    print(f"{'日期':<12} | {'KAMA 均值 (多空)':>16} | {'ADX 均值 (動能)':>16}")
    print("-" * 54)
    for _, row in df_lookback.tail(10).iterrows():
        print(f"{row['date'].strftime('%Y-%m-%d'):<12} | {row['kama_val']:>16.4f} | {row['adx_val']:>16.2f}")
    print("-" * 54)

    # 6. 繪製 Plotly 互動式分格對照圖 (上圖 KAMA / 下圖 ADX)
    min_date_str = df_lookback['date'].min().strftime('%Y-%m-%d')
    max_date_str = df_lookback['date'].max().strftime('%Y-%m-%d')
    title_prefix = "台股高流動性白名單市場水溫觀測圖 (Top 1/3)" if filter_top_liquidity else "台股白名單市場水溫觀測圖"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.55, 0.45],
        subplot_titles=(
            f"<b>KAMA 均溫 (now: {kama_now:.4f} ｜ 60日均: {kama_avg:.4f})</b>",
            f"<b>ADX 趨勢強度 (now: {adx_now:.2f} ｜ 60日均: {adx_avg:.2f})</b>"
        )
    )

    # ── 上圖: KAMA 均溫走勢 (紅線) ──
    fig.add_trace(
        go.Scatter(
            x=df_lookback['date'],
            y=df_lookback['kama_val'],
            mode='lines+markers',
            name='KAMA Ratio',
            line=dict(color='#d62728', width=2.5),
            marker=dict(size=6),
            hovertemplate='日期: %{x|%Y-%m-%d}<br><b>KAMA: %{y:.4f}</b><extra></extra>'
        ),
        row=1, col=1
    )

    # 標出 y=1.00 黑色虛線
    fig.add_hline(
        y=1.00, line_color='black', line_dash='dash', line_width=2,
        annotation_text="<b>強弱界線 (1.00)</b>",
        annotation_position="top right",
        row=1, col=1
    )

    # ── 下圖: ADX 趨勢強度走勢 (藍線) ──
    fig.add_trace(
        go.Scatter(
            x=df_lookback['date'],
            y=df_lookback['adx_val'],
            mode='lines+markers',
            name='ADX',
            line=dict(color='#1f77b4', width=2.5),
            marker=dict(size=6, symbol='triangle-up'),
            hovertemplate='日期: %{x|%Y-%m-%d}<br><b>ADX: %{y:.2f}</b><extra></extra>'
        ),
        row=2, col=1
    )

    # ── 版面與標題設定 (不要圖例 showlegend=False, 主標題空一格留白) ──
    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix} ({min_date_str} ~ {max_date_str})</b><br><br>",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        showlegend=False,             # 不要圖例(不會遮擋線條)
        hovermode='x unified',        # 滑鼠碰觸時顯示整列統一提示窗
        template='plotly_white',
        height=700,
        margin=dict(l=60, r=60, t=115, b=50)
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickformat='%m-%d')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', row=1, col=1, title_text="KAMA")
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', row=2, col=1, title_text="ADX")

    print(f"\n[顯示圖表] 正在開啟 Plotly 互動式圖表... (請於瀏覽器中檢視與 Hover)...")
    fig.show()

    return df_lookback

if __name__ == "__main__":
    analyze_market_temper(lookback_days=60, filter_top_liquidity=True)
