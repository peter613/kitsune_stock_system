"""
engine.py
ETF 基金管理引擎
職責：資料讀取、資產結算
供應：app.py

journal.csv 格式規範（fund flows）：
  買入：id=ETF代號, action=buy,   price=成交價, share=股數
  賣出：id=ETF代號, action=sell,  price=成交價, share=股數

transaction.csv 格式規範（fund subscription/redemption）：
  申購： action=subscribe, price=申購金額(TWD), share=空白
  贖回： action=redeem,    price=空白,          share=贖回單位數
"""
# ────────
# setting
# ────────
START_DATE = None  # 尚未啟用，啟用後設定起始日如 '2026-07-01'

import os
import numpy as np
import pandas as pd
import streamlit as st
import config

# ── ETF 交易稅率 ──
ETF_TAX_RATE = 0.001  # ETF 優惠稅率 0.1%


# ──────────────────────────────────────────
# 輔助：從 etf.csv 寬表讀取報價 → 長表
# ──────────────────────────────────────────
def _load_etf_prices():
    """
    讀取 etf.csv (寬表)，轉為長表格式 (date, id, close)，
    供 engine 和 settlement 使用。
    """
    etf_path = config.MARKET_DATA_DIR / "etf.csv"
    if not os.path.exists(etf_path):
        return pd.DataFrame(columns=['date', 'id', 'close'])

    df_wide = pd.read_csv(etf_path)
    df_wide.columns = df_wide.columns.str.strip()
    df_wide['date'] = pd.to_datetime(df_wide['date'], errors='coerce').dt.normalize()

    # 寬表 → 長表 (melt)
    etf_cols = [c for c in df_wide.columns if c != 'date']
    df_long = df_wide.melt(
        id_vars=['date'],
        value_vars=etf_cols,
        var_name='id',
        value_name='close'
    )
    df_long['close'] = pd.to_numeric(df_long['close'], errors='coerce')
    df_long = df_long.dropna(subset=['close'])
    df_long = df_long.sort_values(['id', 'date']).reset_index(drop=True)

    if START_DATE:
        df_long = df_long[df_long['date'] >= START_DATE].copy()

    return df_long


# ──────────────────────────────────────────
# 模組 1：資料讀取
# ──────────────────────────────────────────
@st.cache_data
def load_data():
    file_path_journal     = config.ETF_JOURNAL_DIR / "journal.csv"
    file_path_transaction = config.FUND_ETF_ADMINISTRATION_DIR / "transaction.csv"

    # ── ETF 交易紀錄（journal.csv）──
    df_record = pd.read_csv(file_path_journal, dtype={'id': str})
    df_record.columns  = df_record.columns.str.strip().str.lower()
    df_record['label'] = df_record['label'].str.strip().str.lower()
    df_record['date']  = pd.to_datetime(df_record['date'], errors='coerce').dt.normalize()
    if START_DATE:
        df_record = df_record[df_record['date'] >= START_DATE].copy()
    df_record['price'] = pd.to_numeric(df_record['price'], errors='coerce').fillna(0)
    df_record['share'] = pd.to_numeric(df_record['share'], errors='coerce').fillna(0)

    # ── 申購/贖回紀錄（transaction.csv）──
    df_flows = pd.read_csv(file_path_transaction, dtype={'id': str}, encoding='utf-8-sig')
    df_flows.columns = df_flows.columns.str.strip().str.lower()
    df_flows['date']  = pd.to_datetime(df_flows['date'], errors='coerce').dt.normalize()
    if START_DATE:
        df_flows = df_flows[df_flows['date'] >= START_DATE].copy()
    df_flows['price'] = pd.to_numeric(df_flows['price'], errors='coerce').fillna(0)
    df_flows['share'] = pd.to_numeric(df_flows['share'], errors='coerce').fillna(0)

    # ── ETF 報價
    df_price = _load_etf_prices()

    # ── Benchmark（從同一張 etf.csv 篩選）──
    def _load_bench(ticker):
        """從寬表 etf.csv 讀取指定 ETF 的收盤價"""
        etf_path = config.MARKET_DATA_DIR / "etf.csv"
        if not os.path.exists(etf_path):
            return pd.DataFrame()
        df_etf = pd.read_csv(etf_path)
        df_etf.columns = df_etf.columns.str.strip()
        if ticker not in df_etf.columns:
            return pd.DataFrame()
        df = df_etf[['date', ticker]].rename(columns={ticker: 'close'}).copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
        df['daily_return'] = df['close'].pct_change()
        df['realized_vol'] = df['daily_return'].rolling(window=20).std() * (252 ** 0.5)
        if START_DATE:
            df = df[df['date'] >= START_DATE].copy()
        return df

    df_0050   = _load_bench("0050")
    df_00631L = _load_bench("00631L")

    return df_record, df_flows, df_price, df_0050, df_00631L


# ──────────────────────────────────────────
# 內部輔助：手續費 / 稅計算 (ETF 優惠稅率)
# ──────────────────────────────────────────
def _tax_rate(stock_id: str) -> float:
    """ETF 交易稅 0.1%"""
    return ETF_TAX_RATE


def _buy_cost(price: float, share: float, stock_id: str) -> float:
    """買入總成本（含手續費）"""
    val = price * share
    return val + int(val * 0.001425)


def _sell_proceeds(price: float, share: float, stock_id: str) -> float:
    """賣出淨收入（扣手續費與交易稅）"""
    val = price * share
    return val - int(val * 0.001425) - int(val * _tax_rate(stock_id))


# ──────────────────────────────────────────
# 模組 2：資產結算引擎
# ──────────────────────────────────────────
@st.cache_data
def calculate_settlement_data(df_record, df_price):
    if df_record.empty:
        return [], [], 0, 0

    df_trade  = df_record.copy()
    latest_px = df_price.groupby('id').tail(1).set_index('id')['close'].to_dict() if not df_price.empty else {}
    unrealized_data, realized_data = [], []

    for aid in df_trade['id'].unique():
        sub        = df_trade[df_trade['id'] == aid].sort_values('date')
        stock_name = sub['name'].iloc[0] if 'name' in sub.columns else ""
        tax_rate   = _tax_rate(aid)

        cum_shares      = 0.0
        total_cost      = 0.0
        total_r_pnl     = 0.0
        total_revenue   = 0.0
        total_sell_cost = 0.0
        last_sell_date  = None

        for _, row in sub.iterrows():
            trade_val = row['price'] * row['share']
            if row['action'] == 'buy':
                fee         = int(trade_val * 0.001425)
                cum_shares += row['share']
                total_cost += trade_val + fee

            elif row['action'] == 'sell':
                fee         = int(trade_val * 0.001425) + int(trade_val * tax_rate)
                net_revenue = trade_val - fee

                avg_cost_now    = total_cost / cum_shares if cum_shares > 0 else 0
                sell_cost       = row['share'] * avg_cost_now

                total_r_pnl    += net_revenue - sell_cost
                total_revenue  += net_revenue
                total_sell_cost += sell_cost

                cum_shares    -= row['share']
                total_cost    -= sell_cost
                last_sell_date = row['date']

        shares_sold = sub[sub['action'] == 'sell']['share'].sum()
        if shares_sold > 0 and last_sell_date is not None:
            realized_data.append({
                "日期":       last_sell_date.strftime("%Y-%m-%d"),
                "代號":       aid,
                "名稱":       stock_name,
                "累積報酬率": f"{(total_r_pnl / total_sell_cost) * 100:.1f}%" if total_sell_cost > 0 else "0%",
                "已實現損益": round(total_r_pnl, 0),
                "回收現金":   round(total_revenue, 0),
            })
        realized_data = sorted(realized_data, key=lambda x: x["日期"], reverse=True)
        
        if cum_shares > 0:
            px           = latest_px.get(aid, 0.0)
            mv           = cum_shares * px
            est_sell_fee = int(mv * 0.001425) + int(mv * tax_rate)
            u_pnl        = mv - total_cost - est_sell_fee
            unrealized_data.append({
                "代號":       aid,
                "名稱":       stock_name,
                "累積報酬率": f"{(u_pnl / total_cost) * 100:.1f}%" if total_cost > 0 else "0%",
                "未實現損益": round(u_pnl, 0),
                "當前市值":   round(mv, 0),
            })

    total_unrealized = sum(item["未實現損益"] for item in unrealized_data)
    total_realized   = sum(item["已實現損益"] for item in realized_data)

    return unrealized_data, realized_data, total_unrealized, total_realized


