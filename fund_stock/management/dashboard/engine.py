"""
engine_quant.py
Kitsune Kernel — 基金管理引擎
職責：資料讀取、資產結算、風控警報
供應：app_quant.py

journal.csv 格式規範（fund flows）：
  買股：id=股票代號, action=buy,   price=股價,          share=股數
  賣股：id=股票代號, action=sell,  price=股價,          share=股數

transaction.csv 格式規範（fund subscription/redemption）：
  申購： action=subscribe, price=申購金額(TWD), share=空白
  贖回： action=redeem,    price=空白,          share=贖回單位數
"""
# ────────
# setting
# ────────
START_DATE = '2026-03-10'

import os
import numpy as np
import pandas as pd
import streamlit as st
import config

# ──────────────────────────────────────────
# 模組 1：資料讀取
# ──────────────────────────────────────────
@st.cache_data
def load_data():
    file_path_journal     = config.STOCK_JOURNAL_DIR / "journal.csv"
    file_path_transaction = config.FUND_STOCK_ADMINISTRATION_DIR / "transaction.csv"

    # ── 股票交易紀錄（journal.csv）──
    df_record = pd.read_csv(file_path_journal, dtype={'id': str})
    df_record.columns  = df_record.columns.str.strip().str.lower()
    df_record['label'] = df_record['label'].str.strip().str.lower()
    df_record['date']  = pd.to_datetime(df_record['date'], errors='coerce').dt.normalize()
    df_record = df_record[df_record['date'] >= START_DATE].copy()
    df_record['price'] = pd.to_numeric(df_record['price'], errors='coerce').fillna(0)
    df_record['share'] = pd.to_numeric(df_record['share'], errors='coerce').fillna(0)

    # ── 申購/贖回紀錄（transaction.csv）──
    df_flows = pd.read_csv(file_path_transaction, dtype={'id': str}, encoding='utf-8-sig')
    df_flows.columns = df_flows.columns.str.strip().str.lower()
    df_flows['date']  = pd.to_datetime(df_flows['date'], errors='coerce').dt.normalize()
    df_flows = df_flows[df_flows['date'] >= START_DATE].copy()
    df_flows['price'] = pd.to_numeric(df_flows['price'], errors='coerce').fillna(0)
    df_flows['share'] = pd.to_numeric(df_flows['share'], errors='coerce').fillna(0)

    def _standardize(df):
        df.columns = df.columns.str.strip().str.lower()
        if 'price' in df.columns and 'close' not in df.columns:
            df = df.rename(columns={'price': 'close'})
        return df

    market_path = config.MARKET_DATA_DIR / "market.parquet"
    if os.path.exists(market_path):
        df_price = pd.read_parquet(market_path)
        df_price = _standardize(df_price)
        df_price['date'] = pd.to_datetime(df_price['date'], errors='coerce').dt.normalize()
        df_price = df_price[df_price['date'] >= START_DATE].copy()
    else:
        df_price = pd.DataFrame(columns=['date', 'id', 'close'])

    def _validate_ids(df_record, df_price):
        record_ids = set(df_record['id'].unique())
        market_ids = set(df_price['id'].unique())
        orphan_ids = record_ids - market_ids
        if orphan_ids:
            st.error(f"偵測到無效代號：{orphan_ids}。請檢查 journal.csv。")

    _validate_ids(df_record, df_price)

    def _load_bench(name):
        path = config.MARKET_DATA_DIR / name
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, dtype={'id': str})
        df = _standardize(df)
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
        df['daily_return'] = df['close'].pct_change()
        df['realized_vol'] = df['daily_return'].rolling(window=20).std() * (252 ** 0.5)
        return df[df['date'] >= START_DATE].copy()

    df_0050   = _load_bench("0050.csv")
    df_00631L = _load_bench("00631.csv")

    return df_record, df_flows, df_price, df_0050, df_00631L


# ──────────────────────────────────────────
# 內部輔助：手續費 / 稅計算
# ──────────────────────────────────────────
def _tax_rate(stock_id: str) -> float:
    """個股 0.3%"""
    return 0.003


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
        yahoo_url  = f"https://tw.stock.yahoo.com/quote/{aid}/technical-analysis"

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
                "代號":       aid,#若未來要超連結改成:yahoo_url,
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
                "代號":       aid,#若未來要超連結改成:yahoo_url,
                "名稱":       stock_name,
                "累積報酬率": f"{(u_pnl / total_cost) * 100:.1f}%" if total_cost > 0 else "0%",
                "未實現損益": round(u_pnl, 0),
                "當前市值":   round(mv, 0),
            })

    total_unrealized = sum(item["未實現損益"] for item in unrealized_data)
    total_realized   = sum(item["已實現損益"] for item in realized_data)

    return unrealized_data, realized_data, total_unrealized, total_realized


# ──────────────────────────────────────────
# 模組 3：風控警報
# ──────────────────────────────────────────
def _check_rebalance_alerts(df_record, df_price):
    if df_record.empty or df_price.empty:
        return []

    df_trade  = df_record[~df_record['id'].str.startswith('00')].copy()
    latest_px = df_price.groupby('id').tail(1).set_index('id')['close'].to_dict()
    holdings  = {}
    pure_equity_mv = 0.0

    for aid in df_trade['id'].unique():
        sub    = df_trade[df_trade['id'] == aid]
        shares = sub[sub['action'] == 'buy']['share'].sum() - sub[sub['action'] == 'sell']['share'].sum()
        if shares > 0:
            px = latest_px.get(aid, 0.0)
            if px == 0.0:
                last_buy = sub[sub['action'] == 'buy']
                px = last_buy['price'].iloc[-1] if not last_buy.empty else 0.0
            if px > 0:
                mv   = shares * px
                name = sub['name'].iloc[0] if 'name' in sub.columns else ""
                holdings[aid]   = {'name': name, 'shares': shares, 'price': px, 'mv': mv}
                pure_equity_mv += mv

    alerts = []
    if pure_equity_mv > 0:
        for aid, data in holdings.items():
            weight = data['mv'] / pure_equity_mv
            if weight > 0.25:
                sell_value  = (data['mv'] - 0.20 * pure_equity_mv) / 0.8
                sell_shares = int(sell_value / data['price'])
                alerts.append(
                    f"【指令 : 重構】{aid} {data['name']} 投組權重大於25% "
                    f"(目前{weight*100:.1f}%)，需賣出市值 {sell_value:,.0f} "
                    f"(約需賣出 {sell_shares} 股)"
                )
    return alerts


def _check_stoploss_alerts(unrealized_data, threshold=-20.0):
    alerts = []
    for item in unrealized_data:
        roi_float = float(item["累積報酬率"].replace("%", ""))
        if roi_float <= threshold:
            alerts.append(
                f"【指令 : 滅止 】{item['代號']} {item['名稱']} 已觸發 -20% 停損保險機制"
            )
    return alerts


def _check_bear_market_alerts(df_0050, threshold=-0.20):
    if df_0050.empty or 'close' not in df_0050.columns:
        return []
    max_price    = df_0050['close'].max()
    latest_price = df_0050['close'].iloc[-1]
    if max_price == 0:
        return []
    drawdown = (latest_price - max_price) / max_price
    if drawdown <= threshold:
        return [f"【指令 : 凜冬 】 大盤已從高點回撤 {drawdown*100:.1f}%，正式進入熊市，全面砍倉。"]
    return []


def generate_system_alerts(df_record, df_price, df_0050, unrealized_data):
    """
    統一收集並回傳所有風控警報。
    """
    alerts = []
    alerts.extend(_check_bear_market_alerts(df_0050))
    alerts.extend(_check_stoploss_alerts(unrealized_data))

    #暫且不比例獲利平衡(心理作用，回測無增加績效)
    #alerts.extend(_check_rebalance_alerts(df_record, df_price))
    
    return alerts