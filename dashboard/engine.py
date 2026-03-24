"""
engine.py
Kitsune Kernel — 計算引擎層
職責：資料讀取、時間序列計算、資產結算、警報判斷
規則：本檔案不得引入任何 streamlit 渲染 (st.write / st.plotly_chart …)
      唯一例外：load_data 內的 st.error 防呆提示保留
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import config

START_DATE = '2026-03-11'
INIT_CAP   = 335800


# ──────────────────────────────────────────
# 模組 1：資料讀取
# ──────────────────────────────────────────
@st.cache_data
def load_data():
    file_path_record = config.JOURNAL_DIR / "journal.csv"

    # ── 交易紀錄 ──────────────────────────
    df_record = pd.read_csv(file_path_record, dtype={'id': str})
    df_record.columns = df_record.columns.str.strip().str.lower()
    df_record['date'] = pd.to_datetime(df_record['date'], errors='coerce').dt.normalize()
    df_record = df_record[df_record['date'] >= START_DATE].copy()
    df_record['price'] = pd.to_numeric(df_record['price'], errors='coerce').fillna(0)
    df_record['share'] = pd.to_numeric(df_record['share'], errors='coerce').fillna(0)
    df_record['cash_flow'] = np.where(
        df_record['action'] == 'in',  df_record['price'],
        np.where(df_record['action'] == 'out', -df_record['price'], 0)
    )

    # ── 內部工具：標準化欄位名稱 ──────────
    def _standardize(df):
        df.columns = df.columns.str.strip().str.lower()
        if 'price' in df.columns and 'close' not in df.columns:
            df = df.rename(columns={'price': 'close'})
        return df

    # ── 市場報價 (寬表 parquet) ────────────
    market_path = config.DATABASE_DIR / "market.parquet"
    if os.path.exists(market_path):
        df_price = pd.read_parquet(market_path)
        df_price = _standardize(df_price)
        df_price['date'] = pd.to_datetime(df_price['date'], errors='coerce').dt.normalize()
        df_price = df_price[df_price['date'] >= START_DATE].copy()
    else:
        df_price = pd.DataFrame(columns=['date', 'id', 'close'])

    # ── 防呆：交易紀錄中找不到對應報價的 ID ─
    def _validate_ids(df_record, df_price):
        record_ids = set(df_record[df_record['id'] != 'cash']['id'].unique())
        market_ids = set(df_price['id'].unique())
        orphan_ids = record_ids - market_ids
        if orphan_ids:
            st.error(f"偵測到無效代號：{orphan_ids}。請檢查 journal.csv 或 market.csv。")

    _validate_ids(df_record, df_price)

    # ── 基準線 (先算波動率再切片) ──────────
    def _load_bench(name):
        path = config.DATABASE_DIR / f'{name}.csv'
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, dtype={'id': str})
        df = _standardize(df)
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
        df.columns = [c.lower() for c in df.columns]
        p_col = 'price' if 'price' in df.columns else 'close'
        df['daily_return']  = df[p_col].pct_change()
        df['realized_vol']  = df['daily_return'].rolling(window=20).std() * (252 ** 0.5)
        return df[df['date'] >= START_DATE].copy()

    df_0050   = _load_bench("0050")
    df_00631L = _load_bench("00631")

    return df_record, df_price, df_0050, df_00631L


# ──────────────────────────────────────────
# 模組 2：時間序列計算引擎 (NAV 淨值法)
# ──────────────────────────────────────────
@st.cache_data
def calculate_daily_history(df_record, df_price, df_0050, df_00631L):
    if df_record.empty:
        return pd.DataFrame()

    dates = sorted(df_price['date'].dropna().unique())
    if not dates:
        dates = sorted(df_record['date'].unique())

    p0_0050   = df_0050['close'].iloc[0]   if not df_0050.empty   else 1.0
    p0_00631L = df_00631L['close'].iloc[0] if not df_00631L.empty else 1.0

    total_units  = 0.0
    nav_per_unit = 1.0
    history      = []

    for d in dates:
        rec_d = df_record[df_record['date'] <= d]
        if rec_d.empty:
            continue

        px_d        = df_price[df_price['date'] <= d].groupby('id').tail(1).set_index('id')['close'].to_dict()
        cash_rec    = rec_d[rec_d['id'] == 'cash']
        principal   = cash_rec['cash_flow'].sum()
        net_cf_today= cash_rec[cash_rec['date'] == d]['cash_flow'].sum()
        trades      = rec_d[rec_d['id'] != 'cash'].copy()
        trades['val'] = trades['price'] * trades['share']

        # ── 子投組計算 (含手續費 + 預扣賣出稅費) ──
        def _calc_sub(label_filter):
            sub = trades[trades['label'] == label_filter] if label_filter else trades
            if sub.empty:
                return 0.0, 0.0, 0.0
            buy_mask  = sub['action'] == 'buy'
            sell_mask = sub['action'] == 'sell'
            buy_fees  = (sub[buy_mask]['val']  * 0.001425).astype(int)
            sell_fees = (sub[sell_mask]['val'] * 0.001425).astype(int) + (sub[sell_mask]['val'] * 0.003).astype(int)
            buy_cost      = sub[buy_mask]['val'].sum()  + buy_fees.sum()
            sell_revenue  = sub[sell_mask]['val'].sum() - sell_fees.sum()
            net_flow      = sell_revenue - buy_cost
            mv = 0.0
            for aid in sub['id'].unique():
                a_sub  = sub[sub['id'] == aid]
                shares = a_sub[a_sub['action'] == 'buy']['share'].sum() - a_sub[a_sub['action'] == 'sell']['share'].sum()
                if shares > 0:
                    raw_mv = shares * px_d.get(aid, 0.0)
                    if str(aid).startswith('00'):
                        est_sell_fee = int(raw_mv * 0.001425) + int(raw_mv * 0.001)
                    else:
                        est_sell_fee = int(raw_mv * 0.001425) + int(raw_mv * 0.003)
                    mv += (raw_mv - est_sell_fee)
            roi = (mv + net_flow) / buy_cost if buy_cost > 0 else 0.0
            return mv, net_flow, roi

        mv_ai,  _, roi_ai  = _calc_sub('ai')
        mv_oth, _, roi_oth = _calc_sub('other')
        mv_tot, flow_tot, _= _calc_sub(None)

        # ── NAV 淨值法更新 ──────────────────
        current_cash = principal + flow_tot
        total_nav    = current_cash + mv_tot

        if total_units == 0.0 and net_cf_today > 0:
            nav_per_unit = 1.0
            total_units  = net_cf_today / nav_per_unit
        elif total_units > 0:
            nav_before_cf = total_nav - net_cf_today
            nav_per_unit  = nav_before_cf / total_units
            if net_cf_today != 0:
                total_units += net_cf_today / nav_per_unit

        roi_total = nav_per_unit - 1.0 if total_units > 0 else 0.0

        # ── 基準線報酬 ──────────────────────
        curr_0050   = df_0050[df_0050['date'] <= d]['close'].iloc[-1]   if not df_0050[df_0050['date'] <= d].empty   else p0_0050
        curr_00631L = df_00631L[df_00631L['date'] <= d]['close'].iloc[-1] if not df_00631L[df_00631L['date'] <= d].empty else p0_00631L
        roi_0050    = (curr_0050   - p0_0050)   / p0_0050
        roi_00631L  = (curr_00631L - p0_00631L) / p0_00631L

        history.append({
            'date': d, 'principal': principal, 'current_cash': current_cash,
            'total_nav': total_nav, 'roi_total': roi_total,
            'roi_ai': roi_ai, 'roi_oth': roi_oth,
            'roi_0050': roi_0050, 'roi_00631L': roi_00631L
        })

    df_res = pd.DataFrame(history)
    if not df_res.empty:
        df_res['max_nav'] = df_res['total_nav'].cummax()
        df_res['dd_pct']  = (df_res['total_nav'] - df_res['max_nav']) / df_res['max_nav']
        df_res['dd_twd']  = df_res['total_nav'] - df_res['max_nav']
    return df_res


# ──────────────────────────────────────────
# 模組 3：資產結算引擎 (供各頁面共用)
# ──────────────────────────────────────────
@st.cache_data
def calculate_settlement_data(df_record, df_price):
    if df_record.empty:
        return [], [], 0, 0

    df_trade  = df_record[df_record['id'] != 'cash'].copy()
    latest_px = df_price.groupby('id').tail(1).set_index('id')['close'].to_dict() if not df_price.empty else {}

    unrealized_data, realized_data = [], []

    for aid in df_trade['id'].unique():
        sub        = df_trade[df_trade['id'] == aid].sort_values('date')
        cum_shares = 0
        total_cost = 0
        shares_sold= 0
        revenue    = 0
        cum_buy_fee  = 0
        cum_sell_fee = 0
        raw_sell_val = 0
        stock_name   = sub['name'].iloc[0] if 'name' in sub.columns else ""

        for _, row in sub.iterrows():
            trade_val = row['price'] * row['share']
            if row['action'] == 'buy':
                fee           = int(trade_val * 0.001425)
                cum_buy_fee  += fee
                cum_shares   += row['share']
                total_cost   += trade_val + fee
            elif row['action'] == 'sell':
                fee           = int(trade_val * 0.001425) + int(trade_val * 0.003)
                cum_sell_fee += fee
                shares_sold  += row['share']
                raw_sell_val += trade_val
                revenue      += trade_val - fee

        curr_shares  = cum_shares - shares_sold
        avg_cost     = total_cost / cum_shares if cum_shares > 0 else 0
        avg_buy_fee  = cum_buy_fee / cum_shares if cum_shares > 0 else 0
        yahoo_url    = f"https://tw.stock.yahoo.com/quote/{aid}"

        # ── 已實現部分 ──────────────────────
        if shares_sold > 0:
            sell_cost            = shares_sold * avg_cost
            r_pnl                = revenue - sell_cost
            r_roi_pct            = (r_pnl / sell_cost) * 100 if sell_cost > 0 else 0.0
            avg_sell_px          = raw_sell_val / shares_sold
            realized_total_fee   = (shares_sold * avg_buy_fee) + cum_sell_fee
            realized_data.append({
                "代號": yahoo_url,
                "名稱": stock_name,
                "累積報酬率": f"{r_roi_pct:.1f}%",
                "已實現損益": round(r_pnl, 0),
                "平均成本":   round(avg_cost, 1),
                "賣出均價":   f"{avg_sell_px:.1f}",
                "買賣手續費": int(realized_total_fee),
                "賣出股數":   int(shares_sold),
                "回收現金":   round(revenue, 0)
            })

        # ── 未實現部分 ──────────────────────
        if curr_shares > 0:
            px             = latest_px.get(aid, 0.0)
            mv             = curr_shares * px
            est_sell_fee   = int(mv * 0.001425) + int(mv * 0.003)
            u_pnl          = mv - (curr_shares * avg_cost) - est_sell_fee
            roi_pct        = (u_pnl / (curr_shares * avg_cost)) * 100 if avg_cost > 0 else 0.0
            unrealized_buy_fee = curr_shares * avg_buy_fee
            unrealized_data.append({
                "代號": yahoo_url,
                "名稱": stock_name,
                "累積報酬率": f"{roi_pct:.1f}%",
                "未實現損益": round(u_pnl, 0),
                "平均成本":   round(avg_cost, 1),
                "最新市價":   f"{px:.1f}",
                "買入手續費": int(unrealized_buy_fee),
                "買入股數":   int(curr_shares),
                "當前市值":   round(mv, 0),
            })

    total_unrealized = sum(item["未實現損益"] for item in unrealized_data)
    total_realized   = sum(item["已實現損益"] for item in realized_data)
    return unrealized_data, realized_data, total_unrealized, total_realized


# ──────────────────────────────────────────
# 模組 4：警報判斷 (純邏輯，回傳字串清單)
# ──────────────────────────────────────────
def check_rebalance_alerts(df_record, df_price):
    """個股權重 > 25% 時，計算應賣出股數 (削回20%)"""
    if df_record.empty or df_price.empty:
        return []

    df_trade  = df_record[df_record['id'] != 'cash'].copy()
    latest_px = df_price.groupby('id').tail(1).set_index('id')['close'].to_dict()

    holdings      = {}
    pure_equity_mv= 0.0

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
                    f"(目前{weight*100:.1f}%)，需賣出市值 {sell_value:,.0f}"
                    f"(約需賣出 {sell_shares} 股)"
                )
    return alerts


def check_market_volatility_alerts(df_0050, threshold=0.30):
    """0050 20日年化波動率 > threshold 時觸發警報"""
    if df_0050.empty or 'realized_vol' not in df_0050.columns:
        return []
    latest_vol = df_0050['realized_vol'].iloc[-1]
    if pd.isna(latest_vol):
        return []
    if latest_vol > threshold:
        return [f"【指令 : 熔毀 】 大盤波動率超過{threshold*100:.0f}% (目前:{latest_vol*100:.1f}%)，需砍倉1/2"]
    return []


def check_stoploss_alerts(unrealized_data, threshold=-20.0):
    """-20% 停損警報"""
    alerts = []
    for item in unrealized_data:
        roi_float = float(item["累積報酬率"].replace("%", ""))
        if roi_float <= threshold:
            alerts.append(
                f"【指令 : 滅止 】{item['代號']} {item['名稱']} 已觸發 -20% 停損保險機制"
            )
    return alerts