"""
engine.py
Kitsune Kernel — 計算引擎層
職責：資料讀取、時間序列計算、資產結算、警報判斷
規則：本檔案不得引入任何 streamlit 渲染 (st.write / st.plotly_chart …)
      唯一例外：load_data 內的 st.error 防呆提示保留

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
START_DATE = '2026-03-11'
BASE_NAV   = 10.0   # 基金初始單位淨值

import os
import numpy as np
import pandas as pd
import streamlit as st
import config
from datetime import datetime
# ──────────────────────────────────────────
# 輔助：策略月份標籤（每月 11 日或順延的第一個交易日為起點）
# ──────────────────────────────────────────
def assign_strategy_month(dates_series: pd.Series, trading_dates: pd.Index) -> pd.Series:
    """
    將日期序列對應到「策略月份」。
    策略月份定義：每個自然月中，第一個 >= 11 日的交易日為該月起始。
    若當月無 >= 11 日的交易日，以當月最後一個交易日為起始。
    """
    trading_dates = pd.DatetimeIndex(trading_dates).normalize()

    entry_map: dict[str, pd.Timestamp] = {}
    for ym, grp in pd.Series(trading_dates).groupby(trading_dates.to_period('M')):
        grp_sorted = grp.sort_values()
        candidates = grp_sorted[grp_sorted.dt.day >= 11]
        entry_map[str(ym)] = candidates.iloc[0] if not candidates.empty else grp_sorted.iloc[-1]

    boundaries = sorted(entry_map.values())

    def _label(d):
        idx = pd.Series(boundaries).searchsorted(d, side='right') - 1
        if idx < 0:
            return boundaries[0].strftime('%Y-%m')
        return boundaries[idx].strftime('%Y-%m')

    return dates_series.apply(_label)


# ──────────────────────────────────────────
# 模組 1：資料讀取
# ──────────────────────────────────────────
@st.cache_data
def load_data():
    file_path_journal     = config.JOURNAL_DIR     / "journal.csv"
    file_path_transaction = config.KITSUNE_FUND_ADMINISTRATION_DIR / "transaction.csv"

    # ── 股票交易紀錄（journal.csv）──
    df_record = pd.read_csv(file_path_journal, dtype={'id': str})
    df_record.columns  = df_record.columns.str.strip().str.lower()
    df_record['label'] = df_record['label'].str.strip().str.lower()
    df_record['date']  = pd.to_datetime(df_record['date'], errors='coerce').dt.normalize()
    df_record = df_record[df_record['date'] >= START_DATE].copy()
    df_record['price'] = pd.to_numeric(df_record['price'], errors='coerce').fillna(0)
    df_record['share'] = pd.to_numeric(df_record['share'], errors='coerce').fillna(0)

    # ── 申購/贖回紀錄（transaction.csv）──
    df_flows = pd.read_csv(file_path_transaction, dtype={'id': str})
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

    market_path = config.DATABASE_DIR / "market.parquet"
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
        path = config.DATABASE_DIR / f'{name}.csv'
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, dtype={'id': str})
        df = _standardize(df)
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
        df['daily_return'] = df['close'].pct_change()
        df['realized_vol'] = df['daily_return'].rolling(window=20).std() * (252 ** 0.5)
        return df[df['date'] >= START_DATE].copy()

    df_0050   = _load_bench("0050")
    df_00631L = _load_bench("00631")

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
# 模組 2：時間序列計算引擎 (純 NAV 單位法)
# ──────────────────────────────────────────
@st.cache_data
def calculate_daily_history(df_record, df_flows, df_price, df_0050, df_00631L):
    """
    唯一事實來源：total_nav = cash + mv
    NAV 漲跌只反映投資績效；申購/贖回只改變 units，不影響 nav_per_unit。
    """
    if df_record.empty and df_flows.empty:
        return pd.DataFrame()

    dates = sorted(df_price['date'].dropna().unique())
    if not dates:
        dates = sorted(
            set(df_record['date'].unique()) | set(df_flows['date'].unique())
        )

    p0_0050   = df_0050['close'].iloc[0]   if not df_0050.empty   else 1.0
    p0_00631L = df_00631L['close'].iloc[0] if not df_00631L.empty else 1.0

    # ── 狀態變數 ──────────────────────────────────────────
    cash         = 0.0
    total_units  = 0.0
    nav_per_unit = BASE_NAV

    sub_cash    = {'ai': 0.0,  'other': 0.0}
    sub_units   = {'ai': 0.0,  'other': 0.0}
    sub_nav_pu  = {'ai': BASE_NAV,  'other': BASE_NAV}

    history = []

    for d in dates:
        # ── 當日最新報價 ──────────────────────────────────
        px_d = (df_price[df_price['date'] <= d]
                .groupby('id').tail(1)
                .set_index('id')['close']
                .to_dict())

        # ── 截至今日的所有股票交易（用於算持倉市值）────────
        trades_all = df_record[df_record['date'] <= d].copy()
        trades_today = df_record[df_record['date'] == d].copy()

        # ── 步驟 1：今日買賣影響現金 ─────────────────────
        for _, row in trades_today.iterrows():
            if row['action'] == 'buy':
                cost = _buy_cost(row['price'], row['share'], row['id'])
                cash -= cost
                sub_cash[row['label']] -= cost
            elif row['action'] == 'sell':
                proceeds = _sell_proceeds(row['price'], row['share'], row['id'])
                cash += proceeds
                sub_cash[row['label']] += proceeds

        # ── 步驟 2：計算持倉市值 ──────────────────────────
        def _mv(subset: pd.DataFrame) -> float:
            mv = 0.0
            for aid in subset['id'].unique():
                a = subset[subset['id'] == aid]
                shares = (a[a['action'] == 'buy']['share'].sum()
                        - a[a['action'] == 'sell']['share'].sum())
                if shares > 0:
                    raw = shares * px_d.get(aid, 0.0)
                    mv += raw - int(raw * 0.001425) - int(raw * _tax_rate(aid))
            return mv

        mv_total = _mv(trades_all)
        total_nav = cash + mv_total

        # ── 步驟 3：申購前更新 nav_per_unit ──────────────
        if total_units > 0:
            nav_per_unit = total_nav / total_units

        # ── 步驟 4：處理今日申購/贖回 ────────────────────
        flows_today = df_flows[df_flows['date'] == d]

        for _, row in flows_today.iterrows():
            if row['action'] == 'subscribe':
                amount = row['price']
                if total_units == 0.0:
                    nav_per_unit = BASE_NAV
                new_units    = amount / nav_per_unit
                total_units += new_units
                cash        += amount
                total_nav   += amount

                # 子組合：申購金額按比例分配到 other（預設），或可依 label 欄位
                label = row['label'] if pd.notna(row.get('label')) and row.get('label') in sub_cash else 'other'
                sub_cash[label]  += amount
                if sub_units[label] == 0.0:
                    sub_nav_pu[label] = BASE_NAV
                sub_units[label] += amount / sub_nav_pu[label]

            elif row['action'] == 'redeem':
                units_out   = row['share']
                cash_out    = units_out * nav_per_unit
                total_units -= units_out
                cash        -= cash_out
                total_nav   -= cash_out

                label = row['label'] if pd.notna(row.get('label')) and row.get('label') in sub_cash else 'other'
                sub_units[label] -= units_out
                sub_cash[label]  -= cash_out

        # ── 步驟 5：子組合 NAV 更新 ───────────────────────
        sub_roi = {}
        for label in ['ai', 'other']:
            sub_trades = trades_all[trades_all['label'] == label]
            mv_sub = _mv(sub_trades)
            sub_total_nav = sub_cash[label] + mv_sub

            if sub_units[label] > 0:
                sub_nav_pu[label] = sub_total_nav / sub_units[label]
                sub_roi[label]    = (sub_nav_pu[label] / BASE_NAV) - 1.0
            else:
                sub_roi[label] = 0.0

        # ── 步驟 6：整體報酬率 ────────────────────────────
        roi_total = (nav_per_unit / BASE_NAV) - 1.0 if total_units > 0 else 0.0

        # ── 基準指數 ──────────────────────────────────────
        curr_0050   = df_0050[df_0050['date'] <= d]['close'].iloc[-1]   if not df_0050[df_0050['date'] <= d].empty   else p0_0050
        curr_00631L = df_00631L[df_00631L['date'] <= d]['close'].iloc[-1] if not df_00631L[df_00631L['date'] <= d].empty else p0_00631L
        roi_0050    = (curr_0050 - p0_0050) / p0_0050
        roi_00631L  = (curr_00631L - p0_00631L) / p0_00631L

        history.append({
            'date':         d,
            'cash':         cash,
            'total_nav':    total_nav,
            'nav_per_unit': nav_per_unit,
            'total_units':  total_units,
            'roi_total':    roi_total,
            'roi_ai':       sub_roi['ai'],
            'roi_oth':      sub_roi['other'],
            'roi_0050':     roi_0050,
            'roi_00631L':   roi_00631L,
        })

    df_res = pd.DataFrame(history)
    if not df_res.empty:
        df_res['max_nav'] = df_res['total_nav'].cummax()
        df_res['dd_pct']  = (df_res['total_nav'] - df_res['max_nav']) / df_res['max_nav']
        df_res['dd_twd']  = df_res['total_nav'] - df_res['max_nav']
        df_res['strategy_month'] = assign_strategy_month(df_res['date'], df_res['date'])

    return df_res


# ──────────────────────────────────────────
# 模組 3：資產結算引擎（不變）
# ──────────────────────────────────────────
@st.cache_data
def calculate_settlement_data(df_record, df_price):
    if df_record.empty: return [], [], 0, 0

    df_trade  = df_record.copy()
    latest_px = df_price.groupby('id').tail(1).set_index('id')['close'].to_dict() if not df_price.empty else {}
    unrealized_data, realized_data = [], []

    for aid in df_trade['id'].unique():
        sub        = df_trade[df_trade['id'] == aid].sort_values('date')
        stock_name = sub['name'].iloc[0] if 'name' in sub.columns else ""
        tax_rate   = _tax_rate(aid)
        yahoo_url  = f"https://tw.stock.yahoo.com/quote/{aid}/institutional-trading"

        cum_shares       = 0.0
        total_cost       = 0.0
        total_r_pnl      = 0.0
        total_revenue    = 0.0
        total_sell_cost  = 0.0
        last_sell_date   = None

        for _, row in sub.iterrows():
            trade_val = row['price'] * row['share']
            if row['action'] == 'buy':
                fee         = int(trade_val * 0.001425)
                cum_shares += row['share']
                total_cost += trade_val + fee

            elif row['action'] == 'sell':
                fee         = int(trade_val * 0.001425) + int(trade_val * tax_rate)
                net_revenue = trade_val - fee

                avg_cost_now  = total_cost / cum_shares if cum_shares > 0 else 0
                sell_cost     = row['share'] * avg_cost_now

                total_r_pnl   += net_revenue - sell_cost
                total_revenue += net_revenue
                total_sell_cost += sell_cost

                cum_shares    -= row['share']
                total_cost    -= sell_cost
                last_sell_date = row['date']

        shares_sold = sub[sub['action'] == 'sell']['share'].sum()
        if shares_sold > 0 and last_sell_date is not None:
            realized_data.append({
                "日期":     last_sell_date.strftime("%Y-%m-%d"),
                "代號":     yahoo_url,
                "名稱":     stock_name,
                "累積報酬率": f"{(total_r_pnl / total_sell_cost) * 100:.1f}%" if total_sell_cost > 0 else "0%",
                "已實現損益": round(total_r_pnl, 0),
                "回收現金":   round(total_revenue, 0),
            })

        if cum_shares > 0:
            px           = latest_px.get(aid, 0.0)
            mv           = cum_shares * px
            est_sell_fee = int(mv * 0.001425) + int(mv * tax_rate)
            u_pnl        = mv - total_cost - est_sell_fee
            unrealized_data.append({
                "代號":     yahoo_url,
                "名稱":     stock_name,
                "累積報酬率": f"{(u_pnl / total_cost) * 100:.1f}%" if total_cost > 0 else "0%",
                "未實現損益": round(u_pnl, 0),
                "當前市值":   round(mv, 0),
            })

    total_unrealized = sum(item["未實現損益"] for item in unrealized_data)
    total_realized   = sum(item["已實現損益"] for item in realized_data)

    return unrealized_data, realized_data, total_unrealized, total_realized


# ──────────────────────────────────────────
# 模組 4：警報判斷
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


# def check_market_volatility_alerts(df_0050, threshold=0.30):
#     if df_0050.empty or 'realized_vol' not in df_0050.columns:
#         return []
#     latest_vol = df_0050['realized_vol'].iloc[-1]
#     if pd.isna(latest_vol):
#         return []
#     if latest_vol > threshold:
#         return [f"【指令 : 熔毀 】 大盤波動率超過{threshold*100:.0f}% (目前:{latest_vol*100:.1f}%)，需砍倉1/2"]
#     return []


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

# ── 統整alert ──
def generate_system_alerts(df_record, df_price, df_0050, unrealized_data):
    """
    統一收集並回傳所有風控警報。
    排列順序：總體經濟 (大盤) -> 個體經濟 (持股)
    """
    alerts = []
    # 1. 宏觀風險 (優先顯示)
    alerts.extend(_check_bear_market_alerts(df_0050))
    # alerts.extend(_check_market_volatility_alerts(df_0050))
    
    # 2. 微觀風險
    alerts.extend(_check_stoploss_alerts(unrealized_data))
    alerts.extend(_check_rebalance_alerts(df_record, df_price))
    
    return alerts

# ──────────────────────────────────────────
# 投資者
# ──────────────────────────────────────────
"""
transaction.csv 格式規範：
  申購：      action=subscribe,    price=申購金額(TWD),  share=空白,       subscriber=PETER/LYNN
  贖回(股數)：action=redeem,       price=空白,           share=贖回單位數, subscriber=PETER/LYNN
  贖回(金額)：action=redeem_cash,  price=贖回金額(TWD),  share=空白,       subscriber=PETER/LYNN
"""
# ──────────────────────────────────────────
# FundSubscriber — 單一投資者
# ──────────────────────────────────────────
class FundSubscriber:
    def __init__(self, name: str):
        self.name = name
 
    def calculate_equity(
        self,
        df_history: pd.DataFrame,
        df_flows: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        回傳欄位：
            date / nav_per_unit / units / equity / roi /
            capital / total_redeemed / net_cashflow
        """
        if df_history.empty or 'nav_per_unit' not in df_history.columns:
            return pd.DataFrame()
 
        my_flows = df_flows[
            df_flows['subscriber'].str.upper() == self.name.upper()
        ].sort_values('date').copy()
 
        if my_flows.empty:
            return pd.DataFrame()
 
        result_rows   = []
        units         = 0.0
        total_cost    = 0.0   # 累積投入現金
        total_redeemed = 0.0  # 累積提領現金
 
        for d_ts, day_hist in df_history.set_index('date').iterrows():
            nav = day_hist['nav_per_unit']
 
            day_flows = my_flows[my_flows['date'] == d_ts]
            for _, row in day_flows.iterrows():
 
                if row['action'] == 'subscribe':
                    amount      = float(row['price'])
                    new_units   = amount / nav if nav > 0 else 0.0
                    units      += new_units
                    total_cost += amount
 
                elif row['action'] == 'redeem':
                    # 填單位數
                    redeem_units = float(row['share'])
                    cash_out     = redeem_units * nav
                    if units > 0:
                        total_cost -= (redeem_units / units) * total_cost
                    units          = max(0.0, units - redeem_units)
                    total_redeemed += cash_out
 
                elif row['action'] == 'redeem_cash':
                    # 填金額，程式算單位數
                    cash_out     = float(row['price'])
                    redeem_units = cash_out / nav if nav > 0 else 0.0
                    if units > 0:
                        total_cost -= (redeem_units / units) * total_cost
                    units          = max(0.0, units - redeem_units)
                    total_redeemed += cash_out
                    
                # === 處理手續費的單位數轉移 (不影響本金與基金總現金) ===
                elif row['action'] == 'fee_in':
                    units += float(row['share'])
                elif row['action'] == 'fee_out':
                    units = max(0.0, units - float(row['share']))
 
            if units > 0:
                equity      = units * nav
                roi         = (equity / total_cost - 1) * 100 if total_cost > 0 else 0.0
                net_cashflow = total_cost - total_redeemed  # >0 淨流入，<0 淨流出
                result_rows.append({
                    'date':           d_ts,
                    'nav_per_unit':   nav,
                    'units':          units,
                    'equity':         equity,
                    'roi':            roi,
                    'capital':        total_cost,
                    'total_redeemed': total_redeemed,
                    'net_cashflow':   net_cashflow,
                })
 
        if not result_rows:
            return pd.DataFrame()
 
        return pd.DataFrame(result_rows).reset_index(drop=True)
 
 
# ──────────────────────────────────────────
# load_subscribers
# ──────────────────────────────────────────
def load_subscribers(df_flows: pd.DataFrame) -> dict:
    if df_flows.empty or 'subscriber' not in df_flows.columns:
        return {}
    names = df_flows['subscriber'].dropna().str.upper().unique()
    return {name: FundSubscriber(name=name) for name in sorted(names)}
