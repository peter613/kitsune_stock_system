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
from datetime import datetime

START_DATE = '2026-03-11'

# ──────────────────────────────────────────
# 輔助：策略月份標籤（每月 11 日或順延的第一個交易日為起點）
# ──────────────────────────────────────────
def assign_strategy_month(dates_series: pd.Series, trading_dates: pd.Index) -> pd.Series:
    """
    將日期序列對應到「策略月份」。
    策略月份定義：每個自然月中，第一個 >= 11 日的交易日為該月起始。
    若當月無 >= 11 日的交易日，以當月最後一個交易日為起始。

    Parameters
    ----------
    dates_series   : 要標記的日期 Series (datetime64)
    trading_dates  : 所有交易日的 DatetimeIndex（用於找到每月實際入場日）

    Returns
    -------
    pd.Series[str]  形如 '2026-03' 的月份標籤
    """
    trading_dates = pd.DatetimeIndex(trading_dates).normalize()

    # 建立每個自然年月 → 實際入場日的對應表
    entry_map: dict[str, pd.Timestamp] = {}
    for ym, grp in pd.Series(trading_dates).groupby(trading_dates.to_period('M')):
        grp_sorted = grp.sort_values()
        candidates = grp_sorted[grp_sorted.dt.day >= 11]
        entry_map[str(ym)] = candidates.iloc[0] if not candidates.empty else grp_sorted.iloc[-1]

    # 依入場日排序，建成切割點列表
    boundaries = sorted(entry_map.values())

    def _label(d):
        # 找到最後一個 <= d 的 boundary
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
    file_path_record = config.JOURNAL_DIR / "journal.csv"

    # ── 交易紀錄 ──────────────────────────
    df_record = pd.read_csv(file_path_record, dtype={'id': str})
    # 強制標準化標籤，確保 'div' 判斷一致
    df_record['label'] = df_record['label'].str.strip().str.lower()
    df_record.columns = df_record.columns.str.strip().str.lower()
    df_record['date'] = pd.to_datetime(df_record['date'], errors='coerce').dt.normalize()
    df_record = df_record[df_record['date'] >= START_DATE].copy()
    df_record['price'] = pd.to_numeric(df_record['price'], errors='coerce').fillna(0)
    df_record['share'] = pd.to_numeric(df_record['share'], errors='coerce').fillna(0)
    df_record['cash_flow'] = np.where(
        df_record['action'] == 'in',  df_record['price'],
        np.where(df_record['action'] == 'out', -df_record['price'], 0)
    )

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
        record_ids = set(df_record[df_record['id'] != 'cash']['id'].unique())
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
        df['daily_return']  = df['close'].pct_change()
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
    if df_record.empty: return pd.DataFrame()

    dates = sorted(df_price['date'].dropna().unique())
    if not dates: dates = sorted(df_record['date'].unique())

    p0_0050   = df_0050['close'].iloc[0]   if not df_0050.empty   else 1.0
    p0_00631L = df_00631L['close'].iloc[0] if not df_00631L.empty else 1.0

    # ── 總組合 NAV ──
    total_units  = 0.0
    nav_per_unit = 1.0

    # ── 子組合 NAV 初始化 ──
    sub_units  = {'ai': 0.0, 'etf': 0.0, 'other': 0.0}
    sub_nav_pu = {'ai': 1.0, 'etf': 1.0, 'other': 1.0}

    history = []

    for d in dates:
        rec_d = df_record[df_record['date'] <= d]
        if rec_d.empty: continue

        px_d = df_price[df_price['date'] <= d].groupby('id').tail(1).set_index('id')['close'].to_dict()

        # ── 現金流處理 ──
        cash_rec     = rec_d[rec_d['id'] == 'cash']
        principal    = cash_rec['cash_flow'].sum()
        net_cf_today = cash_rec[cash_rec['date'] == d]['cash_flow'].sum()
        div_cash     = cash_rec[cash_rec['label'] == 'div']['cash_flow'].sum()
        div_today    = cash_rec[(cash_rec['label'] == 'div') & (cash_rec['date'] == d)]['cash_flow'].sum()

        trades        = rec_d[rec_d['id'] != 'cash'].copy()
        trades['val'] = trades['price'] * trades['share']

        # ── 總組合市值 ──
        def _calc_mv(sub):
            mv = 0.0
            for aid in sub['id'].unique():
                a      = sub[sub['id'] == aid]
                shares = a[a['action'] == 'buy']['share'].sum() - a[a['action'] == 'sell']['share'].sum()
                if shares > 0:
                    raw     = shares * px_d.get(aid, 0.0)
                    tax     = 0.001 if str(aid).startswith('00') else 0.003
                    mv     += raw - int(raw * 0.001425) - int(raw * tax)
            return mv

        def _calc_flow(sub):
            buy_mask  = sub['action'] == 'buy'
            sell_mask = sub['action'] == 'sell'
            buy_fees  = (sub[buy_mask]['val'] * 0.001425).astype(int)
            # 統一使用 id 前綴判斷稅率，與 _calc_mv 一致
            sell_tax  = sub[sell_mask]['id'].apply(lambda x: 0.001 if str(x).startswith('00') else 0.003)
            sell_fees = (sub[sell_mask]['val'] * 0.001425).astype(int) + (sub[sell_mask]['val'] * sell_tax).astype(int)
            return (sub[sell_mask]['val'].sum() - sell_fees.sum()) - (sub[buy_mask]['val'].sum() + buy_fees.sum())

        flow_tot     = _calc_flow(trades)
        mv_tot       = _calc_mv(trades)
        current_cash = principal + flow_tot
        total_nav    = current_cash + mv_tot

        # ── 總組合 NAV 更新 ──
        if total_units == 0.0 and net_cf_today > 0:
            nav_per_unit = 1.0
            total_units  = net_cf_today / nav_per_unit
        elif total_units > 0:
            nav_before_cf = total_nav - net_cf_today
            nav_per_unit  = nav_before_cf / total_units
            if net_cf_today != 0:
                total_units += net_cf_today / nav_per_unit

        roi_total = nav_per_unit - 1.0 if total_units > 0 else 0.0

        # ── 子組合 NAV 更新 ──
        # ai / other：純資本利得，不含配息
        # etf：含當日股利（配息計入收益但不增加 units）
        sub_roi = {}

        def _calc_mv_for_holdings(sub_p):
            """計算給定持倉在今日報價下的市值（含手續費/稅）"""
            mv = 0.0
            for aid in sub_p['id'].unique():
                a = sub_p[sub_p['id'] == aid]
                shares_held = a[a['action'] == 'buy']['share'].sum() - a[a['action'] == 'sell']['share'].sum()
                if shares_held > 0:
                    raw = shares_held * px_d.get(aid, 0.0)
                    tax = 0.001 if str(aid).startswith('00') else 0.003
                    mv += raw - int(raw * 0.001425) - int(raw * tax)
            return mv

        for label in ['ai', 'etf', 'other']:
            sub = trades[trades['label'] == label]

            # 當日該子組合的買賣現金流
            sub_today  = sub[sub['date'] == d]
            buy_today  = sub_today[sub_today['action'] == 'buy']
            sell_today = sub_today[sub_today['action'] == 'sell']

            buy_val  = buy_today['val'].sum()
            sell_val = sell_today['val'].sum()
            buy_fee  = int(buy_val * 0.001425)
            # 統一用 id 前綴判斷稅率
            if not sell_today.empty:
                sell_tax_vec = sell_today['id'].apply(lambda x: 0.001 if str(x).startswith('00') else 0.003)
                sell_fee = int(sell_val * 0.001425) + int((sell_today['val'] * sell_tax_vec).sum())
            else:
                sell_fee = 0

            # net_cf_sub > 0：今日有淨買入（資金流入子組合）
            net_cf_sub = (buy_val + buy_fee) - (sell_val - sell_fee)

            # ETF 加計當日股利（不增加 units，純收益）
            div_sub = div_today if label == 'etf' else 0.0

            # 今日交易前的持倉（截至昨日）× 今日報價 = 今日開盤前的子組合淨值
            sub_prev   = trades[(trades['label'] == label) & (trades['date'] < d)]
            nav_before = _calc_mv_for_holdings(sub_prev) + div_sub

            if sub_units[label] == 0.0 and net_cf_sub > 0:
                # 首次買入：初始化
                sub_nav_pu[label] = 1.0
                sub_units[label]  = net_cf_sub / sub_nav_pu[label]
            elif sub_units[label] > 0:
                # 用昨日持倉 × 今日報價還原每單位淨值（正確還原，不依賴反推）
                sub_nav_pu[label] = nav_before / sub_units[label]

                # 今日淨買入更新 units
                if net_cf_sub != 0:
                    sub_units[label] += net_cf_sub / sub_nav_pu[label]

            sub_roi[label] = sub_nav_pu[label] - 1.0 if sub_units[label] > 0 else 0.0

        # ── 基準指數 ──
        curr_0050   = df_0050[df_0050['date'] <= d]['close'].iloc[-1]   if not df_0050[df_0050['date'] <= d].empty   else p0_0050
        curr_00631L = df_00631L[df_00631L['date'] <= d]['close'].iloc[-1] if not df_00631L[df_00631L['date'] <= d].empty else p0_00631L
        roi_0050    = (curr_0050 - p0_0050) / p0_0050
        roi_00631L  = (curr_00631L - p0_00631L) / p0_00631L

        history.append({
            'date': d, 'principal': principal, 'current_cash': current_cash,
            'div_cash': div_cash,
            'total_nav': total_nav, 'roi_total': roi_total,
            'roi_ai':  sub_roi['ai'],
            'roi_etf': sub_roi['etf'],
            'roi_oth': sub_roi['other'],
            'roi_0050': roi_0050, 'roi_00631L': roi_00631L
        })

    df_res = pd.DataFrame(history)
    if not df_res.empty:
        df_res['max_nav'] = df_res['total_nav'].cummax()
        df_res['dd_pct']  = (df_res['total_nav'] - df_res['max_nav']) / df_res['max_nav']
        df_res['dd_twd']  = df_res['total_nav'] - df_res['max_nav']
        # 附加策略月份標籤（以實際交易日 >= 11 號為月份起始）
        df_res['strategy_month'] = assign_strategy_month(df_res['date'], df_res['date'])

    return df_res

# ──────────────────────────────────────────
# 模組 3：資產結算引擎 (增加 total_div 回傳)
# ──────────────────────────────────────────
@st.cache_data
def calculate_settlement_data(df_record, df_price):
    if df_record.empty: return [], [], 0, 0, 0

    df_trade  = df_record[df_record['id'] != 'cash'].copy()
    latest_px = df_price.groupby('id').tail(1).set_index('id')['close'].to_dict() if not df_price.empty else {}
    unrealized_data, realized_data = [], []

    for aid in df_trade['id'].unique():
        sub        = df_trade[df_trade['id'] == aid].sort_values('date')
        stock_name = sub['name'].iloc[0] if 'name' in sub.columns else ""
        tax_rate   = 0.001 if str(aid).startswith('00') else 0.003
        yahoo_url  = f"https://tw.stock.yahoo.com/quote/{aid}/institutional-trading"

        # ── 逐筆走訪，賣出時快照當下 avg_cost（移動加權平均成本法）──
        # 這樣攤平買入只影響「剩餘持倉」的成本，不回溯修改已結算的損益
        cum_shares  = 0.0   # 目前持有股數
        total_cost  = 0.0   # 目前持倉總成本（含手續費）
        total_r_pnl = 0.0   # 累積已實現損益
        total_revenue = 0.0 # 累積回收現金
        total_sell_cost = 0.0  # 累積已實現成本（用於整體報酬率計算）
        last_sell_date  = None

        for _, row in sub.iterrows():
            trade_val = row['price'] * row['share']
            if row['action'] == 'buy':
                fee         = int(trade_val * 0.001425)
                cum_shares += row['share']
                total_cost += trade_val + fee

            elif row['action'] == 'sell':
                fee = int(trade_val * 0.001425) + int(trade_val * tax_rate)
                net_revenue = trade_val - fee

                # 賣出時快照此刻的平均成本（攤平後的正確值）
                avg_cost_now = total_cost / cum_shares if cum_shares > 0 else 0
                sell_cost    = row['share'] * avg_cost_now

                total_r_pnl    += net_revenue - sell_cost
                total_revenue  += net_revenue
                total_sell_cost+= sell_cost

                # 從持倉扣除：股數與對應成本
                cum_shares -= row['share']
                total_cost -= sell_cost
                last_sell_date = row['date']

        # ── 已實現損益 ──
        shares_sold = sub[sub['action'] == 'sell']['share'].sum()
        if shares_sold > 0 and last_sell_date is not None:
            realized_data.append({
                "日期": last_sell_date.strftime("%Y-%m-%d"),
                "代號": yahoo_url, "名稱": stock_name,
                "累積報酬率": f"{(total_r_pnl / total_sell_cost) * 100:.1f}%" if total_sell_cost > 0 else "0%",
                "已實現損益": round(total_r_pnl, 0),
                "回收現金":   round(total_revenue, 0)
            })

        # ── 未實現損益（剩餘持倉用剩餘成本，avg_cost 是攤平後的正確值）──
        if cum_shares > 0:
            avg_cost_remain = total_cost / cum_shares if cum_shares > 0 else 0
            px   = latest_px.get(aid, 0.0)
            mv   = cum_shares * px
            est_sell_fee = int(mv * 0.001425) + int(mv * tax_rate)
            u_pnl = mv - total_cost - est_sell_fee
            unrealized_data.append({
                "代號": yahoo_url, "名稱": stock_name,
                "累積報酬率": f"{(u_pnl / total_cost) * 100:.1f}%" if total_cost > 0 else "0%",
                "未實現損益": round(u_pnl, 0), "當前市值": round(mv, 0)
            })

    total_unrealized = sum(item["未實現損益"] for item in unrealized_data)
    total_realized   = sum(item["已實現損益"] for item in realized_data)
    total_div = df_record[(df_record['id'] == 'cash') & (df_record['label'] == 'div')]['cash_flow'].sum()

    return unrealized_data, realized_data, total_unrealized, total_realized, total_div


# ──────────────────────────────────────────
# 模組 4：警報判斷 (純邏輯，回傳字串清單)
# ──────────────────────────────────────────
def check_rebalance_alerts(df_record, df_price):
    """個股權重 > 25% 時，計算應賣出股數 (削回20%)"""
    if df_record.empty or df_price.empty:
        return []

    df_trade  = df_record[(df_record['id'] != 'cash') & (df_record['id'].str.startswith('00') == False)].copy()
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

def check_bear_market_alerts(df_0050, threshold=-0.20):
    """0050 從最高點回撤 >= 20% 進入熊市警報"""
    if df_0050.empty or 'close' not in df_0050.columns:
        return []
        
    # 找出區間最高價與最新收盤價
    max_price = df_0050['close'].max()
    latest_price = df_0050['close'].iloc[-1]
    
    if max_price == 0:
        return []
        
    # 計算最大回撤 (Drawdown)
    drawdown = (latest_price - max_price) / max_price
    
    if drawdown <= threshold:
        return [f"【指令 : 凜冬 】 大盤已從高點回撤 {drawdown*100:.1f}%，正式進入熊市，全面砍倉。"]
    
    return []

def check_etf_balance_day():
    """etf季底調整"""
    if datetime.now().month in [3, 6, 9, 12] and datetime.now().day >= 24:
        return [f"【指令 : 重構 】ETF與整體投組季底調整"]
    return []