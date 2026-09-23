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
    df_record = pd.read_csv(file_path_journal, dtype={'id': str}, encoding='utf-8-sig')
    df_record.columns  = df_record.columns.str.strip().str.lower()
    if 'label' in df_record.columns:
        df_record['label'] = df_record['label'].fillna('').astype(str).str.strip().str.lower()
    else:
        df_record['label'] = ''
    if 'reason' in df_record.columns:
        df_record['reason'] = df_record['reason'].fillna('').astype(str).str.strip()
    else:
        df_record['reason'] = ''
    if 'cycle' in df_record.columns:
        df_record['cycle'] = pd.to_numeric(df_record['cycle'], errors='coerce')
    df_record['date']  = pd.to_datetime(df_record['date'], errors='coerce').dt.normalize()
    df_record = df_record[df_record['date'] >= START_DATE].copy()
    df_record['price'] = pd.to_numeric(df_record['price'], errors='coerce').fillna(0)
    df_record['share'] = pd.to_numeric(df_record['share'], errors='coerce').fillna(0)

    # 補充股票名稱（優先從 white_list.csv 依 id 對照）
    wl_path = config.MARKET_DATA_DIR / "white_list.csv"
    if os.path.exists(wl_path):
        df_wl = pd.read_csv(wl_path, dtype={'id': str}, encoding='utf-8-sig')
        name_col = 'nmae' if 'nmae' in df_wl.columns else ('name' if 'name' in df_wl.columns else None)
        if name_col:
            name_map = dict(zip(df_wl['id'].astype(str).str.strip(), df_wl[name_col].astype(str).str.strip()))
            if 'name' not in df_record.columns:
                df_record['name'] = df_record['id'].astype(str).str.strip().map(name_map).fillna('')
            else:
                df_record['name'] = df_record['name'].fillna(df_record['id'].astype(str).str.strip().map(name_map)).fillna('')
    elif 'name' not in df_record.columns:
        df_record['name'] = ''

    # ── 申購/贖回紀錄（transaction.csv）──
    df_flows = pd.read_csv(file_path_transaction, dtype={'id': str}, encoding='utf-8-sig')
    df_flows.columns = df_flows.columns.str.strip().str.lower()
    df_flows['date']  = pd.to_datetime(df_flows['date'], errors='coerce').dt.normalize()
    df_flows = df_flows[df_flows['date'] >= START_DATE].copy()
    df_flows['price'] = pd.to_numeric(df_flows['price'], errors='coerce').fillna(0)
    df_flows['share'] = pd.to_numeric(df_flows['share'], errors='coerce').fillna(0)
    if 'subscriber' in df_flows.columns:
        df_flows['subscriber'] = df_flows['subscriber'].astype(str).str.strip()

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
        return df[df['date'] >= START_DATE].copy()

    df_0050   = _load_bench("0050")
    df_00631L = _load_bench("00631L")

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
    if 'date' in df_trade.columns:
        df_trade['date'] = pd.to_datetime(df_trade['date'], errors='coerce')
    has_multi_year = df_trade['date'].dt.year.nunique() > 1 if 'date' in df_trade.columns else False
    m_fmt = "'%y/%m月" if has_multi_year else "%m月"
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
        cycle_start_date = None
        entry_cycle     = None
        has_trimmed_in_cycle = False

        for _, row in sub.iterrows():
            trade_val = row['price'] * row['share']
            if row['action'] == 'buy':
                if cum_shares <= 0.0001:
                    cycle_start_date = row['date']
                    has_trimmed_in_cycle = False
                    entry_cycle = row['cycle'] if 'cycle' in row and pd.notnull(row['cycle']) else None
                fee         = int(trade_val * 0.001425)
                cum_shares += row['share']
                total_cost += trade_val + fee
                last_buy_date = row['date']

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
                
                # 當部位清空時，結算這一輪的交易
                if cum_shares <= 0.0001:
                    start_dt = cycle_start_date if cycle_start_date is not None else last_buy_date
                    m_prefix = f"{start_dt.strftime(m_fmt)} " if start_dt is not None and pd.notnull(start_dt) else ""
                    if entry_cycle is not None and pd.notnull(entry_cycle):
                        cycle_str = f"{m_prefix}(Cycle {int(entry_cycle)})" if m_prefix else f"Cycle {int(entry_cycle)}"
                    elif pd.notnull(last_buy_date):
                        cycle_str = f"{last_buy_date.strftime('%Y/%m')}"
                    else:
                        cycle_str = "未知"
                    realized_data.append({
                        "日期":       last_sell_date.strftime("%Y-%m-%d"),
                        "進場週期":   cycle_str,
                        "代號":       aid,
                        "名稱":       stock_name,
                        "累積報酬率": f"{(total_r_pnl / total_sell_cost) * 100:.1f}%" if total_sell_cost > 0 else "0%",
                        "已實現損益": round(total_r_pnl, 0),
                        "總成本":     round(total_sell_cost, 0),
                        "回收現金":   round(total_revenue, 0),
                    })
                    # 重置變數以迎接下一次買進
                    total_r_pnl     = 0.0
                    total_revenue   = 0.0
                    total_sell_cost = 0.0
                    last_sell_date  = None
                    cycle_start_date = None
                    entry_cycle     = None
                    has_trimmed_in_cycle = False
                else:
                    if str(row.get('reason', '')).strip().upper() == 'SL_HALF':
                        has_trimmed_in_cycle = True

        # 迴圈結束後，如果還有未清空的已賣出紀錄 (異常防呆)
        if total_sell_cost > 0 and last_sell_date is not None and cum_shares > 0.0001:
            start_dt = cycle_start_date if cycle_start_date is not None else last_buy_date
            m_prefix = f"{start_dt.strftime(m_fmt)} " if start_dt is not None and pd.notnull(start_dt) else ""
            if entry_cycle is not None and pd.notnull(entry_cycle):
                cycle_str = f"{m_prefix}(Cycle {int(entry_cycle)})" if m_prefix else f"Cycle {int(entry_cycle)}"
            elif pd.notnull(last_buy_date):
                cycle_str = f"{last_buy_date.strftime('%Y/%m')}"
            else:
                cycle_str = "未知"
            realized_data.append({
                "日期":       last_sell_date.strftime("%Y-%m-%d"),
                "進場週期":   cycle_str,
                "代號":       aid,
                "名稱":       stock_name,
                "累積報酬率": f"{(total_r_pnl / total_sell_cost) * 100:.1f}%" if total_sell_cost > 0 else "0%",
                "已實現損益": round(total_r_pnl, 0),
                "總成本":     round(total_sell_cost, 0),
                "回收現金":   round(total_revenue, 0),
            })
            
        realized_data = sorted(realized_data, key=lambda x: x["日期"], reverse=True)
        
        if cum_shares > 0:
            px           = latest_px.get(aid, 0.0)
            mv           = cum_shares * px
            est_sell_fee = int(mv * 0.001425) + int(mv * tax_rate)
            u_pnl        = mv - total_cost - est_sell_fee

            # 計算進場後的最高收盤價與高點回撤 (以進場成本為基準起點，最少 0.00%)
            cost_px = total_cost / cum_shares if cum_shares > 0 else px
            peak_px = cost_px
            start_date = cycle_start_date if cycle_start_date is not None else last_buy_date
            if start_date is not None and not df_price.empty:
                stock_hist = df_price[(df_price['id'] == aid) & (df_price['date'] >= start_date)]
                if not stock_hist.empty:
                    hist_max = stock_hist['close'].max()
                    if pd.notnull(hist_max) and hist_max > peak_px:
                        peak_px = hist_max
            if px > peak_px:
                peak_px = px

            roi_pct = (u_pnl / total_cost) * 100 if total_cost > 0 else 0.0
            if peak_px <= cost_px * 1.0001:
                # 從未突破成本：高點回撤即為當前虧損幅度
                dd_pct = min(0.0, roi_pct)
            else:
                # 曾突破成本創高：計算自歷史最高價回撤幅度
                dd_pct = min(0.0, (px / peak_px - 1.0) * 100) if peak_px > 0 else 0.0

            unrealized_data.append({
                "代號":       aid,#若未來要超連結改成:yahoo_url,
                "名稱":       stock_name,
                "股數":       int(cum_shares),
                "累積報酬率": f"{roi_pct:.1f}%",
                "高點回撤":   f"{dd_pct:.2f}%",
                "未實現損益": round(u_pnl, 0),
                "當前市值":   round(mv, 0),
                "has_trimmed": has_trimmed_in_cycle,
            })

    total_unrealized = sum(item["未實現損益"] for item in unrealized_data)
    total_realized   = sum(item["已實現損益"] for item in realized_data)

    return unrealized_data, realized_data, total_unrealized, total_realized


# ──────────────────────────────────────────
# 模組 3：風控警報
# ──────────────────────────────────────────
def _check_micro_stage1_stoploss_alerts(unrealized_data, threshold=-10.0, stage2_threshold=-20.0):
    alerts = []
    for item in unrealized_data:
        # 若本週期該股已執行過減碼（如已砍半），不再重複觸發 Stage 1 減碼警報
        if item.get("has_trimmed", False):
            continue

        dd_float = float(item.get("高點回撤", "0%").replace("%", ""))
        roi_float = float(item["累積報酬率"].replace("%", ""))
        max_loss = min(dd_float, roi_float)

        # 若已跌破 Stage 2 門檻 (-20%)，交由 Stage 2 警報處理，避免重複警報
        if max_loss <= stage2_threshold:
            continue

        if max_loss <= threshold:
            alerts.append(
                f"微觀個股第一階段移動減碼 ： 觸發 {abs(threshold):.0f}% 移動停損 ， {item['代號']} {item['名稱']} 砍半倉，高點回撤已達 {dd_float:.1f}%"
            )
    return alerts

def _check_micro_stage2_stoploss_alerts(unrealized_data, threshold=-20.0):
    alerts = []
    for item in unrealized_data:
        dd_float = float(item.get("高點回撤", "0%").replace("%", ""))
        roi_float = float(item["累積報酬率"].replace("%", ""))
        max_loss = min(dd_float, roi_float)

        if max_loss <= threshold:
            alerts.append(
                f"微觀個股第二階段移動停損 ： 觸發 {abs(threshold):.0f}% 移動停損 ， {item['代號']} {item['名稱']} 砍全倉，高點回撤已達 {dd_float:.1f}%"
            )
    return alerts

def get_macro_cycle_stats(df_record: pd.DataFrame, df_history: pd.DataFrame) -> dict:
    """
    直接以最新 cycle 計算當前週期的宏觀指標 (高點回撤, 累積報酬, 進場日, 是否已1/4停利)
    """
    default_stats = {"macro_dd": 0.0, "macro_roi": 0.0, "entry_date_str": "", "has_tp": False}
    if df_record.empty or df_history.empty or 'cycle' not in df_record.columns:
        return default_stats

    curr = df_record[df_record['cycle'] == df_record['cycle'].max()]
    buys = curr[(curr['action'] == 'buy') & (~curr['id'].str.startswith('cash'))]
    if buys.empty:
        return default_stats

    entry_date = buys['date'].min()
    period = df_history[df_history['date'] >= entry_date]
    if period.empty:
        return default_stats

    cur_nav = df_history['total_nav'].iloc[-1]
    peak_nav = period['total_nav'].max()
    entry_nav = period['total_nav'].iloc[0]

    return {
        "macro_dd": min(0.0, (cur_nav / peak_nav - 1.0) * 100) if peak_nav > 0 else 0.0,
        "macro_roi": (cur_nav / entry_nav - 1.0) * 100 if entry_nav > 0 else 0.0,
        "entry_date_str": entry_date.strftime('%m/%d'),
        "has_tp": (curr['reason'] == 'TP_QUARTER').any(),
    }


def calculate_macro_drawdown(df_record: pd.DataFrame, df_history: pd.DataFrame) -> tuple[float, str]:
    stats = get_macro_cycle_stats(df_record, df_history)
    return stats["macro_dd"], stats["entry_date_str"]


def calculate_macro_profit(df_record: pd.DataFrame, df_history: pd.DataFrame) -> tuple[float, str, bool]:
    stats = get_macro_cycle_stats(df_record, df_history)
    return stats["macro_roi"], stats["entry_date_str"], stats["has_tp"]


def _check_macro_stoploss_alerts(df_record, df_history, threshold=-10.0):
    """
    宏觀總經/投組移動停損警報 (總資產自高點跌 10% 全清倉)
    """
    stats = get_macro_cycle_stats(df_record, df_history)
    if stats["macro_dd"] <= threshold:
        return [
            f"宏觀帳戶高點移動停損 ： 觸發 {abs(threshold):.0f}% 移動停損 ， 本週期{stats['entry_date_str']}投組高點回撤已達 {stats['macro_dd']:.1f}% ， 全面清倉"
        ]
    return []


def _check_macro_takeprofit_alerts(df_record, df_history, threshold=10.0):
    """
    宏觀帳戶%數停利警報 (總資產獲利達 10% 砍 1/4 入袋降波動率)
    """
    stats = get_macro_cycle_stats(df_record, df_history)
    if stats["macro_roi"] >= threshold and not stats["has_tp"]:
        return [
            f"宏觀帳戶停利 ： 本週期{stats['entry_date_str']}總資產獲利已達 {stats['macro_roi']:.1f}% ， 各檔持股砍 1/4 入袋降低波動率"
        ]
    return []


def _check_bear_market_alerts(df_0050, threshold=-0.20):
    if df_0050.empty or 'close' not in df_0050.columns:
        return []
    max_price    = df_0050['close'].max()
    latest_price = df_0050['close'].iloc[-1]
    if max_price == 0:
        return []
    drawdown = (latest_price - max_price) / max_price
    if drawdown <= threshold:
        return [f"大盤熊市通知 ： 正式進入熊市，全面砍倉 ， 大盤已從高點回撤 {drawdown*100:.1f}%"]
    return []


def generate_system_alerts(df_record, df_price, df_0050, unrealized_data, df_history=None):
    """
    統一收集並回傳所有風控警報。
    防呆機制：若目前投組已無任何持倉股票（100% 現金空倉狀態），代表清倉/停損指令皆已執行，不需再發出警報。
    """
    has_active_holdings = len(unrealized_data) > 0 and any(item.get("股數", 0) > 0 for item in unrealized_data)

    alerts = []
    if has_active_holdings:
        alerts.extend(_check_bear_market_alerts(df_0050))
        alerts.extend(_check_micro_stage1_stoploss_alerts(unrealized_data))
        alerts.extend(_check_micro_stage2_stoploss_alerts(unrealized_data))
        if df_history is not None:
            alerts.extend(_check_macro_stoploss_alerts(df_record, df_history))
            alerts.extend(_check_macro_takeprofit_alerts(df_record, df_history))

    return alerts


# ──────────────────────────────────────────
# 模組 4：品質指標與系統共異變分析
# ──────────────────────────────────────────
def calc_beta(r_p, r_m):
    if len(r_p) < 2 or len(r_m) < 2 or r_m.var() == 0:
        return 0.0
    return np.cov(r_p, r_m)[0, 1] / np.var(r_m, ddof=1)


def calc_dd_series(s):
    peak = (1 + s).cummax()
    return ((1 + s) - peak) / peak * 100


def calculate_quality_comparison(df_history):
    """
    產生標的品質指標對照表格 DataFrame
    """
    if df_history.empty:
        return pd.DataFrame()

    ret_0050 = df_history['roi_0050'].diff().dropna()

    def _calc_row(roi_series, label, bm_ret_series=None):
        d_ret = roi_series.diff().dropna()
        if len(d_ret) == 0:
            return {"標的": label, "Sharpe": "0.00", "Sortino": "0.00", "Calmar": "0.00", "波動率": "0.0%", "Beta": "0.00", "最大回撤": "0.00%"}
        first_val = roi_series.iloc[0]
        last_val  = roi_series.iloc[-1]
        roi_win   = ((1 + last_val) / (1 + first_val) - 1) * 100
        m_dd      = calc_dd_series(roi_series).min()
        sh        = (d_ret.mean() / d_ret.std() * (252 ** 0.5)) if d_ret.std() > 0 else 0
        down_std  = d_ret[d_ret < 0].std()
        so        = (d_ret.mean() / down_std * (252 ** 0.5)) if down_std > 0 else 0
        ann_r     = (1 + roi_win / 100) ** (252 / max(len(roi_series), 1)) - 1
        ca        = ann_r / abs(m_dd / 100) if m_dd < 0 else 0
        vo        = d_ret.std() * (252 ** 0.5) * 100
        be        = calc_beta(d_ret, bm_ret_series) if bm_ret_series is not None else 1.00
        return {
            "標的": label,
            "Sharpe": f"{sh:.2f}",
            "Sortino": f"{so:.2f}",
            "Calmar": f"{ca:.2f}",
            "波動率": f"{vo:.1f}%",
            "Beta": f"{be:.2f}",
            "最大回撤": f"{m_dd:.2f}%"
        }

    rows = [
        _calc_row(df_history['roi_total'],  "基金",        ret_0050),
        _calc_row(df_history['roi_0050'],   "大盤(0050)",   ret_0050),
        _calc_row(df_history['roi_00631L'], "正二(00631L)", ret_0050),
    ]
    return pd.DataFrame(rows)


# ──────────────────────────────────────────
# 模組 5：投資人認購與持股比例分析
# ──────────────────────────────────────────
@st.cache_data
def calculate_subscriber_holding_ratio(df_flows=None) -> pd.DataFrame:
    """
    計算各投資人持股單位數與認購持股比例。
    回傳欄位：subscriber, share, ratio (float, %), ratio_str (str, 例如 78.9%)
    由持股比例高到低排序。
    """
    if df_flows is None:
        _, df_flows, _, _, _ = load_data()

    if df_flows.empty or 'subscriber' not in df_flows.columns:
        return pd.DataFrame(columns=['subscriber', 'share', 'ratio', 'ratio_str'])

    subscriber_shares = {}
    for _, row in df_flows.iterrows():
        sub = str(row.get('subscriber', '')).strip()
        if not sub or sub.lower() in ('nan', 'none', ''):
            continue
        if sub not in subscriber_shares:
            subscriber_shares[sub] = 0.0

        action = str(row.get('action', '')).strip().lower()
        share = float(row.get('share', 0.0))
        if action in ['subscribe', 'fee_in', 'discount']:
            subscriber_shares[sub] += share
        elif action in ['redeem', 'fee_out']:
            subscriber_shares[sub] -= share

    total_shares = sum(s for s in subscriber_shares.values() if s > 0)
    if total_shares <= 0:
        return pd.DataFrame(columns=['subscriber', 'share', 'ratio', 'ratio_str'])

    results = []
    for sub, shares in subscriber_shares.items():
        if shares > 0:
            ratio = (shares / total_shares) * 100
            results.append({
                'subscriber': sub,
                'share': round(shares, 4),
                'ratio': round(ratio, 2),
                'ratio_str': f"{ratio:.1f}%"
            })

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values('ratio', ascending=False).reset_index(drop=True)
    return df_res
