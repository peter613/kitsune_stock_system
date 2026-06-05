"""
update_navpu.py (ETF 基金版)
職責：
  1. 計算每日基金帳務 → 輸出 navpu.csv
  2. 回填 transaction.csv 的空白欄位
     - share 空白 → share = price / nav_per_unit
     - price 空白 → price = share * nav_per_unit

規則：不 import engine.py，不依賴 Streamlit，可獨立執行或排程

輸出 CSV 欄位：
    date, nav_per_unit, total_units, total_nav, cash
"""

# ────────
# setting
# ────────
START_DATE = None   # 尚未啟用，啟用後設定起始日如 '2026-07-01'
BASE_NAV   = 10.0

import os
import pandas as pd
from datetime import datetime
import config

# ── ETF 交易稅率 ──
ETF_TAX_RATE = 0.001  # ETF 優惠稅率 0.1%


# ──────────────────────────────────────────
# 手續費 / 稅
# ──────────────────────────────────────────
def _tax_rate(stock_id: str) -> float:
    return ETF_TAX_RATE

def _buy_cost(price: float, share: float, stock_id: str) -> float:
    val = price * share
    return val + int(val * 0.001425)

def _sell_proceeds(price: float, share: float, stock_id: str) -> float:
    val = price * share
    return val - int(val * 0.001425) - int(val * _tax_rate(stock_id))


# ──────────────────────────────────────────
# 從 etf.csv 寬表讀取報價 → 長表
# ──────────────────────────────────────────
def _load_etf_prices():
    """讀取 etf.csv (寬表)，轉為長表格式 (date, id, close)"""
    etf_path = config.MARKET_DATA_DIR / "etf.csv"
    if not os.path.exists(etf_path):
        return pd.DataFrame(columns=['date', 'id', 'close'])

    df_wide = pd.read_csv(etf_path)
    df_wide.columns = df_wide.columns.str.strip()
    df_wide['date'] = pd.to_datetime(df_wide['date'], errors='coerce').dt.normalize()

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
# 讀取原始數據
# ──────────────────────────────────────────
def _load_raw_data():
    # journal.csv (ETF 交易紀錄)
    journal_path = config.ETF_JOURNAL_DIR / "journal.csv"
    if not os.path.exists(journal_path):
        df_record = pd.DataFrame(columns=['date', 'id', 'action', 'price', 'share'])
        df_record.to_csv(journal_path, index=False, encoding='utf-8-sig')
    df_record = pd.read_csv(journal_path, dtype={'id': str})
    df_record.columns  = df_record.columns.str.strip().str.lower()
    df_record['date']  = pd.to_datetime(df_record['date'], errors='coerce').dt.normalize()
    if START_DATE:
        df_record = df_record[df_record['date'] >= START_DATE].copy()
    df_record['price'] = pd.to_numeric(df_record['price'], errors='coerce').fillna(0)
    df_record['share'] = pd.to_numeric(df_record['share'], errors='coerce').fillna(0)

    # transaction.csv (申購/贖回)
    transaction_path = config.FUND_ETF_ADMINISTRATION_DIR / "transaction.csv"
    if not os.path.exists(transaction_path):
        df_flows = pd.DataFrame(columns=['date', 'id', 'action', 'price', 'share'])
        df_flows.to_csv(transaction_path, index=False, encoding='utf-8-sig')
    df_flows = pd.read_csv(transaction_path, encoding='utf-8-sig')
    df_flows.columns = df_flows.columns.str.strip().str.lower()
    df_flows['date']  = pd.to_datetime(df_flows['date'], errors='coerce').dt.normalize()
    df_flows['price'] = pd.to_numeric(df_flows['price'], errors='coerce')  # 保留 NaN
    df_flows['share'] = pd.to_numeric(df_flows['share'], errors='coerce')  # 保留 NaN

    # dividend.csv (ETF 配息)
    dividend_path = config.FUND_ETF_ADMINISTRATION_DIR / "dividend.csv"
    if not os.path.exists(dividend_path):
        df_div = pd.DataFrame(columns=['id', 'ex_dividend_date', 'payment_date', 'dividend_per_share'])
        df_div.to_csv(dividend_path, index=False, encoding='utf-8-sig')
    df_div = pd.read_csv(dividend_path, dtype={'id': str})
    df_div.columns = df_div.columns.str.strip().str.lower()
    df_div['ex_dividend_date'] = pd.to_datetime(df_div['ex_dividend_date'], errors='coerce').dt.normalize()
    df_div['payment_date'] = pd.to_datetime(df_div['payment_date'], errors='coerce').dt.normalize()
    df_div['dividend_per_share'] = pd.to_numeric(df_div['dividend_per_share'], errors='coerce').fillna(0)

    # ETF 報價（從 etf.csv 寬表轉長表）
    df_price = _load_etf_prices()

    return df_record, df_flows, df_price, df_div


# ──────────────────────────────────────────
# 計算每日帳務
# ──────────────────────────────────────────
def _calculate_nav(df_record, df_flows, df_price, df_div):
    if df_record.empty:
        return pd.DataFrame()

    # 第一筆申購初始化
    first_subs = df_flows[df_flows['action'] == 'subscribe'].sort_values('date')
    if first_subs.empty:
        print("錯誤：transaction.csv 找不到任何 subscribe 紀錄。")
        return pd.DataFrame()

    first_sub   = first_subs.iloc[0]
    total_units = first_sub['price'] / BASE_NAV
    cash        = first_sub['price']

    dates = sorted(df_price['date'].dropna().unique())
    if not dates:
        dates = sorted(df_record['date'].unique())

    history = []
    receivables = []  # 應收股息清單：[{'id': '0056', 'amount': 1000, 'payment_date': timestamp}]

    for d in dates:
        # 1. 處理入帳日：應收股息轉現金
        new_receivables = []
        for r in receivables:
            if r['payment_date'] <= d:
                cash += r['amount']
            else:
                new_receivables.append(r)
        receivables = new_receivables

        # 2. 處理除息日：計算新的應收股息
        div_today = df_div[df_div['ex_dividend_date'] == d]
        for _, row in div_today.iterrows():
            aid = row['id']
            # 計算在除息日「前」的持股數
            trades_before = df_record[df_record['date'] < d]
            a = trades_before[trades_before['id'] == aid]
            shares_held = a[a['action'] == 'buy']['share'].sum() - a[a['action'] == 'sell']['share'].sum()
            
            if shares_held > 0:
                receivable_amount = shares_held * row['dividend_per_share']
                receivables.append({
                    'id': aid,
                    'amount': receivable_amount,
                    'payment_date': row['payment_date']
                })

        px_d = (df_price[df_price['date'] <= d]
                .groupby('id').tail(1)
                .set_index('id')['close']
                .to_dict())

        trades_all   = df_record[df_record['date'] <= d].copy()
        trades_today = df_record[df_record['date'] == d].copy()

        # 今日買賣影響現金
        for _, row in trades_today.iterrows():
            if row['action'] == 'buy':
                cash -= _buy_cost(row['price'], row['share'], row['id'])
            elif row['action'] == 'sell':
                cash += _sell_proceeds(row['price'], row['share'], row['id'])

        # 持倉市值
        mv = 0.0
        for aid in trades_all['id'].unique():
            a = trades_all[trades_all['id'] == aid]
            shares = (a[a['action'] == 'buy']['share'].sum()
                    - a[a['action'] == 'sell']['share'].sum())
            if shares > 0:
                raw = shares * px_d.get(aid, 0.0)
                mv += raw - int(raw * 0.001425) - int(raw * _tax_rate(aid))

        total_nav = cash + mv + sum(r['amount'] for r in receivables)

        # 申購前先更新 nav_per_unit
        if total_units > 0:
            nav_per_unit = total_nav / total_units

        # 處理今日申購/贖回（第一筆除外，已初始化）
        flows_today = df_flows[
            (df_flows['date'] == d) &
            (df_flows['date'] != first_sub['date'])  # 跳過初始申購日
        ]
        for _, row in flows_today.iterrows():
            if row['action'] == 'subscribe':
                amount       = row['price']
                new_units    = amount / nav_per_unit
                cash        += amount
                total_units += new_units
                total_nav   += amount

            elif row['action'] == 'redeem':
                units_out    = row['share']
                cash_out     = units_out * nav_per_unit
                cash        -= cash_out
                total_units -= units_out
                total_nav   -= cash_out

        history.append({
            'date':         d,
            'nav_per_unit': round(nav_per_unit, 6),
            'total_units':  round(total_units, 6),
            'total_nav':    round(total_nav, 2),
            'cash':         round(cash, 2),
        })

    return pd.DataFrame(history)

# ──────────────────────────────────────────
# 回填 transaction.csv 空白欄位
# ──────────────────────────────────────────
def _backfill_transaction(df_flows, df_nav, transaction_path):
    df_nav_lookup = df_nav.set_index('date')['nav_per_unit']
    filled = 0

    for idx, row in df_flows.iterrows():
        share_empty = pd.isna(row['share']) or row['share'] == 0
        price_empty = pd.isna(row['price']) or row['price'] == 0

        if share_empty and price_empty:
            print(f"警告：第 {idx+1} 列 {row['date'].date()} share 與 price 皆空白，略過。")
            continue

        if not share_empty and not price_empty:
            continue

        nav_pu = df_nav_lookup.get(row['date'])
        if nav_pu is None or nav_pu == 0:
            print(f"{row['date'].date()} 找不到對應 nav_per_unit，略過。")
            continue

        if share_empty:
            df_flows.at[idx, 'share'] = round(row['price'] / nav_pu, 6)
            filled += 1
        elif price_empty:
            df_flows.at[idx, 'price'] = round(row['share'] * nav_pu, 2)
            filled += 1

    if filled > 0:
        df_flows.to_csv(transaction_path, index=False, encoding = 'utf-8-sig')
        print(f"transaction.csv：回填 {filled} 筆，已寫回。")
    else:
        print("transaction.csv：無需回填，略過。")


# ──────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────
def update_etf_nav():

    if START_DATE is None:
        print("ETF 基金尚未啟用 (START_DATE = None)，略過 NAV 計算。")
        return

    df_record, df_flows, df_price, df_div = _load_raw_data()
    df_nav = _calculate_nav(df_record, df_flows, df_price, df_div)

    if df_nav.empty:
        print("警告：計算結果為空，檔案未更新。")
        return

    # 輸出 1：navpu.csv
    nav_path = config.FUND_ETF_ADMINISTRATION_DIR / "navpu.csv"
    nav_path.parent.mkdir(parents=True, exist_ok=True)
    df_nav.to_csv(nav_path, index=False, encoding='utf-8-sig')
    print(f'ETF 基金帳務更新完成，目前總筆數 {len(df_nav)} 筆')
    print(f"現淨值(NAV) {df_nav['nav_per_unit'].iloc[-1]:.4f}")

    # 輸出 2：回填 transaction.csv
    transaction_path = config.FUND_ETF_ADMINISTRATION_DIR / "transaction.csv"
    _backfill_transaction(df_flows, df_nav, transaction_path)


if __name__ == '__main__':
    update_etf_nav()