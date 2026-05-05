"""
update_fund_ledger.py
職責：
  1. 計算每日基金帳務 → 輸出 quant_navpu.csv
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
START_DATE = '2026-03-10'   # 申購日，確保第一天 NAV = BASE_NAV
BASE_NAV   = 10.0

import os
import pandas as pd
from datetime import datetime
import config


# ──────────────────────────────────────────
# 手續費 / 稅
# ──────────────────────────────────────────
def _tax_rate(stock_id: str) -> float:
    return 0.003

def _buy_cost(price: float, share: float, stock_id: str) -> float:
    val = price * share
    return val + int(val * 0.001425)

def _sell_proceeds(price: float, share: float, stock_id: str) -> float:
    val = price * share
    return val - int(val * 0.001425) - int(val * _tax_rate(stock_id))


# ──────────────────────────────────────────
# 讀取原始數據
# ──────────────────────────────────────────
def _load_raw_data():
    # journal.csv
    df_record = pd.read_csv(config.JOURNAL_DIR / "journal_quant.csv", dtype={'id': str})
    df_record.columns  = df_record.columns.str.strip().str.lower()
    df_record['date']  = pd.to_datetime(df_record['date'], errors='coerce').dt.normalize()
    df_record = df_record[df_record['date'] >= START_DATE].copy()
    df_record['price'] = pd.to_numeric(df_record['price'], errors='coerce').fillna(0)
    df_record['share'] = pd.to_numeric(df_record['share'], errors='coerce').fillna(0)

    # transaction.csv
    transaction_path = config.KITSUNE_QUANT_ADMINISTRATION_DIR / "transaction_quant.csv"
    df_flows = pd.read_csv(transaction_path)
    df_flows.columns = df_flows.columns.str.strip().str.lower()
    df_flows['date']  = pd.to_datetime(df_flows['date'], errors='coerce').dt.normalize()
    df_flows['price'] = pd.to_numeric(df_flows['price'], errors='coerce')  # 保留 NaN
    df_flows['share'] = pd.to_numeric(df_flows['share'], errors='coerce')  # 保留 NaN

    # market.parquet
    market_path = config.DATABASE_DIR / "market.parquet"
    if os.path.exists(market_path):
        df_price = pd.read_parquet(market_path)
        df_price.columns = df_price.columns.str.strip().str.lower()
        if 'price' in df_price.columns and 'close' not in df_price.columns:
            df_price = df_price.rename(columns={'price': 'close'})
        df_price['date'] = pd.to_datetime(df_price['date'], errors='coerce').dt.normalize()
        df_price = df_price[df_price['date'] >= START_DATE].copy()
    else:
        df_price = pd.DataFrame(columns=['date', 'id', 'close'])

    return df_record, df_flows, df_price


# ──────────────────────────────────────────
# 計算每日帳務
# ──────────────────────────────────────────
def _calculate_nav(df_record, df_flows, df_price):
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

    for d in dates:
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

        total_nav = cash + mv

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
        df_flows.to_csv(transaction_path, index=False,encoding = 'utf-8-sig')
        print(f"transaction.csv：回填 {filled} 筆，已寫回。")
    else:
        print("transaction.csv：無需回填，略過。")


# ──────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────
def update_quant():

    df_record, df_flows, df_price = _load_raw_data()
    df_nav = _calculate_nav(df_record, df_flows, df_price)

    if df_nav.empty:
        print("警告：計算結果為空，檔案未更新。")
        return

    # 輸出 1：quant_navpu.csv
    nav_path = config.KITSUNE_QUANT_ADMINISTRATION_DIR / "navpu_quant.csv"
    nav_path.parent.mkdir(parents=True, exist_ok=True)
    df_nav.to_csv(nav_path, index=False)
    print(f'基金帳務更新完成，目前總筆數 {len(df_nav)} 筆')
    print(f"現淨值(NAV) {df_nav['nav_per_unit'].iloc[-1]:.4f}")

    # 輸出 2：回填 transaction.csv
    transaction_path = config.KITSUNE_QUANT_ADMINISTRATION_DIR / "transaction_quant.csv"
    _backfill_transaction(df_flows, df_nav, transaction_path)


if __name__ == '__main__':
    update_quant()