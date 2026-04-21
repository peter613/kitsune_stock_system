"""
engine_port.py
Kitsune Kernel — 投資者入口引擎
職責：NAV 歷史讀取、申購/贖回紀錄讀取、投資者持分計算
供應：app_port.py

transaction.csv 格式規範：
  申購：      action=subscribe,    price=申購金額(TWD),  share=空白,       subscriber=PETER/LYNN
  贖回(股數)：action=redeem,       price=空白,           share=贖回單位數, subscriber=PETER/LYNN
  贖回(金額)：action=redeem_cash,  price=贖回金額(TWD),  share=空白,       subscriber=PETER/LYNN
"""
import pandas as pd
import streamlit as st
import config


# ──────────────────────────────────────────
# 模組 1：資料讀取
# ──────────────────────────────────────────
def _load_navpu(admin_dir) -> pd.DataFrame:
    df = pd.read_csv(admin_dir / "quant_navpu.csv")
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    return df.sort_values('date').reset_index(drop=True)

def _load_flows(admin_dir) -> pd.DataFrame:
    df = pd.read_csv(admin_dir / "transaction.csv", dtype={'id': str})
    df.columns = df.columns.str.strip().str.lower()
    df['date']  = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['share'] = pd.to_numeric(df['share'], errors='coerce').fillna(0)
    return df

@st.cache_data
def load_data():
    quant_nav = _load_navpu(config.KITSUNE_QUANT_ADMINISTRATION_DIR)
    df_flows  = _load_flows(config.KITSUNE_QUANT_ADMINISTRATION_DIR)
    return quant_nav, df_flows

# ──────────────────────────────────────────
# 模組 2：FundSubscriber — 單一投資者
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

        result_rows    = []
        units          = 0.0
        total_cost     = 0.0   # 累積投入現金
        total_redeemed = 0.0   # 累積提領現金

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
                    redeem_units   = float(row['share'])
                    cash_out       = redeem_units * nav
                    if units > 0:
                        total_cost -= (redeem_units / units) * total_cost
                    units          = max(0.0, units - redeem_units)
                    total_redeemed += cash_out

                elif row['action'] == 'redeem_cash':
                    cash_out       = float(row['price'])
                    redeem_units   = cash_out / nav if nav > 0 else 0.0
                    if units > 0:
                        total_cost -= (redeem_units / units) * total_cost
                    units          = max(0.0, units - redeem_units)
                    total_redeemed += cash_out

                elif row['action'] == 'fee_in':
                    units += float(row['share'])
                elif row['action'] == 'fee_out':
                    units = max(0.0, units - float(row['share']))

            if units > 0:
                equity       = units * nav
                roi          = (equity / total_cost - 1) * 100 #包含中途申購
                net_cashflow = total_cost - total_redeemed
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
# 模組 3：load_subscribers
# ──────────────────────────────────────────
def load_subscribers(df_flows: pd.DataFrame) -> dict:
    if df_flows.empty or 'subscriber' not in df_flows.columns:
        return {}
    names = df_flows['subscriber'].dropna().str.upper().unique()
    return {name: FundSubscriber(name=name) for name in sorted(names)}