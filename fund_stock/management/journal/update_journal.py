"""
update_journal.py
Kitsune Kernel — 交易紀錄維護模組
負責 journal.csv 的資料讀取、寫入、復原與白名單對照。
標準格式：cycle, date, id, action, share, price, reason
"""

import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import config

data_path = config.STOCK_JOURNAL_DIR / "journal.csv"
wl_path = config.MARKET_DATA_DIR / "white_list.csv"

COLUMNS = ['cycle', 'date', 'id', 'action', 'share', 'price', 'reason']
REASONS = [
    'ENTRY',        # 每月定期換股進場買進
    'REBALANCE',    # 換股日例行結算出清
    'SL_HALF',      # 微觀移動減碼 10% (砍半)
    'SL_FULL',      # 微觀移動停損 20% (清倉)
    'TP_QUARTER',   # 宏觀獲利 10% 砍 1/4 入袋
    'MACRO_SL',     # 宏觀高點回撤 10% 清倉
    'MANUAL'        # 特殊手動操作 / 當沖
]


def load_stock_names() -> dict[str, str]:
    """讀取 white_list.csv 取得股票代號與名稱對照字典"""
    if not os.path.exists(wl_path):
        return {}
    try:
        df_wl = pd.read_csv(wl_path, dtype={'id': str}, encoding='utf-8-sig')
        name_col = 'nmae' if 'nmae' in df_wl.columns else ('name' if 'name' in df_wl.columns else None)
        if name_col:
            return dict(zip(df_wl['id'].astype(str).str.strip(), df_wl[name_col].astype(str).str.strip()))
    except Exception:
        pass
    return {}


def get_latest_cycle() -> int:
    """取得當前最新週期編號"""
    if not os.path.exists(data_path):
        return 0
    try:
        df = pd.read_csv(data_path, encoding='utf-8-sig')
        if 'cycle' in df.columns and not df['cycle'].dropna().empty:
            return int(pd.to_numeric(df['cycle'], errors='coerce').dropna().max())
    except Exception:
        pass
    return 0


def append_journal_entry(cycle: int, date: str, stock_id: str, action: str, share: int | float, price: float, reason: str) -> tuple[bool, str]:
    """
    新增一筆交易紀錄至 journal.csv
    嚴格遵循 7 欄位標準：cycle, date, id, action, share, price, reason
    """
    # 格式清理與驗證
    stock_id = str(stock_id).strip()
    action = str(action).strip().lower()
    reason = str(reason).strip().upper()
    date_str = str(date).strip()

    if not stock_id:
        return False, "股票代號不能為空"
    if action not in ['buy', 'sell']:
        return False, "動作必須為 buy 或 sell"
    if reason not in REASONS:
        return False, f"原因必須為以下之一: {', '.join(REASONS)}"
    if share <= 0:
        return False, "股數必須大於 0"
    if price <= 0:
        return False, "單價必須大於 0"

    # 讀取現有紀錄
    if os.path.exists(data_path):
        df_old = pd.read_csv(data_path, dtype={'id': str}, encoding='utf-8-sig').dropna(how='all')
    else:
        df_old = pd.DataFrame(columns=COLUMNS)

    new_row = pd.DataFrame([{
        'cycle': int(cycle),
        'date': date_str,
        'id': stock_id,
        'action': action,
        'share': int(share) if float(share).is_integer() else round(float(share), 4),
        'price': round(float(price), 2),
        'reason': reason
    }])

    df_save = pd.concat([df_old[COLUMNS], new_row[COLUMNS]], ignore_index=True)
    df_save.to_csv(data_path, index=False, encoding='utf-8-sig')

    names = load_stock_names()
    name_display = f" {names.get(stock_id, '')}" if stock_id in names else ""
    return True, f"成功寫入: Cycle {cycle} | {date_str} | {stock_id}{name_display} | {action.upper()} | {share}股 | ${price} | {reason}"


def delete_last_entry() -> tuple[bool, str]:
    """刪除 journal.csv 最後一筆交易紀錄 (防呆復原)"""
    if not os.path.exists(data_path):
        return False, "檔案不存在"
    try:
        df = pd.read_csv(data_path, dtype={'id': str}, encoding='utf-8-sig').dropna(how='all')
        if df.empty:
            return False, "目前無任何紀錄可刪除"
        last_row = df.iloc[-1].to_dict()
        df = df.iloc[:-1]
        df.to_csv(data_path, index=False, encoding='utf-8-sig')
        desc = f"已刪除最後一筆: Cycle {last_row.get('cycle')} | {last_row.get('date')} | {last_row.get('id')} | {last_row.get('action')} | {last_row.get('share')}股"
        return True, desc
    except Exception as e:
        return False, f"刪除失敗: {e}"


def get_recent_entries(n: int = 10) -> pd.DataFrame:
    """取得最近 n 筆交易紀錄（含中文名稱顯示）"""
    if not os.path.exists(data_path):
        return pd.DataFrame(columns=['cycle', 'date', 'id', 'name', 'action', 'share', 'price', 'reason'])
    try:
        df = pd.read_csv(data_path, dtype={'id': str}, encoding='utf-8-sig').dropna(how='all')
        if df.empty:
            return pd.DataFrame(columns=['cycle', 'date', 'id', 'name', 'action', 'share', 'price', 'reason'])
        names = load_stock_names()
        df['name'] = df['id'].astype(str).str.strip().map(names).fillna('')
        cols = ['cycle', 'date', 'id', 'name', 'action', 'share', 'price', 'reason']
        return df[cols].tail(n)
    except Exception:
        return pd.DataFrame(columns=['cycle', 'date', 'id', 'name', 'action', 'share', 'price', 'reason'])


# 相容舊版名稱
update_journal = append_journal_entry