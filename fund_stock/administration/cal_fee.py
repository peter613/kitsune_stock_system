import os
import sys
from pathlib import Path

# 將專案根目錄加入 sys.path，避免 ModuleNotFoundError
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from portfolio.dashboard.engine import load_data as load_port_data, FundSubscriber
import config
from datetime import datetime

'''
手續費結算邏輯:
1. 本程式建議於「每月初（如 1 號）」執行。
2. 執行時，會自動設定「上個完整月份」為結算目標（target_month）。
   (例如：7/1 執行時，結算整個 6 月的報酬率)
3. 結算完成後，績效費的扣款日期會押在「下個月 1 號」（也就是當前執行的月份）。
   (例如：結算 6 月份績效，扣款日會押在 7/1)
這樣設計可確保每月獲利週期完整，不遺漏任何一天的績效。
'''

PERFORMANCE_FEE_RATE = 0.05  # 績效費率 5%
MANAGER = 'PETER'            # 基金經理人（績效費收取方）

def run_fee_settlement(target_month):
    """
    透過讀取 navpu 歷史與 transaction 來計算各投資人的績效費
    自動掃描所有非經理人的投資人，各自計算績效費
    若賺錢扣5%，否則不扣費用
    target_month 格式：'2026-03'
    """
    # 1. 調用 port 模組讀取 df_history(navpu) 與 df_flows(transaction)
    df_history, df_flows = load_port_data()

    # 2. 讀取 transaction.csv，找出所有投資人
    tx_path = config.FUND_STOCK_ADMINISTRATION_DIR / 'transaction.csv'
    df_tx = pd.read_csv(tx_path)

    all_subscribers = df_tx['subscriber'].dropna().unique()
    investors = [s for s in all_subscribers if s != MANAGER]

    if not investors:
        print("系統提示：找不到任何非經理人的投資人。")
        return

    print(f"\n{'='*60}")
    print(f"績效費結算：{target_month}")
    print(f"投資人清單：{', '.join(investors)}")
    print(f"{'='*60}")

    new_rows = []

    for investor in investors:
        fee_name = f"{target_month} 手續費扣除"

        # 檢查是否已結算過，避免重複扣款
        already_settled = (
            (df_tx['name'] == fee_name) &
            (df_tx['subscriber'] == investor)
        ).any()

        if already_settled:
            print(f"\n[{investor}] {target_month} 手續費已在紀錄中，跳過。")
            continue

        # 取得該投資人的權益序列
        sub = FundSubscriber(investor)
        df_sub = sub.calculate_equity(df_history, df_flows)

        if df_sub.empty:
            print(f"\n[{investor}] 找不到 {target_month} 的權益資料，跳過。")
            continue

        # 篩選該月份資料
        df_month = df_sub[df_sub['date'].dt.strftime('%Y-%m') == target_month]
        if df_month.empty:
            print(f"\n[{investor}] 月份 {target_month} 無可用資料，跳過。")
            continue

        last_entry  = df_month.iloc[-1]
        first_entry = df_month.iloc[0]

        # 計算月報酬率
        nav_end   = last_entry['nav_per_unit']
        nav_start = first_entry['nav_per_unit']
        monthly_return = (nav_end / nav_start) - 1.0

        if monthly_return <= 0:
            print(f"\n[{investor}] {target_month} 無獲利（月報酬率 {monthly_return*100:.2f}%），免收績效費。")
            continue

        # 計算績效費單位數
        fee_units = last_entry['units'] * monthly_return * PERFORMANCE_FEE_RATE
        
        # 將扣款日期押在結算目標月份的「下個月 1 號」
        # 例如結算 2026-06，則扣款日押在 2026/07/01
        target_dt = datetime.strptime(f"{target_month}-01", '%Y-%m-%d')
        if target_dt.month == 12:
            next_month_dt = target_dt.replace(year=target_dt.year + 1, month=1)
        else:
            next_month_dt = target_dt.replace(month=target_dt.month + 1)
        calc_date = next_month_dt.strftime('%Y/%m/%d')

        print(f"\n[{investor}] 月報酬率 {monthly_return*100:.2f}%，期末單位數 {last_entry['units']:.6f}，"
              f"扣除績效費 {fee_units:.6f} 單位。")

        # 構造扣款與入帳紀錄
        new_rows.append({
            'date':       calc_date,
            'action':     'fee_out',
            'price':      '',
            'share':      fee_units,
            'subscriber': investor,
            'name':       fee_name
        })
        new_rows.append({
            'date':       calc_date,
            'action':     'fee_in',
            'price':      '',
            'share':      fee_units,
            'subscriber': MANAGER,
            'name':       f"{target_month} {investor} 貢獻績效費"
        })

    # 更新並儲存 CSV
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_final = pd.concat([df_tx, df_new], ignore_index=True)
        df_final.to_csv(tx_path, index=False, encoding='utf-8-sig')
        print(f"\n結算完成：共處理 {len(new_rows)//2} 位投資人的績效費。")
    else:
        print(f"\n本月無需扣除任何績效費。")


if __name__ == "__main__":
    now = datetime.now()
    # 自動計算上個月作為 target_month
    if now.month == 1:
        target_month_str = f"{now.year - 1}-12"
    else:
        target_month_str = f"{now.year}-{now.month - 1:02d}"
        
    print(f"系統執行時間: {now.strftime('%Y-%m-%d')}，自動結算上個月 ({target_month_str}) 績效")
    run_fee_settlement(target_month_str)