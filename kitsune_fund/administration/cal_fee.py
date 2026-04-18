import os
import pandas as pd
from kitsune_fund.valuation.engine import load_data,calculate_daily_history,FundSubscriber
import config
from datetime import datetime

fee_rate = 0.0001

def run_fee_settlement(target_month):
    """
    透過 engine 計算出的單位數，執行每月手續費轉移
    target_month 格式：'2026-03'
    """
    # 1. 調用 engine 模組讀取並計算所有數據，取得最新的 nav_per_unit
    df_record, df_flows, df_price, df_0050, df_00631L =load_data()
    df_history = calculate_daily_history(df_record, df_flows, df_price, df_0050, df_00631L)
    
    # 2. 取得 LYNN 的權益序列，以獲取精確單位數 (share)
    lynn = FundSubscriber("LYNN")
    df_lynn = lynn.calculate_equity(df_history, df_flows)
    
    if df_lynn.empty:
        print(f"系統提示：找不到 {target_month} 的 LYNN 權益資料。")
        return

    # 3. 檢查 transaction.csv 是否已存在該月扣款紀錄，避免重複扣款
    tx_path = config.KITSUNE_FUND_ADMINISTRATION_DIR / 'transaction.csv'
    df_tx = pd.read_csv(tx_path)

    fee_name = f"{target_month} 手續費扣除"
    
    # 只要 name 欄位裡有任何一個值等於 fee_name，就會回傳 True
    if (df_tx['name'] == fee_name).any():
        print(f"系統提示：{target_month} 手續費已在紀錄中，跳過執行。")
        return

    # 4. 定位該月份最後一筆資料並計算 0.01% 手續費
    # 篩選出該月份資料
    df_month = df_lynn[df_lynn['date'].dt.strftime('%Y-%m') == target_month]
    if df_month.empty:
        print(f"系統提示：月份 {target_month} 無可用資料。")
        return
        
    last_entry = df_month.iloc[-1]
    current_units = last_entry['units']
    fee_units = current_units * fee_rate
    calc_date = last_entry['date'].strftime('%Y/%m/%d')

    # 5. 構造扣款與入帳紀錄
    # fee_out: 妹妹單位數減少 / fee_in: 哥哥單位數增加
    new_rows = pd.DataFrame([
        {
            'date': calc_date, 
            'action': 'fee_out', 
            'price': '', 
            'share': fee_units, 
            'subscriber': 'LYNN', 
            'name': fee_name
        },
        {
            'date': calc_date, 
            'action': 'fee_in',  
            'price': '', 
            'share': fee_units, 
            'subscriber': 'PETER', 
            'name': f"{target_month} 妹妹上繳手續費"
        }
    ])
    
    # 6. 更新並儲存 CSV
    df_final = pd.concat([df_tx, new_rows], ignore_index=True)
    df_final.to_csv(tx_path, index=False, encoding='utf-8-sig')
    print(f"結算完成：{target_month} 已移轉 {fee_units:.6f} 單位。")

if __name__ == "__main__":
    # 自動抓取當前月份進行結算，亦可手動指定字串如 '2026-03'
    current_month_str = datetime.now().strftime('%Y-%m')
    run_fee_settlement(current_month_str)