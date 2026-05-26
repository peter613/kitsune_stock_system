import os
import pandas as pd
from kitsune_quant.management.dashboard.engine import load_data, calculate_daily_history, FundSubscriber
import config
from datetime import datetime

PERFORMANCE_FEE_RATE = 0.05  # 績效費率 5%

def run_fee_settlement(target_month):
    """
    透過 engine 計算出的單位數，執行每月手續費轉移
    若賺錢扣5%，否則不扣費用
    target_month 格式：'2026-03'
    """
    # 1. 調用 engine 模組讀取並計算所有數據
    df_record, df_flows, df_price, df_0050, df_00631L = load_data()
    df_history = calculate_daily_history(df_record, df_flows, df_price, df_0050, df_00631L)

    # 2. 取得 LYNN 的權益序列
    lynn = FundSubscriber("LYNN")
    df_lynn = lynn.calculate_equity(df_history, df_flows)

    if df_lynn.empty:
        print(f"系統提示：找不到 {target_month} 的 LYNN 權益資料。")
        return

    # 3. 檢查是否已結算過，避免重複扣款
    tx_path = config.kitsune_quant_ADMINISTRATION_DIR / 'transaction.csv'
    df_tx = pd.read_csv(tx_path)

    fee_name = f"{target_month} 手續費扣除"
    if (df_tx['name'] == fee_name).any():
        print(f"系統提示：{target_month} 手續費已在紀錄中，跳過執行。")
        return

    # 4. 篩選該月份資料
    df_month = df_lynn[df_lynn['date'].dt.strftime('%Y-%m') == target_month]
    if df_month.empty:
        print(f"系統提示：月份 {target_month} 無可用資料。")
        return

    last_entry  = df_month.iloc[-1]
    first_entry = df_month.iloc[0]

    # 5. 計算月報酬率，無獲利則免收
    nav_end   = last_entry['nav_per_unit']
    nav_start = first_entry['nav_per_unit']
    monthly_return = (nav_end / nav_start) - 1.0

    if monthly_return <= 0:
        print(f"系統提示：{target_month} 無獲利（月報酬率 {monthly_return*100:.2f}%），免收績效費。")
        return

    # 6. 計算績效費單位數
    fee_units  = last_entry['units'] * monthly_return * PERFORMANCE_FEE_RATE
    calc_date  = last_entry['date'].strftime('%Y/%m/%d')

    print(f"結算資訊：月報酬率 {monthly_return*100:.2f}%，期末單位數 {last_entry['units']:.6f}，"
          f"扣除績效費 {fee_units:.6f} 單位。")

    # 7. 構造扣款與入帳紀錄
    new_rows = pd.DataFrame([
        {
            'date':       calc_date,
            'action':     'fee_out',
            'price':      '',
            'share':      fee_units,
            'subscriber': 'LYNN',
            'name':       fee_name
        },
        {
            'date':       calc_date,
            'action':     'fee_in',
            'price':      '',
            'share':      fee_units,
            'subscriber': 'PETER',
            'name':       f"{target_month} 妹妹上繳績效費"
        }
    ])

    # 8. 更新並儲存 CSV
    df_final = pd.concat([df_tx, new_rows], ignore_index=True)
    df_final.to_csv(tx_path, index=False, encoding='utf-8-sig')
    print(f"結算完成：{target_month} 已移轉 {fee_units:.6f} 單位。")


if __name__ == "__main__":
    current_month_str = datetime.now().strftime('%Y-%m')
    run_fee_settlement(current_month_str)