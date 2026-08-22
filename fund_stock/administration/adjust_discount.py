import pandas as pd
from datetime import datetime
import sys
import os
from pathlib import Path

# Ensure the script can import config from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

def main():
    print("=== 手續費折讓/現金調整工具 ===")
    
    # 1. 檢查是否全部平倉
    journal_path = config.STOCK_JOURNAL_DIR / "journal.csv"
    if not os.path.exists(journal_path):
        print("錯誤：找不到 journal.csv")
        return
        
    df_record = pd.read_csv(journal_path, dtype={'id': str})
    df_record.columns = df_record.columns.str.strip().str.lower()
    df_record['share'] = pd.to_numeric(df_record['share'], errors='coerce').fillna(0)
    
    for aid in df_record['id'].unique():
        sub = df_record[df_record['id'] == aid]
        shares = sub[sub['action'] == 'buy']['share'].sum() - sub[sub['action'] == 'sell']['share'].sum()
        if shares > 0.0001:
            print(f"錯誤：無法調整，請先全部平倉留下現金。目前仍持有 {aid} 共 {shares} 股。")
            return

    # 2. 取得目前持有現金
    nav_path = config.FUND_STOCK_ADMINISTRATION_DIR / "navpu.csv"
    if not os.path.exists(nav_path):
        print("錯誤：找不到 navpu.csv")
        return
        
    df_nav = pd.read_csv(nav_path)
    if df_nav.empty:
        print("錯誤：navpu.csv 為空")
        return
        
    current_cash = df_nav.iloc[-1]['cash']
    current_nav_date = df_nav.iloc[-1]['date']
    print(f"({current_nav_date}) 目前程式持有現金: {current_cash:.2f}")
    
    # 3. 輸入帳戶目前現金
    try:
        actual_cash_str = input("請輸入帳戶目前現金以供調整: ")
        actual_cash = float(actual_cash_str)
    except ValueError:
        print("錯誤：輸入無效，請輸入數字。")
        return
        
    if actual_cash < current_cash:
        print(f"錯誤：輸入的帳戶現金 ({actual_cash}) 小於 目前程式持有現金 ({current_cash})，防呆機制啟動。")
        return
        
    extra_cash = actual_cash - current_cash
    if extra_cash <= 0:
        print("無額外現金需調整。")
        return
        
    print(f"需調整分配的額外現金為: {extra_cash:.2f}")
    
    # 4. 輸入調整日期
    adj_date = input(f"請輸入調整日期 (YYYY-MM-DD，預設為最後一個交易日 {current_nav_date}): ")
    if not adj_date.strip():
        adj_date = current_nav_date
    
    # 5. 按申購人比例分配
    transaction_path = config.FUND_STOCK_ADMINISTRATION_DIR / "transaction.csv"
    if not os.path.exists(transaction_path):
        print("錯誤：找不到 transaction.csv")
        return
        
    df_flows = pd.read_csv(transaction_path, encoding='utf-8-sig')
    
    # 為了計算方便，建立一個小寫欄位的副本，但寫回時保留原始欄位
    df_calc = df_flows.copy()
    df_calc.columns = df_calc.columns.str.strip().str.lower()
    df_calc['share'] = pd.to_numeric(df_calc['share'], errors='coerce').fillna(0)
    
    subscriber_shares = {}
    for _, row in df_calc.iterrows():
        sub = str(row['subscriber']).strip()
        if not sub or sub.lower() == 'nan':
            continue
        if sub not in subscriber_shares:
            subscriber_shares[sub] = 0
            
        if row['action'] in ['subscribe', 'fee_in', 'discount']:
            subscriber_shares[sub] += row['share']
        elif row['action'] in ['redeem', 'fee_out']:
            subscriber_shares[sub] -= row['share']
            
    total_shares = sum(s for s in subscriber_shares.values() if s > 0)
    if total_shares <= 0:
        print("錯誤：目前總申購單位數為 0，無法分配。")
        return
        
    # 建立新紀錄
    new_rows = []
    
    print("\n【分配明細】")
    for sub, shares in subscriber_shares.items():
        if shares > 0:
            ratio = shares / total_shares
            allocated_cash = round(extra_cash * ratio, 2)
            print(f"申購人: {sub}, 單位數: {shares:.2f}, 比例: {ratio*100:.2f}%, 分配現金: {allocated_cash}")
            
            # 使用原始的 df_flows 欄位名稱來建立字典
            new_row_dict = {}
            for col in df_flows.columns:
                col_lower = col.strip().lower()
                if col_lower == 'date':
                    new_row_dict[col] = adj_date
                elif col_lower == 'action':
                    new_row_dict[col] = 'discount'
                elif col_lower == 'price':
                    new_row_dict[col] = allocated_cash
                elif col_lower == 'share':
                    new_row_dict[col] = ''  # 空白，交由 update_navpu.py 回填
                elif col_lower == 'subscriber':
                    new_row_dict[col] = sub
                elif col_lower == 'name':
                    new_row_dict[col] = '手續費折讓調整'
                else:
                    new_row_dict[col] = ''
            new_rows.append(new_row_dict)
            
    if not new_rows:
        return
        
    # 6. 附加到 transaction.csv
    df_new = pd.DataFrame(new_rows)
    df_flows_updated = pd.concat([df_flows, df_new], ignore_index=True)
    df_flows_updated.to_csv(transaction_path, index=False, encoding='utf-8-sig')
    
    print("\n成功寫入 transaction.csv。")
    print("請記得執行 update_navpu.py 來重新計算淨值與回填單位數。")

if __name__ == '__main__':
    main()
