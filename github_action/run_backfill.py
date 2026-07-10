import os
import sys

# 將專案根目錄加入路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from dotenv import load_dotenv

# 確保載入 .env
env_path = config.BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)

def run_backfill_job():
    print("===============================================")
    print("[INFO] 啟動歷史資料雲端回補程式 (Manual Backfill)")
    print("===============================================")
    
    # 從 GitHub Action 的環境變數取得輸入的日期字串
    dates_input = os.environ.get("DATES_TO_FIX", "")
    if not dates_input:
        print("[ERROR] 找不到要回補的日期 (DATES_TO_FIX 為空)。中止。")
        return

    # 清洗字串，處理逗號、空白
    raw_dates = [d.strip() for d in dates_input.split(',')]
    dates_to_fix = [d.replace('-', '') for d in raw_dates if d]

    if not dates_to_fix:
        print("[ERROR] 沒有有效日期待回補。")
        return

    print(f"[INFO] 收到待回補的日期共 {len(dates_to_fix)} 天: {dates_to_fix}\n")

    # 引入爬蟲與資料同步模組
    from market_data.updater.market import update_market_data
    from market_data.sync_data import download_from_supabase
    from datetime import datetime
    import time
    
    current_year = datetime.now().year
    
    # 1. 根據執行環境下載現有的資料庫到本地環境
    is_github_action = os.environ.get("GITHUB_ACTIONS") == "true"
    
    if is_github_action:
        print(f"[INFO] GitHub Actions 環境：正在從 Supabase 下載 {current_year} 年歷史資料庫以供合併...")
        remote_market = f"market/market_{current_year}.parquet"
        fake_local_db = str(config.MARKET_DATA_DIR / "market.parquet") 
        dl_success = download_from_supabase('stock_data_backup', remote_market, fake_local_db)
        
        if not dl_success:
            print("[ERROR] 無法從 Supabase 取得歷史資料庫，為保護資料庫，強制中斷。")
            return
    else:
        print("[INFO] 您似乎在本地執行此程式。請確保本地的 market.parquet 為最新狀態。")

    print("\n[INFO] 開始執行回補迴圈...\n")
    
    # 2. 執行迴圈回補
    for d_str in dates_to_fix:
        # 將格式統一為 YYYY-MM-DD
        try:
            formatted_date = datetime.strptime(d_str, '%Y%m%d').strftime('%Y-%m-%d')
        except Exception as e:
            print(f"[ERROR] 日期 {d_str} 格式無法解析，跳過。錯誤: {e}")
            continue

        print(f"--- 正在回補: {formatted_date} ---")
        
        try:
            result = update_market_data(specific_date=formatted_date)
            
            if result.get("status") == "略過":
                print(f"[RESULT] {formatted_date} : 略過 ({result.get('error')})")
            elif result.get("status") == "失敗":
                print(f"[RESULT] {formatted_date} : 失敗 ({result.get('error')})")
            else:
                print(f"[RESULT] {formatted_date} : 成功")
                print(f"         獲取筆數：{result.get('rows')} 筆, NaN 數量：{result.get('nans')}")
                print(f"         動作：{result.get('action')}")
                
        except Exception as e:
             print(f"[ERROR] 回補 {formatted_date} 時發生嚴重例外: {e}")
             
        # 避免被證交所 Ban IP，強制睡 5 秒
        print("--- 等待 5 秒後繼續下一個日期 ---\n")
        time.sleep(5)
        
    print("=======================================")
    print("[INFO] 所有指定日期的歷史資料回補作業已結束！")
    print("=======================================")

if __name__ == "__main__":
    run_backfill_job()
