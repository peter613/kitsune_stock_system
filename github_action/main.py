import os
import sys

# 將專案根目錄加入路徑，解決找不到 config 模組的問題
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import config
from dotenv import load_dotenv

# 確保載入 .env
env_path = config.BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)

def send_discord_notify(message: str):
    """
    發送訊息到 Discord 指定頻道
    """
    webhook_url = os.environ.get("DC_WEBHOOK_URL")
    
    data = {"content": message}
    
    try:
        # 發送 POST 請求給 Discord
        response = requests.post(webhook_url, json=data)
        
        # Discord Webhook 成功通常回傳 204 No Content
        if response.status_code in (200, 204):
            print("Discord 通知發送成功！")
            return True
        else:
            print(f"Discord 通知發送失敗，狀態碼：{response.status_code}")
            return False
            
    except Exception as e:
        print(f"發送 Webhook 時發生異常：{e}")
        return False

def run_daily_job():
    print("[INFO] Starting Daily Automated Scraper...")
    
    # 這裡引入您寫的爬蟲模組
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from market_data.updater.market import update_market_data
    from market_data.sync_data import download_from_supabase
    from datetime import datetime
    
    current_year = datetime.now().year
    
    # 1. 根據執行環境決定資料庫準備方式 (防呆機制：保護本地資料)
    is_github_action = os.environ.get("GITHUB_ACTIONS") == "true"
    
    if is_github_action:
        print(f"[INFO] GitHub Actions 環境：正在從 Supabase 下載 {current_year} 年歷史資料...")
        remote_market = f"market/market_{current_year}.parquet"
        fake_local_db = str(config.MARKET_DATA_DIR / "market.parquet") 
        dl_success = download_from_supabase('stock_data_backup', remote_market, fake_local_db)
        
        if not dl_success:
            error_msg = (
                "每日股市資料更新報告\n"
                f"日期：{datetime.now().strftime('%Y-%m-%d')}\n"
                "狀態：異常\n\n"
                "‧ ERROR：無法從 Supabase 取得歷史資料庫，強制中斷以保護資料。"
            )
            send_discord_notify(error_msg)
            return
    else:
        print("[INFO] 本地測試環境：啟動安全同步機制 (防止舊年份資料被覆寫)...")
        from market_data.sync_data import sync
        try:
            sync()
        except Exception as e:
            print(f"[ERROR] 本地同步失敗：{e}")
        
    print("[INFO] Executing Market Scraper...")
    
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # 2. 執行市場更新並取得狀態回報
        result = update_market_data()
        
        # 3. 根據回傳的結果，組裝低熵、無雜訊的報告
        if result.get("status") in ["略過", "失敗", "取消"]:
            report = (
                "台股數據更新\n"
                f"日期：{date_str}\n"
                f"狀態：異常\n\n"
                f"‧ ERROR：{result.get('error')}"
            )
        else:
            report = (
                "台股數據更新\n"
                f"日期：{result.get('date')}\n"
                f"狀態：{result.get('status')}\n\n"
                f"‧ 獲取筆數：{result.get('rows')} 筆\n"
                f"‧ NaN 數量：{result.get('nans')}\n"
                f"‧ {result.get('action')}"
            )
            
    except Exception as e:
        # 防呆機制：捕捉未預期的爬蟲崩潰 (Crash)
        report = (
            "台股數據更新\n"
            f"日期：{date_str}\n"
            f"狀態：異常\n\n"
            f"‧ ERROR：{str(e)}"
        )
        
    # 4. 發送至 Discord
    send_discord_notify(report)
    print("[INFO] Daily job finished.")

if __name__ == "__main__":
    run_daily_job()