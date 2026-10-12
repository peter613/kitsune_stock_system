import os
import sys
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv

# 把專案根目錄加入 sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
import config

# 載入 .env
env_path = BASE_DIR / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


def send_database_notify(message: str) -> bool:
    """發送資料庫維運狀態至指定頻道"""
    webhook_url = os.environ.get("DC_DATABASE_WEBHOOK_URL") or os.environ.get("DC_WEBHOOK_URL")
    if not webhook_url:
        print("[WARN] 未設定 DC_DATABASE_WEBHOOK_URL / DC_WEBHOOK_URL")
        return False

    try:
        res = requests.post(webhook_url, json={"content": message}, timeout=15)
        if res.status_code not in (200, 204):
            print(f"[ERROR] Discord Webhook 發送失敗 ({res.status_code}): {res.text}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] 發送資料庫通知失敗: {e}")
        return False


def run_update_pipeline():
    print("[INFO] Starting Market Data & NAV Update Pipeline...")

    from market_data.updater.market import update_market_data
    from market_data.sync_data import download_from_supabase
    from backup_db import backup_to_supabase
    from fund_stock.administration.update_navpu import update_quant

    current_year = datetime.now().year
    date_str = datetime.now().strftime('%Y-%m-%d')
    is_github_action = os.environ.get("GITHUB_ACTIONS") == "true"

    # 1. 下載當年歷史資料庫
    if is_github_action:
        remote_market = f"market/market_{current_year}.parquet"
        fake_local_db = str(config.MARKET_DATA_DIR / "market.parquet")
        dl_success = download_from_supabase('stock_data_backup', remote_market, fake_local_db)

        if not dl_success:
            report = (
                "台股數據更新報告\n"
                f"日期：{date_str}\n"
                "狀態：**異常**\n\n"
                "‧ ERROR：無法從 Supabase 取得歷史資料庫"
            )
            send_database_notify(report)
            return False
    else:
        from market_data.sync_data import sync
        try:
            sync()
        except Exception as e:
            print(f"[ERROR] 本地同步失敗: {e}")

    # 2. 爬取今日市場數據
    try:
        result = update_market_data()
    except Exception as e:
        report = (
            "台股數據更新報告\n"
            f"日期：{date_str}\n"
            "狀態：**異常**\n\n"
            f"‧ ERROR：爬蟲異常崩潰 ({str(e)})"
        )
        send_database_notify(report)
        return False

    if result.get("status") == "略過":
        print(f"[INFO] 略過訊號 (原因: {result.get('error')})，不發送通知。")
        return True

    if result.get("status") == "失敗":
        report = (
            "台股數據更新報告\n"
            f"日期：{date_str}\n"
            "狀態：**異常**\n\n"
            f"‧ ERROR：{result.get('error')}"
        )
        send_database_notify(report)
        return False

    # 3. 計算今日基金淨值 (navpu.csv)
    try:
        update_quant()
    except Exception as e:
        report = (
            "台股數據更新報告\n"
            f"日期：{date_str}\n"
            "狀態：**異常**\n\n"
            f"‧ ERROR：基金淨值計算異常 ({str(e)})"
        )
        send_database_notify(report)
        return False

    # 4. 備份 navpu.csv 至 Supabase
    navpu_file = str(config.FUND_STOCK_ADMINISTRATION_DIR / "navpu.csv")
    backup_ok = backup_to_supabase('stock_data_backup', navpu_file, '/navpu.csv')
    if not backup_ok:
        report = (
            "台股數據更新報告\n"
            f"日期：{date_str}\n"
            "狀態：**異常**\n\n"
            "‧ ERROR：navpu.csv 備份至 Supabase 失敗"
        )
        send_database_notify(report)
        return False

    # 5. 發送純粹、低熵的資料庫報告 (無 emoji)
    db_report = (
        "台股數據更新報告\n"
        f"日期：{result.get('date')}\n"
        "狀態：**成功**\n\n"
        f"‧ 獲取筆數：{result.get('rows')} 筆\n"
        f"‧ NaN 數量：{result.get('nans')}"
    )
    send_database_notify(db_report)
    print("[INFO] Market Data & NAV Update Pipeline completed successfully.")
    return True


if __name__ == "__main__":
    success = run_update_pipeline()
    if not success:
        sys.exit(1)