import config
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import pandas as pd
from datetime import datetime

env_path = config.BASE_DIR/'.env'
load_dotenv(dotenv_path=env_path)

URL = os.environ.get("SUPABASE_URL")
KEY = os.environ.get("SUPABASE_KEY")

if not URL or not KEY:
    raise ValueError("找不到 SUPABASE_URL 或 SUPABASE_SECRET_KEY，請確認 .env 檔案設定")

supabase: Client = create_client(URL, KEY)


def backup_to_supabase(bucket_name: str, local_file_path: str, supabase_file_path: str):
    label = os.path.basename(local_file_path)
    try:
        with open(local_file_path, "rb") as f:
            supabase.storage.from_(bucket_name).upload(
                path=supabase_file_path,
                file=f,
                file_options={"x-upsert": "true"}
            )
        print(f"{label} 上傳成功")
        return True
    except Exception as e:
        print(f"{label} 上傳失敗: {e}")
        return False


def cloud_has_data(bucket_name: str) -> bool:
    try:
        items = supabase.storage.from_(bucket_name).list(path="")
        return bool(items)
    except Exception as e:
        print(f"檢查雲端資料失敗: {e}")
        return False


def backup():
    print("開始備份個人帳務與營收資料到雲端...")
    responses = []

    for local, remote in [
        (f"{config.FUND_STOCK_ADMINISTRATION_DIR}/navpu.csv",                  "/navpu.csv"),
        (f"{config.MARKET_DATA_DIR}/revenue.csv",                              "/revenue.csv"),
        (f"{config.STOCK_JOURNAL_DIR}/journal.csv",                            "/trading_record.csv"),
        (f"{config.FUND_STOCK_ADMINISTRATION_DIR}/transaction.csv",            "/transaction.csv"),
    ]:
        if os.path.exists(local):
            responses.append(backup_to_supabase('stock_data_backup', local, remote))
        else:
            print(f"找不到本地檔案: {local}")

    if responses:
        print("備份作業完畢" if all(responses) else "部分資料上傳失敗")


if __name__ == '__main__':
    backup()