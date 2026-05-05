import config
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import pandas as pd
from datetime import datetime

env_path = config.BASE_DIR/'.env'
load_dotenv(dotenv_path=env_path)

# 2. 讀取環境變數 (括號內的名稱必須與 .env 檔案中設定的一致)
URL = os.environ.get("SUPABASE_URL")
KEY = os.environ.get("SUPABASE_KEY")

# 檢查是否有成功讀取到 (若無設定會回傳 None)
if not URL or not KEY:
    raise ValueError("找不到 SUPABASE_URL 或 SUPABASE_SECRET_KEY，請確認 .env 檔案設定")

# 3. 建立 Supabase 連線
supabase: Client = create_client(URL, KEY)

# 4. 執行備份的函式
def backup_to_supabase(bucket_name: str, local_file_path: str, supabase_file_path: str):
    try:
        with open(local_file_path, "rb") as f:
            supabase.storage.from_(bucket_name).upload(
                path=supabase_file_path,
                file=f,
                file_options={"x-upsert": "true"}
            )
        return True  # 成功
    except Exception as e:
        print(f"上傳失敗: {e}")
        return False


def cloud_has_data(bucket_name: str) -> bool:
    try:
        items = supabase.storage.from_(bucket_name).list(path="")
        return bool(items)
    except Exception as e:
        print(f"檢查雲端資料失敗: {e}")
        return False


def backup():
    current_year = datetime.now().year
    df = pd.read_parquet(f"{config.DATABASE_DIR}/market.parquet")
    df['year'] = df['date'].dt.year

    initialized = cloud_has_data('stock_data_backup')
    if initialized:
        print("雲端已有資料，僅上傳當前年份")
        years_to_upload = {current_year}
    else:
        print("雲端無資料，初始化上傳所有年份")
        years_to_upload = set(df['year'].unique())

    responses = []
    for year, group in df.groupby('year'):
        if year in years_to_upload:
            temp_file = f"{config.DATABASE_DIR}/market_{year}.parquet"
            group.drop(columns=['year']).to_parquet(temp_file)
            response = backup_to_supabase(
                'stock_data_backup',
                temp_file,
                f"market/market_{year}.parquet"
            )
            responses.append(response)
            os.remove(temp_file)  # 刪除臨時文件

    response_revenue = backup_to_supabase(
        'stock_data_backup',
        f"{config.DATABASE_DIR}/revenue.csv",
        "revenue/revenue.csv"
    )
    responses.append(response_revenue)

    if all(responses):
        print("資料上傳成功")
    else:
        print("資料上傳失敗")


if __name__ == '__main__':
    backup()
