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
            os.remove(temp_file)

    for local, remote in [
        (f"{config.DATABASE_DIR}/revenue.csv",                                    "revenue/revenue.csv"),
        (f"{config.JOURNAL_DIR}/journal_quant.csv",                               "trading_record/trading_record.csv"),
        (f"{config.KITSUNE_QUANT_ADMINISTRATION_DIR}/transaction_quant.csv",      "transaction/transaction.csv"),
    ]:
        responses.append(backup_to_supabase('stock_data_backup', local, remote))

    print("\n所有資料上傳成功" if all(responses) else "\n部分資料上傳失敗")


if __name__ == '__main__':
    backup()