import pandas as pd
import os
import config

data_path = config.JOURNAL_DIR / "journal_quant.csv"

def update_journal(id, name, action, label, price, share, date):
    # 1. 讀取舊資料 (加上檔案存在的防呆機制)
    if os.path.exists(data_path):
        df_old = pd.read_csv(data_path, encoding="utf-8-sig").dropna(axis=0, how='all')
    else:
        df_old = pd.DataFrame()

    # 2. 將新資料封裝為 DataFrame (確保數值型別正確)
    df_new = pd.DataFrame([{
        'date': date,
        'action': action,
        'id': str(id),
        'name': name,
        'price': float(price),
        'share': float(share),
        'label': label
    }])

    # 3. 垂直合併 (忽略原本的 index 重新排序)
    df_save = pd.concat([df_old, df_new], ignore_index=True)

    # 4. 覆蓋存檔
    df_save.to_csv(data_path, index=False, encoding="utf-8-sig")
    
    return f"交易已記錄: {date} | {action} | {name} | 價:{price} | 股:{share}"