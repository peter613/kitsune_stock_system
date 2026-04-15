import pandas as pd
import numpy as np
from catboost import CatBoostRanker, Pool
import os
import json
from datetime import datetime
import config

def create_cbm(version=None):
    # 1. 設定路徑

    data_path = config.FEATURES_DIR / f"train_{version}.csv"
    model_save_path = config.CBM_DIR / f"strategy_{version}.cbm"
    feature_save_path_json = config.CBM_DIR / f"feature_names_{version}.json"

    # 2. 讀取資料
    print(f"正在讀取資料: {data_path}")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values(['date', 'id']).reset_index(drop=True)

    feature_cols = [
        #刪掉60日特徵 ("amount60","sma60","r2_60")

        # 均線與技術指標
        "sma20",  "sma240", "kama", "adx",
        
        # 趨勢強度
        "r2_20",
        
        # 流動性 (Log化)
        "amount20",
        
        # 籌碼與法人動能
        "trust_in_5", "t_f_sync",
        
        # 基本面
        "yoy_ma3_bias","yoy",

        #突破與築底
        "breakout_20", "breakout_240",
        "support_20","support_240",
    ]

    print(f"使用 {len(feature_cols)} 個特徵進行訓練")

    # 4. 切分訓練集 (前 80%) 與 驗證集 (後 20%)
    dates = df['date'].unique()
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]

    train_df = df[df['date'] < split_date]
    val_df = df[df['date'] >= split_date]

    # 5. 建立 Pool
    train_pool = Pool(
        data=train_df[feature_cols],
        label=train_df['label'],
        group_id=train_df['date'].astype('category').cat.codes
    )

    val_pool = Pool(
        data=val_df[feature_cols],
        label=val_df['label'],
        group_id=val_df['date'].astype('category').cat.codes
    )

    # 6. 訓練模型
    model = CatBoostRanker(
        iterations=10000,
        learning_rate=0.001,
        depth=6,
        loss_function='YetiRank',
        eval_metric='NDCG:top=5',
        # early_stopping_rounds=5000,
        verbose=100,
        random_seed=42,
        custom_metric=['PrecisionAt:top=5'],
    )

    print("開始訓練模型...")
    model.fit(train_pool, eval_set=val_pool)

    # 7. 封存 (Save)
    # 儲存模型本體 (.cbm)
    model.save_model(model_save_path)
    print(f"模型已儲存至: {model_save_path}")

    # 儲存特徵列表 (.json)
    #這一步超級重要！回測時必須用「一模一樣的順序」餵資料，不然模型會錯亂
    with open(feature_save_path_json, 'w') as f:
        json.dump(feature_cols, f)
    print(f"特徵列表已備份至: {feature_save_path_json}")

if __name__ == "__main__":
    create_cbm(version='v18')