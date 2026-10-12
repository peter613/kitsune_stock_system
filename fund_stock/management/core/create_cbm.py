import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
import pandas as pd
import numpy as np
from catboost import CatBoostRanker, Pool, version
import os
import json
from datetime import datetime
import config
import matplotlib.pyplot as plt
import matplotlib
import shap
from fund_stock.management.pipeline._features import FEATURE_COLS

def plot_shap_values(model,test_pool,test_df,X_col,version):

    shap_results = model.get_feature_importance(data=test_pool, type='ShapValues')
    shap_values_array = shap_results[:, :-1]
    base_values = shap_results[:, -1]
    shap_obj = shap.Explanation(
        values=shap_values_array,        # SHAP 影響力數值
        base_values=base_values,         # 基準值
        data=test_df[X_col].values,      # 原始特徵數值 (用來決定紅高藍低)
        feature_names=X_col              # 特徵名稱
    )

    #繪圖: SHAP 柱狀圖
    print("繪製 SHAP 柱狀圖...")
    plot_dir = config.get_version_dir(config.STOCK_PLOTS_DIR, version)
    shap.plots.bar(shap_obj,max_display=15, show=False)
    plt.savefig(plot_dir / 'shap_bar.png', bbox_inches='tight')

    # 繪圖: SHAP 蜂群圖
    print("繪製 SHAP 蜂群圖...")
    plt.figure() 
    shap.plots.beeswarm(shap_obj,max_display=20, show=False)
    plt.savefig(plot_dir / 'shap_beeswarm.png', bbox_inches='tight')

    plt.show()

def create_cbm(version=None):
    # 1. 設定路徑

    cbm_version_dir = config.get_version_dir(config.STOCK_CBM_DIR, version)
    data_path = config.STOCK_TRAIN_FEATURES_DIR / f"train_{version}.csv"
    model_save_path = cbm_version_dir / "strategy.cbm"
    feature_save_path_json = cbm_version_dir / "feature_names.json"

    # 2. 讀取資料
    print(f"正在讀取資料: {data_path}")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values(['date', 'id']).reset_index(drop=True)

    feature_cols = FEATURE_COLS


    print(f"使用 {len(feature_cols)} 個特徵進行訓練")
    # 4. 切分訓練集 (前 80%) 與 驗證集 (後 20%)
    dates = df['date'].unique()
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]

    train_df = df[df['date'] < split_date]
    test_df = df[df['date'] >= split_date]

    # 5. 建立 Pool
    train_pool = Pool(
        data=train_df[feature_cols],
        label=train_df['label'],
        group_id=train_df['date'].astype('category').cat.codes
    )

    test_pool = Pool(
        data=test_df[feature_cols],
        label=test_df['label'],
        group_id=test_df['date'].astype('category').cat.codes
    )

    # 6. 訓練模型
    model = CatBoostRanker(
        iterations=6000,
        learning_rate=0.005,#原本0.001
        depth=5,#原本5
        loss_function='YetiRank',
        eval_metric='NDCG:top=5',
        custom_metric=['NDCG:top=5','PrecisionAt:top=5'],
        verbose=200,
        random_seed=42,
        thread_count=-1,
        use_best_model=True,
    )

    print("開始訓練模型...")
    model.fit(train_pool, eval_set=test_pool)

    # 7. 封存 (Save)
    # 儲存模型本體 (.cbm)
    model.save_model(model_save_path)
    print(f"模型已儲存至: {model_save_path}")

    # 儲存特徵列表 (.json)
    #這一步超級重要！回測時必須用「一模一樣的順序」餵資料，不然模型會錯亂
    with open(feature_save_path_json, 'w') as f:
        json.dump(feature_cols, f)
    print(f"特徵列表已備份至: {feature_save_path_json}")

    #shap
    plot_shap_values(model,test_pool,test_df,feature_cols,version)

if __name__ == "__main__":
    create_cbm(version='v60')