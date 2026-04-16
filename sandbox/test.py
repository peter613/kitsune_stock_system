import pandas as pd
import json
from catboost import CatBoostRanker, Pool
import shap
import matplotlib.pyplot as plt
import config

version = 19

# 1. 設定路徑 (以 V18 為例)
model_path = config.CBM_DIR / f"strategy_v{version}.cbm"
feature_path = config.CBM_DIR / f"feature_names_v{version}.json"
data_path = config.FEATURES_DIR / f"train_v{version}.csv"

# 2. 載入模型與特徵清單
with open(feature_path, 'r') as f:
    feature_cols = json.load(f)

model = CatBoostRanker()
model.load_model(model_path)
print("模型載入成功。")

# 3. 準備資料 (只需讀取測試集區間，節省記憶體)
df = pd.read_csv(data_path)
# 假設你還是要看最後 20% 的資料
dates = sorted(df['date'].unique())
split_idx = int(len(dates) * 0.8)
test_df = df[df['date'] >= dates[split_idx]]

test_pool = Pool(
    data=test_df[feature_cols],
    label=test_df['label'],
    group_id=test_df['date'].astype('category').cat.codes
)

def plot_shap_values(model,test_pool,test_df,X_col):

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
    shap.plots.bar(shap_obj,max_display=15, show=False)
    plt.savefig(config.PLOTS_REPORT_DIR/f'shap_bar_v{version}.png')

    # 繪圖: SHAP 蜂群圖
    print("繪製 SHAP 蜂群圖...")
    plt.figure() 
    shap.plots.beeswarm(shap_obj,max_display=15, show=False)
    plt.savefig(config.PLOTS_REPORT_DIR/f'shap_beeswarm_v{version}.png')

    print("繪製 Breakout 發動時的 R 與加分交互作用圖...")
    # 修改這一行：直接對 shap_obj 進行切片，它支援字串索引
    interaction_mask = shap_obj[:, "breakout_20"].values > 0
    shap.plots.scatter(shap_obj[interaction_mask, "r_20_abs"], color=shap_obj[interaction_mask, "breakout_20"], show=False)

    plt.show()

# 4. 重新執行繪圖函式 (直接呼叫你原本寫好的那個)
plot_shap_values(model, test_pool, test_df, feature_cols)