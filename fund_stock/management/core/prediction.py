import pandas as pd
import numpy as np
from catboost import CatBoostRanker
import os
import sys
from pathlib import Path

# 將專案根目錄加入 sys.path，避免 ModuleNotFoundError
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
import config
from fund_stock.management.core.position_sizing import format_allocation_report

def search_stock(stock_name: str, version='v20') -> str:
    """
    根據股票名稱查詢排名
    例如: search_stock('台光電') 會告訴你他的排名和排名多少
    """
    # 1. 設定與載入
    MODEL_FILE = config.get_version_dir(config.STOCK_CBM_DIR, version)/"strategy.cbm"
    DATA_FILE = config.STOCK_FEATURES_DIR/"pred.csv"
    
    if not os.path.exists(MODEL_FILE):
        return f"❌ 錯誤：找不到模型檔案 {MODEL_FILE}"
    
    if not os.path.exists(DATA_FILE):
        return f"❌ 錯誤：找不到資料檔案 {DATA_FILE}"
    
    model = CatBoostRanker()
    model.load_model(MODEL_FILE)
    feature_cols = model.feature_names_

    # 2. 資料讀取
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()

    target_df = df[df['date'] == latest_date].copy().reset_index(drop=True)
    target_df = target_df.fillna(0)

    # 3. 預測與排序
    if target_df.empty:
        return "❌ 錯誤：找不到最新日期的資料。"

    missing_cols = [c for c in feature_cols if c not in target_df.columns]
    if missing_cols:
        return f"目前的 pred.csv 缺少模型 ({version}) 所需的特徵。"
    
    scores = model.predict(target_df[feature_cols])
    target_df['predict_score'] = scores 
    target_df = target_df.sort_values('predict_score', ascending=False).reset_index(drop=True)
    target_df['rank'] = range(1, len(target_df) + 1)

    # 4. 搜尋股票
    # 搜尋名稱或代碼
    search_result = target_df[
        (target_df['name'].str.contains(stock_name, na=False)) | 
        (target_df['id'].astype(str).str.contains(stock_name, na=False))
    ]

    if search_result.empty:
        return f"❌ 找不到 '{stock_name}' 股票。請檢查股票名稱或代碼。"
    
    # 取得搜尋結果
    result_row = search_result.iloc[0]
    rank = int(result_row['rank'])
    total_stocks = len(target_df)
    stock_id = result_row['id']
    stock_name_display = result_row['name']
    score = result_row['predict_score']
    
    # 計算百分比排名
    percentile = (1 - rank/total_stocks) * 100
    
    # 5. 特徵分析 - 找出強勢和弱勢特徵
    analysis_df = target_df.copy()
    valid_features = []
    
    for col in feature_cols:
        if col in analysis_df.columns and pd.api.types.is_numeric_dtype(analysis_df[col]):
            analysis_df[f"{col}_pct"] = analysis_df[col].rank(pct=True)
            valid_features.append(col)

    result_str = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
股票代碼: {stock_id}
股票名稱: {stock_name_display}
預測得分: {score:.4f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
排名: 第 {rank} 名（共 {total_stocks} 檔股票）
排名百分位: 前 {(100-percentile):.1f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return result_str


def prediction(version='v8', select_ranks=[1, 2, 3, 4, 5]) -> str: # 加上型別提示
    # 1. 設定與載入
    MODEL_FILE = config.get_version_dir(config.STOCK_CBM_DIR, version)/"strategy.cbm"
    DATA_FILE = config.STOCK_FEATURES_DIR/"pred.csv"
    
    print(f"正在載入模型: {MODEL_FILE}")
    if not os.path.exists(MODEL_FILE):
        error_msg = f"❌ 錯誤：找不到模型檔案 {MODEL_FILE}"
        print(error_msg)
        return error_msg

    try:
        model = CatBoostRanker()
        model.load_model(MODEL_FILE)
        feature_cols = model.feature_names_
    except Exception as e:
        error_msg = f"❌ 錯誤：載入模型失敗 ({e})"
        print(error_msg)
        return error_msg

    # 2. 資料讀取
    if not os.path.exists(DATA_FILE):
        error_msg = f"❌ 錯誤：找不到資料檔案 {DATA_FILE}"
        print(error_msg)
        return error_msg

    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        error_msg = f"❌ 錯誤：讀取 {DATA_FILE} 失敗 ({e})"
        print(error_msg)
        return error_msg
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()

    target_df = df[df['date'] == latest_date].copy().reset_index(drop=True)
    target_df = target_df.fillna(0)
    
    print(f"資料日期: {latest_date.date()} | 篩選排名: {select_ranks}")
    if 'market_kama_mean' in target_df.columns:
        mk_val = float(target_df['market_kama_mean'].iloc[0])
        print(f"\n{format_allocation_report(mk_val)}")

    # 3. 預測與排序
    if target_df.empty:
        return "❌ 錯誤：找不到最新日期的資料。"

    missing_cols = [c for c in feature_cols if c not in target_df.columns]
    if missing_cols:
        error_msg = f"❌ 錯誤：目前的 pred.csv 缺少模型 ({version}) 所需的特徵：\n{missing_cols}\n\n請先執行 train.py 產生新特徵，再執行 create_cbm.py 重新訓練模型！"
        print(error_msg)
        return error_msg
    
    scores = model.predict(target_df[feature_cols])
    target_df['predict_score'] = scores 
    target_df = target_df.sort_values('predict_score', ascending=False).reset_index(drop=True)
    target_df['rank'] = range(1, len(target_df) + 1)

    #--- 核心邏輯修改：改為回傳字串而不是單純 print ---
    def get_feature_analysis_string(all_df, ranks_to_show, features_to_analyze) -> str:
        analysis_df = all_df.copy()
        valid_features = []
        
        # 計算百分位
        for col in features_to_analyze:
            if col in analysis_df.columns and pd.api.types.is_numeric_dtype(analysis_df[col]):
                analysis_df[f"{col}_pct"] = analysis_df[col].rank(pct=True)
                valid_features.append(col)
        
        display_df = analysis_df[analysis_df['rank'].isin(ranks_to_show)].copy()
        
        for _, row in display_df.iterrows():
            stock_info = f"Rank {int(row['rank'])}. [{row['id']}] {row['name']} (得分: {row['predict_score']:.4f})"
            print(f"\n{stock_info}") 
                        
            strong_feats = []
            for feat in valid_features:
                pct_val = row[f"{feat}_pct"]
                if pct_val > 0.75:
                    strong_feats.append((feat, pct_val))
            
            strong_feats = sorted(strong_feats, key=lambda x: x[1], reverse=True)
            
            if strong_feats:
                output_str = ", ".join([f"{k}(PR{v*100:.1f})" for k, v in strong_feats])
                print(output_str) # 保留螢幕輸出
            else:
                print("指標分佈平均，無顯著單一強項。")
            
    # 執行分析，並把結果存進變數
    get_feature_analysis_string(target_df, select_ranks, feature_cols)

    # 4. 存檔
    top_picks = target_df[target_df['rank'].isin(select_ranks)].copy()
    prediction_dir = config.get_version_dir(config.STOCK_PREDICTION_REPORT_DIR, version)
    file_name = f'{latest_date.date().strftime("%Y-%m-%d")}.csv'
    output_dir = prediction_dir / file_name
    try:
        top_picks[['rank', 'id', 'name', 'predict_score', 'close']].to_csv(output_dir, index=False, encoding='utf-8-sig')
        print(f"\n預測結果已成功儲存至: {output_dir}")
    except PermissionError:
        print(f"\n錯誤：無法儲存檔案。請確認 {file_name} 是否正被 Excel 或其他程式開啟！")
    except Exception as e:
        print(f"\n錯誤：儲存檔案時發生例外狀況 ({e})")

if __name__ == "__main__":
    #print(search_stock('台光電', version='v33'))
    result = prediction(version='v38', select_ranks=[c for c in range(1,6)])
    if isinstance(result, str):
        print(result)