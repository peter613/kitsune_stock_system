import matplotlib
matplotlib.use('TkAgg') # 確保視窗能彈出來
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
import config

file_path = config.BACKTEST_TRADELOG_DIR / "trade_log_v16_rank_1_2_3_4_5.csv"

# 讀取 CSV 交易日誌 (加入防呆編碼)
test_df = pd.read_csv(file_path, encoding='utf-8-sig')

def rank_skewness(test_df):
    # 迴圈畫出 Rank 1 到 5 的偏態圖與箱型圖
    for i in range(1, 6):
        rank_df = test_df[test_df['排名'] == i].copy()
        
        # 防呆：如果該名次沒有資料就跳過
        if rank_df.empty:
            continue

        # 計算偏態係數 (Skewness)
        skewness_value = rank_df['該股當月報酬%'].skew()

        # 最低程度修改：建立共用 X 軸的上下子圖 (高度比例 15% 給箱型圖，85% 給直方圖)
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(10, 6))
        
        # 繪製上方箱型圖
        sns.boxplot(x=rank_df['該股當月報酬%'], ax=ax_box, color='red')
        ax_box.set(xlabel='') # 隱藏箱型圖的 X 軸標籤以保持簡潔
        
        # 繪製下方帶有平滑曲線的直方圖 (KDE)
        sns.histplot(x=rank_df['該股當月報酬%'], bins=20, kde=True, ax=ax_hist, color='red')
        
        # 標題改成動態變數 {i}
        ax_hist.set_title(f"Rank {i} return distribution(Skewness: {skewness_value:.2f})")
        ax_hist.set_xlabel("month return (%)")
        ax_hist.set_ylabel("times")
        ax_hist.grid(alpha=0.3)

        # plt.show() 
        print(f"Rank {i} return distribution(Skewness: {skewness_value:.2f})")

def portfolio_skewness(test_df, top_n=3):
    # 1. 架構核心：先嚴格篩選出前 N 名的獨立資料
    df_top_n = test_df[test_df['排名'] <= top_n]

    # 2. 動態重算：用這 N 檔的個別報酬，重新算出「真正的 Top N 組合平均報酬」
    port_returns = df_top_n.groupby('換股日期')['該股當月報酬%'].mean()

    # 3. 計算真實的偏態係數
    skewness_value = port_returns.skew()

    # 4. 繪圖邏輯：加入箱型圖
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(10, 6))
    
    # 繪製上方箱型圖
    sns.boxplot(x=port_returns, ax=ax_box, color='purple')
    ax_box.set(xlabel='')
    
    # 繪製下方直方圖
    sns.histplot(x=port_returns, bins=20, kde=True, ax=ax_hist, color='purple')
    
    ax_hist.set_title(f"Top {top_n} Portfolio Return Distribution (Skewness: {skewness_value:.2f})")
    ax_hist.set_xlabel("Monthly Return (%)")
    ax_hist.set_ylabel("Frequency (Months)")
    ax_hist.grid(alpha=0.3)

    #plt.show() 
    print(f"Top {top_n} Portfolio Return Distribution (Skewness: {skewness_value:.2f})")

def main():
    rank_skewness(test_df)
    # 呼叫時指定只要前 5 名
    portfolio_skewness(test_df, top_n=5)

if __name__ == '__main__':
    main()