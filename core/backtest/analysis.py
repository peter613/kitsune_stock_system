import pandas as pd
import os 
import config

# 讀取 CSV 交易日誌 (加入防呆編碼)
file_path = config.BACKTEST_TRADELOG_DIR / "trade_log_v17_rank_1_2_3_4_5.csv"
test_df = pd.read_csv(file_path, encoding='utf-8-sig')

def rank_skewness(test_df):
    print("\n--- Individual Rank Skewness ---")
    for i in range(1, 6):
        rank_df = test_df[test_df['排名'] == i].copy()
        
        if rank_df.empty:
            continue

        skewness_value = rank_df['該股當月報酬%'].skew()
        print(f"Rank {i} Skewness : {skewness_value:.4f}")

def portfolio_skewness(test_df, top_n=5):
    # 1. 嚴格篩選出前 N 名的獨立資料
    df_top_n = test_df[test_df['排名'] <= top_n]

    # 2. 動態重算：算出「Top N 組合平均報酬」(Time-Series)
    port_returns = df_top_n.groupby('換股日期')['該股當月報酬%'].mean()

    # --- 整合：Time-Series 統計與風險分析 ---
    mean_ret = port_returns.mean()
    std_ret = port_returns.std()
    skewness_value = port_returns.skew()
    
    lower_bound = mean_ret - 2 * std_ret
    upper_bound = mean_ret + 2 * std_ret
    
    # 計算 MDD (最大回撤)
    cum_returns = (1 + port_returns / 100).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / running_max * 100
    mdd = drawdowns.min()

    # 產出終端機量化分析報告
    print("\n" + "="*50)
    print(f"Kitsune System Quantitative Analysis (Top {top_n} Portfolio)")
    print("="*50)
    print(f"Sample Size (Months) : {len(port_returns)}")
    print(f"Monthly Mean         : {mean_ret:.2f}%")
    print(f"Monthly Std          : {std_ret:.2f}%")
    print(f"System Skewness      : {skewness_value:.4f}")
    print(f"VaR Lower Bound (-2σ): {lower_bound:.2f}% (95.4% Confidence)")
    print(f"VaR Upper Bound (+2σ): {upper_bound:.2f}%")
    print(f"Maximum Drawdown     : {mdd:.2f}%")
    print("="*50 + "\n")

def main():
    rank_skewness(test_df)
    portfolio_skewness(test_df, top_n=5)

if __name__ == '__main__':
    main()