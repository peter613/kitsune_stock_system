import pandas as pd
import os 
import config
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def rank_skewness(test_df,top_n=5):
    print("\n--- Individual Rank Skewness ---")
    for i in range(1, top_n+1):
        rank_df = test_df[test_df['排名'] == i].copy()
        
        if rank_df.empty:
            continue

        skewness_value = rank_df['該股當月報酬%'].skew()
        print(f"Rank {i} Skewness : {skewness_value:.4f}")

def portfolio_skewness(test_df, top_n=5,version='v25'):
    # 1. 嚴格篩選出前 N 名的獨立資料
    df_top_n = test_df[test_df['排名'] <= top_n]

    # 2. 動態重算：算出「Top N 組合平均報酬」(Time-Series)
    port_returns = df_top_n.groupby('換股日期')['該股當月報酬%'].mean()

    # --- 整合：Time-Series 統計與風險分析 ---
    mean_ret = port_returns.mean()
    std_ret = port_returns.std()
    skewness_value = port_returns.skew()
    kurtosis_value = port_returns.kurtosis()
    
    # 計算 MDD (最大回撤)
    cum_returns = (1 + port_returns / 100).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / running_max * 100
    mdd = drawdowns.min()

    # 使用 IQR (四分位距) 衡量異常值 - 對偏態分布更穩健
    q1 = port_returns.quantile(0.25)
    q3 = port_returns.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    # Tukey's Method for outliers
    outliers = port_returns[(port_returns < lower_fence) | (port_returns > upper_fence)]

    # 產出終端機量化分析報告
    print("\n" + "="*60)
    print(f"Kitsune System Quantitative Analysis {version}(Top {top_n} Portfolio)")
    print("="*60)
    print(f"Sample Size (Months)     : {len(port_returns)}")
    print(f"Monthly Mean             : {mean_ret:.2f}%")
    print(f"Monthly Std              : {std_ret:.2f}%")
    print(f"System Skewness          : {skewness_value:.4f}")
    print(f"System Kurtosis          : {kurtosis_value:.4f} (常態分佈為3)")
    print(f"Maximum Drawdown         : {mdd:.2f}%")
    print("-"*60)
    print(f'''使用 IQR 方法衡量異常值:
IQR Lower Fence (Q1-1.5*IQR)   : {lower_fence:.2f}%
IQR Upper Fence (Q3+1.5*IQR)   : {upper_fence:.2f}%
異常月份數量/比例              : {len(outliers)} ({len(outliers)/len(port_returns):.1%})
''')
    if len(outliers) > 0:
        print(f"異常月份報酬 : {[f'{x:.2f}%' for x in outliers.values]}")
    print("="*60 + "\n")
    
    return port_returns, mean_ret, std_ret, skewness_value

def plot_distribution(port_returns, top_n=5, version='v27'):
    """繪製報酬分布直方圖並存檔 (連續刻度邊界對齊版 - 2.5% 級距)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Portfolio Monthly Returns Distribution (Top {top_n})', fontsize=14, fontweight='bold')
    
    # 1. 建立明確的連續數值區間 (低熵原則：拔除 math 模組，使用原生 // 運算)
    bin_width = 2.5
    
    # 使用原生的 // (向下取整除法) 來鎖死邊界
    min_val = (port_returns.min() // bin_width) * bin_width
    # 針對最大值，直接向下取整後 +1 個單位，即可達成超越極值的包覆效果
    max_val = ((port_returns.max() // bin_width) + 1) * bin_width
    
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    
    # 2. 直方圖 - 帶入自訂的 bins 陣列
    n, _, patches = ax.hist(port_returns, bins=bins, density=False, alpha=0.7, 
                            color='steelblue', edgecolor='white', label='Frequency')
    
    # 3. 使用 KDE 繪製實際分布曲線（同軸）
    kde = gaussian_kde(port_returns)
    x = np.linspace(min_val, max_val, 200)
    # 將 KDE 密度值縮放為頻率尺度
    kde_scaled = kde(x) * len(port_returns) * bin_width
    ax.plot(x, kde_scaled, 'r-', linewidth=2.5, label='Actual Distribution (KDE)')
    
    # 添加統計信息
    skewness = port_returns.skew()
    kurtosis = port_returns.kurtosis()
    
    # 計算 IQR
    q1 = port_returns.quantile(0.25)
    q3 = port_returns.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    # 中位數
    median = port_returns.median()
    
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.2f}%')
    ax.axvline(lower_fence, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label=f'IQR Fence Lower: {lower_fence:.2f}%')
    ax.axvline(upper_fence, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label=f'IQR Fence Upper: {upper_fence:.2f}%')
    
    # 4. 強制 X 軸刻度對齊柱子的邊界
    ax.set_xticks(bins)
    # 因為級距是 2.5，改用 .1f 顯示小數點第一位，讓視覺依然純粹
    ax.set_xticklabels([f"{b:.1f}" for b in bins], rotation=45, ha='right', fontsize=9)
    # 左右留白半個柱子的空間
    ax.set_xlim(min_val - (bin_width / 2), max_val + (bin_width / 2))
    
    ax.set_xlabel('Monthly Return (%)', fontsize=11)
    ax.set_ylabel('Frequency (Count)', fontsize=11)
    ax.set_title(f'Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}', fontsize=11)
    
    ax.legend(loc='upper right', fontsize=9) 
    ax.grid(axis='y', alpha=0.3) 
    
    plt.tight_layout()
    
    # 存檔
    plot_dir = config.get_version_dir(config.STOCK_PLOTS_DIR, version)
    plot_file = plot_dir / f"distribution_analysis_top{top_n}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"柱狀圖已存檔: {plot_file}")
    
    plt.close()

def plot_score_distribution(test_df, version='v27', top_n=3):
    # 1. 建立分數區間 (以 0.05 為單位)
    bin_width = 0.05
    bins = np.arange(0, 1.0 + bin_width, bin_width)
    
    # 2. 資料處理：標記勝負與分箱
    df_plot = test_df.copy()
    df_plot['Result'] = np.where(df_plot['該股當月報酬%'] > 0, 'Win', 'Loss')
    df_plot['Score_Bin'] = pd.cut(df_plot['預測分數'], bins=bins, right=False)
    
    # 3. 聚合計算次數
    agg_df = df_plot.groupby('Score_Bin', observed=False)['Result'].value_counts().unstack(fill_value=0)
    
    for col in ['Win', 'Loss']:
        if col not in agg_df.columns:
            agg_df[col] = 0
            
    agg_df['Total'] = agg_df['Win'] + agg_df['Loss']
    
    # 剔除無交易次數的區間
    agg_df = agg_df[agg_df['Total'] > 0]
    
    # 4. 繪製圖表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 取得每個區間的「中心點」作為柱子的 X 座標 (例如 0.35 與 0.40 的中心是 0.375)
    x_centers = [interval.mid for interval in agg_df.index]
    
    # 繪製堆疊柱狀圖：設定 width 為區間寬度 (0.05)，讓柱子與刻度完美貼合
    ax.bar(x_centers, agg_df['Loss'], width=bin_width, label='Loss (<= 0%)', color='#d62728', edgecolor='white')
    ax.bar(x_centers, agg_df['Win'], width=bin_width, bottom=agg_df['Loss'], label='Win (> 0%)', color='#1f77b4', edgecolor='white')
    
    # 將次數與佔比結構標示於對應區塊內
    for i, center in enumerate(x_centers):
        total = agg_df['Total'].iloc[i]
        win_count = agg_df['Win'].iloc[i]
        loss_count = agg_df['Loss'].iloc[i]
        
        if loss_count > 0:
            loss_pct = (loss_count / total) * 100
            ax.text(center, loss_count / 2, f"{loss_count}\n({loss_pct:.0f}%)", 
                    ha='center', va='center', color='white', fontsize=9)
            
        if win_count > 0:
            win_pct = (win_count / total) * 100
            ax.text(center, loss_count + (win_count / 2), f"{win_count}\n({win_pct:.0f}%)", 
                    ha='center', va='center', color='white', fontsize=9)
    
    # 5. 格式設定 (將刻度線精準設定在區間的邊界)
    # 動態取得當下有資料的最小與最大邊界，確保畫面不過度留白
    min_edge = agg_df.index[0].left
    max_edge = agg_df.index[-1].right
    
    # 產生要顯示的刻度 (例如 0.35, 0.40, 0.45...)
    active_bins = np.arange(min_edge, max_edge + bin_width, bin_width)
    
    ax.set_xticks(active_bins)
    ax.set_xticklabels([f"{b:.2f}" for b in active_bins], rotation=0, ha='center', fontsize=10)
    
    # 設定 X 軸的顯示範圍，左右各留半個柱子的寬度
    ax.set_xlim(min_edge - (bin_width / 2), max_edge + (bin_width / 2))
    
    ax.set_xlabel('Prediction Score Threshold', fontsize=11)
    ax.set_ylabel('Frequency (Count)', fontsize=11)
    ax.set_title(f'Score Distribution (Version: {version}, Top: {top_n})', fontsize=12)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 6. 存檔與清理
    plot_dir = config.get_version_dir(config.STOCK_PLOTS_DIR, version)
    plot_file = plot_dir / f"score_distribution_top{top_n}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"分數柱狀圖已存檔: {plot_file}")

def bt_analysis(version='v27', top_n=3):
    file_path = config.get_version_dir(config.STOCK_BACKTEST_TRADELOG_DIR, version) / "trade_log_rank_1_2_3.csv"
    test_df = pd.read_csv(file_path, encoding='utf-8-sig')
    rank_skewness(test_df, top_n=top_n)
    port_returns, mean_ret, std_ret, skewness_value = portfolio_skewness(test_df, top_n=top_n, version=version)
    plot_distribution(port_returns, top_n=top_n, version=version)
    plot_score_distribution(test_df, version=version, top_n=top_n)

if __name__ == '__main__':
    bt_analysis(version='v30', top_n=3)