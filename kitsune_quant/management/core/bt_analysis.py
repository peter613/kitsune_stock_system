import pandas as pd
import os 
import config
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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

def plot_distribution(port_returns, top_n=5, version='v20'):
    """繪製報酬分布直方圖並存檔"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Portfolio Monthly Returns Distribution (Top {top_n})', fontsize=14, fontweight='bold')
    
    # 直方圖 - 使用頻率（累計次數）而非密度
    n, bins, patches = ax.hist(port_returns, bins=30, density=False, alpha=0.7, color='steelblue', edgecolor='black', label='Frequency')
    
    # 使用 KDE 繪製實際分布曲線（同軸）
    kde = gaussian_kde(port_returns)
    x = np.linspace(port_returns.min(), port_returns.max(), 100)
    # 將 KDE 密度值縮放為頻率尺度
    kde_scaled = kde(x) * len(port_returns) * (bins[1] - bins[0])
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
    
    ax.set_xlabel('Monthly Return (%)', fontsize=11)
    ax.set_ylabel('Frequency (Count)', fontsize=11)
    ax.set_title(f'Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # 存檔
    plot_path = config.PLOTS_REPORT_DIR
    plot_file = plot_path / f"distribution_analysis_{version}_top{top_n}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"柱狀圖已存檔: {plot_file}")
    plt.show()

def bt_analysis(version='v20', top_n=5):
    file_path = config.BACKTEST_TRADELOG_DIR / f"trade_log_{version}_rank_1_2_3_4_5.csv"
    test_df = pd.read_csv(file_path, encoding='utf-8-sig')
    rank_skewness(test_df, top_n=top_n)
    port_returns, mean_ret, std_ret, skewness_value = portfolio_skewness(test_df, top_n=top_n, version=version)
    plot_distribution(port_returns, top_n=top_n, version=version)

if __name__ == '__main__':
    bt_analysis(version='v27', top_n=3)