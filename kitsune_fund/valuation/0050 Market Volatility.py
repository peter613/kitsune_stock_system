import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick  # 新增 ticker 處理格式
import yfinance as yf
import config

# 1. 取得資料
try:
    df_0050 = pd.read_csv(config.DATABASE_DIR / '0050.csv')
    df_0050['date'] = pd.to_datetime(df_0050['date'])
    df_0050 = df_0050.sort_values('date').set_index('date')
    price_col = 'close'
except:
    print("找不到本地檔案，改從 yfinance 下載最新資料...")
    df_0050 = yf.download("0050.TW", start="2023-01-01")
    price_col = 'Close'

# 2. 計算波動率邏輯
df_0050['daily_return'] = df_0050[price_col].pct_change()
# 計算 20 日滾動年化波動率
df_0050['realized_vol'] = df_0050['daily_return'].rolling(window=20).std() * np.sqrt(252)

# 3. 繪圖
plt.figure(figsize=(15, 7))
plt.plot(df_0050.index, df_0050['realized_vol'], label='20D Realized Volatility', color='#1f77b4', linewidth=1.5)
plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.8, label='Black Swan Threshold (0.3)')
plt.fill_between(df_0050.index, df_0050['realized_vol'], 0.3, 
                 where=(df_0050['realized_vol'] >= 0.3), color='red', alpha=0.2)

# 圖表美化
plt.title('0050 Market Volatility Radar', fontsize=16)
plt.ylabel('Annualized Volatility (%)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')

# 修正：使用 PercentFormatter 解決警告，並設定 Y 軸格式
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.tight_layout()
plt.show()

# 檢查最後五筆，確認 NaN 結束的位置
print(df_0050['realized_vol'].tail(5))