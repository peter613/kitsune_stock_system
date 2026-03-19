import pandas as pd
import config
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from talib import SMA

# 1. 資料載入與預處理
market_path = config.DATABASE_DIR / "market.parquet"
df = pd.read_parquet(market_path)
df = df.sort_values(['id', 'date'])

# 2. 特徵計算函數 (確保 return group)
def compute_all_features(group):
    # 轉換為 float 以符合 TA-Lib 要求
    c = group['close'].values.astype(float)
    
    # 計算均線
    periods = [5, 10, 20, 60]
    for p in periods:
        group[f'ma{p}'] = SMA(c, timeperiod=p)
        
    return group

# 執行分組計算 (group_keys=False 保持索引乾淨)
df = df.groupby('id', group_keys=False).apply(compute_all_features)

# 3. 篩選特定標的與日期
target_id = '2330'
target_date = '2020-01-01'
# 確保 date 欄位為 datetime 格式後進行篩選
df['date'] = pd.to_datetime(df['date'])
target_df = df[(df['id'] == target_id) & (df['date'] > pd.to_datetime(target_date))].copy()

# 4. 建立圖表 (使用子圖：Row 1 價格, Row 2 成交量)
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    row_heights=[0.7, 0.3]
)

# A. 加入 K 線圖 (Row 1)
fig.add_trace(go.Candlestick(
    x=target_df['date'],
    open=target_df['open'],
    high=target_df['high'],
    low=target_df['low'],
    close=target_df['close'],
    name='K線'
), row=1, col=1)

# B. 自動加入所有 MA 均線 (Row 1)
ma_cols = [c for c in target_df.columns if c.startswith('ma')]
for col in ma_cols:
    fig.add_trace(go.Scatter(
        x=target_df['date'],
        y=target_df[col],
        mode='lines',
        name=col.upper(),
        line=dict(width=1)
    ), row=1, col=1)

# C. 加入成交量柱狀圖 (Row 2)
colors = ['red' if c >= o else 'green' for c, o in zip(target_df.close, target_df.open)]
fig.add_trace(go.Bar(
    x=target_df['date'],
    y=target_df['volume'],
    name='成交量',
    marker_color=colors,
    opacity=0.7
), row=2, col=1)

# 5. 佈局設定與移除假日
fig.update_layout(
    title=f'Stock Analysis: {target_id}',
    xaxis_rangeslider_visible=False,
    template='plotly_white',
    height=800,
    showlegend=True
)

# 移除週末與空白交易日
fig.update_xaxes(
    rangebreaks=[
        dict(bounds=["sat", "mon"]), # 移除週六、週日
    ]
)

fig.show()