import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import config

st.set_page_config(page_title="Kitsune Kernel", layout="wide")

# 全域 CSS 注入：統一壓縮標題與子標題的高度與字體大小
st.markdown("""
    <style>
    /* 側邊欄寬度控制 */
    [data-testid='stSidebar'] {
        width: 200px !important;
    }
    
    /* 1. 將 st.title (預設 h1) 降級為 h3 視覺大小，並極限壓縮頂部高度 */
    .stApp h1 {
        font-size: 1.5rem !important;      
        margin-top: -50px !important;      
        margin-bottom: 0px !important;     
        padding-top: 0rem !important;
        padding-bottom: 1rem !important;
        font-weight: 600 !important;
    }

    /* 2. 將 st.subheader (預設 h3) 降級為 h4 視覺大小，並壓縮間距 */
    .stApp h3 {
        font-size: 1.2rem !important;      /* h4 的精緻字體大小 */
        margin-top: -10px !important;      /* 稍微往上拉，減少與上方表格/分隔線的距離 */
        margin-bottom: 0px !important;     /* 消除與下方 Dataframe 的空白 */
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        font-weight: 500 !important;
    }

    /* 3. 針對 st.metric 標題 (大字段落 <p>) 的精準渲染 */
    [data-testid="stMetricLabel"] p {
        font-size: 1.2rem !important;  /* 調整為你適合的大小 */
        font-weight: 400 !important;   /* 增加厚度 */
        color:#101010 !important;    
        opacity: 1 !important;         /* 徹底解決灰濛濛的透明度問題 */
        margin-bottom: 10px !important; /* 壓縮標題與數字之間的間距 */
    }

    /* 4. 針對 st.metric 數字 (小字區塊 <div>) 的精準渲染 */
    [data-testid="stMetricValue"] div {
        font-size: 2.2rem !important;
        font-weight: 600 !important;
        color:#db3b00 !important;
        opacity: 0.95 !important; /* 調整為 0.0 ~ 1.0 之間 */
    }
    /* 5. 修正側邊欄進度小字 (Caption) 的亮度與大小 */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #000000 !important;     /* 移除引號，強制亮灰色 */
        opacity: 1 !important;         /* 徹底取消透明度 */
        font-size: 1.0rem !important;  /* 稍微放大一點 */
    }

    /* 6. 調整側邊欄導航 (st.radio) 的標題大小：例如「選單」 */
    [data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #000000 !important;
    }

    /* 7. 調整側邊欄導航的「選項文字」：例如「累積報酬率」 */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem !important;  /* 調整選項字體 */
        color: #000000 !important;
    }


    /* 8. 側邊欄背景色：很淡很淡的 (#EDE6DB) */
    [data-testid="stSidebar"] {
        background-color: #fcf5ed !important;
    }
    
    /* 9. 調整導航選項間距：讓選項之間稍微拉開一點，不要太擠 */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        margin-bottom: 10px !important;
    }

    </style>
    """, unsafe_allow_html=True)

START_DATE = '2026-03-11'
init_cap = 335800
# ==========================================
# 模組 1：資料讀取 (嚴格過濾 3/11 之後)
# ==========================================
@st.cache_data
def load_data():
    file_path_record = config.JOURNAL_DIR/"journal.csv"
    
    # 讀取交易紀錄 (維持 price 邏輯)
    df_record = pd.read_csv(file_path_record, dtype={'id': str})
    df_record.columns = df_record.columns.str.strip().str.lower()
    df_record['date'] = pd.to_datetime(df_record['date'], errors='coerce').dt.normalize()
    df_record = df_record[df_record['date'] >= START_DATE].copy()
    df_record['price'] = pd.to_numeric(df_record['price'], errors='coerce').fillna(0)
    df_record['share'] = pd.to_numeric(df_record['share'], errors='coerce').fillna(0)
    df_record['cash_flow'] = np.where(df_record['action'] == 'in', df_record['price'], 
                             np.where(df_record['action'] == 'out', -df_record['price'], 0))

    # 內部函數：標準化報價資料庫欄位
    def standardize_market_df(df):
        df.columns = df.columns.str.strip().str.lower()
        if 'price' in df.columns and 'close' not in df.columns:
            df = df.rename(columns={'price': 'close'})
        return df

    # 讀取市場報價 (整合寬表)
    market_path = config.DATABASE_DIR/"market.parquet"
    if os.path.exists(market_path):
        # 最低程度修改：改用 read_parquet，並刪除 dtype={'id': str}
        df_price = pd.read_parquet(market_path)
        df_price = standardize_market_df(df_price)
        df_price['date'] = pd.to_datetime(df_price['date'], errors='coerce').dt.normalize()
        df_price = df_price[df_price['date'] >= START_DATE].copy()
    else:
        df_price = pd.DataFrame(columns=['date', 'id', 'close'])

    # 找出有交易紀錄但報價庫存裡沒有的 ID(防呆)
    def validate_ids(df_record, df_price):
        record_ids = set(df_record[df_record['id'] != 'cash']['id'].unique())
        market_ids = set(df_price['id'].unique())
        
        orphan_ids = record_ids - market_ids
        if orphan_ids:
            import streamlit as st
            st.error(f"偵測到無效代號：{orphan_ids}。請檢查 journal.csv 或 market.csv。")
    validate_ids(df_record, df_price)

    # 讀取基準線
    def load_bench(name):
        path = config.DATABASE_DIR/f'{name}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, dtype={'id': str})
            df = standardize_market_df(df)
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
            return df[df['date'] >= START_DATE].copy()
        return pd.DataFrame()

    df_0050 = load_bench("0050")
    df_00631L = load_bench("00631")
    
    return df_record, df_price, df_0050, df_00631L

# ==========================================
# 模組 2：時間序列計算引擎 (含手續費精準結算)
# ==========================================
@st.cache_data
def calculate_daily_history(df_record, df_price, df_0050, df_00631L):
    if df_record.empty:
        return pd.DataFrame()

    dates = sorted(df_price['date'].dropna().unique())
    if not dates:
        dates = sorted(df_record['date'].unique())

    # 取得大盤基準首日價格
    p0_0050 = df_0050['close'].iloc[0] if not df_0050.empty else 1.0
    p0_00631L = df_00631L['close'].iloc[0] if not df_00631L.empty else 1.0

    # [新增] 淨值法初始狀態
    # 發行初始： 第一天你匯入 100 萬，我們假設初始淨值是 $1.0，所以你擁有了 100 萬個「單位數」。
    # 資產成長： 第二天股票漲了，總資產變成 110 萬。因為單位數還是 100 萬，所以淨值變成 $1.1（報酬率 10%）。
    # 中途入金（關鍵防失真）： 第三天你又匯入了 55 萬。這時候系統不會去改分母，而是讓你用今天的淨值 $1.1，去「申購」新的單位數（55 萬 / 1.1 = 50 萬單位）。
    # 結果： 你的總單位數變成了 150 萬單位，總資產變成 165 萬。但你的淨值依然是 $1.1，報酬率完美維持在 10%，完全排除了入金造成的扭曲。
    total_units = 0.0
    nav_per_unit = 1.0

    history = []
    for d in dates:
        rec_d = df_record[df_record['date'] <= d]
        if rec_d.empty:
            continue

        # 取得當日最新報價切片
        px_d = df_price[df_price['date'] <= d].groupby('id').tail(1).set_index('id')['close'].to_dict()

        # 1. 總本金結算
        cash_rec = rec_d[rec_d['id'] == 'cash']
        principal = cash_rec['cash_flow'].sum()  # 歷史總投入本金
        net_cf_today = cash_rec[cash_rec['date'] == d]['cash_flow'].sum()  # 今日淨現金流
        trades = rec_d[rec_d['id'] != 'cash'].copy()

        trades['val'] = trades['price'] * trades['share']

        # 2. 內部函數：計算各標籤子投組 (加入手續費計算)
        def calc_sub_portfolio(label_filter):
            sub = trades[trades['label'] == label_filter] if label_filter else trades
            if sub.empty: return 0.0, 0.0, 0.0
            
            # 分離買賣行為計算手續費
            buy_mask = sub['action'] == 'buy'
            sell_mask = sub['action'] == 'sell'
            
            buy_fees = (sub[buy_mask]['val'] * 0.001425).astype(int)
            sell_fees = (sub[sell_mask]['val'] * 0.001425).astype(int) + (sub[sell_mask]['val'] * 0.003).astype(int)
            
            buy_cost = sub[buy_mask]['val'].sum() + buy_fees.sum()
            sell_revenue = sub[sell_mask]['val'].sum() - sell_fees.sum()
            net_flow = sell_revenue - buy_cost
            
            mv = 0.0
            for aid in sub['id'].unique():
                a_sub = sub[sub['id'] == aid]
                shares = a_sub[a_sub['action'] == 'buy']['share'].sum() - a_sub[a_sub['action'] == 'sell']['share'].sum()
                if shares > 0:
                    raw_mv = shares * px_d.get(aid, 0.0)
                    
                    # --- [新增] 嚴謹扣除預估賣出手續費與交易稅 ---
                    # 如果是 0050 或是正二 (ETF)，稅率是 0.1%，一般個股是 0.3%
                    if str(aid).startswith('00'):
                        est_sell_fee = int(raw_mv * 0.001425) + int(raw_mv * 0.001)
                    else:
                        est_sell_fee = int(raw_mv * 0.001425) + int(raw_mv * 0.003)
                    
                    # 淨市值 = 原始市值 - 預扣賣出稅費
                    mv += (raw_mv - est_sell_fee)
                    
            roi = (mv + net_flow) / buy_cost if buy_cost > 0 else 0.0
            return mv, net_flow, roi

        mv_ai, _, roi_ai = calc_sub_portfolio('ai')
        mv_oth, _, roi_oth = calc_sub_portfolio('other')
        mv_tot, flow_tot, _ = calc_sub_portfolio(None)

        # 3. 總資產與總報酬結算
        current_cash = principal + flow_tot
        total_nav = current_cash + mv_tot
        if total_units == 0.0 and net_cf_today > 0:
            # 第一天建倉入金，發行初始單位
            nav_per_unit = 1.0
            total_units = net_cf_today / nav_per_unit
        elif total_units > 0:
            # 結算今天的淨值 (扣除今天才剛匯入、還沒參與市場漲跌的本金)
            nav_before_cf = total_nav - net_cf_today
            nav_per_unit = nav_before_cf / total_units
            
            # 若今天有現金流異動，用「今天的淨值」增減單位數
            if net_cf_today != 0:
                total_units += net_cf_today / nav_per_unit

        # 真實的累積報酬率 = (當前淨值 / 初始淨值 1.0) - 1
        roi_total = nav_per_unit - 1.0 if total_units > 0 else 0.0

        # 4. 基準線結算
        curr_0050 = df_0050[df_0050['date'] <= d]['close'].iloc[-1] if not df_0050[df_0050['date'] <= d].empty else p0_0050
        roi_0050 = (curr_0050 - p0_0050) / p0_0050

        curr_00631L = df_00631L[df_00631L['date'] <= d]['close'].iloc[-1] if not df_00631L[df_00631L['date'] <= d].empty else p0_00631L
        roi_00631L = (curr_00631L - p0_00631L) / p0_00631L

        history.append({
            'date': d, 'principal': principal, 'current_cash': current_cash,
            'total_nav': total_nav, 'roi_total': roi_total,
            'roi_ai': roi_ai, 'roi_oth': roi_oth,
            'roi_0050': roi_0050, 'roi_00631L': roi_00631L
        })
    df_res = pd.DataFrame(history)
    if not df_res.empty:
        df_res['max_nav'] = df_res['total_nav'].cummax()
        df_res['dd_pct'] = (df_res['total_nav'] - df_res['max_nav']) / df_res['max_nav']
        df_res['dd_twd'] = df_res['total_nav'] - df_res['max_nav']
        
    return df_res

# ==========================================
# 頁面 1：投組報酬率
# ==========================================
def page_performance_pct(df_history):
    st.title(f"累積報酬率(%)")
    if df_history.empty:
        st.warning("無數據")
        return
    
    last_row = df_history.iloc[-1]
    roi_pct = last_row['roi_total'] * 100 # 真實累積報酬率(%)
    roi_0050 = last_row['roi_0050'] * 100 # 真實累積報酬率(%)
    roi_00631 = last_row['roi_00631L'] * 100 # 真實累積報酬率(%)
    # 使用幾何超額報酬計算，消除長期複利帶來的基期失真
    alpha_0050 = (((1 + last_row['roi_total']) / (1 + last_row['roi_0050'])) - 1) * 100
    alpha_00631 = (((1 + last_row['roi_total']) / (1 + last_row['roi_00631L'])) - 1) * 100
    max_dd = df_history["dd_pct"].min()*100

    col1, col2, col3 = st.columns(3)
    col1.metric("累積報酬", f"{roi_pct:.2f}%")
    col2.metric("ALPHA", f"{alpha_0050:.2f}%")
    col3.metric("最大回撤", f"{max_dd:,.2f}%")

    fig = go.Figure()
    configs = [
        # 核心：戰鬥橘，增加線寬到 3，確保作為投資組合的主體視覺
        {"y": "roi_total", "name": "投資組合 (AI+Other)", "color": "#4F247E", "width": 2, "visible": True},
        
        # 對比：深石板藍，作為穩定的基準線，不搶橘色風采
        {"y": "roi_0050", "name": "大盤 (0050)", "color": "#68CCFD", "width": 2, "visible": True},
        
        # 輔助：沉穩紫色，對應你對正二的高波動但穩健的預期
        {"y": "roi_00631L", "name": "正二 (00631L)", "color":"#8E44AD", "width": 2, "visible": "legendonly"},
        
        # 輔助：深海青色，代表 AI 投資的成長性
        {"y": "roi_ai", "name": "AI 投資", "color": "#16A085", "width": 1.5, "visible": "legendonly"},
        
        # 輔助：灰褐色，完美融入木頭色背景，不產生視覺干擾
        {"y": "roi_oth", "name": "Other", "color": "#7F8C8D", "width": 1.5, "visible": "legendonly"}
    ]

    for c in configs:
        fig.add_trace(go.Scatter(
            x=df_history['date'].dt.strftime('%Y-%m-%d'), y=df_history[c['y']] * 100,
            name=c['name'], mode='lines',
            line=dict(color=c['color'], width=c['width']),
            visible=c['visible'],
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))

    fig.update_layout(
        template="plotly_dark", 
        hovermode="x unified", 
        xaxis=dict(type="category",tickformat="%Y-%m-%d"),
        yaxis=dict(title="累積報酬率", ticksuffix="%")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("風險監控：最大回撤幅度")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df_history['date'].dt.strftime('%Y-%m-%d'), y=df_history['dd_pct'] * 100,
        fill='tozeroy', fillcolor='rgba(255, 51, 51, 0.2)', # 半透明
        line=dict(color="#505050", width=1.5), name="回撤(%)",
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    fig_dd.update_layout(
        template="plotly_dark", height=250, margin=dict(t=10, b=10),
        xaxis=dict(type="category", tickformat="%Y-%m-%d"),
        yaxis=dict(title="回撤 %", ticksuffix="%")
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# ==========================================
# 頁面 2：投組權益淨值
# ==========================================
def page_performance_nav(df_history):
    st.title("累積權益(淨值)")
    if df_history.empty:
        st.warning("無數據")
        return

    # 1. 計算核心數值
    last_row = df_history.iloc[-1]
    tw = last_row['total_nav']  # 總權益 (市值+現金)
    ti = last_row['principal']  # 總投入 (初始+後來所有入金)
    tp = tw - ti                # 股票總損益
    
    # 真實累積報酬率 (用淨值算的那一個，最準)
    total_roi = last_row['roi_total'] * 100 

    # 2. UI 渲染 (最低程度修改)
    col1, col2, col3 = st.columns(3)
    
    # 顯示現在總共有多少錢
    col1.metric("總權益", f"${tw:,.0f}")
    
    # 顯示純交易損益，並把報酬率放在 delta
    col2.metric(
        "股票總損益", 
        f"${tp:,.0f}", 
    )
    
    # 顯示總投入本金，並把「後來追加的部分」放在 delta
    col3.metric(
        "累積追加投入",  
        f"{ti - init_cap:,.0f}",
    )

    # 2. 再建立與顯示下方折線圖
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_history['date'].dt.strftime('%Y-%m-%d'), y=df_history['total_nav'],
        name="總淨值(NAV)", line=dict(color='#4F247E'),
        hovertemplate='%{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df_history['date'].dt.strftime('%Y-%m-%d'), y=df_history['principal'],
        name="投入本金", line=dict(color='#68CCFD', dash='dot'),
        hovertemplate='%{y:,.0f}<extra></extra>'
    ))
    fig.update_layout(template="plotly_dark", yaxis_title="TWD", hovermode="x unified", yaxis=dict(tickformat=",.0f"),xaxis=dict(type='category'))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("風險監控：資產回撤金額")
    fig_dd_twd = go.Figure()
    fig_dd_twd.add_trace(go.Scatter(
        x=df_history['date'].dt.strftime('%Y-%m-%d'), y=df_history['dd_twd'],
        fill='tozeroy', fillcolor='rgba(255, 51, 51, 0.2)',
        line=dict(color='#505050', width=1.5), name="回撤金額",
        hovertemplate='$%{y:,.0f}<extra></extra>'
    ))
    fig_dd_twd.update_layout(
        template="plotly_dark", height=250, margin=dict(t=10, b=10),
        xaxis=dict(type="category", tickformat="%Y-%m-%d"),
        yaxis=dict(title="TWD")
    )
    st.plotly_chart(fig_dd_twd, use_container_width=True)

# ==========================================
# 頁面 3：配置比例 (加入手續費校正現金池)
# ==========================================
def page_allocation(df_record, df_price):
    st.title("目前配置比例")
    if df_record.empty:
        st.warning("無數據")
        return

    df_trade = df_record[df_record['id'] != 'cash'].copy()
    latest_px = df_price.groupby('id').tail(1).set_index('id')['close'].to_dict() if not df_price.empty else {}

    labels, values = [], []
    for aid in df_trade['id'].unique():
        sub = df_trade[df_trade['id'] == aid]
        shares = sub[sub['action'] == 'buy']['share'].sum() - sub[sub['action'] == 'sell']['share'].sum()
        
        if shares > 0:
            px = latest_px.get(aid, 0.0)
            if px == 0.0:
                last_buy = sub[sub['action'] == 'buy']
                px = last_buy['price'].iloc[-1] if not last_buy.empty else 0.0
                
            if px > 0:
                name = sub['name'].iloc[0] if 'name' in sub.columns else ""
                label_str = f"{aid.upper()} {name}".strip()
                labels.append(label_str)
                values.append(shares * px)

    # 精準計算現金 (納入手續費與交易稅)
    cash_rec = df_record[df_record['id'] == 'cash']
    principal = cash_rec['cash_flow'].sum()
    
    buy_mask = df_trade['action'] == 'buy'
    sell_mask = df_trade['action'] == 'sell'
    
    buy_val = df_trade[buy_mask]['price'] * df_trade[buy_mask]['share']
    buy_fees = (buy_val * 0.001425).astype(int)
    
    sell_val = df_trade[sell_mask]['price'] * df_trade[sell_mask]['share']
    sell_fees = (sell_val * 0.001425).astype(int) + (sell_val * 0.003).astype(int)
    
    current_cash = principal - (buy_val.sum() + buy_fees.sum()) + (sell_val.sum() - sell_fees.sum())

    if current_cash > 1:
        labels.append("現金")
        values.append(current_cash)

    if not labels:
        st.info("無持倉配置")
        return

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=.3, textinfo='label+percent',
        hovertemplate="<b>%{label}</b><br>帳面價值: $%{value:,.0f}<br>佔比: %{percent}<extra></extra>",
        marker=dict(colors=["#FFF2AF", '#FFB380', '#A3E4D7', "#B5D0E2", '#CCCCCC','#A996FE'])
    )])
    fig.update_layout(template="plotly_dark",height=400,margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 頁面 4：歷史紀錄結算 (加入獨立手續費欄位)
# ==========================================
def page_history(df_record, df_price):
    st.title("資產結算中心")
    if df_record.empty:
        st.warning("無數據")
        return

    df_trade = df_record[df_record['id'] != 'cash'].copy()
    latest_px = df_price.groupby('id').tail(1).set_index('id')['close'].to_dict() if not df_price.empty else {}

    unrealized_data, realized_data = [], []

    for aid in df_trade['id'].unique():
        sub = df_trade[df_trade['id'] == aid].sort_values('date')
        cum_shares = 0
        total_cost = 0
        shares_sold = 0
        revenue = 0
        
        # 追蹤手續費
        cum_buy_fee = 0
        cum_sell_fee = 0
        raw_sell_val = 0 # 追蹤純賣出價值以計算均價

        stock_name = sub['name'].iloc[0] if 'name' in sub.columns else ""

        for _, row in sub.iterrows():
            trade_val = row['price'] * row['share']
            if row['action'] == 'buy':
                fee = int(trade_val * 0.001425)  # 買進手續費 0.1425%
                cum_buy_fee += fee
                cum_shares += row['share']
                total_cost += trade_val + fee    # 成本 = 買價 + 手續費
            elif row['action'] == 'sell':
                fee = int(trade_val * 0.001425) + int(trade_val * 0.003) # 賣出手續費 + 交易稅
                cum_sell_fee += fee
                shares_sold += row['share']
                raw_sell_val += trade_val
                revenue += trade_val - fee       # 實際入帳 = 賣價 - 手續費

        curr_shares = cum_shares - shares_sold
        # 這裡的 avg_cost 已經包含買入手續費
        avg_cost = total_cost / cum_shares if cum_shares > 0 else 0
        avg_buy_fee = cum_buy_fee / cum_shares if cum_shares > 0 else 0
        
        yahoo_url = f"https://tw.stock.yahoo.com/quote/{aid}"

        if shares_sold > 0:
            # 賣出的總成本 (已包含對應比例的買入手續費)
            sell_cost = shares_sold * avg_cost
            r_pnl = revenue - sell_cost
            r_roi_pct = (r_pnl / sell_cost) * 100 if sell_cost > 0 else 0.0
            
            if r_roi_pct > 0:
                r_roi_str = f"▲ {r_roi_pct:.1f}%"
            elif r_roi_pct < 0:
                r_roi_str = f"▼ -{abs(r_roi_pct):.1f}%"
            else:
                r_roi_str = "− 0.0%"
                
            avg_sell_px = raw_sell_val / shares_sold
            # 已實現總手續費 = 賣出股數攤提的買入手續費 + 實際發生的賣出手續費
            realized_total_fee = (shares_sold * avg_buy_fee) + cum_sell_fee

            realized_data.append({
                "代號": yahoo_url,     
                "名稱": stock_name,    
                "累積報酬率": f"{r_roi_str}",
                "已實現損益": round(r_pnl, 0),
                "平均成本": round(avg_cost, 1), 
                "賣出均價": f"{avg_sell_px:.1f}",
                "買賣手續費": int(realized_total_fee), # 新增手續費欄位
                "賣出股數": int(shares_sold), 
                "回收現金": round(revenue, 0)
            })

        if curr_shares > 0:
            px = latest_px.get(aid, 0.0)
            mv = curr_shares * px

            est_sell_fee = int(mv * 0.001425) + int(mv * 0.003)
            # 未實現損益：目前市值 - (包含買入/賣出手續費的成本)
            u_pnl = mv - (curr_shares * avg_cost) - est_sell_fee
            roi_pct = (u_pnl / (curr_shares * avg_cost)) * 100 if avg_cost > 0 else 0.0

            if roi_pct > 0:
                roi_str = f"▲ {roi_pct:.1f}%"
            elif roi_pct < 0:
                roi_str = f"▼ -{abs(roi_pct):.1f}%"
            else:
                roi_str = "− 0.0%"
                
            unrealized_buy_fee = curr_shares * avg_buy_fee

            unrealized_data.append({
                "代號": yahoo_url,     
                "名稱": stock_name,    
                "累積報酬率":f"{roi_str}",
                "未實現損益": round(u_pnl, 0),
                "平均成本": round(avg_cost, 1), 
                "最新市價": f"{px:.1f}", 
                "買入手續費": int(unrealized_buy_fee), # 新增手續費欄位
                "買入股數": int(curr_shares),
                "當前市值": round(mv, 0),
            })

    # UI 魔法：攔截網址最後面的代號
    column_config_setting = {
        "代號": st.column_config.LinkColumn(
            "代號", 
            display_text=r"https://tw\.stock\.yahoo\.com/quote/(.*)"
        )
    }

    st.subheader("未實現損益")
    if unrealized_data:
        df_un = pd.DataFrame(unrealized_data).sort_values("當前市值", ascending=False)
        st.dataframe(
            df_un, 
            use_container_width=True, 
            hide_index=True,
            column_config=column_config_setting
        )
    
    st.markdown("---")
    st.subheader("已實現損益")
    if realized_data:
        df_re = pd.DataFrame(realized_data)
        st.dataframe(
            df_re, 
            use_container_width=True, 
            hide_index=True,
            column_config=column_config_setting
        )
# ==========================================
# 頁面 5：月度績效結算 (非累積)
# ==========================================
def page_monthly_performance(df_history):
    st.title("月度績效結算")
    if df_history.empty:
        st.warning("無數據")
        return

    # 1. 將每日資料按月份分組，抓取每個月最後一天的紀錄
    df = df_history.copy()
    df['month'] = df['date'].dt.strftime('%Y-%m')
    monthly_end = df.groupby('month').last().reset_index()

    # 2. 計算月度獨立報酬率 (NAV增長率) 與 實質獲利金額
    # 使用 shift(1) 取得上個月的數值，若為第一個月則補 0 或 1
    monthly_end['prev_nav_ratio'] = monthly_end['roi_total'].shift(1).fillna(0) + 1.0
    monthly_end['curr_nav_ratio'] = monthly_end['roi_total'] + 1.0
    monthly_end['monthly_ret_pct'] = (monthly_end['curr_nav_ratio'] / monthly_end['prev_nav_ratio'] - 1) * 100
    
    # 針對第一個月的特例處理
    monthly_end.loc[0, 'monthly_ret_pct'] = (monthly_end.loc[0, 'curr_nav_ratio'] - 1.0) * 100

    # 月度獲利金額 = 本月底總淨值 - 上月底總淨值 - 本月新增投入本金
    monthly_end['prev_total_nav'] = monthly_end['total_nav'].shift(1).fillna(0)
    monthly_end['prev_principal'] = monthly_end['principal'].shift(1).fillna(0)
    monthly_end['principal_added'] = monthly_end['principal'] - monthly_end['prev_principal']
    monthly_end['monthly_profit'] = monthly_end['total_nav'] - monthly_end['prev_total_nav'] - monthly_end['principal_added']

    # 3. 繪製圖表 (折線圖 + 柱狀圖的混合視覺化)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("單月報酬率 (%)")
        fig_ret = go.Figure()
        # 台股直覺：紅漲 (#FF3333) 綠跌 (#00AA00)
        colors_ret = ["#FF7D7D" if val > 0 else "#A4FFA4" for val in monthly_end['monthly_ret_pct']]
        
        fig_ret.add_trace(go.Bar(
            x=monthly_end['month'], 
            y=monthly_end['monthly_ret_pct'],
            marker_color=colors_ret, 
            text=monthly_end['monthly_ret_pct'].round(2).astype(str) + '%',
            textposition='outside', 
            name="月報酬率"
        ))
        
        fig_ret.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#444444"),
            yaxis_ticksuffix="%", 
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(type="category", gridcolor="#EEE"),
            yaxis=dict(gridcolor="#EEE")
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    with col2:
        st.subheader("單月實質獲利 (TWD)")
        fig_prof = go.Figure()
        # 獲利同樣採用紅綠邏輯，或可統一使用系統橘色 (#FF8800)
        colors_prof = ['#FF7D7D' if val > 0 else '#A4FFA4' for val in monthly_end['monthly_profit']]
        
        fig_prof.add_trace(go.Bar(
            x=monthly_end['month'], 
            y=monthly_end['monthly_profit'],
            marker_color=colors_prof, # 這裡建議也用紅綠，視覺邏輯才統一
            text=monthly_end['monthly_profit'].map('{:,.0f}'.format),
            textposition='outside',
            name="獲利金額", 
            hovertemplate='$%{y:,.0f}<extra></extra>'
        ))
        
        fig_prof.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#444444"),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(type="category", gridcolor="#EEE"),
            yaxis=dict(gridcolor="#EEE")
        )
        st.plotly_chart(fig_prof, use_container_width=True)
# ==========================================
# 主程式
# ==========================================
def main():
    # 1. 建立一個暫存佔位符
    loading_placeholder = st.empty()
    
    with loading_placeholder.container():
        # st.image("assets/fox_loading.gif", width=200) 
        st.write("🦊請稍候...")

    df_record, df_price, df_0050, df_00631L = load_data()
    df_history = calculate_daily_history(df_record, df_price, df_0050, df_00631L)

    # 3. 讀取完畢後，把佔位符清空，換成主程式內容
    loading_placeholder.empty()

    with st.sidebar:
        current_nav = df_history['total_nav'].iloc[-1]
        target_progress = (current_nav / 10000000) * 100

        st.title(f"KITSUNE KERNEL")
        selection = st.radio(f"選單",["累積報酬率", "累積權益", "目前配置", "損益結算中心","單月績效報告"])

    if selection == "累積報酬率":
        page_performance_pct(df_history)
    elif selection == "累積權益":
        page_performance_nav(df_history)
    elif selection == "目前配置":
        page_allocation(df_record, df_price)
    elif selection == "損益結算中心":
        page_history(df_record, df_price)
    else:
        page_monthly_performance(df_history)
    

if __name__ == "__main__": 
    main()