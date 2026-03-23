#streamlit run c:/Users/Lynn610/Documents/kitsune_stock_system/dashboard.py

# st.title() → 渲染為 <h1>
# st.header() → 渲染為 <h2>
# st.subheader() → 渲染為 <h3>

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
    [data-testid="stSidebar"] {
        background-color: #fcf5ed !important;
    }
 
    [data-testid="stSidebar"] h1 {
        margin-top: -2rem !important;
        padding-top: 2px !important;
    }
    
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 0rem !important; 
    }

    /* 2. 標題與數據 (完全保留你的設定) */
    .stApp h1 {
        font-size: 1.5rem !important;      
        padding-bottom: 0.8rem !important;
        font-weight: 700 !important;
        color: #1a1a1a;
    }

    .stApp h3 {
        font-size: 1.1rem !important;      
        margin-top: -5px !important;
        padding-bottom: 0.6rem !important;
        font-weight: 400 !important;
    }

    [data-testid="stMetricLabel"] p {
        font-size: 1.0rem !important;
        font-weight: 500 !important;
        color: #333333 !important;
        opacity: 1 !important;
        margin-bottom: 0px !important;
    }

    [data-testid="stMetricValue"] div {
        font-size: 2.0rem !important;
        font-weight: 700 !important;
        color: #C0392B !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #1a1a1a !important;
        font-size: 1.05rem !important;
        margin-bottom: 8px !important;
    }

    [data-testid="stSidebar"] .st-emotion-cache-6qob1r { 
        font-size: 1.1rem !important; 
    }

    /* 3. 警報 (在你的設定基礎上，加入最少跑馬燈邏輯) */
    /* 針對 st.error 的整個容器進行加深處理 */
    div[data-testid="stAlertContainer"]{
        background-color:#1A2B4C!important; /* 背景 */
        padding-top: 1rem !important;    /* 上內距 */
        padding-bottom: 1rem !important; /* 下內距 */
        height:100%!important;
        width: 100% !important;
    }

    /* 針對警報內的文字 */
    div[data-testid="stAlert"] p {
        color:  #ffffff	  !important;            
        font-size: 1.0rem !important;         /* 放大字體 */
        font-weight: 700 !important;          /* 極粗體 */
        letter-spacing: 3.0px !important;     /* 增加字距營造嚴肅感 */
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
            
            # --- 核心修正：在切片(Filter)之前，先算好波動率 ---
            df.columns = [c.lower() for c in df.columns]
            p_col = 'price' if 'price' in df.columns else 'close'
            df['daily_return'] = df[p_col].pct_change()
            # 這裡會用到 CSV 裡所有的歷史資料，絕對夠 20 筆
            df['realized_vol'] = df['daily_return'].rolling(window=20).std() * (252 ** 0.5)
            
            # 算完之後，再根據 START_DATE 切片回傳
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
# 模組3：共用資產結算引擎 (提供給各頁面使用)
# ==========================================
@st.cache_data
def calculate_settlement_data(df_record, df_price):
    if df_record.empty:
        return [], [], 0, 0

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
        raw_sell_val = 0

        stock_name = sub['name'].iloc[0] if 'name' in sub.columns else ""

        for _, row in sub.iterrows():
            trade_val = row['price'] * row['share']
            if row['action'] == 'buy':
                fee = int(trade_val * 0.001425)  
                cum_buy_fee += fee
                cum_shares += row['share']
                total_cost += trade_val + fee    
            elif row['action'] == 'sell':
                fee = int(trade_val * 0.001425) + int(trade_val * 0.003) 
                cum_sell_fee += fee
                shares_sold += row['share']
                raw_sell_val += trade_val
                revenue += trade_val - fee       

        curr_shares = cum_shares - shares_sold
        avg_cost = total_cost / cum_shares if cum_shares > 0 else 0
        avg_buy_fee = cum_buy_fee / cum_shares if cum_shares > 0 else 0
        
        yahoo_url = f"https://tw.stock.yahoo.com/quote/{aid}"

        if shares_sold > 0:
            sell_cost = shares_sold * avg_cost
            r_pnl = revenue - sell_cost
            r_roi_pct = (r_pnl / sell_cost) * 100 if sell_cost > 0 else 0.0
            
            # if r_roi_pct > 0: r_roi_str = f"▲ {r_roi_pct:.1f}%"
            # elif r_roi_pct < 0: r_roi_str = f"▼ -{abs(r_roi_pct):.1f}%"
            # else: r_roi_str = "− 0.0%"
                
            avg_sell_px = raw_sell_val / shares_sold
            realized_total_fee = (shares_sold * avg_buy_fee) + cum_sell_fee

            realized_data.append({
                "代號": yahoo_url,    
                "名稱": stock_name,    
                "累積報酬率": f"{r_roi_pct:.1f}%",
                "已實現損益": round(r_pnl, 0),
                "平均成本": round(avg_cost, 1), 
                "賣出均價": f"{avg_sell_px:.1f}",
                "買賣手續費": int(realized_total_fee),
                "賣出股數": int(shares_sold), 
                "回收現金": round(revenue, 0)
            })

        if curr_shares > 0:
            px = latest_px.get(aid, 0.0)
            mv = curr_shares * px
            est_sell_fee = int(mv * 0.001425) + int(mv * 0.003)
            u_pnl = mv - (curr_shares * avg_cost) - est_sell_fee
            roi_pct = (u_pnl / (curr_shares * avg_cost)) * 100 if avg_cost > 0 else 0.0

            # if roi_pct > 0: roi_str = f"▲ {roi_pct:.1f}%"
            # elif roi_pct < 0: roi_str = f"▼ -{abs(roi_pct):.1f}%"
            # else: roi_str = "− 0.0%"
                
            unrealized_buy_fee = curr_shares * avg_buy_fee

            unrealized_data.append({
                "代號": yahoo_url,    
                "名稱": stock_name,    
                "累積報酬率": f"{roi_pct:.1f}%",
                "未實現損益": round(u_pnl, 0),
                "平均成本": round(avg_cost, 1), 
                "最新市價": f"{px:.1f}", 
                "買入手續費": int(unrealized_buy_fee), 
                "買入股數": int(curr_shares),
                "當前市值": round(mv, 0),
            })

    total_unrealized = sum(item["未實現損益"] for item in unrealized_data)
    total_realized = sum(item["已實現損益"] for item in realized_data)

    return unrealized_data, realized_data, total_unrealized, total_realized

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
    col1.metric("累積報酬", f"{roi_pct:.2f}%",help="包含現金")
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
def page_performance_nav(df_history,df_record, df_price):
    st.title("累積權益(淨值)")
    if df_history.empty:
        st.warning("無數據")
        return

    # 1. 計算核心數值
    last_row = df_history.iloc[-1]
    tw = last_row['total_nav']  # 總權益 (市值+現金)
    ti = last_row['principal']  # 總投入 (初始+後來所有入金)
    tp = tw - ti                # 股票總損益          

    _, _, total_unrealized, total_realized = calculate_settlement_data(df_record, df_price)

    col1, col2, col3 = st.columns(3)
    col1.metric("總權益", f"${tw:,.0f}")
    
    # 3. 完美顯示你的 help
    col2.metric(
        "股票總損益", 
        f"${tp:,.0f}",
        help=f"未實現：${total_unrealized:,.0f}\n\n已實現：${total_realized:,.0f}"
    )
    
    # 顯示總投入本金ta
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
# 頁面 3：配置比例 (雙視角切換 & 回傳權重)
# ==========================================
def page_allocation(df_record, df_price):
    st.title("目前配置比例")
    if df_record.empty:
        st.warning("無數據")
        return {}

    df_trade = df_record[df_record['id'] != 'cash'].copy()
    latest_px = df_price.groupby('id').tail(1).set_index('id')['close'].to_dict() if not df_price.empty else {}

    stock_labels, stock_values = [], []
    pure_equity_mv = 0.0
    
    # 1. 計算純股票部位
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
                mv = shares * px
                
                stock_labels.append(label_str)
                stock_values.append(mv)
                pure_equity_mv += mv

    # 2. 精準計算現金池
    cash_rec = df_record[df_record['id'] == 'cash']
    principal = cash_rec['cash_flow'].sum()
    
    buy_mask = df_trade['action'] == 'buy'
    sell_mask = df_trade['action'] == 'sell'
    
    buy_val = df_trade[buy_mask]['price'] * df_trade[buy_mask]['share']
    buy_fees = (buy_val * 0.001425).astype(int)
    
    sell_val = df_trade[sell_mask]['price'] * df_trade[sell_mask]['share']
    sell_fees = (sell_val * 0.001425).astype(int) + (sell_val * 0.003).astype(int)
    
    current_cash = principal - (buy_val.sum() + buy_fees.sum()) + (sell_val.sum() - sell_fees.sum())

    if not stock_labels and current_cash <= 1:
        st.info("無持倉配置")
        return {}

    # 3. 建立兩種圓餅圖的數據
    total_labels = stock_labels.copy()
    total_values = stock_values.copy()
    if current_cash > 1:
        total_labels.append("現金")
        total_values.append(current_cash)

    # 畫圖設定
    pie_colors = ["#FFF2AF", '#FFB380', '#A3E4D7', "#B5D0E2", '#CCCCCC', '#A996FE']
    fig = go.Figure()

    fig.add_trace(go.Pie(
        labels=total_labels, values=total_values, hole=.3, textinfo='label+percent',
        hovertemplate="<b>%{label}</b><br>帳面價值: $%{value:,.0f}<br>總資產佔比: %{percent}<extra></extra>",
        marker=dict(colors=pie_colors)
    ))

    if fig.data:
        fig.update_layout(template="plotly_dark", height=400, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
# ==========================================
# 頁面 4：歷史紀錄結算 (加入獨立手續費欄位與移動停利)
# ==========================================
def page_history(df_record, df_price):
    st.title("資產結算中心")
    if df_record.empty:
        st.warning("無數據")
        return

    # 1. 直接跟大腦拿算好的資料
    unrealized_data, realized_data, total_unrealized, total_realized = calculate_settlement_data(df_record, df_price)

    # 2. UI 魔法：攔截網址最後面的代號
    column_config_setting = {
        "代號": st.column_config.LinkColumn(
            "代號", 
            display_text=r"https://tw\.stock\.yahoo\.com/quote/(.*)"
        )
    }

    # 3. 畫未實現表格
    st.subheader(f"未實現損益 │ ${total_unrealized:,.0f}")
    if unrealized_data:
        df_un = pd.DataFrame(unrealized_data).sort_values("當前市值", ascending=False)
        st.dataframe(df_un, use_container_width=True, hide_index=True, column_config=column_config_setting)

    st.markdown("---")

    # 4. 畫已實現表格
    st.subheader(f"已實現損益 │ ${total_realized:,.0f}")
    if realized_data:
        df_re = pd.DataFrame(realized_data)
        st.dataframe(df_re, use_container_width=True, hide_index=True, column_config=column_config_setting)
# ==========================================
# 頁面 5：月度績效結算 (非累積)
# ==========================================
def page_monthly_performance(df_history):
    st.title("月度績效結算")
    if df_history.empty:
        st.warning("無數據")
        return

    # 1. 資料處理
    df = df_history.copy()
    df['month'] = df['date'].dt.strftime('%Y-%m')
    df['daily_ret'] = df['roi_total'].diff().fillna(0) # 用於計算月度波動
    
    m_groups = df.groupby('month')
    monthly_end = m_groups.last().reset_index()

    # 計算月報酬、大盤月報酬與 Alpha (幾何邏輯)
    monthly_end['prev_nav'] = monthly_end['roi_total'].shift(1).fillna(0) + 1.0
    monthly_end['monthly_ret_pct'] = ((monthly_end['roi_total'] + 1.0) / monthly_end['prev_nav'] - 1) * 100
    monthly_end.loc[0, 'monthly_ret_pct'] = (monthly_end.loc[0, 'roi_total']) * 100

    monthly_end['prev_0050'] = monthly_end['roi_0050'].shift(1).fillna(0) + 1.0
    monthly_end['m_0050'] = ((monthly_end['roi_0050'] + 1.0) / monthly_end['prev_0050'] - 1) * 100
    monthly_end.loc[0, 'm_0050'] = (monthly_end.loc[0, 'roi_0050']) * 100
    
    monthly_end['monthly_alpha_pct'] = (((1 + monthly_end['monthly_ret_pct']/100) / (1 + monthly_end['m_0050']/100)) - 1) * 100

    # 計算獲利金額
    monthly_end['prev_total_nav'] = monthly_end['total_nav'].shift(1).fillna(0)
    monthly_end['principal_added'] = monthly_end['principal'] - monthly_end['principal'].shift(1).fillna(0)
    monthly_end['monthly_profit'] = monthly_end['total_nav'] - monthly_end['prev_total_nav'] - monthly_end['principal_added']

    # 計算月度 Sharpe/Sortino
    risk_stats = []
    for m, group in m_groups:
        d_ret = group['daily_ret']
        avg, std = d_ret.mean(), d_ret.std()
        sharpe = (avg / std) * np.sqrt(252) if std > 0 else 0
        d_std = d_ret[d_ret < 0].std()
        sortino = (avg / d_std) * np.sqrt(252) if d_std > 0 else (sharpe if avg > 0 else 0)
        risk_stats.append({'month': m, 'Sharpe': sharpe, 'Sortino': sortino})
    monthly_end = monthly_end.merge(pd.DataFrame(risk_stats), on='month')

    # 計算平均值與中位數 (供標題使用)
    avg_ret, med_ret = monthly_end['monthly_ret_pct'].mean(), monthly_end['monthly_ret_pct'].median()
    avg_prof, med_prof = monthly_end['monthly_profit'].mean(), monthly_end['monthly_profit'].median()

    r1_col1, r1_col2 = st.columns(2)
    
    with r1_col1:
        st.subheader("月報酬率")
        st.caption(f"Avg: {avg_ret:.1f}%  |  Med: {med_ret:.1f}%")
        colors_ret = ["#FD9292" if val > 0 else "#A4FFA4" for val in monthly_end['monthly_ret_pct']]
        fig_ret = go.Figure(go.Bar(
            x=monthly_end['month'], 
            y=monthly_end['monthly_ret_pct'], 
            marker_color=colors_ret, 
            text=monthly_end['monthly_ret_pct'].round(1).astype(str)+'%', 
            textposition='outside'
        ))
        fig_ret.update_layout(
            template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
            margin=dict(l=20,r=20,t=30,b=20), height=300,
            xaxis=dict(type="category") # 強制類別型軸，避免時間軸間距問題
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    with r1_col2:
        st.subheader("月損益")
        st.caption(f"Avg: {avg_prof:,.0f}  |  Med: {med_prof:,.0f}")
        colors_prof = ['#FD9292' if val > 0 else '#A4FFA4' for val in monthly_end['monthly_profit']]
        fig_prof = go.Figure(go.Bar(
            x=monthly_end['month'], 
            y=monthly_end['monthly_profit'], 
            marker_color=colors_prof, 
            text=monthly_end['monthly_profit'].map('{:,.0f}'.format), 
            textposition='outside'
        ))
        fig_prof.update_layout(
            template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
            margin=dict(l=20,r=20,t=30,b=20), height=300,
            xaxis=dict(type="category")
        )
        st.plotly_chart(fig_prof, use_container_width=True)

    st.caption("")

    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        st.subheader("Alpha 趨勢")
        # st.caption("幾何超額報酬 (穩定度監控)")
        fig_alpha = go.Figure(go.Scatter(
            x=monthly_end['month'], 
            y=monthly_end['monthly_alpha_pct'], 
            mode='lines+markers', 
            line=dict(color='#16A085', width=3), 
            fill='tozeroy',
            fillcolor='rgba(22, 160, 133, 0.1)'
        ))
        fig_alpha.update_layout(
            template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
            margin=dict(l=20,r=20,t=30,b=20), height=300, 
            yaxis_ticksuffix="%",
            xaxis=dict(type="category")
        )
        st.plotly_chart(fig_alpha, use_container_width=True)

    with r2_col2:
        st.subheader("夏普/索提諾")
        # st.caption("Sharpe (紫) & Sortino (橘)")
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(
            x=monthly_end['month'], y=monthly_end['Sharpe'], 
            name="Sharpe", line=dict(color='#4F247E', width=2)
        ))
        fig_risk.add_trace(go.Scatter(
            x=monthly_end['month'], y=monthly_end['Sortino'], 
            name="Sortino", line=dict(color='#db3b00', width=2, dash='dot')
        ))
        fig_risk.update_layout(
            template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
            margin=dict(l=20,r=20,t=30,b=20), height=300, 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(type="category")
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # 下方保留量化表格，供精確數值查閱
    st.markdown("---")
    report_df = monthly_end[['month', 'monthly_ret_pct', 'monthly_alpha_pct', 'Sharpe', 'Sortino']].copy()
    report_df.columns = ['月份', '報酬率', 'Alpha', 'Sharpe', 'Sortino']
    st.dataframe(report_df.style.format({'報酬率': '{:.2f}%', 'Alpha': '{:.2f}%', 'Sharpe': '{:.2f}', 'Sortino': '{:.2f}'}), use_container_width=True, hide_index=True)

# ==========================================
# 模組 6：再平衡監控警報器 (純股 25% 閾值削回 20%)
# ==========================================
def check_rebalance_alerts(df_record, df_price):
    if df_record.empty or df_price.empty:
        return []

    df_trade = df_record[df_record['id'] != 'cash'].copy()
    latest_px = df_price.groupby('id').tail(1).set_index('id')['close'].to_dict()
    
    holdings = {}
    pure_equity_mv = 0.0
    
    for aid in df_trade['id'].unique():
        sub = df_trade[df_trade['id'] == aid]
        shares = sub[sub['action'] == 'buy']['share'].sum() - sub[sub['action'] == 'sell']['share'].sum()
        
        if shares > 0:
            px = latest_px.get(aid, 0.0)
            if px == 0.0:
                last_buy = sub[sub['action'] == 'buy']
                px = last_buy['price'].iloc[-1] if not last_buy.empty else 0.0
            
            if px > 0:
                mv = shares * px
                name = sub['name'].iloc[0] if 'name' in sub.columns else ""
                holdings[aid] = {'name': name, 'shares': shares, 'price': px, 'mv': mv}
                pure_equity_mv += mv
                
    alerts = []
    if pure_equity_mv > 0:
        for aid, data in holdings.items():
            weight = data['mv'] / pure_equity_mv
            if weight > 0.25:
                # 代入機構再平衡公式：X = (V_A - 0.2 * V_total) / 0.8
                sell_value = (data['mv'] - 0.20 * pure_equity_mv) / 0.8
                sell_shares = int(sell_value / data['price'])
                alerts.append(
                    f"【指令 : 重構】{aid} {data['name']} 投組權重大於25% (目前{weight*100:.1f}%)，需賣出市值 {sell_value:,.0f}(約需賣出 {sell_shares} 股)"
                )
    return alerts

# ==========================================
# 模組 7：大盤波動率斷路警報 (0050 20日年化波動 > 0.3)
# ==========================================
def check_market_volatility_alerts(df_0050, threshold=0.30):
    if df_0050.empty or 'realized_vol' not in df_0050.columns:
        return []

    # 直接拿最後一筆已經算好的數值
    latest_vol = df_0050['realized_vol'].iloc[-1]
    
    if pd.isna(latest_vol): return []

    if latest_vol > threshold:
        return [f"【指令 : 熔毀 】 大盤波動率超過{threshold*100:.0f}% (目前:{latest_vol*100:.1f}%)，需砍倉1/2"]
    return []
# ==========================================
# 主程式
# ==========================================
def main():
    # 1. 建立一個暫存佔位符
    loading_placeholder = st.empty()
    
    #資料初始化與佔位
    with loading_placeholder.container():
        # st.image("assets/fox_loading.gif", width=200) 
        st.write("🦊請稍候...")

    df_record, df_price, df_0050, df_00631L = load_data()
    df_history = calculate_daily_history(df_record, df_price, df_0050, df_00631L)
    # 3. 讀取完畢後，把佔位符清空，換成主程式內容
    loading_placeholder.empty()

    #提醒與通知
    # 第一層：大盤波動率斷路器 (防禦黑天鵝)
    vol_alerts = check_market_volatility_alerts(df_0050)
    for v_alert in vol_alerts:
        st.warning(v_alert) # 熔毀指令

    rebalance_alerts = check_rebalance_alerts(df_record, df_price)
    for rb_alert in rebalance_alerts:
        st.warning(rb_alert) # 重構指令

    with st.sidebar:
        # current_nav = df_history['total_nav'].iloc[-1]
        # target_progress = (current_nav / 10000000) * 100

        st.title("KITSUNE KERNEL")
        st.caption("")
        selection = st.radio(f"選單",["累積報酬率", "累積權益", "目前配置", "損益結算中心","單月績效報告"])

    if selection == "累積報酬率":
        page_performance_pct(df_history)
    elif selection == "累積權益":
        page_performance_nav(df_history,df_record,df_price)
    elif selection == "目前配置":
        page_allocation(df_record, df_price)
    elif selection == "損益結算中心":
        page_history(df_record, df_price)
    else:
        page_monthly_performance(df_history)
    

if __name__ == "__main__": 
    main()