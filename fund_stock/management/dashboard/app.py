# streamlit run c:/Users/Lynn610/Documents/kitsune_stock_system/dashboard/main.py
"""
app_quant.py
Kitsune Kernel — 主程式入口
職責：CSS 注入、資料初始化、警報觸發、Sidebar 路由
啟動：streamlit run app_quant.py
"""

import streamlit as st
import pandas as pd

from fund_stock.management.dashboard.engine import (
    load_data,
    calculate_settlement_data,
    generate_system_alerts,
)
from fund_stock.management.dashboard.pages import (
    page_fund_overview,
    page_fund_nav,
    page_fund_return,
    page_fund_quality,
    page_allocation,
    page_history,
)
import config

# ──────────────────────────────────────────
# 全域設定
# ──────────────────────────────────────────
st.set_page_config(page_title="Kitsune Quant", layout="wide")

_CSS = """
<style>
/* ── 留白修正 ── */
[data-testid="stMainBlockContainer"] { padding-top: 2.0rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #fcf5ed !important;
    color: #1a1a1a !important;
    font-size: 1.05rem !important;
}
/* ── 左頁置頂空白 ── */
[data-testid="stSidebarUserContent"] {
    margin-top: -50px !important;
}
/* ── 選單page間隔 ── */
[data-testid="stSidebar"] .stRadio label {
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}
/* ── 標題 ── */
.stApp h1 {
    font-size: 1.5rem !important;
    padding-top: 1.0rem !important;
    padding-bottom: 0.8rem !important;
    margin-top: 0rem !important;
    font-weight: 700 !important;
    color: #1a1a1a !important;
}
.stApp h3 {
    font-size: 1.2rem !important;
    margin-top: -5px !important;
    padding-bottom: 0.8rem !important;
    font-weight: 400 !important;
}

/* ── Metric label（小字） ── */
[data-testid="stMetricLabel"] {
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    color: #333333 !important;
}

/* ── Metric value（大數字，紅色）── */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] p {
    font-size: 2.0rem !important;
    font-weight: 700 !important;
    color: #C0392B !important;
    padding-top: 0.3rem !important;
}

/* ── 一般 markdown 文字（排除 MetricValue 內的）── */
[data-testid="stMarkdownContainer"] p {
    color: #1a1a1a !important;
    font-size: 1.2rem !important;
    margin-bottom: 5px !important;
}
</style>
"""

def alert(message):
    html_code = f"""
    <div style="
        background-color: #1A2B4C;
        color: #ffffff;
        border: none;
        margin: 15px 20px 15px 0px;
        font-weight: 500;
        font-size: 1.1rem;
        letter-spacing: 2px;
    ">
        {message}
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)


# ──────────────────────────────────────────
# 資料讀取：直接讀 CSV
# ──────────────────────────────────────────
@st.cache_data
def load_nav_history():
    """
    讀取 navpu.csv，補算 pages 需要的衍生欄位：
    roi_total, roi_0050, roi_00631L, dd_pct, total_units, total_nav, cash
    """
    # ── 基金 NAV ──
    df_nav = pd.read_csv(config.KITSUNE_QUANT_ADMINISTRATION_DIR / "navpu_quant.csv")
    df_nav['date'] = pd.to_datetime(df_nav['date']).dt.normalize()
    df_nav = df_nav.sort_values('date').reset_index(drop=True)

    # roi_total：相對第一天的累積報酬
    df_nav['roi_total'] = df_nav['nav_per_unit'] / df_nav['nav_per_unit'].iloc[0] - 1

    # dd_pct：回撤
    df_nav['max_nav'] = df_nav['nav_per_unit'].cummax()
    df_nav['dd_pct']  = (df_nav['nav_per_unit'] - df_nav['max_nav']) / df_nav['max_nav']

    # ── Benchmark ──
    def _load_bench(filename):
        path = config.DATABASE_DIR / filename
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        return df.sort_values('date').reset_index(drop=True)

    df_0050   = _load_bench("0050.csv")
    df_00631L = _load_bench("00631.csv")

    # 對齊起始日
    start = df_nav['date'].iloc[0]
    df_0050   = df_0050[df_0050['date'] >= start].reset_index(drop=True)
    df_00631L = df_00631L[df_00631L['date'] >= start].reset_index(drop=True)

    # benchmark roi
    p0_0050   = df_0050['close'].iloc[0]
    p0_00631L = df_00631L['close'].iloc[0]
    df_0050['roi']   = df_0050['close']   / p0_0050   - 1
    df_00631L['roi'] = df_00631L['close'] / p0_00631L - 1

    # merge benchmark roi 到 df_nav（左連接，交易日對齊）
    df_nav = df_nav.merge(
        df_0050[['date', 'roi']].rename(columns={'roi': 'roi_0050'}),
        on='date', how='left'
    ).merge(
        df_00631L[['date', 'roi']].rename(columns={'roi': 'roi_00631L'}),
        on='date', how='left'
    )
    df_nav[['roi_0050', 'roi_00631L']] = df_nav[['roi_0050', 'roi_00631L']].ffill()

    return df_nav, df_0050, df_00631L


# ──────────────────────────────────────────
# 頁面路由對應表
# ──────────────────────────────────────────
_PAGES = {
    " 基金概況": lambda h, r, p: page_fund_overview(h),
    " 淨值走勢": lambda h, r, p: page_fund_nav(h),
    " 基金報酬": lambda h, r, p: page_fund_return(h),
    " 基金品質": lambda h, r, p: page_fund_quality(h),
    " 資產配置": lambda h, r, p: page_allocation(h, r, p),
    " 交易紀錄": lambda h, r, p: page_history(r, p, r),
}


# ──────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────
def main():
    st.markdown(_CSS, unsafe_allow_html=True)

    loading = st.empty()
    with loading.container():
        st.write("🦊請稍候...")

    # 1. 讀取數據
    df_history, df_0050, df_00631L = load_nav_history()

    # page_allocation / page_history / 警報 仍需原始數據
    df_record,_, df_price, _, _ = load_data()

    loading.empty()

    # 2. 全域警報
    unrealized_data, _, _, _ = calculate_settlement_data(df_record, df_price)
    msgs = generate_system_alerts(df_record, df_price, df_0050, unrealized_data)
    if msgs:
        for m in msgs:
            alert(m)
    
    # 3. Sidebar 導覽
    with st.sidebar:
        st.markdown("""
        <h1 style='margin-bottom: -15px;'>小狐狸量化</h1>
        """, unsafe_allow_html=True)
        st.markdown(
            f'<span style="font-weight: normal; font-size: 16.0px;">'
            f'資料日期 : {df_history["date"].max().strftime("%Y/%m/%d")}</span>',
            unsafe_allow_html=True
        )
        st.divider()
        selection = st.radio("選單", list(_PAGES.keys()))

    # 4. 渲染對應頁面
    _PAGES[selection](df_history, df_record, df_price)


if __name__ == "__main__":
    main()