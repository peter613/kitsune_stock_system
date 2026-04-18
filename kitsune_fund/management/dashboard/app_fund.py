# streamlit run c:/Users/Lynn610/Documents/kitsune_stock_system/dashboard/main.py
"""
dashboard.py
Kitsune Kernel — 主程式入口
職責：CSS 注入、資料初始化、警報觸發、Sidebar 路由
啟動：streamlit run dashboard.py
"""

import streamlit as st

from kitsune_fund.valuation.engine import (
    load_data,
    calculate_daily_history,
    calculate_settlement_data,
    generate_system_alerts
)
from pages_fund import (
    page_fund_overview,
    page_fund_nav,
    page_fund_return,
    page_fund_quality,
    page_allocation,
    page_history,
)

# ──────────────────────────────────────────
# 全域設定
# ──────────────────────────────────────────
st.set_page_config(page_title="Kitsune Kernel", layout="wide")

_CSS = """
<style>
/* ── 留白修正 ── */
[data-testid="stAppViewContainer"] {
    padding-top: 0rem !important;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 3rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #fcf5ed !important;
    color: #1a1a1a !important;
    font-size: 1.05rem !important;
}
/* ── 小狐狸基金置頂空白 ── */
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
    padding-top: 1.5rem !important;
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
[data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] p {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: #C0392B !important;
    padding-top: 0.3rem !important;
}

/* ── 一般 markdown 文字 ── */
[data-testid="stMarkdownContainer"] p {
    color: #1a1a1a !important;
    font-size: 1.2rem !important;
    margin-bottom: 5px !important;
}
</style>
"""
def alert(message):
    """自定義深藍色戰情警報，完全跳過 Streamlit 內建組件"""
    html_code = f"""
    <div style="
        background-color: #1A2B4C;
        color: #ffffff;
        padding: 15px 20px;
        border: none;
        margin: 50px 20px 10px 0px;
        font-weight: 500;
        font-size: 1.1rem;
        letter-spacing: 2px;
    ">
        {message}
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

# ──────────────────────────────────────────
# 頁面路由對應表
# ──────────────────────────────────────────
_PAGES = {
    " 基金概況": lambda h, r, p: page_fund_overview(h),
    " 淨值走勢": lambda h, r, p: page_fund_nav(h),
    " 基金報酬": lambda h, r, p: page_fund_return(h),
    " 基金績效": lambda h, r, p: page_fund_quality(h),
    " 資產配置": lambda h, r, p: page_allocation(h,r, p),
    " 交易紀錄": lambda h, r, p: page_history(r, p,r),
}

# ──────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────
def main():
    st.markdown(_CSS, unsafe_allow_html=True)

    # 1. 資料讀取（含 loading 佔位符）
    loading = st.empty()
    with loading.container():
        st.write("🦊請稍候...")

    df_record, df_flows, df_price, df_0050, df_00631L = load_data()
    df_history = calculate_daily_history(df_record, df_flows, df_price, df_0050, df_00631L)
    loading.empty()

    # 2. 全域警報（大盤波動率 → 再平衡 → 個股停損）
    unrealized_data, _, _, _ = calculate_settlement_data(df_record, df_price)

    msgs = generate_system_alerts(df_record, df_price, df_0050, unrealized_data)
    if msgs:
        for m in msgs:
            alert(m)

    # 3. Sidebar 導覽
    with st.sidebar:
        st.markdown("""
        <h1 style='margin-bottom: -15px;'>小狐狸基金</h1>
        """, unsafe_allow_html=True)
        st.markdown(
            f'<span style="font-weight: normal; font-size: 16.0px;">資料日期 : {df_history["date"].max().strftime("%Y/%m/%d")}</span>'
            , unsafe_allow_html=True
        )
        st.divider()
        selection = st.radio("選單", list(_PAGES.keys()))
        # st.markdown("<br>" * 5, unsafe_allow_html=True)
    # 4. 渲染對應頁面
    _PAGES[selection](df_history, df_record, df_price)

if __name__ == "__main__":
    main()