"""
pages_portfolio.py
Kitsune Kernel — 投資者頁面層
職責：接收 FundSubscriber.calculate_equity() 的結果，負責所有渲染
規則：本檔案不得進行任何業務計算
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ──────────────────────────────────────────
# 共用繪圖格式
# ──────────────────────────────────────────
def _plotly_base(y_name: str) -> dict:
    return dict(
        template="plotly_dark",
        yaxis_title=y_name,
        hovermode="x",
        xaxis=dict(
            type='date', tickformat="%Y/%m/%d",
            rangebreaks=[dict(bounds=["sat", "mon"])],
            showspikes=True,
            spikecolor="rgba(150,150,150,0.8)",
            spikethickness=1, spikedash="solid",
            spikemode="across", spikesnap="cursor",
        ),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#4F247E",
            font=dict(size=13, color="black"),
        ),
    )

UPDATEMENU_STYLE = dict(
    type="buttons", direction="right",
    x=0.0, y=1.15, xanchor="left", yanchor="top",
    active=0, showactive=True,
    bgcolor="#dedede", bordercolor="#555", borderwidth=1,
    font=dict(color="#000000", size=12),
    pad=dict(r=5, t=5, b=2, l=2),
)


# ──────────────────────────────────────────
# 頁面 1：基金走勢
# ──────────────────────────────────────────
def page_subscriber_nav(df_equity: pd.DataFrame, name: str):
    st.title(f"{name} 的基金走勢")

    if df_equity.empty:
        st.info("尚無認購紀錄，或認購日尚未到達。")
        return

    latest  = df_equity.iloc[-1]
    equity  = latest['equity']
    capital = latest['capital']
    pnl     = equity - capital
    roi     = latest['roi']
    units   = latest['units']
    nav     = latest['nav_per_unit']
    net_cf  = latest['net_cashflow']
    redeemed = latest['total_redeemed']

    # ── Metrics ──────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("當前權益",   f"${equity:,.0f}")
    col2.metric("累積損益",   f"${pnl:,.0f}")
    col3.metric("累積報酬",   f"{roi:.2f}%")

    cf_label = "淨現金流入" if net_cf >= 0 else "淨現金流出"
    col4.metric(
        cf_label,
        f"${abs(net_cf):,.0f}",
        help=f"總投入：${capital:,.0f}\n\n總提領：${redeemed:,.0f}"
    )

    # ── 時間段按鈕 ────────────────────────────────────────
    today   = df_equity['date'].max()
    periods = ["ALL", "1M", "3M", "6M", "1Y"]
    cutoffs = {
        "ALL": df_equity['date'].min(),
        "1M":  today - pd.DateOffset(months=1),
        "3M":  today - pd.DateOffset(months=3),
        "6M":  today - pd.DateOffset(months=6),
        "1Y":  today - pd.DateOffset(years=1),
    }

    # 每個時間段各畫三條線（equity / roi / capital），用按鈕切換 visible
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    n_traces = 2  # equity, capital
    
    for i, p in enumerate(periods):
        df_w   = df_equity[df_equity['date'] >= cutoffs[p]]
        show   = (i == 0)

        # 權益
        fig.add_trace(go.Scatter(
            x=df_w['date'], 
            y=df_w['equity'],
            customdata=df_w['roi'],
            name="權益", visible=show,
            line=dict(color='#4F247E', width=2.5),
            hovertemplate='權益：$%{y:,.0f}<br>報酬率：%{customdata:.2f}%<extra></extra>',
        ), secondary_y=False)

        # 成本線
        fig.add_trace(go.Scatter(
            x=df_w['date'], y=df_w['capital'],
            name="投入成本", visible=show,
            line=dict(color='#68CCFD', width=1.5, dash='dot'),
            hovertemplate='投入成本：$%{y:,.0f}<extra></extra>',
        ), secondary_y=False)

    # 按鈕 visible 控制
    buttons = []
    for i, p in enumerate(periods):
        df_w    = df_equity[df_equity['date'] >= cutoffs[p]]
        x_start = df_w['date'].iloc[0].isoformat()
        x_end   = today.isoformat()
        vis = []
        for j in range(len(periods)):
            vis += [j == i] * n_traces
        buttons.append(dict(
            label=p, method="update",
            args=[{"visible": vis}, {"xaxis.range": [x_start, x_end]}]
        ))

    fig.update_layout(
        **_plotly_base(""),
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=50),
        updatemenus=[{**UPDATEMENU_STYLE, "buttons": buttons}],
    )
    fig.update_xaxes(rangeslider=dict(visible=False))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1,
                  secondary_y=True)

    st.plotly_chart(fig, width='stretch')
    
    st.caption(
        f"當前 NAV：{nav:.4f}　｜　持有單位數：{units:,.4f}　｜　"
        f"首次認購：{df_equity.iloc[0]['date'].strftime('%Y-%m-%d')}"
    )

# ──────────────────────────────────────────
# 頁面 2：月績效
# ──────────────────────────────────────────
def page_subscriber_performance(df_equity: pd.DataFrame, name: str):
    st.title(f"{name} 的月績效")

    if df_equity.empty:
        st.info("尚無認購紀錄。")
        return

    # ── 計算月績效 ────────────────────────────────────────
    df = df_equity.copy()
    df['ym'] = df['date'].dt.to_period('M')

    # 每月取最後一筆
    monthly = (
        df.groupby('ym')
        .last()
        .reset_index()
    )
    monthly['month_pnl'] = monthly['equity'] - monthly['equity'].shift(1).fillna(monthly['capital'].iloc[0])
    monthly['month_roi'] = monthly['roi'] - monthly['roi'].shift(1).fillna(0)
    monthly['ym_str']    = monthly['ym'].astype(str)

    # ── 年份選單 ──────────────────────────────────────────
    years = sorted(monthly['date'].dt.year.unique(), reverse=True)
    selected_year = st.selectbox("選擇年份", years)

    df_year = monthly[monthly['date'].dt.year == selected_year].copy()

    if df_year.empty:
        st.info("該年度無資料。")
        return

    # ── 雙Y軸圖：左=單月損益(柱狀)，右=單月報酬率(折線) ──
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bar_colors = [
        '#4F247E' if v >= 0 else '#C0392B'
        for v in df_year['month_pnl']
    ]

    # 單月損益（柱狀）
    fig.add_trace(go.Bar(
        x=df_year['ym_str'],
        y=df_year['month_pnl'],
        customdata=df_year['month_roi'],
        name="單月損益",
        marker_color=bar_colors,
        hovertemplate='損益：$%{y:,.0f}<br>報酬率：%{customdata:.2f}%<extra></extra>',
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)

    fig.update_layout(
        template="plotly_dark",
        hovermode="x",
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=30),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#4F247E",
            font=dict(size=13, color="black"),
        ),
        xaxis=dict(type='category'),
    )
    fig.update_yaxes(title_text="單月損益 (TWD)", tickformat=",.0f",  secondary_y=False)
    fig.update_yaxes(title_text="單月報酬率 (%)", tickformat=".2f",
                     ticksuffix="%", showgrid=False,                  secondary_y=True)

    st.plotly_chart(fig, width='stretch')

    # ── 數字明細表 ────────────────────────────────────────
    df_table = df_year[['ym_str', 'equity', 'month_pnl', 'month_roi', 'roi']].copy()
    df_table = df_table.iloc[::-1].reset_index(drop=True)  # 加這行
    df_table.columns = ['月份', '月底權益', '單月損益', '單月報酬率(%)', '累積報酬率(%)']
    df_table['月底權益']     = df_table['月底權益'].map('${:,.0f}'.format)
    df_table['單月損益']     = df_table['單月損益'].map('${:,.0f}'.format)
    df_table['單月報酬率(%)'] = df_table['單月報酬率(%)'].map('{:.2f}%'.format)
    df_table['累積報酬率(%)'] = df_table['累積報酬率(%)'].map('{:.2f}%'.format)

    st.dataframe(df_table, width='stretch', hide_index=True)
