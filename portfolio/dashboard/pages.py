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
    st.title(f"{name} 的權益")

    if df_equity.empty:
        st.info("尚無認購紀錄，或認購日尚未到達。")
        return

    latest  = df_equity.iloc[-1]
    first   = df_equity.iloc[0]
    equity  = latest['equity']
    capital = latest['capital']
    pnl     = equity - capital
    
    # 權益成長率 = 期末/期初權益 - 1
    first_equity = first['equity']
    equity_growth_rate = (equity / first_equity - 1) * 100 if first_equity > 0 else 0

    units   = latest['units']
    nav     = latest['nav_per_unit']
    net_cf  = latest['net_cashflow']
    redeemed = latest['total_redeemed']

    # ── Metrics ──────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("總權益",   f"${equity:,.0f}")
    col2.metric("權益成長率",   f"{equity_growth_rate:.2f}%")
    col3.metric("資本利得",   f"${pnl:,.0f}")

    col4.metric(
        "現金流量",
        f"${net_cf:,.0f}",
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
            name="權益", visible=show,
            line=dict(color='#4F247E', width=2.5),
            hovertemplate='權益：$%{y:,.0f}<extra></extra>',
        ), secondary_y=False)

        # 成本線 (現金投入)
        fig.add_trace(go.Scatter(
            x=df_w['date'], y=df_w['capital'],
            name="現金流量", visible=show,
            line=dict(color='#68CCFD', width=2.0, dash='dot'),
            hovertemplate='現金流量：$%{y:,.0f}<extra></extra>',
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
# 頁面 2：週期紀錄
# ──────────────────────────────────────────
def page_subscriber_performance(df_equity: pd.DataFrame, name: str):
    st.title(f"{name} 的週期紀錄")

    if df_equity.empty:
        st.info("尚無認購紀錄。")
        return

    # ── 計算週期績效 ────────────────────────────────────────
    from portfolio.dashboard.engine import load_cycle_periods
    df_cycles = load_cycle_periods()

    if df_cycles.empty:
        st.warning("尚未設定交易週期資料 (journal.csv 缺少 cycle 欄位)。")
        return

    first_day_equity = df_equity['equity'].iloc[0]
    rows = []

    for _, c_row in df_cycles.iterrows():
        c_num   = c_row['cycle']
        c_str   = c_row['cycle_str']
        s_date  = c_row['start_date']
        e_date  = c_row['end_date']
        d_range = c_row['date_range']

        # 抓取該週期結束日(或該日之前最新)的權益資料
        sub_df = df_equity[df_equity['date'] <= e_date]
        if sub_df.empty:
            continue
        last_row = sub_df.iloc[-1]

        # 若在此週期結束時尚未進場
        if last_row['date'] < s_date and last_row['equity'] == 0:
            continue

        c_chart = c_row.get('chart_label', f"{c_str}")
        c_table = c_row.get('table_label', f"{c_str}")

        rows.append({
            'cycle': c_num,
            'cycle_str': c_str,
            'chart_label': c_chart,
            'table_label': c_table,
            'date_range': d_range,
            'date': last_row['date'],
            'equity': last_row['equity'],
            'capital': last_row['capital'],
            'net_cashflow': last_row['net_cashflow'],
        })

    cycle_df = pd.DataFrame(rows)
    if cycle_df.empty:
        st.info("該訂閱者在此期間尚無週期參與紀錄。")
        return

    cycle_df['cycle_equity_change'] = cycle_df['equity'] - cycle_df['equity'].shift(1).fillna(0)
    cycle_df['cycle_cash_flow'] = cycle_df['capital'] - cycle_df['capital'].shift(1).fillna(0)
    cycle_df['cycle_capital_gain'] = cycle_df['cycle_equity_change'] - cycle_df['cycle_cash_flow']

    # 期初權益基準：第一期用進場時的第一天權益，後續用上一期期末權益
    cycle_df['prev_equity'] = cycle_df['equity'].shift(1).fillna(first_day_equity)

    # 單期增長率 = 該期末權益 / 上期期末(或期初)權益 - 1
    cycle_df['cycle_equity_growth_rate'] = (cycle_df['equity'] / cycle_df['prev_equity'] - 1) * 100
    cycle_df.loc[cycle_df['prev_equity'] == 0, 'cycle_equity_growth_rate'] = 0

    # 累積增長率 = 該期末權益 / 第一天期初權益 - 1 (防呆純量判斷避免產生 NaN 列)
    if first_day_equity > 0:
        cycle_df['cum_equity_growth_rate'] = (cycle_df['equity'] / first_day_equity - 1) * 100
    else:
        cycle_df['cum_equity_growth_rate'] = 0.0

    # ── 年份選單 ──────────────────────────────────────────
    years = sorted(cycle_df['date'].dt.year.dropna().astype(int).unique(), reverse=True)
    if len(years) > 1:
        selected_year = st.selectbox("選擇年份", ["全部"] + years)
        if selected_year != "全部":
            df_display = cycle_df[cycle_df['date'].dt.year == selected_year].copy()
        else:
            df_display = cycle_df.copy()
    else:
        df_display = cycle_df.copy()

    if df_display.empty:
        st.info("該年度無資料。")
        return

    # ── 雙Y軸圖：左=單期損益(柱狀)，右=單期報酬率(折線) ──
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    custom_data = df_display[['equity', 'cycle_capital_gain', 'cycle_cash_flow', 'date_range']].values

    # 股票(資本利得) - 底層 (X軸顯示：上層月份、下層週期)
    fig.add_trace(go.Bar(
        x=df_display['chart_label'],
        y=df_display['cycle_capital_gain'],
        customdata=custom_data,
        name="股票(資本利得)",
        marker=dict(color='#4F247E', line=dict(width=0)),
        hovertemplate='<b>%{x}</b> (%{customdata[3]})<br>總權益：$%{customdata[0]:,.0f}<br>現金流量：$%{customdata[2]:,.0f}<br>資本利得：$%{customdata[1]:,.0f}<extra></extra>',
    ))
    # 現金流量 - 上層
    fig.add_trace(go.Bar(
        x=df_display['chart_label'],
        y=df_display['cycle_cash_flow'],
        customdata=custom_data,
        name="現金流量",
        marker=dict(color='#B3D4FF', line=dict(width=0)),
        hovertemplate='<b>%{x}</b> (%{customdata[3]})<br>總權益：$%{customdata[0]:,.0f}<br>現金流量：$%{customdata[2]:,.0f}<br>資本利得：$%{customdata[1]:,.0f}<extra></extra>',
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)

    n_bars = len(df_display)
    default_visible = 8
    start_idx = max(0, n_bars - default_visible) - 0.5
    end_idx = n_bars - 0.5

    fig.update_layout(
        template="plotly_dark",
        hovermode="closest",
        barmode='relative',
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=30),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#5A9BD4",
            font=dict(size=15, color="black", family="Arial"),
        ),
        xaxis=dict(
            type='category',
            title="進場月份與週期",
            rangeslider=dict(visible=True, thickness=0.08),
            range=[start_idx, end_idx],
        ),
    )
    fig.update_yaxes(title_text="資產增長 (TWD)", tickformat=",.0f")

    st.plotly_chart(fig, width='stretch')

    # ── 數字明細表 ────────────────────────────────────────
    df_table = df_display[['table_label', 'date_range', 'equity', 'cycle_equity_growth_rate', 'cum_equity_growth_rate']].copy()
    df_table = df_table.dropna(subset=['equity']).iloc[::-1].reset_index(drop=True)
    df_table.columns = ['週期', '區間', '期末權益', '單期增長率(%)', '累積增長率(%)']
    df_table['期末權益']       = df_table['期末權益'].map('${:,.0f}'.format)
    df_table['單期增長率(%)'] = df_table['單期增長率(%)'].map('{:.2f}%'.format)
    df_table['累積增長率(%)'] = df_table['累積增長率(%)'].map('{:.2f}%'.format)

    st.dataframe(df_table, width='stretch', hide_index=True)
