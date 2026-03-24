"""
pages.py
Kitsune Kernel — 頁面渲染層
職責：接收計算結果，負責所有 Streamlit / Plotly 渲染
規則：本檔案不得進行任何業務計算，只調用 engine 模組的函式
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import calculate_settlement_data

# ══════════════════════════════════════════
# 頁面 1：累積報酬率
# ══════════════════════════════════════════
def page_performance_pct(df_history):
    st.title("累積報酬率(%)")
    if df_history.empty:
        st.warning("無數據")
        return

    last_row   = df_history.iloc[-1]
    roi_pct    = last_row['roi_total']  * 100
    roi_0050   = last_row['roi_0050']   * 100
    roi_00631  = last_row['roi_00631L'] * 100
    alpha_0050 = (((1 + last_row['roi_total']) / (1 + last_row['roi_0050']))   - 1) * 100
    max_dd     = df_history["dd_pct"].min() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("累積報酬", f"{roi_pct:.2f}%", help="包含現金")
    col2.metric("超額報酬",  f"{alpha_0050:.2f}%")
    col3.metric("最大回撤",  f"{max_dd:,.2f}%")

    # ── 主要折線圖 ────────────────────────
    fig = go.Figure()
    configs = [
        {"y": "roi_total",  "name": "投資組合 (AI+Other)", "color": "#4F247E", "width": 2,   "visible": True},
        {"y": "roi_0050",   "name": "大盤 (0050)",         "color": "#68CCFD", "width": 2,   "visible": True},
        {"y": "roi_00631L", "name": "正二 (00631L)",       "color": "#8E44AD", "width": 2,   "visible": "legendonly"},
        {"y": "roi_ai",     "name": "AI 投資",             "color": "#16A085", "width": 1.5, "visible": "legendonly"},
        {"y": "roi_oth",    "name": "Other",               "color": "#7F8C8D", "width": 1.5, "visible": "legendonly"},
    ]
    for c in configs:
        fig.add_trace(go.Scatter(
            x=df_history['date'].dt.strftime('%Y-%m-%d'),
            y=df_history[c['y']] * 100,
            name=c['name'], mode='lines',
            line=dict(color=c['color'], width=c['width']),
            visible=c['visible'],
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
    fig.update_layout(
        template="plotly_dark", hovermode="x unified",
        xaxis=dict(type="category", tickformat="%Y-%m-%d"),
        yaxis=dict(title="累積報酬率", ticksuffix="%")
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 回撤折線圖 ────────────────────────
    st.subheader("風險監控：最大回撤幅度")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df_history['date'].dt.strftime('%Y-%m-%d'),
        y=df_history['dd_pct'] * 100,
        fill='tozeroy', fillcolor='rgba(255, 51, 51, 0.2)',
        line=dict(color="#505050", width=1.5), name="回撤(%)",
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    fig_dd.update_layout(
        template="plotly_dark", height=250, margin=dict(t=10, b=10),
        xaxis=dict(type="category", tickformat="%Y-%m-%d"),
        yaxis=dict(title="回撤 %", ticksuffix="%")
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# ══════════════════════════════════════════
# 頁面 2：累積權益淨值
# ══════════════════════════════════════════
def page_performance_nav(df_history, df_record, df_price):
    st.title("累積權益(淨值)")
    if df_history.empty:
        st.warning("無數據")
        return

    last_row = df_history.iloc[-1]
    tw = last_row['total_nav']
    ti = last_row['principal']
    tp = tw - ti

    from engine import INIT_CAP
    _, _, total_unrealized, total_realized = calculate_settlement_data(df_record, df_price)

    col1, col2, col3 = st.columns(3)
    col1.metric("總權益", f"${tw:,.0f}")
    col2.metric(
        "股票總損益", f"${tp:,.0f}",
        help=f"未實現：${total_unrealized:,.0f}\n\n已實現：${total_realized:,.0f}"
    )
    col3.metric("累積追加投入", f"{ti - INIT_CAP:,.0f}")

    # ── 主要折線圖 ────────────────────────
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
    fig.update_layout(
        template="plotly_dark", yaxis_title="TWD", hovermode="x unified",
        yaxis=dict(tickformat=",.0f"), xaxis=dict(type='category')
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 回撤金額折線圖 ────────────────────
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


# ══════════════════════════════════════════
# 頁面 3：配置比例
# ══════════════════════════════════════════
def page_allocation(df_record, df_price):
    st.title("目前配置比例")
    if df_record.empty:
        st.warning("無數據")
        return {}

    df_trade  = df_record[df_record['id'] != 'cash'].copy()
    latest_px = df_price.groupby('id').tail(1).set_index('id')['close'].to_dict() if not df_price.empty else {}

    stock_labels   = []
    stock_values   = []
    pure_equity_mv = 0.0

    # ── 純股票部位 ────────────────────────
    for aid in df_trade['id'].unique():
        sub    = df_trade[df_trade['id'] == aid]
        shares = sub[sub['action'] == 'buy']['share'].sum() - sub[sub['action'] == 'sell']['share'].sum()
        if shares > 0:
            px = latest_px.get(aid, 0.0)
            if px == 0.0:
                last_buy = sub[sub['action'] == 'buy']
                px = last_buy['price'].iloc[-1] if not last_buy.empty else 0.0
            if px > 0:
                name      = sub['name'].iloc[0] if 'name' in sub.columns else ""
                label_str = f"{aid.upper()} {name}".strip()
                mv        = shares * px
                stock_labels.append(label_str)
                stock_values.append(mv)
                pure_equity_mv += mv

    # ── 現金池 ────────────────────────────
    cash_rec  = df_record[df_record['id'] == 'cash']
    principal = cash_rec['cash_flow'].sum()
    buy_mask  = df_trade['action'] == 'buy'
    sell_mask = df_trade['action'] == 'sell'
    buy_val   = df_trade[buy_mask]['price']  * df_trade[buy_mask]['share']
    sell_val  = df_trade[sell_mask]['price'] * df_trade[sell_mask]['share']
    buy_fees  = (buy_val  * 0.001425).astype(int)
    sell_fees = (sell_val * 0.001425).astype(int) + (sell_val * 0.003).astype(int)
    current_cash = principal - (buy_val.sum() + buy_fees.sum()) + (sell_val.sum() - sell_fees.sum())

    if not stock_labels and current_cash <= 1:
        st.info("無持倉配置")
        return {}

    total_labels = stock_labels.copy()
    total_values = stock_values.copy()
    if current_cash > 1:
        total_labels.append("現金")
        total_values.append(current_cash)

    # ── 圓餅圖 ────────────────────────────
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


# ══════════════════════════════════════════
# 頁面 4：資產結算中心
# ══════════════════════════════════════════
def page_history(df_record, df_price):
    st.title("資產結算中心")
    if df_record.empty:
        st.warning("無數據")
        return

    unrealized_data, realized_data, total_unrealized, total_realized = calculate_settlement_data(df_record, df_price)

    column_config_setting = {
        "代號": st.column_config.LinkColumn(
            "代號",
            display_text=r"https://tw\.stock\.yahoo\.com/quote/(.*)"
        )
    }

    st.subheader(f"未實現損益 │ ${total_unrealized:,.0f}")
    if unrealized_data:
        df_un = pd.DataFrame(unrealized_data).sort_values("當前市值", ascending=False)
        st.dataframe(df_un, use_container_width=True, hide_index=True, column_config=column_config_setting)

    st.markdown("---")

    st.subheader(f"已實現損益 │ ${total_realized:,.0f}")
    if realized_data:
        df_re = pd.DataFrame(realized_data)
        st.dataframe(df_re, use_container_width=True, hide_index=True, column_config=column_config_setting)


# ══════════════════════════════════════════
# 頁面 5：月度績效結算
# ══════════════════════════════════════════
def page_monthly_performance(df_history):
    st.title("月度績效結算")
    if df_history.empty:
        st.warning("無數據")
        return

    df          = df_history.copy()
    df['month'] = df['date'].dt.strftime('%Y-%m')
    df['daily_ret'] = df['roi_total'].diff().fillna(0)
    m_groups    = df.groupby('month')
    monthly_end = m_groups.last().reset_index()

    # ── 月報酬、大盤月報酬、Alpha ─────────
    monthly_end['prev_nav']        = monthly_end['roi_total'].shift(1).fillna(0) + 1.0
    monthly_end['monthly_ret_pct'] = ((monthly_end['roi_total'] + 1.0) / monthly_end['prev_nav'] - 1) * 100
    monthly_end.loc[0, 'monthly_ret_pct'] = monthly_end.loc[0, 'roi_total'] * 100

    monthly_end['prev_0050'] = monthly_end['roi_0050'].shift(1).fillna(0) + 1.0
    monthly_end['m_0050']    = ((monthly_end['roi_0050'] + 1.0) / monthly_end['prev_0050'] - 1) * 100
    monthly_end.loc[0, 'm_0050'] = monthly_end.loc[0, 'roi_0050'] * 100

    monthly_end['monthly_alpha_pct'] = (
        ((1 + monthly_end['monthly_ret_pct'] / 100) / (1 + monthly_end['m_0050'] / 100)) - 1
    ) * 100

    # ── 月損益金額 ────────────────────────
    monthly_end['prev_total_nav']   = monthly_end['total_nav'].shift(1).fillna(0)
    monthly_end['principal_added']  = monthly_end['principal'] - monthly_end['principal'].shift(1).fillna(0)
    monthly_end['monthly_profit']   = (
        monthly_end['total_nav'] - monthly_end['prev_total_nav'] - monthly_end['principal_added']
    )

    # ── 月度 Sharpe / Sortino ─────────────
    risk_stats = []
    for m, group in m_groups:
        d_ret   = group['daily_ret']
        avg, std= d_ret.mean(), d_ret.std()
        sharpe  = (avg / std) * np.sqrt(252) if std > 0 else 0
        d_std   = d_ret[d_ret < 0].std()
        sortino = (avg / d_std) * np.sqrt(252) if d_std > 0 else (sharpe if avg > 0 else 0)
        risk_stats.append({'month': m, 'Sharpe': sharpe, 'Sortino': sortino})
    monthly_end = monthly_end.merge(pd.DataFrame(risk_stats), on='month')

    avg_ret, med_ret   = monthly_end['monthly_ret_pct'].mean(),   monthly_end['monthly_ret_pct'].median()
    avg_prof, med_prof = monthly_end['monthly_profit'].mean(),    monthly_end['monthly_profit'].median()

    # ── 第一列：月報酬率 & 月損益 ─────────
    r1_col1, r1_col2 = st.columns(2)

    with r1_col1:
        st.subheader("月報酬率")
        st.caption(f"Avg: {avg_ret:.1f}%  |  Med: {med_ret:.1f}%")
        colors_ret = ["#FD9292" if val > 0 else "#A4FFA4" for val in monthly_end['monthly_ret_pct']]
        fig_ret = go.Figure(go.Bar(
            x=monthly_end['month'], y=monthly_end['monthly_ret_pct'],
            marker_color=colors_ret,
            text=monthly_end['monthly_ret_pct'].round(1).astype(str) + '%',
            textposition='outside'
        ))
        fig_ret.update_layout(
            template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20), height=300,
            xaxis=dict(type="category")
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    with r1_col2:
        st.subheader("月損益")
        st.caption(f"Avg: {avg_prof:,.0f}  |  Med: {med_prof:,.0f}")
        colors_prof = ['#FD9292' if val > 0 else '#A4FFA4' for val in monthly_end['monthly_profit']]
        fig_prof = go.Figure(go.Bar(
            x=monthly_end['month'], y=monthly_end['monthly_profit'],
            marker_color=colors_prof,
            text=monthly_end['monthly_profit'].map('{:,.0f}'.format),
            textposition='outside'
        ))
        fig_prof.update_layout(
            template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20), height=300,
            xaxis=dict(type="category")
        )
        st.plotly_chart(fig_prof, use_container_width=True)

    st.caption("")

    # ── 第二列：Alpha & 夏普/索提諾 ───────
    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        st.subheader("Alpha 趨勢")
        fig_alpha = go.Figure(go.Scatter(
            x=monthly_end['month'], y=monthly_end['monthly_alpha_pct'],
            mode='lines+markers',
            line=dict(color='#16A085', width=3),
            fill='tozeroy', fillcolor='rgba(22, 160, 133, 0.1)'
        ))
        fig_alpha.update_layout(
            template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20), height=300,
            yaxis_ticksuffix="%", xaxis=dict(type="category")
        )
        st.plotly_chart(fig_alpha, use_container_width=True)

    with r2_col2:
        st.subheader("夏普/索提諾")
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
            margin=dict(l=20, r=20, t=30, b=20), height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(type="category")
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # ── 量化表格 ──────────────────────────
    st.markdown("---")
    report_df = monthly_end[['month', 'monthly_ret_pct', 'monthly_alpha_pct', 'Sharpe', 'Sortino']].copy()
    report_df.columns = ['月份', '報酬率', 'Alpha', 'Sharpe', 'Sortino']
    st.dataframe(
        report_df.style.format({'報酬率': '{:.2f}%', 'Alpha': '{:.2f}%', 'Sharpe': '{:.2f}', 'Sortino': '{:.2f}'}),
        use_container_width=True, hide_index=True
    )