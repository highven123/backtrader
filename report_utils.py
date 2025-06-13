import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

def show_kpi_cards(metrics):
    """显示KPI指标卡片"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("年化收益率", metrics['annual_return'])
        st.metric("总盈亏金额", metrics['avg_pnl'])
        
    with col2:
        st.metric("夏普比率", metrics['sharpe_ratio'])
        st.metric("总交易次数", str(metrics['signal_freq']))
        
    with col3:
        st.metric("最大回撤", metrics['max_drawdown'])
        st.metric("收益标准差", metrics['pnl_std'])

def plot_equity_curve(equity_curve, trades=None):
    """绘制权益曲线"""
    if equity_curve is None or equity_curve.empty:
        st.warning("没有权益曲线数据可供显示")
        return
        
    try:
        fig = go.Figure()
        
        # 添加权益曲线
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='权益曲线',
            line=dict(color='#00b894')
        ))
        
        # 添加起始和结束标记
        fig.add_trace(go.Scatter(
            x=[equity_curve.index[0]],
            y=[equity_curve.iloc[0]],
            mode='markers',
            name='起始点',
            marker=dict(color='blue', size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=[equity_curve.index[-1]],
            y=[equity_curve.iloc[-1]],
            mode='markers',
            name='结束点',
            marker=dict(color='red', size=10)
        ))
        
        # 如果有交易记录，添加交易点
        if trades:
            buy_points = [t for t in trades if t['type'] == 'buy']
            sell_points = [t for t in trades if t['type'] == 'sell']
            
            if buy_points:
                fig.add_trace(go.Scatter(
                    x=[t['date'] for t in buy_points],
                    y=[t['price'] for t in buy_points],
                    mode='markers',
                    name='买入点',
                    marker=dict(color='green', size=8, symbol='triangle-up')
                ))
                
            if sell_points:
                fig.add_trace(go.Scatter(
                    x=[t['date'] for t in sell_points],
                    y=[t['price'] for t in sell_points],
                    mode='markers',
                    name='卖出点',
                    marker=dict(color='red', size=8, symbol='triangle-down')
                ))
        
        # 更新布局
        fig.update_layout(
            title="回测权益曲线",
            xaxis_title="日期",
            yaxis_title="权益",
            template="plotly_white",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"绘制权益曲线时出错: {str(e)}")

def show_trade_log(trades):
    """显示交易记录"""
    if not trades:
        st.warning("回测期间没有产生任何交易")
        return
    try:
        st.subheader("交易记录")
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            # 格式化交易记录中的数值
            for col in trades_df.select_dtypes(include=['float64']).columns:
                trades_df[col] = trades_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
            # 计算累计收益
            trades_df['cumulative_profit'] = trades_df['profit'].astype(float).cumsum()
            trades_df['cumulative_profit'] = trades_df['cumulative_profit'].apply(lambda x: f"{x:.4f}")
            # 格式化单位、持仓金额、杠杆等
            if 'unit' in trades_df.columns:
                trades_df['size'] = trades_df.apply(lambda row: f"{row['size']} {row['unit']}" if pd.notnull(row['unit']) else row['size'], axis=1)
            if 'position_value' in trades_df.columns:
                trades_df['position_value'] = trades_df['position_value'].apply(lambda x: f"{float(x):,.2f}")
            if 'actual_leverage' in trades_df.columns:
                trades_df['actual_leverage'] = trades_df['actual_leverage'].apply(lambda x: f"{float(x):.2f}x")
            # 重新排序显示重要字段
            show_cols = [
                'date', 'type', 'side', 'size', 'price', 'position_value', 'actual_leverage',
                'leverage', 'sell_date', 'sell_price', 'profit', 'cumulative_profit'
            ]
            show_cols = [c for c in show_cols if c in trades_df.columns]
            st.dataframe(trades_df[show_cols])
        else:
            st.warning("交易记录为空")
    except Exception as e:
        st.warning(f"处理交易记录时出错: {str(e)}")

def save_backtest_result(symbol, strategy_type, params, results):
    """保存回测结果到历史记录"""
    if results is None:
        return
        
    history_item = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': symbol,
        'strategy_type': strategy_type,
        'params': params,
        'total_return': results.get('total_return', 0.0),
        'sharpe_ratio': results.get('sharpe_ratio', 0.0),
        'max_drawdown': results.get('max_drawdown', 0.0),
        'win_rate': results.get('win_rate', 0.0),
        'total_trades': results.get('total_trades', 0),
        'total_won': results.get('total_won', 0)
    }
    
    if 'backtest_history' not in st.session_state:
        st.session_state.backtest_history = []
        
    st.session_state.backtest_history.append(history_item)

def display_backtest_history():
    """显示回测历史记录"""
    if not st.session_state.get('backtest_history'):
        st.info("还没有回测历史记录")
        return
        
    st.subheader("回测历史记录")
    
    # 转换历史记录为DataFrame
    history_df = pd.DataFrame(st.session_state.backtest_history)
    
    # 格式化数值列
    format_cols = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    for col in format_cols:
        history_df[col] = history_df[col].apply(lambda x: f"{float(x):.2f}%")
    
    # 显示历史记录表格
    st.dataframe(
        history_df,
        column_config={
            'timestamp': '时间',
            'symbol': '交易标的',
            'strategy_type': '策略类型',
            'total_return': '总收益率',
            'sharpe_ratio': '夏普比率',
            'max_drawdown': '最大回撤',
            'win_rate': '胜率',
            'total_trades': '总交易次数',
            'total_won': '盈利交易数'
        },
        hide_index=True
    )
    
    # 添加清除历史记录的按钮
    if st.button("清除历史记录"):
        st.session_state.backtest_history = []
        st.rerun()

def show_signal_heatmap(signal_df: pd.DataFrame):
    st.write("信号热力图（可用于调试信号触发频率）")
    st.dataframe(signal_df)