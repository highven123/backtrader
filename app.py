import streamlit as st
import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import time
import json
import requests
import akshare as ak
import baostock as bs
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
import tempfile
import os
from strategy_handler import (
    MovingAverageStrategy, RSIStrategy, MACDStrategy,
    TrendFollowingStrategy, SupplyDemandStrategy,
    DingerStrategy, ScoringSignalStrategy
)
from data_handler import get_stock_data, fetch_crypto_data, get_stock_data_yfinance
from report_utils import (
    show_kpi_cards, plot_equity_curve, show_trade_log,
    save_backtest_result, display_backtest_history
)

# 初始化回测历史记录
if 'backtest_history' not in st.session_state:
    st.session_state.backtest_history = []

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
    
    st.session_state.backtest_history.append(history_item)

def display_backtest_history():
    """显示回测历史记录"""
    if not st.session_state.backtest_history:
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

# 基础策略类
class BaseStrategy(bt.Strategy):
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        self.trades_log.append(f'{dt.isoformat()} {txt}')

    def __init__(self):
        self.trades_log = []
        self.trades = []
        self.current_trade = None

    def store_trade(self, type, size, price):
        self.current_trade = {
            'type': type,
            'date': self.data.datetime.date(),
            'price': price,
            'size': size,
            'value': size * price
        }

    def close_trade(self, price):
        if self.current_trade:
            self.current_trade.update({
                'sell_date': self.data.datetime.date(),
                'sell_price': price,
                'profit': (price - self.current_trade['price']) * self.current_trade['size']
            })
            self.trades.append(self.current_trade)
            self.current_trade = None

# 双均线策略
class MovingAverageStrategy(BaseStrategy):
    params = (
        ('short_window', 20),
        ('long_window', 50),
        ('position_size', 0.1),
    )
    
    def __init__(self):
        super().__init__()
        self.short_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.short_window)
        self.long_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.long_window)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)
        
    def next(self):
        if not self.position:  # 没有持仓
            if self.crossover > 0:  # 金叉，买入信号
                size = self.broker.getvalue() * self.params.position_size / self.data.close[0]
                self.buy(size=size)
                self.store_trade('buy', size, self.data.close[0])
                
        else:  # 有持仓
            if self.crossover < 0:  # 死叉，卖出信号
                self.close()
                self.close_trade(self.data.close[0])

# 多周期双均线策略
class MultiPeriodMAStrategy(BaseStrategy):
    params = (
        ('daily_short', 5),   # 日线短期均线
        ('daily_long', 20),   # 日线长期均线
        ('weekly_short', 4),  # 周线短期均线
        ('weekly_long', 12),  # 周线长期均线
        ('position_size', 0.1),
    )
    
    def __init__(self):
        super().__init__()
        
        # 日线数据
        self.daily_data = self.datas[0]
        # 周线数据
        self.weekly_data = self.datas[1] if len(self.datas) > 1 else self.datas[0]
        
        # 日线均线
        self.daily_short_ma = bt.indicators.SMA(
            self.daily_data.close, period=self.params.daily_short)
        self.daily_long_ma = bt.indicators.SMA(
            self.daily_data.close, period=self.params.daily_long)
        
        # 周线均线
        self.weekly_short_ma = bt.indicators.SMA(
            self.weekly_data.close, period=self.params.weekly_short)
        self.weekly_long_ma = bt.indicators.SMA(
            self.weekly_data.close, period=self.params.weekly_long)
        
        # 计算趋势方向
        self.daily_trend = self.daily_short_ma > self.daily_long_ma
        self.weekly_trend = self.weekly_short_ma > self.weekly_long_ma
        
    def next(self):
        # 确保有足够的数据进行计算
        if not all([self.daily_short_ma[0], self.daily_long_ma[0],
                   self.weekly_short_ma[0], self.weekly_long_ma[0]]):
            return
            
        if not self.position:  # 没有持仓
            # 当日线趋势向上，且得到周线趋势确认时买入
            if (self.daily_trend[0] and not self.daily_trend[-1] and  # 日线由空转多
                self.weekly_trend[0]):  # 周线处于上升趋势
                size = self.broker.getvalue() * self.params.position_size / self.data0.close[0]
                self.buy(size=size)
                self.store_trade('buy', size, self.data0.close[0])
                
        else:  # 有持仓
            # 当日线趋势向下，或周线趋势向下时卖出
            if ((not self.daily_trend[0] and self.daily_trend[-1]) or  # 日线由多转空
                (not self.weekly_trend[0] and self.weekly_trend[-1])):  # 周线由多转空
                self.close()
                self.close_trade(self.data0.close[0])

# RSI策略
class RSIStrategy(BaseStrategy):
    params = (
        ('rsi_period', 14),
        ('rsi_entry_oversold', 30),
        ('rsi_exit_oversold', 45),
        ('rsi_entry_overbought', 70),
        ('position_size', 0.1),
    )
    
    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(
            self.data.close, period=self.params.rsi_period)
    
    def next(self):
        if not self.position:  # 没有持仓
            if self.rsi < self.params.rsi_entry_oversold:  # RSI超卖，买入信号
                size = self.broker.getvalue() * self.params.position_size / self.data.close[0]
                self.buy(size=size)
                self.store_trade('buy', size, self.data.close[0])
                
        else:  # 有持仓
            if self.rsi > self.params.rsi_entry_overbought:  # RSI超买，卖出信号
                self.close()
                self.close_trade(self.data.close[0])
            elif self.rsi > self.params.rsi_exit_oversold:  # RSI超卖退出，止盈
                self.close()
                self.close_trade(self.data.close[0])

# MACD策略
class MACDStrategy(BaseStrategy):
    params = (
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('position_size', 0.1),
    )
    
    def __init__(self):
        super().__init__()
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal)
    
    def next(self):
        if not self.position:  # 没有持仓
            if self.macd.macd > self.macd.signal:  # MACD金叉，买入信号
                size = self.broker.getvalue() * self.params.position_size / self.data.close[0]
                self.buy(size=size)
                self.store_trade('buy', size, self.data.close[0])
                
        else:  # 有持仓
            if self.macd.macd < self.macd.signal:  # MACD死叉，卖出信号
                self.close()
                self.close_trade(self.data.close[0])

def fetch_crypto_data(exchange_id, symbol, timeframe, since, limit=1000):
    try:
        exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })
        
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        if not ohlcv:
            st.error(f"没有获取到 {symbol} 的数据")
            return None
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # 验证数据有效性
        if df.empty:
            st.error("获取到的数据为空")
            return None
            
        # 检查是否有无效值
        if df.isnull().any().any():
            st.warning("数据中存在无效值，将被移除")
            df = df.dropna()
            
        # 检查是否有零或负值
        invalid_prices = (df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)
        if invalid_prices.any():
            st.warning("数据中存在无效的价格（零或负值），将被移除")
            df = df[~invalid_prices]
            
        if df.empty:
            st.error("清理后的数据为空")
            return None
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # 确保列名称正确
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df
        
    except Exception as e:
        st.error(f"获取加密货币数据时发生错误: {str(e)}")
        return None

def check_network_connection():
    """
    检查是否可以访问Yahoo Finance
    """
    try:
        # 测试连接Yahoo Finance
        response = requests.get("https://finance.yahoo.com", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_stock_data_akshare(symbol, start_date, end_date):
    """
    使用 akshare 获取股票数据
    """
    try:
        # 转换股票代码格式
        if '.SS' in symbol:
            code = f"sh{symbol.replace('.SS', '')}"
        elif '.SZ' in symbol:
            code = f"sz{symbol.replace('.SZ', '')}"
        elif '.HK' in symbol:
            code = f"{symbol.replace('.HK', '')}"
        else:
            code = symbol
            
        # 获取数据
        if '.HK' in symbol:  # 港股
            df = ak.stock_hk_daily(symbol=code, adjust="qfq")
        elif '.SS' in symbol or '.SZ' in symbol:  # A股
            df = ak.stock_zh_a_daily(symbol=code, adjust="qfq")
        else:  # 美股
            df = ak.stock_us_daily(symbol=code, adjust="qfq")
            
        # 统一列名
        df = df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
    
        # 设置日期索引
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 过滤日期范围
        df = df[(df.index >= pd.to_datetime(start_date)) & 
               (df.index <= pd.to_datetime(end_date))]
        
        return df
        
    except Exception as e:
        st.error(f"获取数据失败: {str(e)}")
        return None
        
def get_stock_data_baostock(symbol, start_date, end_date):
    """
    使用 baostock 获取 A 股数据
    """
    try:
        # 登录系统
        bs.login()
        
        # 转换股票代码格式
        if '.SS' in symbol:
            code = f"sh.{symbol.replace('.SS', '')}"
        elif '.SZ' in symbol:
            code = f"sz.{symbol.replace('.SZ', '')}"
        else:
            st.error("Baostock 仅支持 A 股数据")
            return None
            
        # 获取数据
        rs = bs.query_history_k_data(
            code,
            "date,open,high,low,close,volume",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="2"  # 前复权
        )
        
        # 转换为 DataFrame
        df = pd.DataFrame(rs.data, columns=rs.fields)
        
        # 转换数据类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # 统一列名
        df = df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # 设置日期索引
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 登出系统
        bs.logout()
        
        return df
        
    except Exception as e:
        st.error(f"获取数据失败: {str(e)}")
        bs.logout()
        return None

def get_stock_data(symbol, start_date, end_date, data_source='yfinance'):
    """
    根据选择的数据源获取股票数据
    """
    if data_source == 'akshare':
        return get_stock_data_akshare(symbol, start_date, end_date)
    elif data_source == 'baostock':
        return get_stock_data_baostock(symbol, start_date, end_date)
    else:  # yfinance
        return get_stock_data_yfinance(symbol, start_date, end_date)

def display_backtest_results(results):
    if results is None:
        st.error("没有回测结果可供显示")
        return
        
    try:
        # 创建结果展示
        col1, col2, col3 = st.columns(3)
        
        # 安全地获取和格式化指标值
        def format_metric(value, default=0.0, precision=2, percentage=True):
            try:
                if value is None:
                    value = default
                formatted = f"{float(value):.{precision}f}"
                return f"{formatted}%" if percentage else formatted
            except (TypeError, ValueError):
                return f"{default:.{precision}f}%" if percentage else f"{default:.{precision}f}"
        
        # 计算总盈亏金额
        trades = results.get('trades', [])
        total_profit = sum(float(trade['profit']) for trade in trades) if trades else 0.0
        
        with col1:
            total_return = results.get('total_return')
            st.metric("总收益率", format_metric(total_return))
            st.metric("总盈亏金额", f"${total_profit:.2f}")
            
        with col2:
            sharpe_ratio = results.get('sharpe_ratio')
            st.metric("夏普比率", format_metric(sharpe_ratio, percentage=False))
            
            total_trades = results.get('total_trades', 0)
            st.metric("总交易次数", str(total_trades))
            
        with col3:
            max_drawdown = results.get('max_drawdown')
            st.metric("最大回撤", format_metric(max_drawdown))
            
            win_rate = results.get('win_rate')
            st.metric("胜率", format_metric(win_rate))
                
        # 绘制权益曲线
        equity_curve = results.get('equity_curve')
        if equity_curve is not None and not equity_curve.empty:
            try:
                fig = go.Figure()
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
        else:
            st.warning("没有权益曲线数据可供显示")
        
        # 显示交易记录
        trades = results.get('trades', [])
        if trades:
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
                    
                    st.dataframe(trades_df)
                else:
                    st.warning("交易记录为空")
            except Exception as e:
                st.warning(f"处理交易记录时出错: {str(e)}")
        else:
            st.warning("回测期间没有产生任何交易")
            
        # 显示交易日志
        if results.get('trades_log'):
            st.subheader("交易日志")
            for log in results['trades_log']:
                st.text(log)
        
    except Exception as e:
        st.error(f"显示回测结果时发生错误: {str(e)}")
        st.error(f"错误类型: {type(e).__name__}")
        st.error(f"结果数据类型: {type(results)}")
        st.error(f"结果内容: {results}")

def run_backtest(data, strategy_class, strategy_params):
    """运行回测"""
    if data is None or data.empty:
        st.error("没有有效的数据用于回测")
        return None
        
    try:
        cerebro = bt.Cerebro()
        # 创建日线数据源
        data_feed = bt.feeds.PandasData(
            dataname=data,
            datetime=None,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=-1
        )
        cerebro.adddata(data_feed, name='Daily')
        # 从策略参数中分离出初始资金
        initial_cash = strategy_params.pop('initial_cash', 10000)
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001)
        # 添加策略
        cerebro.addstrategy(strategy_class, **strategy_params)
        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        results = cerebro.run()
        if not results or len(results) == 0:
            st.error("回测没有产生结果")
            return None
        strategy = results[0]
        final_value = cerebro.broker.getvalue()
        total_return = ((final_value - initial_cash) / initial_cash * 100)
        # 获取分析结果
        def safe_get(analyzer, *keys, default=0.0):
            try:
                value = analyzer.get_analysis()
                for key in keys:
                    value = getattr(value, key, None)
                    if value is None:
                        return default
                return value
            except (KeyError, AttributeError):
                return default
        sharpe_ratio = safe_get(strategy.analyzers.sharpe, 'sharperatio', default=0.0)
        max_drawdown = safe_get(strategy.analyzers.drawdown, 'max', 'drawdown', default=0.0)
        trades_analysis = strategy.analyzers.trades.get_analysis()
        try:
            total_trades = trades_analysis.total.total if hasattr(trades_analysis, 'total') else 0
            total_won = trades_analysis.won.total if hasattr(trades_analysis, 'won') else 0
            win_rate = (total_won / total_trades * 100) if total_trades > 0 else 0.0
        except Exception as e:
            total_trades = 0
            total_won = 0
            win_rate = 0.0
        # 获取权益曲线数据
        portfolio_value = strategy.broker.get_value()
        equity_curve = pd.Series(index=data.index)
        equity_curve.iloc[0] = initial_cash
        equity_curve.iloc[-1] = portfolio_value
        equity_curve = equity_curve.interpolate(method='linear')
        # 将初始资金重新添加回策略参数
        strategy_params['initial_cash'] = initial_cash
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trades': getattr(strategy, 'trades', []),
            'equity_curve': equity_curve,
            'trades_log': getattr(strategy, 'trades_log', []),
            'total_trades': total_trades,
            'total_won': total_won
        }
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        st.error(f"回测执行过程中发生错误: {str(e)}")
        st.error("详细错误信息:")
        st.code(error_msg)
        st.error(f"策略参数: {strategy_params}")
        st.error(f"数据信息: 行数={len(data) if data is not None else 0}, 列={list(data.columns) if data is not None else []}")
        return None

def main():
    st.set_page_config(page_title="策略回测系统", layout="wide")

    st.title("📈 多市场策略回测系统")

    # 创建标签页
    tab1, tab2 = st.tabs(["回测", "历史记录"])

    with tab1:
        # 侧边栏参数
        st.sidebar.header("参数设置")
        
        # 在侧边栏添加市场选择
        market = st.sidebar.selectbox(
            "市场",
            ["美股", "港股", "沪深A股", "加密货币"],
            help="选择你要回测的市场类型，系统会自动匹配数据源和代码格式。"
        )

        # 根据市场自动设置数据源和代码提示和默认代码
        if market == "美股":
            data_source = "yfinance"
            symbol_hint = "如：AAPL"
            symbol_default = "AAPL"
        elif market == "港股":
            data_source = "akshare"
            symbol_hint = "如：00700.HK"
            symbol_default = "00700.HK"
        elif market == "沪深A股":
            data_source = "akshare"
            symbol_hint = "如：600519.SS 或 000001.SZ"
            symbol_default = "600519.SS"
        else:
            data_source = "ccxt"
            symbol_hint = "如：BTC/USDT"
            symbol_default = "BTC/USDT"

        symbol = st.sidebar.text_input("标的代码", value=symbol_default, help=symbol_hint)
        
        # 日期选择（先开始日期，后结束日期）
        start_date = st.sidebar.date_input(
            "开始日期",
            value=datetime.now() - timedelta(days=365)
        )
        end_date = st.sidebar.date_input("结束日期", value=datetime.now())
        
        # 策略选择及提示
        strategy_name = st.sidebar.selectbox(
            "策略",
            [
                "双均线策略",
                "RSI策略",
                "MACD策略",
                "趋势跟随策略",
                "供给需求策略",
                "丁格尔策略",
                "评分制信号策略"
            ],
            help="选择回测用的交易策略。"
        )

        # 策略说明字典
        strategy_descriptions = {
            "双均线策略": "短期均线与长期均线交叉作为买卖信号，适合趋势行情。参数建议：短期5-20，长期20-60。",
            "RSI策略": "RSI指标用于判断超买超卖，适合震荡行情。参数建议：周期14，超卖30，超买70。",
            "MACD策略": "MACD金叉死叉信号，适合趋势行情。参数建议：快线12，慢线26，信号9。",
            "趋势跟随策略": "结合ATR和趋势线，适合波动较大的市场。参数建议：ATR周期14，乘数2。",
            "供给需求策略": "通过成交量和价格突破识别供需区，适合箱体震荡。参数建议：回看20，放量1.5。",
            "丁格尔策略": "结合成交量和价格波动，识别市场结构变化。参数建议：回看20，放量1.5。",
            "评分制信号策略": "综合多个因子（成交量、价格、波动率等）进行评分，分数超过阈值则买入。适合多因子择时。参数建议：各因子权重1，阈值2。"
        }

        # 在主页面展示策略说明
        st.markdown(f"**策略说明：** {strategy_descriptions.get(strategy_name, '')}")

        with st.sidebar.expander("策略参数", expanded=True):
            st.markdown(f"**参数建议：** {strategy_descriptions.get(strategy_name, '')}")
            if strategy_name == "双均线策略":
                st.info("短期均线与长期均线交叉作为买卖信号，适合趋势行情。")
                short_window = st.number_input("短期均线周期", 5, 50, 20, help="如5日均线")
                long_window = st.number_input("长期均线周期", 10, 200, 50, help="如20日均线")
                position_size = st.number_input("仓位大小", 0.01, 1.0, 0.1, step=0.01, help="每次买入资金占比")
                strategy_class = MovingAverageStrategy
                strategy_params = {
                    'short_window': short_window,
                    'long_window': long_window,
                    'position_size': position_size
                }
            elif strategy_name == "RSI策略":
                st.info("RSI指标用于判断超买超卖，适合震荡行情。")
                rsi_period = st.number_input("RSI周期", 5, 50, 14, help="常用14")
                rsi_entry_oversold = st.number_input("RSI超卖阈值", 10, 40, 30, help="低于此值买入")
                rsi_exit_oversold = st.number_input("RSI超卖退出阈值", 30, 60, 45, help="止盈阈值")
                rsi_entry_overbought = st.number_input("RSI超买阈值", 60, 90, 70, help="高于此值卖出")
                position_size = st.number_input("仓位大小", 0.01, 1.0, 0.1, step=0.01, help="每次买入资金占比")
                strategy_class = RSIStrategy
                strategy_params = {
                    'rsi_period': rsi_period,
                    'rsi_entry_oversold': rsi_entry_oversold,
                    'rsi_exit_oversold': rsi_exit_oversold,
                    'rsi_entry_overbought': rsi_entry_overbought,
                    'position_size': position_size
                }
            elif strategy_name == "MACD策略":
                st.info("MACD金叉死叉信号，适合趋势行情。")
                macd_fast = st.number_input("MACD快线周期", 5, 30, 12, help="常用12")
                macd_slow = st.number_input("MACD慢线周期", 10, 50, 26, help="常用26")
                macd_signal = st.number_input("MACD信号周期", 5, 20, 9, help="常用9")
                position_size = st.number_input("仓位大小", 0.01, 1.0, 0.1, step=0.01, help="每次买入资金占比")
                strategy_class = MACDStrategy
                strategy_params = {
                    'macd_fast': macd_fast,
                    'macd_slow': macd_slow,
                    'macd_signal': macd_signal,
                    'position_size': position_size
                }
            elif strategy_name == "趋势跟随策略":
                st.info("结合ATR和趋势线，适合波动较大的市场。")
                atr_period = st.number_input("ATR周期", 5, 50, 14, help="波动率计算周期")
                atr_multiplier = st.number_input("ATR乘数", 0.5, 5.0, 2.0, step=0.1, help="止损宽度")
                trend_period = st.number_input("趋势周期", 5, 50, 20, help="趋势判断周期")
                position_size = st.number_input("仓位大小", 0.01, 1.0, 0.1, step=0.01, help="每次买入资金占比")
                strategy_class = TrendFollowingStrategy
                strategy_params = {
                    'atr_period': atr_period,
                    'atr_multiplier': atr_multiplier,
                    'trend_period': trend_period,
                    'position_size': position_size
                }
            elif strategy_name == "供给需求策略":
                st.info("通过成交量和价格突破识别供需区，适合箱体震荡。")
                lookback = st.number_input("回看周期", 5, 50, 20, help="历史窗口长度")
                volume_threshold = st.number_input("成交量阈值", 1.0, 3.0, 1.5, step=0.1, help="放量倍数")
                price_threshold = st.number_input("价格阈值", 0.01, 0.05, 0.02, step=0.01, help="价格突破比例")
                position_size = st.number_input("仓位大小", 0.01, 1.0, 0.1, step=0.01, help="每次买入资金占比")
                strategy_class = SupplyDemandStrategy
                strategy_params = {
                    'lookback': lookback,
                    'volume_threshold': volume_threshold,
                    'price_threshold': price_threshold,
                    'position_size': position_size
                }
            elif strategy_name == "丁格尔策略":
                st.info("结合成交量和价格波动，识别市场结构变化。")
                lookback = st.number_input("回看周期", 5, 50, 20, help="历史窗口长度")
                volume_threshold = st.number_input("成交量阈值", 1.0, 3.0, 1.5, step=0.1, help="放量倍数")
                price_threshold = st.number_input("价格阈值", 0.01, 0.05, 0.02, step=0.01, help="价格突破比例")
                position_size = st.number_input("仓位大小", 0.01, 1.0, 0.1, step=0.01, help="每次买入资金占比")
                strategy_class = DingerStrategy
                strategy_params = {
                    'lookback': lookback,
                    'volume_threshold': volume_threshold,
                    'price_threshold': price_threshold,
                    'position_size': position_size
                }
            else:  # 评分制信号策略
                st.info("综合多个因子评分，适合多因子择时。")
                lookback = st.number_input("回看周期", 5, 60, 20, help="历史窗口长度")
                volume_weight = st.number_input("成交量评分权重", 0.0, 5.0, 1.0, step=0.1, help="成交量因子权重")
                price_weight = st.number_input("价格评分权重", 0.0, 5.0, 1.0, step=0.1, help="价格因子权重")
                volatility_weight = st.number_input("波动率评分权重", 0.0, 5.0, 1.0, step=0.1, help="波动率因子权重")
                score_threshold = st.number_input("信号评分阈值", 0.5, 5.0, 2.0, step=0.1, help="生成信号的评分阈值")
                atr_period = st.number_input("ATR周期", 5, 50, 14, help="波动率计算周期")
                atr_multiplier = st.number_input("ATR乘数", 0.5, 5.0, 2.0, step=0.1, help="止损宽度")
                position_size = st.number_input("仓位大小", 0.01, 1.0, 0.1, step=0.01, help="每次买入资金占比")
                strategy_class = ScoringSignalStrategy
                strategy_params = {
                    'lookback': lookback,
                    'volume_weight': volume_weight,
                    'price_weight': price_weight,
                    'volatility_weight': volatility_weight,
                    'score_threshold': score_threshold,
                    'atr_period': atr_period,
                    'atr_multiplier': atr_multiplier,
                    'position_size': position_size
                }
        
        # 添加初始资金设置
        initial_cash = st.sidebar.number_input(
            "初始资金",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000,
            help="回测账户的初始资金"
        )
        strategy_params['initial_cash'] = initial_cash
        
        # 执行回测按钮
        if st.sidebar.button("执行回测"):
            with st.spinner("正在获取数据..."):
                if data_source == "ccxt":
                    data = fetch_crypto_data(
                        "binance",
                        symbol,
                        "1d",
                        int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
                    )
                else:
                    data = get_stock_data(
                        symbol,
                        start_date,
                        end_date,
                        data_source=data_source
                    )
                    
            if data is None or len(data) == 0:
                st.error("数据获取失败，请检查标的代码和数据源！")
            else:
                st.success(f"成功获取 {len(data)} 条数据")
                
                # 执行回测
                results = run_backtest(data, strategy_class, strategy_params)
                
                if results:
                    # 显示回测结果
                    metrics = {
                        'annual_return': f"{results['total_return']:.2f}%",
                        'sharpe_ratio': f"{results['sharpe_ratio']:.2f}",
                        'max_drawdown': f"{results['max_drawdown']:.2f}%",
                        'signal_freq': results['total_trades'],
                        'avg_pnl': f"{sum(t['profit'] for t in results['trades'])/len(results['trades']) if results['trades'] else 0:.2f}",
                        'pnl_std': f"{pd.Series([t['profit'] for t in results['trades']]).std() if results['trades'] else 0:.2f}"
                    }
                    
                    show_kpi_cards(metrics)
                    plot_equity_curve(results['equity_curve'], results['trades'])
                    show_trade_log(results['trades'])
                    
                    # 保存回测结果到历史记录
                    save_backtest_result(symbol, strategy_name, strategy_params, results)
                    
                    # 显示交易日志
                    if results.get('trades_log'):
                        st.subheader("交易日志")
                        for log in results['trades_log']:
                            st.text(log)
                    
    with tab2:
        # 显示回测历史记录
        display_backtest_history()

if __name__ == "__main__":
    main()