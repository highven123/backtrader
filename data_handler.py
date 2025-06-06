import streamlit as st
import yfinance as yf
import pandas as pd
import ccxt
import akshare as ak
import baostock as bs
import requests
from datetime import datetime, timedelta

def check_network_connection():
    """检查网络连接"""
    try:
        response = requests.get("https://finance.yahoo.com", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_stock_data_yfinance(symbol, start_date, end_date):
    """使用 yfinance 获取股票数据"""
    try:
        if not check_network_connection():
            st.error("无法连接到 Yahoo Finance，请检查网络连接")
            return None
            
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            st.error(f"没有获取到 {symbol} 的数据")
            return None
            
        return data
        
    except Exception as e:
        st.error(f"获取数据失败: {str(e)}")
        return None

def get_stock_data_akshare(symbol, start_date, end_date):
    """使用 akshare 获取股票数据"""
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
    """使用 baostock 获取 A 股数据"""
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

def fetch_crypto_data(exchange_id, symbol, timeframe, since, limit=1000):
    """获取加密货币数据"""
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

def get_stock_data(symbol, start_date, end_date, data_source='yfinance'):
    """根据选择的数据源获取股票数据"""
    if data_source == 'akshare':
        return get_stock_data_akshare(symbol, start_date, end_date)
    elif data_source == 'baostock':
        return get_stock_data_baostock(symbol, start_date, end_date)
    else:  # yfinance
        return get_stock_data_yfinance(symbol, start_date, end_date) 