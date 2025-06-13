import yfinance as yf
import pandas as pd
import akshare as ak
import ccxt
import requests
import os
import time

def fetch_stock_data(symbol, start_date, end_date):
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
    if df is None or df.empty:
        return None
    df.rename(columns={'开盘': 'Open', '收盘': 'Close', '最高': 'High', '最低': 'Low', '成交量': 'Volume', '日期': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def fetch_crypto_data(symbol, start_date, end_date, data_source='ccxt'):
    if data_source == 'ccxt':
        try:
            exchange = ccxt.binance()
            since = int(pd.to_datetime(start_date).timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since, limit=1000)
            if not ohlcv:
                print('ccxt 拉取失败，尝试切换到 akshare...')
                # 自动切换到 akshare
                try:
                    df = fetch_crypto_data(symbol, start_date, end_date, data_source='akshare')
                    if df is not None and not df.empty:
                        return df
                except Exception as e:
                    print(f'akshare 加密货币接口异常: {e}')
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('Date', inplace=True)
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            print(f'ccxt 异常: {e}，自动切换到 akshare...')
            return fetch_crypto_data(symbol, start_date, end_date, data_source='akshare')
    elif data_source == 'akshare':
        # akshare 仅查最新价，且无 symbol 参数
        try:
            df = ak.crypto_js_spot()
        except Exception as e:
            print(f"akshare 加密货币接口异常: {e}")
            return None
        if df is None or df.empty:
            return None
        # 过滤出目标币种（如 BTCUSDT、BTCUSD、ETHUSD 等，需与 symbol 适配）
        symbol_str = symbol.replace('/', '').upper()
        df = df[df['交易品种'].str.upper() == symbol_str]
        if df.empty:
            print(f"akshare 未找到币种 {symbol_str} 的最新价")
            return None
        # 补齐字段，确保所有字段存在
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                df[col] = 0
        # 保证所有字段为 float 类型，防止字符串混入
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # 去除所有含有 NaN 的行，防止有非法字符串或缺失
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        # 只保留需要的字段
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        # 如果只有一行，直接返回 None，防止回测端异常
        if len(df) < 2:
            print(f"akshare 仅返回一行最新价，无法用于回测")
            return None
        return df
    else:
        raise ValueError("暂不支持的数据源")

def fetch_forex_data(symbol, start_date, end_date, data_source='yfinance', alpha_vantage_key=None):
    if data_source == 'yfinance':
        try:
            df = yf.download(symbol, start=start_date, end=end_date)
            # 处理多级列名（MultiIndex）情况，降为单层列名
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns.values]
            if df is None or df.empty:
                print('yfinance 拉取失败，尝试切换到 akshare...')
                # 自动切换到 akshare
                try:
                    df = fetch_forex_data(symbol, start_date, end_date, data_source='akshare')
                    if df is not None and not df.empty:
                        return df
                except Exception as e:
                    print(f'akshare 外汇接口异常: {e}')
                return None
            df.index = pd.to_datetime(df.index)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            print(f"yfinance 外汇接口异常: {e}")
            print('yfinance 拉取失败，尝试切换到 akshare...')
            try:
                df = fetch_forex_data(symbol, start_date, end_date, data_source='akshare')
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                print(f'akshare 外汇接口异常: {e}')
            return None
    elif data_source == 'alphavantage':
        if not alpha_vantage_key:
            raise ValueError("需要提供 Alpha Vantage API KEY")
        base, quote = symbol[:3], symbol[3:6]
        url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={base}&to_symbol={quote}&outputsize=full&apikey={alpha_vantage_key}"
        r = requests.get(url)
        data = r.json().get('Time Series FX (Daily)', {})
        if not data:
            return None
        df = pd.DataFrame(data).T.astype(float)
        df.index = pd.to_datetime(df.index)
        df.columns = ['Open', 'High', 'Low', 'Close']
        df = df.sort_index()
        df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        return df
    elif data_source == 'akshare':
        # akshare 历史外汇数据，优先用 forex_hist_em，若 symbol 不支持则 fallback 到 spot
        try:
            # 先尝试 forex_hist_em（支持 EURUSD、USDJPY、EURCNYC 等）
            df = ak.forex_hist_em(symbol=symbol)
            if df is not None and not df.empty:
                df.rename(columns={'开盘': 'Open', '收盘': 'Close', '最高': 'High', '最低': 'Low', '日期': 'Date'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
                return df[['Open', 'High', 'Low', 'Close']]
        except Exception as e:
            print(f"akshare forex_hist_em 异常: {e}")
        # fallback 到 spot（仅查最新价）
        try:
            spot_df = ak.forex_spot_em()
            spot_df = spot_df[spot_df['代码'].str.upper() == symbol.upper()]
            if spot_df.empty:
                print(f"akshare 未找到外汇代码 {symbol} 的最新价")
                return None
            spot_df.rename(columns={'最新价': 'Close', '开盘价': 'Open', '最高价': 'High', '最低价': 'Low', '时间': 'Date'}, inplace=True)
            spot_df['Date'] = pd.to_datetime(spot_df['Date'], errors='coerce')
            spot_df.set_index('Date', inplace=True)
            # 补齐字段，确保所有字段存在
            for col in ['Open', 'High', 'Low', 'Close']:
                if col not in spot_df.columns:
                    spot_df[col] = 0
            # 保证所有字段为 float 类型，防止字符串混入
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in spot_df.columns:
                    spot_df[col] = pd.to_numeric(spot_df[col], errors='coerce')
            # 去除所有含有 NaN 的行，防止有非法字符串或缺失
            spot_df = spot_df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            # 只保留需要的字段
            spot_df = spot_df[['Open', 'High', 'Low', 'Close']]
            # 如果只有一行，直接返回 None，防止回测端异常
            if len(spot_df) < 2:
                print(f"akshare 仅返回一行最新价，无法用于回测")
                return None
            return spot_df
        except Exception as e:
            print(f"akshare forex_spot_em 异常: {e}")
            return None
    else:
        raise ValueError("暂不支持的数据源")

def get_data(market, symbol, start_date, end_date, fx_data_source='yfinance', alpha_vantage_key=None, crypto_data_source='ccxt'):
    if market == '沪深A股':
        return fetch_stock_data(symbol, start_date, end_date)
    elif market == '加密货币':
        return fetch_crypto_data(symbol, start_date, end_date, data_source=crypto_data_source)
    elif market == '外汇':
        return fetch_forex_data(symbol, start_date, end_date, data_source=fx_data_source, alpha_vantage_key=alpha_vantage_key)
    else:
        raise ValueError("暂不支持的市场类型")

def get_data_with_cache(
    market, symbol, start_date, end_date,
    fx_data_source='yfinance', alpha_vantage_key=None, cache_dir='cache', crypto_data_source='ccxt'
):
    """
    获取数据，优先本地缓存，若无则拉取并缓存。每次拉取新数据会覆盖原缓存。
    market: '沪深A股'/'加密货币'/'外汇'
    symbol: 标的代码
    start_date, end_date: 'YYYY-MM-DD'
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{market}_{symbol}_{start_date}_{end_date}.csv")
    # 先查缓存
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0)
            # 强制 index 为 DatetimeIndex，消除 parse_dates 警告
            df.index = pd.to_datetime(df.index, errors='coerce')
            # 修复：强制数值列为 float，去除非法/缺失行
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            if not df.empty:
                return df
        except Exception as e:
            print(f"读取缓存失败: {e}")
    # 拉取数据，自动切换数据源
    for i in range(3):
        try:
            if market == '沪深A股':
                df = fetch_stock_data(symbol, start_date, end_date)
            elif market == '加密货币':
                # 优先用用户指定的数据源，失败自动切换到 akshare/ccxt
                try_sources = [crypto_data_source]
                if crypto_data_source == 'ccxt':
                    try_sources.append('akshare')
                elif crypto_data_source == 'akshare':
                    try_sources.append('ccxt')
                for source in try_sources:
                    try:
                        df = fetch_crypto_data(symbol, start_date, end_date, data_source=source)
                        if df is not None and not df.empty:
                            break
                    except Exception as e:
                        print(f'加密货币数据源 {source} 拉取失败: {e}')
                else:
                    df = None
            elif market == '外汇':
                # 优先用用户指定的数据源，失败自动切换到 akshare/yfinance/alphavantage
                try_sources = [fx_data_source]
                for alt in ['akshare', 'yfinance', 'alphavantage']:
                    if alt not in try_sources:
                        try_sources.append(alt)
                for source in try_sources:
                    try:
                        df = fetch_forex_data(symbol, start_date, end_date, data_source=source, alpha_vantage_key=alpha_vantage_key)
                        if df is not None and not df.empty:
                            break
                    except Exception as e:
                        print(f'外汇数据源 {source} 拉取失败: {e}')
                else:
                    df = None
            else:
                df = None
            # 修复：强制数值列为 float，去除非法/缺失行
            if df is not None and not df.empty:
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
                # 修复：强制 index 为 DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, errors='coerce')
                df = df[~df.index.isna()]
                # 每次都覆盖缓存
                df.to_csv(cache_file)
                return df
        except Exception as e:
            print(f'拉取失败，第{i+1}次重试: {e}')
            time.sleep(2)
    return None