import backtrader as bt
import numpy as np
import pandas as pd

class BaseStrategy(bt.Strategy):
    """基础策略类，提供通用的交易记录和日志功能"""
    
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

class StopLossMixin:
    """止损混入类，提供多种止损方法"""
    
    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        
    def get_atr_stop(self, is_long=True):
        """ATR止损"""
        if is_long:
            return self.data.close[0] - self.atr[0] * self.params.atr_multiplier
        return self.data.close[0] + self.atr[0] * self.params.atr_multiplier
        
    def get_percent_stop(self, is_long=True):
        """百分比止损"""
        if is_long:
            return self.data.close[0] * (1 - self.params.stop_loss_pct)
        return self.data.close[0] * (1 + self.params.stop_loss_pct)
        
    def get_swing_stop(self, is_long=True):
        """波动止损"""
        if is_long:
            return min(self.data.low.get(size=self.params.swing_period))
        return max(self.data.high.get(size=self.params.swing_period))
        
    def get_trailing_stop(self, is_long=True):
        """追踪止损"""
        if not hasattr(self, 'highest_price'):
            self.highest_price = self.data.close[0]
            self.lowest_price = self.data.close[0]
            
        if is_long:
            self.highest_price = max(self.highest_price, self.data.close[0])
            return self.highest_price * (1 - self.params.trailing_pct)
        else:
            self.lowest_price = min(self.lowest_price, self.data.close[0])
            return self.lowest_price * (1 + self.params.trailing_pct)

class MovingAverageStrategy(BaseStrategy):
    """双均线策略"""
    
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

class RSIStrategy(BaseStrategy):
    """RSI策略"""
    
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

class MACDStrategy(BaseStrategy):
    """MACD策略"""
    
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

class TrendFollowingStrategy(BaseStrategy, StopLossMixin):
    """趋势跟随策略"""
    
    params = (
        ('atr_period', 14),
        ('atr_multiplier', 2.0),
        ('trend_period', 20),
        ('position_size', 0.1),
    )
    
    def __init__(self):
        super().__init__()
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.trend_ma = bt.indicators.SMA(self.data.close, period=self.params.trend_period)
        
    def next(self):
        if not self.position:  # 没有持仓
            if self.data.close[0] > self.trend_ma[0]:  # 价格在趋势线上方
                size = self.broker.getvalue() * self.params.position_size / self.data.close[0]
                self.buy(size=size)
                self.store_trade('buy', size, self.data.close[0])
                
        else:  # 有持仓
            stop_price = self.get_atr_stop(is_long=True)
            if self.data.close[0] < stop_price:  # 触发止损
                self.close()
                self.close_trade(self.data.close[0])

class SupplyDemandStrategy(BaseStrategy):
    """供给需求策略"""
    
    params = (
        ('lookback', 20),
        ('volume_threshold', 1.5),
        ('price_threshold', 0.02),
        ('position_size', 0.1),
    )
    
    def __init__(self):
        super().__init__()
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.params.lookback)
        self.price_ma = bt.indicators.SMA(self.data.close, period=self.params.lookback)
        
    def next(self):
        if not self.position:  # 没有持仓
            # 识别供给区
            if (self.data.volume[0] > self.volume_ma[0] * self.params.volume_threshold and
                self.data.close[0] > self.price_ma[0] * (1 + self.params.price_threshold)):
                size = self.broker.getvalue() * self.params.position_size / self.data.close[0]
                self.buy(size=size)
                self.store_trade('buy', size, self.data.close[0])
                
        else:  # 有持仓
            # 识别需求区
            if (self.data.volume[0] > self.volume_ma[0] * self.params.volume_threshold and
                self.data.close[0] < self.price_ma[0] * (1 - self.params.price_threshold)):
                self.close()
                self.close_trade(self.data.close[0])

class DingerStrategy(BaseStrategy):
    """丁格尔策略"""
    
    params = (
        ('lookback', 20),
        ('volume_threshold', 1.5),
        ('price_threshold', 0.02),
        ('position_size', 0.1),
    )
    
    def __init__(self):
        super().__init__()
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.params.lookback)
        self.price_ma = bt.indicators.SMA(self.data.close, period=self.params.lookback)
        
    def next(self):
        if not self.position:  # 没有持仓
            # 识别突破
            if (self.data.volume[0] > self.volume_ma[0] * self.params.volume_threshold and
                self.data.close[0] > self.price_ma[0] * (1 + self.params.price_threshold)):
                size = self.broker.getvalue() * self.params.position_size / self.data.close[0]
                self.buy(size=size)
                self.store_trade('buy', size, self.data.close[0])
                
        else:  # 有持仓
            # 识别反转
            if (self.data.volume[0] > self.volume_ma[0] * self.params.volume_threshold and
                self.data.close[0] < self.price_ma[0] * (1 - self.params.price_threshold)):
                self.close()
                self.close_trade(self.data.close[0])

class ScoringSignalStrategy(BaseStrategy):
    """评分制信号策略"""
    
    params = (
        ('lookback', 20),
        ('volume_weight', 1.0),
        ('price_weight', 1.0),
        ('volatility_weight', 1.0),
        ('score_threshold', 2.0),
        ('atr_period', 14),
        ('atr_multiplier', 2.0),
        ('position_size', 0.1),
    )
    
    def __init__(self):
        super().__init__()
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.params.lookback)
        self.price_ma = bt.indicators.SMA(self.data.close, period=self.params.lookback)
        
    def calculate_score(self):
        """计算综合评分"""
        volume_score = (self.data.volume[0] / self.volume_ma[0]) * self.params.volume_weight
        price_score = ((self.data.close[0] / self.price_ma[0] - 1) * 100) * self.params.price_weight
        volatility_score = (self.atr[0] / self.price_ma[0] * 100) * self.params.volatility_weight
        
        return volume_score + price_score + volatility_score
        
    def next(self):
        if not self.position:  # 没有持仓
            score = self.calculate_score()
            if score > self.params.score_threshold:  # 评分超过阈值，买入信号
                size = self.broker.getvalue() * self.params.position_size / self.data.close[0]
                self.buy(size=size)
                self.store_trade('buy', size, self.data.close[0])
                
        else:  # 有持仓
            score = self.calculate_score()
            if score < -self.params.score_threshold:  # 评分低于负阈值，卖出信号
                self.close()
                self.close_trade(self.data.close[0])

# 你可以在这里继续添加经典策略类，如双均线、RSI、MACD等 