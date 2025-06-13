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
        # 杠杆和方向参数，默认1倍多头
        self.leverage = getattr(self.params, 'leverage', 1)
        self.trade_side = getattr(self.params, 'trade_side', '多头')

    def store_trade(self, type, size, price):
        # 判断单位
        if hasattr(self.data, 'symbol'):
            symbol = getattr(self.data, 'symbol', '')
        else:
            symbol = ''
        # 简单规则：加密货币用“张”，外汇用“手”，其它用“股”
        if 'USD' in symbol or 'EUR' in symbol or 'JPY' in symbol or 'GBP' in symbol:
            unit = '手'
        elif '/' in symbol:
            unit = '张'
        else:
            unit = '股'
        # 持仓金额 = size * price * leverage
        position_value = size * price * self.leverage
        # 实际杠杆 = 持仓金额 / 本金
        capital = self.broker.getvalue() / self.leverage if self.leverage > 0 else self.broker.getvalue()
        actual_leverage = position_value / capital if capital > 0 else self.leverage
        self.current_trade = {
            'type': type,
            'date': self.data.datetime.date(),
            'price': price,
            'size': size,
            'unit': unit,
            'value': size * price,
            'leverage': self.leverage,
            'side': self.trade_side,
            'position_value': position_value,
            'actual_leverage': actual_leverage
        }

    def close_trade(self, price):
        if self.current_trade:
            profit = (price - self.current_trade['price']) * self.current_trade['size']
            # 空头反向
            if self.current_trade.get('side', '多头') == '空头':
                profit = -profit
            self.current_trade.update({
                'sell_date': self.data.datetime.date(),
                'sell_price': price,
                'profit': profit * self.current_trade.get('leverage', 1)
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
        """ 百分比止损"""
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
        size = self.broker.getvalue() * self.params.position_size * self.leverage / self.data.close[0]
        if not self.position:
            if self.crossover > 0:
                if self.trade_side == '多头':
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
                elif self.trade_side == '空头':
                    self.sell(size=size)
                    self.store_trade('sell', size, self.data.close[0])
                elif self.trade_side == '多空混合':
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
            elif self.crossover < 0 and self.trade_side == '多空混合':
                self.sell(size=size)
                self.store_trade('sell', size, self.data.close[0])
        else:
            if self.crossover < 0:
                if self.trade_side in ['多头', '空头']:
                    self.close()
                    self.close_trade(self.data.close[0])
                elif self.trade_side == '多空混合':
                    if getattr(self.params, 'trade_mode', '反向开仓') == '反向开仓':
                        self.close()
                        self.close_trade(self.data.close[0])
                        self.sell(size=size)
                        self.store_trade('sell', size, self.data.close[0])
                    else:  # 仅平仓
                        self.close()
                        self.close_trade(self.data.close[0])
            elif self.crossover > 0 and self.trade_side == '多空混合':
                if getattr(self.params, 'trade_mode', '反向开仓') == '反向开仓':
                    self.close()
                    self.close_trade(self.data.close[0])
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
                else:
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
        size = self.broker.getvalue() * self.params.position_size * self.leverage / self.data.close[0]
        if not self.position:
            if self.rsi < self.params.rsi_entry_oversold:
                if self.trade_side == '多头':
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
                elif self.trade_side == '空头':
                    self.sell(size=size)
                    self.store_trade('sell', size, self.data.close[0])
                elif self.trade_side == '多空混合':
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
            elif self.rsi > self.params.rsi_entry_overbought and self.trade_side == '多空混合':
                self.sell(size=size)
                self.store_trade('sell', size, self.data.close[0])
        else:
            if self.rsi > self.params.rsi_entry_overbought:
                if self.trade_side in ['多头', '空头']:
                    self.close()
                    self.close_trade(self.data.close[0])
                elif self.trade_side == '多空混合':
                    if getattr(self.params, 'trade_mode', '反向开仓') == '反向开仓':
                        self.close()
                        self.close_trade(self.data.close[0])
                        self.sell(size=size)
                        self.store_trade('sell', size, self.data.close[0])
                    else:
                        self.close()
                        self.close_trade(self.data.close[0])
            elif self.rsi < self.params.rsi_entry_oversold and self.trade_side == '多空混合':
                if getattr(self.params, 'trade_mode', '反向开仓') == '反向开仓':
                    self.close()
                    self.close_trade(self.data.close[0])
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
                else:
                    self.close()
                    self.close_trade(self.data.close[0])
            elif self.rsi > self.params.rsi_exit_oversold:
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
        size = self.broker.getvalue() * self.params.position_size * self.leverage / self.data.close[0]
        if not self.position:
            if self.macd.macd > self.macd.signal:
                if self.trade_side == '多头':
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
                elif self.trade_side == '空头':
                    self.sell(size=size)
                    self.store_trade('sell', size, self.data.close[0])
                elif self.trade_side == '多空混合':
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
            elif self.macd.macd < self.macd.signal and self.trade_side == '多空混合':
                self.sell(size=size)
                self.store_trade('sell', size, self.data.close[0])
        else:
            if self.macd.macd < self.macd.signal:
                if self.trade_side in ['多头', '空头']:
                    self.close()
                    self.close_trade(self.data.close[0])
                elif self.trade_side == '多空混合':
                    if getattr(self.params, 'trade_mode', '反向开仓') == '反向开仓':
                        self.close()
                        self.close_trade(self.data.close[0])
                        self.sell(size=size)
                        self.store_trade('sell', size, self.data.close[0])
                    else:
                        self.close()
                        self.close_trade(self.data.close[0])
            elif self.macd.macd > self.macd.signal and self.trade_side == '多空混合':
                if getattr(self.params, 'trade_mode', '反向开仓') == '反向开仓':
                    self.close()
                    self.close_trade(self.data.close[0])
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
                else:
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
        size = self.broker.getvalue() * self.params.position_size * self.leverage / self.data.close[0]
        if not self.position:
            if self.data.close[0] > self.trend_ma[0]:
                if self.trade_side == '多头':
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
                elif self.trade_side == '空头':
                    self.sell(size=size)
                    self.store_trade('sell', size, self.data.close[0])
                elif self.trade_side == '多空混合':
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
            elif self.data.close[0] < self.trend_ma[0] and self.trade_side == '多空混合':
                self.sell(size=size)
                self.store_trade('sell', size, self.data.close[0])
        else:
            stop_price = self.get_atr_stop(is_long=(self.trade_side=='多头'))
            if (self.data.close[0] < stop_price and self.trade_side=='多头') or (self.data.close[0] > stop_price and self.trade_side=='空头'):
                self.close()
                self.close_trade(self.data.close[0])
            elif self.data.close[0] < self.trend_ma[0] and self.trade_side == '多空混合':
                if getattr(self.params, 'trade_mode', '反向开仓') == '反向开仓':
                    self.close()
                    self.close_trade(self.data.close[0])
                    self.sell(size=size)
                    self.store_trade('sell', size, self.data.close[0])
                else:
                    self.close()
                    self.close_trade(self.data.close[0])
            elif self.data.close[0] > self.trend_ma[0] and self.trade_side == '多空混合':
                if getattr(self.params, 'trade_mode', '反向开仓') == '反向开仓':
                    self.close()
                    self.close_trade(self.data.close[0])
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
                else:
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
        size = self.broker.getvalue() * self.params.position_size * self.leverage / self.data.close[0]
        if not self.position:
            if (self.data.volume[0] > self.volume_ma[0] * self.params.volume_threshold and
                self.data.close[0] > self.price_ma[0] * (1 + self.params.price_threshold)):
                if self.trade_side == '多头':
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
                elif self.trade_side == '空头':
                    self.sell(size=size)
                    self.store_trade('sell', size, self.data.close[0])
                elif self.trade_side == '多空混合':
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
            elif (self.data.volume[0] > self.volume_ma[0] * self.params.volume_threshold and
                  self.data.close[0] < self.price_ma[0] * (1 - self.params.price_threshold) and self.trade_side == '多空混合'):
                self.sell(size=size)
                self.store_trade('sell', size, self.data.close[0])
        else:
            if (self.data.volume[0] > self.volume_ma[0] * self.params.volume_threshold and
                self.data.close[0] < self.price_ma[0] * (1 - self.params.price_threshold)):
                if self.trade_side in ['多头', '空头']:
                    self.close()
                    self.close_trade(self.data.close[0])
                elif self.trade_side == '多空混合':
                    if getattr(self.params, 'trade_mode', '反向开仓') == '反向开仓':
                        self.close()
                        self.close_trade(self.data.close[0])
                        self.sell(size=size)
                        self.store_trade('sell', size, self.data.close[0])
                    else:
                        self.close()
                        self.close_trade(self.data.close[0])
            elif (self.data.volume[0] > self.volume_ma[0] * self.params.volume_threshold and
                  self.data.close[0] > self.price_ma[0] * (1 + self.params.price_threshold) and self.trade_side == '多空混合'):
                if getattr(self.params, 'trade_mode', '反向开仓') == '反向开仓':
                    self.close()
                    self.close_trade(self.data.close[0])
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
                else:
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
        size = self.broker.getvalue() * self.params.position_size * self.leverage / self.data.close[0]
        if not self.position:
            if (self.data.volume[0] > self.volume_ma[0] * self.params.volume_threshold and
                self.data.close[0] > self.price_ma[0] * (1 + self.params.price_threshold)):
                if self.trade_side == '多头':
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
                elif self.trade_side == '空头':
                    self.sell(size=size)
                    self.store_trade('sell', size, self.data.close[0])
                elif self.trade_side == '多空混合':
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
            elif (self.data.volume[0] > self.volume_ma[0] * self.params.volume_threshold and
                  self.data.close[0] < self.price_ma[0] * (1 - self.params.price_threshold) and self.trade_side == '多空混合'):
                self.sell(size=size)
                self.store_trade('sell', size, self.data.close[0])
        else:
            if (self.data.volume[0] > self.volume_ma[0] * self.params.volume_threshold and
                self.data.close[0] < self.price_ma[0] * (1 - self.params.price_threshold)):
                if self.trade_side in ['多头', '空头']:
                    self.close()
                    self.close_trade(self.data.close[0])
                elif self.trade_side == '多空混合':
                    if getattr(self.params, 'trade_mode', '反向开仓') == '反向开仓':
                        self.close()
                        self.close_trade(self.data.close[0])
                        self.sell(size=size)
                        self.store_trade('sell', size, self.data.close[0])
                    else:
                        self.close()
                        self.close_trade(self.data.close[0])
            elif (self.data.volume[0] > self.volume_ma[0] * self.params.volume_threshold and
                  self.data.close[0] > self.price_ma[0] * (1 + self.params.price_threshold) and self.trade_side == '多空混合'):
                if getattr(self.params, 'trade_mode', '反向开仓') == '反向开仓':
                    self.close()
                    self.close_trade(self.data.close[0])
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
                else:
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
        size = self.broker.getvalue() * self.params.position_size * self.leverage / self.data.close[0]
        score = self.calculate_score()
        if not self.position:
            if score > self.params.score_threshold:
                if self.trade_side == '多头':
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
                elif self.trade_side == '空头':
                    self.sell(size=size)
                    self.store_trade('sell', size, self.data.close[0])
                elif self.trade_side == '多空混合':
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
            elif score < -self.params.score_threshold and self.trade_side == '多空混合':
                self.sell(size=size)
                self.store_trade('sell', size, self.data.close[0])
        else:
            if score < -self.params.score_threshold:
                if self.trade_side in ['多头', '空头']:
                    self.close()
                    self.close_trade(self.data.close[0])
                elif self.trade_side == '多空混合':
                    if getattr(self.params, 'trade_mode', '反向开仓') == '反向开仓':
                        self.close()
                        self.close_trade(self.data.close[0])
                        self.sell(size=size)
                        self.store_trade('sell', size, self.data.close[0])
                    else:
                        self.close()
                        self.close_trade(self.data.close[0])
            elif score > self.params.score_threshold and self.trade_side == '多空混合':
                if getattr(self.params, 'trade_mode', '反向开仓') == '反向开仓':
                    self.close()
                    self.close_trade(self.data.close[0])
                    self.buy(size=size)
                    self.store_trade('buy', size, self.data.close[0])
                else:
                    self.close()
                    self.close_trade(self.data.close[0])