"""
Trend Following Strategies.
Includes EMA Crossover and MACD strategies.
"""
import pandas as pd
import ta
from typing import Dict, Any, Optional

from .base_strategy import BaseStrategy, Signal


class EMAStrategy(BaseStrategy):
    """
    EMA (Exponential Moving Average) Crossover Strategy.
    
    Generates buy signals when fast EMA crosses above slow EMA.
    Generates sell signals when fast EMA crosses below slow EMA.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize EMA strategy.
        
        Args:
            params: Strategy parameters
                - fast_period: Fast EMA period (default: 9)
                - slow_period: Slow EMA period (default: 21)
        """
        default_params = {
            'fast_period': 9,
            'slow_period': 21
        }
        params = {**default_params, **(params or {})}
        super().__init__("ema_crossover", params)
        
        self.fast_period = params['fast_period']
        self.slow_period = params['slow_period']
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA indicators."""
        df = df.copy()
        
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=self.fast_period)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=self.slow_period)
        
        # Previous values for crossover detection
        df['ema_fast_prev'] = df['ema_fast'].shift(1)
        df['ema_slow_prev'] = df['ema_slow'].shift(1)
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Generate signal based on EMA crossover."""
        if not self.validate_data(df, min_rows=self.slow_period + 5):
            return Signal.HOLD
        
        df = self.calculate_indicators(df)
        
        # Get latest values
        ema_fast = df['ema_fast'].iloc[-1]
        ema_slow = df['ema_slow'].iloc[-1]
        ema_fast_prev = df['ema_fast_prev'].iloc[-1]
        ema_slow_prev = df['ema_slow_prev'].iloc[-1]
        
        # Check for crossover
        if ema_fast_prev <= ema_slow_prev and ema_fast > ema_slow:
            self._logger.info(f"BUY signal: EMA({self.fast_period}) crossed above EMA({self.slow_period})")
            return Signal.BUY
        
        elif ema_fast_prev >= ema_slow_prev and ema_fast < ema_slow:
            self._logger.info(f"SELL signal: EMA({self.fast_period}) crossed below EMA({self.slow_period})")
            return Signal.SELL
        
        return Signal.HOLD
    
    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """Calculate signal strength based on EMA distance."""
        df = self.calculate_indicators(df)
        
        ema_fast = df['ema_fast'].iloc[-1]
        ema_slow = df['ema_slow'].iloc[-1]
        
        # Normalized distance
        distance = abs(ema_fast - ema_slow) / ema_slow
        
        return min(distance * 10, 1.0)


class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy.
    
    Generates buy signals when MACD line crosses above signal line.
    Generates sell signals when MACD line crosses below signal line.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize MACD strategy.
        
        Args:
            params: Strategy parameters
                - fast_period: Fast EMA period (default: 12)
                - slow_period: Slow EMA period (default: 26)
                - signal_period: Signal line period (default: 9)
        """
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
        params = {**default_params, **(params or {})}
        super().__init__("macd", params)
        
        self.fast_period = params['fast_period']
        self.slow_period = params['slow_period']
        self.signal_period = params['signal_period']
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators."""
        df = df.copy()
        
        macd_indicator = ta.trend.MACD(
            close=df['close'],
            window_fast=self.fast_period,
            window_slow=self.slow_period,
            window_sign=self.signal_period
        )
        
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()
        
        # Previous values
        df['macd_prev'] = df['macd'].shift(1)
        df['macd_signal_prev'] = df['macd_signal'].shift(1)
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Generate signal based on MACD crossover."""
        min_rows = self.slow_period + self.signal_period + 5
        if not self.validate_data(df, min_rows=min_rows):
            return Signal.HOLD
        
        df = self.calculate_indicators(df)
        
        if 'macd' not in df.columns:
            return Signal.HOLD
        
        # Get latest values
        macd = df['macd'].iloc[-1]
        signal = df['macd_signal'].iloc[-1]
        macd_prev = df['macd_prev'].iloc[-1]
        signal_prev = df['macd_signal_prev'].iloc[-1]
        hist = df['macd_hist'].iloc[-1]
        
        # Check for crossover
        if macd_prev <= signal_prev and macd > signal:
            # Bullish crossover
            if hist > 0:
                self._logger.info("STRONG BUY signal: MACD bullish crossover above zero")
                return Signal.STRONG_BUY
            self._logger.info("BUY signal: MACD bullish crossover")
            return Signal.BUY
        
        elif macd_prev >= signal_prev and macd < signal:
            # Bearish crossover
            if hist < 0:
                self._logger.info("STRONG SELL signal: MACD bearish crossover below zero")
                return Signal.STRONG_SELL
            self._logger.info("SELL signal: MACD bearish crossover")
            return Signal.SELL
        
        return Signal.HOLD
    
    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """Calculate signal strength based on MACD histogram."""
        df = self.calculate_indicators(df)
        
        if 'macd_hist' not in df.columns:
            return 0.5
        
        hist = df['macd_hist'].iloc[-1]
        hist_max = df['macd_hist'].abs().max()
        
        if hist_max == 0:
            return 0.5
        
        return min(abs(hist) / hist_max, 1.0)
    
    def is_bullish_divergence(self, df: pd.DataFrame) -> bool:
        """Check for bullish divergence."""
        df = self.calculate_indicators(df)
        
        if 'macd_hist' not in df.columns:
            return False
        
        # Simple check: price making lower lows but MACD making higher lows
        price_ll = df['close'].iloc[-1] < df['close'].iloc[-20:-1].min()
        macd_hl = df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-20:-1].min()
        
        return price_ll and macd_hl
    
    def is_bearish_divergence(self, df: pd.DataFrame) -> bool:
        """Check for bearish divergence."""
        df = self.calculate_indicators(df)
        
        if 'macd_hist' not in df.columns:
            return False
        
        # Simple check: price making higher highs but MACD making lower highs
        price_hh = df['close'].iloc[-1] > df['close'].iloc[-20:-1].max()
        macd_lh = df['macd_hist'].iloc[-1] < df['macd_hist'].iloc[-20:-1].max()
        
        return price_hh and macd_lh


class TrendStrengthStrategy(BaseStrategy):
    """
    Trend strength strategy using ADX.
    Only trades in strong trends.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize trend strength strategy.
        
        Args:
            params: Strategy parameters
                - adx_period: ADX period (default: 14)
                - adx_threshold: Minimum ADX for trend (default: 25)
                - ema_period: EMA period for direction (default: 20)
        """
        default_params = {
            'adx_period': 14,
            'adx_threshold': 25,
            'ema_period': 20
        }
        params = {**default_params, **(params or {})}
        super().__init__("trend_strength", params)
        
        self.adx_period = params['adx_period']
        self.adx_threshold = params['adx_threshold']
        self.ema_period = params['ema_period']
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX and trend indicators."""
        df = df.copy()
        
        # ADX
        adx_indicator = ta.trend.ADXIndicator(
            high=df['high'], 
            low=df['low'], 
            close=df['close'], 
            window=self.adx_period
        )
        df['adx'] = adx_indicator.adx()
        df['di_plus'] = adx_indicator.adx_pos()
        df['di_minus'] = adx_indicator.adx_neg()
        
        # EMA for trend direction
        df['ema'] = ta.trend.ema_indicator(df['close'], window=self.ema_period)
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Generate signal based on trend strength."""
        if not self.validate_data(df, min_rows=self.adx_period + 10):
            return Signal.HOLD
        
        df = self.calculate_indicators(df)
        
        if 'adx' not in df.columns:
            return Signal.HOLD
        
        adx = df['adx'].iloc[-1]
        di_plus = df['di_plus'].iloc[-1]
        di_minus = df['di_minus'].iloc[-1]
        close = df['close'].iloc[-1]
        ema = df['ema'].iloc[-1]
        
        # Only trade in strong trends
        if adx < self.adx_threshold:
            return Signal.HOLD
        
        # Bullish trend
        if di_plus > di_minus and close > ema:
            self._logger.info(f"BUY signal: Strong uptrend (ADX={adx:.1f})")
            return Signal.BUY
        
        # Bearish trend
        elif di_minus > di_plus and close < ema:
            self._logger.info(f"SELL signal: Strong downtrend (ADX={adx:.1f})")
            return Signal.SELL
        
        return Signal.HOLD

