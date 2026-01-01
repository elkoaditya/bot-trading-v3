"""
Breakout Strategies.
Includes support/resistance and channel breakout strategies.
"""
import pandas as pd
import ta
import numpy as np
from typing import Dict, Any, Optional, Tuple

from .base_strategy import BaseStrategy, Signal


class BreakoutStrategy(BaseStrategy):
    """
    Support/Resistance Breakout Strategy.
    
    Identifies key support and resistance levels and trades breakouts.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize breakout strategy.
        
        Args:
            params: Strategy parameters
                - lookback: Period for S/R calculation (default: 20)
                - atr_period: ATR period for volatility (default: 14)
                - breakout_factor: ATR multiplier for breakout (default: 0.5)
        """
        default_params = {
            'lookback': 20,
            'atr_period': 14,
            'breakout_factor': 0.5
        }
        params = {**default_params, **(params or {})}
        super().__init__("breakout", params)
        
        self.lookback = params['lookback']
        self.atr_period = params['atr_period']
        self.breakout_factor = params['breakout_factor']
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support, resistance, and ATR."""
        df = df.copy()
        
        # ATR for volatility
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=self.atr_period
        )
        df['atr'] = atr_indicator.average_true_range()
        
        # Rolling high and low for S/R
        df['resistance'] = df['high'].rolling(window=self.lookback).max()
        df['support'] = df['low'].rolling(window=self.lookback).min()
        
        # Previous resistance and support
        df['resistance_prev'] = df['resistance'].shift(1)
        df['support_prev'] = df['support'].shift(1)
        
        # Previous close
        df['close_prev'] = df['close'].shift(1)
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Generate signal based on breakout."""
        if not self.validate_data(df, min_rows=self.lookback + 5):
            return Signal.HOLD
        
        df = self.calculate_indicators(df)
        
        close = df['close'].iloc[-1]
        close_prev = df['close_prev'].iloc[-1]
        resistance = df['resistance'].iloc[-2]  # Use previous bar's resistance
        support = df['support'].iloc[-2]  # Use previous bar's support
        atr = df['atr'].iloc[-1]
        
        breakout_threshold = atr * self.breakout_factor
        
        # Resistance breakout
        if close_prev <= resistance and close > resistance + breakout_threshold:
            self._logger.info(f"BUY signal: Resistance breakout at {resistance:.2f}")
            return Signal.STRONG_BUY
        
        # Support breakdown
        if close_prev >= support and close < support - breakout_threshold:
            self._logger.info(f"SELL signal: Support breakdown at {support:.2f}")
            return Signal.STRONG_SELL
        
        # Bounce from support
        if close <= support + breakout_threshold and close > close_prev:
            self._logger.info(f"BUY signal: Bounce from support at {support:.2f}")
            return Signal.BUY
        
        # Rejection from resistance
        if close >= resistance - breakout_threshold and close < close_prev:
            self._logger.info(f"SELL signal: Rejection from resistance at {resistance:.2f}")
            return Signal.SELL
        
        return Signal.HOLD
    
    def get_levels(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Get current support and resistance levels."""
        df = self.calculate_indicators(df)
        return df['support'].iloc[-1], df['resistance'].iloc[-1]
    
    def get_stop_loss(
        self,
        df: pd.DataFrame,
        signal: Signal,
        entry_price: float
    ) -> Optional[float]:
        """Get stop loss based on S/R levels."""
        df = self.calculate_indicators(df)
        atr = df['atr'].iloc[-1]
        
        if signal in [Signal.BUY, Signal.STRONG_BUY]:
            return df['support'].iloc[-1] - atr
        elif signal in [Signal.SELL, Signal.STRONG_SELL]:
            return df['resistance'].iloc[-1] + atr
        
        return None


class DonchianBreakoutStrategy(BaseStrategy):
    """
    Donchian Channel Breakout Strategy.
    
    Based on the Turtle Trading system.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Donchian breakout strategy.
        
        Args:
            params: Strategy parameters
                - entry_period: Period for entry channel (default: 20)
                - exit_period: Period for exit channel (default: 10)
        """
        default_params = {
            'entry_period': 20,
            'exit_period': 10
        }
        params = {**default_params, **(params or {})}
        super().__init__("donchian", params)
        
        self.entry_period = params['entry_period']
        self.exit_period = params['exit_period']
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Donchian channels."""
        df = df.copy()
        
        # Entry channel
        df['dc_upper'] = df['high'].rolling(window=self.entry_period).max()
        df['dc_lower'] = df['low'].rolling(window=self.entry_period).min()
        df['dc_mid'] = (df['dc_upper'] + df['dc_lower']) / 2
        
        # Exit channel (shorter period)
        df['dc_exit_upper'] = df['high'].rolling(window=self.exit_period).max()
        df['dc_exit_lower'] = df['low'].rolling(window=self.exit_period).min()
        
        # Previous values
        df['dc_upper_prev'] = df['dc_upper'].shift(1)
        df['dc_lower_prev'] = df['dc_lower'].shift(1)
        df['close_prev'] = df['close'].shift(1)
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Generate signal based on Donchian channel breakout."""
        if not self.validate_data(df, min_rows=self.entry_period + 5):
            return Signal.HOLD
        
        df = self.calculate_indicators(df)
        
        close = df['close'].iloc[-1]
        close_prev = df['close_prev'].iloc[-1]
        dc_upper = df['dc_upper'].iloc[-2]  # Previous bar's channel
        dc_lower = df['dc_lower'].iloc[-2]
        
        # Breakout above 20-day high
        if close > dc_upper and close_prev <= dc_upper:
            self._logger.info(f"BUY signal: Donchian breakout above {dc_upper:.2f}")
            return Signal.STRONG_BUY
        
        # Breakout below 20-day low
        if close < dc_lower and close_prev >= dc_lower:
            self._logger.info(f"SELL signal: Donchian breakout below {dc_lower:.2f}")
            return Signal.STRONG_SELL
        
        return Signal.HOLD
    
    def should_exit_long(self, df: pd.DataFrame, entry_price: float) -> bool:
        """Check if should exit long position."""
        df = self.calculate_indicators(df)
        close = df['close'].iloc[-1]
        exit_lower = df['dc_exit_lower'].iloc[-1]
        
        return close < exit_lower
    
    def should_exit_short(self, df: pd.DataFrame, entry_price: float) -> bool:
        """Check if should exit short position."""
        df = self.calculate_indicators(df)
        close = df['close'].iloc[-1]
        exit_upper = df['dc_exit_upper'].iloc[-1]
        
        return close > exit_upper


class VolumeBreakoutStrategy(BaseStrategy):
    """
    Volume Breakout Strategy.
    
    Combines price breakout with volume confirmation.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize volume breakout strategy.
        
        Args:
            params: Strategy parameters
                - price_lookback: Period for price levels (default: 20)
                - volume_lookback: Period for volume average (default: 20)
                - volume_multiplier: Volume threshold multiplier (default: 1.5)
        """
        default_params = {
            'price_lookback': 20,
            'volume_lookback': 20,
            'volume_multiplier': 1.5
        }
        params = {**default_params, **(params or {})}
        super().__init__("volume_breakout", params)
        
        self.price_lookback = params['price_lookback']
        self.volume_lookback = params['volume_lookback']
        self.volume_multiplier = params['volume_multiplier']
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price levels and volume indicators."""
        df = df.copy()
        
        # Price levels
        df['resistance'] = df['high'].rolling(window=self.price_lookback).max()
        df['support'] = df['low'].rolling(window=self.price_lookback).min()
        
        # Volume average
        df['volume_avg'] = df['volume'].rolling(window=self.volume_lookback).mean()
        df['volume_ratio'] = df['volume'] / df['volume_avg']
        
        # Previous values
        df['close_prev'] = df['close'].shift(1)
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Generate signal based on volume-confirmed breakout."""
        max_lookback = max(self.price_lookback, self.volume_lookback)
        if not self.validate_data(df, min_rows=max_lookback + 5):
            return Signal.HOLD
        
        df = self.calculate_indicators(df)
        
        close = df['close'].iloc[-1]
        close_prev = df['close_prev'].iloc[-1]
        resistance = df['resistance'].iloc[-2]
        support = df['support'].iloc[-2]
        volume_ratio = df['volume_ratio'].iloc[-1]
        
        # High volume breakout above resistance
        if close > resistance and close_prev <= resistance:
            if volume_ratio >= self.volume_multiplier:
                self._logger.info(f"STRONG BUY signal: Volume breakout (vol_ratio={volume_ratio:.1f}x)")
                return Signal.STRONG_BUY
            else:
                self._logger.info(f"BUY signal: Breakout with normal volume")
                return Signal.BUY
        
        # High volume breakdown below support
        if close < support and close_prev >= support:
            if volume_ratio >= self.volume_multiplier:
                self._logger.info(f"STRONG SELL signal: Volume breakdown (vol_ratio={volume_ratio:.1f}x)")
                return Signal.STRONG_SELL
            else:
                self._logger.info(f"SELL signal: Breakdown with normal volume")
                return Signal.SELL
        
        return Signal.HOLD
    
    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """Calculate signal strength based on volume ratio."""
        df = self.calculate_indicators(df)
        
        volume_ratio = df['volume_ratio'].iloc[-1]
        
        # Higher volume = stronger signal
        if volume_ratio >= 2.0:
            return 1.0
        elif volume_ratio >= 1.5:
            return 0.8
        elif volume_ratio >= 1.0:
            return 0.5
        
        return 0.3


class ATRBreakoutStrategy(BaseStrategy):
    """
    ATR-based Breakout Strategy.
    
    Uses ATR to define breakout thresholds.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize ATR breakout strategy.
        
        Args:
            params: Strategy parameters
                - atr_period: ATR period (default: 14)
                - atr_multiplier: ATR multiplier for breakout (default: 2.0)
                - ema_period: EMA period for trend direction (default: 50)
        """
        default_params = {
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'ema_period': 50
        }
        params = {**default_params, **(params or {})}
        super().__init__("atr_breakout", params)
        
        self.atr_period = params['atr_period']
        self.atr_multiplier = params['atr_multiplier']
        self.ema_period = params['ema_period']
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR-based levels."""
        df = df.copy()
        
        # ATR
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=self.atr_period
        )
        df['atr'] = atr_indicator.average_true_range()
        
        # EMA for trend
        df['ema'] = ta.trend.ema_indicator(df['close'], window=self.ema_period)
        
        # ATR channels
        df['atr_upper'] = df['ema'] + (df['atr'] * self.atr_multiplier)
        df['atr_lower'] = df['ema'] - (df['atr'] * self.atr_multiplier)
        
        # Previous values
        df['close_prev'] = df['close'].shift(1)
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Generate signal based on ATR breakout."""
        max_period = max(self.atr_period, self.ema_period)
        if not self.validate_data(df, min_rows=max_period + 5):
            return Signal.HOLD
        
        df = self.calculate_indicators(df)
        
        close = df['close'].iloc[-1]
        close_prev = df['close_prev'].iloc[-1]
        atr_upper = df['atr_upper'].iloc[-1]
        atr_lower = df['atr_lower'].iloc[-1]
        ema = df['ema'].iloc[-1]
        
        # Breakout above ATR upper band
        if close_prev <= atr_upper and close > atr_upper:
            self._logger.info(f"BUY signal: ATR breakout above {atr_upper:.2f}")
            return Signal.STRONG_BUY
        
        # Breakout below ATR lower band
        if close_prev >= atr_lower and close < atr_lower:
            self._logger.info(f"SELL signal: ATR breakout below {atr_lower:.2f}")
            return Signal.STRONG_SELL
        
        # Trend continuation
        if close > ema and close > close_prev:
            return Signal.BUY
        elif close < ema and close < close_prev:
            return Signal.SELL
        
        return Signal.HOLD

