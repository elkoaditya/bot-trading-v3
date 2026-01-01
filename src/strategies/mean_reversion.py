"""
Mean Reversion Strategies.
Includes RSI and Bollinger Bands strategies.
"""
import pandas as pd
import ta
from typing import Dict, Any, Optional

from .base_strategy import BaseStrategy, Signal


class RSIStrategy(BaseStrategy):
    """
    RSI (Relative Strength Index) Strategy.
    
    Generates buy signals when RSI crosses above oversold level.
    Generates sell signals when RSI crosses below overbought level.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize RSI strategy.
        
        Args:
            params: Strategy parameters
                - period: RSI period (default: 14)
                - oversold: Oversold level (default: 30)
                - overbought: Overbought level (default: 70)
        """
        default_params = {
            'period': 14,
            'oversold': 30,
            'overbought': 70
        }
        params = {**default_params, **(params or {})}
        super().__init__("rsi", params)
        
        self.period = params['period']
        self.oversold = params['oversold']
        self.overbought = params['overbought']
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator."""
        df = df.copy()
        
        df['rsi'] = ta.momentum.rsi(df['close'], window=self.period)
        df['rsi_prev'] = df['rsi'].shift(1)
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Generate signal based on RSI levels."""
        if not self.validate_data(df, min_rows=self.period + 5):
            return Signal.HOLD
        
        df = self.calculate_indicators(df)
        
        rsi = df['rsi'].iloc[-1]
        rsi_prev = df['rsi_prev'].iloc[-1]
        
        # Check for oversold bounce
        if rsi_prev <= self.oversold and rsi > self.oversold:
            self._logger.info(f"BUY signal: RSI({self.period}) crossed above {self.oversold} (RSI={rsi:.1f})")
            return Signal.BUY
        
        # Check for overbought pullback
        elif rsi_prev >= self.overbought and rsi < self.overbought:
            self._logger.info(f"SELL signal: RSI({self.period}) crossed below {self.overbought} (RSI={rsi:.1f})")
            return Signal.SELL
        
        # Strong oversold
        if rsi < 20:
            return Signal.STRONG_BUY
        
        # Strong overbought
        if rsi > 80:
            return Signal.STRONG_SELL
        
        return Signal.HOLD
    
    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """Calculate signal strength based on RSI extremity."""
        df = self.calculate_indicators(df)
        
        rsi = df['rsi'].iloc[-1]
        
        # More extreme = stronger signal
        if rsi <= self.oversold:
            return min((self.oversold - rsi) / self.oversold + 0.5, 1.0)
        elif rsi >= self.overbought:
            return min((rsi - self.overbought) / (100 - self.overbought) + 0.5, 1.0)
        
        return 0.3
    
    def is_divergence(self, df: pd.DataFrame) -> Optional[str]:
        """
        Check for RSI divergence.
        
        Returns:
            'bullish', 'bearish', or None
        """
        df = self.calculate_indicators(df)
        
        lookback = 20
        
        # Get price and RSI swings
        recent_price = df['close'].iloc[-lookback:]
        recent_rsi = df['rsi'].iloc[-lookback:]
        
        # Bullish divergence: price lower lows, RSI higher lows
        price_ll = recent_price.iloc[-1] < recent_price.min()
        rsi_hl = recent_rsi.iloc[-1] > recent_rsi.min()
        
        if price_ll and rsi_hl:
            return 'bullish'
        
        # Bearish divergence: price higher highs, RSI lower highs
        price_hh = recent_price.iloc[-1] > recent_price.max()
        rsi_lh = recent_rsi.iloc[-1] < recent_rsi.max()
        
        if price_hh and rsi_lh:
            return 'bearish'
        
        return None


class BollingerStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy.
    
    Generates buy signals when price bounces from lower band.
    Generates sell signals when price pulls back from upper band.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Bollinger Bands strategy.
        
        Args:
            params: Strategy parameters
                - period: Moving average period (default: 20)
                - std_dev: Standard deviation multiplier (default: 2)
        """
        default_params = {
            'period': 20,
            'std_dev': 2.0
        }
        params = {**default_params, **(params or {})}
        super().__init__("bollinger", params)
        
        self.period = params['period']
        self.std_dev = params['std_dev']
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        df = df.copy()
        
        bb_indicator = ta.volatility.BollingerBands(
            close=df['close'], 
            window=self.period, 
            window_dev=self.std_dev
        )
        
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_mid'] = bb_indicator.bollinger_mavg()
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_bandwidth'] = bb_indicator.bollinger_wband()
        df['bb_percent'] = bb_indicator.bollinger_pband()
        
        # Previous values
        df['close_prev'] = df['close'].shift(1)
        df['bb_lower_prev'] = df['bb_lower'].shift(1)
        df['bb_upper_prev'] = df['bb_upper'].shift(1)
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Generate signal based on Bollinger Bands."""
        if not self.validate_data(df, min_rows=self.period + 5):
            return Signal.HOLD
        
        df = self.calculate_indicators(df)
        
        if 'bb_lower' not in df.columns:
            return Signal.HOLD
        
        close = df['close'].iloc[-1]
        close_prev = df['close_prev'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_mid = df['bb_mid'].iloc[-1]
        bb_lower_prev = df['bb_lower_prev'].iloc[-1]
        bb_upper_prev = df['bb_upper_prev'].iloc[-1]
        
        # Bounce from lower band
        if close_prev <= bb_lower_prev and close > bb_lower:
            self._logger.info(f"BUY signal: Price bounced from lower Bollinger Band")
            return Signal.BUY
        
        # Touch lower band (potential buy)
        if close <= bb_lower:
            self._logger.info(f"STRONG BUY signal: Price at lower Bollinger Band")
            return Signal.STRONG_BUY
        
        # Pullback from upper band
        if close_prev >= bb_upper_prev and close < bb_upper:
            self._logger.info(f"SELL signal: Price pulled back from upper Bollinger Band")
            return Signal.SELL
        
        # Touch upper band (potential sell)
        if close >= bb_upper:
            self._logger.info(f"STRONG SELL signal: Price at upper Bollinger Band")
            return Signal.STRONG_SELL
        
        return Signal.HOLD
    
    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """Calculate signal strength based on band position."""
        df = self.calculate_indicators(df)
        
        if 'bb_lower' not in df.columns:
            return 0.5
        
        close = df['close'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        
        band_width = bb_upper - bb_lower
        
        if band_width == 0:
            return 0.5
        
        # Position within bands (0 = lower, 1 = upper)
        position = (close - bb_lower) / band_width
        
        # Stronger signal at extremes
        if position < 0.2 or position > 0.8:
            return 0.8
        elif position < 0.3 or position > 0.7:
            return 0.6
        
        return 0.4
    
    def get_stop_loss(
        self,
        df: pd.DataFrame,
        signal: Signal,
        entry_price: float
    ) -> Optional[float]:
        """Get stop loss at opposite band."""
        df = self.calculate_indicators(df)
        
        if signal in [Signal.BUY, Signal.STRONG_BUY]:
            # Stop below lower band
            return df['bb_lower'].iloc[-1] * 0.99
        elif signal in [Signal.SELL, Signal.STRONG_SELL]:
            # Stop above upper band
            return df['bb_upper'].iloc[-1] * 1.01
        
        return None
    
    def get_take_profit(
        self,
        df: pd.DataFrame,
        signal: Signal,
        entry_price: float
    ) -> Optional[float]:
        """Get take profit at middle or opposite band."""
        df = self.calculate_indicators(df)
        
        if signal in [Signal.BUY, Signal.STRONG_BUY]:
            # Target middle band first, then upper
            return df['bb_mid'].iloc[-1]
        elif signal in [Signal.SELL, Signal.STRONG_SELL]:
            # Target middle band first, then lower
            return df['bb_mid'].iloc[-1]
        
        return None
    
    def is_squeeze(self, df: pd.DataFrame, threshold: float = 0.04) -> bool:
        """
        Check if Bollinger Bands are in a squeeze.
        
        Args:
            df: OHLCV DataFrame
            threshold: Bandwidth threshold (default 4%)
            
        Returns:
            True if in squeeze
        """
        df = self.calculate_indicators(df)
        
        if 'bb_lower' not in df.columns:
            return False
        
        bb_lower = df['bb_lower'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_mid = df['bb_mid'].iloc[-1]
        
        bandwidth = (bb_upper - bb_lower) / bb_mid
        
        return bandwidth < threshold


class StochasticStrategy(BaseStrategy):
    """
    Stochastic Oscillator Strategy.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Stochastic strategy.
        
        Args:
            params: Strategy parameters
                - k_period: %K period (default: 14)
                - d_period: %D period (default: 3)
                - oversold: Oversold level (default: 20)
                - overbought: Overbought level (default: 80)
        """
        default_params = {
            'k_period': 14,
            'd_period': 3,
            'oversold': 20,
            'overbought': 80
        }
        params = {**default_params, **(params or {})}
        super().__init__("stochastic", params)
        
        self.k_period = params['k_period']
        self.d_period = params['d_period']
        self.oversold = params['oversold']
        self.overbought = params['overbought']
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic indicators."""
        df = df.copy()
        
        stoch_indicator = ta.momentum.StochasticOscillator(
            high=df['high'], 
            low=df['low'], 
            close=df['close'],
            window=self.k_period, 
            smooth_window=self.d_period
        )
        
        df['stoch_k'] = stoch_indicator.stoch()
        df['stoch_d'] = stoch_indicator.stoch_signal()
        df['stoch_k_prev'] = df['stoch_k'].shift(1)
        df['stoch_d_prev'] = df['stoch_d'].shift(1)
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Generate signal based on Stochastic."""
        if not self.validate_data(df, min_rows=self.k_period + self.d_period + 5):
            return Signal.HOLD
        
        df = self.calculate_indicators(df)
        
        if 'stoch_k' not in df.columns:
            return Signal.HOLD
        
        k = df['stoch_k'].iloc[-1]
        d = df['stoch_d'].iloc[-1]
        k_prev = df['stoch_k_prev'].iloc[-1]
        d_prev = df['stoch_d_prev'].iloc[-1]
        
        # Bullish crossover in oversold zone
        if k_prev <= d_prev and k > d and k < self.oversold + 10:
            self._logger.info(f"BUY signal: Stochastic bullish crossover (K={k:.1f})")
            return Signal.BUY
        
        # Bearish crossover in overbought zone
        if k_prev >= d_prev and k < d and k > self.overbought - 10:
            self._logger.info(f"SELL signal: Stochastic bearish crossover (K={k:.1f})")
            return Signal.SELL
        
        return Signal.HOLD

