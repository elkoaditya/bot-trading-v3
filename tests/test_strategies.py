"""
Unit tests for trading strategies.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.base_strategy import Signal, MultiStrategyDecider
from src.strategies.trend_following import EMAStrategy, MACDStrategy
from src.strategies.mean_reversion import RSIStrategy, BollingerStrategy
from src.strategies.breakout import BreakoutStrategy


def generate_sample_data(n_rows: int = 200, trend: str = 'up') -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Base price
    if trend == 'up':
        base = 100 + np.cumsum(np.random.randn(n_rows) * 0.5 + 0.1)
    elif trend == 'down':
        base = 100 + np.cumsum(np.random.randn(n_rows) * 0.5 - 0.1)
    else:
        base = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    
    # Generate OHLCV
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_rows, freq='5min'),
        'open': base + np.random.randn(n_rows) * 0.5,
        'high': base + abs(np.random.randn(n_rows)) * 1.0,
        'low': base - abs(np.random.randn(n_rows)) * 1.0,
        'close': base + np.random.randn(n_rows) * 0.5,
        'volume': np.random.randint(1000, 10000, n_rows).astype(float)
    })
    
    # Ensure high > low
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df


def generate_oversold_data() -> pd.DataFrame:
    """Generate data with RSI in oversold territory."""
    np.random.seed(42)
    n_rows = 100
    
    # Declining prices to create oversold condition
    base = 100 - np.arange(n_rows) * 0.3 + np.random.randn(n_rows) * 0.2
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_rows, freq='5min'),
        'open': base,
        'high': base + 0.5,
        'low': base - 0.5,
        'close': base,
        'volume': np.random.randint(1000, 10000, n_rows).astype(float)
    })
    
    return df


class TestRSIStrategy:
    """Tests for RSI strategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = RSIStrategy()
        assert strategy.name == "rsi"
        assert strategy.period == 14
        assert strategy.oversold == 30
        assert strategy.overbought == 70
    
    def test_custom_params(self):
        """Test custom parameters."""
        strategy = RSIStrategy(params={'period': 7, 'oversold': 25, 'overbought': 75})
        assert strategy.period == 7
        assert strategy.oversold == 25
        assert strategy.overbought == 75
    
    def test_calculate_indicators(self):
        """Test indicator calculation."""
        strategy = RSIStrategy()
        df = generate_sample_data()
        result = strategy.calculate_indicators(df)
        
        assert 'rsi' in result.columns
        assert 'rsi_prev' in result.columns
        assert not result['rsi'].iloc[-1:].isna().all()
    
    def test_generate_signal(self):
        """Test signal generation."""
        strategy = RSIStrategy()
        df = generate_sample_data()
        signal = strategy.generate_signal(df)
        
        assert isinstance(signal, Signal)
        assert signal in [Signal.BUY, Signal.SELL, Signal.HOLD, Signal.STRONG_BUY, Signal.STRONG_SELL]
    
    def test_validate_data(self):
        """Test data validation."""
        strategy = RSIStrategy()
        
        # Valid data
        df = generate_sample_data()
        assert strategy.validate_data(df) == True
        
        # Too few rows
        df_small = generate_sample_data(10)
        assert strategy.validate_data(df_small) == False
        
        # Empty DataFrame
        df_empty = pd.DataFrame()
        assert strategy.validate_data(df_empty) == False


class TestMACDStrategy:
    """Tests for MACD strategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = MACDStrategy()
        assert strategy.name == "macd"
        assert strategy.fast_period == 12
        assert strategy.slow_period == 26
        assert strategy.signal_period == 9
    
    def test_calculate_indicators(self):
        """Test indicator calculation."""
        strategy = MACDStrategy()
        df = generate_sample_data()
        result = strategy.calculate_indicators(df)
        
        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_hist' in result.columns
    
    def test_generate_signal(self):
        """Test signal generation."""
        strategy = MACDStrategy()
        df = generate_sample_data()
        signal = strategy.generate_signal(df)
        
        assert isinstance(signal, Signal)


class TestEMAStrategy:
    """Tests for EMA strategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = EMAStrategy()
        assert strategy.name == "ema_crossover"
        assert strategy.fast_period == 9
        assert strategy.slow_period == 21
    
    def test_calculate_indicators(self):
        """Test indicator calculation."""
        strategy = EMAStrategy()
        df = generate_sample_data()
        result = strategy.calculate_indicators(df)
        
        assert 'ema_fast' in result.columns
        assert 'ema_slow' in result.columns
    
    def test_generate_signal(self):
        """Test signal generation."""
        strategy = EMAStrategy()
        df = generate_sample_data()
        signal = strategy.generate_signal(df)
        
        assert isinstance(signal, Signal)


class TestBollingerStrategy:
    """Tests for Bollinger Bands strategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = BollingerStrategy()
        assert strategy.name == "bollinger"
        assert strategy.period == 20
        assert strategy.std_dev == 2.0
    
    def test_calculate_indicators(self):
        """Test indicator calculation."""
        strategy = BollingerStrategy()
        df = generate_sample_data()
        result = strategy.calculate_indicators(df)
        
        assert 'bb_lower' in result.columns
        assert 'bb_mid' in result.columns
        assert 'bb_upper' in result.columns
    
    def test_generate_signal(self):
        """Test signal generation."""
        strategy = BollingerStrategy()
        df = generate_sample_data()
        signal = strategy.generate_signal(df)
        
        assert isinstance(signal, Signal)


class TestBreakoutStrategy:
    """Tests for Breakout strategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = BreakoutStrategy()
        assert strategy.name == "breakout"
        assert strategy.lookback == 20
    
    def test_calculate_indicators(self):
        """Test indicator calculation."""
        strategy = BreakoutStrategy()
        df = generate_sample_data()
        result = strategy.calculate_indicators(df)
        
        assert 'resistance' in result.columns
        assert 'support' in result.columns
        assert 'atr' in result.columns
    
    def test_get_levels(self):
        """Test support/resistance levels."""
        strategy = BreakoutStrategy()
        df = generate_sample_data()
        support, resistance = strategy.get_levels(df)
        
        assert isinstance(support, float)
        assert isinstance(resistance, float)
        assert resistance > support


class TestMultiStrategyDecider:
    """Tests for multi-strategy decision making."""
    
    def test_initialization(self):
        """Test decider initialization."""
        strategies = [RSIStrategy(), MACDStrategy(), EMAStrategy()]
        decider = MultiStrategyDecider(strategies, mode="majority")
        
        assert len(decider.strategies) == 3
        assert decider.mode == "majority"
    
    def test_get_combined_signal(self):
        """Test combined signal generation."""
        strategies = [RSIStrategy(), MACDStrategy()]
        decider = MultiStrategyDecider(strategies, mode="majority")
        
        df = generate_sample_data()
        signal = decider.get_combined_signal(df)
        
        assert isinstance(signal, Signal)
    
    def test_get_all_signals(self):
        """Test getting all individual signals."""
        strategies = [RSIStrategy(), MACDStrategy(), EMAStrategy()]
        decider = MultiStrategyDecider(strategies, mode="majority")
        
        df = generate_sample_data()
        signals = decider.get_all_signals(df)
        
        assert len(signals) == 3
        assert "rsi" in signals
        assert "macd" in signals
        assert "ema_crossover" in signals
    
    def test_majority_vote(self):
        """Test majority voting logic."""
        strategies = [RSIStrategy(), MACDStrategy(), EMAStrategy()]
        decider = MultiStrategyDecider(strategies, mode="majority")
        
        # Test with mock signals
        signals = [Signal.BUY, Signal.BUY, Signal.SELL]
        result = decider._majority_vote(signals)
        assert result == Signal.BUY
        
        signals = [Signal.SELL, Signal.SELL, Signal.BUY]
        result = decider._majority_vote(signals)
        assert result == Signal.SELL
        
        signals = [Signal.HOLD, Signal.HOLD, Signal.BUY]
        result = decider._majority_vote(signals)
        assert result == Signal.HOLD
    
    def test_all_agree(self):
        """Test all-agree mode."""
        strategies = [RSIStrategy(), MACDStrategy()]
        decider = MultiStrategyDecider(strategies, mode="all")
        
        signals = [Signal.BUY, Signal.BUY]
        result = decider._all_agree(signals)
        assert result == Signal.BUY
        
        signals = [Signal.BUY, Signal.SELL]
        result = decider._all_agree(signals)
        assert result == Signal.HOLD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
