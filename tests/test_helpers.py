"""
Unit tests for helper utilities.
"""
import pytest
import asyncio
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import (
    retry_with_backoff,
    calculate_position_size,
    round_to_precision,
    round_to_tick_size,
    calculate_pnl,
    format_number,
    timeframe_to_seconds,
    timeframe_to_minutes,
    RateLimiter
)


class TestRetryWithBackoff:
    """Tests for retry decorator."""
    
    def test_successful_call(self):
        """Test successful function call."""
        @retry_with_backoff(max_retries=3)
        def success_func():
            return "success"
        
        result = success_func()
        assert result == "success"
    
    def test_retry_on_failure(self):
        """Test retry on failure."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.1)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = fail_then_succeed()
        assert result == "success"
        assert call_count == 3
    
    def test_max_retries_exceeded(self):
        """Test max retries exceeded."""
        @retry_with_backoff(max_retries=2, base_delay=0.1)
        def always_fail():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fail()


class TestPositionSizing:
    """Tests for position sizing."""
    
    def test_basic_position_size(self):
        """Test basic position size calculation."""
        size = calculate_position_size(
            balance=10000,
            risk_per_trade=0.02,
            entry_price=100,
            stop_loss_price=98,
            leverage=1
        )
        
        # Risk $200 (2% of 10000), stop loss is $2 per unit
        # Position size should be 100 units max, limited by max_position_pct
        assert size > 0
        assert size <= 10  # max_position_pct=0.1, so max $1000/100 = 10 units
    
    def test_leverage_position_size(self):
        """Test position size with leverage."""
        size_no_leverage = calculate_position_size(
            balance=10000,
            risk_per_trade=0.02,
            entry_price=100,
            stop_loss_price=98,
            leverage=1
        )
        
        size_with_leverage = calculate_position_size(
            balance=10000,
            risk_per_trade=0.02,
            entry_price=100,
            stop_loss_price=98,
            leverage=10
        )
        
        # With leverage, position size should be larger
        assert size_with_leverage > size_no_leverage
    
    def test_zero_risk(self):
        """Test with zero risk per unit."""
        size = calculate_position_size(
            balance=10000,
            risk_per_trade=0.02,
            entry_price=100,
            stop_loss_price=100,  # Same as entry
            leverage=1
        )
        
        assert size == 0.0


class TestRounding:
    """Tests for rounding functions."""
    
    def test_round_to_precision(self):
        """Test precision rounding."""
        assert round_to_precision(1.23456789, 2) == 1.23
        assert round_to_precision(1.23456789, 4) == 1.2345
        assert round_to_precision(1.99999, 2) == 1.99
    
    def test_round_to_tick_size(self):
        """Test tick size rounding."""
        assert round_to_tick_size(1.234, 0.1) == 1.2
        assert round_to_tick_size(1.256, 0.1) == 1.2
        assert round_to_tick_size(100.5, 0.5) == 100.5
        assert round_to_tick_size(100.7, 0.5) == 100.5


class TestPnLCalculation:
    """Tests for PnL calculation."""
    
    def test_long_profit(self):
        """Test profitable long trade."""
        result = calculate_pnl(
            entry_price=100,
            exit_price=110,
            quantity=1,
            side='long',
            leverage=1,
            fee_rate=0.001
        )
        
        assert result['gross_pnl'] == 10.0
        assert result['net_pnl'] < 10.0  # After fees
        assert result['pnl_pct'] > 0
    
    def test_long_loss(self):
        """Test losing long trade."""
        result = calculate_pnl(
            entry_price=100,
            exit_price=90,
            quantity=1,
            side='long',
            leverage=1,
            fee_rate=0.001
        )
        
        assert result['gross_pnl'] == -10.0
        assert result['net_pnl'] < -10.0  # After fees
        assert result['pnl_pct'] < 0
    
    def test_short_profit(self):
        """Test profitable short trade."""
        result = calculate_pnl(
            entry_price=100,
            exit_price=90,
            quantity=1,
            side='short',
            leverage=1,
            fee_rate=0.001
        )
        
        assert result['gross_pnl'] == 10.0
        assert result['pnl_pct'] > 0
    
    def test_leverage_effect(self):
        """Test leverage effect on PnL."""
        result_1x = calculate_pnl(
            entry_price=100,
            exit_price=110,
            quantity=1,
            side='long',
            leverage=1,
            fee_rate=0.001
        )
        
        result_10x = calculate_pnl(
            entry_price=100,
            exit_price=110,
            quantity=1,
            side='long',
            leverage=10,
            fee_rate=0.001
        )
        
        # 10x leverage should have 10x gross PnL
        assert result_10x['gross_pnl'] == result_1x['gross_pnl'] * 10


class TestTimeframeConversion:
    """Tests for timeframe conversion."""
    
    def test_to_seconds(self):
        """Test conversion to seconds."""
        assert timeframe_to_seconds('1m') == 60
        assert timeframe_to_seconds('5m') == 300
        assert timeframe_to_seconds('1h') == 3600
        assert timeframe_to_seconds('4h') == 14400
        assert timeframe_to_seconds('1d') == 86400
    
    def test_to_minutes(self):
        """Test conversion to minutes."""
        assert timeframe_to_minutes('1m') == 1
        assert timeframe_to_minutes('5m') == 5
        assert timeframe_to_minutes('1h') == 60
        assert timeframe_to_minutes('4h') == 240


class TestRateLimiter:
    """Tests for rate limiter."""
    
    def test_can_proceed(self):
        """Test rate limiting."""
        limiter = RateLimiter(max_calls=3, time_window=1.0)
        
        # First 3 calls should proceed
        assert limiter.can_proceed() == True
        limiter.record_call()
        
        assert limiter.can_proceed() == True
        limiter.record_call()
        
        assert limiter.can_proceed() == True
        limiter.record_call()
        
        # 4th call should be blocked
        assert limiter.can_proceed() == False
    
    def test_window_reset(self):
        """Test rate limit window reset."""
        limiter = RateLimiter(max_calls=2, time_window=0.1)
        
        limiter.record_call()
        limiter.record_call()
        assert limiter.can_proceed() == False
        
        # Wait for window to reset
        time.sleep(0.15)
        assert limiter.can_proceed() == True


class TestFormatNumber:
    """Tests for number formatting."""
    
    def test_format_with_decimals(self):
        """Test formatting with decimals."""
        assert format_number(1234.567, 2) == "1,234.57"
        assert format_number(1000000, 0) == "1,000,000"
        assert format_number(0.12345, 4) == "0.1235"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
