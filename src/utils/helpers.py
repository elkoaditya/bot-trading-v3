"""
Helper utilities for the trading bot.
Contains retry mechanism, position sizing, and other common functions.
"""
import asyncio
import functools
import time
from typing import TypeVar, Callable, Any, Optional
from decimal import Decimal, ROUND_DOWN


T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        await asyncio.sleep(delay)
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        time.sleep(delay)
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def calculate_position_size(
    balance: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
    leverage: int = 1,
    max_position_pct: float = 0.1
) -> float:
    """
    Calculate position size based on risk management rules.
    
    Args:
        balance: Account balance in quote currency
        risk_per_trade: Maximum risk per trade as decimal (e.g., 0.02 for 2%)
        entry_price: Entry price for the trade
        stop_loss_price: Stop loss price
        leverage: Trading leverage
        max_position_pct: Maximum position size as percentage of balance
        
    Returns:
        Position size in base currency
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0.0
    
    # Calculate risk per unit
    risk_per_unit = abs(entry_price - stop_loss_price)
    
    if risk_per_unit == 0:
        return 0.0
    
    # Maximum amount willing to risk
    max_risk_amount = balance * risk_per_trade
    
    # Position size based on risk
    position_size = (max_risk_amount / risk_per_unit) * leverage
    
    # Apply maximum position limit
    max_position_value = balance * max_position_pct * leverage
    max_position_size = max_position_value / entry_price
    
    return min(position_size, max_position_size)


def round_to_precision(value: float, precision: int = 8) -> float:
    """
    Round a value to specified decimal precision.
    
    Args:
        value: Value to round
        precision: Number of decimal places
        
    Returns:
        Rounded value
    """
    decimal_value = Decimal(str(value))
    precision_str = '0.' + '0' * precision
    return float(decimal_value.quantize(Decimal(precision_str), rounding=ROUND_DOWN))


def round_to_tick_size(value: float, tick_size: float) -> float:
    """
    Round a value to the nearest tick size.
    
    Args:
        value: Value to round
        tick_size: Minimum price increment
        
    Returns:
        Rounded value
    """
    if tick_size <= 0:
        return value
    
    decimal_value = Decimal(str(value))
    decimal_tick = Decimal(str(tick_size))
    
    return float((decimal_value / decimal_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * decimal_tick)


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    quantity: float,
    side: str,
    leverage: int = 1,
    fee_rate: float = 0.001
) -> dict:
    """
    Calculate profit and loss for a trade.
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        quantity: Position quantity
        side: 'long' or 'short'
        leverage: Trading leverage
        fee_rate: Trading fee rate
        
    Returns:
        Dictionary with PnL details
    """
    notional_value = quantity * entry_price
    
    if side.lower() == 'long':
        price_change = exit_price - entry_price
    else:
        price_change = entry_price - exit_price
    
    gross_pnl = price_change * quantity * leverage
    
    # Calculate fees
    entry_fee = notional_value * fee_rate
    exit_fee = quantity * exit_price * fee_rate
    total_fees = entry_fee + exit_fee
    
    net_pnl = gross_pnl - total_fees
    
    # Calculate percentage return
    margin = notional_value / leverage
    pnl_pct = (net_pnl / margin) * 100 if margin > 0 else 0
    
    return {
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'total_fees': total_fees,
        'pnl_pct': pnl_pct,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'quantity': quantity,
        'leverage': leverage
    }


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format a number with thousands separator and decimal places.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{value:,.{decimals}f}"


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert timeframe string to seconds.
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '4h', '1d')
        
    Returns:
        Number of seconds
    """
    multipliers = {
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800
    }
    
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    
    return value * multipliers.get(unit, 60)


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Convert timeframe string to minutes.
    
    Args:
        timeframe: Timeframe string
        
    Returns:
        Number of minutes
    """
    return timeframe_to_seconds(timeframe) // 60


class RateLimiter:
    """
    Simple rate limiter for API calls.
    """
    
    def __init__(self, max_calls: int, time_window: float):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: list = []
    
    def can_proceed(self) -> bool:
        """Check if a call can proceed."""
        now = time.time()
        
        # Remove old calls
        self.calls = [t for t in self.calls if now - t < self.time_window]
        
        return len(self.calls) < self.max_calls
    
    def record_call(self):
        """Record a call."""
        self.calls.append(time.time())
    
    async def wait_if_needed(self):
        """Wait if rate limit is reached."""
        while not self.can_proceed():
            await asyncio.sleep(0.1)
        self.record_call()
