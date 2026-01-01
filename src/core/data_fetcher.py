"""
Data Fetcher for OHLCV data from Bybit.
Includes caching and multi-timeframe support.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union
from datetime import datetime, timedelta
import time
from collections import OrderedDict

from .bybit_client import BybitClient
from ..utils.logger import get_logger
from ..utils.helpers import timeframe_to_minutes


logger = get_logger("data_fetcher")


class DataCache:
    """
    Simple in-memory cache for OHLCV data.
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 60):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
    
    def _make_key(self, symbol: str, timeframe: str) -> str:
        """Create cache key."""
        return f"{symbol}_{timeframe}"
    
    def get(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get cached data if fresh."""
        key = self._make_key(symbol, timeframe)
        
        if key not in self._cache:
            return None
        
        # Check TTL
        if time.time() - self._timestamps.get(key, 0) > self.ttl:
            del self._cache[key]
            del self._timestamps[key]
            return None
        
        # Move to end (LRU)
        self._cache.move_to_end(key)
        
        return self._cache[key].copy()
    
    def set(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Set cached data."""
        key = self._make_key(symbol, timeframe)
        
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            del self._timestamps[oldest]
        
        self._cache[key] = data.copy()
        self._timestamps[key] = time.time()
    
    def invalidate(self, symbol: str, timeframe: str):
        """Invalidate cache entry."""
        key = self._make_key(symbol, timeframe)
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]
    
    def clear(self):
        """Clear all cache."""
        self._cache.clear()
        self._timestamps.clear()


class DataFetcher:
    """
    OHLCV data fetcher from Bybit with caching.
    """
    
    # Bybit interval mapping
    INTERVAL_MAP = {
        '1m': '1',
        '3m': '3',
        '5m': '5',
        '15m': '15',
        '30m': '30',
        '1h': '60',
        '2h': '120',
        '4h': '240',
        '6h': '360',
        '12h': '720',
        '1d': 'D',
        '1w': 'W',
        '1M': 'M'
    }
    
    def __init__(
        self,
        client: Optional[BybitClient] = None,
        cache_ttl: int = 30,
        category: str = "linear"
    ):
        """
        Initialize data fetcher.
        
        Args:
            client: BybitClient instance
            cache_ttl: Cache time-to-live in seconds
            category: Product category (linear, inverse, spot)
        """
        self.client = client or BybitClient()
        self.category = category
        self.cache = DataCache(max_size=200, ttl_seconds=cache_ttl)
        
        logger.info("DataFetcher initialized")
    
    def _convert_interval(self, timeframe: str) -> str:
        """Convert timeframe string to Bybit interval."""
        return self.INTERVAL_MAP.get(timeframe, timeframe)
    
    def _parse_kline_data(self, raw_data: List[List]) -> pd.DataFrame:
        """
        Parse raw kline data into DataFrame.
        
        Args:
            raw_data: Raw kline data from API
            
        Returns:
            DataFrame with OHLCV columns
        """
        if not raw_data:
            return pd.DataFrame()
        
        # Bybit returns: [startTime, open, high, low, close, volume, turnover]
        df = pd.DataFrame(raw_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['turnover'] = df['turnover'].astype(float)
        
        # Sort by timestamp ascending
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 200,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Number of candles to fetch
            use_cache: Whether to use cache
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        if use_cache:
            cached = self.cache.get(symbol, timeframe)
            if cached is not None and len(cached) >= limit:
                logger.debug(f"Cache hit for {symbol} {timeframe}")
                return cached.tail(limit).reset_index(drop=True)
        
        # Fetch from API
        interval = self._convert_interval(timeframe)
        
        try:
            raw_data = self.client.get_kline(
                symbol=symbol,
                interval=interval,
                limit=limit,
                category=self.category
            )
            
            df = self._parse_kline_data(raw_data)
            
            # Update cache
            if use_cache and not df.empty:
                self.cache.set(symbol, timeframe, df)
            
            logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a specific time range.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_time: Start datetime
            end_time: End datetime (default: now)
            
        Returns:
            DataFrame with OHLCV data
        """
        end_time = end_time or datetime.utcnow()
        
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        interval = self._convert_interval(timeframe)
        
        all_data = []
        current_end = end_ms
        
        # Fetch in batches (max 1000 per request)
        while current_end > start_ms:
            try:
                raw_data = self.client.get_kline(
                    symbol=symbol,
                    interval=interval,
                    limit=1000,
                    category=self.category,
                    end_time=current_end
                )
                
                if not raw_data:
                    break
                
                all_data.extend(raw_data)
                
                # Get earliest timestamp from this batch
                earliest = min(int(d[0]) for d in raw_data)
                current_end = earliest - 1
                
                # Avoid hitting rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching range data: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = self._parse_kline_data(all_data)
        
        # Filter to exact range
        df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
        
        return df.drop_duplicates('timestamp').reset_index(drop=True)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest price or None
        """
        try:
            tickers = self.client.get_tickers(symbol=symbol, category=self.category)
            if tickers:
                return float(tickers[0].get('lastPrice', 0))
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
        
        return None
    
    def get_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "5m",
        limit: int = 200
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe
            limit: Number of candles per symbol
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        result = {}
        
        for symbol in symbols:
            df = self.get_ohlcv(symbol, timeframe, limit)
            if not df.empty:
                result[symbol] = df
            
            # Small delay to avoid rate limiting
            time.sleep(0.05)
        
        return result
    
    def get_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str],
        limit: int = 200
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV data for multiple timeframes.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes
            limit: Number of candles per timeframe
            
        Returns:
            Dictionary of timeframe -> DataFrame
        """
        result = {}
        
        for tf in timeframes:
            df = self.get_ohlcv(symbol, tf, limit)
            if not df.empty:
                result[tf] = df
        
        return result
    
    def update_candle(
        self,
        symbol: str,
        timeframe: str,
        candle_data: Dict
    ):
        """
        Update cache with new candle data from WebSocket.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            candle_data: New candle data
        """
        cached = self.cache.get(symbol, timeframe)
        
        if cached is None:
            return
        
        # Parse new candle
        new_candle = {
            'timestamp': pd.to_datetime(int(candle_data.get('start', 0)), unit='ms'),
            'open': float(candle_data.get('open', 0)),
            'high': float(candle_data.get('high', 0)),
            'low': float(candle_data.get('low', 0)),
            'close': float(candle_data.get('close', 0)),
            'volume': float(candle_data.get('volume', 0)),
            'turnover': float(candle_data.get('turnover', 0))
        }
        
        # Check if we should update or append
        if not cached.empty:
            last_ts = cached.iloc[-1]['timestamp']
            new_ts = new_candle['timestamp']
            
            if new_ts == last_ts:
                # Update last candle
                for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                    cached.iloc[-1, cached.columns.get_loc(col)] = new_candle[col]
            elif new_ts > last_ts:
                # Append new candle
                cached = pd.concat([
                    cached,
                    pd.DataFrame([new_candle])
                ], ignore_index=True)
            
            self.cache.set(symbol, timeframe, cached)
    
    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """
        Clear cache.
        
        Args:
            symbol: Specific symbol to clear (optional)
            timeframe: Specific timeframe to clear (optional)
        """
        if symbol and timeframe:
            self.cache.invalidate(symbol, timeframe)
        else:
            self.cache.clear()
