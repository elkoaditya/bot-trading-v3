"""
Bybit API Client wrapper using pybit.
Supports Demo Trading (Mainnet Demo) and Real (Mainnet) environments.

Note: Bybit Demo Trading uses the mainnet infrastructure with virtual funds.
Endpoint: https://api-demo.bybit.com
"""
import os
from typing import Optional, Dict, Any, List
from pybit.unified_trading import HTTP
from dataclasses import dataclass

from ..utils.logger import get_logger
from ..utils.helpers import retry_with_backoff


logger = get_logger("bybit_client")


@dataclass
class BybitCredentials:
    """API credentials for Bybit."""
    api_key: str
    api_secret: str
    is_demo: bool = True


class BybitClient:
    """
    Bybit API client wrapper with support for Demo and Real environments.
    Uses singleton pattern to ensure single connection.
    """
    
    _instance: Optional['BybitClient'] = None
    _initialized: bool = False
    
    # API Endpoints
    # Demo trading uses mainnet demo (not testnet)
    DEMO_URL = "https://api-demo.bybit.com"  # Mainnet Demo
    REAL_URL = "https://api.bybit.com"       # Mainnet Real
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        is_demo: bool = True
    ):
        """
        Initialize Bybit client.
        
        Args:
            api_key: API key (or from environment)
            api_secret: API secret (or from environment)
            is_demo: Use demo/testnet environment
        """
        if self._initialized:
            return
        
        self.is_demo = is_demo
        
        # Get credentials from environment if not provided
        if is_demo:
            self.api_key = api_key or os.getenv('BYBIT_DEMO_API_KEY', '')
            self.api_secret = api_secret or os.getenv('BYBIT_DEMO_API_SECRET', '')
        else:
            self.api_key = api_key or os.getenv('BYBIT_REAL_API_KEY', '')
            self.api_secret = api_secret or os.getenv('BYBIT_REAL_API_SECRET', '')
        
        # Initialize HTTP session
        self.base_url = self.DEMO_URL if is_demo else self.REAL_URL
        
        # For Bybit Demo Trading (Mainnet Demo), we use:
        # - testnet=False (because it's mainnet infrastructure)
        # - demo=True (to enable demo trading mode)
        if is_demo:
            self.session = HTTP(
                testnet=False,
                demo=True,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
        else:
            self.session = HTTP(
                testnet=False,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
        
        self._initialized = True
        
        env_name = "Demo" if is_demo else "Real"
        logger.info(f"Bybit client initialized for {env_name} environment")
    
    @classmethod
    def reset(cls):
        """Reset the singleton instance."""
        cls._instance = None
        cls._initialized = False
    
    def _handle_response(self, response: Dict) -> Dict:
        """
        Handle API response and check for errors.
        
        Args:
            response: API response dictionary
            
        Returns:
            Response result data
            
        Raises:
            Exception: If API returns an error
        """
        if response.get('retCode') != 0:
            error_msg = response.get('retMsg', 'Unknown error')
            raise Exception(f"Bybit API Error: {error_msg}")
        
        return response.get('result', {})
    
    # ==================== Account Methods ====================
    
    @retry_with_backoff(max_retries=3)
    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict:
        """
        Get wallet balance.
        
        Args:
            account_type: Account type (UNIFIED, CONTRACT, SPOT)
            
        Returns:
            Wallet balance data
        """
        response = self.session.get_wallet_balance(accountType=account_type)
        return self._handle_response(response)
    
    @retry_with_backoff(max_retries=3)
    def get_positions(
        self,
        category: str = "linear",
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        Get current positions.
        
        Args:
            category: Product category (linear, inverse)
            symbol: Trading symbol (optional)
            
        Returns:
            List of position data
        """
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        
        response = self.session.get_positions(**params)
        result = self._handle_response(response)
        
        return result.get('list', [])
    
    # ==================== Market Data Methods ====================
    
    @retry_with_backoff(max_retries=3)
    def get_kline(
        self,
        symbol: str,
        interval: str = "5",
        limit: int = 200,
        category: str = "linear",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[List]:
        """
        Get kline/candlestick data.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Timeframe (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            limit: Number of candles to fetch (max 1000)
            category: Product category
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            
        Returns:
            List of OHLCV data
        """
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        
        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time
        
        response = self.session.get_kline(**params)
        result = self._handle_response(response)
        
        return result.get('list', [])
    
    @retry_with_backoff(max_retries=3)
    def get_tickers(
        self,
        symbol: Optional[str] = None,
        category: str = "linear"
    ) -> List[Dict]:
        """
        Get ticker information.
        
        Args:
            symbol: Trading symbol (optional, gets all if None)
            category: Product category
            
        Returns:
            List of ticker data
        """
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        
        response = self.session.get_tickers(**params)
        result = self._handle_response(response)
        
        return result.get('list', [])
    
    @retry_with_backoff(max_retries=3)
    def get_orderbook(
        self,
        symbol: str,
        category: str = "linear",
        limit: int = 50
    ) -> Dict:
        """
        Get orderbook.
        
        Args:
            symbol: Trading symbol
            category: Product category
            limit: Depth (1-500)
            
        Returns:
            Orderbook data
        """
        response = self.session.get_orderbook(
            category=category,
            symbol=symbol,
            limit=limit
        )
        return self._handle_response(response)
    
    # ==================== Trading Methods ====================
    
    @retry_with_backoff(max_retries=3)
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: str,
        category: str = "linear",
        price: Optional[str] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        close_on_trigger: bool = False,
        position_idx: int = 0,
        take_profit: Optional[str] = None,
        stop_loss: Optional[str] = None
    ) -> Dict:
        """
        Place a new order.
        
        Args:
            symbol: Trading symbol
            side: Order side ('Buy' or 'Sell')
            order_type: Order type ('Market' or 'Limit')
            qty: Order quantity
            category: Product category
            price: Limit price (required for Limit orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            reduce_only: Reduce only flag
            close_on_trigger: Close on trigger flag
            position_idx: Position index (0=one-way, 1=buy-side, 2=sell-side)
            take_profit: Take profit price
            stop_loss: Stop loss price
            
        Returns:
            Order result data
        """
        params = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
            "timeInForce": time_in_force,
            "reduceOnly": reduce_only,
            "closeOnTrigger": close_on_trigger,
            "positionIdx": position_idx
        }
        
        if price and order_type == "Limit":
            params["price"] = price
        
        if take_profit:
            params["takeProfit"] = take_profit
        
        if stop_loss:
            params["stopLoss"] = stop_loss
        
        logger.info(f"Placing order: {side} {qty} {symbol} @ {order_type}")
        
        response = self.session.place_order(**params)
        result = self._handle_response(response)
        
        logger.info(f"Order placed successfully: {result.get('orderId')}")
        
        return result
    
    @retry_with_backoff(max_retries=3)
    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        category: str = "linear"
    ) -> Dict:
        """
        Cancel an order.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID
            order_link_id: Custom order ID
            category: Product category
            
        Returns:
            Cancellation result
        """
        params = {
            "category": category,
            "symbol": symbol
        }
        
        if order_id:
            params["orderId"] = order_id
        if order_link_id:
            params["orderLinkId"] = order_link_id
        
        response = self.session.cancel_order(**params)
        return self._handle_response(response)
    
    @retry_with_backoff(max_retries=3)
    def cancel_all_orders(
        self,
        symbol: Optional[str] = None,
        category: str = "linear"
    ) -> Dict:
        """
        Cancel all orders.
        
        Args:
            symbol: Trading symbol (optional)
            category: Product category
            
        Returns:
            Cancellation result
        """
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        
        response = self.session.cancel_all_orders(**params)
        return self._handle_response(response)
    
    @retry_with_backoff(max_retries=3)
    def get_open_orders(
        self,
        symbol: Optional[str] = None,
        category: str = "linear"
    ) -> List[Dict]:
        """
        Get open orders.
        
        Args:
            symbol: Trading symbol (optional)
            category: Product category
            
        Returns:
            List of open orders
        """
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        
        response = self.session.get_open_orders(**params)
        result = self._handle_response(response)
        
        return result.get('list', [])
    
    @retry_with_backoff(max_retries=3)
    def get_order_history(
        self,
        symbol: Optional[str] = None,
        category: str = "linear",
        limit: int = 50
    ) -> List[Dict]:
        """
        Get order history.
        
        Args:
            symbol: Trading symbol (optional)
            category: Product category
            limit: Number of orders to fetch
            
        Returns:
            List of historical orders
        """
        params = {
            "category": category,
            "limit": limit
        }
        if symbol:
            params["symbol"] = symbol
        
        response = self.session.get_order_history(**params)
        result = self._handle_response(response)
        
        return result.get('list', [])
    
    # ==================== Position Management ====================
    
    @retry_with_backoff(max_retries=3)
    def set_leverage(
        self,
        symbol: str,
        buy_leverage: str,
        sell_leverage: str,
        category: str = "linear"
    ) -> Dict:
        """
        Set position leverage.
        
        Args:
            symbol: Trading symbol
            buy_leverage: Leverage for buy side
            sell_leverage: Leverage for sell side
            category: Product category
            
        Returns:
            Result data
        """
        response = self.session.set_leverage(
            category=category,
            symbol=symbol,
            buyLeverage=buy_leverage,
            sellLeverage=sell_leverage
        )
        return self._handle_response(response)
    
    @retry_with_backoff(max_retries=3)
    def set_trading_stop(
        self,
        symbol: str,
        take_profit: Optional[str] = None,
        stop_loss: Optional[str] = None,
        position_idx: int = 0,
        category: str = "linear"
    ) -> Dict:
        """
        Set take profit and stop loss.
        
        Args:
            symbol: Trading symbol
            take_profit: Take profit price
            stop_loss: Stop loss price
            position_idx: Position index
            category: Product category
            
        Returns:
            Result data
        """
        params = {
            "category": category,
            "symbol": symbol,
            "positionIdx": position_idx
        }
        
        if take_profit:
            params["takeProfit"] = take_profit
        if stop_loss:
            params["stopLoss"] = stop_loss
        
        response = self.session.set_trading_stop(**params)
        return self._handle_response(response)
    
    # ==================== Instrument Info ====================
    
    @retry_with_backoff(max_retries=3)
    def get_instruments_info(
        self,
        symbol: Optional[str] = None,
        category: str = "linear"
    ) -> List[Dict]:
        """
        Get instrument specifications.
        
        Args:
            symbol: Trading symbol (optional)
            category: Product category
            
        Returns:
            List of instrument info
        """
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        
        response = self.session.get_instruments_info(**params)
        result = self._handle_response(response)
        
        return result.get('list', [])
    
    def get_symbol_info(self, symbol: str, category: str = "linear") -> Optional[Dict]:
        """
        Get info for a specific symbol.
        
        Args:
            symbol: Trading symbol
            category: Product category
            
        Returns:
            Symbol info or None
        """
        instruments = self.get_instruments_info(symbol=symbol, category=category)
        return instruments[0] if instruments else None
    
    # ==================== Health Check ====================
    
    def health_check(self) -> bool:
        """
        Check API connection health.
        
        Returns:
            True if healthy
        """
        try:
            response = self.session.get_server_time()
            return response.get('retCode') == 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_server_time(self) -> int:
        """
        Get server time.
        
        Returns:
            Server timestamp in milliseconds
        """
        response = self.session.get_server_time()
        result = self._handle_response(response)
        return int(result.get('timeSecond', 0)) * 1000
