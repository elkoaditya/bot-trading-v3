"""
Order Executor for placing and managing orders.
Handles order placement, validation, and tracking.
"""
import asyncio
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from ..core.bybit_client import BybitClient
from ..utils.logger import get_logger, TradingLogger
from ..utils.helpers import retry_with_backoff, round_to_precision, round_to_tick_size


logger = get_logger("order_executor")


class OrderStatus(Enum):
    """Order status types."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class OrderSide(Enum):
    """Order side types."""
    BUY = "Buy"
    SELL = "Sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "Market"
    LIMIT = "Limit"


@dataclass
class Order:
    """Order data class."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: int = 10
    reduce_only: bool = False
    
    # Tracking fields
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    exchange_order_id: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'leverage': self.leverage,
            'status': self.status.value,
            'filled_qty': self.filled_qty,
            'avg_price': self.avg_price,
            'exchange_order_id': self.exchange_order_id,
            'created_at': self.created_at.isoformat(),
        }


class OrderValidator:
    """Validates orders before submission."""
    
    def __init__(self, client: BybitClient):
        self.client = client
        self._symbol_info_cache: Dict[str, Dict] = {}
    
    async def validate(self, order: Order) -> Tuple[bool, str]:
        """
        Validate an order.
        
        Args:
            order: Order to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Get symbol info
        info = await self._get_symbol_info(order.symbol)
        
        if not info:
            return False, f"Symbol {order.symbol} not found"
        
        # Validate quantity
        min_qty = float(info.get('lotSizeFilter', {}).get('minOrderQty', 0))
        max_qty = float(info.get('lotSizeFilter', {}).get('maxOrderQty', float('inf')))
        qty_step = float(info.get('lotSizeFilter', {}).get('qtyStep', 0.001))
        
        if order.quantity < min_qty:
            return False, f"Quantity {order.quantity} below minimum {min_qty}"
        
        if order.quantity > max_qty:
            return False, f"Quantity {order.quantity} above maximum {max_qty}"
        
        # Validate price for limit orders
        if order.order_type == OrderType.LIMIT and order.price:
            min_price = float(info.get('priceFilter', {}).get('minPrice', 0))
            max_price = float(info.get('priceFilter', {}).get('maxPrice', float('inf')))
            tick_size = float(info.get('priceFilter', {}).get('tickSize', 0.01))
            
            if order.price < min_price:
                return False, f"Price {order.price} below minimum {min_price}"
            
            if order.price > max_price:
                return False, f"Price {order.price} above maximum {max_price}"
        
        return True, ""
    
    async def _get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol info with caching."""
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]
        
        try:
            info = self.client.get_symbol_info(symbol)
            if info:
                self._symbol_info_cache[symbol] = info
            return info
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            return None
    
    def normalize_quantity(self, symbol: str, quantity: float) -> float:
        """Normalize quantity to valid step size."""
        info = self._symbol_info_cache.get(symbol)
        if not info:
            return quantity
        
        qty_step = float(info.get('lotSizeFilter', {}).get('qtyStep', 0.001))
        return round_to_tick_size(quantity, qty_step)
    
    def normalize_price(self, symbol: str, price: float) -> float:
        """Normalize price to valid tick size."""
        info = self._symbol_info_cache.get(symbol)
        if not info:
            return price
        
        tick_size = float(info.get('priceFilter', {}).get('tickSize', 0.01))
        return round_to_tick_size(price, tick_size)


class OrderExecutor:
    """
    Executor for placing and managing orders.
    """
    
    def __init__(
        self,
        client: Optional[BybitClient] = None,
        category: str = "linear",
        default_leverage: int = 10
    ):
        """
        Initialize order executor.
        
        Args:
            client: BybitClient instance
            category: Product category (linear, inverse)
            default_leverage: Default leverage for positions
        """
        self.client = client or BybitClient()
        self.category = category
        self.default_leverage = default_leverage
        self.validator = OrderValidator(self.client)
        
        # Order tracking
        self._pending_orders: Dict[str, Order] = {}
        self._completed_orders: List[Order] = []
        self._leverage_set: Dict[str, bool] = {}
        
        logger.info("Order executor initialized")
    
    async def execute_order(self, order: Order) -> Order:
        """
        Execute an order.
        
        Args:
            order: Order to execute
            
        Returns:
            Updated order with status
        """
        trade_logger = TradingLogger(order.symbol)
        
        try:
            # Validate order
            is_valid, error_msg = await self.validator.validate(order)
            if not is_valid:
                order.status = OrderStatus.REJECTED
                order.error_message = error_msg
                trade_logger.error(f"Order rejected: {error_msg}")
                return order
            
            # Set leverage if not already set
            await self._ensure_leverage(order.symbol, order.leverage)
            
            # Normalize quantity and price
            qty = self.validator.normalize_quantity(order.symbol, order.quantity)
            price = None
            if order.price:
                price = self.validator.normalize_price(order.symbol, order.price)
            
            # Prepare stop loss and take profit
            sl = None
            tp = None
            if order.stop_loss:
                sl = str(self.validator.normalize_price(order.symbol, order.stop_loss))
            if order.take_profit:
                tp = str(self.validator.normalize_price(order.symbol, order.take_profit))
            
            # Place order
            trade_logger.info(
                f"Placing {order.order_type.value} order: "
                f"{order.side.value} {qty} @ {price or 'market'}"
            )
            
            result = self.client.place_order(
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                qty=str(qty),
                category=self.category,
                price=str(price) if price else None,
                reduce_only=order.reduce_only,
                stop_loss=sl,
                take_profit=tp
            )
            
            # Update order with result
            order.exchange_order_id = result.get('orderId')
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.utcnow()
            
            # Track order
            self._pending_orders[order.order_id] = order
            
            trade_logger.trade(
                action="PLACED",
                price=price or 0,
                quantity=qty,
                side=order.side.value
            )
            
            return order
            
        except Exception as e:
            order.status = OrderStatus.FAILED
            order.error_message = str(e)
            order.updated_at = datetime.utcnow()
            
            trade_logger.error(f"Order failed: {e}")
            
            return order
    
    async def execute_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        leverage: int = None,
        stop_loss: float = None,
        take_profit: float = None,
        reduce_only: bool = False
    ) -> Order:
        """
        Execute a market order.
        
        Args:
            symbol: Trading symbol
            side: Order side (Buy/Sell)
            quantity: Order quantity
            leverage: Leverage (optional)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            reduce_only: Reduce only flag
            
        Returns:
            Executed order
        """
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            leverage=leverage or self.default_leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reduce_only=reduce_only
        )
        
        return await self.execute_order(order)
    
    async def execute_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        leverage: int = None,
        stop_loss: float = None,
        take_profit: float = None,
        reduce_only: bool = False
    ) -> Order:
        """
        Execute a limit order.
        
        Args:
            symbol: Trading symbol
            side: Order side (Buy/Sell)
            quantity: Order quantity
            price: Limit price
            leverage: Leverage (optional)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            reduce_only: Reduce only flag
            
        Returns:
            Executed order
        """
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            leverage=leverage or self.default_leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reduce_only=reduce_only
        )
        
        return await self.execute_order(order)
    
    async def close_position(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> Order:
        """
        Close a position.
        
        Args:
            symbol: Trading symbol
            side: Position side ('long' or 'short')
            quantity: Position quantity to close
            
        Returns:
            Executed order
        """
        # Opposite side to close
        order_side = OrderSide.SELL if side == 'long' else OrderSide.BUY
        
        return await self.execute_market_order(
            symbol=symbol,
            side=order_side,
            quantity=quantity,
            reduce_only=True
        )
    
    async def cancel_order(
        self,
        symbol: str,
        order_id: str = None,
        exchange_order_id: str = None
    ) -> bool:
        """
        Cancel an order.
        
        Args:
            symbol: Trading symbol
            order_id: Internal order ID
            exchange_order_id: Exchange order ID
            
        Returns:
            True if cancelled successfully
        """
        try:
            if order_id and order_id in self._pending_orders:
                order = self._pending_orders[order_id]
                exchange_order_id = order.exchange_order_id
            
            if not exchange_order_id:
                logger.warning("No exchange order ID provided for cancellation")
                return False
            
            self.client.cancel_order(
                symbol=symbol,
                order_id=exchange_order_id,
                category=self.category
            )
            
            # Update local tracking
            if order_id and order_id in self._pending_orders:
                order = self._pending_orders.pop(order_id)
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.utcnow()
                self._completed_orders.append(order)
            
            logger.info(f"Order cancelled: {exchange_order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.
        
        Args:
            symbol: Specific symbol (optional)
            
        Returns:
            Number of orders cancelled
        """
        try:
            result = self.client.cancel_all_orders(
                symbol=symbol,
                category=self.category
            )
            
            cancelled_count = len(result.get('list', []))
            
            # Update local tracking
            for order_id in list(self._pending_orders.keys()):
                order = self._pending_orders.get(order_id)
                if order and (symbol is None or order.symbol == symbol):
                    self._pending_orders.pop(order_id)
                    order.status = OrderStatus.CANCELLED
                    self._completed_orders.append(order)
            
            logger.info(f"Cancelled {cancelled_count} orders")
            return cancelled_count
            
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return 0
    
    async def get_order_status(self, order: Order) -> Order:
        """
        Get updated order status from exchange.
        
        Args:
            order: Order to check
            
        Returns:
            Updated order
        """
        if not order.exchange_order_id:
            return order
        
        try:
            orders = self.client.get_order_history(
                symbol=order.symbol,
                category=self.category,
                limit=50
            )
            
            for o in orders:
                if o.get('orderId') == order.exchange_order_id:
                    status = o.get('orderStatus', '')
                    
                    if status == 'Filled':
                        order.status = OrderStatus.FILLED
                        order.filled_qty = float(o.get('cumExecQty', 0))
                        order.avg_price = float(o.get('avgPrice', 0))
                    elif status == 'PartiallyFilled':
                        order.status = OrderStatus.PARTIAL
                        order.filled_qty = float(o.get('cumExecQty', 0))
                    elif status == 'Cancelled':
                        order.status = OrderStatus.CANCELLED
                    elif status == 'Rejected':
                        order.status = OrderStatus.REJECTED
                    
                    order.updated_at = datetime.utcnow()
                    break
            
            return order
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return order
    
    async def _ensure_leverage(self, symbol: str, leverage: int):
        """Ensure leverage is set for symbol."""
        if symbol in self._leverage_set:
            return
        
        try:
            self.client.set_leverage(
                symbol=symbol,
                buy_leverage=str(leverage),
                sell_leverage=str(leverage),
                category=self.category
            )
            self._leverage_set[symbol] = True
            logger.info(f"Set leverage for {symbol}: {leverage}x")
        except Exception as e:
            # Leverage might already be set
            if "leverage not modified" not in str(e).lower():
                logger.warning(f"Could not set leverage for {symbol}: {e}")
            self._leverage_set[symbol] = True
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get list of pending orders."""
        orders = list(self._pending_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
    
    def get_completed_orders(self, limit: int = 100) -> List[Order]:
        """Get list of completed orders."""
        return self._completed_orders[-limit:]
    
    async def sync_orders(self):
        """Sync pending orders with exchange."""
        for order_id, order in list(self._pending_orders.items()):
            updated_order = await self.get_order_status(order)
            
            if updated_order.status in [
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED
            ]:
                self._pending_orders.pop(order_id, None)
                self._completed_orders.append(updated_order)


# Add missing import
from typing import Tuple
