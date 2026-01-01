"""
WebSocket Manager for real-time data from Bybit.
Handles kline/trade/orderbook streams with auto-reconnect.
"""
import asyncio
import json
from typing import Optional, Dict, List, Callable, Any
from datetime import datetime
import websockets
from websockets.exceptions import ConnectionClosed

from ..utils.logger import get_logger


logger = get_logger("websocket")


class WebSocketManager:
    """
    WebSocket manager for Bybit real-time data streams.
    """
    
    # WebSocket endpoints
    DEMO_WS_PUBLIC = "wss://stream-demo.bybit.com/v5/public/linear"
    DEMO_WS_PRIVATE = "wss://stream-demo.bybit.com/v5/private"
    
    REAL_WS_PUBLIC = "wss://stream.bybit.com/v5/public/linear"
    REAL_WS_PRIVATE = "wss://stream.bybit.com/v5/private"
    
    def __init__(
        self,
        is_demo: bool = True,
        ping_interval: int = 20,
        reconnect_delay: int = 5
    ):
        """
        Initialize WebSocket manager.
        
        Args:
            is_demo: Use demo environment
            ping_interval: Ping interval in seconds
            reconnect_delay: Delay before reconnection in seconds
        """
        self.is_demo = is_demo
        self.ping_interval = ping_interval
        self.reconnect_delay = reconnect_delay
        
        # Connection state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._subscriptions: Dict[str, Callable] = {}
        self._message_handlers: Dict[str, List[Callable]] = {}
        
        # Select endpoint
        self.public_url = self.DEMO_WS_PUBLIC if is_demo else self.REAL_WS_PUBLIC
        
        logger.info(f"WebSocket manager initialized (demo={is_demo})")
    
    async def connect(self):
        """Establish WebSocket connection."""
        try:
            self._ws = await websockets.connect(
                self.public_url,
                ping_interval=self.ping_interval,
                ping_timeout=10
            )
            self._running = True
            logger.info(f"WebSocket connected to {self.public_url}")
            
            # Resubscribe to all topics
            for topic in self._subscriptions:
                await self._subscribe_topic(topic)
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Close WebSocket connection."""
        self._running = False
        
        if self._ws:
            await self._ws.close()
            self._ws = None
        
        logger.info("WebSocket disconnected")
    
    async def _subscribe_topic(self, topic: str):
        """Send subscribe message for a topic."""
        if not self._ws:
            return
        
        subscribe_msg = {
            "op": "subscribe",
            "args": [topic]
        }
        
        await self._ws.send(json.dumps(subscribe_msg))
        logger.debug(f"Subscribed to {topic}")
    
    async def _unsubscribe_topic(self, topic: str):
        """Send unsubscribe message for a topic."""
        if not self._ws:
            return
        
        unsubscribe_msg = {
            "op": "unsubscribe",
            "args": [topic]
        }
        
        await self._ws.send(json.dumps(unsubscribe_msg))
        logger.debug(f"Unsubscribed from {topic}")
    
    def subscribe_kline(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Dict], None]
    ):
        """
        Subscribe to kline/candlestick stream.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Interval (1, 5, 15, 30, 60, 240, D, W)
            callback: Callback function for kline updates
        """
        topic = f"kline.{interval}.{symbol}"
        self._subscriptions[topic] = callback
        
        if topic not in self._message_handlers:
            self._message_handlers[topic] = []
        self._message_handlers[topic].append(callback)
        
        logger.info(f"Registered kline subscription: {topic}")
    
    def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[Dict], None]
    ):
        """
        Subscribe to ticker stream.
        
        Args:
            symbol: Trading symbol
            callback: Callback function for ticker updates
        """
        topic = f"tickers.{symbol}"
        self._subscriptions[topic] = callback
        
        if topic not in self._message_handlers:
            self._message_handlers[topic] = []
        self._message_handlers[topic].append(callback)
        
        logger.info(f"Registered ticker subscription: {topic}")
    
    def subscribe_orderbook(
        self,
        symbol: str,
        depth: int = 50,
        callback: Callable[[Dict], None] = None
    ):
        """
        Subscribe to orderbook stream.
        
        Args:
            symbol: Trading symbol
            depth: Order book depth (1, 50, 200, 500)
            callback: Callback function for orderbook updates
        """
        topic = f"orderbook.{depth}.{symbol}"
        self._subscriptions[topic] = callback
        
        if topic not in self._message_handlers:
            self._message_handlers[topic] = []
        if callback:
            self._message_handlers[topic].append(callback)
        
        logger.info(f"Registered orderbook subscription: {topic}")
    
    def subscribe_trade(
        self,
        symbol: str,
        callback: Callable[[Dict], None]
    ):
        """
        Subscribe to public trade stream.
        
        Args:
            symbol: Trading symbol
            callback: Callback function for trade updates
        """
        topic = f"publicTrade.{symbol}"
        self._subscriptions[topic] = callback
        
        if topic not in self._message_handlers:
            self._message_handlers[topic] = []
        self._message_handlers[topic].append(callback)
        
        logger.info(f"Registered trade subscription: {topic}")
    
    def unsubscribe(self, symbol: str, stream_type: str):
        """
        Unsubscribe from a stream.
        
        Args:
            symbol: Trading symbol
            stream_type: Type of stream (kline, ticker, orderbook, trade)
        """
        # Find and remove matching subscriptions
        to_remove = []
        for topic in list(self._subscriptions.keys()):
            if symbol in topic and stream_type in topic:
                to_remove.append(topic)
        
        for topic in to_remove:
            del self._subscriptions[topic]
            if topic in self._message_handlers:
                del self._message_handlers[topic]
        
        logger.info(f"Unsubscribed from {stream_type} for {symbol}")
    
    def add_handler(self, topic: str, callback: Callable):
        """Add a message handler for a topic."""
        if topic not in self._message_handlers:
            self._message_handlers[topic] = []
        self._message_handlers[topic].append(callback)
    
    def remove_handler(self, topic: str, callback: Callable):
        """Remove a message handler."""
        if topic in self._message_handlers:
            self._message_handlers[topic] = [
                h for h in self._message_handlers[topic] if h != callback
            ]
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            # Handle subscription confirmation
            if data.get('op') == 'subscribe':
                if data.get('success'):
                    logger.debug(f"Subscription confirmed: {data.get('conn_id')}")
                else:
                    logger.warning(f"Subscription failed: {data.get('ret_msg')}")
                return
            
            # Handle pong
            if data.get('op') == 'pong':
                return
            
            # Handle data messages
            topic = data.get('topic', '')
            
            if topic and topic in self._message_handlers:
                msg_data = data.get('data', [])
                
                for handler in self._message_handlers[topic]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(msg_data)
                        else:
                            handler(msg_data)
                    except Exception as e:
                        logger.error(f"Handler error for {topic}: {e}")
            
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message received")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def listen(self):
        """Main listening loop with auto-reconnect."""
        while self._running:
            try:
                if not self._ws:
                    await self.connect()
                
                async for message in self._ws:
                    await self._handle_message(message)
                    
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                self._ws = None
                
                if self._running:
                    logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                    await asyncio.sleep(self.reconnect_delay)
                    
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self._ws = None
                
                if self._running:
                    await asyncio.sleep(self.reconnect_delay)
    
    async def start(self):
        """Start WebSocket manager."""
        logger.info("Starting WebSocket manager...")
        
        await self.connect()
        
        # Subscribe to all registered topics
        for topic in list(self._subscriptions.keys()):
            await self._subscribe_topic(topic)
            await asyncio.sleep(0.1)  # Small delay between subscriptions
        
        # Start listening
        await self.listen()
    
    async def stop(self):
        """Stop WebSocket manager."""
        logger.info("Stopping WebSocket manager...")
        await self.disconnect()
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws is not None and self._ws.open


class KlineAggregator:
    """
    Aggregates kline updates and triggers callbacks on candle close.
    """
    
    def __init__(self, on_candle_close: Callable[[str, str, Dict], None]):
        """
        Initialize aggregator.
        
        Args:
            on_candle_close: Callback when a candle closes (symbol, interval, candle_data)
        """
        self.on_candle_close = on_candle_close
        self._current_candles: Dict[str, Dict] = {}
    
    def handle_kline(self, symbol: str, interval: str, data: List[Dict]):
        """
        Handle kline update.
        
        Args:
            symbol: Trading symbol
            interval: Interval
            data: Kline data list
        """
        for candle in data:
            key = f"{symbol}_{interval}"
            
            is_confirm = candle.get('confirm', False)
            
            if is_confirm and self.on_candle_close:
                # Candle closed
                self.on_candle_close(symbol, interval, candle)
            
            # Update current candle
            self._current_candles[key] = candle
    
    def get_current_candle(self, symbol: str, interval: str) -> Optional[Dict]:
        """Get current (incomplete) candle."""
        key = f"{symbol}_{interval}"
        return self._current_candles.get(key)
