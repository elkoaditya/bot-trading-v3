"""
Trading Bot Main Entry Point.
Async main loop with graceful shutdown handling.
"""
import asyncio
import signal
import sys
from datetime import datetime, date
from typing import Optional
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.config_loader import ConfigLoader, BotConfig
from src.core.bybit_client import BybitClient
from src.core.data_fetcher import DataFetcher
from src.core.websocket_manager import WebSocketManager, KlineAggregator
from src.strategies.base_strategy import Signal
from src.trading.decision_engine import DecisionEngine, TradeAction
from src.trading.order_executor import OrderExecutor, OrderSide
from src.trading.risk_manager import RiskManager, PositionInfo
from src.database.models import Database, Trade, Position
from src.notifications.telegram_notifier import TelegramNotifier
from src.utils.logger import setup_logger, get_logger


# Setup logger
logger = setup_logger("trading_bot", log_level=os.getenv("LOG_LEVEL", "INFO"))


class TradingBot:
    """
    Main trading bot class.
    """
    
    def __init__(self, config_path: str = "config/bot_config.json"):
        """
        Initialize trading bot.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Optional[BotConfig] = None
        
        # Components
        self.client: Optional[BybitClient] = None
        self.data_fetcher: Optional[DataFetcher] = None
        self.ws_manager: Optional[WebSocketManager] = None
        self.decision_engine: Optional[DecisionEngine] = None
        self.order_executor: Optional[OrderExecutor] = None
        self.risk_manager: Optional[RiskManager] = None
        self.database: Optional[Database] = None
        self.notifier: Optional[TelegramNotifier] = None
        
        # State
        self._running = False
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing trading bot...")
        
        # Load configuration
        config_loader = ConfigLoader(self.config_path)
        self.config = config_loader.load()
        
        logger.info(f"Environment: {'Demo' if self.config.is_demo else 'Real'}")
        logger.info(f"Enabled coins: {list(c for c, cfg in self.config.coins.items() if cfg.enabled)}")
        
        # Initialize Bybit client
        self.client = BybitClient(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            is_demo=self.config.is_demo
        )
        
        # Check connection
        if not self.client.health_check():
            raise Exception("Failed to connect to Bybit API")
        
        logger.info("Bybit API connection successful")
        
        # Initialize data fetcher
        self.data_fetcher = DataFetcher(
            client=self.client,
            category=self.config.global_settings.category
        )
        
        # Initialize WebSocket manager
        self.ws_manager = WebSocketManager(is_demo=self.config.is_demo)
        
        # Initialize decision engine
        self.decision_engine = DecisionEngine(
            coin_configs=self.config.coins
        )
        
        # Initialize order executor
        self.order_executor = OrderExecutor(
            client=self.client,
            category=self.config.global_settings.category,
            default_leverage=self.config.global_settings.leverage
        )
        
        # Initialize risk manager
        coin_risk_configs = {
            name: cfg.risk_management
            for name, cfg in self.config.coins.items()
        }
        self.risk_manager = RiskManager(
            global_settings=self.config.global_settings,
            coin_risk_configs=coin_risk_configs
        )
        
        # Initialize database
        self.database = Database()
        await self.database.connect()
        
        # Load open positions from database
        await self._load_positions()
        
        # Initialize Telegram notifier
        self.notifier = TelegramNotifier(
            bot_token=self.config.telegram_token,
            chat_id=self.config.telegram_chat_id,
            enabled=self.config.telegram_enabled
        )
        
        # Update balance from exchange
        await self._update_balance()
        
        logger.info("Trading bot initialized successfully")
    
    async def _load_positions(self):
        """Load open positions from database."""
        positions = await self.database.get_open_positions()
        for pos in positions:
            position_info = PositionInfo(
                symbol=pos.symbol,
                side=pos.side,
                quantity=pos.quantity,
                entry_price=pos.entry_price,
                current_price=pos.entry_price,
                leverage=pos.leverage,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit
            )
            self.risk_manager.open_position(position_info)
        
        if positions:
            logger.info(f"Loaded {len(positions)} open positions from database")
    
    async def _update_balance(self):
        """Update balance from exchange."""
        try:
            wallet = self.client.get_wallet_balance()
            if wallet and 'list' in wallet:
                for account in wallet['list']:
                    if account.get('accountType') == 'UNIFIED':
                        balance = float(account.get('totalEquity', 0))
                        self.risk_manager.update_balance(balance)
                        logger.info(f"Updated balance: ${balance:,.2f}")
                        break
        except Exception as e:
            logger.error(f"Failed to update balance: {e}")
    
    async def start(self):
        """Start the trading bot."""
        await self.initialize()
        
        self._running = True
        
        # Notify start
        enabled_coins = [c for c, cfg in self.config.coins.items() if cfg.enabled]
        await self.notifier.notify_bot_started(
            coins=enabled_coins,
            leverage=self.config.global_settings.leverage,
            capital=self.risk_manager.get_balance()
        )
        
        logger.info("Trading bot started")
        
        # Start main loop
        try:
            await self._main_loop()
        except asyncio.CancelledError:
            logger.info("Bot cancelled")
        except Exception as e:
            logger.error(f"Bot error: {e}")
            await self.notifier.notify_error(str(e), "Main loop")
        finally:
            await self.stop()
    
    async def _main_loop(self):
        """Main trading loop."""
        timeframe = self.config.global_settings.timeframe
        interval_seconds = self._get_interval_seconds(timeframe)
        
        logger.info(f"Starting main loop with {timeframe} timeframe ({interval_seconds}s interval)")
        
        while self._running:
            try:
                # Process each enabled coin
                for coin_name, coin_config in self.config.coins.items():
                    if not coin_config.enabled:
                        continue
                    
                    await self._process_coin(coin_name, coin_config.symbol)
                
                # Update balance periodically
                await self._update_balance()
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)  # Wait before retry
    
    async def _process_coin(self, coin_name: str, symbol: str):
        """Process a single coin."""
        try:
            # Fetch OHLCV data
            timeframe = self.config.global_settings.timeframe
            df = self.data_fetcher.get_ohlcv(symbol, timeframe, limit=200)
            
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Update position price if exists
            position = self.risk_manager.get_position(symbol)
            if position:
                self.risk_manager.update_position_price(symbol, current_price)
                
                # Check stop conditions
                should_close, reason = self.risk_manager.check_stop_conditions(symbol)
                if should_close:
                    await self._close_position(symbol, current_price, reason)
                    return
            
            # Get current position info for decision
            current_pos = None
            if position:
                current_pos = {'side': position.side, 'entry_price': position.entry_price}
            
            # Make trading decision
            decision = self.decision_engine.make_decision(
                coin_name=coin_name,
                df=df,
                current_position=current_pos,
                current_price=current_price
            )
            
            # Execute decision
            if decision.action == TradeAction.OPEN_LONG:
                await self._open_position(
                    symbol=symbol,
                    side='long',
                    price=current_price,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                    reason=decision.reason
                )
            
            elif decision.action == TradeAction.OPEN_SHORT:
                await self._open_position(
                    symbol=symbol,
                    side='short',
                    price=current_price,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                    reason=decision.reason
                )
            
            elif decision.action == TradeAction.CLOSE_LONG:
                await self._close_position(symbol, current_price, decision.reason)
            
            elif decision.action == TradeAction.CLOSE_SHORT:
                await self._close_position(symbol, current_price, decision.reason)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    async def _open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        stop_loss: float = None,
        take_profit: float = None,
        reason: str = ""
    ):
        """Open a new position."""
        # Check if can open
        can_open, msg = self.risk_manager.can_open_position(symbol)
        if not can_open:
            logger.info(f"Cannot open position for {symbol}: {msg}")
            return
        
        # Calculate position size
        if stop_loss:
            quantity = self.risk_manager.calculate_position_size(
                symbol=symbol,
                entry_price=price,
                stop_loss_price=stop_loss,
                side=side
            )
        else:
            # Default position size
            balance = self.risk_manager.get_balance()
            position_pct = self.config.global_settings.position_size
            quantity = (balance * position_pct * self.config.global_settings.leverage) / price
        
        if quantity <= 0:
            logger.warning(f"Invalid quantity for {symbol}")
            return
        
        # Place order
        order_side = OrderSide.BUY if side == 'long' else OrderSide.SELL
        order = await self.order_executor.execute_market_order(
            symbol=symbol,
            side=order_side,
            quantity=quantity,
            leverage=self.config.global_settings.leverage,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        if order.status.value in ['submitted', 'filled']:
            # Track position
            position_info = PositionInfo(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                leverage=self.config.global_settings.leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time=datetime.utcnow()
            )
            self.risk_manager.open_position(position_info)
            
            # Save to database
            await self.database.insert_position(Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                leverage=self.config.global_settings.leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time=datetime.utcnow().isoformat()
            ))
            
            # Notify
            await self.notifier.notify_trade_opened(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                leverage=self.config.global_settings.leverage,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            logger.info(f"Opened {side} position for {symbol}: {quantity} @ {price}")
    
    async def _close_position(self, symbol: str, exit_price: float, reason: str = ""):
        """Close an existing position."""
        position = self.risk_manager.get_position(symbol)
        if not position:
            return
        
        # Close position via order
        order = await self.order_executor.close_position(
            symbol=symbol,
            side=position.side,
            quantity=position.quantity
        )
        
        # Record trade
        trade_result = self.risk_manager.close_position(symbol, exit_price, reason)
        
        if trade_result:
            # Save to database
            await self.database.insert_trade(Trade(
                symbol=symbol,
                side=trade_result['side'],
                quantity=trade_result['quantity'],
                entry_price=trade_result['entry_price'],
                exit_price=trade_result['exit_price'],
                entry_time=trade_result['entry_time'].isoformat() if trade_result.get('entry_time') else "",
                exit_time=datetime.utcnow().isoformat(),
                leverage=trade_result['leverage'],
                pnl=trade_result['pnl'],
                pnl_pct=trade_result['pnl_pct'],
                fees=trade_result['fees'],
                reason=reason
            ))
            
            # Remove position from database
            await self.database.close_position(symbol)
            
            # Notify
            await self.notifier.notify_trade_closed(
                symbol=symbol,
                side=trade_result['side'],
                quantity=trade_result['quantity'],
                entry_price=trade_result['entry_price'],
                exit_price=exit_price,
                pnl=trade_result['pnl'],
                pnl_pct=trade_result['pnl_pct'],
                reason=reason
            )
            
            logger.info(f"Closed position for {symbol}: PnL ${trade_result['pnl']:.2f}")
    
    def _get_interval_seconds(self, timeframe: str) -> int:
        """Convert timeframe to seconds."""
        multipliers = {'m': 60, 'h': 3600, 'd': 86400}
        unit = timeframe[-1].lower()
        value = int(timeframe[:-1])
        return value * multipliers.get(unit, 60)
    
    async def stop(self):
        """Stop the trading bot."""
        logger.info("Stopping trading bot...")
        self._running = False
        
        # Close all positions optionally
        # (Add flag to control this behavior)
        
        # Disconnect database
        if self.database:
            await self.database.disconnect()
        
        # Disconnect WebSocket
        if self.ws_manager:
            await self.ws_manager.stop()
        
        # Notify stop
        if self.notifier:
            await self.notifier.notify_bot_stopped("Graceful shutdown")
        
        logger.info("Trading bot stopped")
    
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signal."""
        logger.info(f"Received shutdown signal {signum}")
        self._running = False


async def main():
    """Main entry point."""
    bot = TradingBot()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, bot.handle_shutdown)
    signal.signal(signal.SIGTERM, bot.handle_shutdown)
    
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
