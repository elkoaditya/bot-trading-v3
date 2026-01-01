"""
Decision Engine for trading signals.
Combines multiple strategies and makes trading decisions.
"""
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..strategies.base_strategy import BaseStrategy, Signal, MultiStrategyDecider
from ..strategies.trend_following import EMAStrategy, MACDStrategy
from ..strategies.mean_reversion import RSIStrategy, BollingerStrategy
from ..strategies.breakout import BreakoutStrategy
from ..config.config_loader import CoinConfig, StrategyConfig
from ..utils.logger import get_logger


logger = get_logger("decision_engine")


class TradeAction(Enum):
    """Trading action types."""
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    HOLD = "hold"


@dataclass
class TradeDecision:
    """Trade decision result."""
    action: TradeAction
    symbol: str
    signal: Signal
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity: Optional[float] = None
    reason: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class StrategyFactory:
    """Factory for creating strategy instances."""
    
    STRATEGY_MAP = {
        'rsi': RSIStrategy,
        'macd': MACDStrategy,
        'ema_crossover': EMAStrategy,
        'ema': EMAStrategy,
        'bollinger': BollingerStrategy,
        'breakout': BreakoutStrategy,
    }
    
    @classmethod
    def create(cls, config: StrategyConfig) -> Optional[BaseStrategy]:
        """
        Create a strategy instance from config.
        
        Args:
            config: Strategy configuration
            
        Returns:
            Strategy instance or None
        """
        strategy_class = cls.STRATEGY_MAP.get(config.name.lower())
        
        if strategy_class is None:
            logger.warning(f"Unknown strategy: {config.name}")
            return None
        
        return strategy_class(params=config.params)
    
    @classmethod
    def create_from_list(cls, configs: List[StrategyConfig]) -> List[BaseStrategy]:
        """Create multiple strategies from config list."""
        strategies = []
        for config in configs:
            strategy = cls.create(config)
            if strategy:
                strategies.append(strategy)
        return strategies


class DecisionEngine:
    """
    Decision engine for making trading decisions.
    """
    
    def __init__(
        self,
        coin_configs: Dict[str, CoinConfig],
        default_mode: str = "majority"
    ):
        """
        Initialize decision engine.
        
        Args:
            coin_configs: Dictionary of coin configurations
            default_mode: Default strategy combination mode
        """
        self.coin_configs = coin_configs
        self.default_mode = default_mode
        self._deciders: Dict[str, MultiStrategyDecider] = {}
        self._positions: Dict[str, dict] = {}  # Track current positions
        
        self._initialize_deciders()
        
        logger.info(f"Decision engine initialized with {len(coin_configs)} coins")
    
    def _initialize_deciders(self):
        """Initialize strategy deciders for each coin."""
        for coin_name, config in self.coin_configs.items():
            if not config.enabled:
                continue
            
            # Create strategies
            strategies = StrategyFactory.create_from_list(config.strategies)
            
            if not strategies:
                logger.warning(f"No valid strategies for {coin_name}")
                continue
            
            # Create decider
            mode = config.strategy_mode or self.default_mode
            self._deciders[coin_name] = MultiStrategyDecider(
                strategies=strategies,
                mode=mode
            )
            
            logger.info(
                f"Created decider for {coin_name}: "
                f"{len(strategies)} strategies, mode={mode}"
            )
    
    def get_signal(self, coin_name: str, df: pd.DataFrame) -> Signal:
        """
        Get combined signal for a coin.
        
        Args:
            coin_name: Coin name (e.g., 'BTC')
            df: OHLCV DataFrame
            
        Returns:
            Combined signal
        """
        if coin_name not in self._deciders:
            return Signal.HOLD
        
        return self._deciders[coin_name].get_combined_signal(df)
    
    def get_all_signals(self, coin_name: str, df: pd.DataFrame) -> Dict[str, Signal]:
        """
        Get signals from all strategies for a coin.
        
        Args:
            coin_name: Coin name
            df: OHLCV DataFrame
            
        Returns:
            Dictionary of strategy name -> signal
        """
        if coin_name not in self._deciders:
            return {}
        
        return self._deciders[coin_name].get_all_signals(df)
    
    def make_decision(
        self,
        coin_name: str,
        df: pd.DataFrame,
        current_position: Optional[dict] = None,
        current_price: Optional[float] = None
    ) -> TradeDecision:
        """
        Make trading decision for a coin.
        
        Args:
            coin_name: Coin name
            df: OHLCV DataFrame
            current_position: Current position info (if any)
            current_price: Current price (optional, uses close price)
            
        Returns:
            TradeDecision with action and details
        """
        config = self.coin_configs.get(coin_name)
        if not config or not config.enabled:
            return TradeDecision(
                action=TradeAction.HOLD,
                symbol=f"{coin_name}USDT",
                signal=Signal.HOLD,
                confidence=0.0,
                reason="Coin not enabled"
            )
        
        # Get combined signal
        signal = self.get_signal(coin_name, df)
        
        # Get current price
        price = current_price or (df['close'].iloc[-1] if len(df) > 0 else 0)
        
        # Determine action based on signal and current position
        action, reason = self._determine_action(signal, current_position)
        
        # Calculate stop loss and take profit
        stop_loss = None
        take_profit = None
        
        if action in [TradeAction.OPEN_LONG, TradeAction.OPEN_SHORT]:
            risk_mgmt = config.risk_management
            
            if action == TradeAction.OPEN_LONG:
                stop_loss = price * (1 - risk_mgmt.stop_loss_pct / 100)
                take_profit = price * (1 + risk_mgmt.take_profit_pct / 100)
            else:  # SHORT
                stop_loss = price * (1 + risk_mgmt.stop_loss_pct / 100)
                take_profit = price * (1 - risk_mgmt.take_profit_pct / 100)
        
        # Calculate confidence
        confidence = self._calculate_confidence(coin_name, df, signal)
        
        return TradeDecision(
            action=action,
            symbol=config.symbol,
            signal=signal,
            confidence=confidence,
            entry_price=price if action in [TradeAction.OPEN_LONG, TradeAction.OPEN_SHORT] else None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason
        )
    
    def _determine_action(
        self,
        signal: Signal,
        current_position: Optional[dict]
    ) -> Tuple[TradeAction, str]:
        """Determine trading action based on signal and position."""
        has_long = current_position and current_position.get('side') == 'long'
        has_short = current_position and current_position.get('side') == 'short'
        
        if signal in [Signal.BUY, Signal.STRONG_BUY]:
            if has_short:
                return TradeAction.CLOSE_SHORT, "Buy signal while in short position"
            elif not has_long:
                return TradeAction.OPEN_LONG, f"Opening long on {signal.value} signal"
            else:
                return TradeAction.HOLD, "Already in long position"
        
        elif signal in [Signal.SELL, Signal.STRONG_SELL]:
            if has_long:
                return TradeAction.CLOSE_LONG, "Sell signal while in long position"
            elif not has_short:
                return TradeAction.OPEN_SHORT, f"Opening short on {signal.value} signal"
            else:
                return TradeAction.HOLD, "Already in short position"
        
        return TradeAction.HOLD, "No actionable signal"
    
    def _calculate_confidence(
        self,
        coin_name: str,
        df: pd.DataFrame,
        signal: Signal
    ) -> float:
        """Calculate confidence score for the signal."""
        if coin_name not in self._deciders:
            return 0.0
        
        if signal == Signal.HOLD:
            return 0.0
        
        # Get individual signals
        all_signals = self._deciders[coin_name].get_all_signals(df)
        
        # Count agreeing signals
        buy_signals = [Signal.BUY, Signal.STRONG_BUY]
        sell_signals = [Signal.SELL, Signal.STRONG_SELL]
        
        if signal in buy_signals:
            agreeing = sum(1 for s in all_signals.values() if s in buy_signals)
        elif signal in sell_signals:
            agreeing = sum(1 for s in all_signals.values() if s in sell_signals)
        else:
            agreeing = 0
        
        total = len(all_signals)
        
        if total == 0:
            return 0.0
        
        base_confidence = agreeing / total
        
        # Bonus for strong signals
        if signal in [Signal.STRONG_BUY, Signal.STRONG_SELL]:
            base_confidence = min(base_confidence + 0.1, 1.0)
        
        return round(base_confidence, 2)
    
    def should_close_position(
        self,
        coin_name: str,
        df: pd.DataFrame,
        position: dict
    ) -> Tuple[bool, str]:
        """
        Check if a position should be closed.
        
        Args:
            coin_name: Coin name
            df: OHLCV DataFrame
            position: Current position info
            
        Returns:
            Tuple of (should_close, reason)
        """
        signal = self.get_signal(coin_name, df)
        side = position.get('side', 'long')
        entry_price = position.get('entry_price', 0)
        current_price = df['close'].iloc[-1] if len(df) > 0 else 0
        
        # Check for opposite signal
        if side == 'long' and signal in [Signal.SELL, Signal.STRONG_SELL]:
            return True, "Received sell signal"
        
        if side == 'short' and signal in [Signal.BUY, Signal.STRONG_BUY]:
            return True, "Received buy signal"
        
        # Check for stop loss / take profit
        config = self.coin_configs.get(coin_name)
        if config:
            risk_mgmt = config.risk_management
            
            pnl_pct = ((current_price - entry_price) / entry_price * 100
                       if side == 'long'
                       else (entry_price - current_price) / entry_price * 100)
            
            if pnl_pct <= -risk_mgmt.stop_loss_pct:
                return True, f"Stop loss triggered ({pnl_pct:.2f}%)"
            
            if pnl_pct >= risk_mgmt.take_profit_pct:
                return True, f"Take profit triggered ({pnl_pct:.2f}%)"
        
        return False, ""
    
    def update_position(self, coin_name: str, position: Optional[dict]):
        """Update tracked position for a coin."""
        if position:
            self._positions[coin_name] = position
        elif coin_name in self._positions:
            del self._positions[coin_name]
    
    def get_position(self, coin_name: str) -> Optional[dict]:
        """Get tracked position for a coin."""
        return self._positions.get(coin_name)
    
    def add_strategy(
        self,
        coin_name: str,
        strategy: BaseStrategy
    ):
        """Add a strategy to a coin's decider."""
        if coin_name in self._deciders:
            self._deciders[coin_name].strategies.append(strategy)
            logger.info(f"Added strategy {strategy.name} to {coin_name}")
    
    def get_strategy_info(self, coin_name: str) -> List[Dict]:
        """Get info about strategies for a coin."""
        if coin_name not in self._deciders:
            return []
        
        return [s.get_info() for s in self._deciders[coin_name].strategies]
