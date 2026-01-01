"""
Base Strategy class and Signal enum.
All trading strategies should inherit from BaseStrategy.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List
import pandas as pd

from ..utils.logger import get_logger


logger = get_logger("strategy")


class Signal(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self._logger = get_logger(f"strategy.{name}")
    
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Generate trading signal from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Signal enum value
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators and add to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        pass
    
    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """
        Get signal strength (0-1).
        Override in subclass for weighted voting.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Signal strength between 0 and 1
        """
        return 0.5
    
    def get_entry_price(self, df: pd.DataFrame, signal: Signal) -> Optional[float]:
        """
        Get suggested entry price.
        
        Args:
            df: DataFrame with OHLCV data
            signal: Generated signal
            
        Returns:
            Suggested entry price or None for market order
        """
        return None  # Default to market order
    
    def get_stop_loss(
        self,
        df: pd.DataFrame,
        signal: Signal,
        entry_price: float
    ) -> Optional[float]:
        """
        Get suggested stop loss price.
        Override in subclass for strategy-specific SL.
        
        Args:
            df: DataFrame with OHLCV data
            signal: Generated signal
            entry_price: Entry price
            
        Returns:
            Stop loss price or None
        """
        return None
    
    def get_take_profit(
        self,
        df: pd.DataFrame,
        signal: Signal,
        entry_price: float
    ) -> Optional[float]:
        """
        Get suggested take profit price.
        Override in subclass for strategy-specific TP.
        
        Args:
            df: DataFrame with OHLCV data
            signal: Generated signal
            entry_price: Entry price
            
        Returns:
            Take profit price or None
        """
        return None
    
    def validate_data(self, df: pd.DataFrame, min_rows: int = 50) -> bool:
        """
        Validate input data.
        
        Args:
            df: DataFrame with OHLCV data
            min_rows: Minimum required rows
            
        Returns:
            True if data is valid
        """
        if df is None or df.empty:
            self._logger.warning("Empty DataFrame provided")
            return False
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            self._logger.warning(f"Missing columns: {missing}")
            return False
        
        if len(df) < min_rows:
            self._logger.warning(f"Insufficient data: {len(df)} rows < {min_rows} required")
            return False
        
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'params': self.params,
            'type': self.__class__.__name__
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"


class MultiStrategyDecider:
    """
    Combines signals from multiple strategies.
    """
    
    def __init__(
        self,
        strategies: List[BaseStrategy],
        mode: str = "majority",
        weights: Optional[List[float]] = None
    ):
        """
        Initialize multi-strategy decider.
        
        Args:
            strategies: List of strategy instances
            mode: Decision mode ('majority', 'weighted', 'all', 'any')
            weights: Optional weights for weighted mode
        """
        self.strategies = strategies
        self.mode = mode
        self.weights = weights or [1.0] * len(strategies)
        
        if len(self.weights) != len(self.strategies):
            raise ValueError("Weights must match number of strategies")
    
    def get_combined_signal(self, df: pd.DataFrame) -> Signal:
        """
        Get combined signal from all strategies.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Combined signal
        """
        signals = []
        
        for strategy in self.strategies:
            try:
                signal = strategy.generate_signal(df)
                signals.append(signal)
            except Exception as e:
                logger.error(f"Error in {strategy.name}: {e}")
                signals.append(Signal.HOLD)
        
        return self._combine_signals(signals)
    
    def _combine_signals(self, signals: List[Signal]) -> Signal:
        """Combine signals based on mode."""
        if self.mode == "majority":
            return self._majority_vote(signals)
        elif self.mode == "weighted":
            return self._weighted_vote(signals)
        elif self.mode == "all":
            return self._all_agree(signals)
        elif self.mode == "any":
            return self._any_signal(signals)
        else:
            return Signal.HOLD
    
    def _majority_vote(self, signals: List[Signal]) -> Signal:
        """Majority voting."""
        buy_count = sum(1 for s in signals if s in [Signal.BUY, Signal.STRONG_BUY])
        sell_count = sum(1 for s in signals if s in [Signal.SELL, Signal.STRONG_SELL])
        
        threshold = len(signals) / 2
        
        if buy_count > threshold:
            return Signal.BUY
        elif sell_count > threshold:
            return Signal.SELL
        else:
            return Signal.HOLD
    
    def _weighted_vote(self, signals: List[Signal]) -> Signal:
        """Weighted voting."""
        buy_weight = 0.0
        sell_weight = 0.0
        
        for signal, weight in zip(signals, self.weights):
            if signal in [Signal.BUY, Signal.STRONG_BUY]:
                buy_weight += weight
            elif signal in [Signal.SELL, Signal.STRONG_SELL]:
                sell_weight += weight
        
        total_weight = sum(self.weights)
        threshold = total_weight / 2
        
        if buy_weight > threshold:
            return Signal.BUY
        elif sell_weight > threshold:
            return Signal.SELL
        else:
            return Signal.HOLD
    
    def _all_agree(self, signals: List[Signal]) -> Signal:
        """All strategies must agree."""
        buy_signals = [Signal.BUY, Signal.STRONG_BUY]
        sell_signals = [Signal.SELL, Signal.STRONG_SELL]
        
        if all(s in buy_signals for s in signals):
            return Signal.BUY
        elif all(s in sell_signals for s in signals):
            return Signal.SELL
        else:
            return Signal.HOLD
    
    def _any_signal(self, signals: List[Signal]) -> Signal:
        """Any buy/sell signal triggers."""
        for signal in signals:
            if signal in [Signal.STRONG_BUY]:
                return Signal.BUY
            elif signal in [Signal.STRONG_SELL]:
                return Signal.SELL
        
        for signal in signals:
            if signal == Signal.BUY:
                return Signal.BUY
            elif signal == Signal.SELL:
                return Signal.SELL
        
        return Signal.HOLD
    
    def get_all_signals(self, df: pd.DataFrame) -> Dict[str, Signal]:
        """Get signals from all strategies."""
        result = {}
        for strategy in self.strategies:
            try:
                result[strategy.name] = strategy.generate_signal(df)
            except Exception as e:
                logger.error(f"Error in {strategy.name}: {e}")
                result[strategy.name] = Signal.HOLD
        return result
