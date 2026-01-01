"""
Risk Manager for position sizing and risk control.
Implements position sizing, drawdown limits, and trade limits.
"""
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, date
from collections import defaultdict

from ..config.config_loader import RiskManagement, GlobalSettings
from ..utils.logger import get_logger
from ..utils.helpers import calculate_position_size, calculate_pnl


logger = get_logger("risk_manager")


@dataclass
class PositionInfo:
    """Position information."""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    leverage: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: datetime = None
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized PnL."""
        if self.side == 'long':
            return (self.current_price - self.entry_price) * self.quantity * self.leverage
        else:
            return (self.entry_price - self.current_price) * self.quantity * self.leverage
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized PnL percentage."""
        notional = self.entry_price * self.quantity
        if notional == 0:
            return 0.0
        return (self.unrealized_pnl / notional) * 100
    
    @property
    def is_profitable(self) -> bool:
        """Check if position is profitable."""
        return self.unrealized_pnl > 0


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0


class RiskManager:
    """
    Risk manager for controlling position sizes and risk.
    """
    
    def __init__(
        self,
        global_settings: GlobalSettings,
        coin_risk_configs: Dict[str, RiskManagement] = None
    ):
        """
        Initialize risk manager.
        
        Args:
            global_settings: Global trading settings
            coin_risk_configs: Per-coin risk management configs
        """
        self.global_settings = global_settings
        self.coin_risk_configs = coin_risk_configs or {}
        
        # State tracking
        # Balance starts at 0, will be updated from exchange
        self._current_balance: float = 0.0
        self._peak_balance: float = 0.0
        self._initial_balance: float = 0.0  # Track initial balance for return calculation
        self._positions: Dict[str, PositionInfo] = {}
        self._daily_stats: Dict[date, DailyStats] = defaultdict(lambda: DailyStats(date.today()))
        self._trade_history: List[Dict] = []
        
        logger.info("Risk manager initialized (balance will be fetched from exchange)")
    
    def get_risk_config(self, symbol: str) -> RiskManagement:
        """Get risk config for a symbol."""
        # Extract coin name from symbol
        coin = symbol.replace('USDT', '').replace('PERP', '')
        return self.coin_risk_configs.get(coin, RiskManagement())
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        side: str = 'long'
    ) -> float:
        """
        Calculate appropriate position size based on risk rules.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            side: 'long' or 'short'
            
        Returns:
            Position size in base currency
        """
        risk_config = self.get_risk_config(symbol)
        
        # Risk per trade (base on account)
        risk_amount = self._current_balance * (risk_config.stop_loss_pct / 100)
        
        # Risk per unit
        if side == 'long':
            risk_per_unit = entry_price - stop_loss_price
        else:
            risk_per_unit = stop_loss_price - entry_price
        
        if risk_per_unit <= 0:
            logger.warning(f"Invalid risk per unit for {symbol}")
            return 0.0
        
        # Position size based on risk
        position_size = (risk_amount / risk_per_unit) * self.global_settings.leverage
        
        # Apply maximum position size limit
        max_position_value = self._current_balance * risk_config.max_position_size * self.global_settings.leverage
        max_position_size = max_position_value / entry_price
        
        final_size = min(position_size, max_position_size)
        
        logger.info(
            f"Position size for {symbol}: {final_size:.6f} "
            f"(risk-based: {position_size:.6f}, max: {max_position_size:.6f})"
        )
        
        return final_size
    
    def can_open_position(self, symbol: str) -> tuple:
        """
        Check if a new position can be opened.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (can_open, reason)
        """
        risk_config = self.get_risk_config(symbol)
        today = date.today()
        daily_stats = self._daily_stats[today]
        
        # Check daily trade limit
        if daily_stats.trades_count >= risk_config.max_daily_trades:
            return False, f"Daily trade limit reached ({risk_config.max_daily_trades})"
        
        # Check existing position
        if symbol in self._positions:
            return False, f"Already have position in {symbol}"
        
        # Check drawdown limit
        drawdown = self._calculate_drawdown()
        if drawdown >= risk_config.max_drawdown_limit:
            return False, f"Max drawdown limit reached ({drawdown:.2f}%)"
        
        # Check available margin
        total_exposure = self._calculate_total_exposure()
        max_exposure = self._current_balance * self.global_settings.leverage
        
        if total_exposure >= max_exposure * 0.9:
            return False, "Near maximum exposure limit"
        
        return True, "OK"
    
    def open_position(self, position: PositionInfo):
        """
        Register a new position.
        
        Args:
            position: Position information
        """
        self._positions[position.symbol] = position
        
        # Update daily stats
        today = date.today()
        self._daily_stats[today].trades_count += 1
        
        logger.info(
            f"Position opened: {position.symbol} {position.side} "
            f"qty={position.quantity:.6f} @ {position.entry_price:.2f}"
        )
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str = ""
    ) -> Optional[Dict]:
        """
        Close a position and record PnL.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            exit_reason: Reason for closing
            
        Returns:
            Trade result dictionary
        """
        if symbol not in self._positions:
            logger.warning(f"No position to close for {symbol}")
            return None
        
        position = self._positions.pop(symbol)
        
        # Calculate PnL
        pnl_result = calculate_pnl(
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            side=position.side,
            leverage=position.leverage,
            fee_rate=self.global_settings.trading_fee
        )
        
        # Update balance
        self._current_balance += pnl_result['net_pnl']
        
        # Update peak balance
        if self._current_balance > self._peak_balance:
            self._peak_balance = self._current_balance
        
        # Update daily stats
        today = date.today()
        stats = self._daily_stats[today]
        stats.total_pnl += pnl_result['net_pnl']
        
        if pnl_result['net_pnl'] > 0:
            stats.winning_trades += 1
        else:
            stats.losing_trades += 1
        
        # Record trade
        trade_record = {
            'symbol': symbol,
            'side': position.side,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'entry_time': position.entry_time,
            'exit_time': datetime.utcnow(),
            'leverage': position.leverage,
            'pnl': pnl_result['net_pnl'],
            'pnl_pct': pnl_result['pnl_pct'],
            'fees': pnl_result['total_fees'],
            'reason': exit_reason
        }
        
        self._trade_history.append(trade_record)
        
        logger.info(
            f"Position closed: {symbol} | PnL: ${pnl_result['net_pnl']:.2f} "
            f"({pnl_result['pnl_pct']:.2f}%) | Reason: {exit_reason}"
        )
        
        return trade_record
    
    def update_position_price(self, symbol: str, current_price: float):
        """Update current price for a position."""
        if symbol in self._positions:
            self._positions[symbol].current_price = current_price
    
    def check_stop_conditions(self, symbol: str) -> tuple:
        """
        Check if stop loss or take profit is triggered.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (should_close, reason)
        """
        if symbol not in self._positions:
            return False, ""
        
        position = self._positions[symbol]
        current_price = position.current_price
        
        # Check stop loss
        if position.stop_loss:
            if position.side == 'long' and current_price <= position.stop_loss:
                return True, "Stop loss triggered"
            elif position.side == 'short' and current_price >= position.stop_loss:
                return True, "Stop loss triggered"
        
        # Check take profit
        if position.take_profit:
            if position.side == 'long' and current_price >= position.take_profit:
                return True, "Take profit triggered"
            elif position.side == 'short' and current_price <= position.take_profit:
                return True, "Take profit triggered"
        
        # Check max drawdown for position
        if position.unrealized_pnl_pct < -10:  # -10% threshold
            return True, "Position drawdown limit"
        
        return False, ""
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown percentage."""
        if self._peak_balance == 0:
            return 0.0
        
        return ((self._peak_balance - self._current_balance) / self._peak_balance) * 100
    
    def _calculate_total_exposure(self) -> float:
        """Calculate total exposure across all positions."""
        total = 0.0
        for position in self._positions.values():
            notional = position.entry_price * position.quantity
            total += notional
        return total
    
    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position for a symbol."""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, PositionInfo]:
        """Get all open positions."""
        return self._positions.copy()
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized PnL."""
        return sum(p.unrealized_pnl for p in self._positions.values())
    
    def get_balance(self) -> float:
        """Get current balance."""
        return self._current_balance
    
    def get_equity(self) -> float:
        """Get current equity (balance + unrealized PnL)."""
        return self._current_balance + self.get_unrealized_pnl()
    
    def update_balance(self, new_balance: float):
        """Update current balance from exchange."""
        # Set initial balance on first update
        if self._initial_balance == 0:
            self._initial_balance = new_balance
            logger.info(f"Initial balance set from exchange: ${new_balance:,.2f}")
        
        self._current_balance = new_balance
        if new_balance > self._peak_balance:
            self._peak_balance = new_balance
    
    def get_daily_stats(self, target_date: date = None) -> DailyStats:
        """Get daily statistics."""
        target_date = target_date or date.today()
        return self._daily_stats.get(target_date, DailyStats(target_date))
    
    def get_performance_metrics(self) -> Dict:
        """Calculate overall performance metrics."""
        if not self._trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'current_balance': self._current_balance,
                'return_pct': 0.0
            }
        
        wins = [t for t in self._trade_history if t['pnl'] > 0]
        losses = [t for t in self._trade_history if t['pnl'] <= 0]
        
        total_wins = sum(t['pnl'] for t in wins) if wins else 0
        total_losses = abs(sum(t['pnl'] for t in losses)) if losses else 0
        
        return {
            'total_trades': len(self._trade_history),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(self._trade_history) * 100 if self._trade_history else 0,
            'total_pnl': sum(t['pnl'] for t in self._trade_history),
            'avg_win': total_wins / len(wins) if wins else 0,
            'avg_loss': total_losses / len(losses) if losses else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else total_wins,
            'max_drawdown': self._calculate_drawdown(),
            'current_balance': self._current_balance,
            'initial_balance': self._initial_balance,
            'return_pct': ((self._current_balance - self._initial_balance) 
                          / self._initial_balance * 100) if self._initial_balance > 0 else 0
        }
    
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get recent trade history."""
        return self._trade_history[-limit:]
    
    def set_stop_loss(
        self,
        symbol: str,
        stop_loss: float
    ):
        """Set stop loss for a position."""
        if symbol in self._positions:
            self._positions[symbol].stop_loss = stop_loss
            logger.info(f"Stop loss set for {symbol}: {stop_loss}")
    
    def set_take_profit(
        self,
        symbol: str,
        take_profit: float
    ):
        """Set take profit for a position."""
        if symbol in self._positions:
            self._positions[symbol].take_profit = take_profit
            logger.info(f"Take profit set for {symbol}: {take_profit}")
    
    def update_trailing_stop(
        self,
        symbol: str,
        trail_pct: float = 2.0
    ):
        """
        Update trailing stop for a position.
        
        Args:
            symbol: Trading symbol
            trail_pct: Trail percentage
        """
        if symbol not in self._positions:
            return
        
        position = self._positions[symbol]
        current_price = position.current_price
        
        if position.side == 'long':
            new_stop = current_price * (1 - trail_pct / 100)
            if position.stop_loss is None or new_stop > position.stop_loss:
                position.stop_loss = new_stop
                logger.info(f"Trailing stop updated for {symbol}: {new_stop:.2f}")
        else:
            new_stop = current_price * (1 + trail_pct / 100)
            if position.stop_loss is None or new_stop < position.stop_loss:
                position.stop_loss = new_stop
                logger.info(f"Trailing stop updated for {symbol}: {new_stop:.2f}")
