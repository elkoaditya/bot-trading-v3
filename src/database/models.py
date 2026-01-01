"""
Database models for trade logging.
Uses SQLite with async support via aiosqlite.
"""
import aiosqlite
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json

from ..utils.logger import get_logger


logger = get_logger("database")


@dataclass
class Trade:
    """Trade record."""
    id: Optional[int] = None
    symbol: str = ""
    side: str = ""  # 'long' or 'short'
    quantity: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    entry_time: str = ""
    exit_time: str = ""
    leverage: int = 1
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    strategy: str = ""
    reason: str = ""
    created_at: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Position:
    """Position record."""
    id: Optional[int] = None
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    entry_price: float = 0.0
    leverage: int = 1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: str = ""
    status: str = "open"  # 'open', 'closed'
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Signal:
    """Signal log record."""
    id: Optional[int] = None
    symbol: str = ""
    signal_type: str = ""  # 'buy', 'sell', 'hold'
    strategy: str = ""
    confidence: float = 0.0
    price: float = 0.0
    timestamp: str = ""
    executed: bool = False


class Database:
    """
    Async SQLite database for trade logging.
    """
    
    def __init__(self, db_path: str = "data/trading.db"):
        """
        Initialize database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[aiosqlite.Connection] = None
    
    async def connect(self):
        """Connect to database and create tables."""
        self._connection = await aiosqlite.connect(str(self.db_path))
        self._connection.row_factory = aiosqlite.Row
        await self._create_tables()
        logger.info(f"Database connected: {self.db_path}")
    
    async def disconnect(self):
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database disconnected")
    
    async def _create_tables(self):
        """Create database tables."""
        await self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                leverage INTEGER DEFAULT 1,
                pnl REAL DEFAULT 0,
                pnl_pct REAL DEFAULT 0,
                fees REAL DEFAULT 0,
                strategy TEXT,
                reason TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                leverage INTEGER DEFAULT 1,
                stop_loss REAL,
                take_profit REAL,
                entry_time TEXT NOT NULL,
                status TEXT DEFAULT 'open'
            );
            
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                strategy TEXT,
                confidence REAL DEFAULT 0,
                price REAL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                executed INTEGER DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                starting_balance REAL,
                ending_balance REAL,
                total_pnl REAL,
                trades_count INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0
            );
            
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
            CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
            CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
        """)
        await self._connection.commit()
    
    # ==================== Trade Operations ====================
    
    async def insert_trade(self, trade: Trade) -> int:
        """Insert a trade record."""
        cursor = await self._connection.execute("""
            INSERT INTO trades (
                symbol, side, quantity, entry_price, exit_price,
                entry_time, exit_time, leverage, pnl, pnl_pct,
                fees, strategy, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.symbol, trade.side, trade.quantity, trade.entry_price,
            trade.exit_price, trade.entry_time, trade.exit_time,
            trade.leverage, trade.pnl, trade.pnl_pct, trade.fees,
            trade.strategy, trade.reason
        ))
        await self._connection.commit()
        logger.info(f"Trade inserted: {trade.symbol} {trade.side} PnL: {trade.pnl:.2f}")
        return cursor.lastrowid
    
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Trade]:
        """Get trade records."""
        query = "SELECT * FROM trades"
        params = []
        
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY entry_time DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        
        return [Trade(**dict(row)) for row in rows]
    
    async def get_trade_by_id(self, trade_id: int) -> Optional[Trade]:
        """Get a trade by ID."""
        cursor = await self._connection.execute(
            "SELECT * FROM trades WHERE id = ?", (trade_id,)
        )
        row = await cursor.fetchone()
        return Trade(**dict(row)) if row else None
    
    async def get_trades_summary(self) -> Dict:
        """Get trades summary statistics."""
        cursor = await self._connection.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl) as total_pnl,
                AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as avg_win,
                AVG(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as avg_loss
            FROM trades
        """)
        row = await cursor.fetchone()
        
        if row:
            total = row['total_trades'] or 0
            wins = row['winning_trades'] or 0
            return {
                'total_trades': total,
                'winning_trades': wins,
                'losing_trades': row['losing_trades'] or 0,
                'win_rate': (wins / total * 100) if total > 0 else 0,
                'total_pnl': row['total_pnl'] or 0,
                'avg_win': row['avg_win'] or 0,
                'avg_loss': row['avg_loss'] or 0
            }
        return {}
    
    # ==================== Position Operations ====================
    
    async def insert_position(self, position: Position) -> int:
        """Insert or update a position."""
        cursor = await self._connection.execute("""
            INSERT OR REPLACE INTO positions (
                symbol, side, quantity, entry_price, leverage,
                stop_loss, take_profit, entry_time, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            position.symbol, position.side, position.quantity,
            position.entry_price, position.leverage, position.stop_loss,
            position.take_profit, position.entry_time, position.status
        ))
        await self._connection.commit()
        return cursor.lastrowid
    
    async def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        cursor = await self._connection.execute(
            "SELECT * FROM positions WHERE status = 'open'"
        )
        rows = await cursor.fetchall()
        return [Position(**dict(row)) for row in rows]
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        cursor = await self._connection.execute(
            "SELECT * FROM positions WHERE symbol = ?", (symbol,)
        )
        row = await cursor.fetchone()
        return Position(**dict(row)) if row else None
    
    async def update_position(self, symbol: str, **kwargs):
        """Update position fields."""
        if not kwargs:
            return
        
        set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [symbol]
        
        await self._connection.execute(
            f"UPDATE positions SET {set_clause} WHERE symbol = ?",
            values
        )
        await self._connection.commit()
    
    async def close_position(self, symbol: str):
        """Mark position as closed."""
        await self._connection.execute(
            "UPDATE positions SET status = 'closed' WHERE symbol = ?",
            (symbol,)
        )
        await self._connection.commit()
    
    async def delete_position(self, symbol: str):
        """Delete a position record."""
        await self._connection.execute(
            "DELETE FROM positions WHERE symbol = ?", (symbol,)
        )
        await self._connection.commit()
    
    # ==================== Signal Operations ====================
    
    async def insert_signal(self, signal: Signal) -> int:
        """Insert a signal record."""
        cursor = await self._connection.execute("""
            INSERT INTO signals (
                symbol, signal_type, strategy, confidence, price, executed
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            signal.symbol, signal.signal_type, signal.strategy,
            signal.confidence, signal.price, signal.executed
        ))
        await self._connection.commit()
        return cursor.lastrowid
    
    async def get_recent_signals(
        self,
        symbol: Optional[str] = None,
        limit: int = 50
    ) -> List[Signal]:
        """Get recent signals."""
        query = "SELECT * FROM signals"
        params = []
        
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        
        return [Signal(**dict(row)) for row in rows]
    
    # ==================== Performance Operations ====================
    
    async def save_daily_performance(
        self,
        date: str,
        starting_balance: float,
        ending_balance: float,
        total_pnl: float,
        trades_count: int,
        win_rate: float,
        max_drawdown: float
    ):
        """Save daily performance record."""
        await self._connection.execute("""
            INSERT OR REPLACE INTO performance (
                date, starting_balance, ending_balance, total_pnl,
                trades_count, win_rate, max_drawdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            date, starting_balance, ending_balance, total_pnl,
            trades_count, win_rate, max_drawdown
        ))
        await self._connection.commit()
    
    async def get_performance_history(self, days: int = 30) -> List[Dict]:
        """Get performance history."""
        cursor = await self._connection.execute("""
            SELECT * FROM performance 
            ORDER BY date DESC 
            LIMIT ?
        """, (days,))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    # ==================== Export Operations ====================
    
    async def export_trades_csv(self, filepath: str):
        """Export trades to CSV."""
        import csv
        
        trades = await self.get_trades(limit=10000)
        
        with open(filepath, 'w', newline='') as f:
            if trades:
                writer = csv.DictWriter(f, fieldnames=trades[0].to_dict().keys())
                writer.writeheader()
                for trade in trades:
                    writer.writerow(trade.to_dict())
        
        logger.info(f"Exported {len(trades)} trades to {filepath}")
    
    async def export_trades_json(self, filepath: str):
        """Export trades to JSON."""
        trades = await self.get_trades(limit=10000)
        
        with open(filepath, 'w') as f:
            json.dump([t.to_dict() for t in trades], f, indent=2)
        
        logger.info(f"Exported {len(trades)} trades to {filepath}")
