"""
Logger utility with rotating file handler.
Provides consistent logging across the application.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Optional


# Global logger cache
_loggers: dict = {}


def setup_logger(
    name: str = "trading_bot",
    log_level: str = "INFO",
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup and configure a logger with rotating file handler.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep
        console_output: Whether to also output to console
        
    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers = []  # Clear existing handlers
    
    # Log format
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Detailed format for file logging
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Rotating file handler
    log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Cache the logger
    _loggers[name] = logger
    
    return logger


def get_logger(name: str = "trading_bot") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


class TradingLogger:
    """
    Specialized logger for trading operations with context.
    """
    
    def __init__(self, symbol: Optional[str] = None):
        self.logger = get_logger("trading_bot")
        self.symbol = symbol
    
    def _format_message(self, message: str) -> str:
        """Add symbol context to message if available."""
        if self.symbol:
            return f"[{self.symbol}] {message}"
        return message
    
    def debug(self, message: str):
        self.logger.debug(self._format_message(message))
    
    def info(self, message: str):
        self.logger.info(self._format_message(message))
    
    def warning(self, message: str):
        self.logger.warning(self._format_message(message))
    
    def error(self, message: str):
        self.logger.error(self._format_message(message))
    
    def critical(self, message: str):
        self.logger.critical(self._format_message(message))
    
    def trade(self, action: str, price: float, quantity: float, side: str):
        """Log a trade execution."""
        self.info(
            f"TRADE {action.upper()} | Side: {side} | "
            f"Price: {price:.8f} | Qty: {quantity:.8f}"
        )
    
    def signal(self, signal_type: str, strategy: str, details: str = ""):
        """Log a trading signal."""
        self.info(f"SIGNAL {signal_type} | Strategy: {strategy} | {details}")
    
    def position(self, action: str, size: float, entry_price: float):
        """Log position changes."""
        self.info(
            f"POSITION {action.upper()} | Size: {size:.8f} | Entry: {entry_price:.8f}"
        )
