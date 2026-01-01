"""
Configuration loader with hot-reload support.
Handles JSON/YAML configs and environment variables.
"""
import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv
import hashlib
import asyncio

from ..utils.logger import get_logger


# Load environment variables
load_dotenv()


logger = get_logger("config")


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskManagement:
    """Risk management settings for a coin."""
    max_position_size: float = 0.1
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    max_daily_trades: int = 10
    max_drawdown_limit: float = 10.0


@dataclass
class CoinConfig:
    """Configuration for a single coin."""
    symbol: str
    enabled: bool
    strategies: List[StrategyConfig]
    strategy_mode: str = "majority"  # majority, weighted, all
    risk_management: RiskManagement = field(default_factory=RiskManagement)
    entry_conditions: List[Dict] = field(default_factory=list)
    exit_conditions: List[Dict] = field(default_factory=list)


@dataclass
class GlobalSettings:
    """Global bot settings."""
    initial_capital: float = 10000.0
    trading_fee: float = 0.001
    position_size: float = 0.1
    leverage: int = 10
    timeframe: str = "5m"
    category: str = "linear"
    exchange: str = "bybit"


@dataclass
class BotConfig:
    """Complete bot configuration."""
    global_settings: GlobalSettings
    coins: Dict[str, CoinConfig]
    api_key: str = ""
    api_secret: str = ""
    is_demo: bool = True
    telegram_enabled: bool = False
    telegram_token: str = ""
    telegram_chat_id: str = ""


class ConfigLoader:
    """
    Configuration loader with hot-reload support.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to config file (JSON or YAML)
        """
        self.config_path = Path(config_path)
        self._config: Optional[BotConfig] = None
        self._last_hash: str = ""
        self._watching = False
    
    def load(self) -> BotConfig:
        """
        Load configuration from file.
        
        Returns:
            BotConfig instance
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        # Read file content
        content = self.config_path.read_text(encoding='utf-8')
        
        # Parse based on file extension
        if self.config_path.suffix.lower() in ['.yaml', '.yml']:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        
        # Update hash
        self._last_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Parse configuration
        self._config = self._parse_config(data)
        
        logger.info(f"Configuration loaded from {self.config_path}")
        
        return self._config
    
    def _parse_config(self, data: Dict) -> BotConfig:
        """
        Parse raw config data into BotConfig.
        
        Args:
            data: Raw config dictionary
            
        Returns:
            BotConfig instance
        """
        # Parse global settings
        global_data = data.get('global_settings', {})
        global_settings = GlobalSettings(
            initial_capital=global_data.get('initial_capital', 10000.0),
            trading_fee=global_data.get('trading_fee', 0.001),
            position_size=global_data.get('position_size', 0.1),
            leverage=global_data.get('leverage', 10),
            timeframe=global_data.get('timeframe', '5m'),
            category=global_data.get('category', 'linear'),
            exchange=global_data.get('exchange', 'bybit')
        )
        
        # Parse coins
        coins: Dict[str, CoinConfig] = {}
        coins_data = data.get('coins', {})
        
        for coin_name, coin_data in coins_data.items():
            # Parse strategies
            strategy_data = coin_data.get('strategy', {})
            strategies = []
            
            for s in strategy_data.get('strategies', []):
                strategies.append(StrategyConfig(
                    name=s.get('name', ''),
                    params=s.get('params', {})
                ))
            
            # Parse risk management
            rm_data = coin_data.get('risk_management', {})
            risk_mgmt = RiskManagement(
                max_position_size=rm_data.get('max_position_size', 0.1),
                stop_loss_pct=rm_data.get('stop_loss_pct', 2.0),
                take_profit_pct=rm_data.get('take_profit_pct', 4.0),
                max_daily_trades=rm_data.get('max_daily_trades', 10),
                max_drawdown_limit=rm_data.get('max_drawdown_limit', 10.0)
            )
            
            coins[coin_name] = CoinConfig(
                symbol=coin_data.get('symbol', f"{coin_name}USDT"),
                enabled=coin_data.get('enabled', True),
                strategies=strategies,
                strategy_mode=strategy_data.get('mode', 'majority'),
                risk_management=risk_mgmt,
                entry_conditions=coin_data.get('entry_conditions', []),
                exit_conditions=coin_data.get('exit_conditions', [])
            )
        
        # Get environment based credentials
        is_demo = os.getenv('TRADING_ENVIRONMENT', 'demo').lower() == 'demo'
        
        if is_demo:
            api_key = os.getenv('BYBIT_DEMO_API_KEY', '')
            api_secret = os.getenv('BYBIT_DEMO_API_SECRET', '')
        else:
            api_key = os.getenv('BYBIT_REAL_API_KEY', '')
            api_secret = os.getenv('BYBIT_REAL_API_SECRET', '')
        
        # Telegram settings
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        return BotConfig(
            global_settings=global_settings,
            coins=coins,
            api_key=api_key,
            api_secret=api_secret,
            is_demo=is_demo,
            telegram_enabled=bool(telegram_token and telegram_chat_id),
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id
        )
    
    def has_changed(self) -> bool:
        """
        Check if config file has changed.
        
        Returns:
            True if config has changed
        """
        if not self.config_path.exists():
            return False
        
        content = self.config_path.read_text(encoding='utf-8')
        current_hash = hashlib.md5(content.encode()).hexdigest()
        
        return current_hash != self._last_hash
    
    def reload_if_changed(self) -> Optional[BotConfig]:
        """
        Reload config if file has changed.
        
        Returns:
            New BotConfig if changed, None otherwise
        """
        if self.has_changed():
            logger.info("Configuration file changed, reloading...")
            return self.load()
        return None
    
    async def watch_for_changes(self, callback=None, interval: float = 5.0):
        """
        Watch for config changes and reload automatically.
        
        Args:
            callback: Optional callback function called on reload
            interval: Check interval in seconds
        """
        self._watching = True
        
        while self._watching:
            try:
                new_config = self.reload_if_changed()
                if new_config and callback:
                    await callback(new_config)
            except Exception as e:
                logger.error(f"Error checking config changes: {e}")
            
            await asyncio.sleep(interval)
    
    def stop_watching(self):
        """Stop watching for config changes."""
        self._watching = False
    
    @property
    def config(self) -> Optional[BotConfig]:
        """Get current config."""
        return self._config
    
    def get_enabled_coins(self) -> List[str]:
        """
        Get list of enabled coin names.
        
        Returns:
            List of coin names
        """
        if not self._config:
            return []
        
        return [
            name for name, coin in self._config.coins.items()
            if coin.enabled
        ]
    
    def get_coin_config(self, coin_name: str) -> Optional[CoinConfig]:
        """
        Get configuration for a specific coin.
        
        Args:
            coin_name: Name of the coin (e.g., 'BTC')
            
        Returns:
            CoinConfig or None
        """
        if not self._config:
            return None
        
        return self._config.coins.get(coin_name)
