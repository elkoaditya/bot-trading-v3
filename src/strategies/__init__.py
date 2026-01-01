# Strategies module
from .base_strategy import BaseStrategy, Signal
from .trend_following import EMAStrategy, MACDStrategy
from .mean_reversion import RSIStrategy, BollingerStrategy
from .breakout import BreakoutStrategy

__all__ = [
    'BaseStrategy', 'Signal',
    'EMAStrategy', 'MACDStrategy',
    'RSIStrategy', 'BollingerStrategy',
    'BreakoutStrategy'
]
