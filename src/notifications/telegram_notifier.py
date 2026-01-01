"""
Telegram Notifier for trade alerts and updates.
"""
import asyncio
from typing import Optional, List
from datetime import datetime, date

from ..utils.logger import get_logger


logger = get_logger("telegram")


class TelegramNotifier:
    """
    Telegram notification service for trading alerts.
    """
    
    def __init__(
        self,
        bot_token: str = "",
        chat_id: str = "",
        enabled: bool = True
    ):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID to send messages
            enabled: Enable/disable notifications
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bool(bot_token and chat_id)
        self._bot = None
        
        if self.enabled:
            try:
                from telegram import Bot
                self._bot = Bot(token=bot_token)
                logger.info("Telegram notifier initialized")
            except ImportError:
                logger.warning("python-telegram-bot not installed, notifications disabled")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
                self.enabled = False
    
    async def send_message(self, message: str, parse_mode: str = "HTML"):
        """
        Send a message to Telegram.
        
        Args:
            message: Message text
            parse_mode: Parse mode (HTML, Markdown)
        """
        if not self.enabled or not self._bot:
            logger.debug(f"Telegram disabled, would send: {message[:50]}...")
            return
        
        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            logger.debug("Telegram message sent")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    async def notify_trade_opened(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        leverage: int,
        stop_loss: float = None,
        take_profit: float = None
    ):
        """Notify about opened trade."""
        emoji = "🟢" if side == "long" else "🔴"
        side_text = "LONG" if side == "long" else "SHORT"
        
        message = f"""
{emoji} <b>Trade Opened</b>

<b>Symbol:</b> {symbol}
<b>Side:</b> {side_text}
<b>Quantity:</b> {quantity:.6f}
<b>Entry Price:</b> ${entry_price:,.2f}
<b>Leverage:</b> {leverage}x
"""
        
        if stop_loss:
            message += f"<b>Stop Loss:</b> ${stop_loss:,.2f}\n"
        if take_profit:
            message += f"<b>Take Profit:</b> ${take_profit:,.2f}\n"
        
        message += f"\n<i>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</i>"
        
        await self.send_message(message)
    
    async def notify_trade_closed(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        reason: str = ""
    ):
        """Notify about closed trade."""
        emoji = "💰" if pnl > 0 else "💔"
        pnl_emoji = "📈" if pnl > 0 else "📉"
        
        message = f"""
{emoji} <b>Trade Closed</b>

<b>Symbol:</b> {symbol}
<b>Side:</b> {"LONG" if side == "long" else "SHORT"}
<b>Quantity:</b> {quantity:.6f}
<b>Entry:</b> ${entry_price:,.2f}
<b>Exit:</b> ${exit_price:,.2f}

{pnl_emoji} <b>PnL:</b> ${pnl:,.2f} ({pnl_pct:+.2f}%)
"""
        
        if reason:
            message += f"\n<b>Reason:</b> {reason}"
        
        message += f"\n\n<i>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</i>"
        
        await self.send_message(message)
    
    async def notify_signal(
        self,
        symbol: str,
        signal_type: str,
        strategy: str,
        confidence: float,
        price: float
    ):
        """Notify about trading signal."""
        emoji = "🔔"
        signal_emoji = "⬆️" if signal_type.lower() in ['buy', 'strong_buy'] else "⬇️"
        
        message = f"""
{emoji} <b>Trading Signal</b>

<b>Symbol:</b> {symbol}
<b>Signal:</b> {signal_emoji} {signal_type.upper()}
<b>Strategy:</b> {strategy}
<b>Confidence:</b> {confidence * 100:.0f}%
<b>Price:</b> ${price:,.2f}

<i>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
"""
        
        await self.send_message(message)
    
    async def notify_error(self, error_message: str, context: str = ""):
        """Notify about an error."""
        message = f"""
⚠️ <b>Bot Error</b>

<b>Context:</b> {context}
<b>Error:</b> {error_message}

<i>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
"""
        
        await self.send_message(message)
    
    async def notify_daily_summary(
        self,
        date_str: str,
        starting_balance: float,
        ending_balance: float,
        total_pnl: float,
        trades_count: int,
        win_rate: float,
        max_drawdown: float
    ):
        """Send daily performance summary."""
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        pnl_sign = "+" if total_pnl >= 0 else ""
        
        message = f"""
📊 <b>Daily Summary - {date_str}</b>

<b>Starting Balance:</b> ${starting_balance:,.2f}
<b>Ending Balance:</b> ${ending_balance:,.2f}

{pnl_emoji} <b>Daily PnL:</b> {pnl_sign}${total_pnl:,.2f}
<b>Total Trades:</b> {trades_count}
<b>Win Rate:</b> {win_rate:.1f}%
<b>Max Drawdown:</b> {max_drawdown:.2f}%

<i>Generated at {datetime.utcnow().strftime('%H:%M:%S')} UTC</i>
"""
        
        await self.send_message(message)
    
    async def notify_bot_started(self, coins: List[str], leverage: int, capital: float):
        """Notify bot has started."""
        coins_text = ", ".join(coins[:5])
        if len(coins) > 5:
            coins_text += f" +{len(coins) - 5} more"
        
        message = f"""
🚀 <b>Trading Bot Started</b>

<b>Coins:</b> {coins_text}
<b>Leverage:</b> {leverage}x
<b>Capital:</b> ${capital:,.2f}

<i>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
"""
        
        await self.send_message(message)
    
    async def notify_bot_stopped(self, reason: str = "Manual stop"):
        """Notify bot has stopped."""
        message = f"""
🛑 <b>Trading Bot Stopped</b>

<b>Reason:</b> {reason}

<i>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
"""
        
        await self.send_message(message)
    
    async def notify_position_update(
        self,
        symbol: str,
        side: str,
        current_price: float,
        entry_price: float,
        unrealized_pnl: float,
        unrealized_pnl_pct: float
    ):
        """Notify about position update."""
        emoji = "📊"
        pnl_emoji = "📈" if unrealized_pnl >= 0 else "📉"
        
        message = f"""
{emoji} <b>Position Update</b>

<b>Symbol:</b> {symbol} {"LONG" if side == "long" else "SHORT"}
<b>Entry:</b> ${entry_price:,.2f}
<b>Current:</b> ${current_price:,.2f}

{pnl_emoji} <b>Unrealized PnL:</b> ${unrealized_pnl:,.2f} ({unrealized_pnl_pct:+.2f}%)
"""
        
        await self.send_message(message)


# Factory function
def create_notifier(
    token: str = "",
    chat_id: str = "",
    enabled: bool = True
) -> TelegramNotifier:
    """Create a Telegram notifier instance."""
    return TelegramNotifier(
        bot_token=token,
        chat_id=chat_id,
        enabled=enabled
    )
