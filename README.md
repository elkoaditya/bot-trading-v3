# Bybit Trading Bot

Bot trading crypto otomatis dengan multi-strategi untuk Bybit Exchange.

## Fitur Utama

- **Koneksi API Bybit**: Support Demo dan Real Market
- **Multi-Strategy System**: RSI, MACD, Bollinger Bands, EMA Crossover, Breakout
- **Real-time WebSocket**: Update data candle secara real-time
- **Risk Management**: Stop-loss, take-profit, position sizing, drawdown limit
- **Order Execution**: Market dan limit orders dengan retry mechanism
- **Database Logging**: SQLite untuk menyimpan history trade
- **Telegram Notifications**: Alert untuk trade dan daily summary

## Struktur Project

```
BE/
├── src/
│   ├── core/              # Bybit API client, data fetcher, websocket
│   ├── strategies/        # Trading strategies (RSI, MACD, Bollinger, etc)
│   ├── trading/           # Decision engine, order executor, risk manager
│   ├── database/          # SQLite database models
│   ├── notifications/     # Telegram notifier
│   ├── config/            # Configuration loader
│   └── utils/             # Logger, helpers
├── config/
│   └── bot_config.json    # Bot configuration
├── tests/                 # Unit tests
├── logs/                  # Log files
├── main.py                # Entry point
├── requirements.txt       # Dependencies
└── .env.example           # Environment variables template
```

## Instalasi

### 1. Clone repository
```bash
cd /Users/elko/My\ Work/Bot\ Trading\ NEW/BE
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup environment variables
```bash
cp .env.example .env
```

Edit `.env` dan isi dengan API keys:
```env
# Demo/Testnet
BYBIT_DEMO_API_KEY=your_demo_api_key
BYBIT_DEMO_API_SECRET=your_demo_api_secret

# Real/Mainnet
BYBIT_REAL_API_KEY=your_real_api_key
BYBIT_REAL_API_SECRET=your_real_api_secret

# Environment: demo atau real
TRADING_ENVIRONMENT=demo

# Telegram (opsional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Konfigurasi

Edit `config/bot_config.json` untuk menyesuaikan:

- **Global Settings**: Capital, leverage, timeframe, trading fee
- **Coin Configuration**: Symbol, strategies, risk management per coin
- **Entry/Exit Conditions**: Kondisi untuk buka dan tutup posisi

### Contoh Konfigurasi Coin

```json
{
  "BTC": {
    "symbol": "BTCUSDT",
    "enabled": true,
    "strategy": {
      "type": "multi",
      "mode": "majority",
      "strategies": [
        {"name": "rsi", "params": {"period": 14, "oversold": 30, "overbought": 70}},
        {"name": "macd", "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9}}
      ]
    },
    "risk_management": {
      "max_position_size": 0.1,
      "stop_loss_pct": 2.0,
      "take_profit_pct": 4.0,
      "max_daily_trades": 5
    }
  }
}
```

## Menjalankan Bot

### Mode Normal
```bash
python main.py
```

### Dengan Custom Config
```bash
python main.py --config path/to/config.json
```

### Background Mode
```bash
nohup python main.py > logs/bot.log 2>&1 &
```

## Strategies

### 1. RSI (Relative Strength Index)
- Buy ketika RSI crosses above oversold level (default: 30)
- Sell ketika RSI crosses below overbought level (default: 70)

### 2. MACD
- Buy ketika MACD line crosses above signal line
- Sell ketika MACD line crosses below signal line

### 3. EMA Crossover
- Buy ketika fast EMA crosses above slow EMA
- Sell ketika fast EMA crosses below slow EMA

### 4. Bollinger Bands
- Buy ketika price bounces from lower band
- Sell ketika price pulls back from upper band

### 5. Breakout
- Buy ketika price breaks above resistance
- Sell ketika price breaks below support

## Risk Management

- **Position Sizing**: Berdasarkan risk per trade dan stop loss
- **Stop Loss**: Automatic stop loss berdasarkan persentase
- **Take Profit**: Automatic take profit berdasarkan persentase
- **Max Drawdown**: Limit maksimum drawdown harian
- **Daily Trade Limit**: Maksimum jumlah trade per hari

## Logs

Log files disimpan di folder `logs/`:
- `trading_bot_YYYYMMDD.log` - General bot logs

## Database

Trade history disimpan di SQLite:
- `data/trading.db`

Export trades:
```python
from src.database.models import Database
db = Database()
await db.connect()
await db.export_trades_csv("trades_export.csv")
```

## API Reference

### BybitClient
```python
from src.core.bybit_client import BybitClient

client = BybitClient(api_key="...", api_secret="...", is_demo=True)
balance = client.get_wallet_balance()
klines = client.get_kline(symbol="BTCUSDT", interval="5", limit=200)
```

### DataFetcher
```python
from src.core.data_fetcher import DataFetcher

fetcher = DataFetcher(client)
df = fetcher.get_ohlcv("BTCUSDT", "5m", limit=200)
```

### Strategies
```python
from src.strategies.mean_reversion import RSIStrategy

strategy = RSIStrategy(params={"period": 14, "oversold": 30, "overbought": 70})
signal = strategy.generate_signal(df)
```

## Testing

```bash
python -m pytest tests/ -v
```

## Troubleshooting

### Connection Error
- Pastikan API key dan secret sudah benar
- Cek koneksi internet
- Pastikan environment (demo/real) sesuai dengan API key

### No Trades
- Cek apakah coin enabled di config
- Cek log untuk signal yang dihasilkan
- Pastikan balance mencukupi

### Rate Limit
- Bot sudah include retry mechanism
- Jika masih error, kurangi jumlah coin yang dimonitor

## License

MIT License
