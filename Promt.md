Judul: Sistem Backend Bot Trading Crypto dengan Multi-Strategi untuk Bybit

Deskripsi: Buatkan sistem backend untuk bot trading crypto yang menggunakan API Bybit dengan spesifikasi sebagai berikut:

Fitur Inti:

Koneksi API Bybit:

Implementasi menggunakan library pybit atau ccxt

Dukungan untuk dua environment: Real Market dan Mainnet Demo

Sistem otentikasi dengan API Key dan Secret untuk kedua environment

Auto-detection environment berdasarkan konfigurasi

Manajemen Data Candle:

Endpoint untuk fetch data candle OHLCV dari Bybit

Support multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d)

Sistem caching data untuk mengurangi API calls

Real-time websocket connection untuk data terbaru

Sistem Multi-Strategi:

Konfigurasi strategi melalui file YAML/JSON

Support multiple strategy patterns:

Trend Following (MA Crossover, MACD)

Mean Reversion (RSI, Bollinger Bands)

Breakout strategies

Sistem weighting untuk kombinasi strategi

Parameter optimization melalui config

Engine Trading:

Decision making system untuk entry/exit positions

Auto leverage management (10x sesuai konfigurasi)

Support Long dan Short positions

Risk management dengan stop-loss dan take-profit

Position sizing berdasarkan balance

Order Management:

Market dan Limit orders

Order validation sebelum eksekusi

Error handling untuk failed orders

Retry mechanism dengan exponential backoff

Persyaratan Teknis:

Python 3.9+

Async/await untuk performance

Database untuk logging trades (SQLite/PostgreSQL)

Logging system dengan rotation

Unit tests untuk core functions

Error notification (Telegram/Email)

Fitur Tambahan:

Health check endpoint

Performance metrics calculation

Trade history export

Config hot reload

Backtesting simulation mode

Security Considerations:

API keys encryption

Rate limiting implementation

Input validation

Secure credential storage

Deliverables:

Working backend system

Documentation API

Deployment instructions

Example configuration files