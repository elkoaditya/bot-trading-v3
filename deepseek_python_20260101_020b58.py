project/
├── src/
│   ├── core/
│   │   ├── bybit_client.py
│   │   ├── data_fetcher.py
│   │   └── websocket_manager.py
│   ├── strategies/
│   │   ├── base_strategy.py
│   │   ├── trend_following.py
│   │   ├── mean_reversion.py
│   │   └── breakout.py
│   ├── trading/
│   │   ├── decision_engine.py
│   │   ├── order_executor.py
│   │   └── risk_manager.py
│   ├── config/
│   │   ├── config_loader.py
│   │   └── strategies.yaml
│   └── utils/
│       ├── logger.py
│       └── helpers.py
├── tests/
├── requirements.txt
└── main.py