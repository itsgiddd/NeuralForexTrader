# 🚀 LIVE NEURAL TRADING BOT CONFIGURATION
# ======================================
# 
# Configure your trading bot settings here
# 

# Trading Mode: demo, live, backtest
TRADING_MODE = "demo"  # CHANGE TO "live" FOR REAL TRADING

# Neural Network Settings
CONFIDENCE_THRESHOLD = 0.78  # Minimum confidence for trades (78%)
MAX_RISK_PER_TRADE = 0.02    # Maximum risk per trade (2%)
MAX_DAILY_RISK = 0.05        # Maximum daily risk (5%)
MAX_DRAWDOWN_LIMIT = 0.15    # Maximum drawdown limit (15%)
MAX_OPEN_POSITIONS = 5        # Maximum simultaneous open positions

# Symbols to trade (add your preferred forex pairs)
SYMBOLS = [
    "EURUSD",
    "GBPUSD", 
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "EURJPY",
    "GBPJPY"
]

# Risk Management Settings
MIN_ACCOUNT_BALANCE = 1000.0   # Minimum account balance to trade
MIN_MARGIN_LEVEL = 200.0      # Minimum margin level required
MAX_SPREAD_PIPS = 3.0         # Maximum spread in pips to trade

# Trading Hours (UTC)
TRADING_HOURS = {
    "start": "00:00",  # Start time (UTC)
    "end": "23:59",    # End time (UTC)
    "avoid_news": True # Avoid trading during major news
}

# Position Sizing
BASE_LOT_SIZE = 0.01          # Base lot size for trades
LOT_SIZE_MULTIPLIER = 1.0    # Multiplier for position sizing
MIN_LOT_SIZE = 0.01           # Minimum lot size
MAX_LOT_SIZE = 1.0            # Maximum lot size

# Neural Network Parameters
NEURAL_SETTINGS = {
    "use_neural": True,           # Enable neural network
    "fallback_to_original": False, # Fallback to original system
    "model_path": "models/",      # Path to trained models
    "retrain_interval": 24,       # Retrain every 24 hours
    "feature_count": 100         # Number of features to use
}

# Logging Settings
LOGGING = {
    "level": "INFO",
    "file": "neural_trading_bot.log",
    "console": True,
    "max_file_size": "10MB",
    "backup_count": 5
}

# Performance Targets
TARGETS = {
    "win_rate": 0.78,           # Target win rate (78%)
    "profit_factor": 2.0,       # Target profit factor
    "max_drawdown": 0.10,       # Maximum drawdown (10%)
    "sharpe_ratio": 1.5,        # Target Sharpe ratio
    "monthly_return": 0.10      # Target monthly return (10%)
}

# Alert Settings (Optional)
ALERTS = {
    "email": False,              # Enable email alerts
    "telegram": False,           # Enable Telegram alerts
    "discord": False,           # Enable Discord alerts
    "trade_notifications": True, # Notify on trades
    "error_notifications": True  # Notify on errors
}

# Safety Settings
SAFETY = {
    "emergency_stop_loss": 0.05,  # Stop trading if 5% loss
    "max_consecutive_losses": 5,  # Stop after 5 consecutive losses
    "kill_switch": False,         # Emergency kill switch
    "demo_mode_first": True       # Force demo mode first
}

# Economic Calendar (Optional)
ECONOMIC_CALENDAR = {
    "avoid_high_impact_news": True,
    "avoid_news_minutes_before": 30,
    "avoid_news_minutes_after": 30,
    "high_impact_events": [
        "Non Farm Payrolls",
        "Federal Funds Rate",
        "ECB Interest Rate Decision",
        "BOE Interest Rate Decision",
        "GDP",
        "Inflation Rate",
        "Employment Rate"
    ]
}

# ======================================
# ⚠️  IMPORTANT SAFETY REMINDERS
# ======================================
# 
# 1. START WITH DEMO MODE FIRST!
# 2. Test thoroughly before live trading
# 3. Use proper risk management
# 4. Monitor performance regularly
# 5. Have a stop-loss plan
# 6. Don't risk more than you can afford to lose
# 
# ======================================