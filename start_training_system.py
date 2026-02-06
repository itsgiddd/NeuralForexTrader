#!/usr/bin/env python3
"""
Start the Enhanced Trading System - Simplified Version
"""

import sys
import os
import time
import threading
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def start_initial_training():
    """Start the initial training process"""
    print("Starting Enhanced Neural Trading System Training...")
    print("=" * 60)
    
    try:
        # Import and run the training system
        from mt5_neural_training_system import main as train_main
        
        print("Initializing MT5 connection...")
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            print(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        account_info = mt5.account_info()
        if account_info:
            print(f"Connected to account: {account_info.login}")
            print(f"Balance: ${account_info.balance:.2f}")
        else:
            print("Failed to get account info")
            return False
        
        print("\nStarting neural network training on MT5 historical data...")
        success = train_main()
        
        if success:
            print("\nTraining completed successfully!")
            print("Neural model has been trained on 365 days of MT5 data")
            return True
        else:
            print("\nTraining failed!")
            return False
            
    except Exception as e:
        print(f"Training error: {e}")
        return False

def start_continuous_learning():
    """Start the continuous learning system"""
    print("\nStarting continuous learning system...")
    
    try:
        from continuous_learning_loop import AdaptiveLearningLoop, TrainingConfig
        
        config = TrainingConfig()
        learning_system = AdaptiveLearningLoop(config)
        
        print("Continuous learning system initialized")
        print("System will:")
        print("- Collect new MT5 data every 30 minutes")
        print("- Retrain neural network continuously")
        print("- Improve performance over time")
        print("- Deploy better models automatically")
        
        # Start learning in background
        learning_system.start_continuous_learning()
        
        return learning_system
        
    except Exception as e:
        print(f"Continuous learning error: {e}")
        return None

def start_enhanced_trading():
    """Start the enhanced trading bot"""
    print("\nStarting enhanced trading bot...")
    
    try:
        from enhanced_live_trading_bot import EnhancedLiveTradingBot, TradingMode
        
        # Initialize trading bot
        bot = EnhancedLiveTradingBot(trading_mode=TradingMode.DEMO)
        
        if not bot.initialize():
            print("Failed to initialize trading bot")
            return None
        
        print("Enhanced trading bot initialized")
        print("Bot will:")
        print("- Use trained neural models")
        print("- Execute frequent trades (50-100/day)")
        print("- Target 82%+ win rate")
        print("- Implement advanced risk management")
        
        # Start trading
        if bot.start_trading():
            print("Enhanced trading bot started successfully!")
            return bot
        else:
            print("Failed to start trading bot")
            return None
            
    except Exception as e:
        print(f"Trading bot error: {e}")
        return None

def monitor_system(learning_system, trading_bot):
    """Monitor the system performance"""
    print("\nSystem monitoring started...")
    print("=" * 60)
    
    try:
        while True:
            print(f"\nSystem Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 40)
            
            # Check learning system
            if learning_system:
                perf = learning_system.get_current_performance()
                print(f"Learning Active: {perf.get('learning_active', False)}")
                print(f"Model Version: {perf.get('model_version', 0)}")
                print(f"Best Model: v{perf.get('best_model_version', 0)}")
            
            # Check trading bot
            if trading_bot and hasattr(trading_bot, 'risk_manager'):
                win_rate = trading_bot.risk_manager.get_win_rate()
                daily_pnl = trading_bot.risk_manager.daily_stats['total_pnl']
                trades_count = trading_bot.risk_manager.daily_stats['trades_count']
                
                print(f"Current Win Rate: {win_rate:.1%}")
                print(f"Daily P&L: ${daily_pnl:.2f}")
                print(f"Trades Today: {trades_count}")
            
            print("-" * 40)
            print("System is training and trading continuously...")
            print("Press Ctrl+C to stop")
            
            # Wait 5 minutes before next update
            time.sleep(300)
            
    except KeyboardInterrupt:
        print("\nStopping system...")
        
        # Stop all components
        if learning_system:
            learning_system.stop_continuous_learning()
            print("Learning system stopped")
        
        if trading_bot:
            trading_bot.stop_trading()
            print("Trading bot stopped")
        
        print("System stopped successfully")

def main():
    """Main function to start the complete system"""
    print("Enhanced Neural Trading System - Auto Training Mode")
    print("=" * 60)
    print("This system will:")
    print("1. Train neural network on MT5 historical data")
    print("2. Start continuous learning and improvement")
    print("3. Begin enhanced frequent trading")
    print("4. Monitor and optimize performance")
    print("5. Self-improve until optimal performance is achieved")
    print("=" * 60)
    
    # Step 1: Initial training
    print("\nSTEP 1: Initial Neural Network Training")
    if not start_initial_training():
        print("Cannot proceed without successful training")
        return False
    
    # Step 2: Start continuous learning
    print("\nSTEP 2: Starting Continuous Learning")
    learning_system = start_continuous_learning()
    
    # Step 3: Start enhanced trading
    print("\nSTEP 3: Starting Enhanced Trading")
    trading_bot = start_enhanced_trading()
    
    # Step 4: Monitor and optimize
    print("\nSTEP 4: System Monitoring and Optimization")
    print("The system will now:")
    print("- Continue learning from new market data")
    print("- Improve neural network performance")
    print("- Execute frequent profitable trades")
    print("- Monitor and adapt to market changes")
    print("- Self-optimize until 82%+ win rate achieved")
    
    # Start monitoring
    monitor_system(learning_system, trading_bot)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nEnhanced Trading System completed successfully!")
        else:
            print("\nSystem encountered errors")
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
    except Exception as e:
        print(f"\nSystem error: {e}")
