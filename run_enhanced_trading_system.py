#!/usr/bin/env python3
"""
Enhanced Neural Trading System - Main Orchestrator
=================================================

Complete system that combines:
1. MT5 historical data collection and training
2. Continuous neural network learning
3. Enhanced live trading with frequent trades
4. Real-time performance monitoring
5. Self-improvement mechanisms

Run this script to start the entire system.
"""

import sys
import os
import argparse
import threading
import time
from datetime import datetime
from pathlib import Path
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all components
from mt5_neural_training_system import main as train_initial_model
from continuous_learning_loop import AdaptiveLearningLoop, TrainingConfig
from enhanced_live_trading_bot import EnhancedLiveTradingBot, TradingMode

class EnhancedTradingOrchestrator:
    """Main orchestrator for the entire enhanced trading system"""
    
    def __init__(self, mode: str = "demo", enable_learning: bool = True):
        self.mode = TradingMode.DEMO if mode.lower() == "demo" else TradingMode.LIVE
        self.enable_learning = enable_learning
        
        # Initialize components
        self.config = TrainingConfig()
        self.learning_system = None
        self.trading_bot = None
        self.is_running = False
        
        # Performance tracking
        self.system_stats = {
            'start_time': None,
            'training_cycles': 0,
            'trades_executed': 0,
            'system_uptime': 0,
            'last_update': None
        }
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('enhanced_trading_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_system(self) -> bool:
        """Initialize the entire trading system"""
        self.logger.info("🚀 Initializing Enhanced Neural Trading System")
        
        try:
            # Step 1: Initial model training
            if not self._perform_initial_training():
                self.logger.error("❌ Initial training failed")
                return False
            
            # Step 2: Initialize trading bot
            if not self._initialize_trading_bot():
                self.logger.error("❌ Trading bot initialization failed")
                return False
            
            # Step 3: Initialize learning system
            if self.enable_learning:
                if not self._initialize_learning_system():
                    self.logger.error("❌ Learning system initialization failed")
                    return False
            
            self.system_stats['start_time'] = datetime.now()
            self.logger.info("✅ System initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            return False
    
    def _perform_initial_training(self) -> bool:
        """Perform initial neural network training"""
        self.logger.info("🧠 Performing initial neural network training...")
        
        try:
            # Run the training system
            success = train_initial_model()
            
            if success:
                self.logger.info("✅ Initial training completed successfully")
                self.system_stats['training_cycles'] += 1
                return True
            else:
                self.logger.error("❌ Initial training failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Training error: {e}")
            return False
    
    def _initialize_trading_bot(self) -> bool:
        """Initialize the enhanced trading bot"""
        self.logger.info("🤖 Initializing enhanced trading bot...")
        
        try:
            self.trading_bot = EnhancedLiveTradingBot(trading_mode=self.mode)
            
            if not self.trading_bot.initialize():
                return False
            
            self.logger.info(f"✅ Trading bot initialized in {self.mode.value} mode")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Trading bot initialization error: {e}")
            return False
    
    def _initialize_learning_system(self) -> bool:
        """Initialize the continuous learning system"""
        self.logger.info("🔄 Initializing continuous learning system...")
        
        try:
            self.learning_system = AdaptiveLearningLoop(self.config)
            self.logger.info("✅ Learning system initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Learning system initialization error: {e}")
            return False
    
    def start_system(self) -> bool:
        """Start the entire enhanced trading system"""
        if not self.initialize_system():
            return False
        
        self.logger.info("🎯 Starting Enhanced Neural Trading System")
        self.logger.info("=" * 60)
        
        try:
            self.is_running = True
            
            # Start learning system (in background)
            if self.learning_system:
                learning_thread = threading.Thread(
                    target=self._run_learning_system,
                    daemon=True
                )
                learning_thread.start()
                self.logger.info("🔄 Continuous learning started in background")
            
            # Start trading bot (main thread)
            if self.trading_bot:
                if self.trading_bot.start_trading():
                    self._run_monitoring_loop()
                else:
                    self.logger.error("❌ Failed to start trading bot")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System startup error: {e}")
            return False
    
    def _run_learning_system(self):
        """Run the continuous learning system in background"""
        try:
            self.learning_system.start_continuous_learning()
            
            # Keep learning system running
            while self.is_running:
                time.sleep(60)
                
        except Exception as e:
            self.logger.error(f"❌ Learning system error: {e}")
    
    def _run_monitoring_loop(self):
        """Main monitoring and status loop"""
        self.logger.info("📊 Starting system monitoring...")
        
        try:
            while self.is_running:
                # Update system stats
                self._update_system_stats()
                
                # Print status every 5 minutes
                if int(time.time()) % 300 == 0:
                    self._print_system_status()
                
                # Check for system health
                if not self._check_system_health():
                    self.logger.warning("⚠️  System health check failed")
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Received stop signal")
        except Exception as e:
            self.logger.error(f"❌ Monitoring loop error: {e}")
        finally:
            self._shutdown_system()
    
    def _update_system_stats(self):
        """Update system performance statistics"""
        try:
            # Get trading bot stats
            if self.trading_bot and hasattr(self.trading_bot, 'risk_manager'):
                win_rate = self.trading_bot.risk_manager.get_win_rate()
                daily_pnl = self.trading_bot.risk_manager.daily_stats['total_pnl']
                trades_count = self.trading_bot.risk_manager.daily_stats['trades_count']
                
                self.system_stats['trades_executed'] = trades_count
                self.system_stats['current_win_rate'] = win_rate
                self.system_stats['daily_pnl'] = daily_pnl
            
            # Get learning system stats
            if self.learning_system:
                perf = self.learning_system.get_current_performance()
                self.system_stats['model_version'] = perf.get('model_version', 0)
                self.system_stats['best_model'] = perf.get('best_model_version', 0)
            
            self.system_stats['last_update'] = datetime.now()
            
            # Calculate uptime
            if self.system_stats['start_time']:
                uptime = datetime.now() - self.system_stats['start_time']
                self.system_stats['system_uptime'] = uptime.total_seconds()
                
        except Exception as e:
            self.logger.error(f"❌ Stats update error: {e}")
    
    def _print_system_status(self):
        """Print comprehensive system status"""
        print(f"\n{'='*60}")
        print(f"🕐 System Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Uptime
        if self.system_stats['system_uptime'] > 0:
            hours = int(self.system_stats['system_uptime'] // 3600)
            minutes = int((self.system_stats['system_uptime'] % 3600) // 60)
            print(f"⏰ Uptime: {hours}h {minutes}m")
        
        # Trading Performance
        if hasattr(self, 'current_win_rate'):
            print(f"🎯 Win Rate: {self.system_stats.get('current_win_rate', 0):.1%}")
            print(f"💰 Daily P&L: ${self.system_stats.get('daily_pnl', 0):.2f}")
            print(f"📈 Trades Today: {self.system_stats.get('trades_executed', 0)}")
        
        # Learning System
        if self.learning_system:
            print(f"🧠 Learning Active: {self.learning_system.is_learning}")
            print(f"🔄 Model Version: {self.system_stats.get('model_version', 0)}")
            print(f"🏆 Best Model: v{self.system_stats.get('best_model', 0)}")
        
        # Neural Network Status
        if self.trading_bot and self.trading_bot.neural_predictor:
            nn_status = "Active" if self.trading_bot.neural_predictor.is_trained else "Inactive"
            print(f"🤖 Neural Network: {nn_status}")
        
        print(f"{'='*60}")
    
    def _check_system_health(self) -> bool:
        """Check overall system health"""
        try:
            # Check MT5 connection
            if self.trading_bot and not self.trading_bot.mt5_initialized:
                self.logger.warning("⚠️  MT5 connection lost")
                return False
            
            # Check if trading bot is responsive
            if self.trading_bot and not self.trading_bot.is_running:
                self.logger.warning("⚠️  Trading bot stopped")
                return False
            
            # Check learning system
            if self.learning_system and not self.learning_system.is_learning:
                self.logger.warning("⚠️  Learning system stopped")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Health check error: {e}")
            return False
    
    def _shutdown_system(self):
        """Gracefully shutdown the system"""
        self.logger.info("🛑 Shutting down Enhanced Trading System...")
        
        self.is_running = False
        
        # Stop trading bot
        if self.trading_bot:
            self.trading_bot.stop_trading()
        
        # Stop learning system
        if self.learning_system:
            self.learning_system.stop_continuous_learning()
        
        # Final stats
        self._print_final_statistics()
        
        self.logger.info("✅ System shutdown completed")
    
    def _print_final_statistics(self):
        """Print final system statistics"""
        print(f"\n{'='*60}")
        print(f"📊 Final System Statistics")
        print(f"{'='*60}")
        
        if self.system_stats['start_time']:
            uptime = datetime.now() - self.system_stats['start_time']
            hours = int(uptime.total_seconds() // 3600)
            print(f"⏰ Total Uptime: {hours}h")
        
        print(f"🧠 Training Cycles: {self.system_stats['training_cycles']}")
        print(f"📈 Total Trades: {self.system_stats['trades_executed']}")
        
        if hasattr(self, 'current_win_rate'):
            print(f"🎯 Final Win Rate: {self.system_stats.get('current_win_rate', 0):.1%}")
            print(f"💰 Final P&L: ${self.system_stats.get('daily_pnl', 0):.2f}")
        
        print(f"{'='*60}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Neural Trading System")
    parser.add_argument(
        "--mode", 
        choices=["demo", "live"], 
        default="demo",
        help="Trading mode: demo or live"
    )
    parser.add_argument(
        "--no-learning", 
        action="store_true",
        help="Disable continuous learning system"
    )
    parser.add_argument(
        "--quick-start", 
        action="store_true",
        help="Quick start without initial training"
    )
    
    args = parser.parse_args()
    
    print("🚀 Enhanced Neural Trading System")
    print("=" * 60)
    print("This system will:")
    print("1. 📊 Collect MT5 historical data")
    print("2. 🧠 Train advanced neural networks")
    print("3. 🔄 Continuously learn and improve")
    print("4. 🤖 Execute frequent, profitable trades")
    print("5. 📈 Minimize losses through AI")
    print("6. 🎯 Target 82%+ win rate")
    print("\nMode:", args.mode.upper())
    print("Learning:", "Enabled" if not args.no_learning else "Disabled")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = EnhancedTradingOrchestrator(
        mode=args.mode,
        enable_learning=not args.no_learning
    )
    
    try:
        # Start the system
        if orchestrator.start_system():
            print("\n🎉 Enhanced Trading System started successfully!")
            print("📊 Monitoring system performance...")
            print("🛑 Press Ctrl+C to stop")
            
            # Keep main thread alive
            while orchestrator.is_running:
                time.sleep(1)
                
        else:
            print("\n❌ Failed to start Enhanced Trading System")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
