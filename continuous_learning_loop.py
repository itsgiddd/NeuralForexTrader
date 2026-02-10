#!/usr/bin/env python3
"""
Continuous Learning and Self-Improvement System
============================================

This system continuously:
1. Collects new MT5 data
2. Retrains the neural network
3. Tests performance improvements
4. Updates the live trading system
5. Monitors performance metrics
6. Adapts to market changes
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import time
import threading
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our training system
from mt5_neural_training_system import (
    MT5DataCollector, 
    AdvancedFeatureEngineer, 
    ContinuousLearningSystem,
    TradingOptimizer,
    TrainingConfig
)

class PerformanceMonitor:
    """Real-time performance monitoring and adaptation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.performance_history = []
        self.adaptation_threshold = 0.05  # 5% performance drop triggers adaptation
        
    def record_performance(self, win_rate: float, daily_pnl: float, 
                         trades_count: int, max_drawdown: float):
        """Record trading performance metrics"""
        performance = {
            'timestamp': datetime.now(),
            'win_rate': win_rate,
            'daily_pnl': daily_pnl,
            'trades_count': trades_count,
            'max_drawdown': max_drawdown,
            'timestamp_unix': time.time()
        }
        
        self.performance_history.append(performance)
        
        # Keep only last 30 days of data
        cutoff_time = datetime.now() - timedelta(days=30)
        self.performance_history = [
            p for p in self.performance_history 
            if p['timestamp'] > cutoff_time
        ]
        
        # Check if adaptation is needed
        if self._needs_adaptation():
            print(f"⚠️  Performance degradation detected - triggering adaptation")
            return True
        
        return False
    
    def _needs_adaptation(self) -> bool:
        """Check if the model needs adaptation based on performance"""
        if len(self.performance_history) < 7:  # Need at least a week of data
            return False
        
        # Calculate recent vs historical performance
        recent_performance = np.mean([p['win_rate'] for p in self.performance_history[-7:]])
        historical_performance = np.mean([p['win_rate'] for p in self.performance_history[:-7]])
        
        performance_drop = historical_performance - recent_performance
        
        return performance_drop > self.adaptation_threshold
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get current performance summary"""
        if not self.performance_history:
            return {'win_rate': 0.0, 'avg_daily_pnl': 0.0, 'total_trades': 0}
        
        win_rates = [p['win_rate'] for p in self.performance_history]
        daily_pnls = [p['daily_pnl'] for p in self.performance_history]
        trade_counts = [p['trades_count'] for p in self.performance_history]
        
        return {
            'current_win_rate': np.mean(win_rates),
            'avg_daily_pnl': np.mean(daily_pnls),
            'total_trades': sum(trade_counts),
            'performance_trend': self._calculate_trend(win_rates),
            'consistency_score': 1.0 - np.std(win_rates)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate performance trend (positive = improving)"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope

class ModelVersionManager:
    """Manages different versions of trained models"""
    
    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.current_version = 0
        self.best_model_version = None
        self.best_performance = 0.0
        
    def save_model_version(self, model, metrics: Dict[str, Any], 
                          performance_score: float) -> str:
        """Save a new model version"""
        self.current_version += 1
        
        model_path = self.base_path / f"model_v{self.current_version}.pth"
        metadata = {
            'version': self.current_version,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'performance_score': performance_score
        }
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }, model_path)
        
        # Update best model if this one is better
        if performance_score > self.best_performance:
            self.best_performance = performance_score
            self.best_model_version = self.current_version
            
            # Copy to best model
            best_path = self.base_path / "best_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'metadata': metadata
            }, best_path)
            
            print(f"🏆 New best model: v{self.current_version} (score: {performance_score:.3f})")
        
        print(f"💾 Saved model version v{self.current_version}")
        return str(model_path)
    
    def load_best_model(self):
        """Load the best performing model"""
        best_path = self.base_path / "best_model.pth"
        
        if not best_path.exists():
            print("⚠️  No best model found")
            return None, None
        
        checkpoint = torch.load(best_path)
        return checkpoint['model_state_dict'], checkpoint['metadata']

class LiveTradingIntegration:
    """Integration with live trading system for real-time improvements"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.is_running = False
        self.last_update_time = None
        
    def update_live_model(self, new_model_state_dict: Dict[str, Any]):
        """Update the live trading bot with new model"""
        try:
            # Load the new model state
            if self.model is None:
                # Import the advanced neural network
                from mt5_neural_training_system import AdvancedNeuralNetwork
                
                # Initialize with dummy input dim (will be updated)
                self.model = AdvancedNeuralNetwork(input_dim=100)
            
            # Update model weights
            self.model.load_state_dict(new_model_state_dict)
            self.model.eval()
            
            # Update the live trading bot
            self._update_trading_bot_model()
            
            self.last_update_time = datetime.now()
            print(f"✅ Live trading model updated at {self.last_update_time}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to update live model: {e}")
            return False
    
    def _update_trading_bot_model(self):
        """Update the actual trading bot's model"""
        try:
            # Save the updated model to a file that the trading bot can load
            model_path = "current_neural_model.pth"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'last_update': datetime.now().isoformat()
            }, model_path)
            
            print(f"🔄 Trading bot model updated: {model_path}")
            
        except Exception as e:
            print(f"❌ Failed to update trading bot: {e}")

class AdaptiveLearningLoop:
    """Main continuous learning loop"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_collector = MT5DataCollector()
        self.feature_engineer = AdvancedFeatureEngineer(config)
        self.learning_system = ContinuousLearningSystem(config)
        self.optimizer = TradingOptimizer(config)
        self.performance_monitor = PerformanceMonitor(config)
        self.model_manager = ModelVersionManager()
        self.integration = LiveTradingIntegration(config)
        
        self.is_learning = False
        self.learning_thread = None
        
    def start_continuous_learning(self):
        """Start the continuous learning process"""
        if self.is_learning:
            print("⚠️  Learning loop already running")
            return
        
        self.is_learning = True
        self.learning_thread = threading.Thread(target=self._learning_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()
        
        print("🚀 Continuous learning started!")
    
    def stop_continuous_learning(self):
        """Stop the continuous learning process"""
        self.is_learning = False
        
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        
        print("🛑 Continuous learning stopped")
    
    def _learning_loop(self):
        """Main learning loop that runs continuously"""
        print("🔄 Starting learning loop...")
        
        while self.is_learning:
            try:
                print(f"\n{'='*50}")
                print(f"Learning Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 1. Collect new data
                print("📊 Collecting recent market data...")
                new_data = self.data_collector.collect_historical_data(days_back=30)
                
                if not new_data:
                    print("⚠️  No new data collected, waiting...")
                    time.sleep(3600)  # Wait 1 hour
                    continue
                
                # 2. Create features
                print("🔧 Creating features...")
                features, labels = self.feature_engineer.create_features(new_data)
                
                if len(features) < self.config.min_data_points:
                    print(f"⚠️  Insufficient data: {len(features)} samples")
                    time.sleep(1800)  # Wait 30 minutes
                    continue
                
                # 3. Initialize model if needed
                if self.learning_system.model is None:
                    input_dim = len(features[0])
                    self.learning_system.initialize_model(input_dim)
                
                # 4. Train new model version
                print("🧠 Training new model...")
                training_metrics = self.learning_system.train_model(features, labels)
                
                # 5. Optimize parameters
                print("⚡ Optimizing parameters...")
                optimized_params = self.optimizer.optimize_trading_parameters(
                    self.learning_system.model, features, labels
                )
                
                # 6. Calculate performance score
                performance_score = self._calculate_performance_score(
                    training_metrics, optimized_params
                )
                
                # 7. Save model version
                model_path = self.model_manager.save_model_version(
                    self.learning_system.model, 
                    training_metrics, 
                    performance_score
                )
                
                # 8. Update live trading system
                if self._should_deploy_model(performance_score):
                    print("🚀 Deploying new model to live trading...")
                    self.integration.update_live_model(
                        self.learning_system.model.state_dict()
                    )
                
                # 9. Update performance metrics
                self._record_training_performance(training_metrics, optimized_params)
                
                # 10. Wait before next cycle
                print(f"⏰ Learning cycle completed. Waiting {self.config.epochs_per_cycle * 60} seconds...")
                time.sleep(self.config.epochs_per_cycle * 60)  # Wait based on training time
                
            except Exception as e:
                print(f"❌ Error in learning loop: {e}")
                time.sleep(1800)  # Wait 30 minutes on error
    
    def _calculate_performance_score(self, training_metrics: Dict[str, Any], 
                                   optimized_params: Dict[str, Any]) -> float:
        """Calculate overall performance score for model comparison"""
        # Weighted combination of metrics
        win_rate_score = optimized_params.get('win_rate', 0)
        accuracy_score = training_metrics.get('final_val_accuracy', 0)
        frequency_score = min(optimized_params.get('trades_per_day', 0) / 20, 1.0)  # Normalize to max 20 trades/day
        
        # Weighted average (prioritize win rate)
        performance_score = (
            0.5 * win_rate_score +
            0.3 * accuracy_score +
            0.2 * frequency_score
        )
        
        return performance_score
    
    def _should_deploy_model(self, performance_score: float) -> bool:
        """Determine if model should be deployed to live trading"""
        # Deploy if significantly better than best model
        threshold = self.model_manager.best_performance + 0.02  # 2% improvement
        
        # Only deploy if above minimum threshold
        minimum_threshold = self.config.min_accuracy_threshold
        
        return performance_score > max(threshold, minimum_threshold)
    
    def _record_training_performance(self, training_metrics: Dict[str, Any], 
                                   optimized_params: Dict[str, Any]):
        """Record training performance for monitoring"""
        # This would typically update a database or file
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'training_accuracy': training_metrics.get('final_val_accuracy', 0),
            'win_rate': optimized_params.get('win_rate', 0),
            'trades_per_day': optimized_params.get('trades_per_day', 0),
            'total_samples': training_metrics.get('total_samples', 0)
        }
        
        # Save to file
        with open('training_performance_log.json', 'a') as f:
            json.dump(performance_record, f)
            f.write('\n')
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current system performance"""
        return {
            'learning_active': self.is_learning,
            'model_version': self.model_manager.current_version,
            'best_model_version': self.model_manager.best_model_version,
            'last_update': self.integration.last_update_time,
            'performance_summary': self.performance_monitor.get_performance_summary()
        }

def main():
    """Main function to start continuous learning"""
    print("🚀 Continuous Learning and Self-Improvement System")
    print("=" * 60)
    
    # Initialize configuration
    config = TrainingConfig()
    
    # Initialize learning system
    learning_system = AdaptiveLearningLoop(config)
    
    print("Starting continuous learning system...")
    print("This will:")
    print("1. Collect MT5 historical data")
    print("2. Train neural networks continuously")
    print("3. Improve trading performance")
    print("4. Deploy better models automatically")
    print("5. Monitor and adapt to market changes")
    print("\nPress Ctrl+C to stop...")
    
    try:
        # Start continuous learning
        learning_system.start_continuous_learning()
        
        # Keep main thread alive
        while True:
            time.sleep(60)
            
            # Print status every 10 minutes
            if int(time.time()) % 600 == 0:
                performance = learning_system.get_current_performance()
                print(f"\n📊 System Status:")
                print(f"   Learning Active: {performance['learning_active']}")
                print(f"   Model Version: {performance['model_version']}")
                print(f"   Best Model: v{performance['best_model_version']}")
                print(f"   Last Update: {performance['last_update']}")
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping continuous learning...")
        learning_system.stop_continuous_learning()
        print("✅ System stopped safely")

if __name__ == "__main__":
    main()
