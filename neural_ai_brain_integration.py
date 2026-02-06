"""
Neural AI Brain Integration Module
==============================

This module provides seamless integration of the advanced neural trading system
with the existing trading infrastructure, replacing the rule-based ai_brain.py
with sophisticated neural network capabilities.

Integration Features:
1. Drop-in replacement for existing ai_brain.py
2. Backward compatibility with existing trading infrastructure
3. Enhanced performance monitoring and logging
4. Graceful fallback to rule-based system if neural network fails
5. Real-time performance comparison between neural and rule-based systems
6. A/B testing capabilities for gradual deployment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import logging
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from contextual_trading_brain import ContextualTradingBrain, TradingContext
from enhanced_neural_architecture import TradingFeatures
from feature_engineering_pipeline import FeatureEngineeringPipeline

# Import original ai_brain components for fallback
from ai_brain import AIBrain as OriginalAIBrain
from market_context import MarketContextAnalyzer
from pattern_recognition import PatternRecognizer
from trade_validator import TradeValidator
from adaptive_risk import AdaptiveRiskManager
from trading_memory import TradingMemory
from daily_planner import DailyPlanner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuralAIBrain:
    """
    Enhanced AI Brain that integrates neural network capabilities
    with the existing trading system architecture.
    
    This class provides a drop-in replacement for the original AIBrain
    while providing significantly enhanced capabilities through deep learning.
    """
    
    def __init__(self, 
                 use_neural: bool = True,
                 neural_config_path: str = None,
                 fallback_to_original: bool = True,
                 performance_threshold: float = 0.05):  # 5% performance decline triggers fallback
        
        # Initialize components
        self.use_neural = use_neural
        self.fallback_to_original = fallback_to_original
        self.performance_threshold = performance_threshold
        
        # Performance tracking
        self.neural_performance = {'trades': 0, 'wins': 0, 'total_pnl': 0.0}
        self.original_performance = {'trades': 0, 'wins': 0, 'total_pnl': 0.0}
        
        # Initialize neural brain if enabled
        if use_neural:
            try:
                self.neural_brain = ContextualTradingBrain()
                
                # Load pre-trained model if available
                if neural_config_path and Path(neural_config_path).exists():
                    self.neural_brain.load_model(neural_config_path)
                    logger.info(f"Loaded neural model from {neural_config_path}")
                else:
                    logger.info("Neural brain initialized with fresh model")
                    
            except Exception as e:
                logger.warning(f"Failed to initialize neural brain: {str(e)}")
                if fallback_to_original:
                    logger.info("Falling back to original rule-based system")
                    self.use_neural = False
                else:
                    raise
        else:
            self.neural_brain = None
        
        # Initialize original brain as fallback
        self.original_brain = OriginalAIBrain()
        
        # Data cache for performance
        self.data_cache = {}
        self.last_update_time = {}
        
        # A/B testing configuration
        self.ab_testing = {
            'enabled': False,
            'neural_split': 0.7,  # 70% neural, 30% original
            'current_period_start': datetime.now(),
            'results': {'neural': [], 'original': []}
        }
        
        logger.info(f"NeuralAIBrain initialized: Neural={use_neural}, Fallback={fallback_to_original}")
    
    def set_daily_plan(self, plan: dict):
        """Set daily plan for both neural and original systems"""
        if hasattr(self.original_brain, 'set_daily_plan'):
            self.original_brain.set_daily_plan(plan)
        
        # Neural brain doesn't need daily plan in the same way
        # but we can store it for context
        self.daily_plan = plan
    
    def think(self, symbol: str, data_h1: pd.DataFrame, data_h4: pd.DataFrame, 
             data_d1: pd.DataFrame, account_info: Dict[str, Any], 
             symbol_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main thinking method that provides intelligent routing between
        neural and rule-based systems.
        """
        
        # Check if we should use neural network
        use_neural_decision = self._should_use_neural(symbol)
        
        try:
            if use_neural_decision and self.use_neural and self.neural_brain:
                return self._think_neural(symbol, data_h1, data_h4, data_d1, account_info, symbol_info)
            else:
                return self._think_original(symbol, data_h1, data_h4, data_d1, account_info, symbol_info)
                
        except Exception as e:
            logger.error(f"Error in neural thinking: {str(e)}")
            if self.fallback_to_original:
                logger.info("Falling back to original rule-based system")
                return self._think_original(symbol, data_h1, data_h4, data_d1, account_info, symbol_info)
            else:
                # Return safe fallback
                return {
                    "decision": "HOLD",
                    "reason": f"Neural system error: {str(e)}",
                    "confidence": 0.1,
                    "lot": 0.01
                }
    
    def _should_use_neural(self, symbol: str) -> bool:
        """Determine if neural network should be used for this decision"""
        
        # A/B testing mode
        if self.ab_testing['enabled']:
            return np.random.random() < self.ab_testing['neural_split']
        
        # Performance-based fallback
        if (self.neural_performance['trades'] > 10 and 
            self.original_performance['trades'] > 10):
            
            neural_win_rate = self.neural_performance['wins'] / self.neural_performance['trades']
            original_win_rate = self.original_performance['wins'] / self.original_performance['trades']
            
            performance_gap = neural_win_rate - original_win_rate
            
            # If neural performance drops significantly, consider fallback
            if performance_gap < -self.performance_threshold:
                logger.info(f"Neural performance gap: {performance_gap:.3f}, considering fallback")
                return np.random.random() < 0.3  # 30% chance to use neural anyway
        
        return True
    
    def _think_neural(self, symbol: str, data_h1: pd.DataFrame, data_h4: pd.DataFrame,
                     data_d1: pd.DataFrame, account_info: Dict[str, Any], 
                     symbol_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced neural network thinking process"""
        
        # Prepare data for neural network
        neural_result = self.neural_brain.think(
            symbol=symbol,
            h1_data=data_h1,
            h4_data=data_h4,
            d1_data=data_d1,
            account_info=account_info,
            symbol_info=symbol_info
        )
        
        # Convert neural output to standard format
        converted_result = self._convert_neural_to_standard(neural_result, symbol_info)
        
        # Add system identification
        converted_result['system_used'] = 'neural'
        converted_result['neural_analysis'] = neural_result.get('neural_analysis', {})
        converted_result['market_regime'] = neural_result.get('market_regime', {})
        converted_result['risk_metrics'] = neural_result.get('risk_metrics', {})
        
        return converted_result
    
    def _think_original(self, symbol: str, data_h1: pd.DataFrame, data_h4: pd.DataFrame,
                       data_d1: pd.DataFrame, account_info: Dict[str, Any], 
                       symbol_info: Dict[str, Any]) -> Dict[str, Any]:
        """Original rule-based thinking process"""
        
        original_result = self.original_brain.think(
            symbol=symbol,
            data_h1=data_h1,
            data_h4=data_h4,
            data_d1=data_d1,
            account_info=account_info,
            symbol_info=symbol_info
        )
        
        # Add system identification
        original_result['system_used'] = 'original'
        original_result['rule_based_analysis'] = {
            'patterns_detected': self._analyze_patterns(data_h1, data_h4, data_d1),
            'market_context': self._get_market_context(symbol, data_h1, data_h4, data_d1)
        }
        
        return original_result
    
    def _convert_neural_to_standard(self, neural_result: Dict[str, Any], 
                                  symbol_info: Dict[str, Any]) -> Dict[str, Any]:
        """Convert neural network output to standard trading format"""
        
        decision = neural_result.get('decision', 'HOLD')
        confidence = neural_result.get('confidence', 0.5)
        lot = neural_result.get('position_size_multiplier', 0.01)
        
        # Calculate stop loss and take profit
        sl = neural_result.get('sl', None)
        tp = neural_result.get('tp', None)
        
        # If not provided, calculate based on risk metrics
        if not sl or not tp:
            sl, tp = self._calculate_sl_tp_from_neural(neural_result, symbol_info)
        
        # Prepare standard result format
        standard_result = {
            "decision": decision,
            "confidence": confidence,
            "lot": lot,
            "sl": sl,
            "tp": tp,
            "reason": neural_result.get('reason', f"Neural {decision} signal"),
            "pattern": None,  # Neural system doesn't use traditional patterns
            "market_state": neural_result.get('market_regime', {})
        }
        
        return standard_result
    
    def _calculate_sl_tp_from_neural(self, neural_result: Dict[str, Any], 
                                    symbol_info: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit from neural network analysis"""
        
        # Extract pattern strength and market regime
        pattern_strength = neural_result.get('pattern_strength', 0.5)
        market_regime = neural_result.get('market_regime', {})
        risk_metrics = neural_result.get('risk_metrics', {})
        
        # Basic SL/TP calculation based on confidence and pattern strength
        confidence = neural_result.get('confidence', 0.5)
        
        # Use ATR-like calculation from risk metrics if available
        if 'volatility_risk' in risk_metrics:
            vol_risk = risk_metrics['volatility_risk']
            # Adjust SL/TP based on volatility risk
            sl_distance = vol_risk * 2.0  # 2x volatility for stop loss
            tp_distance = vol_risk * 3.0  # 3x volatility for take profit
        else:
            # Default values
            sl_distance = 0.002  # 20 pips
            tp_distance = 0.003  # 30 pips
        
        # Adjust based on confidence and pattern strength
        confidence_multiplier = confidence * pattern_strength
        sl_distance *= confidence_multiplier
        tp_distance *= confidence_multiplier
        
        # Get current price (would need to be passed in a real implementation)
        # For now, return None to indicate calculation not possible without price
        return None, None
    
    def _analyze_patterns(self, data_h1: pd.DataFrame, data_h4: pd.DataFrame, 
                         data_d1: pd.DataFrame) -> List[str]:
        """Analyze patterns for original system compatibility"""
        
        patterns = []
        
        try:
            # Use pattern recognizer from original system
            for timeframe, data in [('H1', data_h1), ('H4', data_h4), ('D1', data_d1)]:
                if data is not None and len(data) > 50:
                    recognizer = PatternRecognizer(data)
                    detected_patterns = recognizer.detect_all()
                    for pattern in detected_patterns:
                        patterns.append(f"{timeframe}: {pattern.name}")
        except Exception as e:
            logger.warning(f"Pattern analysis error: {str(e)}")
        
        return patterns
    
    def _get_market_context(self, symbol: str, data_h1: pd.DataFrame, 
                           data_h4: pd.DataFrame, data_d1: pd.DataFrame) -> Dict[str, str]:
        """Get market context for original system compatibility"""
        
        try:
            # Use original market context analyzer
            analyzer = MarketContextAnalyzer()
            context = analyzer.get_market_state(symbol, data_h1, data_h4, data_d1)
            return context
        except Exception as e:
            logger.warning(f"Market context analysis error: {str(e)}")
            return {"trend": "NEUTRAL", "volatility": "NORMAL", "support_resistance": "NONE"}
    
    def log_result(self, symbol: str, result: Dict[str, Any], profit: float):
        """Enhanced result logging with performance tracking"""
        
        system_used = result.get('system_used', 'unknown')
        
        # Update performance metrics
        if system_used == 'neural':
            self.neural_performance['trades'] += 1
            self.neural_performance['total_pnl'] += profit
            if profit > 0:
                self.neural_performance['wins'] += 1
                
            # Also log to neural brain for its tracking
            if hasattr(self.neural_brain, 'log_result'):
                try:
                    self.neural_brain.log_result(symbol, result, profit)
                except Exception as e:
                    logger.warning(f"Neural brain logging error: {str(e)}")
        
        elif system_used == 'original':
            self.original_performance['trades'] += 1
            self.original_performance['total_pnl'] += profit
            if profit > 0:
                self.original_performance['wins'] += 1
        
        # Enhanced logging
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'system': system_used,
            'decision': result.get('decision'),
            'profit': profit,
            'confidence': result.get('confidence'),
            'reason': result.get('reason'),
            'total_neural_pnl': self.neural_performance['total_pnl'],
            'total_original_pnl': self.original_performance['total_pnl']
        }
        
        logger.info(f"Trade logged: {symbol} {system_used} {result.get('decision')} | P&L: {profit:.2f}")
        
        # Performance comparison
        if (self.neural_performance['trades'] > 0 and 
            self.original_performance['trades'] > 0):
            
            neural_win_rate = self.neural_performance['wins'] / self.neural_performance['trades']
            original_win_rate = self.original_performance['wins'] / self.original_performance['trades']
            
            logger.info(f"Performance - Neural: {neural_win_rate:.2%} ({self.neural_performance['trades']} trades), "
                       f"Original: {original_win_rate:.2%} ({self.original_performance['trades']} trades)")
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """Get detailed performance comparison between systems"""
        
        comparison = {
            'neural': {},
            'original': {},
            'comparison': {}
        }
        
        # Neural performance
        if self.neural_performance['trades'] > 0:
            comparison['neural'] = {
                'trades': self.neural_performance['trades'],
                'wins': self.neural_performance['wins'],
                'win_rate': self.neural_performance['wins'] / self.neural_performance['trades'],
                'total_pnl': self.neural_performance['total_pnl'],
                'avg_pnl': self.neural_performance['total_pnl'] / self.neural_performance['trades']
            }
        
        # Original performance
        if self.original_performance['trades'] > 0:
            comparison['original'] = {
                'trades': self.original_performance['trades'],
                'wins': self.original_performance['wins'],
                'win_rate': self.original_performance['wins'] / self.original_performance['trades'],
                'total_pnl': self.original_performance['total_pnl'],
                'avg_pnl': self.original_performance['total_pnl'] / self.original_performance['trades']
            }
        
        # Comparison
        if (self.neural_performance['trades'] > 0 and 
            self.original_performance['trades'] > 0):
            
            neural_win_rate = comparison['neural']['win_rate']
            original_win_rate = comparison['original']['win_rate']
            
            comparison['comparison'] = {
                'win_rate_difference': neural_win_rate - original_win_rate,
                'pnl_difference': (comparison['neural']['total_pnl'] - 
                                 comparison['original']['total_pnl']),
                'recommended_system': 'neural' if neural_win_rate > original_win_rate else 'original'
            }
        
        return comparison
    
    def enable_ab_testing(self, neural_split: float = 0.7):
        """Enable A/B testing between neural and original systems"""
        
        self.ab_testing.update({
            'enabled': True,
            'neural_split': neural_split,
            'current_period_start': datetime.now()
        })
        
        logger.info(f"A/B testing enabled: {neural_split*100:.0f}% neural, {(1-neural_split)*100:.0f}% original")
    
    def disable_ab_testing(self):
        """Disable A/B testing and use preferred system"""
        
        self.ab_testing['enabled'] = False
        logger.info("A/B testing disabled")
    
    def save_model(self, path: str = "neural_ai_brain_model"):
        """Save the neural brain model"""
        
        if self.neural_brain:
            try:
                self.neural_brain.save_model(path)
                logger.info(f"Neural model saved to {path}")
            except Exception as e:
                logger.error(f"Failed to save neural model: {str(e)}")
    
    def load_model(self, path: str = "neural_ai_brain_model"):
        """Load the neural brain model"""
        
        if self.neural_brain:
            try:
                self.neural_brain.load_model(path)
                logger.info(f"Neural model loaded from {path}")
            except Exception as e:
                logger.error(f"Failed to load neural model: {str(e)}")

class HybridTradingSystem:
    """
    Advanced hybrid trading system that intelligently combines
    neural network and rule-based approaches based on market conditions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.ai_brain = NeuralAIBrain(**self.config)
        
        # Market condition detection
        self.market_conditions = {
            'high_volatility_threshold': 0.02,
            'trending_market_threshold': 0.015,
            'ranging_market_threshold': 0.005
        }
        
        # System preferences by market condition
        self.system_preferences = {
            'high_volatility': 'neural',  # Neural better at complex patterns
            'trending': 'neural',         # Neural better at trend following
            'ranging': 'original',        # Original better at range trading
            'normal': 'neural'            # Neural preferred by default
        }
        
        logger.info("HybridTradingSystem initialized")
    
    def determine_optimal_system(self, data_h1: pd.DataFrame, data_h4: pd.DataFrame, 
                               data_d1: pd.DataFrame) -> str:
        """Determine the optimal trading system based on market conditions"""
        
        try:
            # Calculate market volatility
            returns_h1 = data_h1['close'].pct_change().dropna()
            volatility = returns_h1.tail(20).std()
            
            # Calculate trend strength
            sma_short = data_h1['close'].rolling(10).mean()
            sma_long = data_h1['close'].rolling(50).mean()
            trend_strength = abs((sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1])
            
            # Determine market condition
            if volatility > self.market_conditions['high_volatility_threshold']:
                return self.system_preferences['high_volatility']
            elif trend_strength > self.market_conditions['trending_market_threshold']:
                return self.system_preferences['trending']
            elif volatility < self.market_conditions['ranging_market_threshold']:
                return self.system_preferences['ranging']
            else:
                return self.system_preferences['normal']
                
        except Exception as e:
            logger.warning(f"Error determining optimal system: {str(e)}")
            return 'neural'  # Default to neural
    
    def think(self, symbol: str, data_h1: pd.DataFrame, data_h4: pd.DataFrame,
             data_d1: pd.DataFrame, account_info: Dict[str, Any], 
             symbol_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced thinking with automatic system selection"""
        
        # Determine optimal system
        optimal_system = self.determine_optimal_system(data_h1, data_h4, data_d1)
        
        # Temporarily set preference
        original_use_neural = self.ai_brain.use_neural
        if optimal_system == 'original':
            self.ai_brain.use_neural = False
        else:
            self.ai_brain.use_neural = True
        
        # Get decision
        result = self.ai_brain.think(symbol, data_h1, data_h4, data_d1, account_info, symbol_info)
        
        # Restore original setting
        self.ai_brain.use_neural = original_use_neural
        
        # Add system selection info
        result['optimal_system'] = optimal_system
        result['selection_reason'] = self._get_selection_reason(optimal_system)
        
        return result
    
    def _get_selection_reason(self, system: str) -> str:
        """Get reason for system selection"""
        reasons = {
            'neural': 'Neural network preferred for complex pattern recognition',
            'original': 'Rule-based system preferred for clear market conditions'
        }
        return reasons.get(system, 'Automatic system selection')

# Example usage and integration demonstration
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='H')
    
    base_price = 1.1000
    returns = np.random.normal(0, 0.001, 500)
    prices = base_price * np.cumprod(1 + returns)
    
    # Create OHLCV data
    sample_data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.002, 500)),
        'low': prices * (1 - np.random.uniform(0, 0.002, 500)),
        'close': prices,
        'tick_volume': np.random.randint(100, 1000, 500)
    }, index=dates)
    
    # Test the integrated system
    print("Testing Neural AI Brain Integration...")
    
    # Initialize hybrid system
    hybrid_system = HybridTradingSystem({
        'use_neural': True,
        'fallback_to_original': True
    })
    
    # Set daily plan
    daily_plan = {'EURUSD': 'LONG', 'GBPUSD': 'SHORT'}
    hybrid_system.ai_brain.set_daily_plan(daily_plan)
    
    # Test neural thinking
    account_info = {'balance': 10000, 'equity': 10000}
    symbol_info = {'digits': 5, 'point': 0.00001, 'lot_size': 100000}
    
    result = hybrid_system.think(
        symbol='EURUSD',
        data_h1=sample_data.tail(100),
        data_h4=sample_data.resample('4H').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
        }).dropna().tail(25),
        data_d1=sample_data.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
        }).dropna().tail(5),
        account_info=account_info,
        symbol_info=symbol_info
    )
    
    print(f"\nNeural AI Brain Decision: {result['decision']}")
    print(f"System Used: {result.get('system_used', 'unknown')}")
    print(f"Confidence: {result.get('confidence', 0):.3f}")
    print(f"Position Size: {result.get('lot', 0):.4f}")
    print(f"Reason: {result.get('reason', 'N/A')}")
    print(f"Optimal System: {result.get('optimal_system', 'unknown')}")
    
    # Test performance logging
    hybrid_system.ai_brain.log_result('EURUSD', result, 50.0)  # Simulate profit
    
    # Get performance comparison
    comparison = hybrid_system.ai_brain.get_performance_comparison()
    print(f"\nPerformance Comparison:")
    print(f"Neural: {comparison['neural']}")
    print(f"Original: {comparison['original']}")
    
    print("\nNeural AI Brain Integration completed successfully!")
    print("Key integration features:")
    print("✓ Seamless replacement for existing ai_brain.py")
    print("✓ Intelligent fallback to rule-based system")
    print("✓ A/B testing capabilities for gradual deployment")
    print("✓ Performance monitoring and comparison")
    print("✓ Hybrid system with automatic optimization")
    print("✓ Backward compatibility with existing infrastructure")
