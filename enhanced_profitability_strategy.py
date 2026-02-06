#!/usr/bin/env python3
"""
Enhanced Profitability Strategy for Neural Trading Bot
Comprehensive improvements for consistent profitability.
"""

import numpy as np
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class PerformanceMetrics:
    """Enhanced performance tracking for consistent profitability."""
    symbol: str
    total_trades: int
    winning_trades: int
    total_profit: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    
class ProfitabilityOptimizer:
    """
    Advanced profitability optimization system for consistent trading profits.
    """
    
    def __init__(self):
        self.performance_history = {}
        self.symbol_weights = {'EURUSD': 0.33, 'GBPUSD': 0.33, 'USDJPY': 0.34}
        self.risk_budget = 0.02  # 2% risk per trade
        self.confidence_threshold = 0.78  # 78% confidence minimum
        
    def calculate_optimal_position_size(self, symbol: str, confidence: float, 
                                    balance: float, volatility: float) -> float:
        """Calculate optimal position size based on multiple factors."""
        
        # Base risk percentage
        base_risk = self.risk_budget
        
        # Adjust for confidence (higher confidence = higher risk)
        confidence_multiplier = min(confidence / 0.90, 1.5)  # Max 1.5x
        
        # Adjust for symbol volatility
        volatility_multiplier = max(0.5, 1.0 - volatility)  # Lower risk for high volatility
        
        # Historical performance adjustment
        symbol_performance = self.get_symbol_performance(symbol)
        performance_multiplier = 1.0 + (symbol_performance.win_rate - 0.5) * 0.5
        
        # Calculate final position size
        adjusted_risk = base_risk * confidence_multiplier * volatility_multiplier * performance_multiplier
        risk_amount = balance * adjusted_risk
        
        # Convert to lot size (simplified calculation)
        lot_size = risk_amount / (1000 * balance / 1000)  # Assuming $1000 per lot risk
        return max(0.01, min(lot_size, 0.5))  # Between 0.01 and 0.5 lots
    
    def get_symbol_performance(self, symbol: str) -> PerformanceMetrics:
        """Get historical performance for a symbol."""
        if symbol not in self.performance_history:
            return PerformanceMetrics(
                symbol=symbol, total_trades=0, winning_trades=0, total_profit=0.0,
                avg_win=0.0, avg_loss=0.0, profit_factor=0.0, max_drawdown=0.0,
                sharpe_ratio=0.0, win_rate=0.0
            )
        return self.performance_history[symbol]
    
    def update_performance_metrics(self, symbol: str, trade_result: Dict):
        """Update performance metrics after each trade."""
        if symbol not in self.performance_history:
            self.performance_history[symbol] = PerformanceMetrics(
                symbol=symbol, total_trades=0, winning_trades=0, total_profit=0.0,
                avg_win=0.0, avg_loss=0.0, profit_factor=0.0, max_drawdown=0.0,
                sharpe_ratio=0.0, win_rate=0.0
            )
        
        metrics = self.performance_history[symbol]
        metrics.total_trades += 1
        
        if trade_result['profit'] > 0:
            metrics.winning_trades += 1
        
        # Update profit tracking
        old_total_profit = metrics.total_profit
        metrics.total_profit += trade_result['profit']
        
        # Update averages
        if trade_result['profit'] > 0:
            metrics.avg_win = (metrics.avg_win * (metrics.winning_trades - 1) + trade_result['profit']) / metrics.winning_trades
        else:
            metrics.avg_loss = (metrics.avg_loss * (metrics.total_trades - metrics.winning_trades - 1) + abs(trade_result['profit'])) / (metrics.total_trades - metrics.winning_trades)
        
        # Calculate win rate
        metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        # Calculate profit factor
        total_wins = metrics.winning_trades * metrics.avg_win if metrics.winning_trades > 0 else 0
        total_losses = (metrics.total_trades - metrics.winning_trades) * metrics.avg_loss if metrics.total_trades > metrics.winning_trades else 0
        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    def get_market_session_filter(self) -> Dict[str, bool]:
        """Filter trades based on optimal market sessions."""
        return {
            'EURUSD': True,  # Best during London/NY overlap
            'GBPUSD': True,  # Best during London session
            'USDJPY': True   # Active during all sessions
        }
    
    def should_trade_symbol(self, symbol: str, confidence: float, 
                          market_conditions: Dict) -> Tuple[bool, str]:
        """Determine if a symbol should be traded based on multiple factors."""
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return False, f"Confidence {confidence:.1%} below threshold {self.confidence_threshold:.1%}"
        
        # Check market session
        session_filter = self.get_market_session_filter()
        if not session_filter.get(symbol, False):
            return False, f"Poor market session for {symbol}"
        
        # Check symbol performance
        performance = self.get_symbol_performance(symbol)
        if performance.total_trades > 10 and performance.win_rate < 0.4:
            return False, f"Poor historical performance: {performance.win_rate:.1%} win rate"
        
        # Check market volatility
        if market_conditions.get('volatility', 0) > 0.03:  # 3% volatility threshold
            return False, "High market volatility detected"
        
        return True, "All checks passed"
    
    def optimize_trading_schedule(self) -> Dict[str, List[str]]:
        """Optimize trading schedule for maximum profitability."""
        return {
            'EURUSD': ['08:00-12:00', '13:00-17:00'],  # London/NY overlap
            'GBPUSD': ['08:00-12:00', '13:00-16:00'],   # London session
            'USDJPY': ['00:00-03:00', '08:00-12:00', '13:00-17:00']  # All sessions
        }

# Advanced Signal Quality Enhancement
class SignalQualityEnhancer:
    """Enhance signal quality for more consistent profits."""
    
    @staticmethod
    def validate_signal_quality(features: Dict, confidence: float, 
                              market_regime: Dict) -> Tuple[float, str]:
        """Validate and enhance signal quality."""
        
        quality_score = confidence
        issues = []
        
        # Check trend consistency
        trend_alignment = market_regime.get('trend_alignment', 1.0)
        if trend_alignment < 0.7:
            quality_score *= 0.8
            issues.append("Weak trend alignment")
        
        # Check volatility regime
        volatility = market_regime.get('volatility', 0.01)
        if volatility > 0.025:  # 2.5% volatility
            quality_score *= 0.9
            issues.append("High volatility")
        
        # Check volume confirmation (if available)
        volume_trend = features.get('volume_trend', 'neutral')
        if volume_trend == 'declining':
            quality_score *= 0.95
            issues.append("Declining volume")
        
        quality_message = "; ".join(issues) if issues else "High quality signal"
        return min(quality_score, 0.95), quality_message
    
    @staticmethod
    def apply_market_filters(features: Dict) -> Dict[str, bool]:
        """Apply additional market filters."""
        filters = {
            'trend_strength': abs(features.get('trend_strength', 0)) > 0.3,
            'momentum': features.get('rsi', 50) > 30 and features.get('rsi', 50) < 70,
            'support_resistance': features.get('near_support', False) or features.get('near_resistance', False),
            'volatility_range': 0.005 < features.get('atr', 0.01) < 0.03
        }
        return filters

# Performance Dashboard Generator
def generate_performance_report(optimizer: ProfitabilityOptimizer) -> Dict:
    """Generate comprehensive performance report."""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'overall_performance': {},
        'symbol_performance': {},
        'recommendations': []
    }
    
    total_profit = sum(p.total_profit for p in optimizer.performance_history.values())
    total_trades = sum(p.total_trades for p in optimizer.performance_history.values())
    winning_trades = sum(p.winning_trades for p in optimizer.performance_history.values())
    
    report['overall_performance'] = {
        'total_profit': total_profit,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'overall_win_rate': winning_trades / total_trades if total_trades > 0 else 0,
        'best_symbol': max(optimizer.performance_history.keys(), 
                          key=lambda x: optimizer.performance_history[x].total_profit) if optimizer.performance_history else None
    }
    
    for symbol, metrics in optimizer.performance_history.items():
        report['symbol_performance'][symbol] = {
            'total_profit': metrics.total_profit,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'total_trades': metrics.total_trades
        }
    
    # Generate recommendations
    if total_trades > 0:
        if winning_trades / total_trades < 0.6:
            report['recommendations'].append("Consider tightening confidence threshold")
        
        if total_profit < 0:
            report['recommendations'].append("Review risk management settings")
        
        best_symbol = report['overall_performance']['best_symbol']
        if best_symbol:
            report['recommendations'].append(f"Focus more on {best_symbol} - best performing symbol")
    
    return report

if __name__ == "__main__":
    # Example usage
    optimizer = ProfitabilityOptimizer()
    
    # Sample trade result
    sample_trade = {
        'profit': 2.50,
        'confidence': 0.85,
        'symbol': 'USDJPY'
    }
    
    optimizer.update_performance_metrics('USDJPY', sample_trade)
    
    # Generate report
    report = generate_performance_report(optimizer)
    print(json.dumps(report, indent=2))
    
    print("\n" + "="*50)
    print("ENHANCED PROFITABILITY OPTIMIZER READY!")
    print("="*50)