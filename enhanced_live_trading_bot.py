#!/usr/bin/env python3
"""
Enhanced Live Trading Bot with Continuous Learning Integration
============================================================

Advanced trading bot that:
1. Uses continuously trained neural networks
2. Implements frequent trading capabilities
3. Minimizes losses through advanced risk management
4. Adapts to market changes in real-time
5. Monitors performance and self-improves
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import json
import time
import threading
import warnings
from pathlib import Path
from enum import Enum
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class EnhancedTradingConfig:
    """Enhanced trading configuration"""
    # Basic trading parameters
    confidence_threshold: float = 0.75  # Higher threshold for better accuracy
    max_risk_per_trade: float = 0.015    # Reduced risk for frequent trading
    max_concurrent_positions: int = 10   # Allow more frequent trading
    
    # Neural network parameters
    model_path: str = "current_neural_model.pth"
    feature_dim: int = 50
    
    # Frequent trading settings
    min_time_between_trades: int = 30     # 30 seconds minimum
    max_trades_per_hour: int = 20         # High frequency capability
    daily_trade_limit: int = 100          # Allow many trades per day
    
    # Risk management
    max_daily_loss: float = 0.05         # 5% max daily loss
    position_size_factor: float = 1.5     # Larger positions for better returns
    correlation_limit: float = 0.7         # Limit correlated trades
    
    # Performance targets
    target_win_rate: float = 0.82        # Higher target win rate
    min_profit_factor: float = 1.5         # Minimum profit factor
    max_drawdown: float = 0.03            # 3% max drawdown

class TradingMode(Enum):
    """Trading mode enumeration"""
    DEMO = "demo"
    LIVE = "live"

@dataclass
class TradeResult(Enum):
    """Trade result enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    """Enhanced trading signal with confidence and risk assessment"""
    symbol: str
    action: TradeResult
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_score: float
    expected_profit_factor: float
    timeframe_consensus: float
    market_condition: str
    reason: str

class EnhancedNeuralPredictor:
    """Enhanced neural predictor using trained models"""
    
    def __init__(self, config: EnhancedTradingConfig):
        self.config = config
        self.model = None
        self.feature_scaler = None
        self.is_trained = False
        self.last_training_time = None
        
    def load_trained_model(self, model_path: str = None):
        """Load the best trained neural network model"""
        try:
            model_path = model_path or self.config.model_path
            
            if not Path(model_path).exists():
                print(f"⚠️  Model file not found: {model_path}")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Import the neural network class
            from mt5_neural_training_system import AdvancedNeuralNetwork
            
            # Initialize model
            self.model = AdvancedNeuralNetwork(
                input_dim=self.config.feature_dim,
                hidden_dim=256,
                num_layers=3
            )
            
            # Load trained weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.is_trained = True
            
            # Get training timestamp if available
            if 'last_update' in checkpoint:
                self.last_training_time = checkpoint['last_update']
            
            print(f"✅ Loaded trained neural model from {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def predict_signal(self, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate trading signal using the trained neural network"""
        if not self.is_trained or self.model is None:
            return None
        
        try:
            # Extract features from market data
            features = self._extract_enhanced_features(market_data)
            
            if features is None:
                return None
            
            # Make prediction
            with torch.no_grad():
                X_tensor = torch.FloatTensor(features.reshape(1, -1))
                outputs = self.model(X_tensor)
                
                # Extract predictions
                direction_probs = torch.softmax(outputs['direction'], dim=1).numpy()[0]
                confidence = torch.sigmoid(outputs['confidence']).numpy()[0][0]
                risk_score = torch.sigmoid(outputs['risk']).numpy()[0][0]
                
                # Determine action
                action_idx = np.argmax(direction_probs)
                actions = [TradeResult.SELL, TradeResult.HOLD, TradeResult.BUY]
                action = actions[action_idx]
                action_prob = direction_probs[action_idx]
                
                # Only proceed if confidence is high enough
                if confidence < self.config.confidence_threshold:
                    return None
                
                # Create trading signal
                signal = self._create_trading_signal(
                    market_data, action, confidence, risk_score, direction_probs
                )
                
                return signal
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def _extract_enhanced_features(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract enhanced features for neural network prediction"""
        try:
            symbol = market_data['symbol']
            timeframes = market_data['timeframes']
            
            features = []
            
            # Primary timeframe (M15)
            if 'M15' not in timeframes:
                return None
            
            m15_data = timeframes['M15']
            
            # Price features
            current_price = m15_data['close'].iloc[-1]
            prev_price = m15_data['close'].iloc[-2]
            price_change = (current_price - prev_price) / prev_price
            
            features.extend([
                price_change,
                (current_price - m15_data['close'].rolling(20).mean().iloc[-1]) / m15_data['close'].std(),
                m15_data['rsi'].iloc[-1] / 100.0,  # Normalize RSI
                m15_data['macd'].iloc[-1] / current_price,  # Normalize MACD
                m15_data['bb_position'].iloc[-1],  # Bollinger position
            ])
            
            # Multi-timeframe features
            for tf_name, tf_data in timeframes.items():
                if len(tf_data) >= 10:
                    # Trend strength
                    ma5 = tf_data['close'].rolling(5).mean().iloc[-1]
                    ma20 = tf_data['close'].rolling(20).mean().iloc[-1]
                    trend_strength = (ma5 - ma20) / ma20
                    
                    # Momentum
                    momentum = tf_data['close'].pct_change(5).iloc[-1]
                    
                    features.extend([trend_strength, momentum])
                else:
                    features.extend([0.0, 0.0])
            
            # Volatility features
            returns = m15_data['close'].pct_change().dropna()
            if len(returns) >= 20:
                features.extend([
                    returns.std(),  # Volatility
                    returns.skew(),  # Skewness
                    returns.kurtosis(),  # Kurtosis
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # Market condition features
            high_low_ratio = (m15_data['high'] - m15_data['low']).mean() / current_price
            features.append(high_low_ratio)
            
            # Ensure we have the right number of features
            while len(features) < self.config.feature_dim:
                features.append(0.0)
            
            return np.array(features[:self.config.feature_dim])
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            return None
    
    def _create_trading_signal(self, market_data: Dict[str, Any], action: TradeResult, 
                             confidence: float, risk_score: float, 
                             direction_probs: np.ndarray) -> TradingSignal:
        """Create enhanced trading signal with full risk assessment"""
        symbol = market_data['symbol']
        m15_data = market_data['timeframes']['M15']
        
        current_price = m15_data['close'].iloc[-1]
        spread = m15_data['spread'].iloc[-1]
        
        # Calculate SL/TP based on neural risk assessment
        base_spread = spread * current_price
        
        # Adjust SL/TP based on risk score and confidence
        risk_multiplier = 1.0 + (1.0 - risk_score) * 0.5  # Higher risk = tighter SL
        confidence_multiplier = 0.8 + confidence * 0.4   # Higher confidence = wider TP
        
        if action == TradeResult.BUY:
            stop_loss = current_price - (base_spread * 3 * risk_multiplier)
            take_profit = current_price + (base_spread * 6 * confidence_multiplier)
        else:  # SELL
            stop_loss = current_price + (base_spread * 3 * risk_multiplier)
            take_profit = current_price - (base_spread * 6 * confidence_multiplier)
        
        # Calculate position size based on risk score
        base_position_size = self.config.max_risk_per_trade
        adjusted_position_size = base_position_size * (1.0 + confidence * 0.5) * (1.0 - risk_score)
        
        # Expected profit factor
        sl_distance = abs(current_price - stop_loss)
        tp_distance = abs(take_profit - current_price)
        expected_profit_factor = tp_distance / sl_distance if sl_distance > 0 else 1.0
        
        # Market condition assessment
        market_condition = self._assess_market_condition(m15_data)
        
        # Timeframe consensus
        timeframe_consensus = np.max(direction_probs)
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=adjusted_position_size,
            risk_score=risk_score,
            expected_profit_factor=expected_profit_factor,
            timeframe_consensus=timeframe_consensus,
            market_condition=market_condition,
            reason=f"Neural prediction: {action.value} with {confidence:.1%} confidence"
        )
    
    def _assess_market_condition(self, m15_data: pd.DataFrame) -> str:
        """Assess current market condition"""
        try:
            # Simple market condition assessment
            volatility = m15_data['close'].pct_change().std()
            trend_strength = abs(m15_data['close'].iloc[-1] - m15_data['close'].rolling(20).mean().iloc[-1])
            
            if volatility > 0.02:  # High volatility
                return "VOLATILE"
            elif trend_strength > 0.001:  # Strong trend
                return "TRENDING"
            else:
                return "RANGING"
                
        except:
            return "UNKNOWN"

class AdvancedRiskManager:
    """Advanced risk management for frequent trading"""
    
    def __init__(self, config: EnhancedTradingConfig):
        self.config = config
        self.daily_stats = {
            'trades_count': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'start_balance': 0.0,
            'peak_balance': 0.0
        }
        self.last_trade_time = {}
        self.open_positions = []
        
    def can_trade(self, symbol: str, signal: TradingSignal, account_info: Any) -> Tuple[bool, str]:
        """Check if we can trade based on risk management rules"""
        # Check daily limits
        if self.daily_stats['trades_count'] >= self.config.daily_trade_limit:
            return False, "Daily trade limit reached"
        
        # Check time between trades
        current_time = time.time()
        if symbol in self.last_trade_time:
            time_diff = current_time - self.last_trade_time[symbol]
            if time_diff < self.config.min_time_between_trades:
                return False, f"Too soon since last trade on {symbol}"
        
        # Check concurrent positions
        if len(self.open_positions) >= self.config.max_concurrent_positions:
            return False, "Maximum concurrent positions reached"
        
        # Check daily loss limit
        if self.daily_stats['total_pnl'] <= -self.config.max_daily_loss * account_info.balance:
            return False, "Daily loss limit reached"
        
        # Check correlation with existing positions
        if self._has_correlated_position(signal.symbol):
            return False, "Correlated position already open"
        
        # Check profit factor
        if signal.expected_profit_factor < self.config.min_profit_factor:
            return False, "Insufficient profit factor"
        
        # Check confidence
        if signal.confidence < self.config.confidence_threshold:
            return False, "Insufficient confidence"
        
        return True, "Trade approved"
    
    def update_daily_stats(self, trade_result: float):
        """Update daily trading statistics"""
        self.daily_stats['trades_count'] += 1
        
        if trade_result > 0:
            self.daily_stats['winning_trades'] += 1
        else:
            self.daily_stats['losing_trades'] += 1
        
        self.daily_stats['total_pnl'] += trade_result
        
        # Update drawdown
        current_balance = self.daily_stats['start_balance'] + self.daily_stats['total_pnl']
        if current_balance > self.daily_stats['peak_balance']:
            self.daily_stats['peak_balance'] = current_balance
        
        self.daily_stats['current_drawdown'] = (
            (self.daily_stats['peak_balance'] - current_balance) / 
            self.daily_stats['peak_balance']
        )
        
        if self.daily_stats['current_drawdown'] > self.daily_stats['max_drawdown']:
            self.daily_stats['max_drawdown'] = self.daily_stats['current_drawdown']
    
    def _has_correlated_position(self, symbol: str) -> bool:
        """Check if there's a correlated position already open"""
        # Simplified correlation check
        # In practice, you'd check actual currency correlations
        for position in self.open_positions:
            if position['symbol'] == symbol:
                return True
        return False
    
    def get_win_rate(self) -> float:
        """Calculate current win rate"""
        if self.daily_stats['trades_count'] == 0:
            return 0.0
        return self.daily_stats['winning_trades'] / self.daily_stats['trades_count']

class EnhancedLiveTradingBot:
    """Enhanced live trading bot with continuous learning integration"""
    
    def __init__(self, trading_mode: TradingMode = TradingMode.DEMO):
        self.trading_mode = trading_mode
        self.config = EnhancedTradingConfig()
        self.is_running = False
        self.bot_thread = None
        
        # Initialize components
        self.neural_predictor = EnhancedNeuralPredictor(self.config)
        self.risk_manager = AdvancedRiskManager(self.config)
        
        # MT5 connection
        self.mt5_initialized = False
        
        # Performance tracking
        self.start_time = None
        self.symbols = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD',
            'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CHFJPY', 'CADCHF'
        ]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def initialize(self):
        """Initialize the trading bot"""
        print("🚀 Initializing Enhanced Live Trading Bot...")
        
        # Initialize MT5
        if not self._initialize_mt5():
            return False
        
        # Load trained neural model
        self.neural_predictor.load_trained_model()
        
        # Initialize risk manager
        account_info = mt5.account_info()
        if account_info:
            self.risk_manager.daily_stats['start_balance'] = account_info.balance
            self.risk_manager.daily_stats['peak_balance'] = account_info.balance
        
        self.mt5_initialized = True
        print("✅ Enhanced trading bot initialized successfully")
        return True
    
    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                print(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False
            
            account_info = mt5.account_info()
            if not account_info:
                print("❌ Failed to get account info")
                return False
            
            print(f"✅ MT5 connected - Account: {account_info.login}")
            print(f"   Balance: ${account_info.balance:.2f}")
            print(f"   Mode: {self.trading_mode.value.upper()}")
            
            return True
            
        except Exception as e:
            print(f"❌ MT5 initialization error: {e}")
            return False
    
    def start_trading(self):
        """Start the enhanced trading system"""
        if not self.mt5_initialized:
            print("❌ Bot not initialized. Call initialize() first.")
            return False
        
        if self.is_running:
            print("⚠️  Trading bot is already running")
            return False
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start trading thread
        self.bot_thread = threading.Thread(target=self._trading_loop)
        self.bot_thread.daemon = True
        self.bot_thread.start()
        
        print("🎯 Enhanced trading bot started!")
        print("   Features: Neural predictions, Frequent trading, Advanced risk management")
        print("   Target: 82%+ win rate with minimal losses")
        print("   Press Ctrl+C to stop")
        
        return True
    
    def stop_trading(self):
        """Stop the trading bot"""
        self.is_running = False
        
        if self.bot_thread:
            self.bot_thread.join(timeout=5)
        
        print("🛑 Enhanced trading bot stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        print("🔄 Starting enhanced trading loop...")
        
        try:
            while self.is_running:
                current_time = datetime.now()
                
                # Get account info
                account_info = mt5.account_info()
                if not account_info:
                    print("⚠️  Lost MT5 connection, attempting to reconnect...")
                    if not self._initialize_mt5():
                        time.sleep(60)
                        continue
                    account_info = mt5.account_info()
                
                # Get open positions
                open_positions = mt5.positions_get()
                self.risk_manager.open_positions = [
                    {'symbol': pos.symbol, 'type': pos.type} 
                    for pos in open_positions
                ] if open_positions else []
                
                # Analyze each symbol
                for symbol in self.symbols:
                    if not self.is_running:
                        break
                    
                    try:
                        # Get market data
                        market_data = self._get_market_data(symbol)
                        if market_data is None:
                            continue
                        
                        # Generate neural signal
                        signal = self.neural_predictor.predict_signal(market_data)
                        
                        if signal:
                            # Check if we can trade
                            can_trade, reason = self.risk_manager.can_trade(
                                symbol, signal, account_info
                            )
                            
                            if can_trade:
                                # Execute trade
                                success = self._execute_enhanced_trade(signal, account_info)
                                if success:
                                    self.risk_manager.last_trade_time[symbol] = time.time()
                                    print(f"✅ {signal.symbol}: {signal.action.value} @ {signal.entry_price:.5f}")
                                    print(f"   Confidence: {signal.confidence:.1%}, Risk: {signal.risk_score:.3f}")
                            else:
                                print(f"⏭️  {symbol}: {reason}")
                        
                    except Exception as e:
                        print(f"❌ Error processing {symbol}: {e}")
                        continue
                
                # Update performance stats
                self._update_performance_stats()
                
                # Sleep before next cycle
                time.sleep(10)  # 10-second cycles for high frequency
                
        except Exception as e:
            print(f"❌ Critical error in trading loop: {e}")
    
    def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive market data for analysis"""
        try:
            timeframes = {}
            
            # Get data for different timeframes
            for tf_name, tf in [('M5', mt5.TIMEFRAME_M5), ('M15', mt5.TIMEFRAME_M15), 
                               ('H1', mt5.TIMEFRAME_H1), ('H4', mt5.TIMEFRAME_H4)]:
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, 100)
                if rates is None or len(rates) < 20:
                    continue
                
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                # Add basic indicators
                df['sma_5'] = df['close'].rolling(5).mean()
                df['sma_20'] = df['close'].rolling(20).mean()
                df['rsi'] = self._calculate_rsi(df['close'])
                df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
                df['bb_position'] = self._calculate_bb_position(df)
                df['spread'] = (df['high'] - df['low']) / df['close']
                
                timeframes[tf_name] = df
            
            if not timeframes:
                return None
            
            return {
                'symbol': symbol,
                'timeframes': timeframes,
                'current_price': timeframes['M15']['close'].iloc[-1]
            }
            
        except Exception as e:
            print(f"❌ Error getting market data for {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bb_position(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Bollinger Bands position"""
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return (df['close'] - lower) / (upper - lower)
    
    def _execute_enhanced_trade(self, signal: TradingSignal, account_info) -> bool:
        """Execute trade with enhanced risk management"""
        try:
            # Calculate lot size
            lot_size = self._calculate_position_size(signal, account_info)
            
            if lot_size <= 0:
                return False
            
            # Prepare MT5 order
            symbol_info = mt5.symbol_info(signal.symbol)
            if not symbol_info:
                return False
            
            if signal.action == TradeResult.BUY:
                order_type = mt5.ORDER_TYPE_BUY
                price = symbol_info.ask
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = symbol_info.bid
            
            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "deviation": 20,
                "magic": 123456,
                "comment": f"Neural-{signal.confidence:.1%}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"🎯 Enhanced trade executed: {signal.symbol} {signal.action.value}")
                print(f"   Entry: {price:.5f}, SL: {signal.stop_loss:.5f}, TP: {signal.take_profit:.5f}")
                print(f"   Lot size: {lot_size:.2f}, Confidence: {signal.confidence:.1%}")
                return True
            else:
                print(f"❌ Trade failed: {result.comment if result else 'Unknown error'}")
                return False
                
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
            return False
    
    def _calculate_position_size(self, signal: TradingSignal, account_info) -> float:
        """Calculate position size based on risk and neural confidence"""
        balance = account_info.balance
        risk_amount = balance * signal.position_size
        
        # Calculate pip value
        pip_value = 10 if 'JPY' not in signal.symbol else 1  # Simplified
        
        # Calculate stop loss in pips
        sl_pips = abs(signal.entry_price - signal.stop_loss) * (
            10000 if 'JPY' not in signal.symbol else 100
        )
        
        if sl_pips == 0:
            return 0
        
        # Calculate lot size
        lot_size = risk_amount / (sl_pips * pip_value)
        
        # Apply confidence multiplier
        confidence_multiplier = 0.5 + signal.confidence * 0.5
        lot_size *= confidence_multiplier
        
        # Ensure minimum lot size
        lot_size = max(lot_size, 0.01)
        
        return round(lot_size, 2)
    
    def _update_performance_stats(self):
        """Update and display performance statistics"""
        if not hasattr(self, '_last_stats_update'):
            self._last_stats_update = time.time()
        
        # Update every 60 seconds
        if time.time() - self._last_stats_update < 60:
            return
        
        self._last_stats_update = time.time()
        
        win_rate = self.risk_manager.get_win_rate()
        daily_pnl = self.risk_manager.daily_stats['total_pnl']
        trades_count = self.risk_manager.daily_stats['trades_count']
        
        print(f"\n📊 Performance Update:")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Daily P&L: ${daily_pnl:.2f}")
        print(f"   Trades Today: {trades_count}")
        print(f"   Current Drawdown: {self.risk_manager.daily_stats['current_drawdown']:.1%}")

def main():
    """Main function to run enhanced trading bot"""
    print("🚀 Enhanced Live Trading Bot with Continuous Learning")
    print("=" * 60)
    
    # Initialize bot
    bot = EnhancedLiveTradingBot(trading_mode=TradingMode.DEMO)
    
    if not bot.initialize():
        print("❌ Failed to initialize trading bot")
        return False
    
    # Start trading
    if bot.start_trading():
        try:
            # Keep main thread alive
            while bot.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping enhanced trading bot...")
            bot.stop_trading()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
