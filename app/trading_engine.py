#!/usr/bin/env python3
"""
Neural Trading Engine
====================

Professional trading engine that integrates neural network predictions
with MT5 trading operations for automated forex trading.

Features:
- Neural network signal generation
- Automated trade execution
- Risk management
- Position monitoring
- Performance tracking
- Real-time trading loop
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Import app modules
from .mt5_connector import MT5Connector
from .model_manager import NeuralModelManager

class TradingSignal:
    """Trading signal data structure"""
    
    def __init__(self, symbol: str, action: str, confidence: float, 
                 entry_price: float, stop_loss: float, take_profit: float,
                 position_size: float, reason: str):
        self.symbol = symbol
        self.action = action  # 'BUY', 'SELL', 'HOLD'
        self.confidence = confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.reason = reason
        self.timestamp = datetime.now()
        self.executed = False
        self.order_ticket = None

class Position:
    """Position tracking data structure"""
    
    def __init__(self, ticket: int, symbol: str, action: str, 
                 entry_price: float, stop_loss: float, take_profit: float,
                 position_size: float):
        self.ticket = ticket
        self.symbol = symbol
        self.action = action
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.open_time = datetime.now()
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.status = 'OPEN'  # OPEN, CLOSED, PARTIAL

class TradingEngine:
    """Professional neural trading engine"""
    
    def __init__(self, mt5_connector: MT5Connector, model_manager: NeuralModelManager,
                 risk_per_trade: float = 0.015, confidence_threshold: float = 0.65,
                 trading_pairs: List[str] = None, max_concurrent_positions: int = 5):
        
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.mt5_connector = mt5_connector
        self.model_manager = model_manager
        
        # Trading parameters
        self.risk_per_trade = risk_per_trade  # 1.5% default
        self.confidence_threshold = confidence_threshold  # 65% default
        self.trading_pairs = trading_pairs or ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        self.max_concurrent_positions = max_concurrent_positions
        
        # Trading state
        self.is_running = False
        self.trading_thread = None
        self.positions: Dict[int, Position] = {}
        self.signals_history: List[TradingSignal] = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0
        }
        
        # Feature engineering cache
        self.feature_cache = {}
        self.last_update = {}
        
        # Timeframes for analysis
        self.timeframes = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1
        }
        
        # Performance tracking
        self.start_time = None
        self.last_performance_update = None
        
        self.logger.info("Neural Trading Engine initialized")
    
    def start(self):
        """Start the trading engine"""
        if self.is_running:
            self.logger.warning("Trading engine already running")
            return
        
        # Validate prerequisites
        if not self.mt5_connector.is_connected():
            raise Exception("MT5 not connected")
        
        if not self.model_manager.is_model_loaded():
            raise Exception("Neural model not loaded")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start trading thread
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        
        self.logger.info("Neural Trading Engine started")
    
    def stop(self):
        """Stop the trading engine"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for trading thread to finish
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)
        
        self.logger.info("Neural Trading Engine stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        self.logger.info("Starting trading loop")
        
        try:
            while self.is_running:
                current_time = datetime.now()
                
                # Update positions
                self._update_positions()
                
                # Generate signals for trading pairs
                for symbol in self.trading_pairs:
                    if not self.is_running:
                        break
                    
                    try:
                        signal = self._generate_signal(symbol)
                        if signal:
                            self._process_signal(signal)
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                # Update performance metrics
                if self.last_performance_update is None or \
                   (current_time - self.last_performance_update).seconds >= 60:
                    self._update_performance_metrics()
                    self.last_performance_update = current_time
                
                # Sleep before next iteration
                time.sleep(5)  # 5-second cycles
                
        except Exception as e:
            self.logger.error(f"Critical error in trading loop: {e}")
        finally:
            self.is_running = False
    
    def _generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate neural trading signal for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            TradingSignal object or None
        """
        try:
            # Get market data
            market_data = self._get_market_data(symbol)
            if not market_data:
                return None
            
            # Extract features
            features = self._extract_features(market_data)
            if features is None:
                return None
            
            # Get neural prediction
            prediction = self.model_manager.predict(features)
            if not prediction:
                return None
            
            # Check confidence threshold
            if prediction['confidence'] < self.confidence_threshold:
                return None
            
            # Get symbol info for trading
            symbol_info = self.mt5_connector.get_symbol_info(symbol)
            if not symbol_info:
                return None
            
            # Calculate trading parameters
            entry_price = symbol_info['ask'] if prediction['action'] == 'BUY' else symbol_info['bid']
            stop_loss, take_profit = self._calculate_sl_tp(
                symbol, prediction['action'], entry_price, symbol_info
            )
            
            # Calculate position size
            position_size = self._calculate_position_size(symbol, entry_price, stop_loss, symbol_info)
            
            if position_size <= 0:
                return None
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                action=prediction['action'],
                confidence=prediction['confidence'],
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=f"Neural prediction: {prediction['action']} ({prediction['confidence']:.1%} confidence)"
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive market data for a symbol"""
        try:
            market_data = {}
            
            # Get data for multiple timeframes
            for tf_name, tf_constant in self.timeframes.items():
                rates = self.mt5_connector.get_rates(symbol, tf_constant, 0, 100)
                if rates:
                    market_data[tf_name] = rates
                else:
                    self.logger.warning(f"No {tf_name} data for {symbol}")
                    return None
            
            # Get symbol info
            symbol_info = self.mt5_connector.get_symbol_info(symbol)
            if symbol_info:
                market_data['symbol_info'] = symbol_info
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _extract_features(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract features for neural network"""
        try:
            features = []
            
            # Use M15 as primary timeframe
            if 'M15' not in market_data:
                return None
            
            m15_data = market_data['M15']
            if len(m15_data) < 20:
                return None
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(m15_data)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Calculate technical indicators
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['rsi'] = self._calculate_rsi(df['close'])
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Create feature vector
            current_price = latest['close']
            prev_price = df.iloc[-2]['close']
            
            # Price features
            price_change = (current_price - prev_price) / prev_price
            z_score = (current_price - df['close'].mean()) / df['close'].std()
            
            # SMA features
            sma_5_ratio = latest['sma_5'] / current_price - 1
            sma_20_ratio = latest['sma_20'] / current_price - 1
            
            # Technical indicators
            rsi_norm = latest['rsi'] / 100.0
            volatility_norm = latest['volatility'] * 100
            
            features = [
                price_change,
                z_score,
                sma_5_ratio,
                sma_20_ratio,
                rsi_norm,
                volatility_norm
            ]
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_sl_tp(self, symbol: str, action: str, entry_price: float, 
                         symbol_info: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        try:
            spread = symbol_info['spread'] * symbol_info['point']
            
            # Base SL/TP calculation
            if action == 'BUY':
                stop_loss = entry_price - (spread * 30)  # 30 spread stop loss
                take_profit = entry_price + (spread * 60)  # 60 spread take profit
            else:  # SELL
                stop_loss = entry_price + (spread * 30)
                take_profit = entry_price - (spread * 60)
            
            # Adjust for JPY pairs
            if 'JPY' in symbol:
                stop_loss = round(stop_loss, 3)
                take_profit = round(take_profit, 3)
            else:
                stop_loss = round(stop_loss, 5)
                take_profit = round(take_profit, 5)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating SL/TP for {symbol}: {e}")
            return 0.0, 0.0
    
    def _calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float,
                               symbol_info: Dict[str, Any]) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account info
            account_info = self.mt5_connector.get_account_info()
            if not account_info:
                return 0.0
            
            balance = account_info['balance']
            risk_amount = balance * self.risk_per_trade
            
            # Calculate pip value
            pip_value = 10 if 'JPY' not in symbol else 1
            
            # Calculate stop loss in pips
            sl_distance = abs(entry_price - stop_loss)
            sl_pips = sl_distance * (10000 if 'JPY' not in symbol else 100)
            
            if sl_pips == 0:
                return 0.0
            
            # Calculate position size
            position_size = risk_amount / (sl_pips * pip_value)
            
            # Apply symbol constraints
            volume_min = symbol_info.get('volume_min', 0.01)
            volume_max = symbol_info.get('volume_max', 100.0)
            volume_step = symbol_info.get('volume_step', 0.01)
            
            # Round to step size
            position_size = round(position_size / volume_step) * volume_step
            
            # Ensure within limits
            position_size = max(volume_min, min(volume_max, position_size))
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    def _process_signal(self, signal: TradingSignal):
        """Process a trading signal"""
        try:
            # Check if we can take this trade
            if not self._can_trade(signal):
                return
            
            # Execute trade
            order_result = self._execute_trade(signal)
            if order_result:
                signal.executed = True
                signal.order_ticket = order_result.get('order')
                
                # Create position tracking
                position = Position(
                    ticket=order_result['order'],
                    symbol=signal.symbol,
                    action=signal.action,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=signal.position_size
                )
                
                self.positions[position.ticket] = position
                self.signals_history.append(signal)
                
                self.logger.info(f"Trade executed: {signal.symbol} {signal.action} @ {signal.entry_price}")
                self.logger.info(f"SL: {signal.stop_loss}, TP: {signal.take_profit}")
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
    
    def _can_trade(self, signal: TradingSignal) -> bool:
        """Check if we can execute the trade"""
        # Check maximum concurrent positions
        if len(self.positions) >= self.max_concurrent_positions:
            return False
        
        # Check if we already have a position in this symbol
        for position in self.positions.values():
            if position.symbol == signal.symbol and position.status == 'OPEN':
                return False
        
        # Check risk parameters
        if signal.confidence < self.confidence_threshold:
            return False
        
        # Additional risk checks can be added here
        
        return True
    
    def _execute_trade(self, signal: TradingSignal) -> Optional[Dict[str, Any]]:
        """Execute the actual trade"""
        try:
            # Prepare order request
            symbol_info = self.mt5_connector.get_symbol_info(signal.symbol)
            if not symbol_info:
                return None
            
            # Determine order type and price
            if signal.action == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY
                price = symbol_info['ask']
            else:  # SELL
                order_type = mt5.ORDER_TYPE_SELL
                price = symbol_info['bid']
            
            # Create order request
            order_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": signal.position_size,
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
            result = self.mt5_connector.send_order(order_request)
            
            if result and result.get('retcode') == mt5.TRADE_RETCODE_DONE:
                return result
            else:
                self.logger.error(f"Trade execution failed: {result}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    def _update_positions(self):
        """Update position information and check for exits"""
        try:
            # Get current positions from MT5
            mt5_positions = self.mt5_connector.get_positions()
            
            # Update our position tracking
            for ticket, position in list(self.positions.items()):
                # Find position in MT5
                mt5_pos = None
                for pos in mt5_positions:
                    if pos['ticket'] == ticket:
                        mt5_pos = pos
                        break
                
                if mt5_pos:
                    # Update position data
                    position.current_price = mt5_pos['price_current']
                    position.unrealized_pnl = mt5_pos['profit']
                    
                    # Check exit conditions
                    if self._should_close_position(position, mt5_pos):
                        self._close_position(position)
                else:
                    # Position no longer exists in MT5, mark as closed
                    position.status = 'CLOSED'
                    self.logger.info(f"Position {ticket} closed")
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _should_close_position(self, position: Position, mt5_pos: Dict[str, Any]) -> bool:
        """Determine if position should be closed"""
        try:
            current_price = mt5_pos['price_current']
            
            if position.action == 'BUY':
                # Check stop loss
                if current_price <= position.stop_loss:
                    return True
                # Check take profit
                if current_price >= position.take_profit:
                    return True
            else:  # SELL
                # Check stop loss
                if current_price >= position.stop_loss:
                    return True
                # Check take profit
                if current_price <= position.take_profit:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return False
    
    def _close_position(self, position: Position):
        """Close a position"""
        try:
            # This would implement position closing logic
            # For now, just mark as closed
            position.status = 'CLOSED'
            self.logger.info(f"Position {position.ticket} closed: {position.symbol} {position.action}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def _update_performance_metrics(self):
        """Update trading performance metrics"""
        try:
            # Calculate metrics from positions and signals
            closed_positions = [p for p in self.positions.values() if p.status == 'CLOSED']
            
            if closed_positions:
                winning_trades = sum(1 for p in closed_positions if p.unrealized_pnl > 0)
                losing_trades = sum(1 for p in closed_positions if p.unrealized_pnl < 0)
                total_trades = len(closed_positions)
                
                self.performance_metrics.update({
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                    'total_pnl': sum(p.unrealized_pnl for p in closed_positions),
                    'current_drawdown': self._calculate_drawdown(closed_positions)
                })
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _calculate_drawdown(self, positions: List[Position]) -> float:
        """Calculate current drawdown"""
        # Simplified drawdown calculation
        if not positions:
            return 0.0
        
        profits = [p.unrealized_pnl for p in positions]
        if not profits:
            return 0.0
        
        peak = max(profits)
        current = sum(profits)
        
        if peak <= 0:
            return 0.0
        
        return max(0, (peak - current) / peak)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def get_active_signals(self) -> List[Dict[str, Any]]:
        """Get list of recent signals"""
        recent_signals = [s for s in self.signals_history[-20:]]  # Last 20 signals
        
        return [
            {
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol,
                'action': signal.action,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'position_size': signal.position_size,
                'reason': signal.reason,
                'executed': signal.executed
            }
            for signal in recent_signals
        ]
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get list of active positions"""
        active_positions = [p for p in self.positions.values() if p.status == 'OPEN']
        
        return [
            {
                'ticket': position.ticket,
                'symbol': position.symbol,
                'action': position.action,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'position_size': position.position_size,
                'unrealized_pnl': position.unrealized_pnl,
                'open_time': position.open_time.isoformat(),
                'status': position.status
            }
            for position in active_positions
        ]
