#!/usr/bin/env python3
"""
Neural Trading Bot - Uses Trained Model
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for forex prediction"""
    
    def __init__(self, input_dim):
        super(SimpleNeuralNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # 3 classes: SELL, HOLD, BUY
        )
    
    def forward(self, x):
        return self.network(x)

def setup_mt5():
    """Setup MT5 connection"""
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return False
    
    account_info = mt5.account_info()
    if account_info:
        print(f"Connected to account: {account_info.login}")
        print(f"Balance: ${account_info.balance:.2f}")
        return True
    return False

def load_model():
    """Load the trained neural model"""
    try:
        checkpoint = torch.load('neural_model.pth', map_location='cpu')
        
        model = SimpleNeuralNetwork(input_dim=6)  # 6 features as in training
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("Neural model loaded successfully")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def get_market_data(symbol):
    """Get current market data for a symbol"""
    try:
        # Get M15 data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 50)
        if rates is None or len(rates) < 20:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Add indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = calculate_rsi(df['close'])
        df['returns'] = df['close'].pct_change()
        
        return df.dropna()
        
    except Exception as e:
        print(f"Error getting data for {symbol}: {e}")
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_features(df):
    """Create features for neural network prediction"""
    try:
        if len(df) < 20:
            return None
        
        # Get latest data point
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        
        # Create features
        features = []
        
        # Price change
        price_change = (current_price - prev_price) / prev_price
        features.append(price_change)
        
        # Z-score
        z_score = (current_price - df['close'].mean()) / df['close'].std()
        features.append(z_score)
        
        # SMA ratios
        sma_5_ratio = df['sma_5'].iloc[-1] / current_price - 1
        sma_20_ratio = df['sma_20'].iloc[-1] / current_price - 1
        features.extend([sma_5_ratio, sma_20_ratio])
        
        # RSI
        rsi_value = df['rsi'].iloc[-1] / 100.0
        features.append(rsi_value)
        
        # Volatility
        volatility = df['returns'].std()
        features.append(volatility)
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error creating features: {e}")
        return None

def predict_signal(model, features):
    """Make prediction using neural network"""
    try:
        with torch.no_grad():
            X = torch.FloatTensor(features.reshape(1, -1))
            outputs = model(X)
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1).numpy()[0]
            predicted_class = torch.argmax(outputs, dim=1).item()
            
            # Map classes
            classes = ['SELL', 'HOLD', 'BUY']
            predicted_action = classes[predicted_class]
            confidence = probabilities[predicted_class]
            
            return {
                'action': predicted_action,
                'confidence': confidence,
                'probabilities': {
                    'SELL': probabilities[0],
                    'HOLD': probabilities[1], 
                    'BUY': probabilities[2]
                }
            }
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def calculate_position_size(signal, account_info, symbol_info):
    """Calculate position size based on risk management"""
    try:
        balance = account_info.balance
        risk_per_trade = 0.015  # 1.5% risk per trade
        
        # Simple position size calculation
        lot_size = 0.01  # Start with minimum lot size
        
        # Adjust based on confidence
        if signal['confidence'] > 0.8:
            lot_size = 0.02
        elif signal['confidence'] > 0.6:
            lot_size = 0.015
        
        return lot_size
        
    except Exception as e:
        print(f"Error calculating position size: {e}")
        return 0.01

def execute_trade(symbol, signal, account_info):
    """Execute trade based on neural signal"""
    try:
        if signal['confidence'] < 0.6:  # Minimum confidence threshold
            print(f"{symbol}: Low confidence ({signal['confidence']:.1%}) - skipping trade")
            return False
        
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"{symbol}: Symbol info not available")
            return False
        
        # Calculate position size
        lot_size = calculate_position_size(signal, account_info, symbol_info)
        
        # Prepare order
        if signal['action'] == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = symbol_info.ask
        elif signal['action'] == 'SELL':
            order_type = mt5.ORDER_TYPE_SELL
            price = symbol_info.bid
        else:
            print(f"{symbol}: HOLD signal - no trade")
            return False
        
        # Calculate SL/TP (simple version)
        spread = symbol_info.spread * symbol_info.point
        
        if signal['action'] == 'BUY':
            sl = price - (spread * 30)  # 30 spread stop loss
            tp = price + (spread * 60)  # 60 spread take profit
        else:  # SELL
            sl = price + (spread * 30)
            tp = price - (spread * 60)
        
        # Create order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": f"Neural-{signal['confidence']:.1%}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ {symbol}: {signal['action']} executed")
            print(f"   Price: {price:.5f}")
            print(f"   SL: {sl:.5f}, TP: {tp:.5f}")
            print(f"   Confidence: {signal['confidence']:.1%}")
            print(f"   Lot Size: {lot_size}")
            return True
        else:
            print(f"❌ {symbol}: Trade failed - {result.comment if result else 'Unknown error'}")
            return False
            
    except Exception as e:
        print(f"❌ {symbol}: Trade execution error - {e}")
        return False

def main_trading_loop(model):
    """Main trading loop"""
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    
    print("Starting neural trading bot...")
    print("Trading symbols:", symbols)
    print("Minimum confidence: 60%")
    print("Risk per trade: 1.5%")
    print("=" * 50)
    
    trade_count = 0
    
    try:
        while True:
            account_info = mt5.account_info()
            if not account_info:
                print("Account info not available")
                time.sleep(30)
                continue
            
            print(f"\nAnalyzing markets... ({datetime.now().strftime('%H:%M:%S')})")
            
            for symbol in symbols:
                try:
                    # Get market data
                    df = get_market_data(symbol)
                    if df is None:
                        continue
                    
                    # Create features
                    features = create_features(df)
                    if features is None:
                        continue
                    
                    # Make prediction
                    signal = predict_signal(model, features)
                    if signal is None:
                        continue
                    
                    print(f"{symbol}: {signal['action']} ({signal['confidence']:.1%})")
                    
                    # Execute trade if confident enough
                    if signal['action'] in ['BUY', 'SELL']:
                        success = execute_trade(symbol, signal, account_info)
                        if success:
                            trade_count += 1
                    
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    continue
            
            # Print summary every 10 trades
            if trade_count > 0 and trade_count % 10 == 0:
                print(f"\n📊 Trading Summary:")
                print(f"   Total trades executed: {trade_count}")
                print(f"   Account balance: ${account_info.balance:.2f}")
            
            # Wait before next analysis
            print("Waiting 60 seconds for next analysis...")
            import time
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user")
    except Exception as e:
        print(f"\nTrading loop error: {e}")

def main():
    """Main function"""
    print("Neural Trading Bot")
    print("=" * 50)
    
    # Setup MT5
    if not setup_mt5():
        return False
    
    # Load neural model
    model = load_model()
    if model is None:
        print("Cannot proceed without trained model")
        return False
    
    # Start trading
    main_trading_loop(model)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("Neural trading bot completed")
        else:
            print("Neural trading bot failed")
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"\nBot error: {e}")
