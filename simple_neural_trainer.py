#!/usr/bin/env python3
"""
Simple Neural Network Trainer - Core Functionality Only
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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

def collect_historical_data(symbols=['EURUSD', 'GBPUSD'], days=30):
    """Collect historical data from MT5"""
    print(f"Collecting {days} days of data for {len(symbols)} symbols...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    all_data = {}
    
    for symbol in symbols:
        print(f"Processing {symbol}...")
        symbol_data = {}
        
        # Get M15 data
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, start_date, end_date)
        if rates is None or len(rates) < 50:
            print(f"  Insufficient data for {symbol}")
            continue
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Add basic indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = calculate_rsi(df['close'])
        df['returns'] = df['close'].pct_change()
        
        symbol_data['M15'] = df
        all_data[symbol] = symbol_data
        print(f"  {symbol}: {len(df)} candles")
    
    print(f"Collected data for {len(all_data)} symbols")
    return all_data

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_features_and_labels(data, lookback=20, horizon=5):
    """Create features and labels for training"""
    print("Creating features and labels...")
    
    all_features = []
    all_labels = []
    
    for symbol, timeframes in data.items():
        df = timeframes['M15'].dropna()
        
        if len(df) < lookback + horizon:
            continue
        
        for i in range(lookback, len(df) - horizon):
            # Get historical window
            window = df.iloc[i-lookback:i]
            
            # Create features
            features = []
            
            # Price features
            current_price = df.iloc[i]['close']
            prev_price = df.iloc[i-1]['close']
            price_change = (current_price - prev_price) / prev_price
            
            features.extend([
                price_change,
                (current_price - window['close'].mean()) / window['close'].std(),
                window['sma_5'].iloc[-1] / current_price - 1,
                window['sma_20'].iloc[-1] / current_price - 1,
                window['rsi'].iloc[-1] / 100.0,
                window['returns'].std()
            ])
            
            # Create label (next 5 candles direction)
            future_prices = df.iloc[i:i+horizon]['close']
            future_return = (future_prices.iloc[-1] / current_price) - 1
            
            # Label: 0 for sell, 1 for hold, 2 for buy
            if future_return > 0.001:  # 0.1% threshold
                label = 2  # Buy
            elif future_return < -0.001:
                label = 0  # Sell
            else:
                label = 1  # Hold
            
            all_features.append(features)
            all_labels.append(label)
    
    print(f"Created {len(all_features)} training samples")
    return np.array(all_features), np.array(all_labels)

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

def train_neural_network(features, labels, epochs=50):
    """Train the neural network"""
    print(f"Training neural network for {epochs} epochs...")
    
    # Convert to tensors
    X = torch.FloatTensor(features)
    y = torch.LongTensor(labels)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Initialize model
    model = SimpleNeuralNetwork(features.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_accuracy = (val_predictions == y_val).float().mean().item()
                
                # Calculate win rate
                trading_mask = val_predictions != 1  # Exclude HOLD (label 1)
                if torch.sum(trading_mask) > 0:
                    correct_trades = torch.sum(val_predictions[trading_mask] == y_val[trading_mask])
                    win_rate = correct_trades.float() / torch.sum(trading_mask)
                    print(f"  Epoch {epoch}: Loss={loss.item():.4f}, Val Acc={val_accuracy:.3f}, Win Rate={win_rate:.3f}")
                else:
                    print(f"  Epoch {epoch}: Loss={loss.item():.4f}, Val Acc={val_accuracy:.3f}")
            
            model.train()
    
    return model

def save_model(model, path="neural_model.pth"):
    """Save the trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_date': datetime.now().isoformat()
    }, path)
    print(f"Model saved to {path}")

def test_model(model, features, labels):
    """Test the trained model"""
    print("Testing trained model...")
    
    model.eval()
    with torch.no_grad():
        X = torch.FloatTensor(features)
        y = torch.LongTensor(labels)
        
        outputs = model(X)
        predictions = torch.argmax(outputs, dim=1)
        
        # Calculate metrics
        accuracy = (predictions == y).float().mean().item()
        
        # Win rate (only for BUY/SELL predictions)
        trading_mask = predictions != 1  # Exclude HOLD (label 1)
        if torch.sum(trading_mask) > 0:
            correct_trades = torch.sum(predictions[trading_mask] == y[trading_mask])
            win_rate = correct_trades.float() / torch.sum(trading_mask)
        else:
            win_rate = 0.0
        
        print(f"Final Test Results:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Win Rate: {win_rate:.3f}")
        
        return accuracy, win_rate

def main():
    """Main training function"""
    print("Simple Neural Network Training System")
    print("=" * 50)
    
    # Step 1: Setup MT5
    if not setup_mt5():
        print("Failed to setup MT5")
        return False
    
    # Step 2: Collect data
    data = collect_historical_data()
    if not data:
        print("No data collected")
        return False
    
    # Step 3: Create features and labels
    features, labels = create_features_and_labels(data)
    if len(features) < 100:
        print("Insufficient training data")
        return False
    
    # Step 4: Train neural network
    model = train_neural_network(features, labels, epochs=100)
    
    # Step 5: Test model
    accuracy, win_rate = test_model(model, features, labels)
    
    # Step 6: Save model
    save_model(model)
    
    # Step 7: Summary
    print("\nTraining Summary:")
    print(f"  Data samples: {len(features)}")
    print(f"  Final accuracy: {accuracy:.1%}")
    print(f"  Final win rate: {win_rate:.1%}")
    print(f"  Model saved: neural_model.pth")
    
    if win_rate >= 0.65:  # 65% threshold
        print("\nSUCCESS: Neural network training completed!")
        print("Model is ready for trading.")
        return True
    else:
        print("\nModel needs more training or different parameters.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nNeural network training completed successfully!")
    else:
        print("\nTraining failed or needs improvement.")
