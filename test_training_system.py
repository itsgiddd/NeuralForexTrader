#!/usr/bin/env python3
"""
Simple test of the training system without emojis
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_mt5_connection():
    """Test MT5 connection"""
    print("Testing MT5 connection...")
    
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            print(f"MT5 init failed: {mt5.last_error()}")
            return False
        
        print("MT5 initialized")
        
        account_info = mt5.account_info()
        if account_info:
            print(f"Account: {account_info.login}")
            print(f"Balance: ${account_info.balance:.2f}")
        else:
            print("No account info")
            return False
        
        # Test EURUSD
        rates = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_M15, 0, 100)
        if rates is not None:
            print(f"Got {len(rates)} EURUSD M15 candles")
            return True
        else:
            print("Failed to get EURUSD rates")
            return False
            
    except Exception as e:
        print(f"MT5 error: {e}")
        return False

def test_data_collector():
    """Test data collection"""
    print("\nTesting data collector...")
    
    try:
        from mt5_neural_training_system import MT5DataCollector
        
        collector = MT5DataCollector(['EURUSD', 'GBPUSD'])
        data = collector.collect_historical_data(days_back=30)
        
        print(f"Collected data for {len(data)} symbols")
        
        for symbol, timeframes in data.items():
            for tf, df in timeframes.items():
                print(f"  {symbol} {tf}: {len(df)} candles")
        
        return True
        
    except Exception as e:
        print(f"Data collector error: {e}")
        return False

def test_neural_components():
    """Test neural network components"""
    print("\nTesting neural components...")
    
    try:
        import torch
        
        # Test basic tensor operations
        x = torch.randn(10, 50)
        print(f"PyTorch working: {x.shape}")
        
        # Test neural network import
        from mt5_neural_training_system import AdvancedNeuralNetwork
        
        model = AdvancedNeuralNetwork(input_dim=50, hidden_dim=64, num_layers=2)
        print(f"Neural network created: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        output = model(x)
        print(f"Forward pass successful: {output['direction'].shape}")
        
        return True
        
    except Exception as e:
        print(f"Neural components error: {e}")
        return False

def main():
    """Main test function"""
    print("Enhanced Trading System - Component Tests")
    print("=" * 50)
    
    tests = [
        ("MT5 Connection", test_mt5_connection),
        ("Data Collection", test_data_collector),
        ("Neural Components", test_neural_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\nRunning: {name}")
        try:
            if test_func():
                print(f"{name} PASSED")
                passed += 1
            else:
                print(f"{name} FAILED")
        except Exception as e:
            print(f"{name} CRASHED: {e}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All core components working!")
        print("Ready to run enhanced training system")
    else:
        print("WARNING: Some components failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
