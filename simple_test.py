#!/usr/bin/env python3
"""
Simple Test Script for Neural Trading System
"""

import sys
import traceback
import time

def test_mt5():
    """Test MT5 connection"""
    print("Testing MT5 connection...")
    
    try:
        import MetaTrader5 as mt5
        print("MT5 imported successfully")
        
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
        rates = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_M15, 0, 5)
        if rates is not None:
            print(f"Got {len(rates)} EURUSD candles")
        else:
            print("Failed to get EURUSD rates")
            return False
        
        mt5.shutdown()
        print("MT5 test PASSED")
        return True
        
    except Exception as e:
        print(f"MT5 error: {str(e)}")
        traceback.print_exc()
        return False

def test_neural_brain():
    """Test neural brain"""
    print("Testing neural brain...")
    
    try:
        sys.path.append('./')
        from contextual_trading_brain import ContextualTradingBrain
        print("ContextualTradingBrain imported")
        
        brain = ContextualTradingBrain()
        print("ContextualTradingBrain created")
        
        # Test that the brain has the think method
        if hasattr(brain, 'think'):
            print("Neural brain has 'think' method")
            print("Neural brain test PASSED")
            return True
        else:
            print("Neural brain missing 'think' method")
            return False
        
    except Exception as e:
        print(f"Neural brain error: {str(e)}")
        traceback.print_exc()
        return False

def test_trading_systems():
    """Test trading systems"""
    print("Testing trading systems...")
    
    try:
        sys.path.append('./')
        
        # Test quick trading
        from clean_live_trading_bot import LiveNeuralTradingBot, TradingMode, TradeResult, TradeSignal
        print("Trading classes imported from clean_live_trading_bot")
        
        # Test that we can import from quick_trade_decision
        import quick_trade_decision
        print("Quick trade decision module imported")
        
        print("Trading systems test PASSED")
        return True
        
    except Exception as e:
        print(f"Trading systems error: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main test"""
    print("=== Neural Trading System Tests ===")
    
    tests = [
        ("MT5 Connection", test_mt5),
        ("Neural Brain", test_neural_brain),
        ("Trading Systems", test_trading_systems)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, func in tests:
        print(f"\nRunning: {name}")
        try:
            if func():
                print(f"{name} PASSED")
                passed += 1
            else:
                print(f"{name} FAILED")
        except Exception as e:
            print(f"{name} CRASHED: {e}")
    
    print(f"\n=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("SUCCESS: All tests passed! System is ready.")
    else:
        print("WARNING: Some tests failed.")
    
    return passed == total

if __name__ == "__main__":
    main()
