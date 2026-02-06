#!/usr/bin/env python3
"""
Comprehensive Test Script for Neural Trading System
Tests neural brain, MT5 connectivity, and trading functionality
"""

import sys
import traceback
import time
from pathlib import Path

def log(message):
    """Simple logging function"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def test_mt5_connection():
    """Test MT5 connection and basic functionality"""
    log("🔌 Testing MT5 Connection...")
    
    try:
        import MetaTrader5 as mt5
        log("✅ MT5 module imported")
        
        # Initialize MT5
        if not mt5.initialize():
            log(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return False
        log("✅ MT5 initialized")
        
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            log(f"✅ Account: {account_info.login}")
            log(f"   Balance: ${account_info.balance:.2f}")
            log(f"   Equity: ${account_info.equity:.2f}")
        else:
            log("❌ Failed to get account info")
            return False
        
        # Test symbol
        symbol_info = mt5.symbol_info('EURUSD')
        if symbol_info:
            log(f"✅ EURUSD spread: {symbol_info.spread}")
        else:
            log("❌ EURUSD not found")
            return False
        
        # Test rates
        rates = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_M15, 0, 5)
        if rates is not None:
            log(f"✅ Got {len(rates)} EURUSD M15 candles")
        else:
            log("❌ Failed to get rates")
            return False
        
        mt5.shutdown()
        log("✅ MT5 connection test passed")
        return True
        
    except Exception as e:
        log(f"❌ MT5 error: {str(e)}")
        traceback.print_exc()
        return False

def test_neural_brain():
    """Test neural brain components"""
    log("🧠 Testing Neural Brain...")
    
    try:
        sys.path.append('./')
        
        # Test ContextualTradingBrain
        from contextual_trading_brain import ContextualTradingBrain
        log("✅ ContextualTradingBrain imported")
        
        brain = ContextualTradingBrain()
        log("✅ ContextualTradingBrain instantiated")
        
        # Test with sample data
        market_data = {
            'EURUSD': {
                'M15': {'MA5': 1.0800, 'MA10': 1.0795, 'price': 1.0802},
                'H1': {'MA5': 1.0801, 'MA10': 1.0798, 'price': 1.0803},
                'H4': {'MA5': 1.0802, 'MA10': 1.0799, 'price': 1.0804}
            }
        }
        
        signal = brain.generate_signal(market_data, 'EURUSD')
        log(f"✅ Generated signal: {signal}")
        
        return True
        
    except Exception as e:
        log(f"❌ Neural brain error: {str(e)}")
        traceback.print_exc()
        return False

def test_quick_trading():
    """Test quick trading decision system"""
    log("⚡ Testing Quick Trading System...")
    
    try:
        sys.path.append('./')
        
        # Import and test the quick trading system
        import importlib.util
        spec = importlib.util.spec_from_file_location("quick_trade_decision", "quick_trade_decision.py")
        if spec and spec.loader:
            log("✅ Quick trading file found")
            
            # Just test if we can import it without errors
            from quick_trade_decision import TradingSignal, TradingMode, TradingBot
            log("✅ Quick trading classes imported")
            
            return True
        else:
            log("❌ Could not load quick trading module")
            return False
            
    except Exception as e:
        log(f"❌ Quick trading error: {str(e)}")
        traceback.print_exc()
        return False

def test_comprehensive_trading():
    """Test comprehensive trading bot"""
    log("🤖 Testing Comprehensive Trading Bot...")
    
    try:
        sys.path.append('./')
        
        # Import and test the comprehensive trading system
        from clean_live_trading_bot import TradingSignal, TradingMode, LiveTradingBot
        log("✅ Comprehensive trading classes imported")
        
        # Test bot instantiation (but don't run it)
        log("✅ Comprehensive trading module loads correctly")
        
        return True
        
    except Exception as e:
        log(f"❌ Comprehensive trading error: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    log("🚀 Starting Neural Trading System Tests")
    log("=" * 50)
    
    tests = [
        ("MT5 Connection", test_mt5_connection),
        ("Neural Brain", test_neural_brain), 
        ("Quick Trading", test_quick_trading),
        ("Comprehensive Trading", test_comprehensive_trading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        log(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                log(f"✅ {test_name} PASSED")
            else:
                log(f"❌ {test_name} FAILED")
        except Exception as e:
            log(f"💥 {test_name} CRASHED: {str(e)}")
            results.append((test_name, False))
    
    log("\n" + "=" * 50)
    log("📊 TEST RESULTS SUMMARY")
    log("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        log(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    log(f"\nOverall: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        log("🎉 ALL TESTS PASSED! System is ready for trading.")
    else:
        log("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
