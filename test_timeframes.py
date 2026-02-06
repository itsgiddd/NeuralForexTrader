#!/usr/bin/env python3
"""
Test script to verify 15-minute, 1-hour, and 4-hour timeframe analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clean_live_trading_bot import LiveNeuralTradingBot, TradingMode
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_timeframe_analysis():
    """Test the multi-timeframe analysis"""
    print("=" * 60)
    print("TESTING MULTI-TIMEFRAME ANALYSIS (15m, 1h, 4h)")
    print("=" * 60)
    
    # Create bot instance
    bot = LiveNeuralTradingBot(
        trading_mode=TradingMode.DEMO,
        confidence_threshold=0.78,
        symbols=['EURUSD']
    )
    
    print("✓ Bot initialized successfully")
    print(f"✓ Confidence threshold: {bot.confidence_threshold:.1%}")
    print(f"✓ Trading symbols: {', '.join(bot.symbols)}")
    
    # Test MT5 connection
    print("\n" + "=" * 40)
    print("TESTING MT5 CONNECTION")
    print("=" * 40)
    
    if bot.connect_to_mt5():
        print("✓ Successfully connected to MT5")
    else:
        print("✗ Failed to connect to MT5")
        return
    
    # Test account info
    print("\n" + "=" * 40)
    print("TESTING ACCOUNT INFO")
    print("=" * 40)
    
    account_info = bot.get_account_info()
    if account_info:
        print(f"✓ Account balance: {account_info.balance}")
        print(f"✓ Account equity: {account_info.equity}")
        print(f"✓ Margin level: {account_info.margin_level}")
        print(f"✓ Currency: {account_info.currency}")
    else:
        print("✗ Could not get account info")
        return
    
    # Test market data collection
    print("\n" + "=" * 40)
    print("TESTING MARKET DATA COLLECTION")
    print("=" * 40)
    
    market_data = bot.get_market_data('EURUSD')
    if market_data:
        print(f"✓ EURUSD bid: {market_data.bid}")
        print(f"✓ EURUSD ask: {market_data.ask}")
        print(f"✓ Spread: {market_data.spread}")
        print(f"✓ H1 data points: {len(market_data.h1_data)}")
        print(f"✓ H4 data points: {len(market_data.h4_data)}")
        
        # Test M15 data specifically
        m15_data = bot._get_m15_data('EURUSD')
        print(f"✓ M15 data points: {len(m15_data)}")
        
        if not m15_data.empty:
            print(f"✓ M15 latest close: {m15_data['close'].iloc[-1]}")
            print(f"✓ M15 time range: {m15_data['time'].iloc[0]} to {m15_data['time'].iloc[-1]}")
    else:
        print("✗ Could not get market data")
        return
    
    # Test signal generation
    print("\n" + "=" * 40)
    print("TESTING SIGNAL GENERATION")
    print("=" * 40)
    
    signal = bot.generate_neural_signal(market_data, account_info)
    if signal:
        print(f"✓ Signal generated: {signal.action.value}")
        print(f"✓ Confidence: {signal.confidence:.1%}")
        print(f"✓ Reason: {signal.reason}")
        print(f"✓ Lot size: {signal.lot_size}")
        print(f"✓ Stop loss: {signal.stop_loss}")
        print(f"✓ Take profit: {signal.take_profit}")
        
        if signal.confidence >= 0.78:
            print(f"✓ HIGH CONFIDENCE SIGNAL! Ready to trade!")
        else:
            print(f"• Signal below 78% threshold - correctly rejected")
    else:
        print("✗ No signal generated")
    
    print("\n" + "=" * 60)
    print("MULTI-TIMEFRAME ANALYSIS TEST COMPLETE")
    print("=" * 60)
    
    # Shutdown MT5
    bot.stop_trading()

if __name__ == "__main__":
    test_timeframe_analysis()