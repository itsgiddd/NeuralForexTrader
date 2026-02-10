#!/usr/bin/env python3
"""Debug why trades are being rejected"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clean_live_trading_bot import LiveNeuralTradingBot, TradingMode, AccountInfo
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_risk_management():
    """Debug why trades are being rejected"""
    print("DEBUGGING RISK MANAGEMENT")
    print("=" * 50)
    
    # Initialize bot
    bot = LiveNeuralTradingBot(
        trading_mode=TradingMode.DEMO,
        confidence_threshold=0.78,  # System threshold
        symbols=['EURUSD', 'GBPUSD', 'USDJPY']
    )
    
    # Connect to MT5
    if not bot.connect_to_mt5():
        print("Failed to connect to MT5")
        return
    
    print(f"Bot confidence threshold: {bot.confidence_threshold:.1%}")
    print(f"Max open positions: {bot.max_open_positions}")
    
    # Get account info
    account_info = bot.get_account_info()
    if not account_info:
        print("Could not get account info")
        return
    
    print(f"\nAccount Info:")
    print(f"  Balance: {account_info.balance}")
    print(f"  Equity: {account_info.equity}")
    print(f"  Margin: {account_info.margin}")
    print(f"  Free Margin: {account_info.margin_free}")
    print(f"  Margin Level: {account_info.margin_level}%")
    
    # Check positions
    positions = bot.positions
    print(f"\nCurrent open positions: {len(positions)}")
    for symbol, pos in positions.items():
        print(f"  {symbol}: {pos['action']} @ {pos['entry_price']}")
    
    # Analyze each symbol
    for symbol in bot.symbols:
        print(f"\n{'='*40}")
        print(f"ANALYZING {symbol}")
        print(f"{'='*40}")
        
        # Get market data
        market_data = bot.get_market_data(symbol)
        if not market_data:
            print(f"No market data for {symbol}")
            continue
        
        # Generate signal
        signal = bot.generate_neural_signal(market_data, account_info)
        if not signal:
            print("No signal generated")
            continue
        
        print(f"Signal: {signal.action.value}")
        print(f"Confidence: {signal.confidence:.1%}")
        print(f"Reason: {signal.reason}")
        
        # Check each risk condition
        print(f"\nRisk Management Checks:")
        
        # Check 1: Max open positions
        check1 = len(bot.positions) < bot.max_open_positions
        print(f"  1. Max positions check: {check1} ({len(bot.positions)}/{bot.max_open_positions})")
        
        # Check 2: Already have position
        check2 = symbol not in bot.positions
        print(f"  2. No existing position: {check2} ({symbol in bot.positions})")
        
        # Check 3: Minimum confidence
        check3 = signal.confidence >= bot.confidence_threshold
        print(f"  3. Confidence >= {bot.confidence_threshold:.1%}: {check3} ({signal.confidence:.1%})")
        
        # Check 4: Margin level
        check4 = True
        if account_info.margin > 0:
            check4 = account_info.margin_level >= 100
            print(f"  4. Margin level >= 100%: {check4} ({account_info.margin_level:.1f}%)")
        else:
            print(f"  4. Margin level check: SKIPPED (no margin used)")
        
        # Overall result
        all_checks = check1 and check2 and check3 and check4
        print(f"\nOVERALL: {'PASS' if all_checks else 'FAIL'}")
        
        if not all_checks:
            failed_checks = []
            if not check1: failed_checks.append("max_positions")
            if not check2: failed_checks.append("existing_position")
            if not check3: failed_checks.append("confidence")
            if not check4: failed_checks.append("margin_level")
            print(f"Failed checks: {', '.join(failed_checks)}")
    
    bot.stop_trading()
    print("\nDone!")

if __name__ == "__main__":
    debug_risk_management()
