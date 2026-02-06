#!/usr/bin/env python3
"""Simple trade execution test"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5

def test_trade():
    """Test trade execution"""
    print("TESTING TRADE EXECUTION")
    print("=" * 50)
    
    # Initialize MT5
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
    
    # Get account info
    account_info = mt5.account_info()
    print(f"Balance: {account_info.balance}")
    print(f"Equity: {account_info.equity}")
    print(f"Margin: {account_info.margin}")
    print(f"Free Margin: {account_info.margin_free}")
    
    # Test trade
    symbol = "EURUSD"
    symbol_info = mt5.symbol_info(symbol)
    
    print(f"\n{symbol}: Bid={symbol_info.bid}, Ask={symbol_info.ask}")
    print(f"Min Lot: {symbol_info.volume_min}")
    
    # Try different filling modes
    filling_modes = [
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_FOK,
        mt5.ORDER_FILLING_RETURN
    ]
    
    for fill_mode in filling_modes:
        print(f"\nTrying filling mode {fill_mode}...")
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_SELL,
            "price": symbol_info.bid,
            "deviation": 20,
            "magic": 123456,
            "comment": "Test trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": fill_mode,
        }
        
        result = mt5.order_send(request)
        
        if result is not None:
            print(f"  Result: retcode={result.retcode}")
            print(f"  Comment: {result.comment}")
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print("  SUCCESS! Trade executed.")
                
                # Close the trade
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": 0.01,
                    "type": mt5.ORDER_TYPE_BUY,
                    "position": result.order,
                    "price": mt5.symbol_info(symbol).ask,
                    "deviation": 20,
                    "magic": 123456,
                    "comment": "Close test",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": fill_mode,
                }
                close_result = mt5.order_send(close_request)
                if close_result is not None:
                    print(f"  Close result: retcode={close_result.retcode}")
                
                break
        else:
            print("  order_send returned None")
    
    mt5.shutdown()
    print("\nDone!")

if __name__ == "__main__":
    test_trade()