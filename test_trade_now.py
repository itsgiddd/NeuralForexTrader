#!/usr/bin/env python3
"""Direct trade execution test"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5
from datetime import datetime

def execute_direct_trade():
    """Execute a direct trade test"""
    print("DIRECT TRADE EXECUTION TEST")
    print("=" * 50)
    
    # Initialize MT5
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
    
    # Get account info
    account_info = mt5.account_info()
    print(f"Account: {account_info.login}")
    print(f"Balance: {account_info.balance}")
    print(f"Equity: {account_info.equity}")
    print(f"Margin: {account_info.margin}")
    print(f"Free Margin: {account_info.margin_free}")
    print(f"Margin Level: {account_info.margin_level}%")
    
    # Try a direct trade
    symbol = "EURUSD"
    symbol_info = mt5.symbol_info(symbol)
    
    print(f"\n{symbol}: Bid={symbol_info.bid}, Ask={symbol_info.ask}")
    
    # Direct trade with FOK filling mode
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01,  # Standard 0.01 lot
        "type": mt5.ORDER_TYPE_SELL,
        "price": symbol_info.bid,
        "deviation": 20,
        "magic": 123456,
        "comment": "Direct neural trade test",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,  # Use FOK (works!)
    }
    
    print("\nSending trade request...")
    result = mt5.order_send(request)
    
    if result is not None:
        print(f"Result: retcode={result.retcode}")
        print(f"Comment: {result.comment}")
        print(f"Order: {result.order}")
        print(f"Volume: {result.volume}")
        print(f"Price: {result.price}")
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print("\nSUCCESS! Trade executed!")
            print(f"Trade Details: SELL 0.01 {symbol} @ {result.price}")
            
            # Check position
            positions = mt5.positions_get(symbol=symbol)
            if positions is not None and len(positions) > 0:
                pos = positions[0]
                print(f"Open Position: {pos.symbol} {pos.type} {pos.volume} @ {pos.price_open}, P/L: {pos.profit}")
                
                # Close the position
                print("\nClosing position...")
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": mt5.symbol_info(symbol).ask,
                    "deviation": 20,
                    "magic": 123456,
                    "comment": "Close test position",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
                close_result = mt5.order_send(close_request)
                if close_result is not None:
                    print(f"Close result: retcode={close_result.retcode}")
                    print(f"Close comment: {close_result.comment}")
        else:
            print(f"FAILED! Error: {result.comment}")
    else:
        print("order_send returned None")
    
    # Check final positions
    print("\nFinal positions:")
    positions = mt5.positions_get()
    if positions is not None and len(positions) > 0:
        for pos in positions:
            print(f"  {pos.symbol}: {pos.type} {pos.volume} @ {pos.price_open}, P/L: {pos.profit}")
    else:
        print("  No open positions")
    
    mt5.shutdown()
    print("\nDone!")

if __name__ == "__main__":
    execute_direct_trade()