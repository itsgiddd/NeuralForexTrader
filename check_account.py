#!/usr/bin/env python3
"""Check account status and fix trading issues"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5

def check_account():
    """Check account status and see what's preventing trades"""
    print("CHECKING ACCOUNT STATUS")
    print("=" * 50)
    
    # Initialize MT5
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
    
    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        print("Could not get account info")
        mt5.shutdown()
        return
    
    print(f"Account: {account_info.login}")
    print(f"Server: {account_info.server}")
    print(f"Balance: {account_info.balance}")
    print(f"Equity: {account_info.equity}")
    print(f"Margin: {account_info.margin}")
    print(f"Free Margin: {account_info.margin_free}")
    print(f"Margin Level: {account_info.margin_level}%")
    print(f"Currency: {account_info.currency}")
    
    # Check open positions
    print("\\nOPEN POSITIONS:")
    positions = mt5.positions_get()
    if positions is not None and len(positions) > 0:
        for pos in positions:
            print(f"  {pos.symbol}: {pos.type} {pos.volume} lots @ {pos.price_open}, P/L: {pos.profit}")
    else:
        print("  No open positions")
    
    # Check orders
    print("\\nPENDING ORDERS:")
    orders = mt5.orders_get()
    if orders is not None and len(orders) > 0:
        for order in orders:
            print(f"  {order.symbol}: {order.type} {order.volume_initial} lots @ {order.price_open}")
    else:
        print("  No pending orders")
    
    # Check available symbols
    print("\\nAVAILABLE SYMBOLS:")
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    for symbol in symbols:
        info = mt5.symbol_info(symbol)
        if info is not None:
            print(f"  {symbol}: Bid={info.bid}, Ask={info.ask}, Spread={info.spread}")
            print(f"    Min Lot: {info.volume_min}, Max Lot: {info.volume_max}, Step: {info.volume_step}")
            print(f"    Margin Rate: {info.margin_rate}")
        else:
            print(f"  {symbol}: Not available")
    
    # Try a test trade with very small lot
    print("\\nTRYING TEST TRADE:")
    print("Attempting to open a 0.01 lot SELL on EURUSD...")
    
    symbol = "EURUSD"
    symbol_info = mt5.symbol_info(symbol)
    
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
        "type_filling": mt5.ORDER_FILLING_RETURN,  # Try different filling mode
    }
    
    result = mt5.order_send(request)
    
    if result is not None:
        print(f"Result: retcode={result.retcode}")
        print(f"Comment: {result.comment}")
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print("SUCCESS! Trade executed.")
            # Close the trade immediately
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
            }
            close_result = mt5.order_send(close_request)
            if close_result is not None:
                print(f"Close result: retcode={close_result.retcode}")
        else:
            print(f"FAILED! Error: {result.comment}")
    else:
        print("order_send returned None")
    
    mt5.shutdown()
    print("\\nDone!")

if __name__ == "__main__":
    check_account()