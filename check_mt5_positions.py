#!/usr/bin/env python3
"""Check what trades are actually in MT5"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5
from datetime import datetime

def check_mt5_positions():
    """Check all positions in MT5"""
    print("CHECKING MT5 POSITIONS")
    print("=" * 50)
    
    # Initialize MT5
    if not mt5.initialize():
        print("Failed to initialize MT5")
        error = mt5.last_error()
        print(f"Error code: {error[0]}, Message: {error[1]}")
        return
    
    # Get account info
    account_info = mt5.account_info()
    print(f"Account: {account_info.login}")
    print(f"Server: {account_info.server}")
    print(f"Balance: {account_info.balance}")
    
    # Get all positions
    positions = mt5.positions_get()
    
    print(f"\nTotal positions: {len(positions) if positions else 0}")
    
    if positions and len(positions) > 0:
        for pos in positions:
            print(f"\nPosition:")
            print(f"  Symbol: {pos.symbol}")
            print(f"  Type: {'BUY' if pos.type == 0 else 'SELL'}")
            print(f"  Volume: {pos.volume}")
            print(f"  Entry Price: {pos.price_open}")
            print(f"  Current Price: {pos.price_current}")
            print(f"  P/L: {pos.profit}")
            print(f"  Time: {datetime.fromtimestamp(pos.time)}")
    else:
        print("No open positions in MT5")
    
    # Check recent trades
    print("\n" + "=" * 50)
    print("CHECKING DEALS/HISTORY")
    
    # Get deals from today
    from_date = datetime(2026, 2, 5)
    deals = mt5.history_deals_get(from_date, datetime.now())
    
    if deals and len(deals) > 0:
        print(f"Total deals today: {len(deals)}")
        for deal in deals[-10:]:  # Last 10 deals
            print(f"  {deal.time}: {deal.symbol} {deal.type} {deal.volume} @ {deal.price} - P/L: {deal.profit}")
    else:
        print("No deals in history")
    
    mt5.shutdown()
    print("\nDone!")

if __name__ == "__main__":
    check_mt5_positions()