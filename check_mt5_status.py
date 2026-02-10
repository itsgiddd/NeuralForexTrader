"""Quick MT5 status check"""
import MetaTrader5 as mt5

print('=' * 50)
print('MT5 Connection Check')
print('=' * 50)

# Initialize
print('\n1. Initializing MT5...')
if not mt5.initialize():
    print('   FAILED: MT5 initialization failed')
    print(f'   Error: {mt5.last_error()}')
    quit()
print('   SUCCESS: MT5 initialized')

# Get account info
print('\n2. Getting account info...')
account = mt5.account_info()
if account is None:
    print(f'   FAILED: Could not get account info')
    print(f'   Error: {mt5.last_error()}')
else:
    print(f'   Server: {account.server}')
    print(f'   Login: {account.login}')
    print(f'   Balance: ${account.balance:.2f}')
    print(f'   Equity: ${account.equity:.2f}')
    print(f'   Margin: ${account.margin:.2f}')
    print(f'   Free Margin: ${account.margin_free:.2f}')

# Get positions
print('\n3. Checking open positions...')
positions = mt5.positions_get()
if positions is None:
    print(f'   Error getting positions: {mt5.last_error()}')
else:
    print(f'   Open positions: {len(positions)}')
    for i, p in enumerate(positions):
        direction = "BUY" if p.type == 0 else "SELL"
        print(f'   [{i+1}] {p.symbol}: {direction} {p.volume} lots @ {p.price_open}')
        print(f'       Current: {p.price_current} | P/L: ${p.profit:.2f}')

# Get recent trades
print('\n4. Checking deal history (last 10)...')
from datetime import datetime, timedelta
to_date = datetime.now()
from_date = to_date - timedelta(days=1)
deals = mt5.history_deals_get(from_date, to_date)
if deals is None:
    print(f'   Error getting history: {mt5.last_error()}')
else:
    print(f'   Total deals: {len(deals)}')
    for d in deals[-10:]:
        print(f'   {d.time}: {d.symbol} {d.type} {d.volume} @ {d.price} = ${d.profit}')

print('\n' + '=' * 50)
print('Check Complete')
print('=' * 50)
