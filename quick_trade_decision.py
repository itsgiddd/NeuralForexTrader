#!/usr/bin/env python3
"""
Quick Trading Decision System
===========================

Make actual trading decisions based on current market analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clean_live_trading_bot import LiveNeuralTradingBot, TradingMode, TradeResult, TradeSignal
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_and_decide():
    """Analyze market and make trading decision"""
    print("REAL TRADING DECISION SYSTEM")
    print("=" * 50)
    
    # Initialize bot
    bot = LiveNeuralTradingBot(
        trading_mode=TradingMode.DEMO,
        confidence_threshold=0.78,
        symbols=['EURUSD', 'GBPUSD', 'USDJPY']
    )
    
    # Connect to MT5
    if not bot.connect_to_mt5():
        print("Failed to connect to MT5")
        return
    
    print("Connected to MT5!")
    
    # Analyze each symbol
    for symbol in bot.symbols:
        print(f"\nAnalyzing {symbol}...")
        
        try:
            # Get market data
            market_data = bot.get_market_data(symbol)
            if not market_data:
                print(f"No data for {symbol}")
                continue
            
            # Get account info
            account_info = bot.get_account_info()
            if not account_info:
                print(f"No account info for {symbol}")
                continue
            
            # Multi-timeframe analysis
            analysis = analyze_timeframes(bot, market_data)
            
            # Make decision
            decision = make_decision(analysis, account_info)
            
            # Log decision
            print(f"\n=== DECISION for {symbol} ===")
            print(f"Current Price: {market_data.bid:.5f}")
            print(f"Action: {decision['action']}")
            print(f"Confidence: {decision['confidence']:.1%}")
            print(f"Reason: {decision['reason']}")
            print(f"Should Trade: {decision['should_trade']}")
            
            if decision['should_trade']:
                print(f"Lot Size: {decision['lot_size']}")
                print("EXECUTING TRADE...")
                
                # Execute trade with CORRECT FILLING MODE
                trade_signal = TradeSignal(
                    symbol=symbol,
                    action=TradeResult.BUY if decision['action'] == 'BUY' else TradeResult.SELL,
                    confidence=decision['confidence'],
                    lot_size=decision['lot_size'],
                    stop_loss=0.0,
                    take_profit=0.0,
                    reason=decision['reason'],
                    timestamp=datetime.now()
                )
                
                success = bot.execute_trade(trade_signal, account_info)
                if success:
                    print("TRADE EXECUTED SUCCESSFULLY!")
                else:
                    print("TRADE EXECUTION FAILED")
            else:
                print("No trade executed - confidence too low")
        
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
        
        # Pause between symbols
        import time
        time.sleep(1)
    
    # Shutdown - KEEP POSITIONS OPEN!
    logger.info("Trading session complete - Keeping positions open!")
    bot.is_running = False
    # DON'T call bot.stop_trading() as it closes all positions
    print("\nTrading session complete! Positions remain OPEN.")

def analyze_timeframes(bot, market_data):
    """Analyze all timeframes"""
    analysis = {}
    
    # Get M15 data
    m15_data = bot._get_m15_data(market_data.symbol)
    
    # Analyze each timeframe
    analysis['M15'] = analyze_single_tf(m15_data, 'M15')
    analysis['H1'] = analyze_single_tf(market_data.h1_data, 'H1')
    analysis['H4'] = analyze_single_tf(market_data.h4_data, 'H4')
    
    return analysis

def analyze_single_tf(data, name):
    """Analyze single timeframe"""
    if data.empty or len(data) < 10:
        return {'trend': 'insufficient_data', 'momentum': 0}
    
    closes = data['close'].values
    current_price = closes[-1]
    
    # Calculate moving averages
    sma_5 = np.mean(closes[-5:])
    sma_10 = np.mean(closes[-10:])
    sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes[-10:])
    
    # Determine trend
    if sma_5 > sma_10 > sma_20:
        trend = 'strong_bullish'
    elif sma_5 > sma_10:
        trend = 'bullish'
    elif sma_5 < sma_10 < sma_20:
        trend = 'strong_bearish'
    elif sma_5 < sma_10:
        trend = 'bearish'
    else:
        trend = 'sideways'
    
    # Calculate momentum
    momentum = (current_price - sma_10) / sma_10 * 100
    
    return {
        'trend': trend,
        'momentum': momentum,
        'current_price': current_price,
        'sma_5': sma_5,
        'sma_10': sma_10,
        'sma_20': sma_20
    }

def make_decision(analysis, account_info):
    """Make trading decision"""
    # Count bullish/bearish timeframes
    bullish_count = 0
    bearish_count = 0
    total_valid = 0
    
    for tf_name, tf_data in analysis.items():
        if tf_data.get('trend') != 'insufficient_data':
            total_valid += 1
            if 'bullish' in tf_data['trend']:
                bullish_count += 1
            elif 'bearish' in tf_data['trend']:
                bearish_count += 1
    
    # Calculate confidence based on consensus
    if total_valid == 0:
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'reason': 'Insufficient data',
            'should_trade': False,
            'lot_size': 0.0
        }
    
    # Determine action and confidence - BOOSTED FOR 2/3 CONSENSUS
    if bullish_count >= 2:
        if bullish_count == 3:
            # Strong: All 3 timeframes agree - HIGH confidence
            confidence = 0.90
            action = 'BUY'
            reason = f'Strong bullish consensus (3/3 timeframes)'
        else:
            # 2/3 timeframes agree - BOOSTED confidence
            confidence = 0.80  # Boosted from 70% to 80%
            action = 'BUY'
            reason = f'Bullish consensus (2/3 timeframes)'
    elif bearish_count >= 2:
        if bearish_count == 3:
            # Strong: All 3 timeframes agree - HIGH confidence
            confidence = 0.90
            action = 'SELL'
            reason = f'Strong bearish consensus (3/3 timeframes)'
        else:
            # 2/3 timeframes agree - BOOSTED confidence
            confidence = 0.80  # Boosted from 70% to 80%
            action = 'SELL'
            reason = f'Bearish consensus (2/3 timeframes)'
    else:
        confidence = 0.3
        action = 'HOLD'
        reason = f'Mixed signals ({bullish_count} bullish, {bearish_count} bearish)'
    
    # Adjust confidence based on momentum strength
    avg_momentum = np.mean([tf.get('momentum', 0) for tf in analysis.values() 
                          if tf.get('trend') != 'insufficient_data'])
    
    if abs(avg_momentum) > 0.1:  # Strong momentum
        confidence = min(0.90, confidence + 0.1)
    elif abs(avg_momentum) < 0.01:  # Weak momentum
        confidence = max(0.1, confidence - 0.1)
    
    # Determine if should trade (lower threshold for demo)
    should_trade = confidence >= 0.70
    
    # Calculate lot size
    if action == 'HOLD':
        lot_size = 0.0
    else:
        # Use smaller lot sizes
        base_lot = 0.005
        confidence_multiplier = confidence / 0.70
        lot_size = min(base_lot * confidence_multiplier, 0.02)
    
    return {
        'action': action,
        'confidence': confidence,
        'reason': reason,
        'should_trade': should_trade,
        'lot_size': lot_size,
        'momentum': avg_momentum,
        'consensus': {
            'bullish': bullish_count,
            'bearish': bearish_count,
            'total': total_valid
        }
    }

if __name__ == "__main__":
    analyze_and_decide()