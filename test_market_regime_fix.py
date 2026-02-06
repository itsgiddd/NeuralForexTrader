#!/usr/bin/env python3
"""
Test script to verify that MarketRegimeDetector dimension mismatch is fixed
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to the path to import our modules
sys.path.append('.')

from contextual_trading_brain import MarketRegimeDetector

def test_market_regime_detector():
    """Test MarketRegimeDetector with various input dimensions"""
    print("Testing MarketRegimeDetector with different input dimensions...")
    
    # Initialize the detector
    detector = MarketRegimeDetector(input_dim=220)  # Original expected dimension
    
    # Test with small dimension (like the 14 we were getting)
    small_features = torch.randn(1, 14)  # Batch size 1, 14 features
    print(f"Testing with input shape: {small_features.shape}")
    
    try:
        results = detector(small_features)
        print("✅ Success with 14-dimensional input!")
        print(f"Market regime shape: {results['market_regime'].shape}")
        print(f"Volatility level shape: {results['volatility_level'].shape}")
        print(f"Trend strength shape: {results['trend_strength'].shape}")
    except Exception as e:
        print(f"❌ Failed with 14-dimensional input: {e}")
        return False
    
    # Test with medium dimension
    medium_features = torch.randn(1, 50)
    print(f"\nTesting with input shape: {medium_features.shape}")
    
    try:
        results = detector(medium_features)
        print("✅ Success with 50-dimensional input!")
        print(f"Market regime shape: {results['market_regime'].shape}")
        print(f"Volatility level shape: {results['volatility_level'].shape}")
        print(f"Trend strength shape: {results['trend_strength'].shape}")
    except Exception as e:
        print(f"❌ Failed with 50-dimensional input: {e}")
        return False
    
    # Test with original expected dimension
    large_features = torch.randn(1, 220)
    print(f"\nTesting with input shape: {large_features.shape}")
    
    try:
        results = detector(large_features)
        print("✅ Success with 220-dimensional input!")
        print(f"Market regime shape: {results['market_regime'].shape}")
        print(f"Volatility level shape: {results['volatility_level'].shape}")
        print(f"Trend strength shape: {results['trend_strength'].shape}")
    except Exception as e:
        print(f"❌ Failed with 220-dimensional input: {e}")
        return False
    
    print("\n🎉 All tests passed! MarketRegimeDetector is now dimension-flexible!")
    return True

def test_regime_features_calculation():
    """Test the actual regime_features calculation from contextual_trading_brain"""
    print("\n" + "="*60)
    print("Testing actual regime_features calculation...")
    
    # Simulate the actual data that would come from feature engineering
    # This simulates what happens in contextual_trading_brain.py lines 377-381
    
    # Mock h1_features, h4_features, and context_features
    # These would normally come from feature engineering
    h1_features = np.random.randn(100, 5)  # 100 time steps, 5 features (much smaller than expected 220)
    h4_features = np.random.randn(50, 3)   # 50 time steps, 3 features (much smaller than expected 220)
    context_features = np.array([0.5, 0.3, 0.1, 0.0, 1.0, 0.0, 0.02, 0.015, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 15 features
    
    # Calculate regime_features the same way as in contextual_trading_brain.py
    regime_features = np.concatenate([
        h1_features[-20:].mean(axis=0),  # Recent average features
        h4_features[-10:].mean(axis=0),  # Recent average features
        context_features
    ])
    
    print(f"h1_features shape: {h1_features.shape}")
    print(f"h4_features shape: {h4_features.shape}")
    print(f"context_features shape: {context_features.shape}")
    print(f"Final regime_features shape: {regime_features.shape}")
    print(f"Total regime_features dimensions: {len(regime_features)}")
    
    # Test with MarketRegimeDetector
    detector = MarketRegimeDetector(input_dim=220)
    regime_input = torch.FloatTensor(regime_features).unsqueeze(0)  # Add batch dimension
    
    print(f"Tensor input shape: {regime_input.shape}")
    
    try:
        results = detector(regime_input)
        print("✅ MarketRegimeDetector successfully processed the actual regime_features!")
        print(f"Market regime output shape: {results['market_regime'].shape}")
        return True
    except Exception as e:
        print(f"❌ MarketRegimeDetector failed with actual regime_features: {e}")
        return False

if __name__ == "__main__":
    print("Testing MarketRegimeDetector dimension fix...")
    print("="*60)
    
    # Test 1: Basic dimension flexibility
    test1_passed = test_market_regime_detector()
    
    # Test 2: Real-world scenario
    test2_passed = test_regime_features_calculation()
    
    print("\n" + "="*60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! The dimension mismatch issue is FIXED! 🎉")
    else:
        print("❌ Some tests failed. The issue may not be fully resolved.")
    
    print("="*60)