"""
Enhanced Neural Trading Architecture
===================================

A sophisticated multi-timeframe neural network system for forex trading
with contextual understanding and adaptive learning capabilities.

Architecture Components:
1. Multi-timeframe LSTM/GRU encoder
2. Attention mechanism for pattern recognition
3. Market context analyzer using transformers
4. Ensemble decision maker
5. Risk assessment neural network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import math

@dataclass
class TradingFeatures:
    """Container for all trading features across timeframes"""
    h1_features: torch.Tensor    # Shape: (batch_size, seq_len_h1, n_features)
    h4_features: torch.Tensor    # Shape: (batch_size, seq_len_h4, n_features)
    d1_features: torch.Tensor    # Shape: (batch_size, seq_len_d1, n_features)
    market_context: torch.Tensor # Shape: (batch_size, n_context_features)
    volume_profile: torch.Tensor  # Shape: (batch_size, n_volume_bins)
    sentiment_data: torch.Tensor # Shape: (batch_size, n_sentiment_features)

class MultiTimeframeEncoder(nn.Module):
    """
    Multi-timeframe encoder using LSTM/GRU with attention mechanisms.
    Processes H1, H4, and D1 data to extract temporal patterns.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Separate encoders for each timeframe
        self.h1_encoder = nn.GRU(input_dim, hidden_dim, num_layers, 
                               dropout=dropout, batch_first=True, bidirectional=True)
        self.h4_encoder = nn.GRU(input_dim, hidden_dim, num_layers,
                               dropout=dropout, batch_first=True, bidirectional=True)
        self.d1_encoder = nn.GRU(input_dim, hidden_dim, num_layers,
                               dropout=dropout, batch_first=True, bidirectional=True)
        
        # Attention mechanisms
        self.attention_h1 = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)
        self.attention_h4 = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)
        self.attention_d1 = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)
        
        # Cross-timeframe attention
        self.cross_attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)
        
        # Feature fusion layer
        self.fusion_layer = nn.Linear(hidden_dim * 2 * 3, hidden_dim * 2)
        self.norm_layer = nn.LayerNorm(hidden_dim * 2)
        
    def forward(self, h1_data: torch.Tensor, h4_data: torch.Tensor, d1_data: torch.Tensor) -> torch.Tensor:
        batch_size = h1_data.size(0)
        
        # Encode each timeframe
        h1_out, _ = self.h1_encoder(h1_data)
        h4_out, _ = self.h4_encoder(h4_data)
        d1_out, _ = self.d1_encoder(d1_data)
        
        # Apply self-attention to each timeframe
        h1_attn, _ = self.attention_h1(h1_out, h1_out, h1_out)
        h4_attn, _ = self.attention_h4(h4_out, h4_out, h4_out)
        d1_attn, _ = self.attention_d1(d1_out, d1_out, d1_out)
        
        # Global average pooling to get timeframe representations
        h1_repr = torch.mean(h1_attn, dim=1)  # (batch_size, hidden_dim * 2)
        h4_repr = torch.mean(h4_attn, dim=1)
        d1_repr = torch.mean(d1_attn, dim=1)
        
        # Cross-timeframe attention
        combined = torch.stack([h1_repr, h4_repr, d1_repr], dim=1)  # (batch_size, 3, hidden_dim * 2)
        
        # Apply cross-attention
        cross_attn_out, _ = self.cross_attention(combined, combined, combined)
        # Flatten the 3D output to 2D for fusion layer
        cross_attn_out = cross_attn_out.reshape(cross_attn_out.size(0), -1)  # (batch, 3 * hidden_dim * 2)
        
        # Fuse features
        fused = self.fusion_layer(cross_attn_out)
        fused = self.norm_layer(fused)
        
        return fused


class MultiTimeframeEncoderFlex(nn.Module):
    """
    Flexible multi-timeframe encoder that adapts to any input feature dimension.
    Uses dynamic projection at runtime to handle variable input sizes.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.base_input_dim = input_dim
        
        # Store projection layers for each possible dimension (predefined common sizes)
        # We'll use runtime detection and create projections as needed
        
        # Separate encoders for each timeframe (use flexible input)
        self.h1_encoder = nn.GRU(hidden_dim * 2, hidden_dim, num_layers, 
                               dropout=dropout, batch_first=True, bidirectional=True)
        self.h4_encoder = nn.GRU(hidden_dim * 2, hidden_dim, num_layers,
                               dropout=dropout, batch_first=True, bidirectional=True)
        self.d1_encoder = nn.GRU(hidden_dim * 2, hidden_dim, num_layers,
                               dropout=dropout, batch_first=True, bidirectional=True)
        
        # Attention mechanisms
        self.attention_h1 = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)
        self.attention_h4 = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)
        self.attention_d1 = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)
        
        # Cross-timeframe attention
        self.cross_attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)
        
        # Feature fusion layer
        self.fusion_layer = nn.Linear(hidden_dim * 2 * 3, hidden_dim * 2)
        self.norm_layer = nn.LayerNorm(hidden_dim * 2)
        
        # Cache for dynamic projections
        self._proj_cache = {}
        
    def _get_projection(self, actual_dim: int) -> nn.Linear:
        """Get or create a projection layer for the given input dimension."""
        if actual_dim not in self._proj_cache:
            # Create a new projection layer
            proj = nn.Linear(actual_dim, self.hidden_dim * 2)
            self._proj_cache[actual_dim] = proj
        return self._proj_cache[actual_dim]
        
    def forward(self, h1_data: torch.Tensor, h4_data: torch.Tensor, d1_data: torch.Tensor) -> torch.Tensor:
        batch_size = h1_data.size(0)
        
        # Detect actual input dimensions from the first timeframe
        actual_dim = h1_data.size(-1)  # Features per timestep
        
        # Get dynamic projections
        h1_proj_fn = self._get_projection(actual_dim)
        h4_proj_fn = self._get_projection(actual_dim)
        d1_proj_fn = self._get_projection(actual_dim)
        
        # Project inputs to expected dimension
        # Reshape to (batch * seq, features) for Linear, then reshape back
        h1_flat = h1_data.reshape(-1, actual_dim)
        h4_flat = h4_data.reshape(-1, actual_dim)
        d1_flat = d1_data.reshape(-1, actual_dim)
        
        h1_proj = h1_proj_fn(h1_flat).reshape(batch_size, -1, self.hidden_dim * 2)
        h4_proj = h4_proj_fn(h4_flat).reshape(batch_size, -1, self.hidden_dim * 2)
        d1_proj = d1_proj_fn(d1_flat).reshape(batch_size, -1, self.hidden_dim * 2)
        
        # Encode each timeframe
        h1_out, _ = self.h1_encoder(h1_proj)
        h4_out, _ = self.h4_encoder(h4_proj)
        d1_out, _ = self.d1_encoder(d1_proj)
        
        # Apply self-attention to each timeframe
        h1_attn, _ = self.attention_h1(h1_out, h1_out, h1_out)
        h4_attn, _ = self.attention_h4(h4_out, h4_out, h4_out)
        d1_attn, _ = self.attention_d1(d1_out, d1_out, d1_out)
        
        # Global average pooling to get timeframe representations
        h1_repr = torch.mean(h1_attn, dim=1)  # (batch_size, hidden_dim * 2)
        h4_repr = torch.mean(h4_attn, dim=1)
        d1_repr = torch.mean(d1_attn, dim=1)
        
        # Cross-timeframe attention
        combined = torch.stack([h1_repr, h4_repr, d1_repr], dim=1)  # (batch_size, 3, hidden_dim * 2)
        
        # Apply cross-attention
        cross_attn_out, _ = self.cross_attention(combined, combined, combined)
        # Flatten the 3D output to 2D for fusion layer
        cross_attn_out = cross_attn_out.reshape(cross_attn_out.size(0), -1)  # (batch, 3 * hidden_dim * 2)
        
        # Fuse features
        fused = self.fusion_layer(cross_attn_out)
        fused = self.norm_layer(fused)
        
        return fused

class MarketContextAnalyzer(nn.Module):
    """
    Market context analyzer using transformer architecture.
    Analyzes market regimes, volatility patterns, and macroeconomic conditions.
    """
    def __init__(self, context_dim: int, num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.context_dim = context_dim
        
        # Market regime detection
        self.regime_classifier = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # Trending up, Trending down, Ranging, Volatile
        )
        
        # Volatility clustering analysis
        self.volatility_predictor = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predicted volatility
        )
        
        # Risk sentiment analyzer
        self.sentiment_analyzer = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Risk-on, Risk-off, Neutral
        )
        
    def forward(self, context_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        regime_probs = F.softmax(self.regime_classifier(context_features), dim=-1)
        volatility_pred = self.volatility_predictor(context_features)
        sentiment_probs = F.softmax(self.sentiment_analyzer(context_features), dim=-1)
        
        return {
            'market_regime': regime_probs,
            'volatility': volatility_pred,
            'sentiment': sentiment_probs
        }

class PatternRecognitionModule(nn.Module):
    """
    Advanced pattern recognition using attention mechanisms.
    Identifies complex price patterns beyond traditional technical analysis.
    """
    def __init__(self, input_dim: int, pattern_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.pattern_dim = pattern_dim
        
        # Pattern encoder
        self.pattern_encoder = nn.MultiheadAttention(pattern_dim, num_heads, batch_first=True)
        
        # Pattern classifier
        self.pattern_classifier = nn.Sequential(
            nn.Linear(pattern_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 10 pattern types
        )
        
        # Pattern strength estimator
        self.strength_estimator = nn.Sequential(
            nn.Linear(pattern_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, price_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Apply attention to identify patterns
        pattern_features, _ = self.pattern_encoder(price_sequence, price_sequence, price_sequence)
        pattern_features = torch.mean(pattern_features, dim=1)  # Global average pooling
        
        # Classify pattern type
        pattern_probs = F.softmax(self.pattern_classifier(pattern_features), dim=-1)
        
        # Estimate pattern strength
        strength = torch.sigmoid(self.strength_estimator(pattern_features))
        
        return {
            'pattern_type': pattern_probs,
            'pattern_strength': strength,
            'pattern_features': pattern_features
        }

class RiskAssessmentNetwork(nn.Module):
    """
    Neural network for comprehensive risk assessment.
    Evaluates multiple risk factors and provides dynamic risk scores.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Multi-layer risk assessment
        self.risk_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Risk type specific outputs
        self.market_risk = nn.Linear(hidden_dim // 4, 1)
        self.volatility_risk = nn.Linear(hidden_dim // 4, 1)
        self.liquidity_risk = nn.Linear(hidden_dim // 4, 1)
        self.correlation_risk = nn.Linear(hidden_dim // 4, 1)
        
        # Overall risk score
        self.overall_risk = nn.Linear(hidden_dim // 4, 1)
        
    def forward(self, risk_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.risk_layers(risk_features)
        
        return {
            'market_risk': torch.sigmoid(self.market_risk(hidden)),
            'volatility_risk': torch.sigmoid(self.volatility_risk(hidden)),
            'liquidity_risk': torch.sigmoid(self.liquidity_risk(hidden)),
            'correlation_risk': torch.sigmoid(self.correlation_risk(hidden)),
            'overall_risk': torch.sigmoid(self.overall_risk(hidden))
        }

class EnhancedTradingBrain(nn.Module):
    """
    Main neural trading brain that orchestrates all components.
    Combines multi-timeframe analysis, market context, pattern recognition,
    and risk assessment for intelligent trading decisions.
    """
    def __init__(self, 
                 feature_dim: int = 150,  # Total number of features
                 hidden_dim: int = 256,
                 num_heads: int = 8):
        super().__init__()
        
        # Store dim for reference
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Core components - use flexible encoder that adapts to actual input
        self.multi_timeframe_encoder = MultiTimeframeEncoderFlex(
            input_dim=feature_dim,  # Pass actual feature dim, encoder adapts
            hidden_dim=hidden_dim
        )
        
        self.market_context_analyzer = MarketContextAnalyzer(
            context_dim=hidden_dim, 
            num_heads=num_heads
        )
        
        self.pattern_recognizer = PatternRecognitionModule(
            input_dim=feature_dim,
            pattern_dim=hidden_dim
        )
        
        self.risk_assessor = RiskAssessmentNetwork(
            input_dim=hidden_dim * 3,  # Combined features from multiple modules
            hidden_dim=hidden_dim
        )
        
        # Decision making layers
        self.decision_network = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # BUY, SELL, HOLD
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Position sizing network
        self.position_sizer = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output: 0 to 1 (position size multiplier)
        )
        
    def forward(self, trading_features: TradingFeatures) -> Dict[str, torch.Tensor]:
        # Extract features for each component
        h1_features = trading_features.h1_features
        h4_features = trading_features.h4_features
        d1_features = trading_features.d1_features
        
        # Multi-timeframe encoding - MultiTimeframeEncoderFlex handles projection internally
        temporal_features = self.multi_timeframe_encoder(h1_features, h4_features, d1_features)
        
        # Market context analysis
        market_context = self.market_context_analyzer(trading_features.market_context)
        
        # Pattern recognition
        # Combine price data for pattern analysis (using closing prices from H1)
        price_sequence = h1_features[:, :, -1:]  # Assuming last feature is close price
        pattern_analysis = self.pattern_recognizer(price_sequence)
        
        # Risk assessment
        # Combine all features for risk analysis
        combined_features = torch.cat([
            temporal_features,
            trading_features.market_context,
            pattern_analysis['pattern_features']
        ], dim=-1)
        
        risk_assessment = self.risk_assessor(combined_features)
        
        # Final decision making
        decision_input = torch.cat([
            temporal_features,
            trading_features.market_context,
            pattern_analysis['pattern_features'],
            torch.cat([
                market_context['market_regime'],
                market_context['volatility'],
                market_context['sentiment']
            ], dim=-1)
        ], dim=-1)
        
        # Trading decision
        decision_logits = self.decision_network(decision_input)
        decision_probs = F.softmax(decision_logits, dim=-1)
        
        # Confidence and position sizing
        confidence = self.confidence_estimator(decision_input)
        position_size = self.position_sizer(decision_input)
        
        return {
            'decision': decision_probs,
            'confidence': confidence,
            'position_size': position_size,
            'market_regime': market_context['market_regime'],
            'pattern_analysis': pattern_analysis,
            'risk_assessment': risk_assessment,
            'temporal_features': temporal_features
        }
    
    def predict(self, 
                h1_data: np.ndarray,
                h4_data: np.ndarray, 
                d1_data: np.ndarray,
                market_context: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        High-level prediction method for trading decisions.
        
        Args:
            h1_data: H1 timeframe data (n_samples, n_features)
            h4_data: H4 timeframe data (n_samples, n_features)
            d1_data: D1 timeframe data (n_samples, n_features)
            market_context: Market context features (n_samples, n_context_features)
            
        Returns:
            Dictionary containing trading decision and confidence metrics
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensors
            h1_tensor = torch.FloatTensor(h1_data).unsqueeze(0)
            h4_tensor = torch.FloatTensor(h4_data).unsqueeze(0)
            d1_tensor = torch.FloatTensor(d1_data).unsqueeze(0)
            context_tensor = torch.FloatTensor(market_context).unsqueeze(0)
            
            # Create trading features object
            trading_features = TradingFeatures(
                h1_features=h1_tensor,
                h4_features=h4_tensor,
                d1_features=d1_tensor,
                market_context=context_tensor,
                volume_profile=torch.zeros(1, 20),  # Placeholder
                sentiment_data=torch.zeros(1, 10)   # Placeholder
            )
            
            # Forward pass
            results = self.forward(trading_features)
            
            # Extract key outputs
            decision_probs = results['decision'].squeeze(0).cpu().numpy()
            confidence = results['confidence'].squeeze(0).item()
            position_size = results['position_size'].squeeze(0).item()
            
            # Determine trading action
            actions = ['BUY', 'SELL', 'HOLD']
            action = actions[np.argmax(decision_probs)]
            
            return {
                'action': action,
                'probabilities': {
                    'BUY': float(decision_probs[0]),
                    'SELL': float(decision_probs[1]),
                    'HOLD': float(decision_probs[2])
                },
                'confidence': confidence,
                'position_size_multiplier': position_size,
                'market_regime': results['market_regime'].squeeze(0).cpu().numpy(),
                'risk_metrics': {k: v.squeeze(0).item() for k, v in results['risk_assessment'].items()},
                'pattern_strength': results['pattern_analysis']['pattern_strength'].squeeze(0).item()
            }

# Example usage and testing
if __name__ == "__main__":
    # Model configuration
    feature_dim = 150
    hidden_dim = 256
    
    # Initialize model
    model = EnhancedTradingBrain(feature_dim=feature_dim, hidden_dim=hidden_dim)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print("\nModel Architecture:")
    print("==================")
    for name, module in model.named_children():
        print(f"{name}: {module.__class__.__name__}")
    
    # Test with dummy data
    batch_size = 32
    seq_len_h1, seq_len_h4, seq_len_d1 = 100, 50, 20
    
    # Create dummy features
    trading_features = TradingFeatures(
        h1_features=torch.randn(batch_size, seq_len_h1, feature_dim // 3),
        h4_features=torch.randn(batch_size, seq_len_h4, feature_dim // 3),
        d1_features=torch.randn(batch_size, seq_len_d1, feature_dim // 3),
        market_context=torch.randn(batch_size, 50),
        volume_profile=torch.randn(batch_size, 20),
        sentiment_data=torch.randn(batch_size, 10)
    )
    
    # Forward pass test
    results = model(trading_features)
    print(f"\nTest Results:")
    print(f"Decision shape: {results['decision'].shape}")
    print(f"Confidence shape: {results['confidence'].shape}")
    print(f"Position size shape: {results['position_size'].shape}")
    
    print("\nArchitecture implementation completed successfully!")
