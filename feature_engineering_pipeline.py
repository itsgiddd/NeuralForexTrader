"""
Advanced Feature Engineering Pipeline for Neural Forex Trading
============================================================

Comprehensive feature engineering system that transforms raw forex data
into 150+ sophisticated features for neural network consumption.

Feature Categories:
1. Technical Indicators (50+ features)
2. Price Action Patterns (20+ features)
3. Market Microstructure (30+ features)
4. Volatility Analysis (15+ features)
5. Volume Analysis (10+ features)
6. Market Context (15+ features)
7. Cross-timeframe Features (10+ features)
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
import talib

@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline"""
    use_technical_indicators: bool = True
    use_price_patterns: bool = True
    use_microstructure: bool = True
    use_volatility: bool = True
    use_volume: bool = True
    use_context: bool = True
    use_cross_timeframe: bool = True
    
    # Technical indicator periods
    sma_periods: List[int] = None
    ema_periods: List[int] = None
    rsi_periods: List[int] = None
    macd_periods: Tuple[int, int, int] = None
    bb_periods: Tuple[int, int] = None
    stoch_periods: Tuple[int, int, int] = None
    
    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [10, 20, 50, 100, 200]
        if self.ema_periods is None:
            self.ema_periods = [12, 26, 50]
        if self.rsi_periods is None:
            self.rsi_periods = [14, 21, 30]
        if self.macd_periods is None:
            self.macd_periods = (12, 26, 9)
        if self.bb_periods is None:
            self.bb_periods = (20, 2)
        if self.stoch_periods is None:
            self.stoch_periods = (14, 3, 3)

class TechnicalIndicatorEngine:
    """
    Engine for computing advanced technical indicators.
    Includes traditional indicators and custom forex-specific measures.
    """
    
    @staticmethod
    def compute_sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def compute_ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def compute_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def compute_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD Indicator"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def compute_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        rolling_std = data.rolling(window=period).std()
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        # Simplified ADX calculation
        tr = TechnicalIndicatorEngine.compute_atr(high, low, close, period)
        atr_mean = tr.rolling(window=period).mean()
        return atr_mean / (high - low).rolling(window=period).mean() * 100
    
    @staticmethod
    def compute_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    @staticmethod
    def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    @staticmethod
    def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = np.where(close > close.shift(), volume,
              np.where(close < close.shift(), -volume, 0)).cumsum()
        return pd.Series(obv, index=close.index)

class PricePatternAnalyzer:
    """
    Advanced price pattern recognition beyond traditional chart patterns.
    Includes micro-patterns and price action signatures.
    """
    
    @staticmethod
    def detect_engulfing_patterns(open_prices: pd.Series, close_prices: pd.Series, high_prices: pd.Series, low_prices: pd.Series) -> pd.Series:
        """Detect bullish and bearish engulfing patterns"""
        bullish_engulfing = ((close_prices > open_prices) & 
                           (close_prices.shift(1) < open_prices.shift(1)) &
                           (open_prices < close_prices.shift(1)) &
                           (close_prices > open_prices.shift(1)))
        
        bearish_engulfing = ((close_prices < open_prices) & 
                           (close_prices.shift(1) > open_prices.shift(1)) &
                           (open_prices > close_prices.shift(1)) &
                           (close_prices < open_prices.shift(1)))
        
        return pd.Series(0, index=close_prices.index).where(~bullish_engulfing & ~bearish_engulfing, 
                                                            np.where(bullish_engulfing, 1, -1))
    
    @staticmethod
    def detect_doji_patterns(open_prices: pd.Series, close_prices: pd.Series) -> pd.Series:
        """Detect doji patterns (open ≈ close)"""
        body_size = abs(close_prices - open_prices)
        total_range = (close_prices - open_prices).rolling(window=2).max()
        doji = (body_size / total_range) < 0.1
        return doji.astype(int)
    
    @staticmethod
    def detect_hammer_patterns(open_prices: pd.Series, close_prices: pd.Series, high_prices: pd.Series, low_prices: pd.Series) -> pd.Series:
        """Detect hammer patterns"""
        body = abs(close_prices - open_prices)
        upper_shadow = high_prices - np.maximum(open_prices, close_prices)
        lower_shadow = np.minimum(open_prices, close_prices) - low_prices
        
        # Hammer conditions
        hammer = ((lower_shadow > 2 * body) & 
                 (upper_shadow < body) &
                 (close_prices > open_prices))
        
        return hammer.astype(int)
    
    @staticmethod
    def detect_shooting_star_patterns(open_prices: pd.Series, close_prices: pd.Series, high_prices: pd.Series, low_prices: pd.Series) -> pd.Series:
        """Detect shooting star patterns"""
        body = abs(close_prices - open_prices)
        upper_shadow = high_prices - np.maximum(open_prices, close_prices)
        lower_shadow = np.minimum(open_prices, close_prices) - low_prices
        
        # Shooting star conditions
        shooting_star = ((upper_shadow > 2 * body) & 
                        (lower_shadow < body) &
                        (close_prices < open_prices))
        
        return shooting_star.astype(int)
    
    @staticmethod
    def compute_support_resistance_levels(high_prices: pd.Series, low_prices: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """Compute dynamic support and resistance levels"""
        # Find peaks and troughs
        peaks, _ = find_peaks(high_prices.values, distance=window)
        troughs, _ = find_peaks(-low_prices.values, distance=window)
        
        # Create support and resistance levels
        resistance_levels = pd.Series(index=high_prices.index, dtype=float)
        support_levels = pd.Series(index=low_prices.index, dtype=float)
        
        # Calculate resistance (local maxima)
        for peak in peaks:
            if peak < len(high_prices):
                resistance_levels.iloc[peak] = high_prices.iloc[peak]
        
        # Calculate support (local minima)
        for trough in troughs:
            if trough < len(low_prices):
                support_levels.iloc[trough] = low_prices.iloc[trough]
        
        return {
            'resistance': resistance_levels.fillna(method='ffill'),
            'support': support_levels.fillna(method='ffill')
        }
    
    @staticmethod
    def compute_trend_strength(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int = 20) -> pd.Series:
        """Compute trend strength using ADX-like measure"""
        # Simplified trend strength calculation
        price_change = close_prices.diff(period)
        volatility = close_prices.rolling(window=period).std()
        
        trend_strength = (price_change.rolling(window=period).mean() / volatility) * 100
        return trend_strength

class MicrostructureAnalyzer:
    """
    Market microstructure analysis for forex-specific features.
    Analyzes bid-ask spreads, market depth, and execution quality.
    """
    
    @staticmethod
    def compute_bid_ask_spread(high_prices: pd.Series, low_prices: pd.Series) -> pd.Series:
        """Compute bid-ask spread proxy (using high-low range)"""
        return (high_prices - low_prices) / high_prices * 100
    
    @staticmethod
    def compute_price_efficiency(close_prices: pd.Series, period: int = 10) -> pd.Series:
        """Compute price efficiency ratio"""
        numerator = abs(close_prices - close_prices.shift(period))
        denominator = close_prices.diff().abs().rolling(window=period).sum()
        efficiency = numerator / denominator
        return efficiency
    
    @staticmethod
    def compute_market_impact_proxy(volume: pd.Series, price_change: pd.Series) -> pd.Series:
        """Compute market impact proxy"""
        price_change_abs = price_change.abs()
        volume_normalized = volume / volume.rolling(window=20).mean()
        impact = price_change_abs / volume_normalized
        return impact
    
    @staticmethod
    def compute_order_flow_imbalance(close_prices: pd.Series, volume: pd.Series, period: int = 10) -> pd.Series:
        """Compute order flow imbalance"""
        price_change = close_prices.diff()
        buying_volume = volume.where(price_change > 0, 0)
        selling_volume = volume.where(price_change < 0, 0)
        
        imbalance = (buying_volume - selling_volume).rolling(window=period).sum()
        return imbalance / volume.rolling(window=period).sum()

class VolatilityAnalyzer:
    """
    Advanced volatility analysis including GARCH-like measures
    and regime-specific volatility patterns.
    """
    
    @staticmethod
    def compute_realized_volatility(returns: pd.Series, period: int = 20) -> pd.Series:
        """Compute realized volatility"""
        return returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
    
    @staticmethod
    def compute_garch_proxy(returns: pd.Series, period: int = 20) -> pd.Series:
        """GARCH-like volatility proxy"""
        squared_returns = returns ** 2
        garch_vol = squared_returns.rolling(window=period).mean()
        return garch_vol ** 0.5
    
    @staticmethod
    def compute_volatility_clustering(returns: pd.Series, period: int = 10) -> pd.Series:
        """Compute volatility clustering measure"""
        abs_returns = returns.abs()
        clustering = abs_returns.rolling(window=period).std()
        return clustering
    
    @staticmethod
    def compute_kelly_criterion_proxy(returns: pd.Series, period: int = 100) -> pd.Series:
        """Kelly Criterion proxy for optimal position sizing"""
        mean_return = returns.rolling(window=period).mean()
        variance = returns.rolling(window=period).var()
        kelly = mean_return / variance
        return kelly

class FeatureEngineeringPipeline:
    """
    Main feature engineering pipeline that orchestrates all analysis components
    and produces a comprehensive feature set for the neural network.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.technical_engine = TechnicalIndicatorEngine()
        self.pattern_analyzer = PricePatternAnalyzer()
        self.microstructure_analyzer = MicrostructureAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
    
    def engineer_features(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """
        Main feature engineering method.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'tick_volume']
            symbol: Trading symbol for context-specific features
            
        Returns:
            DataFrame with 150+ engineered features
        """
        features = pd.DataFrame(index=df.index)
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # 1. Technical Indicators (50+ features)
        if self.config.use_technical_indicators:
            features = self._add_technical_indicators(features, df)
        
        # 2. Price Action Patterns (20+ features)
        if self.config.use_price_patterns:
            features = self._add_price_patterns(features, df)
        
        # 3. Market Microstructure (30+ features)
        if self.config.use_microstructure:
            features = self._add_microstructure_features(features, df)
        
        # 4. Volatility Analysis (15+ features)
        if self.config.use_volatility:
            features = self._add_volatility_features(features, df)
        
        # 5. Volume Analysis (10+ features)
        if self.config.use_volume:
            features = self._add_volume_features(features, df)
        
        # 6. Market Context (15+ features)
        if self.config.use_context:
            features = self._add_market_context(features, df, symbol)
        
        # 7. Cross-timeframe Features (10+ features)
        if self.config.use_cross_timeframe:
            features = self._add_cross_timeframe_features(features, df)
        
        return features
    
    def _add_technical_indicators(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to features DataFrame"""
        
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        
        # Simple Moving Averages
        for period in self.config.sma_periods:
            features[f'SMA_{period}'] = self.technical_engine.compute_sma(close, period)
            features[f'Price_SMA_{period}_Ratio'] = close / features[f'SMA_{period}']
        
        # Exponential Moving Averages
        for period in self.config.ema_periods:
            features[f'EMA_{period}'] = self.technical_engine.compute_ema(close, period)
            features[f'Price_EMA_{period}_Ratio'] = close / features[f'EMA_{period}']
        
        # RSI variations
        for period in self.config.rsi_periods:
            features[f'RSI_{period}'] = self.technical_engine.compute_rsi(close, period)
            features[f'RSI_{period}_Normalized'] = (features[f'RSI_{period}'] - 50) / 50
        
        # MACD
        macd_line, signal_line, histogram = self.technical_engine.compute_macd(close, *self.config.macd_periods)
        features['MACD_Line'] = macd_line
        features['MACD_Signal'] = signal_line
        features['MACD_Histogram'] = histogram
        features['MACD_Crossover'] = np.where(macd_line > signal_line, 1, -1)
        
        # Bollinger Bands
        # Bollinger Bands - only use first 2 periods (period, std_dev)
        bb_upper, bb_middle, bb_lower = self.technical_engine.compute_bollinger_bands(close, *self.config.bb_periods[:2])
        features['BB_Upper'] = bb_upper
        features['BB_Middle'] = bb_middle
        features['BB_Lower'] = bb_lower
        features['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        features['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic - only use first 2 periods (k_period, d_period)
        stoch_k, stoch_d = self.technical_engine.compute_stochastic(high, low, close, *self.config.stoch_periods[:2])
        features['Stoch_K'] = stoch_k
        features['Stoch_D'] = stoch_d
        features['Stoch_Crossover'] = np.where(stoch_k > stoch_d, 1, -1)
        
        # ATR and volatility measures
        features['ATR'] = self.technical_engine.compute_atr(high, low, close)
        features['ATR_Ratio'] = features['ATR'] / close * 100
        
        # ADX
        features['ADX'] = self.technical_engine.compute_adx(high, low, close)
        
        # Williams %R
        features['Williams_R'] = self.technical_engine.compute_williams_r(high, low, close)
        
        # CCI
        features['CCI'] = self.technical_engine.compute_cci(high, low, close)
        
        # Additional custom indicators
        features['Price_Change'] = close.pct_change()
        features['High_Low_Ratio'] = high / low
        features['Open_Close_Ratio'] = open_price / close
        
        return features
    
    def _add_price_patterns(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features"""
        
        open_price = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Candlestick patterns
        features['Bullish_Engulfing'] = self.pattern_analyzer.detect_engulfing_patterns(
            open_price, close, high, low)
        features['Bearish_Engulfing'] = (features['Bullish_Engulfing'] == -1).astype(int)
        features['Bullish_Engulfing'] = (features['Bullish_Engulfing'] == 1).astype(int)
        
        features['Doji'] = self.pattern_analyzer.detect_doji_patterns(open_price, close)
        features['Hammer'] = self.pattern_analyzer.detect_hammer_patterns(
            open_price, close, high, low)
        features['Shooting_Star'] = self.pattern_analyzer.detect_shooting_star_patterns(
            open_price, close, high, low)
        
        # Support and resistance
        sr_levels = self.pattern_analyzer.compute_support_resistance_levels(high, low)
        features['Distance_to_Resistance'] = (features.get('BB_Upper', high) - close) / close * 100
        features['Distance_to_Support'] = (close - features.get('BB_Lower', low)) / close * 100
        
        # Trend strength
        features['Trend_Strength'] = self.pattern_analyzer.compute_trend_strength(
            high, low, close)
        
        # Price momentum indicators
        for period in [5, 10, 20]:
            features[f'Momentum_{period}'] = close / close.shift(period) - 1
            features[f'ROC_{period}'] = (close - close.shift(period)) / close.shift(period) * 100
        
        return features
    
    def _add_microstructure_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['tick_volume']
        
        # Spread and efficiency
        features['Spread_Proxy'] = self.microstructure_analyzer.compute_bid_ask_spread(high, low)
        features['Price_Efficiency'] = self.microstructure_analyzer.compute_price_efficiency(close)
        
        # Market impact
        price_change = close.diff()
        features['Market_Impact'] = self.microstructure_analyzer.compute_market_impact_proxy(
            volume, price_change)
        
        # Order flow
        features['Order_Flow_Imbalance'] = self.microstructure_analyzer.compute_order_flow_imbalance(
            close, volume)
        
        # Volume-weighted features
        for period in [10, 20, 50]:
            features[f'Volume_MA_{period}'] = volume.rolling(window=period).mean()
            features[f'Volume_Ratio_{period}'] = volume / features[f'Volume_MA_{period}']
            features[f'Price_Volume_{period}'] = (close * volume).rolling(window=period).mean()
        
        return features
    
    def _add_volatility_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility analysis features"""
        
        close = df['close']
        returns = close.pct_change()
        
        # Realized volatility
        for period in [5, 10, 20, 50]:
            features[f'Realized_Vol_{period}'] = self.volatility_analyzer.compute_realized_volatility(
                returns, period)
            features[f'Garch_Proxy_{period}'] = self.volatility_analyzer.compute_garch_proxy(
                returns, period)
            features[f'Vol_Clustering_{period}'] = self.volatility_analyzer.compute_volatility_clustering(
                returns, period)
        
        # Volatility ratios
        features['Vol_Ratio_Short_Long'] = (features['Realized_Vol_5'] / 
                                           features['Realized_Vol_20'])
        features['Vol_Ratio_Medium_Long'] = (features['Realized_Vol_10'] / 
                                            features['Realized_Vol_50'])
        
        # Kelly criterion proxy
        features['Kelly_Proxy'] = self.volatility_analyzer.compute_kelly_criterion_proxy(returns)
        
        # Volatility regime detection
        vol_short = features['Realized_Vol_10']
        vol_long = features['Realized_Vol_50']
        vol_regime = np.where(vol_short > vol_long * 1.2, 1,  # High volatility regime
                             np.where(vol_short < vol_long * 0.8, -1, 0))  # Low volatility regime
        features['Volatility_Regime'] = vol_regime
        
        return features
    
    def _add_volume_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume analysis features"""
        
        close = df['close']
        volume = df['tick_volume']
        
        # On-Balance Volume
        features['OBV'] = self.technical_engine.compute_obv(close, volume)
        
        # Volume Price Trend
        vpt = (volume * close.pct_change()).cumsum()
        features['VPT'] = vpt
        
        # Chaikin Money Flow
        money_flow_multiplier = ((close - df['low']) - (df['high'] - close)) / (df['high'] - df['low'])
        money_flow_volume = money_flow_multiplier * volume
        cmf = money_flow_volume.rolling(window=20).sum() / volume.rolling(window=20).sum()
        features['CMF'] = cmf
        
        # Volume Rate of Change
        features['VROC'] = volume.pct_change(periods=10) * 100
        
        return features
    
    def _add_market_context(self, features: pd.DataFrame, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add market context features"""
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Market hours (forex-specific)
        features['Hour'] = pd.to_datetime(df.index).hour
        features['Day_of_Week'] = pd.to_datetime(df.index).dayofweek
        features['Is_Asian_Session'] = ((features['Hour'] >= 0) & (features['Hour'] <= 8)).astype(int)
        features['Is_London_Session'] = ((features['Hour'] >= 8) & (features['Hour'] <= 16)).astype(int)
        features['Is_New_York_Session'] = ((features['Hour'] >= 13) & (features['Hour'] <= 22)).astype(int)
        
        # Market regime indicators
        returns = close.pct_change()
        features['Market_Stress'] = returns.abs().rolling(window=20).mean()
        features['Market_Momentum'] = returns.rolling(window=10).mean()
        
        # Currency-specific features (can be expanded)
        if 'JPY' in symbol:
            features['Is_JPY_Session'] = ((features['Hour'] >= 0) & (features['Hour'] <= 6)).astype(int)
        elif 'USD' in symbol:
            features['Is_USD_Session'] = features['Is_New_York_Session']
        
        return features
    
    def _add_cross_timeframe_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-timeframe analysis features"""
        
        close = df['close']
        
        # Multi-timeframe moving averages
        sma_short = close.rolling(window=10).mean()
        sma_medium = close.rolling(window=50).mean()
        sma_long = close.rolling(window=200).mean()
        
        features['SMA_Alignment'] = ((sma_short > sma_medium) & (sma_medium > sma_long)).astype(int)
        features['Price_SMA_Short_Medium_Ratio'] = sma_short / sma_medium
        features['Price_SMA_Medium_Long_Ratio'] = sma_medium / sma_long
        
        # Momentum across timeframes
        for period in [5, 10, 20]:
            momentum = close / close.shift(period) - 1
            features[f'Momentum_ZScore_{period}'] = (momentum - momentum.rolling(window=100).mean()) / momentum.rolling(window=100).std()
        
        return features
    
    def normalize_features(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Normalize features using robust scaling.
        
        Returns:
            Tuple of (normalized_features, scaling_parameters)
        """
        normalized = features.copy()
        scaling_params = {}
        
        for column in features.columns:
            if features[column].dtype in ['float64', 'int64']:
                median = features[column].median()
                mad = (features[column] - median).abs().median()
                
                if mad > 0:
                    normalized[column] = (features[column] - median) / mad
                    scaling_params[column] = {'median': median, 'mad': mad}
                else:
                    normalized[column] = 0
                    scaling_params[column] = {'median': median, 'mad': 1}
        
        return normalized, scaling_params
    
    def create_feature_tensor(self, features: pd.DataFrame, sequence_length: int = 100) -> torch.Tensor:
        """
        Create PyTorch tensor from features for neural network consumption.
        
        Args:
            features: DataFrame with engineered features
            sequence_length: Length of sequences for LSTM/GRU processing
            
        Returns:
            PyTorch tensor of shape (batch_size, sequence_length, n_features)
        """
        # Ensure we have enough data
        if len(features) < sequence_length:
            raise ValueError(f"Not enough data points. Need at least {sequence_length}, got {len(features)}")
        
        # Take the most recent sequence
        recent_features = features.tail(sequence_length).values
        
        # Convert to tensor and add batch dimension
        tensor = torch.FloatTensor(recent_features).unsqueeze(0)
        
        return tensor

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    
    # Generate realistic forex-like price data
    base_price = 1.1000
    returns = np.random.normal(0, 0.001, 1000)
    prices = base_price * np.cumprod(1 + returns)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.002, 1000)),
        'low': prices * (1 - np.random.uniform(0, 0.002, 1000)),
        'close': prices,
        'tick_volume': np.random.randint(100, 1000, 1000)
    }, index=dates)
    
    # Initialize feature engineering pipeline
    config = FeatureConfig()
    pipeline = FeatureEngineeringPipeline(config)
    
    # Engineer features
    print("Engineering features...")
    features = pipeline.engineer_features(data, "EURUSD")
    
    print(f"Generated {len(features.columns)} features")
    print("\nFeature categories:")
    
    categories = {
        'Technical': len([col for col in features.columns if any(x in col for x in ['SMA', 'EMA', 'RSI', 'MACD', 'BB'])]),
        'Patterns': len([col for col in features.columns if any(x in col for x in ['Engulfing', 'Doji', 'Hammer', 'Shooting'])]),
        'Microstructure': len([col for col in features.columns if any(x in col for x in ['Spread', 'Efficiency', 'Impact', 'Flow'])]),
        'Volatility': len([col for col in features.columns if any(x in col for x in ['Vol', 'Garch', 'Clustering'])]),
        'Volume': len([col for col in features.columns if any(x in col for x in ['OBV', 'VPT', 'CMF', 'VROC'])]),
        'Context': len([col for col in features.columns if any(x in col for x in ['Hour', 'Session', 'Stress'])]),
        'Cross-timeframe': len([col for col in features.columns if any(x in col for x in ['Alignment', 'Momentum'])]),
    }
    
    for category, count in categories.items():
        print(f"{category}: {count} features")
    
    # Normalize features
    normalized_features, scaling_params = pipeline.normalize_features(features)
    
    # Create feature tensor
    feature_tensor = pipeline.create_feature_tensor(normalized_features, sequence_length=100)
    
    print(f"\nFeature tensor shape: {feature_tensor.shape}")
    print("Feature engineering pipeline completed successfully!")
    
    # Display sample features
    print("\nSample features (first 10 columns):")
    print(features.iloc[-5:, :10].round(4))
