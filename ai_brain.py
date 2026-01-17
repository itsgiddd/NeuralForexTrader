from market_context import MarketContextAnalyzer
from pattern_recognition import PatternRecognizer
from trade_validator import TradeValidator, TradeDecision
from adaptive_risk import AdaptiveRiskManager
from trading_memory import TradingMemory
from neural_decision_model import NeuralDecisionModel
from typing import Optional
import os
import pandas as pd

class AIBrain:
    """
    Coordinator mimicking human brain layers:
    1. Context (Occipital/Parietal)
    2. Pattern (Temporal)
    3. Validation (Pre-frontal)
    4. Memory (Hippocampus)
    5. Risk (Amygdala regulation)
    """
    def __init__(self):
        self.market_analyzer = MarketContextAnalyzer()
        # PatternRecognizer is instantiated per data slice usually, or we can keep a persistent one?
        # Usually per data. We will instantiate it inside 'think'.
        self.trade_validator = TradeValidator()
        self.risk_manager = AdaptiveRiskManager()
        self.memory = TradingMemory()
        self.reasoning_engine = ReasoningEngine()
        self.model_path = "neural_model.json"
        self.pattern_index = self._build_pattern_index()
        self.neural_model = self._load_neural_model()
        self.pending_features = {}
        
    def think(self, symbol: str, data_h1: pd.DataFrame, data_h4: pd.DataFrame, data_d1: pd.DataFrame, account_info, symbol_info) -> dict:
        """
        Returns a dict with decision and execution details.
        """
        # 0. Check Memory (Revenge Trading / Kill Switch)
        if not self.memory.can_trade(symbol):
            return {"decision": "REJECT", "reason": "Memory Block (Loss Streak or Cooldown)"}
            
        # 1. Market Context
        market_state = self.market_analyzer.get_market_state(symbol, data_h1, data_h4, data_d1)
        
        # 2. Pattern Recognition (on Data H1/H4? Usually patterns on H1 or H4)
        # We assume H1 for patterns for now, or we iterate. 
        # For simplicity, let's say we check H1 patterns.
        recognizer = PatternRecognizer(data_h1)
        patterns = recognizer.detect_all()
        
        # Filter freshness
        current_len = len(data_h1)
        fresh_patterns = [p for p in patterns if p.index_end >= current_len - 2]
        
        if not fresh_patterns:
            return {"decision": "WAIT", "reason": "No fresh patterns"}
            
        # 3. Validation & Selection
        best_decision = None
        best_pattern = None
        best_rationale = []
        
        for pattern in fresh_patterns:
            # Extract features for this pattern (Stub for features needed by validator)
            # In live_trader we extract complex features. 
            # Here let's pass dummy features or minimal needed.
            # Validator needs: vol_anomaly -> pattern.volume_score
            features = {"vol_anomaly": pattern.volume_score * 2} # Approximate mapping
            
            decision = self.trade_validator.validate(pattern, market_state, features)
            
            if decision.should_trade:
                if best_decision is None or decision.confluence_score > best_decision.confluence_score:
                    best_decision = decision
                    best_pattern = pattern
                    best_rationale = decision.rationale
        
        if not best_decision or not best_decision.should_trade:
             # Log the rejection of the last one?
             msg = best_decision.rejection_reason if best_decision else "All patterns rejected"
             rationale = best_decision.rationale if best_decision else []
             return {"decision": "REJECT", "reason": msg, "reasoning": rationale}
             
        # 4. Risk Calculation
        price = data_h1['close'].iloc[-1]
        sl = best_pattern.details.get('stop_loss', price)
        target_distance = best_pattern.details.get('height', 0)

        if target_distance <= 0:
            return {"decision": "REJECT", "reason": "No measured move target", "reasoning": best_rationale + ["Rejected: missing measured move"]}

        if best_pattern.direction == "bullish" and sl >= price:
            return {"decision": "REJECT", "reason": "Invalid SL for bullish setup", "reasoning": best_rationale + ["Rejected: SL above price"]}
        if best_pattern.direction == "bearish" and sl <= price:
            return {"decision": "REJECT", "reason": "Invalid SL for bearish setup", "reasoning": best_rationale + ["Rejected: SL below price"]}

        tp = price + target_distance if best_pattern.direction == "bullish" else price - target_distance
        risk = abs(price - sl)
        reward = abs(tp - price)
        if risk == 0:
            return {"decision": "REJECT", "reason": "Zero risk distance", "reasoning": best_rationale + ["Rejected: zero risk distance"]}
        rr = reward / risk
        if rr < 2.0:
            return {"decision": "REJECT", "reason": f"RR below 1:2 ({rr:.2f})", "reasoning": best_rationale + [f"Rejected: RR {rr:.2f} < 2.0"]}

        nn_features = self._build_nn_features(best_pattern, market_state, rr)
        nn_score = self.neural_model.predict(nn_features) if self.neural_model else None
        if nn_score is not None and self.neural_model.ready() and nn_score < 0.55:
            return {
                "decision": "REJECT",
                "reason": f"Neural filter rejected ({nn_score:.2f})",
                "reasoning": best_rationale + [f"Rejected: NN score {nn_score:.2f} < 0.55"],
            }

        reasoning_notes = self.reasoning_engine.build_reasoning(
            best_pattern,
            market_state,
            rr,
            best_rationale,
            price,
            sl,
            tp,
            nn_score
        )
        
        lot = self.risk_manager.calculate_lot_size(
            symbol, 
            price, 
            sl, 
            best_decision.confluence_score,
            account_info,
            symbol_info
        )
        
        if lot <= 0:
             return {"decision": "REJECT", "reason": "Risk Calc = 0 Lot", "reasoning": best_rationale + ["Rejected: lot size below minimum"]}

        self.pending_features[symbol] = nn_features

        return {
            "decision": "TRADE",
            "pattern": best_pattern,
            "lot": lot,
            "sl": sl,
            "tp": tp,
            "reason": f"Score {best_decision.confluence_score} | {best_decision.rejection_reason} | RR {rr:.2f}",
            "reasoning": reasoning_notes,
            "confidence": best_decision.confidence,
            "market_state": market_state
        }
        
    def log_result(self, symbol, result, profit):
        self.memory.close_trade(symbol, profit)
        label = 1 if profit > 0 else 0
        features = self.pending_features.pop(symbol, None)
        if features and self.neural_model:
            self.neural_model.train(features, label)
            self.neural_model.save(self.model_path)

    def _build_nn_features(self, pattern, market_state: dict, rr: float) -> list:
        volatility = market_state.get("volatility", "normal")
        volatility_score = {"low": -1.0, "normal": 0.0, "high": 1.0}.get(volatility, 0.0)
        pattern_id = self.pattern_index.get(pattern.name, 0)
        pattern_norm = pattern_id / max(len(self.pattern_index), 1)

        return [
            1.0 if pattern.direction == "bullish" else -1.0,
            float(pattern.push_count),
            float(pattern.volume_score),
            float(pattern.confidence),
            float(pattern.details.get("height", 0)),
            float(market_state.get("strength", 0)),
            float(market_state.get("session", 0)),
            float(market_state.get("global_trend", 0)),
            float(market_state.get("trend_d1", 0)),
            float(market_state.get("trend_h4", 0)),
            float(market_state.get("momentum_h1", 0)),
            float(volatility_score),
            float(pattern_norm),
            float(rr),
        ]

    def _load_neural_model(self) -> NeuralDecisionModel:
        if os.path.exists(self.model_path):
            try:
                model = NeuralDecisionModel.load(self.model_path)
                if model.input_size != self._feature_size():
                    return NeuralDecisionModel(input_size=self._feature_size())
                return model
            except Exception:
                return NeuralDecisionModel(input_size=self._feature_size())
        return NeuralDecisionModel(input_size=self._feature_size())

    def _feature_size(self) -> int:
        return len(self._build_nn_features(self._feature_stub(), {}, 2.0))

    def _feature_stub(self):
        class StubPattern:
            name = "Stub"
            direction = "bullish"
            push_count = 1
            volume_score = 0.5
            confidence = 0.5
            details = {"height": 0.0}
        return StubPattern()

    def _build_pattern_index(self) -> dict:
        patterns = [
            "Bull Flag",
            "Bear Flag",
            "Bullish Pennant",
            "Bearish Pennant",
            "Bullish Rectangle",
            "Bearish Rectangle",
            "Ascending Triangle",
            "Descending Triangle",
            "Symmetrical Triangle",
            "Rising Wedge",
            "Falling Wedge",
            "Head and Shoulders (Top)",
            "Head and Shoulders (Bottom)",
            "Double Top",
            "Double Bottom",
            "Bullish Diamond",
            "Bearish Diamond",
            "Rounding Top",
            "Rounding Bottom",
            "Teacup",
        ]
        return {name: idx + 1 for idx, name in enumerate(patterns)}


class ReasoningEngine:
    def build_reasoning(self, pattern, market_state: dict, rr: float, validator_notes: list, price: float, sl: float, tp: float, nn_score: Optional[float]) -> list:
        notes = list(validator_notes)
        notes.append(f"Pattern: {pattern.name} ({pattern.direction})")
        notes.append(f"Pattern height: {pattern.details.get('height', 0):.5f}")
        notes.append(f"SL distance: {abs(price - sl):.5f}")
        notes.append(f"TP distance: {abs(tp - price):.5f}")
        notes.append(f"RR check: {rr:.2f} >= 2.0")
        if nn_score is not None:
            notes.append(f"Neural score: {nn_score:.2f}")
        notes.append(f"Session score: {market_state.get('session', 0):.2f}")
        notes.append(f"Strength score: {market_state.get('strength', 0):.2f}")
        return notes
