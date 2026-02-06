from market_context import MarketContextAnalyzer
from pattern_recognition import PatternRecognizer
from trade_validator import TradeValidator
from adaptive_risk import AdaptiveRiskManager
from trading_memory import TradingMemory
from daily_planner import DailyPlanner
import pandas as pd

class AIBrain:
    """
    Human-like AI Brain with Daily Planning.
    Only trades in the direction of the daily bias.
    """
    def __init__(self):
        self.market_analyzer = MarketContextAnalyzer()
        self.trade_validator = TradeValidator()
        self.risk_manager = AdaptiveRiskManager()
        self.memory = TradingMemory()
<<<<<<< Updated upstream
        self.reasoning_engine = ReasoningEngine()
        self.required_columns = {"open", "high", "low", "close"}

    def _sanitize_data(self, data: pd.DataFrame, name: str) -> pd.DataFrame:
        if data is None or data.empty:
            raise ValueError(f"{name} data is empty")
        missing = self.required_columns - set(data.columns)
        if missing:
            raise ValueError(f"{name} data missing columns: {', '.join(sorted(missing))}")
        cleaned = data.dropna(subset=list(self.required_columns)).copy()
        if cleaned.empty:
            raise ValueError(f"{name} data has no usable rows after cleanup")
        if "time" in cleaned.columns:
            cleaned = cleaned.sort_values("time")
        return cleaned

    def _validate_symbol_info(self, symbol_info) -> None:
        required_attrs = ("point", "volume_step", "volume_min", "volume_max", "trade_tick_value")
        missing = [attr for attr in required_attrs if not hasattr(symbol_info, attr)]
        if missing:
            raise ValueError(f"symbol_info missing fields: {', '.join(missing)}")

    def _fallback_stop_loss(self, pattern, data_h1: pd.DataFrame) -> float:
        lookback = data_h1.tail(20)
        if pattern.direction == "bullish":
            return float(lookback["low"].min())
        return float(lookback["high"].max())

    def _fallback_target_distance(self, data_h1: pd.DataFrame, risk: float) -> float:
        lookback = data_h1.tail(20)
        recent_range = float(lookback["high"].max() - lookback["low"].min())
        return max(recent_range, risk * 2)
        
    def think(self, symbol: str, data_h1: pd.DataFrame, data_h4: pd.DataFrame, data_d1: pd.DataFrame, account_info, symbol_info) -> dict:
        """
        Returns a dict with decision and execution details.
        """
        try:
            data_h1 = self._sanitize_data(data_h1, "H1")
            data_h4 = self._sanitize_data(data_h4, "H4")
            data_d1 = self._sanitize_data(data_d1, "D1")
            self._validate_symbol_info(symbol_info)
        except ValueError as exc:
            return {"decision": "REJECT", "reason": str(exc)}

        # 0. Check Memory (Revenge Trading / Kill Switch)
        if not self.memory.can_trade(symbol):
            return {"decision": "REJECT", "reason": "Memory Block (Loss Streak or Cooldown)"}

        if len(data_h1) < 60 or len(data_h4) < 50 or len(data_d1) < 20:
            return {"decision": "WAIT", "reason": "Insufficient market history"}
=======
        self.planner = DailyPlanner()
        
    def set_daily_plan(self, plan: dict):
        """Set the daily plan from external source."""
        self.planner.plan = plan
        
    def think(self, symbol: str, data_h1: pd.DataFrame, data_h4: pd.DataFrame, data_d1: pd.DataFrame, account_info, symbol_info) -> dict:
        # 0. Check Daily Bias
        daily_bias = self.planner.get_bias(symbol)
        if daily_bias == "NEUTRAL":
            return {"decision": "REJECT", "reason": "Daily Bias: NEUTRAL - No trading"}
        
        # 1. Check Memory
        if not self.memory.can_trade(symbol):
            return {"decision": "REJECT", "reason": "Memory Block"}
>>>>>>> Stashed changes
            
        # 2. Market Context
        market_state = self.market_analyzer.get_market_state(symbol, data_h1, data_h4, data_d1)
        
        # 3. Pattern Recognition - All Timeframes
        all_patterns = []
        
        for tf_name, df in [('H1', data_h1), ('H4', data_h4), ('D1', data_d1)]:
            if df is None: continue
            recognizer = PatternRecognizer(df)
            patterns = recognizer.detect_all()
            fresh = [p for p in patterns if p.index_end >= len(df) - 2]
            for p in fresh:
                all_patterns.append((tf_name, p, df))
        
        if not all_patterns:
            return {"decision": "WAIT", "reason": "No fresh patterns"}
        
        # 4. Filter by Daily Bias
        valid_patterns = []
        for tf, pattern, data in all_patterns:
            pattern_dir = 'bullish' if pattern.direction == 'bullish' else 'bearish'
            bias_match = (daily_bias == "LONG" and pattern_dir == "bullish") or \
                         (daily_bias == "SHORT" and pattern_dir == "bearish")
            if bias_match:
                valid_patterns.append((tf, pattern, data))
        
        if not valid_patterns:
            return {"decision": "REJECT", "reason": f"Patterns don't match {daily_bias} bias"}
            
        # 5. Validation & Selection
        best_decision = None
        best_pattern = None
<<<<<<< Updated upstream
        best_rationale = []
=======
        best_tf = None
        best_data = None
>>>>>>> Stashed changes
        
        for tf, pattern, data in valid_patterns:
            features = {"vol_anomaly": pattern.volume_score * 2}
            decision = self.trade_validator.validate(pattern, market_state, features)
            
            if decision.should_trade:
                if best_decision is None or decision.confluence_score > best_decision.confluence_score:
                    best_decision = decision
                    best_pattern = pattern
<<<<<<< Updated upstream
                    best_rationale = decision.rationale
        
        if not best_decision or not best_decision.should_trade:
             # Log the rejection of the last one?
             msg = best_decision.rejection_reason if best_decision else "All patterns rejected"
             rationale = best_decision.rationale if best_decision else []
             return {"decision": "REJECT", "reason": msg, "reasoning": rationale}
             
        # 4. Risk Calculation
        price = data_h1['close'].iloc[-1]
        sl = best_pattern.details.get('stop_loss')
        if sl is None:
            sl = self._fallback_stop_loss(best_pattern, data_h1)
        sl = float(sl)
        target_distance = best_pattern.details.get('height', 0)

        risk = abs(price - sl)
        if risk == 0:
            return {"decision": "REJECT", "reason": "Zero risk distance", "reasoning": best_rationale + ["Rejected: zero risk distance"]}

        if target_distance <= 0:
            target_distance = self._fallback_target_distance(data_h1, risk)
            if target_distance <= 0:
                return {"decision": "REJECT", "reason": "No measured move target", "reasoning": best_rationale + ["Rejected: missing measured move"]}

        if best_pattern.direction == "bullish" and sl >= price:
            return {"decision": "REJECT", "reason": "Invalid SL for bullish setup", "reasoning": best_rationale + ["Rejected: SL above price"]}
        if best_pattern.direction == "bearish" and sl <= price:
            return {"decision": "REJECT", "reason": "Invalid SL for bearish setup", "reasoning": best_rationale + ["Rejected: SL below price"]}

        tp = price + target_distance if best_pattern.direction == "bullish" else price - target_distance
        reward = abs(tp - price)
        rr = reward / risk
        if rr < 2.0:
            return {"decision": "REJECT", "reason": f"RR below 1:2 ({rr:.2f})", "reasoning": best_rationale + [f"Rejected: RR {rr:.2f} < 2.0"]}

        reasoning_notes = self.reasoning_engine.build_reasoning(
            best_pattern,
            market_state,
            rr,
            best_rationale,
            price,
            sl,
            tp
        )
=======
                    best_tf = tf
                    best_data = data
        
        if not best_decision or not best_decision.should_trade:
            msg = best_decision.rejection_reason if best_decision else "All patterns rejected"
            return {"decision": "REJECT", "reason": msg}
             
        # 6. Risk Calculation
        price = best_data['close'].iloc[-1]
        sl = best_pattern.details.get('stop_loss', price)
>>>>>>> Stashed changes
        
        lot = self.risk_manager.calculate_lot_size(
            symbol, price, sl, 
            best_decision.confluence_score,
            account_info, symbol_info
        )
        
        if lot <= 0:
<<<<<<< Updated upstream
             return {"decision": "REJECT", "reason": "Risk Calc = 0 Lot", "reasoning": best_rationale + ["Rejected: lot size below minimum"]}
=======
            return {"decision": "REJECT", "reason": "Risk Calc = 0 Lot"}
>>>>>>> Stashed changes
             
        return {
            "decision": "TRADE",
            "pattern": best_pattern,
            "lot": lot,
            "sl": sl,
<<<<<<< Updated upstream
            "tp": tp,
            "reason": f"Score {best_decision.confluence_score} | {best_decision.rejection_reason} | RR {rr:.2f}",
            "reasoning": reasoning_notes,
=======
            "tp": best_pattern.details.get('height', 0) + price,
            "reason": f"[{best_tf}] Score {best_decision.confluence_score} | {daily_bias}",
>>>>>>> Stashed changes
            "confidence": best_decision.confidence,
            "market_state": market_state
        }
        
    def log_result(self, symbol, result, profit):
        self.memory.close_trade(symbol, profit)


class ReasoningEngine:
    def build_reasoning(self, pattern, market_state: dict, rr: float, validator_notes: list, price: float, sl: float, tp: float) -> list:
        notes = list(validator_notes)
        notes.append(f"Pattern: {pattern.name} ({pattern.direction})")
        notes.append(f"Pattern height: {pattern.details.get('height', 0):.5f}")
        notes.append(f"SL distance: {abs(price - sl):.5f}")
        notes.append(f"TP distance: {abs(tp - price):.5f}")
        notes.append(f"RR check: {rr:.2f} >= 2.0")
        notes.append(f"Session score: {market_state.get('session', 0):.2f}")
        notes.append(f"Strength score: {market_state.get('strength', 0):.2f}")
        return notes
