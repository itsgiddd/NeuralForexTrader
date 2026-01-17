import importlib.util
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from market_context import MarketContextAnalyzer
from pattern_recognition import PatternRecognizer


@dataclass
class PatternSample:
    features: List[float]
    label: int
    timeframe: str
    pattern_name: str


QUALITY_GRADE_MAP = {
    "A": 1.0,
    "B": 0.75,
    "C": 0.5,
    "D": 0.25,
}


def _window_slice(data: pd.DataFrame, end_index: int) -> pd.DataFrame:
    end_index = min(end_index, len(data) - 1)
    return data.iloc[: end_index + 1]


def _label_outcome(
    data: pd.DataFrame,
    entry_index: int,
    direction: str,
    horizon: int,
    threshold_pct: float,
) -> int:
    entry_index = min(entry_index, len(data) - 1)
    entry_price = float(data["close"].iloc[entry_index])
    future = data.iloc[entry_index + 1 : entry_index + 1 + horizon]
    if future.empty:
        return 0

    max_high = float(future["high"].max())
    min_low = float(future["low"].min())
    threshold_move = entry_price * threshold_pct

    if direction == "bullish":
        favorable = max_high - entry_price
        adverse = entry_price - min_low
    else:
        favorable = entry_price - min_low
        adverse = max_high - entry_price

    return int(favorable >= threshold_move and adverse <= threshold_move)


def build_pattern_dataset(
    data_h1: pd.DataFrame,
    data_h4: pd.DataFrame,
    data_d1: pd.DataFrame,
    horizon: int = 24,
    threshold_pct: float = 0.002,
) -> List[PatternSample]:
    analyzer = MarketContextAnalyzer()
    samples: List[PatternSample] = []

    for timeframe, data in {"H1": data_h1, "H4": data_h4, "D1": data_d1}.items():
        recognizer = PatternRecognizer(data)
        patterns = recognizer.detect_all()
        for pattern in patterns:
            pattern.timeframe = timeframe
            slice_data = _window_slice(data, pattern.index_end)
            market_state = analyzer.get_market_state("FX", data_h1, data_h4, data_d1)
            trend_alignment = 1.0 if (
                market_state.get("global_trend", 0) == 1 and pattern.direction == "bullish"
                or market_state.get("global_trend", 0) == -1 and pattern.direction == "bearish"
            ) else 0.0
            quality = QUALITY_GRADE_MAP.get(pattern.quality_grade, 0.5)

            features = [
                float(pattern.confidence),
                float(pattern.push_count),
                float(pattern.volume_score),
                quality,
                float(market_state.get("strength", 0)),
                float(market_state.get("session", 0)),
                float(market_state.get("momentum_h1", 0)),
                trend_alignment,
            ]

            label = _label_outcome(
                slice_data,
                pattern.index_end,
                pattern.direction,
                horizon,
                threshold_pct,
            )

            samples.append(
                PatternSample(
                    features=features,
                    label=label,
                    timeframe=timeframe,
                    pattern_name=pattern.name,
                )
            )

    return samples


def train_tensorflow_model(
    samples: List[PatternSample],
    epochs: int = 25,
    batch_size: int = 32,
) -> Tuple["tf.keras.Model", pd.DataFrame]:
    if importlib.util.find_spec("tensorflow") is None:
        raise RuntimeError("TensorFlow is not installed. Install it to train the model.")

    import tensorflow as tf

    features = np.array([sample.features for sample in samples], dtype=np.float32)
    labels = np.array([sample.label for sample in samples], dtype=np.float32)

    if len(features) == 0:
        raise ValueError("No samples available for training.")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(features.shape[1],)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(features, labels, epochs=epochs, batch_size=batch_size, verbose=1)

    history_df = pd.DataFrame(history.history)
    return model, history_df
