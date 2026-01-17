import json
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class NeuralDecisionModel:
    input_size: int
    hidden_sizes: List[int] = field(default_factory=lambda: [32, 16])
    learning_rate: float = 0.005
    min_samples: int = 200
    weights1: np.ndarray = field(init=False)
    bias1: np.ndarray = field(init=False)
    weights2: np.ndarray = field(init=False)
    bias2: np.ndarray = field(init=False)
    weights3: np.ndarray = field(init=False)
    bias3: np.ndarray = field(init=False)
    samples_seen: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(42)
        h1, h2 = self.hidden_sizes
        self.weights1 = rng.normal(0, 0.1, (self.input_size, h1))
        self.bias1 = np.zeros(h1)
        self.weights2 = rng.normal(0, 0.1, (h1, h2))
        self.bias2 = np.zeros(h2)
        self.weights3 = rng.normal(0, 0.1, (h2, 1))
        self.bias3 = np.zeros(1)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def predict(self, features: List[float]) -> float:
        x = np.array(features, dtype=float)
        h1 = self._relu(x @ self.weights1 + self.bias1)
        h2 = self._relu(h1 @ self.weights2 + self.bias2)
        y = self._sigmoid(h2 @ self.weights3 + self.bias3)
        return float(y.squeeze())

    def train(self, features: List[float], label: int):
        x = np.array(features, dtype=float)
        y_true = np.array([label], dtype=float)

        h1 = self._relu(x @ self.weights1 + self.bias1)
        h2 = self._relu(h1 @ self.weights2 + self.bias2)
        y_pred = self._sigmoid(h2 @ self.weights3 + self.bias3)

        # Binary cross-entropy derivative
        error = y_pred - y_true
        grad_w3 = np.outer(h2, error)
        grad_b3 = error

        grad_h2 = error @ self.weights3.T
        grad_h2[h2 <= 0] = 0
        grad_w2 = np.outer(h1, grad_h2)
        grad_b2 = grad_h2

        grad_h1 = grad_h2 @ self.weights2.T
        grad_h1[h1 <= 0] = 0
        grad_w1 = np.outer(x, grad_h1)
        grad_b1 = grad_h1

        self.weights3 -= self.learning_rate * grad_w3
        self.bias3 -= self.learning_rate * grad_b3
        self.weights2 -= self.learning_rate * grad_w2
        self.bias2 -= self.learning_rate * grad_b2
        self.weights1 -= self.learning_rate * grad_w1
        self.bias1 -= self.learning_rate * grad_b1

        self.samples_seen += 1

    def ready(self) -> bool:
        return self.samples_seen >= self.min_samples

    def save(self, path: str):
        payload = {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "learning_rate": self.learning_rate,
            "min_samples": self.min_samples,
            "weights1": self.weights1.tolist(),
            "bias1": self.bias1.tolist(),
            "weights2": self.weights2.tolist(),
            "bias2": self.bias2.tolist(),
            "weights3": self.weights3.tolist(),
            "bias3": self.bias3.tolist(),
            "samples_seen": self.samples_seen,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "NeuralDecisionModel":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        model = cls(
            input_size=payload["input_size"],
            hidden_sizes=payload["hidden_sizes"],
            learning_rate=payload["learning_rate"],
            min_samples=payload["min_samples"],
        )
        model.weights1 = np.array(payload["weights1"], dtype=float)
        model.bias1 = np.array(payload["bias1"], dtype=float)
        model.weights2 = np.array(payload["weights2"], dtype=float)
        model.bias2 = np.array(payload["bias2"], dtype=float)
        model.weights3 = np.array(payload["weights3"], dtype=float)
        model.bias3 = np.array(payload["bias3"], dtype=float)
        model.samples_seen = payload.get("samples_seen", 0)
        return model
