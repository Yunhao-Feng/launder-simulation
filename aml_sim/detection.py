from __future__ import annotations

import random
from typing import Dict


class RiskModel:
    def __init__(self, threshold: float, config: Dict[str, float]):
        self.threshold = threshold
        self.laundering_amount_threshold = config.get("laundering_amount_threshold", 80000)
        self.high_risk_amount = config.get("high_risk_amount", 100000)

    def score_transaction(
        self,
        amount: float,
        is_laundering: bool = False,
        illicit_amount: float = 0.0,
        illicit_fraction: float = 0.0,
    ) -> float:
        base = amount / self.high_risk_amount
        illicit_weight = illicit_fraction * 0.4
        illicit_magnitude = illicit_amount / max(self.high_risk_amount, 1)
        noise = random.random() * 0.2
        score = min(1.0, base + illicit_weight + illicit_magnitude + noise)
        if is_laundering:
            score = min(1.0, score + 0.3)
        if amount >= self.laundering_amount_threshold:
            score = min(1.0, score + 0.2)
        return score

    def is_high_risk(self, amount: float, is_laundering: bool = False) -> bool:
        return self.score_transaction(amount, is_laundering) >= self.threshold