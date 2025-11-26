from __future__ import annotations

import random
from typing import Dict


class RiskModel:
    def __init__(self, threshold: float, config: Dict[str, float]):
        self.threshold = threshold
        self.laundering_amount_threshold = config.get("laundering_amount_threshold", 80000)
        self.high_risk_amount = config.get("high_risk_amount", 100000)

    def score_transaction(self, amount: float, is_laundering: bool = False) -> float:
        base = amount / self.high_risk_amount
        noise = random.random() * 0.2
        score = min(1.0, base + noise)
        if is_laundering:
            score = min(1.0, score + 0.3)
        if amount >= self.laundering_amount_threshold:
            score = min(1.0, score + 0.2)
        return score

    def is_high_risk(self, amount: float, is_laundering: bool = False) -> bool:
        return self.score_transaction(amount, is_laundering) >= self.threshold