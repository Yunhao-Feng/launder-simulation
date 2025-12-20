from __future__ import annotations

import random
from typing import Dict


class RiskModel:
    def __init__(self, threshold: float, config: Dict[str, float]):
        self.threshold = threshold
        self.laundering_amount_threshold = config.get("laundering_amount_threshold", 80000)
        self.high_risk_amount = config.get("high_risk_amount", 100000)
        self.illicit_fraction_weight = config.get("illicit_fraction_weight", 0.4)
        self.illicit_amount_weight = config.get("illicit_amount_weight", 0.5)
        self.cross_border_weight = config.get("cross_border_weight", 0.15)
        self.channel_risk = {
            "wire": 0.05,
            "ach": 0.04,
            "cheque": 0.02,
            "credit_card": 0.03,
            "cash": 0.08,
            "crypto": 0.12,
        }
        self.channel_risk.update(config.get("channel_risk", {}))

    def score_transaction(
        self,
        amount: float,
        is_laundering: bool = False,
        illicit_amount: float = 0.0,
        illicit_fraction: float = 0.0,
        cross_bank: bool = False,
        cross_currency: bool = False,
        channel: str | None = None,
    ) -> float:
        base = amount / self.high_risk_amount
        illicit_weight = illicit_fraction * self.illicit_fraction_weight
        illicit_magnitude = (illicit_amount / max(self.high_risk_amount, 1)) * self.illicit_amount_weight
        cross_penalty = self.cross_border_weight * (1 if cross_bank else 0) + (self.cross_border_weight * 0.5 if cross_currency else 0)
        channel_penalty = self.channel_risk.get(channel or "wire", 0.05)
        noise = random.random() * 0.2
        score = min(1.0, base + illicit_weight + illicit_magnitude + cross_penalty + channel_penalty + noise)
        if is_laundering:
            score = min(1.0, score + 0.3)
        if amount >= self.laundering_amount_threshold:
            score = min(1.0, score + 0.2)
        return score

    def is_high_risk(
        self,
        amount: float,
        is_laundering: bool = False,
        illicit_amount: float = 0.0,
        illicit_fraction: float = 0.0,
        cross_bank: bool = False,
        cross_currency: bool = False,
        channel: str | None = None,
    ) -> bool:
        return (
            self.score_transaction(
                amount,
                is_laundering=is_laundering,
                illicit_amount=illicit_amount,
                illicit_fraction=illicit_fraction,
                cross_bank=cross_bank,
                cross_currency=cross_currency,
                channel=channel,
            )
            >= self.threshold
        )
