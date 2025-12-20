from __future__ import annotations

import random
from typing import Dict, List, Optional


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
        self.dynamic_rules: List[Dict[str, object]] = []

    def set_threshold(self, new_threshold: float) -> None:
        """Update the SAR trigger threshold at runtime."""

        self.threshold = float(new_threshold)

    def update_channel_weight(self, channel: str, weight: float) -> None:
        """Adjust the penalty weight for a specific payment rail."""

        self.channel_risk[channel] = float(weight)

    def bulk_update_channel_weights(self, updates: Dict[str, float]) -> None:
        for channel, weight in updates.items():
            self.update_channel_weight(channel, weight)

    def add_rule(
        self,
        description: str,
        tx_type: Optional[str] = None,
        amount_threshold: Optional[float] = None,
        channel: Optional[str] = None,
        penalty: float = 0.15,
    ) -> None:
        """Register a new rule such as 'flag deposits above 50k'."""

        rule = {
            "description": description,
            "tx_type": tx_type,
            "amount_threshold": amount_threshold,
            "channel": channel,
            "penalty": penalty,
        }
        self.dynamic_rules.append(rule)

    def active_policy_summary(self) -> str:
        if not self.dynamic_rules:
            return "No custom SAR/KYC rules."
        return "\n".join([f"- {rule['description']}" for rule in self.dynamic_rules])

    def _policy_penalty(self, amount: float, tx_type: str | None, channel: str | None) -> float:
        penalty = 0.0
        for rule in self.dynamic_rules:
            matches_type = rule.get("tx_type") is None or rule.get("tx_type") == tx_type
            matches_channel = rule.get("channel") is None or rule.get("channel") == channel
            amount_threshold = rule.get("amount_threshold")
            matches_amount = amount_threshold is None or amount >= float(amount_threshold)
            if matches_type and matches_channel and matches_amount:
                penalty += float(rule.get("penalty", 0.1))
        return penalty

    def score_transaction(
        self,
        amount: float,
        is_laundering: bool = False,
        illicit_amount: float = 0.0,
        illicit_fraction: float = 0.0,
        cross_bank: bool = False,
        cross_currency: bool = False,
        channel: str | None = None,
        tx_type: str | None = None,
    ) -> float:
        base = amount / self.high_risk_amount
        illicit_weight = illicit_fraction * self.illicit_fraction_weight
        illicit_magnitude = (illicit_amount / max(self.high_risk_amount, 1)) * self.illicit_amount_weight
        cross_penalty = self.cross_border_weight * (1 if cross_bank else 0) + (self.cross_border_weight * 0.5 if cross_currency else 0)
        channel_penalty = self.channel_risk.get(channel or "wire", 0.05)
        policy_penalty = self._policy_penalty(amount, tx_type=tx_type, channel=channel)
        noise = random.random() * 0.2
        score = min(1.0, base + illicit_weight + illicit_magnitude + cross_penalty + channel_penalty + policy_penalty + noise)
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
        tx_type: str | None = None,
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
                tx_type=tx_type,
            )
            >= self.threshold
        )
