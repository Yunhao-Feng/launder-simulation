from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, List


@dataclasses.dataclass
class Bank:
    bank_id: str
    name: str
    supported_currencies: List[str]
    transfer_fee: float = 0.0
    interbank_delay_days: int = 0

    def supports(self, currency: str) -> bool:
        return currency in self.supported_currencies


def load_banks(raw_banks: Iterable[Dict[str, object]] | None) -> Dict[str, Bank]:
    """Create a mapping of bank_id -> :class:`Bank` with sensible defaults."""

    banks: Dict[str, Bank] = {}
    for entry in raw_banks or []:
        bank = Bank(
            bank_id=str(entry.get("bank_id")),
            name=str(entry.get("name", entry.get("bank_id", "Bank"))),
            supported_currencies=list(entry.get("supported_currencies", [])) or ["CNY"],
            transfer_fee=float(entry.get("transfer_fee", 0.0)),
            interbank_delay_days=int(entry.get("interbank_delay_days", 0)),
        )
        banks[bank.bank_id] = bank

    if not banks:
        default_bank = Bank("B1", "Default Bank", ["CNY"], transfer_fee=0.0, interbank_delay_days=0)
        banks[default_bank.bank_id] = default_bank
    return banks