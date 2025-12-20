from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn.functional import cosine_similarity

from .knowledge import DPRTextEmbedder


def now_timestamp() -> str:
    return datetime.utcnow().isoformat()


dataclass_options = dict(frozen=True)


class CommonSenseMemory:
    """Lightweight commonsense snippets specialised for AML patterns."""

    def __init__(self) -> None:
        self.common_sense = {
            "Banking Oversight": [
                "Bank tellers must aggregate same-day deposits when screening for SARs.",
                "Amounts just under reportable thresholds are often structuring attempts.",
                "KYC anomalies (new address, sudden cash activity) should trigger manual review.",
            ],
            "Layering Patterns": [
                "Fan-out transfers split a lump sum into many small payouts across accounts.",
                "Scatter-gather typology sends small amounts to many recipients before reconsolidation.",
                "Cycles and peel chains try to obfuscate origin via repeated circular payments.",
            ],
            "Integration Cues": [
                "Inflated payroll or supplier invoices can hide illicit proceeds.",
                "Cash-heavy businesses often disguise placement as daily revenue spikes.",
                "Inconsistent tax filings versus bank inflows indicate suspicious integration.",
            ],
        }
    
    def retrieve(self, knowledge_types: Optional[Sequence[str]] = None) -> str:
        commonsense_prompt = "\n"
        selected_keys = knowledge_types or self.common_sense.keys()
        for knowledge_type in selected_keys:
            if knowledge_type not in self.common_sense:
                continue
            commonsense_prompt += ("*" * 5 + knowledge_type + ":" + "*" * 5 + "\n")
            for rule in self.common_sense[knowledge_type]:
                commonsense_prompt += ("- " + rule + "\n")
        return commonsense_prompt.strip()


@dataclasses.dataclass
class MemoryEntry:
    text: str
    timestamp: str
    embedding: torch.Tensor
    importance: float
    event_type: str | None = None


class AgentMemory:
    """Vector-backed episodic memory for agents."""

    def __init__(
        self,
        max_items: int = 500,
        commonsense: Optional[CommonSenseMemory] = None,
        embedder: DPRTextEmbedder | None = None,
    ) -> None:
        self.max_items = max_items
        self.entries: List[MemoryEntry] = []
        self.commonsense = commonsense or CommonSenseMemory()
        self.embedder = embedder or DPRTextEmbedder()

    def _compute_importance(self, entry: str, event_type: str | None = None) -> float:
        """Heuristically score a memory using event type and content.

        High-impact regulatory events (SAR, investigations) receive the strongest
        weight, followed by financial windfalls, relationship shifts, and gossip.
        """

        base = 1.0
        if event_type:
            if event_type.lower() in {"sar", "investigation", "regulator_alert"}:
                return 2.6
            if event_type.lower() in {"plan", "reflection"}:
                return 1.6
            if event_type.lower() in {"parcel", "deposit", "business_intake"}:
                base = 1.3
        lowered = entry.lower()
        if any(keyword in lowered for keyword in ["sar", "suspicious", "investigation", "flagged"]):
            base = max(base, 2.2)
        if any(keyword in lowered for keyword in ["profit", "windfall", "high margin", "big payout"]):
            base = max(base, 1.7)
        if any(keyword in lowered for keyword in ["rumor", "gossip", "tip", "warning"]):
            base = max(base, 1.2)
        if any(keyword in lowered for keyword in ["large deposit", "100000", "threshold"]):
            base = max(base, 1.5)
        return base

    def _recency_weight(self, timestamp: str) -> float:
        """Return a decay-based recency weight in [0.2, 1.0]."""

        try:
            ts = datetime.fromisoformat(timestamp)
            delta_days = max((datetime.utcnow() - ts).days, 0)
        except Exception:
            delta_days = 0
        return max(0.2, min(1.0, 1.0 / (1 + 0.2 * delta_days)))

    def add(self, entry: str, event_type: str | None = None, importance: float | None = None) -> None:
        """Add a generic memory entry with embedding, timestamp, and salience weight."""

        embedding = self.embedder.embed(entry)
        scored_importance = importance if importance is not None else self._compute_importance(entry, event_type)
        self.entries.append(
            MemoryEntry(
                text=entry,
                timestamp=now_timestamp(),
                embedding=embedding,
                importance=float(scored_importance),
                event_type=event_type,
            )
        )
        if len(self.entries) > self.max_items:
            self.entries = self.entries[-self.max_items :]

    def add_important(self, entry: str) -> None:
        """Add an important memory with an explicit salience boost."""

        marked = f"[IMPORTANT] {entry}"
        self.add(marked, importance=2.8, event_type="important")

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        if not self.entries:
            return []
        query_vec = self.embedder.embed(query)
        embeddings = torch.stack([e.embedding for e in self.entries], dim=0)
        sims = cosine_similarity(embeddings, query_vec.unsqueeze(0), dim=1)

        recency_weights = torch.tensor([self._recency_weight(e.timestamp) for e in self.entries], device=sims.device)
        importance_weights = torch.tensor([e.importance for e in self.entries], device=sims.device)
        salience = importance_weights * recency_weights
        weighted = sims * 0.6 + salience * 0.4
        top_indices = torch.argsort(weighted, descending=True)[:top_k]
        return [f"[{self.entries[i].timestamp}] {self.entries[i].text}" for i in top_indices.tolist()]

    def commonsense_snapshot(self, topics: Optional[Sequence[str]] = None) -> str:
        return self.commonsense.retrieve(topics)

    def important_snapshot(self, top_k: int = 3) -> List[str]:
        """Return the most recent high-importance memories."""

        important_entries = sorted(
            [e for e in self.entries if e.text.startswith("[IMPORTANT]")],
            key=lambda e: e.importance * self._recency_weight(e.timestamp),
            reverse=True,
        )
        return [f"[{e.timestamp}] {e.text}" for e in important_entries[:top_k]]

    def salient_summary(self, top_k: int = 5) -> str:
        """Summarise the most salient memories for prompt conditioning."""

        if not self.entries:
            return "(no salient memories)"
        scored = [
            (e, e.importance * self._recency_weight(e.timestamp))
            for e in self.entries
        ]
        scored.sort(key=lambda tup: tup[1], reverse=True)
        lines = []
        for entry, score in scored[:top_k]:
            tag = entry.event_type or "memory"
            lines.append(f"- [{tag} | {entry.timestamp} | salience {score:.2f}] {entry.text}")
        return "\n".join(lines)


@dataclasses.dataclass(**dataclass_options)
class EventLog:
    log_id: int
    timestamp: str
    agent_id: str
    agent_role: str
    event_type: str
    target_id: str | None
    description: str


@dataclasses.dataclass(**dataclass_options)
class Transaction:
    tx_id: int
    timestamp: str
    sender_account: str
    receiver_account: str
    amount: float
    channel: str
    clean_amount: float
    illicit_amount: float
    illicit_fraction: float
    currency: str
    fee_amount: float
    tx_type: str
    cross_bank: bool
    cross_currency: bool
    is_money_laundering: bool
    ml_typology: str | None
    ml_pattern: str | None
    pattern_scheme_id: str | int | None
    settlement_day: int | None


@dataclasses.dataclass
class AccountState:
    account_id: str
    bank_id: str
    currency: str
    clean_balance: float = 0.0
    illicit_balance: float = 0.0


class EventRecorder:
    def __init__(
        self,
        banks: Dict[str, object] | None = None,
        fx_rates: Dict[Tuple[str, str], float] | Dict[str, float] | None = None,
        default_currency: str = "CNY",
    ) -> None:
        self.logs: List[EventLog] = []
        self.transactions: List[Transaction] = []
        self.accounts: Dict[str, AccountState] = {}
        self.pending_settlements: List[Tuple[int, str, float, float]] = []
        self.banks = banks or {}
        self.fx_rates: Dict[Tuple[str, str], float] = {}
        if fx_rates:
            for k, v in fx_rates.items():
                if isinstance(k, tuple):
                    self.fx_rates[(str(k[0]), str(k[1]))] = float(v)
                else:
                    # string key formatted as "SRC-DEST"
                    if "-" in k:
                        src, dst = k.split("-", 1)
                        self.fx_rates[(src, dst)] = float(v)
        self.default_currency = default_currency

    def register_account(
        self,
        account_id: str,
        bank_id: str | None = None,
        currency: str | None = None,
        clean_balance: float = 0.0,
        illicit_balance: float = 0.0,
    ) -> AccountState:
        existing = self.accounts.get(account_id)
        if existing:
            return existing
        chosen_bank = bank_id or next(iter(self.banks.keys()), "B1")
        chosen_currency = currency or self.default_currency
        state = AccountState(
            account_id=account_id,
            bank_id=chosen_bank,
            currency=chosen_currency,
            clean_balance=float(clean_balance),
            illicit_balance=float(illicit_balance),
        )
        self.accounts[account_id] = state
        return state

    def _get_account(self, account_id: str) -> AccountState:
        return self.accounts.get(account_id) or self.register_account(account_id)

    def release_settlements(self, current_day: int) -> None:
        ready, future = [], []
        for due_day, account_id, clean_amt, illicit_amt in self.pending_settlements:
            (ready if due_day <= current_day else future).append((due_day, account_id, clean_amt, illicit_amt))
        self.pending_settlements = future
        for _, account_id, clean_amt, illicit_amt in ready:
            acct = self._get_account(account_id)
            acct.clean_balance += clean_amt
            acct.illicit_balance += illicit_amt

    def _apply_outgoing(self, account: AccountState, total_amount: float) -> float:
        total_balance = account.clean_balance + account.illicit_balance
        if total_balance <= 0:
            return 0.0
        illicit_fraction = account.illicit_balance / total_balance if total_balance else 0.0
        illicit_deduction = min(account.illicit_balance, total_amount * illicit_fraction)
        clean_deduction = total_amount - illicit_deduction
        account.clean_balance = max(0.0, account.clean_balance - clean_deduction)
        account.illicit_balance = max(0.0, account.illicit_balance - illicit_deduction)
        return illicit_fraction

    def _apply_incoming(self, account: AccountState, clean_amount: float, illicit_amount: float) -> None:
        account.clean_balance += clean_amount
        account.illicit_balance += illicit_amount

    def record_event(
        self,
        agent_id: str,
        agent_role: str,
        event_type: str,
        description: str,
        target_id: Optional[str] = None,
    ) -> EventLog:
        log = EventLog(
            log_id=len(self.logs) + 1,
            timestamp=now_timestamp(),
            agent_id=agent_id,
            agent_role=agent_role,
            event_type=event_type,
            target_id=target_id,
            description=description,
        )
        self.logs.append(log)
        return log
    
    def _convert_currency(self, amount: float, src_currency: str, dst_currency: str) -> float:
        if src_currency == dst_currency:
            return amount
        rate = self.fx_rates.get((src_currency, dst_currency), 1.0)
        return amount * rate

    def record_transaction(
        self,
        sender_account: str,
        receiver_account: str,
        amount: float,
        currency: str,
        tx_type: str,
        channel: str | None,
        is_money_laundering: bool,
        ml_typology: str | None,
        ml_pattern: str | None = None,
        pattern_scheme_id: str | int | None = None,
        current_day: int | None = None,
    ) -> Transaction:
        sender_state = self._get_account(sender_account)
        receiver_state = self._get_account(receiver_account)

        sender_total = sender_state.clean_balance + sender_state.illicit_balance
        illicit_fraction = sender_state.illicit_balance / sender_total if sender_total else 0.0
        illicit_amount = float(amount) * illicit_fraction
        clean_amount = float(amount) - illicit_amount
        tx_currency = currency or sender_state.currency

        sender_bank = self.banks.get(sender_state.bank_id)
        receiver_bank = self.banks.get(receiver_state.bank_id)
        cross_bank = sender_state.bank_id != receiver_state.bank_id if (sender_bank and receiver_bank) else False
        fee_amount = float(amount) * sender_bank.transfer_fee if (sender_bank and cross_bank) else 0.0
        settlement_day = current_day
        if cross_bank and sender_bank:
            settlement_day = (current_day or 0) + sender_bank.interbank_delay_days

        # Deduct outgoing amount and fee proportionally
        self._apply_outgoing(sender_state, float(amount) + fee_amount)

        incoming_clean = clean_amount
        incoming_illicit = illicit_amount
        if sender_state.currency != receiver_state.currency:
            incoming_clean = self._convert_currency(clean_amount, sender_state.currency, receiver_state.currency)
            incoming_illicit = self._convert_currency(illicit_amount, sender_state.currency, receiver_state.currency)

        if cross_bank and settlement_day is not None and current_day is not None and settlement_day > current_day:
            self.pending_settlements.append((settlement_day, receiver_state.account_id, incoming_clean, incoming_illicit))
        else:
            self._apply_incoming(receiver_state, incoming_clean, incoming_illicit)
            
        tx = Transaction(
            tx_id=len(self.transactions) + 1,
            timestamp=now_timestamp(),
            sender_account=sender_account,
            receiver_account=receiver_account,
            amount=float(amount),
            channel=channel or "wire",
            clean_amount=clean_amount,
            illicit_amount=illicit_amount,
            illicit_fraction=illicit_fraction,
            currency=tx_currency,
            fee_amount=fee_amount,
            tx_type=tx_type,
            cross_bank=cross_bank,
            cross_currency=sender_state.currency != receiver_state.currency,
            is_money_laundering=is_money_laundering,
            ml_typology=ml_typology,
            ml_pattern=ml_pattern,
            pattern_scheme_id=pattern_scheme_id,
            settlement_day=settlement_day,
        )
        self.transactions.append(tx)
        return tx

    def export_logs(self, path: str) -> None:
        import csv

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "log_id",
                "timestamp",
                "agent_id",
                "agent_role",
                "event_type",
                "target_id",
                "description",
            ])
            for log in self.logs:
                writer.writerow(
                    [
                        log.log_id,
                        log.timestamp,
                        log.agent_id,
                        log.agent_role,
                        log.event_type,
                        log.target_id or "",
                        log.description,
                    ]
                )

    def export_transactions(self, path: str) -> None:
        import csv

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "tx_id",
                    "timestamp",
                    "sender_account",
                    "receiver_account",
                    "amount",
                    "channel",
                    "clean_amount",
                    "illicit_amount",
                    "illicit_fraction",
                    "currency",
                    "fee_amount",
                    "tx_type",
                    "cross_bank",
                    "cross_currency",
                    "is_money_laundering",
                    "ml_typology",
                    "ml_pattern",
                    "pattern_scheme_id",
                    "settlement_day",
                ]
            )
            for tx in self.transactions:
                writer.writerow(
                    [
                        tx.tx_id,
                        tx.timestamp,
                        tx.sender_account,
                        tx.receiver_account,
                        tx.amount,
                        tx.channel,
                        tx.clean_amount,
                        tx.illicit_amount,
                        tx.illicit_fraction,
                        tx.currency,
                        tx.fee_amount,
                        tx.tx_type,
                        int(tx.cross_bank),
                        int(tx.cross_currency),
                        int(tx.is_money_laundering),
                        tx.ml_typology or "",
                        tx.ml_pattern or "",
                        tx.pattern_scheme_id or "",
                        tx.settlement_day if tx.settlement_day is not None else "",
                    ]
                )
