from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import List, Optional, Sequence

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


class AgentMemory:
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

    def add(self, entry: str) -> None:
        embedding = self.embedder.embed(entry)
        self.entries.append(MemoryEntry(text=entry, timestamp=now_timestamp(), embedding=embedding))
        if len(self.entries) > self.max_items:
            self.entries = self.entries[-self.max_items :]

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        if not self.entries:
            return []
        query_vec = self.embedder.embed(query)
        embeddings = torch.stack([e.embedding for e in self.entries], dim=0)
        sims = cosine_similarity(embeddings, query_vec.unsqueeze(0), dim=1)

        # Recency bias: later entries (more recent) get a slight boost
        recency_weights = torch.linspace(0.2, 1.0, steps=len(self.entries), device=sims.device)
        weighted = sims * 0.8 + recency_weights * 0.2
        top_indices = torch.argsort(weighted, descending=True)[:top_k]
        return [f"[{self.entries[i].timestamp}] {self.entries[i].text}" for i in top_indices.tolist()]

    def commonsense_snapshot(self, topics: Optional[Sequence[str]] = None) -> str:
        return self.commonsense.retrieve(topics)


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
    currency: str
    tx_type: str
    is_money_laundering: bool
    ml_typology: str | None


class EventRecorder:
    def __init__(self) -> None:
        self.logs: List[EventLog] = []
        self.transactions: List[Transaction] = []

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

    def record_transaction(
        self,
        sender_account: str,
        receiver_account: str,
        amount: float,
        currency: str,
        tx_type: str,
        is_money_laundering: bool,
        ml_typology: str | None,
    ) -> Transaction:
        tx = Transaction(
            tx_id=len(self.transactions) + 1,
            timestamp=now_timestamp(),
            sender_account=sender_account,
            receiver_account=receiver_account,
            amount=float(amount),
            currency=currency,
            tx_type=tx_type,
            is_money_laundering=is_money_laundering,
            ml_typology=ml_typology,
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
                    "currency",
                    "tx_type",
                    "is_money_laundering",
                    "ml_typology",
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
                        tx.currency,
                        tx.tx_type,
                        int(tx.is_money_laundering),
                        tx.ml_typology or "",
                    ]
                )