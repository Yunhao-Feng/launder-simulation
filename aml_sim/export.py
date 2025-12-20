"""Benchmark export utilities for AML graph datasets."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict


def export_accounts(sim, path) -> None:
    """Export account-level balances and metadata to CSV."""

    filepath = Path(path)
    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "account_id",
                "owner_entity",
                "owner_type",
                "bank_id",
                "currency",
                "clean_balance",
                "illicit_balance",
                "illicit_fraction",
                "account_type",
            ]
        )
        for account_id, state in sim.event_recorder.accounts.items():
            owner_entity = ""
            owner_type = ""
            account_type = ""
            for agent in sim.all_agents:
                if any(acct.account_id == account_id for acct in agent.accounts):
                    owner_entity = agent.id
                    owner_type = "agent"
                    account_type = agent.role
                    break
            if not owner_entity:
                for company in sim.companies:
                    if account_id in company.accounts:
                        owner_entity = company.id
                        owner_type = "company_shell" if company.is_shell else "company"
                        account_type = "business"
                        break
            writer.writerow(
                [
                    account_id,
                    owner_entity,
                    owner_type,
                    state.bank_id,
                    state.currency,
                    state.clean_balance,
                    state.illicit_balance,
                    state.illicit_balance / max(1.0, state.clean_balance + state.illicit_balance),
                    account_type,
                ]
            )


def export_entities(sim, path) -> None:
    """Export entities and their relationships to CSV."""

    filepath = Path(path)
    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "entity_id",
                "type",
                "ownership_edges",
                "supplier_edges",
                "customer_edges",
                "accounts",
            ]
        )
        for person in sim.person_entities:
            writer.writerow([person.id, "person", "", "", "", person.accounts])
        for company in sim.companies:
            owner_ids = [owner.id for owner in company.owners]
            supplier_ids = [sup.id for sup in company.suppliers]
            customer_ids = [cust.id for cust in company.customers]
            writer.writerow(
                [
                    company.id,
                    "shell_company" if company.is_shell else "company",
                    owner_ids,
                    supplier_ids,
                    customer_ids,
                    company.accounts,
                ]
            )


def export_edges(sim, path) -> None:
    """Export transactions as edges suitable for graph processing."""

    filepath = Path(path)
    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tx_id",
                "timestamp",
                "sender_account",
                "receiver_account",
                "amount",
                "clean_amount",
                "illicit_amount",
                "illicit_fraction",
                "currency",
                "channel",
                "fee_amount",
                "tx_type",
                "cross_bank",
                "cross_currency",
                "is_money_laundering",
                "ml_pattern",
                "ml_typology",
                "pattern_scheme_id",
                "timestamp_day",
            ]
        )
        for tx in sim.event_recorder.transactions:
            writer.writerow(
                [
                    tx.tx_id,
                    tx.timestamp,
                    tx.sender_account,
                    tx.receiver_account,
                    tx.amount,
                    tx.clean_amount,
                    tx.illicit_amount,
                    tx.illicit_fraction,
                    tx.currency,
                    tx.channel,
                    tx.fee_amount,
                    tx.tx_type,
                    int(tx.cross_bank),
                    int(tx.cross_currency),
                    int(tx.is_money_laundering),
                    tx.ml_pattern or (tx.ml_typology or ""),
                    tx.ml_typology or "",
                    tx.pattern_scheme_id or "",
                    tx.settlement_day if tx.settlement_day is not None else "",
                ]
            )


def export_graph_features(sim, path) -> None:
    """Compute and export lightweight graph features per account node."""

    filepath = Path(path)
    adjacency: Dict[str, set] = defaultdict(set)
    indegree: Dict[str, int] = defaultdict(int)
    outdegree: Dict[str, int] = defaultdict(int)
    amount_in: Dict[str, float] = defaultdict(float)
    amount_out: Dict[str, float] = defaultdict(float)
    illicit_in: Dict[str, float] = defaultdict(float)
    fee_out: Dict[str, float] = defaultdict(float)

    for tx in sim.event_recorder.transactions:
        adjacency[tx.sender_account].add(tx.receiver_account)
        outdegree[tx.sender_account] += 1
        indegree[tx.receiver_account] += 1
        amount_out[tx.sender_account] += tx.amount
        amount_in[tx.receiver_account] += tx.amount
        illicit_in[tx.receiver_account] += tx.illicit_amount
        fee_out[tx.sender_account] += tx.fee_amount

    def count_cycles(node: str, depth: int = 3) -> int:
        visited = set()

        def dfs(current: str, target: str, remaining: int) -> int:
            if remaining == 0:
                return 0
            total = 0
            for nxt in adjacency.get(current, []):
                if nxt == target:
                    total += 1
                if nxt not in visited:
                    visited.add(nxt)
                    total += dfs(nxt, target, remaining - 1)
                    visited.discard(nxt)
            return total

        visited.add(node)
        return dfs(node, node, depth)

    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "account_id",
                "in_degree",
                "out_degree",
                "cycles",
                "fan_in",
                "fan_out",
                "amount_in",
                "amount_out",
                "illicit_ratio_in",
                "avg_fee_paid",
            ]
        )
        for account_id in sim.event_recorder.accounts.keys():
            inflow = amount_in.get(account_id, 0.0)
            illicit_ratio = illicit_in.get(account_id, 0.0) / inflow if inflow else 0.0
            tx_out_count = outdegree.get(account_id, 0)
            writer.writerow(
                [
                    account_id,
                    indegree.get(account_id, 0),
                    outdegree.get(account_id, 0),
                    count_cycles(account_id),
                    len([src for src, tgts in adjacency.items() if account_id in tgts]),
                    len(adjacency.get(account_id, set())),
                    inflow,
                    amount_out.get(account_id, 0.0),
                    illicit_ratio,
                    fee_out.get(account_id, 0.0) / tx_out_count if tx_out_count else 0.0,
                ]
            )
