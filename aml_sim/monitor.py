"""Lightweight monitoring utilities for day-over-day simulation tracking."""

from __future__ import annotations

from typing import Any, Dict, List


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _transactions_for_day(sim, day_index: int):
    recorder = getattr(sim, "event_recorder", None)
    if not recorder:
        return []
    return [tx for tx in recorder.transactions if getattr(tx, "event_day", tx.settlement_day) == day_index]


def _summarize_affect_changes(sim, day_index: int) -> List[Dict[str, Any]]:
    snapshots = sim.daily_affect_snapshots.get(day_index, {})
    start = snapshots.get("start", {})
    end = snapshots.get("end", {})
    changes: List[Dict[str, Any]] = []
    for agent_id, end_state in end.items():
        start_state = start.get(agent_id, {})
        mood_changed = start_state.get("mood") != end_state.get("mood")
        risk_before = start_state.get("risk_tolerance")
        if risk_before is None:
            risk_before = end_state.get("risk_tolerance", 0.0)
        risk_after = end_state.get("risk_tolerance", risk_before or 0.0)
        risk_shift = abs((risk_after or 0.0) - (risk_before or 0.0))
        fatigue_before = start_state.get("fatigue", end_state.get("fatigue", 0.0))
        fatigue_change = abs(end_state.get("fatigue", 0.0) - (fatigue_before or 0.0))
        if mood_changed or risk_shift > 0.05 or fatigue_change > 0.05 or not start_state:
            changes.append(
                {
                    "agent_id": agent_id,
                    "mood_before": start_state.get("mood"),
                    "mood_after": end_state.get("mood"),
                    "risk_tolerance_before": start_state.get("risk_tolerance"),
                    "risk_tolerance_after": risk_after,
                    "risk_tolerance_delta": risk_shift if start_state else 0.0,
                    "fatigue_after": end_state.get("fatigue"),
                    "stress_after": end_state.get("stress"),
                    "confidence_after": end_state.get("confidence"),
                }
            )
    return changes


def generate_daily_summary(sim, day_index: int) -> Dict[str, Any]:
    """Aggregate monitoring metrics for a specific simulation day."""

    tx_for_day = _transactions_for_day(sim, day_index)
    tx_count = len(tx_for_day)
    risk_scores = sim.daily_risk_scores.get(day_index, [])
    avg_risk = _safe_mean(risk_scores)
    max_risk = max(risk_scores) if risk_scores else 0.0
    affect_changes = _summarize_affect_changes(sim, day_index)
    policies = sim.risk_model.active_policy_summary() if getattr(sim, "risk_model", None) else ""
    high_risk_alerts = sim.daily_alerts.get(day_index, 0)

    change_strings = []
    for change in affect_changes[:5]:
        mood_before = change.get("mood_before") or "unknown"
        mood_after = change.get("mood_after") or "unknown"
        risk_before = change.get("risk_tolerance_before")
        risk_after = change.get("risk_tolerance_after")
        risk_before = 0.0 if risk_before is None else risk_before
        risk_after = 0.0 if risk_after is None else risk_after
        change_strings.append(
            f"{change['agent_id']}: mood {mood_before}->{mood_after}, "
            f"risk {risk_before:.2f}->{risk_after:.2f}"
        )
    change_summary = "; ".join(change_strings) if change_strings else "No notable affect changes."

    summary_text = (
        f"Day {day_index}: {tx_count} transactions, {high_risk_alerts} high-risk alerts. "
        f"Avg risk {avg_risk:.3f}, max {max_risk:.3f}. "
        f"Affect shifts: {change_summary}. Active policies: {policies}"
    )

    return {
        "day_index": day_index,
        "transactions": tx_count,
        "high_risk_alerts": high_risk_alerts,
        "avg_risk_score": avg_risk,
        "max_risk_score": max_risk,
        "agent_affect_changes": affect_changes,
        "active_policies": policies,
        "summary_text": summary_text,
    }
