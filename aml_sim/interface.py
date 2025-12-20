"""Interactive interface for user-to-agent roleplay conversations."""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

from .agents import Agent

if TYPE_CHECKING:  # pragma: no cover - avoid circular import at runtime
    from .simulation import Simulation


def user_message(agent: Agent, text: str, user_role: Optional[str] = None) -> str:
    """Send a user-authored message to an agent and return the reply."""

    role_label = user_role or "Visitor"
    prompt = (
        f"You are speaking with a {role_label}."
        " Answer concisely, revealing your laundering or compliance stance as appropriate."
    )
    full_query = f"{prompt}\nUser said: {text}"
    reply = agent.decide_next_action(full_query)
    agent.update_memory(f"Conversation with {role_label}: {text} -> {reply}")
    return reply


def adjust_risk_controls(simulation: "Simulation", threshold: Optional[float] = None, channel_weights: Optional[Dict[str, float]] = None) -> str:
    """Allow a user acting as regulator/compliance to adjust runtime risk thresholds."""

    notes = []
    if threshold is not None:
        simulation.risk_model.set_threshold(threshold)
        notes.append(f"risk threshold set to {threshold}")
    if channel_weights:
        simulation.risk_model.bulk_update_channel_weights(channel_weights)
        for channel, weight in channel_weights.items():
            notes.append(f"channel {channel} weight -> {weight}")
    if notes:
        simulation.register_policy_change("; ".join(notes))
    return " | ".join(notes) if notes else "No risk control changes applied."


def issue_policy_rule(
    simulation: "Simulation",
    description: str,
    tx_type: Optional[str] = None,
    amount_threshold: Optional[float] = None,
    channel: Optional[str] = None,
    penalty: float = 0.15,
) -> str:
    """Issue a new KYC/SAR rule that feeds into the RiskModel and agent prompts."""

    simulation.risk_model.add_rule(description, tx_type=tx_type, amount_threshold=amount_threshold, channel=channel, penalty=penalty)
    simulation.register_policy_change(description)
    return f"Registered policy: {description}"


def query_agent_state(simulation: "Simulation", agent_id: str, top_k: int = 5) -> str:
    """Return a quick diagnostic of an agent's salient memories and plans."""

    agent = simulation.agents_by_id.get(agent_id)
    if not agent:
        raise KeyError(f"Agent {agent_id} not found")
    salient = agent.memory.salient_summary(top_k=top_k)
    plan = agent.current_plan or "(no plan)"
    return f"Agent {agent.id} ({agent.role})\nMood: {agent.mood} | Risk tolerance: {agent.risk_tolerance:.2f}\nPlan: {plan}\nSalient:\n{salient}"
