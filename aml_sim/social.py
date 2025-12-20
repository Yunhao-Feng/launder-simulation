"""Social interaction utilities for generative agents.

This module enables simple language-based interactions between agents and
maintains a light-weight relationship graph that can influence behavior.
"""

from __future__ import annotations

import random
from typing import Dict

from .agents import Agent


def generate_social_event(agent_a: Agent, agent_b: Agent, context_str: str, rumor: str | None = None) -> str:
    """Generate a brief dialogue between two agents using their reasoning engines."""

    rumor_line = f"Fresh rumor: {rumor}. " if rumor else ""
    query = (
        "You are roleplaying a short conversation with another character. "
        f"Your partner is {agent_b.role} ({agent_b.id}). Context: {context_str}. "
        f"{rumor_line}"
        "Share actionable gossip about banking scrutiny or laundering tactics."
    )
    return agent_a.retrieve_context(query)


def update_relationship(
    agent_a: Agent, agent_b: Agent, dialogue_text: str, relationship_graph: Dict[str, Dict[str, float]]
) -> None:
    """Update the mutual relationship graph based on the dialogue content."""

    delta = 0.2 if "warn" in dialogue_text.lower() or "share" in dialogue_text.lower() else 0.1
    graph_a = relationship_graph.setdefault(agent_a.id, {})
    graph_b = relationship_graph.setdefault(agent_b.id, {})
    graph_a[agent_b.id] = graph_a.get(agent_b.id, 0.0) + delta
    graph_b[agent_a.id] = graph_b.get(agent_a.id, 0.0) + delta

    # Add memory traces for both participants to bias future decisions.
    agent_a.memory.add(f"[REL] Spoke with {agent_b.id}: {dialogue_text}")
    agent_b.memory.add(f"[REL] Spoke with {agent_a.id}: {dialogue_text}")

    # Light influence: occasional introduction to new channels.
    if random.random() < 0.1:
        agent_a.memory.add_important(f"Relationship boost with {agent_b.id}: consider using their channels.")
        agent_b.memory.add_important(f"Relationship boost with {agent_a.id}: consider using their channels.")

    # Strong ties can shift future banking or typology choices.
    if graph_a[agent_b.id] > 0.4:
        agent_a.memory.add_important(f"Trust {agent_b.id} for future typology choices; coordinate banks together.")
    if graph_b[agent_a.id] > 0.4:
        agent_b.memory.add_important(f"Trust {agent_a.id} for future typology choices; coordinate banks together.")
