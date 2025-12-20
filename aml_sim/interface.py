"""Interactive interface for user-to-agent roleplay conversations."""

from __future__ import annotations

from typing import Optional

from .agents import Agent


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
