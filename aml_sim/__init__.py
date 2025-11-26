from __future__ import annotations

import dataclasses
import importlib.util
import os
from typing import List

from openai import OpenAI

from .knowledge import KnowledgeBase
from .memory import AgentMemory


class ReasoningEngine:
    """LLM-backed reasoning with a deterministic fallback.

    The engine first tries to call GPT-4o when an API key is present; otherwise
    it composes a deterministic response grounded in retrieved memory and
    knowledge base context.
    """

    def __init__(self, model_name: str = "gpt-4o") -> None:
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")

    def _llm_reason(self, prompt: str):
        if not self.api_key or OpenAI is None:
            return None
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an AML reasoning assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=240,
            )
        except Exception:
            return None
        return response.choices[0].message.content if response.choices else None

    def _fallback_reason(self, role: str, query: str, context: str) -> str:
        summary_lines = [
            f"Role: {role}",
            "Context cues:",
        ]
        for line in context.splitlines():
            if line.strip():
                summary_lines.append(f" - {line.strip()}")
        summary_lines.append(f"Proposed action for '{query}':")
        if "investigation" in query.lower() or "suspicious" in query.lower():
            summary_lines.append(" - Escalate with SAR, attach recent memory evidence, and monitor related accounts.")
        elif "transfer" in query.lower() or "parcel" in query.lower():
            summary_lines.append(" - Execute structured transfer while keeping amounts within expected typology bands.")
        else:
            summary_lines.append(" - Continue normal operations while logging observations for compliance review.")
        return "\n".join(summary_lines)

    def run(self, role: str, query: str, memory_block: str, knowledge_block: str, commonsense_block: str) -> str:
        prompt = (
            f"Role: {role}\n"
            "Context from personal memory:\n"
            f"{memory_block}\n\n"
            "Retrieved regulations and cases:\n"
            f"{knowledge_block}\n\n"
            "Commonsense AML heuristics:\n"
            f"{commonsense_block}\n\n"
            f"Query: {query}\n"
            "Respond with a concise decision and short justification."
        )
        llm_answer = self._llm_reason(prompt)
        if llm_answer:
            return llm_answer.strip()
        return self._fallback_reason(role, query, "\n".join([memory_block, knowledge_block, commonsense_block]))


@dataclasses.dataclass
class Agent:
    id: str
    role: str
    accounts: List[str]
    memory: AgentMemory
    knowledge_base: KnowledgeBase
    reasoner: ReasoningEngine = dataclasses.field(default_factory=ReasoningEngine)

    def update_memory(self, entry: str) -> None:
        self.memory.add(entry)

    def retrieve_context(self, query: str) -> str:
        memory_hits = self.memory.retrieve(query)
        commonsense = self.memory.commonsense_snapshot()
        kb_docs = self.knowledge_base.query(query)
        kb_texts = [f"{doc.title}: {doc.text}" for doc in kb_docs]
        memory_block = "\n".join(memory_hits) or "(no direct matches, showing recent history)"
        knowledge_block = "\n".join(kb_texts) or "(no retrieved cases)"
        return self.reasoner.run(self.role, query, memory_block, knowledge_block, commonsense)

    def decide_next_action(self, query: str) -> str:
        return self.retrieve_context(query)


class BusinessOwnerAgent(Agent):
    business_type: str

    def receive_parcel(self, amount: float) -> str:
        note = f"Received parcel of {amount:.2f} for business inflow."
        self.update_memory(note)
        return note

    def integrate_funds(self, amount: float) -> str:
        note = f"Integrating {amount:.2f} via payroll and suppliers."
        self.update_memory(note)
        return note


class AccountantAgent(Agent):
    def design_typology(self, typology: str) -> str:
        note = f"Constructed {typology} layering path."
        self.update_memory(note)
        return note


class MuleAgent(Agent):
    def redistribute(self, amounts: List[float]) -> str:
        note = f"Redistributing {len(amounts)} incoming parcels totaling {sum(amounts):.2f}."
        self.update_memory(note)
        return note


class MastermindAgent(Agent):
    def parcel_funds(self, parcels: List[float]) -> str:
        note = f"Split incoming funds into parcels: {', '.join([str(int(p)) for p in parcels])}."
        self.update_memory(note)
        return note


class RegulatorAgent(Agent):
    def open_investigation(self, tx_id: int, score: float) -> str:
        note = f"Opened SAR for transaction {tx_id} with risk {score:.2f}."
        self.update_memory(note)
        return note


class BankTellerAgent(Agent):
    def flag_deposit(self, amount: float) -> str:
        note = f"Filed SAR for large deposit {amount:.2f}."
        self.update_memory(note)
        return note


class BackOfficeAgent(Agent):
    def approve_transfer(self, amount: float) -> str:
        note = f"Approved large transfer of {amount:.2f}."
        self.update_memory(note)
        return note


class ResidentAgent(Agent):
    def normal_activity(self, amount: float) -> str:
        note = f"Completed normal spend of {amount:.2f}."
        self.update_memory(note)
        return note