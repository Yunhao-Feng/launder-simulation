from __future__ import annotations

import dataclasses
import importlib.util
import os
from typing import List, Optional

try:  # OpenAI is optional for deterministic offline runs.
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback when SDK is unavailable.
    OpenAI = None

from .knowledge import KnowledgeBase
from .memory import AgentMemory


@dataclasses.dataclass
class AccountProfile:
    account_id: str
    bank_id: str
    currency: str
    clean_balance: float = 0.0
    illicit_balance: float = 0.0


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

    def _llm_reason(self, prompt: str) -> Optional[str]:
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

    def run(
        self,
        role: str,
        query: str,
        memory_block: str,
        knowledge_block: str,
        commonsense_block: str,
        plan_block: str | None = None,
        relationship_block: str | None = None,
        affect_block: str | None = None,
        policy_block: str | None = None,
    ) -> str:
        plan_section = f"Long-term plan:\n{plan_block}\n\n" if plan_block else ""
        relationship_section = f"Relationship context:\n{relationship_block}\n\n" if relationship_block else ""
        affect_section = f"Affect and risk posture:\n{affect_block}\n\n" if affect_block else ""
        policy_section = f"Active policies and rules:\n{policy_block}\n\n" if policy_block else ""
        prompt = (
            f"Role: {role}\n"
            "Context from personal memory:\n"
            f"{memory_block}\n\n"
            "Retrieved regulations and cases:\n"
            f"{knowledge_block}\n\n"
            "Commonsense AML heuristics:\n"
            f"{commonsense_block}\n\n"
            f"{plan_section}"
            f"{relationship_section}"
            f"{affect_section}"
            f"{policy_section}"
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
    accounts: List[AccountProfile]
    memory: AgentMemory
    knowledge_base: KnowledgeBase
    reasoner: ReasoningEngine = dataclasses.field(default_factory=ReasoningEngine)
    current_plan: str | None = None
    mood: str = "calm"
    risk_tolerance: float = 0.5
    policy_context: str | None = None

    def update_memory(self, entry: str, event_type: str | None = None, importance: float | None = None) -> None:
        self.memory.add(entry, event_type=event_type, importance=importance)
        self.update_affect_from_event(event_type, entry)

    def update_affect_from_event(self, event_type: str | None, description: str) -> None:
        """Nudge mood and risk tolerance based on notable events."""

        if not event_type:
            return
        lowered = f"{event_type} {description}".lower()
        if any(keyword in lowered for keyword in ["sar", "investigation", "flag", "regulator"]):
            self.mood = "nervous"
            self.risk_tolerance = max(0.15, self.risk_tolerance - 0.1)
        if any(keyword in lowered for keyword in ["profit", "windfall", "successful", "high margin"]):
            self.mood = "aggressive"
            self.risk_tolerance = min(1.0, self.risk_tolerance + 0.1)
        if any(keyword in lowered for keyword in ["gossip", "rumor", "warning", "monitored"]):
            self.mood = "alert"
            self.risk_tolerance = max(0.2, self.risk_tolerance - 0.05)

    def affect_block(self) -> str:
        return f"Mood: {self.mood}; Risk tolerance (0=avoidant,1=reckless): {self.risk_tolerance:.2f}"

    def retrieve_context(self, query: str, plan_context: str | None = None, relationship_context: str | None = None) -> str:
        memory_hits = self.memory.retrieve(query)
        commonsense = self.memory.commonsense_snapshot()
        kb_docs = self.knowledge_base.query(query)
        kb_texts = [f"{doc.title}: {doc.text}" for doc in kb_docs]
        memory_block = "\n".join(memory_hits) or "(no direct matches, showing recent history)"
        knowledge_block = "\n".join(kb_texts) or "(no retrieved cases)"
        plan_block = plan_context or self.current_plan or "\n".join(self.memory.important_snapshot())
        policy_block = self.policy_context or ""
        salient_block = self.memory.salient_summary(top_k=4)
        return self.reasoner.run(
            self.role,
            query,
            memory_block,
            knowledge_block,
            commonsense,
            plan_block=plan_block,
            relationship_block=relationship_context,
            affect_block=f"{self.affect_block()}\nSalient cues:\n{salient_block}",
            policy_block=policy_block,
        )

    @property
    def primary_account(self) -> str:
        return self.accounts[0].account_id if self.accounts else ""
    
    def decide_next_action(self, query: str, observation: str | None = None, relationship_context: str | None = None) -> str:
        plan_context = self.current_plan or "\n".join(self.memory.important_snapshot())
        relationship_context = relationship_context or getattr(self, "relationship_context", None)
        decision = self.retrieve_context(query, plan_context=plan_context, relationship_context=relationship_context)
        memo_line = observation or query
        self.update_memory(f"{memo_line} | decision: {decision}")
        return decision

    def reflect(self, day_summary: str) -> str:
        """Produce an end-of-day reflection summarizing notable events."""

        reflection_prompt = (
            "Review today's events and extract key lessons for laundering risk and operational adjustments.\n"
            f"Events: {day_summary}"
        )
        reflection = self.retrieve_context(reflection_prompt)
        self.memory.add_important(f"Reflection: {reflection}")
        return reflection

    def update_long_term_plan(self, reflection: str) -> str:
        """Generate or adjust medium-horizon goals based on the latest reflection."""

        plan_prompt = (
            "Using the reflection, propose strategic goals for the coming days. Focus on AML evasion or compliance duties. "
            f"Reflection: {reflection}"
        )
        plan = self.retrieve_context(plan_prompt)
        self.memory.add_important(f"Plan: {plan}")
        self.current_plan = plan
        return plan


class BossAgent(Agent):
    def plan_strategy(self) -> str:
        return self.decide_next_action("Coordinate laundering parcels and minimize visibility")


class CourierAgent(Agent):
    def move_parcels(self, parcel_count: int, total_value: float) -> str:
        return self.decide_next_action(
            "Deliver parcels from mastermind to businesses and mules",
            observation=f"Moved {parcel_count} parcels totaling {total_value:.2f}",
        )


class BusinessOwnerAgent(Agent):
    business_type: str

    def receive_parcel(self, amount: float) -> str:
        return self.decide_next_action(
            "Handle illicit inflow while masking as revenue",
            observation=f"Received parcel of {amount:.2f} for business inflow",
        )

    def integrate_funds(self, amount: float) -> str:
        return self.decide_next_action(
            "Blend funds through payroll and supplier payments",
            observation=f"Integrating {amount:.2f} via payroll and suppliers",
        )


class AccountantAgent(Agent):
    def design_typology(self, typology: str) -> str:
        return self.decide_next_action(
            "Construct layering typology",
            observation=f"Constructed {typology} layering path",
        )


class EmployeeAgent(Agent):
    def perform_shift(self, business: str) -> str:
        return self.decide_next_action(
            "Serve customers and record legitimate revenue",
            observation=f"Completed shift at {business}",
        )


class MuleAgent(Agent):
    def redistribute(self, amounts: List[float]) -> str:
        return self.decide_next_action(
            "Redistribute parcels without detection",
            observation=f"Redistributing {len(amounts)} parcels totaling {sum(amounts):.2f}",
        )


class MastermindAgent(Agent):
    def parcel_funds(self, parcels: List[float]) -> str:
        return self.decide_next_action(
            "Split bulk funds into parcels for placement",
            observation=f"Split incoming funds into parcels: {', '.join([str(int(p)) for p in parcels])}",
        )


class RegulatorAgent(Agent):
    def open_investigation(self, tx_id: int, score: float) -> str:
        return self.decide_next_action(
            "Investigate high-risk transaction",
            observation=f"Opened SAR for transaction {tx_id} with risk {score:.2f}",
        )


class BankTellerAgent(Agent):
    def flag_deposit(self, amount: float) -> str:
        return self.decide_next_action(
            "Review cash deposit for SAR threshold",
            observation=f"Filed SAR for large deposit {amount:.2f}",
        )


class BackOfficeAgent(Agent):
    def approve_transfer(self, amount: float) -> str:
        return self.decide_next_action(
            "Approve large outbound transfer",
            observation=f"Approved large transfer of {amount:.2f}",
        )


class ResidentAgent(Agent):
    def normal_activity(self, amount: float) -> str:
        return self.decide_next_action(
            "Perform normal spending or savings",
            observation=f"Completed normal spend of {amount:.2f}",
        )
