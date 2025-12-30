from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .agents import (
    Agent,
    AccountProfile,
    AccountantAgent,
    BackOfficeAgent,
    BankAgent,
    BankTellerAgent,
    BossAgent,
    BusinessOwnerAgent,
    CourierAgent,
    EmployeeAgent,
    LLMConfig,
    MastermindAgent,
    MuleAgent,
    ReasoningEngine,
    RegulatorAgent,
    ResidentAgent,
)
from .banking import Bank, load_banks
from .config import SimulationConfig
from .detection import RiskModel
from .entities import Company, Person
from .knowledge import DPRTextEmbedder, KnowledgeBase
from .memory import AgentMemory, EventRecorder
from .social import diffuse_information, generate_recruitment_event, generate_social_event, update_relationship
from . import interface, monitor
from .patterns import (
    ALL_LAUNDERING_PATTERNS,
    BIPARTITE,
    FAN_IN,
    FAN_OUT,
    GATHER_SCATTER,
    RANDOM,
    SCATTER_GATHER,
    SIMPLE_CYCLE,
    STACK,
)


ACCOUNTING_TYPOLOGIES = ALL_LAUNDERING_PATTERNS + ["peel chain"]


class Simulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        random.seed(config.seed)
        self.population_scale = config.population_scale or 1.0
        self.transaction_scale = config.transaction_scale or 1.0
        self.laundering_intensity = config.laundering_intensity
        self.banks: Dict[str, Bank] = load_banks(config.banks)
        self.fx_rates: Dict[Tuple[str, str], float] = {
            (k[0], k[1]): float(v) if isinstance(k, tuple) else float(v)
            for k, v in ((key if isinstance(key, tuple) else key.split("-"), val) for key, val in config.fx_rates.items())
            if len(k) == 2
        }
        self.default_currency = config.default_currency
        self.available_currencies: List[str] = sorted(
            {self.default_currency, *(c for bank in self.banks.values() for c in bank.supported_currencies)}
        )
        self.pattern_generation = config.pattern_generation or {"enabled": False}
        self.embedder = DPRTextEmbedder(model_path=config.knowledge_model_path, device=config.knowledge_device)
        self.knowledge_base = KnowledgeBase.from_config(
            config.knowledge_documents,
            model_path=config.knowledge_model_path,
            device=config.knowledge_device,
            embedder=self.embedder,
        )
        self.event_recorder = EventRecorder(
            banks=self.banks,
            fx_rates=self.fx_rates,
            default_currency=self.default_currency,
        )
        self.risk_model = RiskModel(config.risk_threshold, config.risk_model)
        self.policy_directives: List[str] = []
        self.llm_config = LLMConfig(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            model=config.llm_model,
        )
        self.affect_config = config.affect_config or {}
        self.calibration = config.calibration or {}
        self.evaluation = config.evaluation or {}
        raw_amount_cfg = self.calibration.get("amounts") or self.calibration.get("amount_distributions")
        if not raw_amount_cfg:
            raw_amount_cfg = {k: v for k, v in self.calibration.items() if isinstance(v, dict) and v.get("type")}
        self.amount_calibrations = raw_amount_cfg or {}
        self.channel_frequency = self.calibration.get("channel_frequencies", {})
        self.relationship_graph: Dict[str, Dict[str, float]] = {}
        self.daily_risk_scores: Dict[int, List[float]] = {}
        self.daily_alerts: Dict[int, int] = {}
        self.daily_transaction_ids: Dict[int, List[int]] = {}
        self.daily_affect_snapshots: Dict[int, Dict[str, Dict[str, Dict[str, float] | str]]] = {}
        self.daily_summaries: Dict[int, Dict[str, object]] = {}
        self.current_day_index: int = 0

        self.bank_agents: List[BankAgent] = self._create_bank_agents()
        self.boss: BossAgent = self._create_boss()
        self.mastermind: MastermindAgent = self._create_mastermind()
        self.courier: CourierAgent = self._create_courier()
        self.business_owners: List[BusinessOwnerAgent] = self._create_business_owners()
        self.accountants: List[AccountantAgent] = self._create_accountants()
        self.employees: List[EmployeeAgent] = self._create_employees()
        self.mules: List[MuleAgent] = self._create_mules()
        self.residents: List[ResidentAgent] = self._create_residents()
        self.bank_tellers: List[BankTellerAgent] = self._create_tellers()
        self.back_office: BackOfficeAgent = self._create_back_office()
        self.regulators: List[RegulatorAgent] = self._create_regulators()

        self.person_entities: List[Person] = [
            Person(resident.id, resident.role, [acct.account_id for acct in resident.accounts])
            for resident in self.residents
        ]
        self.companies: List[Company] = self._create_companies()
        self.company_staff = self._assign_staff_to_companies()

        self._index_agents()
        self._update_policy_contexts()

    def _memory(self, max_items: int = 500) -> AgentMemory:
        return AgentMemory(max_items=max_items, embedder=self.embedder)

    def _make_reasoner(self) -> ReasoningEngine:
        return ReasoningEngine.from_config(self.llm_config)

    def _affect_for_role(self, role_key: str) -> Dict[str, Any]:
        """Return initial affective state for a role with safe defaults."""

        default_mood = self.affect_config.get("default_mood", "calm")
        default_risk = float(self.affect_config.get("default_risk_tolerance", 0.5))
        default_fatigue = float(self.affect_config.get("default_fatigue", 0.1))
        default_stress = float(self.affect_config.get("default_stress", 0.2))
        default_confidence = float(self.affect_config.get("default_confidence", 0.5))
        overrides = self.affect_config.get("role_overrides", {}) or {}
        role_cfg = overrides.get(role_key) or overrides.get(role_key.replace(" ", "_"), {})
        return {
            "mood": role_cfg.get("mood", default_mood),
            "risk_tolerance": float(role_cfg.get("risk_tolerance", default_risk)),
            "fatigue": float(role_cfg.get("fatigue", default_fatigue)),
            "stress": float(role_cfg.get("stress", default_stress)),
            "confidence": float(role_cfg.get("confidence", default_confidence)),
        }

    def _account_state(self, account_id: str):
        return self.event_recorder.accounts.get(account_id) or self.event_recorder.register_account(account_id)

    def _account_currency(self, account_id: str) -> str:
        return self._account_state(account_id).currency or self.default_currency

    def _sample_amount(self, key: str, fallback_range: Tuple[int, int]) -> float:
        """Sample an amount using calibrated distributions when available.

        Calibrations can be provided in ``config.calibration.amounts`` (or
        ``amount_distributions``) with entries such as:

        {"payroll": {"type": "histogram", "bins": [0, 5_000, 10_000], "weights": [0.7, 0.3]}}
        {"p2p": {"type": "lognormal", "mean": 7.2, "sigma": 0.35, "scale": 1.0}}

        If no calibrated distribution is present, the simulator falls back to
        uniform sampling over ``fallback_range``.
        """

        cfg = self.amount_calibrations.get(key) or self.calibration.get(key, {})
        lower, upper = fallback_range
        if cfg:
            dist_type = cfg.get("type", "").lower()
            if dist_type == "lognormal":
                mean = float(cfg.get("mean", 1.0))
                sigma = float(cfg.get("sigma", 0.5))
                scale = float(cfg.get("scale", 1.0))
                sampled = random.lognormvariate(mean, sigma) * scale
                return max(lower, min(sampled, upper * 2))
            if dist_type == "histogram":
                bins = cfg.get("bins", [])
                weights = cfg.get("weights") or cfg.get("frequency") or []
                if bins and weights and len(bins) in {len(weights), len(weights) + 1}:
                    # Ensure bin edges length = weights + 1
                    if len(bins) == len(weights):
                        bins = list(bins) + [bins[-1]]
                    total = sum(weights) or 1.0
                    probs = [w / total for w in weights]
                    idx = random.choices(range(len(probs)), weights=probs, k=1)[0]
                    left, right = bins[idx], bins[idx + 1]
                    return random.uniform(left, right)
        return random.uniform(lower, upper)

    def _risk_scaled_amount(self, base_amount: float, agent: Agent | None) -> float:
        if agent is None:
            return base_amount
        modifier = 0.8 + agent.risk_tolerance * 0.5
        return base_amount * modifier

    def _choose_channel(self, tx_type: str, is_money_laundering: bool = False) -> str:
        """Pick a realistic payment rail for the given transaction context."""

        laundering_channels = ["wire", "crypto", "cash", "ach"]
        normal_channels = ["ach", "credit_card", "cheque", "cash"]
        if tx_type in {"deposit", "placement"}:
            return "cash"
        if tx_type in {"salary", "payroll", "revenue", "supplier_payment"}:
            return "ach"
        if tx_type in {"bill", "p2p", "spend"}:
            return random.choice(["credit_card", "ach", "crypto"])
        if self.channel_frequency:
            weights: Dict[str, float] = {}
            if all(isinstance(v, (int, float)) for v in self.channel_frequency.values()):
                weights = dict(self.channel_frequency)
            else:
                tx_specific = self.channel_frequency.get(tx_type)
                default_block = self.channel_frequency.get("default")
                if isinstance(tx_specific, dict):
                    weights = tx_specific
                elif isinstance(default_block, dict):
                    weights = default_block
                else:
                    weights = {k: v for k, v in self.channel_frequency.items() if isinstance(v, (int, float))}
            if is_money_laundering:
                weights["crypto"] = weights.get("crypto", 0.1) * 1.6
                weights["cash"] = weights.get("cash", 0.1) * 1.4
            population = list(weights.keys())
            probs = [weights[k] for k in population]
            total = sum(probs) or 1.0
            probs = [p / total for p in probs]
            return random.choices(population, weights=probs, k=1)[0]
        if is_money_laundering:
            return random.choice(laundering_channels)
        return random.choice(normal_channels)

    def _seed_illicit_balance(self, account_id: str, amount: float) -> None:
        acct = self._account_state(account_id)
        acct.illicit_balance += float(amount)
    
    def _scale_count(self, value: int) -> int:
        scaled = int(value * self.population_scale)
        return scaled if scaled > 0 else value

    def _choose_bank_and_currency(self) -> Tuple[str, str]:
        bank = random.choice(list(self.banks.values()))
        currency = self.default_currency if self.default_currency in bank.supported_currencies else bank.supported_currencies[0]
        return bank.bank_id, currency

    def _register_account(
        self,
        account_id: str,
        clean_balance: float = 0.0,
        illicit_balance: float = 0.0,
        bank_id: str | None = None,
        currency: str | None = None,
    ) -> AccountProfile:
        chosen_bank_id, chosen_currency = self._choose_bank_and_currency()
        state = self.event_recorder.register_account(
            account_id=account_id,
            bank_id=bank_id or chosen_bank_id,
            currency=currency or chosen_currency,
            clean_balance=clean_balance,
            illicit_balance=illicit_balance,
        )
        return AccountProfile(
            account_id=state.account_id,
            bank_id=state.bank_id,
            currency=state.currency,
            clean_balance=state.clean_balance,
            illicit_balance=state.illicit_balance,
        )

    def _base_agent(self, idx: int, role: str):
        account = self._register_account(f"{role}-{idx}-acct", clean_balance=10000)
        return self._memory(), [account], self._make_reasoner()

    def _create_bank_agents(self) -> List[BankAgent]:
        agents: List[BankAgent] = []
        for bank in self.banks.values():
            account = self._register_account(
                f"bank-{bank.bank_id}-treasury",
                clean_balance=500000,
                bank_id=bank.bank_id,
                currency=bank.supported_currencies[0] if bank.supported_currencies else self.default_currency,
            )
            affect = self._affect_for_role("bank")
            bank_agent = BankAgent(
                f"bank-{bank.bank_id}",
                bank.name,
                [account],
                self._memory(),
                self.knowledge_base,
                self._make_reasoner(),
                mood=affect.get("mood", "calm"),
                risk_tolerance=float(affect.get("risk_tolerance", 0.35)),
                fatigue=float(affect.get("fatigue", 0.05)),
                stress=float(affect.get("stress", 0.15)),
                confidence=float(affect.get("confidence", 0.65)),
            )
            setattr(bank_agent, "bank", bank)
            agents.append(bank_agent)
        return agents

    def _create_boss(self) -> BossAgent:
        mem, accounts, reasoner = self._base_agent(1, "boss")
        affect = self._affect_for_role("boss")
        return BossAgent(
            "boss-1",
            "Money Laundering Boss",
            accounts,
            mem,
            self.knowledge_base,
            reasoner,
            mood=affect["mood"],
            risk_tolerance=affect["risk_tolerance"],
            fatigue=affect["fatigue"],
            stress=affect["stress"],
            confidence=affect["confidence"],
        )

    def _create_mastermind(self) -> MastermindAgent:
        mem, accounts, reasoner = self._base_agent(1, "mastermind")
        affect = self._affect_for_role("mastermind")
        return MastermindAgent(
            "mastermind-1",
            "Money Shop Mastermind",
            accounts,
            mem,
            self.knowledge_base,
            reasoner,
            mood=affect["mood"],
            risk_tolerance=affect["risk_tolerance"],
            fatigue=affect["fatigue"],
            stress=affect["stress"],
            confidence=affect["confidence"],
        )

    def _create_courier(self) -> CourierAgent:
        mem, accounts, reasoner = self._base_agent(1, "courier")
        affect = self._affect_for_role("courier")
        return CourierAgent(
            "courier-1",
            "Courier",
            accounts,
            mem,
            self.knowledge_base,
            reasoner,
            mood=affect["mood"],
            risk_tolerance=affect["risk_tolerance"],
            fatigue=affect["fatigue"],
            stress=affect["stress"],
            confidence=affect["confidence"],
        )

    def _create_business_owners(self) -> List[BusinessOwnerAgent]:
        owners = []
        business_names = ["restaurant", "car_shop", "e_shop"]
        for idx, btype in enumerate(business_names, 1):
            mem, accounts, reasoner = self._base_agent(idx, f"owner-{btype}")
            affect = self._affect_for_role("business_owner")
            owner = BusinessOwnerAgent(
                f"owner-{idx}",
                "Front Business Owner",
                accounts,
                mem,
                self.knowledge_base,
                reasoner,
                mood=affect["mood"],
                risk_tolerance=affect["risk_tolerance"],
                fatigue=affect["fatigue"],
                stress=affect["stress"],
                confidence=affect["confidence"],
            )
            owner.business_type = btype
            owners.append(owner)
        return owners

    def _create_accountants(self) -> List[AccountantAgent]:
        accountants = []
        for idx in range(1, 4):
            mem, accounts, reasoner = self._base_agent(idx, "accountant")
            affect = self._affect_for_role("accountant")
            accountants.append(
                AccountantAgent(
                    f"accountant-{idx}",
                    "Accountant",
                    accounts,
                    mem,
                    self.knowledge_base,
                    reasoner,
                    mood=affect["mood"],
                    risk_tolerance=affect["risk_tolerance"],
                    fatigue=affect["fatigue"],
                    stress=affect["stress"],
                    confidence=affect["confidence"],
                )
            )
        return accountants

    def _create_employees(self) -> List[EmployeeAgent]:
        employees: List[EmployeeAgent] = []
        for idx in range(1, self._scale_count(self.config.num_employees) + 1):
            mem, accounts, reasoner = self._base_agent(idx, "employee")
            affect = self._affect_for_role("employee")
            employees.append(
                EmployeeAgent(
                    f"employee-{idx}",
                    "Employee",
                    accounts,
                    mem,
                    self.knowledge_base,
                    reasoner,
                    mood=affect["mood"],
                    risk_tolerance=affect["risk_tolerance"],
                    fatigue=affect["fatigue"],
                    stress=affect["stress"],
                    confidence=affect["confidence"],
                )
            )
        return employees

    def _create_mules(self) -> List[MuleAgent]:
        mules = []
        for idx in range(1, self._scale_count(self.config.num_mules) + 1):
            accounts = [f"mule-{idx}-acct-1", f"mule-{idx}-acct-2"]
            mem = self._memory(max_items=300)
            affect = self._affect_for_role("mule")
            mules.append(
                MuleAgent(
                    f"mule-{idx}",
                    "Money Mule",
                    [self._register_account(acc) for acc in accounts],
                    mem,
                    self.knowledge_base,
                    self._make_reasoner(),
                    mood=affect["mood"],
                    risk_tolerance=affect["risk_tolerance"],
                    fatigue=affect["fatigue"],
                    stress=affect["stress"],
                    confidence=affect["confidence"],
                )
            )
        return mules

    def _create_residents(self) -> List[ResidentAgent]:
        residents = []
        for idx in range(1, self._scale_count(self.config.num_residents) + 1):
            mem, accounts, reasoner = self._base_agent(idx, "resident")
            affect = self._affect_for_role("resident")
            residents.append(
                ResidentAgent(
                    f"resident-{idx}",
                    "Resident",
                    accounts,
                    mem,
                    self.knowledge_base,
                    reasoner,
                    mood=affect["mood"],
                    risk_tolerance=affect["risk_tolerance"],
                    fatigue=affect["fatigue"],
                    stress=affect["stress"],
                    confidence=affect["confidence"],
                )
            )
        return residents

    def _create_tellers(self) -> List[BankTellerAgent]:
        tellers = []
        for idx in range(1, 3):
            mem, accounts, reasoner = self._base_agent(idx, "teller")
            affect = self._affect_for_role("bank_teller")
            tellers.append(
                BankTellerAgent(
                    f"teller-{idx}",
                    "Bank Teller",
                    accounts,
                    mem,
                    self.knowledge_base,
                    reasoner,
                    mood=affect["mood"],
                    risk_tolerance=affect["risk_tolerance"],
                    fatigue=affect["fatigue"],
                    stress=affect["stress"],
                    confidence=affect["confidence"],
                )
            )
        return tellers

    def _create_back_office(self) -> BackOfficeAgent:
        mem, accounts, reasoner = self._base_agent(1, "back-office")
        affect = self._affect_for_role("back_office")
        return BackOfficeAgent(
            "back-office-1",
            "Back Office Operator",
            accounts,
            mem,
            self.knowledge_base,
            reasoner,
            mood=affect["mood"],
            risk_tolerance=affect["risk_tolerance"],
            fatigue=affect["fatigue"],
            stress=affect["stress"],
            confidence=affect["confidence"],
        )

    def _create_regulators(self) -> List[RegulatorAgent]:
        regulators = []
        for idx in range(1, self._scale_count(self.config.num_regulators) + 1):
            mem, accounts, reasoner = self._base_agent(idx, "regulator")
            affect = self._affect_for_role("regulator")
            regulators.append(
                RegulatorAgent(
                    f"regulator-{idx}",
                    "Regulator",
                    accounts,
                    mem,
                    self.knowledge_base,
                    reasoner,
                    mood=affect["mood"],
                    risk_tolerance=affect["risk_tolerance"],
                    fatigue=affect["fatigue"],
                    stress=affect["stress"],
                    confidence=affect["confidence"],
                )
            )
        return regulators
    
    def _create_companies(self) -> List[Company]:
        companies: List[Company] = []
        total_real = self._scale_count(self.config.num_real_companies)
        total_shell = self._scale_count(self.config.num_shell_companies)

        # Real operating companies
        for idx in range(1, total_real + 1):
            account = self._register_account(f"company-real-{idx}")
            owner_agent = random.choice(self.business_owners)
            company = Company(
                id=f"company-real-{idx}",
                name=f"RealCo-{idx}",
                accounts=[account.account_id],
                owners=[Person(owner_agent.id, owner_agent.role, [acct.account_id for acct in owner_agent.accounts])],
                suppliers=[],
                customers=[],
                is_shell=False,
            )
            companies.append(company)

        # Shells with seed illicit balances
        for idx in range(1, total_shell + 1):
            illicit_seed = 20000 * self.laundering_intensity or 10000
            account = self._register_account(f"company-shell-{idx}", illicit_balance=illicit_seed)
            owner_choice = random.choice(companies) if companies else random.choice(self.business_owners)
            owner_entity = (
                owner_choice
                if isinstance(owner_choice, Company)
                else Person(owner_choice.id, getattr(owner_choice, "role", "owner"), [acct.account_id for acct in owner_choice.accounts])
            )
            company = Company(
                id=f"company-shell-{idx}",
                name=f"ShellCo-{idx}",
                accounts=[account.account_id],
                owners=[owner_entity],
                suppliers=[],
                customers=[],
                is_shell=True,
            )
            companies.append(company)

        # Enrich ownership graph depth by chaining companies through shell layers
        if companies:
            max_depth = max(1, int(self.config.ownership_graph_depth))
            for company in companies:
                current_depth = len([o for o in company.owners if isinstance(o, Company)])
                while current_depth < max_depth:
                    potential_owner = random.choice(companies)
                    if potential_owner.id == company.id or potential_owner in company.owners:
                        break
                    company.owners.append(potential_owner)
                    current_depth += 1

        # Supplier and customer graphs
        for company in companies:
            degree = max(1, self.config.supplier_graph_degree)
            partners = [c for c in companies if c.id != company.id]
            random.shuffle(partners)
            for supplier in partners[:degree]:
                company.add_supplier(supplier)
            for _ in range(degree):
                company.add_customer(random.choice(self.person_entities) if self.person_entities else company)
        return companies

    def _assign_staff_to_companies(self):
        staff_map: Dict[str, List[EmployeeAgent]] = {c.id: [] for c in self.companies}
        if not staff_map:
            return staff_map
        for emp in self.employees:
            company = random.choice(self.companies)
            staff_map[company.id].append(emp)
        return staff_map

    @property
    def all_agents(self):
        return (
            self.bank_agents
            + [self.boss, self.mastermind, self.courier, self.back_office]
            + self.accountants
            + self.business_owners
            + self.employees
            + self.mules
            + self.residents
            + self.bank_tellers
            + self.regulators
        )

    def _index_agents(self) -> None:
        self.agents_by_id: Dict[str, object] = {agent.id: agent for agent in self.all_agents}

    def _relationship_summary(self, agent_id: str) -> str:
        links = self.relationship_graph.get(agent_id, {})
        if not links:
            return ""
        sorted_links = sorted(links.items(), key=lambda kv: kv[1], reverse=True)[:3]
        return ", ".join([f"{peer} (trust {score:.2f})" for peer, score in sorted_links])

    def _apply_relationship_contexts(self) -> None:
        for agent in self.all_agents:
            setattr(agent, "relationship_context", self._relationship_summary(agent.id))
            self._update_policy_context(agent)

    def _capture_affect_snapshot(self, day_index: int, label: str) -> None:
        """Record a snapshot of all agents' affective states for monitoring."""

        snapshot = {
            agent.id: {
                "mood": agent.mood,
                "risk_tolerance": agent.risk_tolerance,
                "fatigue": getattr(agent, "fatigue", 0.0),
                "stress": getattr(agent, "stress", 0.0),
                "confidence": getattr(agent, "confidence", 0.0),
            }
            for agent in self.all_agents
        }
        day_record = self.daily_affect_snapshots.setdefault(day_index, {})
        day_record[label] = snapshot

    def _update_policy_context(self, agent: Agent) -> None:
        policy_lines = [self.risk_model.active_policy_summary(), *self.policy_directives]
        agent.policy_context = "\n".join([line for line in policy_lines if line])

    def _update_policy_contexts(self) -> None:
        for agent in self.all_agents:
            self._update_policy_context(agent)

    def register_policy_change(self, description: str) -> None:
        """Record a new policy directive and propagate to agents' prompts."""

        self.policy_directives.append(description)
        if len(self.policy_directives) > 10:
            self.policy_directives = self.policy_directives[-10:]
        for agent in self.all_agents:
            agent.update_memory(f"Policy change: {description}", event_type="policy", importance=1.8)
        self._update_policy_contexts()

    def _log_event(self, agent: Agent, event_type: str, description: str, target_id: Optional[str] = None, importance: float | None = None) -> None:
        """Centralized event logging to keep AgentMemory and affect in sync with the recorder."""

        self.event_recorder.record_event(agent.id, agent.role, event_type, description, target_id=target_id)
        if hasattr(agent, "update_memory"):
            agent.update_memory(f"{event_type}: {description}", event_type=event_type, importance=importance)

    def _record_transaction(
        self,
        sender_account: str,
        receiver_account: str,
        amount: float,
        tx_type: str,
        is_money_laundering: bool,
        ml_typology: Optional[str],
        ml_pattern: Optional[str] = None,
        pattern_scheme_id: str | int | None = None,
        currency: Optional[str] = None,
        current_day: Optional[int] = None,
        channel: Optional[str] = None,
    ):
        tx_currency = currency or self._account_currency(sender_account)
        tx_channel = channel or self._choose_channel(tx_type, is_money_laundering)
        tx = self.event_recorder.record_transaction(
            sender_account=sender_account,
            receiver_account=receiver_account,
            amount=amount,
            currency=tx_currency,
            tx_type=tx_type,
            channel=tx_channel,
            is_money_laundering=is_money_laundering,
            ml_typology=ml_typology,
            ml_pattern=ml_pattern,
            pattern_scheme_id=pattern_scheme_id,
            current_day=current_day,
        )
        day_key = tx.event_day if tx.event_day is not None else self.current_day_index
        self.daily_transaction_ids.setdefault(day_key, []).append(tx.tx_id)
        return tx
    
    def run(self) -> None:
        start_date = datetime(2024, 1, 1)
        for day_idx in range(self.config.simulation_days):
            current_date = start_date + timedelta(days=day_idx)
            self._run_day(current_date, day_idx)

    def send_user_message(self, agent_id: str, text: str, user_role: str | None = None) -> str:
        """Route a user-authored message to a specific agent."""

        agent = self.agents_by_id.get(agent_id)
        if not agent:
            raise KeyError(f"Agent {agent_id} not found")
        reply = interface.user_message(agent, text, user_role=user_role)
        self._log_event(agent, "interaction", reply)
        return reply
    
    def _run_day(self, current_date: datetime, day_index: int) -> None:
        self.current_day_index = day_index
        weekday = current_date.strftime("%A")
        schedules = self.config.daily_schedules
        self.event_recorder.release_settlements(day_index)
        day_start_idx = len(self.event_recorder.logs)
        self._apply_relationship_contexts()
        self._capture_affect_snapshot(day_index, "start")

        # Strategy and planning by boss and accountants
        boss_plan = self.boss.plan_strategy()
        self._log_event(self.boss, "plan", boss_plan)
        for accountant in self.accountants:
            typology = random.choice(ACCOUNTING_TYPOLOGIES)
            desc = accountant.design_typology(typology)
            self._log_event(accountant, "typology", desc)

        # Canonical laundering pattern injection
        self._generate_laundering_patterns(day_index)

        # Placement stage
        transfer_cfg = schedules.get("mastermind_transfers", {})
        max_amount = transfer_cfg.get("max_amount", 500000)
        parcel_range = transfer_cfg.get("parcel_range", [10000, 50000])
        for _ in transfer_cfg.get("times", []):
            incoming = random.randint(int(max_amount * 0.6), max_amount)
            parcels = []
            remaining = incoming
            while remaining > 0:
                size = random.randint(parcel_range[0], parcel_range[1])
                size = min(size, remaining)
                parcels.append(size)
                remaining -= size
            mm_desc = self.mastermind.parcel_funds(parcels)
            self._log_event(self.mastermind, "parcel", mm_desc)

            courier_desc = self.courier.move_parcels(len(parcels), sum(parcels))
            self._log_event(self.courier, "pickup", courier_desc, target_id=self.mastermind.id)

            courier_tx = self._record_transaction(
                sender_account=self.mastermind.primary_account,
                receiver_account=self.courier.primary_account,
                amount=sum(parcels),
                tx_type="placement",
                is_money_laundering=True,
                ml_typology="placement",
                current_day=day_index,
            )
            self._evaluate_risk(courier_tx)

            # deliver parcels to business owners and mules
            receivers = self.business_owners + self.mules
            random.shuffle(receivers)
            for parcel, receiver in zip(parcels, receivers):
                tx = self._record_transaction(
                    sender_account=self.courier.primary_account,
                    receiver_account=receiver.primary_account,
                    amount=parcel,
                    tx_type="placement",
                    is_money_laundering=True,
                    ml_typology="placement",
                    current_day=day_index,
                )
                desc = receiver.receive_parcel(parcel) if isinstance(receiver, BusinessOwnerAgent) else receiver.redistribute([parcel])
                self._log_event(receiver, "receive_parcel", desc, target_id=self.courier.id, importance=1.4)
                self._evaluate_risk(tx)

        # Layering for businesses
        intake_cfg = schedules.get("business_intake", {})
        for owner in self.business_owners:
            rule = intake_cfg.get(getattr(owner, "business_type", ""), {})
            if weekday in rule.get("days", []):
                amount = rule.get("amount", 20000)
                tx = self._record_transaction(
                    sender_account=self.mastermind.primary_account,
                    receiver_account=owner.primary_account,
                    amount=amount,
                    tx_type="layering",
                    is_money_laundering=True,
                    ml_typology="business_layering",
                    current_day=day_index,
                )
                self._log_event(owner, "business_intake", f"Layering inflow {amount}")
                if hasattr(owner, "integrate_funds"):
                    owner.integrate_funds(amount)
                self._evaluate_risk(tx)

        # Company-level activities (B2B, suppliers, payroll)
        for company in self.companies:
            company_account = company.accounts[0] if company.accounts else ""
            shell_bias = self.laundering_intensity if company.is_shell else self.laundering_intensity * 0.2

            # Revenue from customers
            for _ in range(max(1, int(self.transaction_scale))):
                if not company.customers:
                    break
                customer = random.choice(company.customers)
                sender_account = customer.accounts[0] if isinstance(customer, Person) else (customer.accounts[0] if customer.accounts else company_account)
                amount = int(self._sample_amount("revenue", (3000, 12000)))
                tx = self._record_transaction(
                    sender_account=sender_account,
                    receiver_account=company_account,
                    amount=amount,
                    tx_type="revenue",
                    is_money_laundering=company.is_shell and random.random() < shell_bias,
                    ml_typology="customer_inflow" if company.is_shell else None,
                    current_day=day_index,
                )
                self._evaluate_risk(tx)

            # Supplier payments
            for supplier in company.suppliers:
                if random.random() > self.transaction_scale:
                    continue
                amount = int(self._sample_amount("supplier_payment", (4000, 15000)))
                tx = self._record_transaction(
                    sender_account=company_account,
                    receiver_account=supplier.accounts[0] if supplier.accounts else company_account,
                    amount=amount,
                    tx_type="supplier_payment",
                    is_money_laundering=company.is_shell and random.random() < shell_bias,
                    ml_typology="supplier_layering" if company.is_shell else None,
                    current_day=day_index,
                )
                self._evaluate_risk(tx)

            # Payroll to staff
            for emp in self.company_staff.get(company.id, []):
                base_pay = self._sample_amount("payroll", (3000, 8000))
                pay_amount = int(self._risk_scaled_amount(base_pay, None))
                tx = self._record_transaction(
                    sender_account=company_account,
                    receiver_account=emp.primary_account,
                    amount=pay_amount,
                    tx_type="payroll",
                    is_money_laundering=company.is_shell and random.random() < shell_bias,
                    ml_typology="inflated_payroll" if company.is_shell else None,
                    current_day=day_index,
                )
                self._evaluate_risk(tx)
                
        # Employee shifts generate legitimate revenue events
        for business in self.business_owners:
            staffed = [emp for emp in self.employees if int(emp.id.split("-")[1]) % 3 == self.business_owners.index(business) % 3]
            for emp in staffed:
                shift_note = emp.perform_shift(getattr(business, "business_type", "business"))
                self._log_event(emp, "shift", shift_note, target_id=business.id)

        # Mules distribution schedule
        mule_cfg = schedules.get("mule_payments", {})
        if weekday in mule_cfg.get("days", []):
            for mule in self.mules:
                payment_count = int(mule_cfg.get("payment_count", 5) * self.transaction_scale) or 1
                base_payment = mule_cfg.get("payment_amount", 2000)
                calibrated = self._sample_amount("mule_payment", (base_payment * 0.6, base_payment * 1.5))
                payments = [int(self._risk_scaled_amount(calibrated, mule)) for _ in range(payment_count)]
                mule.redistribute(payments)
                for amt in payments:
                    target = random.choice(self.residents + self.business_owners)
                    tx = self._record_transaction(
                        sender_account=mule.primary_account,
                        receiver_account=target.primary_account,
                        amount=amt,
                        tx_type="layering",
                        is_money_laundering=True,
                        ml_typology="mule_scatter",
                        current_day=day_index,
                    )
                    self._log_event(mule, "mule_transfer", f"Sent {amt} to {target.id}")
                    self._evaluate_risk(tx)

        # Integration stage for payroll and suppliers
        for owner in self.business_owners:
            payroll_amount = int(self._sample_amount("integration_payroll", (5000, 15000)))
            desc = owner.integrate_funds(payroll_amount)
            self._log_event(owner, "integration", desc)
            tx = self._record_transaction(
                sender_account=owner.primary_account,
                receiver_account=random.choice(self.residents).primary_account,
                amount=payroll_amount,
                tx_type="integration",
                is_money_laundering=False,
                ml_typology=None,
                current_day=day_index,
            )
            self._evaluate_risk(tx)

        # Ordinary resident salaries
        if current_date.day in schedules.get("salary_days", [1, 15]):
            for resident in self.residents:
                salary = int(self._sample_amount("salary", (20000, 50000)))
                resident.normal_activity(salary)
                tx = self._record_transaction(
                    sender_account=self.back_office.primary_account,
                    receiver_account=resident.primary_account,
                    amount=salary,
                    tx_type="salary",
                    is_money_laundering=False,
                    ml_typology=None,
                    current_day=day_index,
                )
                self._log_event(self.back_office, "salary", f"Paid salary {salary} to {resident.id}")
                self._evaluate_risk(tx)

        # Deposits at bank tellers (can trigger SAR)
        depositors = self.business_owners + self.mules + self.residents
        for depositor in depositors:
            amount = int(self._sample_amount("deposit", (2000, 140000)))
            amount = int(self._risk_scaled_amount(amount, depositor))
            teller = random.choice(self.bank_tellers)
            is_launder = isinstance(depositor, (BusinessOwnerAgent, MuleAgent, MastermindAgent))
            tx = self._record_transaction(
                sender_account=depositor.primary_account,
                receiver_account=teller.primary_account,
                amount=amount,
                tx_type="deposit",
                is_money_laundering=is_launder,
                ml_typology="structuring" if is_launder and amount > 90000 else None,
                current_day=day_index,
            )
            desc = f"Deposit of {amount} by {depositor.id}"
            if amount >= self.risk_model.high_risk_amount:
                teller_note = teller.flag_deposit(amount)
                self._log_event(teller, "sar", teller_note, target_id=depositor.id, importance=1.8)
            self._log_event(depositor, "deposit", desc, target_id=teller.id, importance=1.2)
            self._evaluate_risk(tx)

        # Random consumer spending
        for resident in self.residents:
            spend = int(self._sample_amount("spend", (500, 3000)))
            resident.normal_activity(spend)
            tx = self._record_transaction(
                sender_account=resident.primary_account,
                receiver_account=random.choice(self.business_owners).primary_account,
                amount=spend,
                tx_type="spend",
                is_money_laundering=False,
                ml_typology=None,
                current_day=day_index,
            )
            self._log_event(resident, "spend", f"Spent {spend}")
            self._evaluate_risk(tx)

        # Peer-to-peer transfers and bill payments to public facilities
        public_accounts = ["city_hall", "tax_office", "hospital", "school"]
        p2p_iterations = max(1, int(len(self.residents) * self.transaction_scale))
        for _ in range(p2p_iterations):
            sender = random.choice(self.residents)
            receiver = random.choice(self.residents)
            if sender.id == receiver.id:
                receiver = random.choice(self.residents)
            amount = int(self._sample_amount("p2p", (200, 5000)))
            tx = self._record_transaction(
                sender_account=sender.primary_account,
                receiver_account=receiver.primary_account,
                amount=amount,
                tx_type="p2p",
                is_money_laundering=False,
                ml_typology=None,
                current_day=day_index,
            )
            self._log_event(sender, "p2p", f"Sent {amount} to {receiver.id}")
            self._evaluate_risk(tx)

        for _ in range(max(1, int(2 * self.transaction_scale))):
            payer = random.choice(self.residents)
            facility = random.choice(public_accounts)
            amount = int(self._sample_amount("bill", (800, 4000)))
            tx = self._record_transaction(
                sender_account=payer.primary_account,
                receiver_account=facility,
                amount=amount,
                tx_type="bill",
                is_money_laundering=False,
                ml_typology=None,
                current_day=day_index,
            )
            self._log_event(payer, "bill", f"Paid {amount} to {facility}")
            self._evaluate_risk(tx)

        # Social interactions between agents
        if self.config.enable_social and self.all_agents:
            interactions = max(1, int(self.config.social_interactions_per_day * self.population_scale))
            rumor_bank = random.choice(list(self.banks.values()))
            rumor_currency = random.choice(self.available_currencies)
            rumor_pattern = random.choice(ALL_LAUNDERING_PATTERNS)
            for _ in range(interactions):
                if len(self.all_agents) < 2:
                    break
                agent_a, agent_b = random.sample(self.all_agents, 2)
                context_str = self._relationship_summary(agent_a.id)
                rumor = f"{rumor_bank.name} is tightening checks on {rumor_currency} {rumor_pattern} flows"
                dialogue = generate_social_event(agent_a, agent_b, context_str, rumor=rumor)
                self._log_event(agent_a, "social", dialogue, target_id=agent_b.id)
                update_relationship(agent_a, agent_b, dialogue, self.relationship_graph)
                if random.random() < 0.25:
                    diffuse_information(self.all_agents, rumor)

            # Targeted recruitment into illicit roles
            recruiter = random.choice([self.mastermind, self.boss])
            prospect = random.choice(self.residents)
            dialogue, success = generate_recruitment_event(recruiter, prospect, target_role="money mule", relationship_graph=self.relationship_graph)
            self._log_event(recruiter, "recruitment", dialogue, target_id=prospect.id, importance=1.6)
            if success:
                self._log_event(prospect, "recruitment", f"Considering mule work after talk with {recruiter.id}", target_id=recruiter.id, importance=1.4)

        # End-of-day reflection and planning
        if self.config.enable_reflection:
            today_logs = self.event_recorder.logs[day_start_idx:]
            by_agent: Dict[str, List[str]] = {}
            for log in today_logs:
                by_agent.setdefault(log.agent_id, []).append(f"{log.event_type}: {log.description}")
            for agent in self.all_agents:
                events = by_agent.get(agent.id, [])
                if not events:
                    continue
                salient = [e for e in events if any(key in e for key in ["sar", "ml_pattern", "deposit", "layering"])]
                summary = "; ".join(salient or events[-5:])
                reflection = agent.reflect(summary)
                plan = agent.update_long_term_plan(reflection)
                self._log_event(agent, "reflection", reflection, importance=1.5)
                self._log_event(agent, "plan", plan, importance=1.4)

        self._capture_affect_snapshot(day_index, "end")
        self._apply_relationship_contexts()
        self.daily_summaries[day_index] = monitor.generate_daily_summary(self, day_index)

    def _evaluate_risk(self, tx) -> None:
        score = self.risk_model.score_transaction(
            tx.amount,
            tx.is_money_laundering,
            tx.illicit_amount,
            tx.illicit_fraction,
            cross_bank=tx.cross_bank,
            cross_currency=tx.cross_currency,
            channel=tx.channel,
            tx_type=tx.tx_type,
        )
        day_index = tx.event_day if getattr(tx, "event_day", None) is not None else self.current_day_index
        self.daily_risk_scores.setdefault(day_index, []).append(score)
        if score >= self.risk_model.threshold:
            self.daily_alerts[day_index] = self.daily_alerts.get(day_index, 0) + 1
            regulator = random.choice(self.regulators)
            note = regulator.open_investigation(tx.tx_id, score)
            self._log_event(regulator, "sar", note, target_id=str(tx.tx_id), importance=2.0)

    def get_daily_summary(self, day_index: Optional[int] = None, as_json: bool = False):
        """Return monitoring summary for a given simulation day.

        If ``day_index`` is not provided, the most recent computed day is used.
        Set ``as_json=True`` to receive a structured dictionary instead of text.
        """

        target_day = day_index if day_index is not None else (max(self.daily_summaries.keys()) if self.daily_summaries else 0)
        summary = self.daily_summaries.get(target_day) or monitor.generate_daily_summary(self, target_day)
        if as_json:
            return summary
        return summary.get("summary_text", str(summary))
    
    def _split_amount(self, total_amount: float, parts: int) -> List[float]:
        """Split ``total_amount`` into ``parts`` positive values that sum to the original."""

        if parts <= 0:
            return []
        weights = [random.random() for _ in range(parts)]
        weight_sum = sum(weights) or 1.0
        allocations = [round(total_amount * w / weight_sum, 2) for w in weights]
        remainder = round(total_amount - sum(allocations), 2)
        if allocations:
            allocations[0] = round(allocations[0] + remainder, 2)
        return allocations

    def _generate_laundering_patterns(self, day_index: int) -> None:
        cfg = self.pattern_generation or {}
        if not cfg.get("enabled", False):
            return
        per_day = max(1, int(cfg.get("per_day", 1) * self.transaction_scale))
        base_amount = float(cfg.get("base_amount", 120000)) * max(self.laundering_intensity, 0.05)

        account_pool = list(
            {
                acct.account_id
                for agent in self.all_agents
                for acct in getattr(agent, "accounts", [])
            }
            | {acct for company in self.companies for acct in company.accounts}
        )
        if len(account_pool) < 6:
            return

        patterns = ALL_LAUNDERING_PATTERNS
        random.shuffle(account_pool)

        for idx in range(per_day):
            pattern_name = patterns[(idx + day_index) % len(patterns)]
            scheme_id = f"day{day_index}-{pattern_name}-{idx}"
            amount = base_amount * random.uniform(0.5, 1.2)

            def sample_accounts(count: int) -> List[str]:
                return random.sample(account_pool, min(count, len(account_pool)))

            try:
                if pattern_name == FAN_OUT:
                    controller, *recipients = sample_accounts(4)
                    self._seed_illicit_balance(controller, amount)
                    tx_ids = self._generate_fan_out_pattern(
                        controller,
                        recipients,
                        amount,
                        currency=self._account_currency(controller),
                        scheme_id=scheme_id,
                        current_day=day_index,
                    )
                elif pattern_name == FAN_IN:
                    sink, *sources = sample_accounts(4)
                    self._seed_illicit_balance(sources[0], amount)
                    tx_ids = self._generate_fan_in_pattern(
                        sources,
                        sink,
                        amount,
                        currency=self._account_currency(sources[0]),
                        scheme_id=scheme_id,
                        current_day=day_index,
                    )
                elif pattern_name == GATHER_SCATTER:
                    gather, src_a, src_b, dst_a, dst_b = sample_accounts(5)
                    self._seed_illicit_balance(src_a, amount / 2)
                    self._seed_illicit_balance(src_b, amount / 2)
                    tx_ids = self._generate_gather_scatter_pattern(
                        gather,
                        [src_a, src_b],
                        [dst_a, dst_b],
                        amount,
                        currency=self._account_currency(gather),
                        scheme_id=scheme_id,
                        current_day=day_index,
                    )
                elif pattern_name == SCATTER_GATHER:
                    source, sink, *intermediates = sample_accounts(4)
                    self._seed_illicit_balance(source, amount)
                    tx_ids = self._generate_scatter_gather_pattern(
                        source,
                        sink,
                        intermediates[:2],
                        amount,
                        currency=self._account_currency(source),
                        scheme_id=scheme_id,
                        current_day=day_index,
                    )
                elif pattern_name == SIMPLE_CYCLE:
                    nodes = sample_accounts(4)
                    self._seed_illicit_balance(nodes[0], amount)
                    tx_ids = self._generate_simple_cycle_pattern(
                        nodes,
                        amount,
                        currency=self._account_currency(nodes[0]),
                        scheme_id=scheme_id,
                        current_day=day_index,
                    )
                elif pattern_name == RANDOM:
                    path = sample_accounts(5)
                    self._seed_illicit_balance(path[0], amount)
                    tx_ids = self._generate_random_pattern(
                        path,
                        amount,
                        currency=self._account_currency(path[0]),
                        scheme_id=scheme_id,
                        current_day=day_index,
                    )
                elif pattern_name == BIPARTITE:
                    senders = sample_accounts(3)
                    receivers = sample_accounts(3)
                    self._seed_illicit_balance(senders[0], amount)
                    tx_ids = self._generate_bipartite_pattern(
                        senders,
                        receivers,
                        amount,
                        currency=self._account_currency(senders[0]),
                        scheme_id=scheme_id,
                        current_day=day_index,
                    )
                elif pattern_name == STACK:
                    layer1 = sample_accounts(2)
                    layer2 = sample_accounts(2)
                    layer3 = sample_accounts(2)
                    self._seed_illicit_balance(layer1[0], amount)
                    tx_ids = self._generate_stack_pattern(
                        [layer1, layer2, layer3],
                        amount,
                        currency=self._account_currency(layer1[0]),
                        scheme_id=scheme_id,
                        current_day=day_index,
                    )
                else:
                    continue
                self._log_event(self.mastermind, "ml_pattern", f"Generated {pattern_name} scheme {scheme_id} with tx {tx_ids}", importance=1.6)
            except ValueError as exc:
                self._log_event(self.mastermind, "ml_pattern_error", f"Failed {pattern_name}: {exc}")
    
    def _generate_fan_out_pattern(
        self,
        controller_account: str,
        recipient_accounts: List[str],
        total_amount: float,
        currency: str,
        scheme_id: str,
        current_day: Optional[int] = None,
    ) -> List[int]:
        """
        Generate a fan-out pattern where a single source splits funds to >=2 recipients.

        Funds from ``controller_account`` are split across ``recipient_accounts``. All
        transactions are labeled with the fan-out pattern and linked by ``scheme_id``.
        Returns the list of created transaction ids.
        """

        if len(recipient_accounts) < 2:
            raise ValueError("Fan-out pattern requires at least two recipient accounts")

        allocations = self._split_amount(total_amount, len(recipient_accounts))
        tx_ids: List[int] = []
        for receiver, amount in zip(recipient_accounts, allocations):
            tx = self._record_transaction(
                sender_account=controller_account,
                receiver_account=receiver,
                amount=amount,
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=FAN_OUT,
                ml_pattern=FAN_OUT,
                pattern_scheme_id=scheme_id,
                current_day=current_day,
            )
            tx_ids.append(tx.tx_id)
        return tx_ids

    def _generate_fan_in_pattern(
        self,
        source_accounts: List[str],
        sink_account: str,
        total_amount: float,
        currency: str,
        scheme_id: str,
        current_day: Optional[int] = None,
    ) -> List[int]:
        """
        Generate a fan-in pattern where multiple sources consolidate into a sink.
        """

        if len(source_accounts) < 2:
            raise ValueError("Fan-in pattern requires at least two source accounts")

        allocations = self._split_amount(total_amount, len(source_accounts))
        tx_ids: List[int] = []
        for sender, amount in zip(source_accounts, allocations):
            tx = self._record_transaction(
                sender_account=sender,
                receiver_account=sink_account,
                amount=amount,
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=FAN_IN,
                ml_pattern=FAN_IN,
                pattern_scheme_id=scheme_id,
                current_day=current_day,
            )
            tx_ids.append(tx.tx_id)
        return tx_ids

    def _generate_gather_scatter_pattern(
        self,
        gather_account: str,
        source_accounts: List[str],
        recipient_accounts: List[str],
        total_amount: float,
        currency: str,
        scheme_id: str,
        current_day: Optional[int] = None,
    ) -> List[int]:
        """
        Generate a gather-scatter pattern where one hub gathers then redistributes.
        """

        if len(source_accounts) < 2 or len(recipient_accounts) < 2:
            raise ValueError("Gather-scatter requires at least two sources and two recipients")

        incoming_amounts = self._split_amount(total_amount, len(source_accounts))
        outgoing_amounts = self._split_amount(total_amount, len(recipient_accounts))
        tx_ids: List[int] = []

        for sender, amount in zip(source_accounts, incoming_amounts):
            tx = self._record_transaction(
                sender_account=sender,
                receiver_account=gather_account,
                amount=amount,
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=GATHER_SCATTER,
                ml_pattern=GATHER_SCATTER,
                pattern_scheme_id=scheme_id,
                current_day=current_day,
            )
            tx_ids.append(tx.tx_id)

        for receiver, amount in zip(recipient_accounts, outgoing_amounts):
            tx = self._record_transaction(
                sender_account=gather_account,
                receiver_account=receiver,
                amount=amount,
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=GATHER_SCATTER,
                ml_pattern=GATHER_SCATTER,
                pattern_scheme_id=scheme_id,
                current_day=current_day,
            )
            tx_ids.append(tx.tx_id)
        return tx_ids

    def _generate_scatter_gather_pattern(
        self,
        source_account: str,
        sink_account: str,
        intermediate_accounts: List[str],
        total_amount: float,
        currency: str,
        scheme_id: str,
        current_day: Optional[int] = None,
    ) -> List[int]:
        """
        Generate a scatter-gather pattern using shared intermediates between source and sink.
        """

        if len(intermediate_accounts) < 2:
            raise ValueError("Scatter-gather requires at least two intermediates")

        scatter_amounts = self._split_amount(total_amount, len(intermediate_accounts))
        gather_amounts = self._split_amount(total_amount, len(intermediate_accounts))
        tx_ids: List[int] = []

        for receiver, amount in zip(intermediate_accounts, scatter_amounts):
            tx = self._record_transaction(
                sender_account=source_account,
                receiver_account=receiver,
                amount=amount,
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=SCATTER_GATHER,
                ml_pattern=SCATTER_GATHER,
                pattern_scheme_id=scheme_id,
                current_day=current_day,
            )
            tx_ids.append(tx.tx_id)

        for sender, amount in zip(intermediate_accounts, gather_amounts):
            tx = self._record_transaction(
                sender_account=sender,
                receiver_account=sink_account,
                amount=amount,
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=SCATTER_GATHER,
                ml_pattern=SCATTER_GATHER,
                pattern_scheme_id=scheme_id,
                current_day=current_day,
            )
            tx_ids.append(tx.tx_id)
        return tx_ids

    def _generate_simple_cycle_pattern(
        self,
        accounts: List[str],
        total_amount: float,
        currency: str,
        scheme_id: str,
        current_day: Optional[int] = None,
    ) -> List[int]:
        """
        Generate a simple directed cycle across distinct accounts.
        """

        if len(accounts) < 3:
            raise ValueError("Simple cycle requires at least three distinct accounts")

        allocations = self._split_amount(total_amount, len(accounts))
        tx_ids: List[int] = []
        for idx, sender in enumerate(accounts):
            receiver = accounts[(idx + 1) % len(accounts)]
            tx = self._record_transaction(
                sender_account=sender,
                receiver_account=receiver,
                amount=allocations[idx],
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=SIMPLE_CYCLE,
                ml_pattern=SIMPLE_CYCLE,
                pattern_scheme_id=scheme_id,
                current_day=current_day,
            )
            tx_ids.append(tx.tx_id)
        return tx_ids

    def _generate_random_pattern(
        self,
        account_path: List[str],
        total_amount: float,
        currency: str,
        scheme_id: str,
        current_day: Optional[int] = None,
    ) -> List[int]:
        """
        Generate a random-walk-style chain without returning to the origin.
        """

        if len(account_path) < 2:
            raise ValueError("Random pattern requires at least two accounts")

        allocations = self._split_amount(total_amount, len(account_path) - 1)
        tx_ids: List[int] = []
        for idx in range(len(account_path) - 1):
            tx = self._record_transaction(
                sender_account=account_path[idx],
                receiver_account=account_path[idx + 1],
                amount=allocations[idx],
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=RANDOM,
                ml_pattern=RANDOM,
                pattern_scheme_id=scheme_id,
                current_day=current_day,
            )
            tx_ids.append(tx.tx_id)
        return tx_ids

    def _generate_bipartite_pattern(
        self,
        sender_accounts: List[str],
        receiver_accounts: List[str],
        total_amount: float,
        currency: str,
        scheme_id: str,
        current_day: Optional[int] = None,
    ) -> List[int]:
        """
        Generate a bipartite pattern with edges only from senders to receivers.
        """

        if not sender_accounts or not receiver_accounts:
            raise ValueError("Bipartite pattern requires sender and receiver accounts")

        edge_count = len(sender_accounts) * len(receiver_accounts)
        allocations = self._split_amount(total_amount, edge_count)
        tx_ids: List[int] = []
        for sender in sender_accounts:
            for receiver in receiver_accounts:
                amount = allocations.pop(0)
                tx = self._record_transaction(
                    sender_account=sender,
                    receiver_account=receiver,
                    amount=amount,
                    currency=currency,
                    tx_type="layering",
                    is_money_laundering=True,
                    ml_typology=BIPARTITE,
                    ml_pattern=BIPARTITE,
                    pattern_scheme_id=scheme_id,
                    current_day=current_day,
                )
                tx_ids.append(tx.tx_id)
        return tx_ids

    def _generate_stack_pattern(
        self,
        layers: List[List[str]],
        total_amount: float,
        currency: str,
        scheme_id: str,
        current_day: Optional[int] = None,
    ) -> List[int]:
        """
        Generate a stacked multi-layer bipartite structure across successive layers.
        """

        if len(layers) < 2:
            raise ValueError("Stack pattern requires at least two layers of accounts")

        tx_ids: List[int] = []
        layer_amounts = total_amount
        for idx in range(len(layers) - 1):
            src_layer = layers[idx]
            dst_layer = layers[idx + 1]
            if not src_layer or not dst_layer:
                raise ValueError("Each stack layer must contain at least one account")
            edge_count = len(src_layer) * len(dst_layer)
            allocations = self._split_amount(layer_amounts, edge_count)
            for sender in src_layer:
                for receiver in dst_layer:
                    amount = allocations.pop(0)
                    tx = self._record_transaction(
                        sender_account=sender,
                        receiver_account=receiver,
                        amount=amount,
                        currency=currency,
                        tx_type="layering",
                        is_money_laundering=True,
                        ml_typology=STACK,
                        ml_pattern=STACK,
                        pattern_scheme_id=scheme_id,
                        current_day=current_day,
                    )
                    tx_ids.append(tx.tx_id)
        return tx_ids
