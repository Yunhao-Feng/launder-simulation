from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List

from .agents import (
    AccountantAgent,
    BackOfficeAgent,
    BankTellerAgent,
    BossAgent,
    BusinessOwnerAgent,
    CourierAgent,
    EmployeeAgent,
    MastermindAgent,
    MuleAgent,
    RegulatorAgent,
    ResidentAgent,
)
from .config import SimulationConfig
from .detection import RiskModel
from .knowledge import DPRTextEmbedder, KnowledgeBase
from .memory import AgentMemory, EventRecorder
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
        self.embedder = DPRTextEmbedder(model_path=config.knowledge_model_path, device=config.knowledge_device)
        self.knowledge_base = KnowledgeBase.from_config(
            config.knowledge_documents,
            model_path=config.knowledge_model_path,
            device=config.knowledge_device,
            embedder=self.embedder,
        )
        self.event_recorder = EventRecorder()
        self.risk_model = RiskModel(config.risk_threshold, config.risk_model)

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

        self.currencies = "CNY"

    def _memory(self, max_items: int = 500) -> AgentMemory:
        return AgentMemory(max_items=max_items, embedder=self.embedder)

    def _base_agent(self, idx: int, role: str):
        account = f"{role}-{idx}-acct"
        return self._memory(), [account]

    def _create_boss(self) -> BossAgent:
        mem, accounts = self._base_agent(1, "boss")
        return BossAgent("boss-1", "Money Laundering Boss", accounts, mem, self.knowledge_base)

    def _create_mastermind(self) -> MastermindAgent:
        mem, accounts = self._base_agent(1, "mastermind")
        return MastermindAgent("mastermind-1", "Money Shop Mastermind", accounts, mem, self.knowledge_base)

    def _create_courier(self) -> CourierAgent:
        mem, accounts = self._base_agent(1, "courier")
        return CourierAgent("courier-1", "Courier", accounts, mem, self.knowledge_base)

    def _create_business_owners(self) -> List[BusinessOwnerAgent]:
        owners = []
        business_names = ["restaurant", "car_shop", "e_shop"]
        for idx, btype in enumerate(business_names, 1):
            mem, accounts = self._base_agent(idx, f"owner-{btype}")
            owner = BusinessOwnerAgent(
                f"owner-{idx}",
                "Front Business Owner",
                accounts,
                mem,
                self.knowledge_base,
            )
            owner.business_type = btype
            owners.append(owner)
        return owners

    def _create_accountants(self) -> List[AccountantAgent]:
        accountants = []
        for idx in range(1, 4):
            mem, accounts = self._base_agent(idx, "accountant")
            accountants.append(
                AccountantAgent(
                    f"accountant-{idx}",
                    "Accountant",
                    accounts,
                    mem,
                    self.knowledge_base,
                )
            )
        return accountants

    def _create_employees(self) -> List[EmployeeAgent]:
        employees: List[EmployeeAgent] = []
        for idx in range(1, self.config.num_employees + 1):
            mem, accounts = self._base_agent(idx, "employee")
            employees.append(EmployeeAgent(f"employee-{idx}", "Employee", accounts, mem, self.knowledge_base))
        return employees

    def _create_mules(self) -> List[MuleAgent]:
        mules = []
        for idx in range(1, self.config.num_mules + 1):
            accounts = [f"mule-{idx}-acct-1", f"mule-{idx}-acct-2"]
            mem = self._memory(max_items=300)
            mules.append(MuleAgent(f"mule-{idx}", "Money Mule", accounts, mem, self.knowledge_base))
        return mules

    def _create_residents(self) -> List[ResidentAgent]:
        residents = []
        for idx in range(1, self.config.num_residents + 1):
            mem, accounts = self._base_agent(idx, "resident")
            residents.append(ResidentAgent(f"resident-{idx}", "Resident", accounts, mem, self.knowledge_base))
        return residents

    def _create_tellers(self) -> List[BankTellerAgent]:
        tellers = []
        for idx in range(1, 3):
            mem, accounts = self._base_agent(idx, "teller")
            tellers.append(BankTellerAgent(f"teller-{idx}", "Bank Teller", accounts, mem, self.knowledge_base))
        return tellers

    def _create_back_office(self) -> BackOfficeAgent:
        mem, accounts = self._base_agent(1, "back-office")
        return BackOfficeAgent("back-office-1", "Back Office Operator", accounts, mem, self.knowledge_base)

    def _create_regulators(self) -> List[RegulatorAgent]:
        regulators = []
        for idx in range(1, self.config.num_regulators + 1):
            mem, accounts = self._base_agent(idx, "regulator")
            regulators.append(RegulatorAgent(f"regulator-{idx}", "Regulator", accounts, mem, self.knowledge_base))
        return regulators
    
    def run(self) -> None:
        start_date = datetime(2024, 1, 1)
        for day_idx in range(self.config.simulation_days):
            current_date = start_date + timedelta(days=day_idx)
            self._run_day(current_date)
    
    def _run_day(self, current_date: datetime) -> None:
        weekday = current_date.strftime("%A")
        schedules = self.config.daily_schedules

        # Strategy and planning by boss and accountants
        boss_plan = self.boss.plan_strategy()
        self.event_recorder.record_event(self.boss.id, self.boss.role, "plan", boss_plan)
        for accountant in self.accountants:
            typology = random.choice(ACCOUNTING_TYPOLOGIES)
            desc = accountant.design_typology(typology)
            self.event_recorder.record_event(accountant.id, accountant.role, "typology", desc)

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
            self.event_recorder.record_event(self.mastermind.id, self.mastermind.role, "parcel", mm_desc)

            courier_desc = self.courier.move_parcels(len(parcels), sum(parcels))
            self.event_recorder.record_event(self.courier.id, self.courier.role, "pickup", courier_desc, target_id=self.mastermind.id)

            courier_tx = self.event_recorder.record_transaction(
                sender_account=self.mastermind.accounts[0],
                receiver_account=self.courier.accounts[0],
                amount=sum(parcels),
                currency=self.currencies,
                tx_type="placement",
                is_money_laundering=True,
                ml_typology="placement",
            )
            self._evaluate_risk(courier_tx)

            # deliver parcels to business owners and mules
            receivers = self.business_owners + self.mules
            random.shuffle(receivers)
            for parcel, receiver in zip(parcels, receivers):
                tx = self.event_recorder.record_transaction(
                    sender_account=self.courier.accounts[0],
                    receiver_account=receiver.accounts[0],
                    amount=parcel,
                    currency=self.currencies,
                    tx_type="placement",
                    is_money_laundering=True,
                    ml_typology="placement",
                )
                desc = receiver.receive_parcel(parcel) if isinstance(receiver, BusinessOwnerAgent) else receiver.redistribute([parcel])
                self.event_recorder.record_event(
                    agent_id=receiver.id,
                    agent_role=receiver.role,
                    event_type="receive_parcel",
                    description=desc,
                    target_id=self.courier.id,
                )
                self._evaluate_risk(tx)

        # Layering for businesses
        intake_cfg = schedules.get("business_intake", {})
        for owner in self.business_owners:
            rule = intake_cfg.get(getattr(owner, "business_type", ""), {})
            if weekday in rule.get("days", []):
                amount = rule.get("amount", 20000)
                tx = self.event_recorder.record_transaction(
                    sender_account=self.mastermind.accounts[0],
                    receiver_account=owner.accounts[0],
                    amount=amount,
                    currency=self.currencies,
                    tx_type="layering",
                    is_money_laundering=True,
                    ml_typology="business_layering",
                )
                self.event_recorder.record_event(owner.id, owner.role, "business_intake", f"Layering inflow {amount}")
                if hasattr(owner, "integrate_funds"):
                    owner.integrate_funds(amount)
                self._evaluate_risk(tx)

        # Employee shifts generate legitimate revenue events
        for business in self.business_owners:
            staffed = [emp for emp in self.employees if int(emp.id.split("-")[1]) % 3 == self.business_owners.index(business) % 3]
            for emp in staffed:
                shift_note = emp.perform_shift(getattr(business, "business_type", "business"))
                self.event_recorder.record_event(emp.id, emp.role, "shift", shift_note, target_id=business.id)

        # Mules distribution schedule
        mule_cfg = schedules.get("mule_payments", {})
        if weekday in mule_cfg.get("days", []):
            for mule in self.mules:
                payments = [mule_cfg.get("payment_amount", 2000)] * mule_cfg.get("payment_count", 5)
                mule.redistribute(payments)
                for amt in payments:
                    target = random.choice(self.residents + self.business_owners)
                    tx = self.event_recorder.record_transaction(
                        sender_account=mule.accounts[0],
                        receiver_account=target.accounts[0],
                        amount=amt,
                        currency=self.currencies,
                        tx_type="layering",
                        is_money_laundering=True,
                        ml_typology="mule_scatter",
                    )
                    self.event_recorder.record_event(mule.id, mule.role, "mule_transfer", f"Sent {amt} to {target.id}")
                    self._evaluate_risk(tx)

        # Integration stage for payroll and suppliers
        for owner in self.business_owners:
            payroll_amount = random.randint(5000, 15000)
            desc = owner.integrate_funds(payroll_amount)
            self.event_recorder.record_event(owner.id, owner.role, "integration", desc)
            tx = self.event_recorder.record_transaction(
                sender_account=owner.accounts[0],
                receiver_account=random.choice(self.residents).accounts[0],
                amount=payroll_amount,
                currency=self.currencies,
                tx_type="integration",
                is_money_laundering=False,
                ml_typology=None,
            )
            self._evaluate_risk(tx)

        # Ordinary resident salaries
        if current_date.day in schedules.get("salary_days", [1, 15]):
            for resident in self.residents:
                salary = random.randint(20000, 50000)
                resident.normal_activity(salary)
                tx = self.event_recorder.record_transaction(
                    sender_account=self.back_office.accounts[0],
                    receiver_account=resident.accounts[0],
                    amount=salary,
                    currency=self.currencies,
                    tx_type="salary",
                    is_money_laundering=False,
                    ml_typology=None,
                )
                self.event_recorder.record_event(self.back_office.id, self.back_office.role, "salary", f"Paid salary {salary} to {resident.id}")
                self._evaluate_risk(tx)

        # Deposits at bank tellers (can trigger SAR)
        depositors = self.business_owners + self.mules + self.residents
        for depositor in depositors:
            amount = random.randint(2000, 140000)
            teller = random.choice(self.bank_tellers)
            is_launder = isinstance(depositor, (BusinessOwnerAgent, MuleAgent, MastermindAgent))
            tx = self.event_recorder.record_transaction(
                sender_account=depositor.accounts[0],
                receiver_account=teller.accounts[0],
                amount=amount,
                currency=self.currencies,
                tx_type="deposit",
                is_money_laundering=is_launder,
                ml_typology="structuring" if is_launder and amount > 90000 else None,
            )
            desc = f"Deposit of {amount} by {depositor.id}"
            if amount >= self.risk_model.high_risk_amount:
                teller_note = teller.flag_deposit(amount)
                self.event_recorder.record_event(teller.id, teller.role, "sar", teller_note, target_id=depositor.id)
            self.event_recorder.record_event(depositor.id, depositor.role, "deposit", desc, target_id=teller.id)
            self._evaluate_risk(tx)

        # Random consumer spending
        for resident in self.residents:
            spend = random.randint(500, 3000)
            resident.normal_activity(spend)
            tx = self.event_recorder.record_transaction(
                sender_account=resident.accounts[0],
                receiver_account=random.choice(self.business_owners).accounts[0],
                amount=spend,
                currency=self.currencies,
                tx_type="spend",
                is_money_laundering=False,
                ml_typology=None,
            )
            self.event_recorder.record_event(resident.id, resident.role, "spend", f"Spent {spend}")
            self._evaluate_risk(tx)

        # Peer-to-peer transfers and bill payments to public facilities
        public_accounts = ["city_hall", "tax_office", "hospital", "school"]
        for _ in range(len(self.residents)):
            sender = random.choice(self.residents)
            receiver = random.choice(self.residents)
            if sender.id == receiver.id:
                receiver = random.choice(self.residents)
            amount = random.randint(200, 5000)
            tx = self.event_recorder.record_transaction(
                sender_account=sender.accounts[0],
                receiver_account=receiver.accounts[0],
                amount=amount,
                currency=self.currencies,
                tx_type="p2p",
                is_money_laundering=False,
                ml_typology=None,
            )
            self.event_recorder.record_event(sender.id, sender.role, "p2p", f"Sent {amount} to {receiver.id}")
            self._evaluate_risk(tx)

        for _ in range(2):
            payer = random.choice(self.residents)
            facility = random.choice(public_accounts)
            amount = random.randint(800, 4000)
            tx = self.event_recorder.record_transaction(
                sender_account=payer.accounts[0],
                receiver_account=facility,
                amount=amount,
                currency=self.currencies,
                tx_type="bill",
                is_money_laundering=False,
                ml_typology=None,
            )
            self.event_recorder.record_event(payer.id, payer.role, "bill", f"Paid {amount} to {facility}")
            self._evaluate_risk(tx)

    def _evaluate_risk(self, tx) -> None:
        score = self.risk_model.score_transaction(tx.amount, tx.is_money_laundering)
        if score >= self.config.risk_threshold:
            regulator = random.choice(self.regulators)
            note = regulator.open_investigation(tx.tx_id, score)
            self.event_recorder.record_event(regulator.id, regulator.role, "sar", note, target_id=str(tx.tx_id))
    
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
    
    def _generate_fan_out_pattern(
        self,
        controller_account: str,
        recipient_accounts: List[str],
        total_amount: float,
        currency: str,
        scheme_id: str,
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
            tx = self.event_recorder.record_transaction(
                sender_account=controller_account,
                receiver_account=receiver,
                amount=amount,
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=FAN_OUT,
                ml_pattern=FAN_OUT,
                pattern_scheme_id=scheme_id,
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
    ) -> List[int]:
        """
        Generate a fan-in pattern where multiple sources consolidate into a sink.
        """

        if len(source_accounts) < 2:
            raise ValueError("Fan-in pattern requires at least two source accounts")

        allocations = self._split_amount(total_amount, len(source_accounts))
        tx_ids: List[int] = []
        for sender, amount in zip(source_accounts, allocations):
            tx = self.event_recorder.record_transaction(
                sender_account=sender,
                receiver_account=sink_account,
                amount=amount,
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=FAN_IN,
                ml_pattern=FAN_IN,
                pattern_scheme_id=scheme_id,
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
            tx = self.event_recorder.record_transaction(
                sender_account=sender,
                receiver_account=gather_account,
                amount=amount,
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=GATHER_SCATTER,
                ml_pattern=GATHER_SCATTER,
                pattern_scheme_id=scheme_id,
            )
            tx_ids.append(tx.tx_id)

        for receiver, amount in zip(recipient_accounts, outgoing_amounts):
            tx = self.event_recorder.record_transaction(
                sender_account=gather_account,
                receiver_account=receiver,
                amount=amount,
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=GATHER_SCATTER,
                ml_pattern=GATHER_SCATTER,
                pattern_scheme_id=scheme_id,
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
            tx = self.event_recorder.record_transaction(
                sender_account=source_account,
                receiver_account=receiver,
                amount=amount,
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=SCATTER_GATHER,
                ml_pattern=SCATTER_GATHER,
                pattern_scheme_id=scheme_id,
            )
            tx_ids.append(tx.tx_id)

        for sender, amount in zip(intermediate_accounts, gather_amounts):
            tx = self.event_recorder.record_transaction(
                sender_account=sender,
                receiver_account=sink_account,
                amount=amount,
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=SCATTER_GATHER,
                ml_pattern=SCATTER_GATHER,
                pattern_scheme_id=scheme_id,
            )
            tx_ids.append(tx.tx_id)
        return tx_ids

    def _generate_simple_cycle_pattern(
        self,
        accounts: List[str],
        total_amount: float,
        currency: str,
        scheme_id: str,
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
            tx = self.event_recorder.record_transaction(
                sender_account=sender,
                receiver_account=receiver,
                amount=allocations[idx],
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=SIMPLE_CYCLE,
                ml_pattern=SIMPLE_CYCLE,
                pattern_scheme_id=scheme_id,
            )
            tx_ids.append(tx.tx_id)
        return tx_ids

    def _generate_random_pattern(
        self,
        account_path: List[str],
        total_amount: float,
        currency: str,
        scheme_id: str,
    ) -> List[int]:
        """
        Generate a random-walk-style chain without returning to the origin.
        """

        if len(account_path) < 2:
            raise ValueError("Random pattern requires at least two accounts")

        allocations = self._split_amount(total_amount, len(account_path) - 1)
        tx_ids: List[int] = []
        for idx in range(len(account_path) - 1):
            tx = self.event_recorder.record_transaction(
                sender_account=account_path[idx],
                receiver_account=account_path[idx + 1],
                amount=allocations[idx],
                currency=currency,
                tx_type="layering",
                is_money_laundering=True,
                ml_typology=RANDOM,
                ml_pattern=RANDOM,
                pattern_scheme_id=scheme_id,
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
                tx = self.event_recorder.record_transaction(
                    sender_account=sender,
                    receiver_account=receiver,
                    amount=amount,
                    currency=currency,
                    tx_type="layering",
                    is_money_laundering=True,
                    ml_typology=BIPARTITE,
                    ml_pattern=BIPARTITE,
                    pattern_scheme_id=scheme_id,
                )
                tx_ids.append(tx.tx_id)
        return tx_ids

    def _generate_stack_pattern(
        self,
        layers: List[List[str]],
        total_amount: float,
        currency: str,
        scheme_id: str,
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
                    tx = self.event_recorder.record_transaction(
                        sender_account=sender,
                        receiver_account=receiver,
                        amount=amount,
                        currency=currency,
                        tx_type="layering",
                        is_money_laundering=True,
                        ml_typology=STACK,
                        ml_pattern=STACK,
                        pattern_scheme_id=scheme_id,
                    )
                    tx_ids.append(tx.tx_id)
        return tx_ids