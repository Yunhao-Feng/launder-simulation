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


ACCOUNTING_TYPOLOGIES = [
    "fan-out",
    "scatter-gather",
    "cycle",
    "peel chain",
]


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