from __future__ import annotations

import dataclasses
from typing import List, Union


@dataclasses.dataclass
class Person:
    id: str
    name: str
    accounts: List[str]


@dataclasses.dataclass
class Company:
    id: str
    name: str
    accounts: List[str]
    owners: List[Union[Person, "Company"]]
    suppliers: List["Company"]
    customers: List[Union[Person, "Company"]]
    is_shell: bool = False

    def add_supplier(self, supplier: "Company") -> None:
        if supplier not in self.suppliers:
            self.suppliers.append(supplier)

    def add_customer(self, customer: Union[Person, "Company"]) -> None:
        if customer not in self.customers:
            self.customers.append(customer)