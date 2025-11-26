from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclasses.dataclass
class SimulationConfig:
    seed: int
    simulation_days: int
    risk_threshold: float
    num_mules: int
    num_residents: int
    num_employees: int
    num_regulators: int
    role_descriptions: Dict[str, str]
    daily_schedules: Dict[str, Any]
    knowledge_documents: List[Dict[str, str]]
    knowledge_model_path: str
    knowledge_device: str | None
    risk_model: Dict[str, Any]


def _load_raw_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if yaml:
        return yaml.safe_load(text)
    return json.loads(text)


def load_config(path: str | Path) -> SimulationConfig:
    raw = _load_raw_config(path)
    knowledge_cfg = raw.get("knowledge", {})
    return SimulationConfig(
        seed=raw.get("seed", 0),
        simulation_days=raw.get("simulation_days", 1),
        risk_threshold=float(raw.get("risk_threshold", 0.8)),
        num_mules=raw.get("num_mules", 0),
        num_residents=raw.get("num_residents", 0),
        num_employees=raw.get("num_employees", 0),
        num_regulators=raw.get("num_regulators", 0),
        role_descriptions=raw.get("role_descriptions", {}),
        daily_schedules=raw.get("daily_schedules", {}),
        knowledge_documents=knowledge_cfg.get("documents", []),
        knowledge_model_path=knowledge_cfg.get("model_path", "facebook/dpr-ctx_encoder-single-nq-base"),
        knowledge_device=knowledge_cfg.get("device"),
        risk_model=raw.get("risk_model", {}),
    )


def ensure_output_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path