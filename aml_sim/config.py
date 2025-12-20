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
    num_shell_companies: int
    num_real_companies: int
    ownership_graph_depth: int
    supplier_graph_degree: int
    scenario: str
    population_scale: float
    transaction_scale: float
    laundering_intensity: float
    banks: List[Dict[str, Any]]
    fx_rates: Dict[str, float]
    role_descriptions: Dict[str, str]
    daily_schedules: Dict[str, Any]
    knowledge_documents: List[Dict[str, str]]
    knowledge_model_path: str
    knowledge_device: str | None
    risk_model: Dict[str, Any]
    default_currency: str
    pattern_generation: Dict[str, Any]
    intensity_presets: Dict[str, Dict[str, float]]
    enable_reflection: bool
    enable_social: bool
    social_interactions_per_day: int


def _load_raw_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if yaml:
        return yaml.safe_load(text)
    return json.loads(text)


def load_config(path: str | Path) -> SimulationConfig:
    raw = _load_raw_config(path)
    knowledge_cfg = raw.get("knowledge", {})
    scenario = raw.get("scenario", "LI")
    default_presets = {
        "HI": {"population_scale": 3.0, "transaction_scale": 4.0, "laundering_intensity": 0.08},
        "LI": {"population_scale": 0.6, "transaction_scale": 0.8, "laundering_intensity": 0.001},
        "PROTO": {"population_scale": 0.2, "transaction_scale": 0.3, "laundering_intensity": 0.0005},
    }
    custom_presets = raw.get("intensity_presets", {})
    presets = {**default_presets, **custom_presets}
    preset_values = presets.get(scenario.upper(), {})
    population_scale = float(raw.get("population_scale", preset_values.get("population_scale", 1.0)))
    transaction_scale = float(raw.get("transaction_scale", preset_values.get("transaction_scale", 1.0)))
    laundering_intensity = float(raw.get("laundering_intensity", preset_values.get("laundering_intensity", 0.01)))
    return SimulationConfig(
        seed=raw.get("seed", 0),
        simulation_days=raw.get("simulation_days", 1),
        risk_threshold=float(raw.get("risk_threshold", 0.8)),
        num_mules=raw.get("num_mules", 0),
        num_residents=raw.get("num_residents", 0),
        num_employees=raw.get("num_employees", 0),
        num_regulators=raw.get("num_regulators", 0),
        num_shell_companies=raw.get("num_shell_companies", 0),
        num_real_companies=raw.get("num_real_companies", 0),
        ownership_graph_depth=raw.get("ownership_graph_depth", 1),
        supplier_graph_degree=raw.get("supplier_graph_degree", 1),
        scenario=scenario,
        population_scale=population_scale,
        transaction_scale=transaction_scale,
        laundering_intensity=laundering_intensity,
        banks=raw.get("banks", []),
        fx_rates=raw.get("fx_rates", {}),
        role_descriptions=raw.get("role_descriptions", {}),
        daily_schedules=raw.get("daily_schedules", {}),
        knowledge_documents=knowledge_cfg.get("documents", []),
        knowledge_model_path=knowledge_cfg.get("model_path", "facebook/dpr-ctx_encoder-single-nq-base"),
        knowledge_device=knowledge_cfg.get("device"),
        risk_model=raw.get("risk_model", {}),
        default_currency=raw.get("default_currency", "CNY"),
        pattern_generation=raw.get("pattern_generation", {"enabled": True, "per_day": 1, "base_amount": 120000}),
        intensity_presets=presets,
        enable_reflection=bool(raw.get("enable_reflection", True)),
        enable_social=bool(raw.get("enable_social", True)),
        social_interactions_per_day=int(raw.get("social_interactions_per_day", 10)),
    )


def ensure_output_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
