from __future__ import annotations

import argparse
from pathlib import Path

from aml_sim.config import ensure_output_dir, load_config
from aml_sim import export
from aml_sim.simulation import Simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AML multi-agent simulation")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--output", type=str, default="outputs", help="Directory for CSV outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_output_dir(args.output)
    simulation = Simulation(config)
    simulation.run(output_dir)
    print(f"Simulation complete. Logs at {output_dir}")


if __name__ == "__main__":
    main()
