# launder-simulation

A prototype multi-agent AML simulation driven by GPT-4o prompts, retrieval-augmented knowledge, and configurable schedules.

## Features
- Modular agent classes for masterminds, business owners, accountants, mules, residents, bank staff, and regulators.
- YAML/JSON-driven configuration for role counts, schedules, typology amounts, and risk thresholds.
- Lightweight knowledge base with deterministic embeddings and cosine retrieval to prime GPT-4o prompts.
- Event recorder producing behaviour logs and labelled transactions CSVs, with AML risk scoring and regulator SAR creation.

## Requirements
- Python 3.11+
- Optional: `pyyaml` if you prefer YAML parsing (the default config is also valid JSON and works without it).

## Running the simulation
```bash
python main.py --config config/default.yaml --output outputs
```
This produces `behaviour_log.csv` and `transactions.csv` inside the chosen output directory.

## Configuration
Edit `config/default.yaml` to adjust:
- `simulation_days` and random `seed`
- Role counts (mules, residents, regulators) and role descriptions
- Daily schedules for mastermind parcels, business intakes, mule scatter payments, and payroll days
- AML risk threshold and model scaling factors
- Knowledge documents to be embedded for retrieval