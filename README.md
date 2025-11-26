# launder-simulation

A prototype multi-agent AML simulation driven by GPT-4o prompts, retrieval-augmented knowledge, and configurable schedules.

## Features
- Modular agent classes for masterminds, business owners, accountants, mules, residents, bank staff, and regulators.
- YAML/JSON-driven configuration for role counts, schedules, typology amounts, and risk thresholds.
- DPR-powered knowledge base embeddings with cosine retrieval to prime GPT-4o prompts.
- Commonsense AML memory plus per-agent experience memory with embedding-based retrieval to ground decisions.
- Event recorder producing behaviour logs and labelled transactions CSVs, with AML risk scoring and regulator SAR creation.

## Requirements
- Python 3.11+
- Optional: `pyyaml` if you prefer YAML parsing (the default config is also valid JSON and works without it).
- `torch` and `transformers` for DPR embeddings; set `knowledge.device` to `cuda` to use GPUs when available.
- Optional: `openai` with `OPENAI_API_KEY`/`OPENAI_API_BASE` for live GPT-4o reasoning; otherwise a deterministic rule-based fallback is used.

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
- Knowledge documents to be embedded for retrieval, plus `knowledge.model_path`/`knowledge.device` for DPR settings

## How the simulation runs
1. **Load configuration**: Parse YAML/JSON to get simulation days, role counts, money parcel sizes, typology probabilities, and knowledge/DPR settings.
2. **Prepare knowledge**: Embed configured knowledge documents with the DPR context encoder and store vectors for cosine-similarity retrieval.
3. **Instantiate agents and accounts**: Create all role-specific agents, attach bank accounts, seed their memories with role briefs and initial knowledge, and share the DPR embedder.
4. **Run daily loop**:
   - Execute scheduled events (mastermind parcel drops, business intakes, mule scatter/gather, resident payroll, teller KYC, regulator scans).
   - For each action, retrieve similar knowledge/memory via DPR + cosine similarity, craft the GPT-4o prompt (or deterministic fallback), and decide the next action.
   - Record behaviour events and any resulting transactions; score transactions with the AML risk model and trigger regulator SAR events when thresholds are exceeded.
5. **Export outputs**: At the end of the run, write `behaviour_log.csv` and `transactions.csv` to the output directory.

### Flow outline
```
Config → DPR embed knowledge → Init agents/accounts →
For each day:
  schedule events → retrieve (DPR + cosine) → GPT-4o reasoning →
  record logs/transactions → AML risk scoring/SARs
→ Export CSVs
```