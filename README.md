# Launder Simulation / 洗钱小镇仿真

A lightweight multi-agent playground that simulates laundering behaviour, everyday finance, and compliance responses in a small town. 每个主体（居民、洗钱者、银行、监管者）都由同一套 LLM 推理引擎驱动，可在无外部 API 的离线环境下运行。

## Why this project / 项目特点
- **LLM reasoning with safe defaults / 统一的 LLM 推理**: All agents share the same client configuration (model, base URL, API key, `max_tokens`). The default cap is **1024 tokens** to avoid runaway generations. When no API key is provided, deterministic fallbacks are used to keep the simulation moving.【F:aml_sim/agents.py†L18-L88】【F:config/default.yaml†L28-L35】
- **Rich daily loop / 丰富的日常循环**: Placement, layering, integration, payroll, deposits, social interactions, and regulator monitoring all run each simulated day.【F:aml_sim/simulation.py†L702-L959】
- **Streaming CSV exports / 运行中持续导出**: Behaviour logs, transactions, accounts, entities, and graph features are written to CSV **after every simulated day**, so partial results remain available even if a long run stops early.【F:aml_sim/simulation.py†L710-L725】
- **Graph-friendly outputs / 适配图挖掘的导出**: Edge lists and per-account graph features are generated for downstream analytics.【F:aml_sim/export.py†L1-L187】

## Quickstart / 快速开始
1. Install dependencies (PyTorch + Transformers for DPR; OpenAI SDK for live LLM calls).
2. (Optional) Set `OPENAI_API_KEY`/`OPENAI_BASE_URL` env vars, or edit `config/default.yaml` under `llm` to point to your endpoint and adjust `max_tokens` (default 1024).【F:config/default.yaml†L28-L35】
3. Run the simulation:
   ```bash
   python main.py --config config/default.yaml --output outputs
   ```
4. Check the `outputs/` directory for continuously updated CSV files: `behaviour_log.csv`, `transactions.csv`, `accounts.csv`, `entities.csv`, `graph_edges.csv`, `graph_features.csv`.

## Configuration highlights / 关键配置
- **Scale & intensity / 规模与强度**: `simulation_days`, `population_scale`, `transaction_scale`, `laundering_intensity`.【F:config/default.yaml†L3-L23】
- **LLM client / LLM 配置**: `llm.model`, `llm.base_url`, `llm.api_key`, `llm.max_tokens` (defaults to 1024).【F:config/default.yaml†L28-L35】
- **Schedules / 时间表**: Daily transfer schedules, business intakes, mule payouts, and salary days under `daily_schedules`.【F:config/default.yaml†L36-L70】
- **Knowledge / 知识库**: DPR model path and seed documents for retrieval-augmented prompts under `knowledge`.【F:config/default.yaml†L71-L101】

## Outputs / 结果
- `behaviour_log.csv`: Agent events and narrative logs.
- `transactions.csv`: Transaction-level details with AML annotations.
- `accounts.csv`, `entities.csv`: Snapshot of balances and ownership structures.
- `graph_edges.csv`, `graph_features.csv`: Graph-ready exports for downstream analysis.

## Tips / 使用建议
- Keep `llm.max_tokens` conservative (e.g., 1024) to reduce token spend and prevent overlong generations.【F:aml_sim/agents.py†L18-L88】
- Because exports run after each simulated day, you can interrupt a run and still keep all CSVs generated so far.【F:aml_sim/simulation.py†L710-L725】
