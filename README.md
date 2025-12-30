# Launder Simulation / 洗钱小镇仿真

## Overview / 项目简介
This project simulates a small town where every participant—from regulators and residents to banks—acts as an LLM-driven agent. Agents reason over personal memory, shared knowledge, and affective context to produce laundering behaviours, compliance actions, and everyday financial flows.

本项目模拟一个由多类主体组成的洗钱小镇：从监管者、居民到银行，都通过 LLM 推理驱动。代理会结合个人记忆、知识库与情绪状态，生成洗钱操作、合规响应和日常金融活动。

## Key features / 主要特性
- **LLM-based agents for everyone**: residents, laundering actors, bank staff, institutional banks, and regulators all share the same configurable reasoning engine.【F:aml_sim/agents.py†L20-L117】【F:aml_sim/simulation.py†L86-L133】
- **Configurable LLM client**: API key, base URL, and model are set in `config/default.yaml` and passed to every agent; calls follow a single `client.chat.completions.create` pattern.【F:aml_sim/agents.py†L44-L83】【F:config/default.yaml†L28-L34】
- **Rich simulation loop**: scheduled events, laundering pattern generation, social interactions, reflections, and monitoring with risk scoring and SAR triggers.【F:aml_sim/simulation.py†L280-L496】【F:aml_sim/simulation.py†L612-L770】
- **Knowledge-grounded reasoning**: DPR embeddings prime agent memory and knowledge retrieval for more realistic decisions.【F:aml_sim/simulation.py†L58-L80】
- **Comprehensive exports**: behaviour logs, transactions, account/entity snapshots, and graph features written to CSV outputs.【F:main.py†L16-L27】【F:aml_sim/export.py†L6-L94】

## Configuration / 配置
Edit `config/default.yaml` (JSON-compatible) to control simulation parameters. Important fields include:
- `llm.api_key`, `llm.base_url`, `llm.model`: LLM client settings for every agent.【F:config/default.yaml†L28-L34】
- `simulation_days`, `population_scale`, `transaction_scale`, `laundering_intensity`: overall scenario sizing.【F:config/default.yaml†L3-L11】
- Role counts, schedules, and pattern generation blocks for laundering and routine activity.【F:config/default.yaml†L12-L26】【F:config/default.yaml†L35-L70】
- Knowledge documents and DPR encoder settings in `knowledge` for retrieval-augmented prompts.【F:config/default.yaml†L71-L101】

在 `config/default.yaml`（可兼容 JSON 语法）中调整仿真参数，重点字段：
- `llm.api_key`、`llm.base_url`、`llm.model`：统一的 LLM 客户端配置，所有代理共用。【F:config/default.yaml†L28-L34】
- `simulation_days`、`population_scale`、`transaction_scale`、`laundering_intensity`：控制场景规模。【F:config/default.yaml†L3-L11】
- 角色数量、时间表及洗钱模式生成配置。【F:config/default.yaml†L12-L26】【F:config/default.yaml†L35-L70】
- `knowledge` 区块里的文档与 DPR 编码器，用于增强检索式提示。【F:config/default.yaml†L71-L101】

## Usage / 使用说明
1. Install dependencies (PyTorch + Transformers for DPR; OpenAI SDK for live LLM calls).
2. Set `llm.api_key`/`llm.base_url`/`llm.model` in `config/default.yaml` as needed.
3. Run the simulation:
   ```bash
   python main.py --config config/default.yaml --output outputs
   ```
4. Review outputs (`behaviour_log.csv`, `transactions.csv`, `accounts.csv`, `entities.csv`, `graph_edges.csv`, `graph_features.csv`) in the chosen directory.【F:main.py†L16-L27】

1. 安装依赖（DPR 需要 PyTorch 与 Transformers；在线推理需 OpenAI SDK）。
2. 在 `config/default.yaml` 中填写 `llm.api_key`、`llm.base_url`、`llm.model`。
3. 运行模拟：
   ```bash
   python main.py --config config/default.yaml --output outputs
   ```
4. 在输出目录查看 `behaviour_log.csv`、`transactions.csv`、`accounts.csv`、`entities.csv`、`graph_edges.csv`、`graph_features.csv` 等结果。【F:main.py†L16-L27】

## Notes / 说明
- If the LLM client is unavailable, agents fall back to deterministic reasoning blocks while keeping the same interface.【F:aml_sim/agents.py†L61-L117】
- Bank agents are instantiated for each configured bank so institutional behaviour also uses the LLM pipeline.【F:aml_sim/simulation.py†L114-L133】
- The configuration file is JSON-compatible, so it can be edited as YAML or JSON without changing the loader.【F:aml_sim/config.py†L6-L38】

若 LLM 客户端不可用，代理会使用确定性逻辑作为后备，但接口保持一致。【F:aml_sim/agents.py†L61-L117】
每家配置的银行都会生成对应的银行代理，使机构层面也使用 LLM 推理。【F:aml_sim/simulation.py†L114-L133】
配置文件兼容 JSON 与 YAML，可按任意格式编辑而无需修改加载逻辑。【F:aml_sim/config.py†L6-L38】
