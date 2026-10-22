# SentientAI: Autonomous Cognitive Runtime

SentientAI is a local, persistent, autonomous AI runtime designed to function as a continuous digital cognitive system. Moving beyond stateless chat interfaces, this project builds a long-running agent equipped with multi-tiered memory, reasoning, tool use, environmental interaction, and the capacity for recursive self-improvement.

Built entirely for local hardware, SentientAI aims to create the closest practical version of a persistent AI "brain" using open-weight models, massive local storage, and structured agentic workflows—operating safely, continuously, and observably.

The mind that runs on SentientAI is named **Fuzzy**. The name lives in `identity.name` in the config; Fuzzy introduces itself by it, carries it across reboots, and uses it in its self-model and dashboard.

> **Status:** the full runtime described below is implemented: all seven phases, in a typed
> Python package with strict typing, 140+ tests (unit and integration against real PostgreSQL +
> pgvector, Redis and Docker), a scenario-based eval suite, Docker/Compose/systemd deployment and
> CI. What remains is running it on your GPUs: see [Quick start](#quick-start) and
> [docs/hardware.md](docs/hardware.md).

---

## Quick Start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[all]"   # Python 3.11+
source .venv/bin/activate
sentientai init -o sentient.yaml        # commented config: models, storage tiers, safety, autonomy
sentientai -c sentient.yaml doctor      # checks DB, memory, embeddings, both model tiers, sandbox, audit
sentientai -c sentient.yaml run         # always-on daemon + API + dashboard at http://127.0.0.1:8765/
sentientai chat                         # talk to it (in another terminal)
```

With no edits, the defaults run fully embedded (SQLite, on-disk vector index, in-process working
memory, offline hashing embeddings). Point `models.*` at your llama.cpp / vLLM / Ollama servers for
real cognition, and `storage.*` at PostgreSQL + pgvector + Redis for production. The full stack,
including the model servers, is in [`deploy/docker-compose.yml`](deploy/docker-compose.yml).

| Guide | What's inside |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Layers, the runtime loop, a task end to end, persistence, extension points |
| [docs/cognition.md](docs/cognition.md) | Self-model, affect and drives, global workspace, metacognition, memory dynamics, inner life |
| [docs/safety.md](docs/safety.md) | Risk policy, approvals, loop breakers, kill switch (incl. hardware button), sandbox, audit chain |
| [docs/hardware.md](docs/hardware.md) | V340 + 1660 Ti serving plan (verified specs), llama.cpp builds, model sizing, storage tiers |
| [docs/operations.md](docs/operations.md) | Install, CLI cheat sheet, monitoring, backup/restore, upgrades, incident playbook |

---

## Key Features

### 🧠 Multi-Tiered Memory Architecture

* **Short-Term Working Memory:** Fast, low-latency state tracking using Redis to handle active context and multi-turn conversations.
* **Durable Long-Term Storage:** Structured factual memory stored in PostgreSQL to retain user preferences, system configurations, and historical context.
* **Semantic Vector Recall:** Dense vector search using Qdrant/pgvector to instantly retrieve relevant notes, documentation, and code snippets based on semantic meaning.
* **Episodic & Procedural Memory:** Chronological logs of past experiences paired with reusable strategies for successful tool workflows.
* **Memory Consolidation & Decay:** Automated background processes that summarize long conversations, extract high-value facts, and decay or prune outdated vector embeddings.

### ⚡ Model Orchestration & Tiered Routing

* **Dual-Model Hardware Allocation:** Routes simple background tasks, memory summarization, and embeddings to a dedicated secondary GPU (e.g., GTX 1660 Ti), preserving primary GPUs (e.g., Dual V340s) for heavy reasoning.
* **Dynamic Model Selection:** Automatically uses smaller 8B–14B quantized models for rapid tool routing and active state updates, switching to 32B–70B+ open-weight models for deep reasoning and code generation.
* **Context Compression:** Keeps active prompts clean by dynamically summarizing older turns in long-running tasks.

### 🛠️ Environment & Tool Interaction

* **Sandboxed Code Execution:** Safely tests and executes Python, Shell, and system commands inside isolated Docker containers.
* **Codebase & Repository Inspection:** Analyzes directory structures, parses files, and manages source code across large repositories.
* **Continuous Knowledge Ingestion:** Automated pipeline to ingest, chunk, embed, and store PDFs, web pages, research papers, and technical books.
* **Web & API Integration:** Performs autonomous web searches, scrapes pages, and interacts with RESTful APIs.

### 🔄 Persistent Runtime & Self-Evolution

* **Always-On Execution Loop:** Operates as a background service that persists across session restarts and system reboots.
* **Recursive Self-Improvement:** Analyzes its own execution logs, identifies performance bottlenecks, and drafts code optimizations to refine its own source code under human supervision.
* **Loop-Breaking Mechanisms:** Hard deterministic limits to prevent execution spirals, infinite retries, or excessive resource consumption during autonomous runs.

### 🧩 Cognitive Core

* **Persistent Self-Model:** Identity, values, capabilities, known limitations and an autobiographical narrative, plus temporal continuity ("awakening #N, offline for 9 hours, last working on…").
* **Affect & Homeostatic Drives:** Curiosity, competence, energy, social connection, coherence, valence and mood. They are appraised from events, relax over time, and *modulate behaviour* (model tier, caution and planning, sampling temperature, initiative).
* **Global Workspace Attention:** Operator requests, schedules, watchers, approvals, goals and drives compete for attention. The winners are broadcast and decide what happens next.
* **Metacognition:** Calibrated confidence on every answer. Low-confidence fast-tier answers are re-checked by the heavy model.
* **Inner Life & Intrinsic Goals:** When idle and curious, it reflects, records insights and adopts bounded self-generated goals, executed under stricter autonomous safety thresholds.

### 🛡️ Safety & Observability

* **Human-in-the-Loop Approval Gates:** Requires explicit operator consent before executing high-risk commands, such as system modifications or repository commits.
* **Immutable Audit Logging:** Formats and writes every action, reasoning step, and tool result to permanent, readable execution logs.
* **Emergency Pathways:** Includes software and hardware kill-switches to immediately stop autonomous execution.

---

## System Architecture

SentientAI operates on a six-layer modular architecture to strictly separate reasoning, storage, execution, and safety.

```
+-------------------------------------------------------------------+
|                        User / Environment                         |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                       Orchestration Layer                         |
|  - State Restore   - Task Planning   - Tool Routing               |
|  - Loop Breakers   - Context Framing - Output Parsers             |
+-------------------------------------------------------------------+
       |                          |                          |
       v                          v                          v
+--------------+          +---------------+          +--------------+
| Model Layer  |          | Memory Layer  |          | Tool Layer   |
| - Fast Routing|          | - Vector DB   |          | - Sandbox    |
| - Heavy Reason|         | - Postgres    |          | - File System|
| - Quantized  |          | - Redis       |          | - Web Search |
+--------------+          +---------------+          +--------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                   Reflection & Learning Layer                     |
|  - Fact Extraction   - Summarization   - Memory Decay             |
|  - Tool Evaluation   - Code Refactoring Proposals                 |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                     Safety & Governance Layer                     |
|  - Permission Gates  - Container Isolation  - Immutable Audit Logs |
+-------------------------------------------------------------------+

```

---

## Hardware and Storage Strategy

Designed to maximize local compute efficiency through strategic workload distribution.

**Compute Allocation:**

* **Primary Stack (2x Radeon Pro V340 = 4 Vega 10 GPUs x 16 GB = 64 GB):** Dedicated to the heavy reasoning model. The model's layers are split across all four GPUs with llama.cpp, which fits a 70B model at ~4-bit or a 32B model at 8-bit. *Correction:* current ROCm and vLLM do not support the V340's `gfx900` architecture, so vLLM tensor parallelism is not a viable path on these cards. llama.cpp's Vulkan backend is. Details and sources are in [docs/hardware.md](docs/hardware.md).
* **Supporting Stack (1x GTX 1660 Ti, 6 GB):** Fast routing, reflection, self-assessment, summarization and embeddings (CUDA), keeping background cognition off the primary GPUs.

**Storage Tiering:**

* **Fast NVMe SSDs:** Active working memory, Redis cache, and live vector indexes.
* **Relational DBs (SSD/HDD):** Structured logs, configurations, and episodic memory sequences.
* **Mass HDD Storage:** Archival data, raw document ingestion (books, PDFs), and massive knowledge graphs.

---

## The Continuous Runtime Loop

Unlike a stateless script, SentientAI operates on an infinite background loop:

1. **Boot & Restore:** Load state, configuration, the self-model and affective state; recover interrupted tasks.
2. **Monitor & Attend:** Perceive user input, schedules, file watchers, approvals, goals and drives; let them compete in the global workspace.
3. **Retrieve:** Pull relevant procedural and semantic context from databases.
4. **Plan:** Formulate an execution strategy and select required tools.
5. **Execute:** Run actions, capture standard output, and validate success.
6. **Reflect:** Summarize results, update long-term memory, and actively forget deprecated data.
7. **Loop:** Return to the monitoring state.

---

## Development Master Plan

### Phase 1: Local Foundation

* [x] Configure local GPU inference (vLLM, Ollama, or llama.cpp). *Any OpenAI-compatible server; llama.cpp configs for the V340/1660 Ti in `deploy/`.*
* [x] Establish a stable chat interface.
* [ ] Validate baseline inference speed and memory usage for 8B and 32B models. *Needs your GPUs: run `sentientai bench`.*

### Phase 2: Memory & Retrieval Engine

* [x] Deploy PostgreSQL and Vector DB (Qdrant/pgvector).
* [x] Build the data ingestion pipeline (chunking, embeddings, metadata tagging).
* [x] Enable semantic retrieval over stored conversations and documents.

### Phase 3: Tool-Use & Execution Sandbox

* [x] Implement robust file access and repository inspection tools.
* [x] Build the shell and code execution sandbox (Docker-based).
* [x] Develop multi-step task workflows and deterministic loop breakers.
* [x] Establish the automated evaluation suite for nightly regression testing.

### Phase 4: Reflection & Continuity

* [x] Implement structured logging of all decisions and tool actions.
* [x] Build the background memory summarization engine.
* [x] Enable long-term learning loops (procedural memory formulation).
* [x] Establish session continuity (restoring exact state after a reboot).

### Phase 5: Autonomous Workflows

* [x] Schedule background tasks and event-driven actions.
* [x] Add persistent task loops (e.g., monitoring a folder and categorizing new files).
* [x] Develop project-aware agent behavior (managing context across days or weeks).

### Phase 6: Optimization & Safety Refinement

* [x] Implement vector memory decay (forgetting outdated facts).
* [ ] Upgrade to larger reasoning models as local compute permits. *Ongoing; a config change (`models.heavy`).*
* [x] Refine reflection logic to minimize junk data ingestion.
* [x] Finalize the safety layer (approval gates, rate limits, kill-switches).

### Phase 7: Recursive Self-Improvement

* [x] Enable the agent to analyze its own architecture and execution logs.
* [x] Allow the agent to write, test, and propose optimizations to its own Python/Rust codebase. *Python; diffs to the safety layer are blocked.*
* [x] Implement automated staging environments for the AI to benchmark its own upgrades before operator approval.

---

## Technology Stack

* **Model Serving:** llama.cpp (Vulkan for the V340s, CUDA for the 1660 Ti); vLLM or Ollama on supported GPUs. Anything OpenAI-compatible works.
* **Inference Models:** Llama 3.x, Qwen 2.5/3, DeepSeek (Quantized GGUF)
* **Vector Database:** Qdrant or PostgreSQL + pgvector
* **Relational Database:** PostgreSQL
* **Short-Term Cache:** Redis
* **Orchestration:** Custom asyncio Python runtime. It was chosen over LangGraph so safety gates, loop breakers and the audit trail stay in a small, fully controlled code path.
* **Data Pipeline:** Python
* **Environment:** Docker (Sandboxing)
* **API & CLI:** FastAPI (+ SSE event stream, Prometheus metrics, operator dashboard), Typer

---

## Development

```bash
make install      # venv + editable install with dev tools
make lint typecheck test     # ruff, mypy --strict, pytest (+ integration tests when Postgres/Redis/Docker are reachable)
make evals        # scripted scenario suite (no GPUs needed)
```

Integration tests read `SENTIENT_TEST_POSTGRES_URL` (needs pgvector) and `SENTIENT_TEST_REDIS_URL`,
and skip cleanly when those services are absent.

---

## License

This project is developed as an open research and experimentation platform. Use, modification, and contribution are encouraged under the [MIT License](LICENSE).
