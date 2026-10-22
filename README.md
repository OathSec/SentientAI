Here is the updated, high-quality **README.md** with a dedicated **Key Features** section added, cleanly integrated into the complete document structure.

---

# SentientAI: Autonomous Cognitive Runtime

SentientAI is a local, persistent, autonomous AI runtime designed to function as a continuous digital cognitive system. Moving beyond stateless chat interfaces, this project builds a long-running agent equipped with multi-tiered memory, reasoning, tool use, environmental interaction, and the capacity for recursive self-improvement.

Built entirely for local hardware, SentientAI aims to create the closest practical version of a persistent AI "brain" using open-weight models, massive local storage, and structured agentic workflows—operating safely, continuously, and observably.

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

* **Primary Stack (e.g., 2x V340 GPUs):** Pooled via tensor parallelism exclusively for the heavy reasoning model.
* **Supporting Stack (e.g., 1x 1660 Ti):** Dedicated to fast embedding generation, background summarization, and small-model tool routing to prevent primary inference bottlenecks.

**Storage Tiering:**

* **Fast NVMe SSDs:** Active working memory, Redis cache, and live vector indexes.
* **Relational DBs (SSD/HDD):** Structured logs, configurations, and episodic memory sequences.
* **Mass HDD Storage:** Archival data, raw document ingestion (books, PDFs), and massive knowledge graphs.

---

## The Continuous Runtime Loop

Unlike a stateless script, SentientAI operates on an infinite background loop:

1. **Boot & Restore:** Load state, configuration, and recent episodic memory.
2. **Monitor:** Wait for user input or scheduled background triggers.
3. **Retrieve:** Pull relevant procedural and semantic context from databases.
4. **Plan:** Formulate an execution strategy and select required tools.
5. **Execute:** Run actions, capture standard output, and validate success.
6. **Reflect:** Summarize results, update long-term memory, and actively forget deprecated data.
7. **Loop:** Return to the monitoring state.

---

## Development Master Plan

### Phase 1: Local Foundation

* [ ] Configure local GPU inference (vLLM, Ollama, or llama.cpp).
* [ ] Establish a stable chat interface.
* [ ] Validate baseline inference speed and memory usage for 8B and 32B models.

### Phase 2: Memory & Retrieval Engine

* [ ] Deploy PostgreSQL and Vector DB (Qdrant/pgvector).
* [ ] Build the data ingestion pipeline (chunking, embeddings, metadata tagging).
* [ ] Enable semantic retrieval over stored conversations and documents.

### Phase 3: Tool-Use & Execution Sandbox

* [ ] Implement robust file access and repository inspection tools.
* [ ] Build the shell and code execution sandbox (Docker-based).
* [ ] Develop multi-step task workflows and deterministic loop breakers.
* [ ] Establish the automated evaluation suite for nightly regression testing.

### Phase 4: Reflection & Continuity

* [ ] Implement structured logging of all decisions and tool actions.
* [ ] Build the background memory summarization engine.
* [ ] Enable long-term learning loops (procedural memory formulation).
* [ ] Establish session continuity (restoring exact state after a reboot).

### Phase 5: Autonomous Workflows

* [ ] Schedule background tasks and event-driven actions.
* [ ] Add persistent task loops (e.g., monitoring a folder and categorizing new files).
* [ ] Develop project-aware agent behavior (managing context across days or weeks).

### Phase 6: Optimization & Safety Refinement

* [ ] Implement vector memory decay (forgetting outdated facts).
* [ ] Upgrade to larger reasoning models as local compute permits.
* [ ] Refine reflection logic to minimize junk data ingestion.
* [ ] Finalize the safety layer (approval gates, rate limits, kill-switches).

### Phase 7: Recursive Self-Improvement

* [ ] Enable the agent to analyze its own architecture and execution logs.
* [ ] Allow the agent to write, test, and propose optimizations to its own Python/Rust codebase.
* [ ] Implement automated staging environments for the AI to benchmark its own upgrades before operator approval.

---

## Technology Stack

* **Model Serving:** vLLM, llama.cpp
* **Inference Models:** Llama 3.1, Qwen 2.5, DeepSeek (Quantized)
* **Vector Database:** Qdrant or PostgreSQL + pgvector
* **Relational Database:** PostgreSQL
* **Short-Term Cache:** Redis
* **Orchestration:** LangGraph or custom Python runtime
* **Data Pipeline:** Python
* **Environment:** Docker (Sandboxing)

---

## License

This project is developed as an open research and experimentation platform. Use, modification, and contribution are encouraged under the [MIT License](https://www.google.com/search?q=LICENSE).
