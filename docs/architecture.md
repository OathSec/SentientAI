# Architecture

SentientAI is a long-running asyncio daemon built from small, layered modules.
Every external dependency (models, databases, vector index, sandbox) sits behind a
narrow interface with an embedded fallback, so the same code runs on a laptop, in
CI and on the GPU server.

```
                         operator ── CLI / HTTP API / dashboard / SSE
                                          │
┌─────────────────────────────── runtime/ ┴────────────────────────────────────┐
│ Runtime: boot/restore → perceive → attend (global workspace) → act → reflect │
│ Scheduler (cron/interval) · FolderWatchers · checkpointing · signals          │
└──────┬──────────────────────┬──────────────────────┬──────────────────────────┘
       │                      │                      │
 orchestration/          cognition/             reflection/
 CognitiveAgent          Mind = self-model      Reflector (episodes, lessons,
 ContextBuilder          + affect/drives        facts, procedures, narrative)
 ContextCompressor       + global workspace     PerformanceAnalyzer
 Planner                 + goals, metacognition SelfImprovementEngine
       │                 + inner life
       ├───────────────┬──────────────────┬──────────────────────┐
   models/          memory/            tools/                safety/
   ModelRouter      MemorySystem       ToolExecutor ◀── PolicyEngine, ApprovalGate,
   (fast/heavy,     working (Redis)    filesystem, repo,      KillSwitch, LoopGuard,
   circuit          facts/episodes/    sandbox (Docker),      TokenBucket, AuditLog
   breakers)        procedures (SQL)   web, memory, goals
   embeddings       vectors (pgvector/ ingestion/ pipeline
                    Qdrant/numpy)
                          │
                      storage/ Database (PostgreSQL | SQLite) + repositories
```

## Module map

| Layer (README) | Package | Key types |
|---|---|---|
| Orchestration | `orchestration/`, `runtime/` | `CognitiveAgent`, `ContextBuilder`, `ContextCompressor`, `Planner`, `Runtime`, `Scheduler`, `FolderWatcher` |
| Model | `models/` | `ModelRouter`, `OpenAICompatibleClient`, `ScriptedLLM`, `HashingEmbedder`, `OpenAIEmbedder` |
| Memory | `memory/`, `storage/` | `MemorySystem`, `MemoryConsolidator`, `RedisWorkingMemory`, `PgVectorStore`, `QdrantVectorStore`, `Database` |
| Tools | `tools/`, `ingestion/` | `ToolExecutor`, `Workspace`, `DockerSandbox`, `WebClient`, `IngestionPipeline` |
| Reflection & learning | `reflection/`, `cognition/` | `Reflector`, `PerformanceAnalyzer`, `SelfImprovementEngine`, `Mind`, `InnerLife` |
| Safety & governance | `safety/` | `PolicyEngine`, `ApprovalGate`, `KillSwitch`, `LoopGuard`, `AuditLog` |

## The runtime loop

Each tick (`runtime.tick_interval_s`):

1. **Relax** the affective state toward its set points (time-based homeostasis).
2. **Perceive.** Submit percepts to the global workspace for queued tasks (salience
   by source and priority), pending approvals, active goals and drives (proactive
   mode only), and due maintenance.
3. **Attend.** Run the salience competition. Winners become the current contents of
   awareness, are broadcast on the event bus, and are rendered into prompts.
4. **Act** on the top actionable percept: run a task, pursue a goal, think, or
   consolidate memory. Concurrency is bounded by `runtime.max_concurrent_tasks`.
5. After every task, **reflect** and update affect, the self-model, procedures and
   goals.

Interrupted tasks are re-queued at the next boot (with a retry budget). Approvals
from a dead process expire. The mind is checkpointed every
`runtime.checkpoint_interval_s` and at shutdown.

## One task, end to end

1. `memory.recall(goal)` runs hybrid search (dense + keyword) and re-ranks by
   similarity, recency, importance and strength. `find_procedures(goal)` looks for
   learned skills.
2. Novelty (nothing similar in memory) raises curiosity. Low competence or coherence
   makes the agent cautious, which turns on planning.
3. `ContextBuilder` frames identity, continuity, inner state, awareness, goals,
   rules, memories, procedures, the plan and conversation history, all within a
   token budget.
4. The model/tool loop runs under `LoopGuard`. Each tool call goes through
   `ToolExecutor`: validation, policy, approval, rate limit, timeout, truncation,
   invocation record and audit. `ContextCompressor` keeps long loops inside the
   context window without orphaning tool results.
5. `Metacognition` blends a model self-rating with evidence signals. A low-confidence
   fast-tier answer is re-checked on the heavy tier.
6. `Reflector` stores the episode, durable facts and (for successful multi-step
   work) a reusable procedure, and updates the reliability of procedures it used.

## Persistence

| Data | Store | Notes |
|---|---|---|
| Memories (facts, episodes, procedures, documents, summaries, reflections, thoughts) | `memories` table | Source of truth; vectors are a rebuildable index (`memory reindex`) |
| Vectors | pgvector (HNSW, cosine) / Qdrant / `vectors.npz` | Keyed by memory id |
| Working memory | Redis lists with TTL / in-process | Recent turns + rolling summary |
| Tasks, approvals, goals, proposals, documents, tool invocations, conversations | SQL tables | Durable queue and history |
| Mind checkpoint (self-model, affect), scheduler and watcher state | `kv_state` | JSON |
| Audit log | `data/audit/audit-NNNNNN.jsonl` | Hash/HMAC chained, file-locked |
| Raw documents | `archive_dir/<sha[:2]>/<sha>.<ext>` | Content-addressed |

The schema is created and version-checked at startup (`schema_meta`). SQLite runs
in WAL mode for concurrent readers.

## Extending

- **New tool.** Subclass `Tool[Args]`, declare `name`, `description`, `args_model`
  and a base `risk`, and override `assess()` for argument-aware risk. Register it in
  `runtime/container.py` or pass `extra_tools`.
- **New model backend.** Implement the `LLMClient` protocol (`chat`, `health`, `aclose`).
- **New vector store.** Implement the `VectorStore` protocol and add it to
  `memory/factory.py`.
- **New eval.** Add a scenario to `evals/scenarios/*.yaml`.
