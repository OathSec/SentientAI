# SentientAI

Project repository: https://github.com/OathSec/SentientAI

A local, persistent, autonomous AI project designed to function as a digital cognitive system: a long-running agent with memory, reasoning, tool use, and environmental interaction. The goal is not simply to make a chat bot, but to build the closest practical version of a persistent AI "brain" using local hardware, massive storage, memory retrieval, and agentic workflows.

This project is intended to evolve from a strong local base model into a self-improving, context-aware AI assistant that can code, reason, research, plan, and retain knowledge over time.

## Mission

Build a system that behaves like an always-on cognitive agent:

- constantly running in the background
- keeping long-term memory
- retrieving relevant information from large knowledge stores
- planning and executing tasks
- using tools to interact with files, code, search, APIs, and the environment
- learning from experience through reflection and memory consolidation
- remaining controllable, observable, and safe

## Important principle

This is not a claim that the project will magically become conscious. The practical, realistic objective is:

- persistent AI runtime
- memory-driven reasoning
- retrieval over large knowledge corpora
- autonomous task execution
- self-reflection and memory updates
- strong local inference using open-weight models

This is the closest practical version of a "sentient-like" AI system.

## Why this project exists

Most AI projects stop at a chat interface. That is useful, but not enough for a true cognitive system.

To create something closer to a real digital mind, the system needs:

- a base model
- long-term memory
- semantic retrieval
- short-term working memory
- task planning and execution
- tool use
- persistent state across restarts
- self-review and memory updates
- environmental awareness and interaction
- safe operational boundaries

## Core concept

SentientAI is built as a layered agent architecture:

1. Base model layer
   - strong open-weight model for reasoning and generation
   - quantized models for memory efficiency
   - smaller specialist model for quick tasks
   - larger model for deep reasoning

2. Memory layer
   - short-term working memory for current task context
   - long-term memory for facts, learned experiences, and preferences
   - semantic memory via vector search for knowledge retrieval
   - episodic memory for events and actions over time
   - procedural memory for learned workflows and patterns

3. Tool-use layer
   - file system operations
   - shell and command execution
   - repo and code analysis
   - web search
   - API and external integrations
   - task execution and automation

4. Orchestration layer
   - state tracking
   - planning
   - memory retrieval
   - tool selection
   - execution flow
   - result logging

5. Reflection and learning layer
   - summarization of recent actions
   - extraction of useful facts
   - memory updates
   - feedback loops for self-improvement
   - learning from outcomes

6. Safety layer
   - permission controls
   - sandboxing
   - audit logs
   - emergency stop paths
   - operator review for sensitive actions

## High-level system model

```text
User / Environment
        |
        v
   Input triggers
        |
        v
   Orchestration layer
   - state restore
   - memory retrieval
   - planning
   - tool routing
   - decision making
        |
        +------------------------+
        |                        |
        v                        v
   Model layer          Memory layer
   - base model        - vector DB
   - reasoning model   - Postgres
   - smaller model     - Redis
   - quantized model   - knowledge graph / summaries
        |
        v
   Tool-use layer
   - files
   - shell
   - web
   - repos
   - APIs
   - local apps
        |
        v
   Reflection and learning
   - record events
   - summarize experiences
   - extract facts
   - update memory
   - improve future planning
```

## Hardware strategy

### Planned hardware

- 3 GPUs total
- 2x V340 GPUs as part of the main inference stack
- 1x 1660 Ti as a supporting or main worker GPU for lighter tasks
- large mass storage for persistent memory and knowledge corpus

### Hardware usage strategy

- Use the strongest GPUs for the main model inference and heavier reasoning loads
- Use the 1660 Ti for supporting workloads, smaller models, background operations, embeddings, and orchestration tasks if needed
- Put the active memory, logs, retrieval corpus, and knowledge stores on NVMe SSDs or fast local disks
- Use larger storage arrays or HDDs for archival data and massive document stores

### Storage plan

- active system memory and working indexes on fast SSDs
- long-term archives on large storage
- structured memory in relational DB
- vector embeddings in a dedicated vector database
- snapshots and logs stored persistently for continuity

## Base model strategy

Use a strong open-weight model as the foundation instead of trying to train a giant model from scratch.

Recommended categories:

- Llama 3.x / 3.1
- Qwen 2.5
- Mistral
- DeepSeek family models

Recommended practical approach:

- Start with 8B or 14B for fast local testing and daily use
- Move into 32B or 70B class models when compute permits
- Use 4-bit or 8-bit quantization for memory efficiency
- Use model routing:
  - small model for rapid responses
  - medium model for code and reasoning work
  - large model for difficult or long-form analysis

This is far more realistic and efficient than attempting a full custom training run immediately.

## Memory architecture

The actual "brain" is not just the model. It is the memory system.

### 1. Short-term memory

Used for:

- active conversation
- current task state
- immediate goals
- realtime reasoning context

### 2. Long-term memory

Used for:

- learned facts
- repeated user preferences
- project history
- earlier conversations
- helpful summaries and discoveries

### 3. Semantic memory

Used for:

- searching large knowledge corpora
- retrieving similar documents, notes, code, and conversations
- context recall based on embedding similarity

### 4. Episodic memory

Used for:

- remembering what happened, when it happened, and why it mattered
- tracking project progress over time
- preserving continuity between sessions

### 5. Procedural memory

Used for:

- learned workflows
- repeated actions that work well
- common problem-solving patterns
- internal task strategies

## Knowledge ingestion pipeline

To build a large brain, the system should continuously ingest data:

- books and reference materials
- PDFs and docs
- web pages and research articles
- code repositories
- notes and project documentation
- user interactions and generated summaries
- logs and execution history

### Pipeline

1. collect raw content
2. parse and normalize data
3. split into chunks
4. generate embeddings
5. store in vector database
6. tag metadata and sources
7. retrieve context on demand
8. summarize important facts into durable memory

This allows the system to remain highly informed without relying solely on model context size.

## Tooling and environment interaction

A strong autonomous system needs tools.

### Planned tools

- shell command execution
- filesystem reading and writing
- code repository inspection
- code execution inside controlled environments
- browsing and web research
- API access
- document parsing and OCR
- summarization tools
- task and workflow automation
- event or log monitoring

The AI should not merely answer questions. It should perform tasks, reason about outcomes, and update memory.

## Always-on runtime design

The project is intended to run continuously, similar to a background digital agent.

### Runtime loop

1. boot the system
2. restore memory state
3. load identity and system configuration
4. wait for user input or task triggers
5. retrieve relevant memories
6. reason and plan
7. choose tools if needed
8. execute actions
9. log outcomes
10. summarize results
11. store key facts in long-term memory
12. continue to monitor and reason

This creates a persistent, stateful, cognitive runtime instead of a stateless chat interface.

## Self-learning and reflection

If the system is going to feel alive, it needs a controlled self-improvement loop.

### Reflection cycle

- observe events
- decide what matters
- store facts
- summarize experiences
- identify patterns
- update memory
- improve future planning

### Important constraint

The system should not blindly learn from random output without validation. Instead, it should:

- analyze recent actions
- retain only useful and meaningful information
- summarize high-value experiences
- track what strategies work
- avoid low-quality or dangerous memory insertion

This is much safer and more stable than uncontrolled self-training.

## Safety model

Because the system is autonomous and may interact with files, commands, APIs, and the environment, safety matters.

### Safety requirements

- sandboxed code execution
- permission checks for shell access
- no destructive actions without explicit approval
- approval gates for powerful commands
- structured logs of all tool usage
- rollback or snapshotting when possible
- rate limits and watchdog monitoring
- emergency stop procedure

## Technology stack

### Recommended core stack

- model serving: vLLM, Ollama, or llama.cpp
- local inference: quantized LLMs
- vector DB: Qdrant or pgvector
- relational DB: PostgreSQL
- cache / short-term memory: Redis
- orchestration: LangGraph or custom agent runtime
- retrieval framework: custom or LlamaIndex-inspired pattern
- data pipeline: Python
- storage: NVMe SSD + large archive storage
- monitoring: structured logging and task event tracking

### Example software direction

- model runtime for local inference
- database-backed memory and state
- vector retrieval for semantic context
- custom tool execution framework
- activity logging and analytics
- dashboard for monitoring the AI

## Project stages

### Phase 1: Local foundation

- get a base model running locally
- configure GPU inference
- build a stable chat interface
- validate inference speed and memory usage

### Phase 2: Memory and retrieval

- set up Postgres + vector DB
- store conversations and documents
- enable semantic retrieval over stored data
- add chunking and embeddings

### Phase 3: Tool-use and task execution

- implement file access tools
- implement repo reading tools
- add shell and code execution sandbox
- allow multi-step task workflows

### Phase 4: Reflection and continuity

- log all decisions and actions
- build memory summarization
- add long-term learning loops
- create continuity across sessions

### Phase 5: Autonomous workflows

- schedule background tasks
- enable monitoring and event-driven actions
- add persistent task loops
- create project-aware agent behavior

### Phase 6: Optimization and refinement

- upgrade to larger models when feasible
- improve retrieval quality
- refine reflection logic
- improve coding and reasoning performance

### Phase 7: Advanced autonomy

- smarter task prioritization
- better self-evaluation
- work across multiple connected systems
- more advanced environment awareness

## What success looks like

This project should eventually be able to do the following:

- hold a conversation over extended periods without losing continuity
- remember relevant facts across sessions
- retrieve information from large knowledge stores
- reason about complex tasks
- code and debug software
- search and summarize external information
- manage project context over time
- use tools to complete tasks autonomously
- improve with memory and reflection

## What this project is not

This project is not:

- a literal claim of sentience
- a magic autonomous consciousness engine
- a random uncontrolled self-training loop
- a full AGI prototype in a single step
- a system without safeguards

It is a realistic path toward building a persistent, memory-rich, tool-using AI system that behaves in a highly intelligent and autonomous way.

## Recommended development mindset

The right direction is:

- build a cognitive system, not a chatbot
- keep everything persistent
- use memory as the central feature
- let tools expand capability
- treat reflection as essential
- control autonomy carefully
- scale gradually with real evidence

## Roadmap summary

1. Run a local base model reliably
2. Add memory storage and semantic retrieval
3. Add tool execution and environment access
4. Create an always-on agent loop
5. Add reflection and memory consolidation
6. Add long-running autonomous tasks
7. Improve behavior, reliability, and performance over time

## Final objective

The long-term goal is to create the closest practical version of a sentient-like digital assistant using local infrastructure, large storage, strong open-weight models, and persistent memory systems.

This project is meant to be a serious engineering effort toward:

- a local AI brain
- a persistent cognitive system
- autonomous reasoning and action
- coding and general intelligence capabilities
- a self-updating memory architecture

## License

This project is intended to be developed as an open research and experimentation project. Use and modification are encouraged in line with the repository's chosen license.

## Notes

This README is intended as a planning document and architectural direction for the project. It can be expanded over time as the codebase and system architecture evolve.

---

This repository is currently being structured as a persistent local AI project focused on memory, reasoning, tool use, and long-term autonomy.

If you want to turn this into a real implementation next, the next milestone should be:

- choose the base model
- set up the inference stack
- build the memory database
- implement the agent loop
- add tool execution and retrieval
- then begin reflection and self-learning

That is the correct order for building the most effective version of SentientAI.
