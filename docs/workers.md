# Workers: lend a computer's GPU to Fuzzy

A **worker** is a small Windows program (`FuzzyWorker.exe`, source in `worker/`) that connects
*out* to the Fuzzy server and runs jobs on a model server running on the same PC. Nothing needs
opening on the router: it long-polls over HTTPS, so it works behind home routers and through the
Cloudflare Tunnel.

## What a worker can do

| Job | What it is | Which computers |
|---|---|---|
| `llm` | one chat completion (Fuzzy's replies, its thinking, its projects) | **private** only (yours) |
| `image` | one picture | private: any picture. shared: only Fuzzy's own **public** art |

If no worker is free, or it fails, the server uses its own models exactly as before; a worker
going to sleep never breaks anything. A worker does one job at a time, so a busy PC is simply
skipped and the load spreads between it and the server.

## Why two kinds (trust)

Fuzzy's thinking includes private conversations and what it knows about people. That must
never be sent to somebody else's computer, so:

* `--private` (your own PC): may run chat/thinking jobs and any picture.
* default (**shared**, a friend's PC): never sees a conversation, a private picture's prompt, or
  anything about a person. It only draws pictures Fuzzy makes for its own public gallery.

A friend's PC could in principle return a bad picture; pictures are checked to be real images
but not for content, so only give shared tokens to people you trust.

## What the program can and can't do

* Runs only the two job kinds above, against model servers **on that PC** (`127.0.0.1`). The
  server cannot aim it at the rest of the network, and it never runs commands, runs code or
  reads files.
* Only the request fields a model server needs are passed on (messages, tools, sampling).
* Limits the user sets: only after N minutes without keyboard/mouse, pause while the GPU is busy.
* Stop (or closing the window) disconnects at once. Revoke a token on the server any time.

## Setting one up

On the server:

```bash
docker compose -f deploy/docker-compose.yml exec sentientai sentientai workers create owens-pc --private
docker compose -f deploy/docker-compose.yml exec sentientai sentientai workers create zanders-pc   # a friend
docker compose -f deploy/docker-compose.yml exec sentientai sentientai workers list
docker compose -f deploy/docker-compose.yml exec sentientai sentientai workers revoke zanders-pc
```

The token is shown once. Give it to the person with the site address (the same one they chat on).

On the PC:

1. Run a model server: **Ollama** or **LM Studio** for chat (pull the same model Fuzzy uses,
   e.g. `ollama pull qwen3:14b`), and/or **Stable Diffusion WebUI** (started with `--api`)
   for pictures.
2. Open `FuzzyWorker.exe`, paste the site address and token, tick what to share, press
   **Check my setup**, then **Start**.

Context window: the server plans each request for the context in `models.*.context_window`
(12288 here), so give the PC's chat model at least that (Ollama: `OLLAMA_CONTEXT_LENGTH=12288`).

Building the exe: `powershell -ExecutionPolicy Bypass -File worker/build.ps1` (needs Python 3.12).

## For operators

`GET /v1/workers` (operator token) lists registered and online workers. Job hand-outs are in the
audit log as `worker.job`. Protocol: `POST /worker/poll`, `/worker/result`, `/worker/bye` with
`Authorization: Bearer fzw_...`.
