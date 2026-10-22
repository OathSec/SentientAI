# Operations runbook

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[all]"   # or: pip install "sentientai[all]"
sentientai init -o sentient.yaml                             # commented example config
sentientai -c sentient.yaml doctor                           # check every subsystem
```

### Option A: Docker Compose

```bash
cp deploy/.env.example deploy/.env        # secrets, absolute storage paths, model files
cp sentient.yaml deploy/sentient.yaml
docker compose -f deploy/docker-compose.yml up -d                     # runtime + Postgres + Redis
docker compose -f deploy/docker-compose.yml --profile models up -d    # + llama.cpp servers
```

### Option B: systemd (bare metal)

See the header of `deploy/systemd/sentientai.service` and
`deploy/systemd/llama-server@.service`. The daemon restarts on failure, drains
running tasks on `SIGTERM`, and engages the kill switch on `SIGUSR1`.

## Daily use

| Task | Command |
|---|---|
| Talk | `sentientai chat` (or the dashboard at `http://127.0.0.1:8765/`) |
| One-shot question | `sentientai ask "…"` |
| Background task | `sentientai submit "…" --wait` |
| What is it doing / feeling? | `sentientai state`, `sentientai events` |
| Pending approvals | `sentientai approvals list` then `approve <id>` or `deny <id> --note "…"` |
| Add knowledge | `sentientai ingest docs/ https://example.org/paper.pdf`, or drop files in a watched folder |
| Memory | `sentientai memory search "…"`, `remember "…" --key topic.key`, `forget <id>`, `stats`, `consolidate` |
| Goals | `sentientai goals add "…"`, `goals list` |
| Emergency stop | `sentientai kill "reason"`, then `sentientai resume` |
| Self-improvement | `improve propose`, `improve show <id>`, `improve stage <id>`, `improve apply <id>` |
| Model speed baseline | `sentientai bench` |

All `/v1` API routes require `Authorization: Bearer $SENTIENT_API__AUTH_TOKEN` when a
token is configured. The daemon refuses to bind a non-loopback address without one.
OpenAPI docs are served at `/docs`.

## Monitoring

- `GET /health` is an unauthenticated liveness probe. `GET /ready` checks the
  database, working memory and model tiers (503 if not ready).
- `GET /metrics` is Prometheus text: LLM requests, latency and tokens by tier; tool
  calls and latency; tasks by source and status; loop breaks; approvals; policy
  denials; memory items; affect; uptime; kill-switch state.

```yaml
# prometheus.yml
scrape_configs:
  - job_name: sentientai
    authorization: {credentials: "<api token>"}
    static_configs: [{targets: ["127.0.0.1:8765"]}]
```

Useful alerts: `sentient_kill_switch_engaged == 1`,
`increase(sentient_loop_breaks_total[1h]) > 5`,
`rate(sentient_llm_requests_total{outcome="error"}[5m]) > 0`, and pending approvals
older than your SLA.

## Backup and restore

| What | How |
|---|---|
| Postgres (memories, tasks, goals, approvals, proposals, history) | `pg_dump -Fc sentientai > sentient.dump` |
| Data dir (vector index if embedded, audit log, archive) | snapshot `storage.data_dir` and `archive_dir` |
| Redis (working memory) | AOF on by default in compose; losing it only loses recent turns |
| Config & secrets | `sentient.yaml`, `deploy/.env` |

Vectors can always be rebuilt from Postgres: stop the daemon, then run
`sentientai memory reindex`. Run `sentientai audit verify` after restoring.

## Upgrades

1. `sentientai kill "upgrade"`, wait for running tasks to stop, stop the service.
2. Back up (above).
3. Install the new version. The schema version is checked at startup, and a newer
   database is refused by an older build.
4. Start, then run `doctor`, `audit verify`, `resume`.

## Incident playbook

| Symptom | Action |
|---|---|
| Agent doing something unexpected | `sentientai kill`, then inspect with `audit tail -n 100` and `state` |
| `audit verify` fails | Treat as tampering or disk corruption. Preserve the segment files, check `.torn` sidecars, restore from backup, rotate the HMAC key. |
| Model tier down | Router falls back to the other tier. Check `doctor`, the llama.cpp logs, and `sentient_llm_requests_total{outcome="error"}`. |
| Frequent loop breaks | `sentientai improve propose` or the `self_analyze` tool show which limits trip. Tune `limits.*` or the prompts. |
| Memory recall poor | Check embedding prefixes and dimensions, then `memory reindex`. Raise `recall_k`. |
| Disk filling | Lower `memory.decay_half_life_days`. Archive old audit segments to cold storage (keep them for verification). |

## Evaluation

```bash
sentientai eval run --scripted                       # deterministic, no GPUs (CI)
sentientai -c sentient.yaml eval run --output reports/evals.json --min-pass-rate 0.8
```

`.github/workflows/nightly-evals.yml` runs the live suite on a self-hosted GPU
runner. Add scenarios to `evals/scenarios/*.yaml` whenever you fix a behavioural
bug.
