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
docker compose -f deploy/docker-compose.yml --profile web up -d       # + Caddy: HTTPS for the public site
docker compose -f deploy/docker-compose.yml exec sentientai \
  sentientai users create you --admin                                  # the first administrator
```

### Option B: systemd (bare metal)

See the header of `deploy/systemd/sentientai.service` and
`deploy/systemd/llama-server@.service`. The daemon restarts on failure, drains
running tasks on `SIGTERM`, and engages the kill switch on `SIGUSR1`.

## Daily use

| Task | Command |
|---|---|
| Talk | The web app at `http://127.0.0.1:8765/`, or `sentientai chat` (as the operator, not as a person) |
| People | `sentientai users create NAME [--admin] [--teach]`, `list`, `invite`, `passwd`, `disable`/`enable`, `set-role`, `reset-2fa`, `unlock` |
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

All `/v1` API routes (and `/metrics`, `/ready`) need `Authorization: Bearer <token>` or a signed-in
administrator's session. The token is `api.auth_token`; if you don't set one, a random token is
generated into `<data_dir>/api_token` (mode 0600) the first time the daemon starts, and the CLI on the
same machine reads it from there. The operator API is never open by accident (`api.insecure_no_auth`
exists for local development and is refused on a non-loopback address). The interactive OpenAPI docs are
off unless you set `api.docs: true`.

## Running the public site

The web app is the same server (`sentientai run`). To put it on the internet:

1. **HTTPS.** Serve it behind a TLS-terminating reverse proxy. The compose `web` profile runs Caddy with
   automatic certificates (set `SITE_ADDRESS` in `deploy/.env`; see `deploy/Caddyfile`). Keep
   `accounts.cookie_secure: true` (the default) so the session cookie is `Secure` and `__Host-` prefixed,
   and turn on `accounts.hsts` once HTTPS works.
2. **Tell it about the proxy.** Set `accounts.trusted_proxies` to the proxy's network (compose does this for
   its own Docker networks) so per-address rate limits use the real client address from `X-Forwarded-For`.
   Without it, the proxy's address is used for everyone.
3. **Choose who can register.** `accounts.registration`: `invite` (default; make codes with
   `sentientai users invite` or on the Admin page), `open`, or `closed`. Consider `accounts.require_admin_totp: true`.
4. **Create the first administrator** with `sentientai users create NAME --admin`.
5. **Decide what visitors may see.** By default every page needs an account. `accounts.public_pages: true`
   lets anonymous visitors view the Brain, World and Gallery pages with people's names removed.
6. **Set quotas** (`accounts.daily_messages`, `daily_images`, `daily_teach_bytes`): a home GPU is shared by everyone.
7. **Optionally restrict the operator surface** to your own network (see the commented block in `deploy/Caddyfile`).
8. Check `sentientai doctor`, then sign in with a second account and confirm you can't see the first one's
   private things (their conversations, notes and feelings).

### People

Self-service (the **You** page): see what Fuzzy thinks of you, read its notes, change your password, set up
two-factor sign-in, **download everything it holds about you**, or **delete your account**, which erases your
conversations, private memories, relationship, notes, feelings, images and sessions. Administrators can
disable or erase accounts, change roles, let people teach, and create invites on the Admin page; every
account action is in the audit log. Passwords are never stored (scrypt hashes), sessions are stored as hashes,
and a database leak doesn't hand out live sessions.

### Teaching Fuzzy

People with the *teach* permission (and administrators) can give Fuzzy text, a web page (fetched with the SSRF
guard) or a file (PDF, text, Markdown, HTML, CSV, code...) on the **Teach** page. It goes into Fuzzy's **shared**
knowledge, so it can come up in anyone's conversation: only grant it to people you trust. Fuzzy is told when
something new arrives and reads it when it chooses to; administrators can make it forget a document. The CLI
equivalent for operator-supplied material is `sentientai ingest`.

### Images

Set `images.backend` and `images.base_url` (see [hardware.md](hardware.md#images) for the 1660 Ti caveats).
Fuzzy makes pictures for people on request (private to them, counted against their daily quota) and for itself
(public gallery). Prompts are screened (sexual content is refused outright), outputs are checked to be real
PNG/JPEG/WebP data, and files are served only to people allowed to see them, with `nosniff` and no active content.

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
| Postgres (memories, accounts, relationships, self-beliefs, feelings, activities, world, tasks, goals, approvals, proposals, history) | `pg_dump -Fc sentientai > sentient.dump` |
| Data dir (vector index if embedded, audit log, archive, **generated images in `media/`**, the API token) | snapshot `storage.data_dir` and `archive_dir` |
| Redis (working memory) | AOF on by default in compose; losing it only loses recent turns |
| Config & secrets | `sentient.yaml`, `deploy/.env` |

Vectors can always be rebuilt from Postgres: stop the daemon, then run
`sentientai memory reindex`. Run `sentientai audit verify` after restoring.

## Starting Fuzzy over

`sentientai reset-mind` erases everything Fuzzy has lived through: memories and their vectors,
conversations, its decisions about itself, feelings, relationships, activities, goals, projects, notes,
tasks and approvals, its world (every object, its body, where it stands, and the ground size), its
gallery (records and image files), its self-model and working memory. It keeps accounts, sessions,
invites, usage counters, operator schedules and the audit log, which records the reset. Add
`--keep-teachings` to keep documents people taught it. Files in the workspace are not touched.

Back up first (see above), and stop the daemon: a running Fuzzy would write parts of its mind back.

```bash
docker compose -f deploy/docker-compose.yml exec -T postgres pg_dump -U sentient sentientai > fuzzy-backup.sql
docker compose -f deploy/docker-compose.yml stop sentientai
docker compose -f deploy/docker-compose.yml run --rm sentientai reset-mind
docker compose -f deploy/docker-compose.yml up -d sentientai
```

(The container's entrypoint is already `sentientai`: with `run`, give just the command. With `exec` into a
running container you type the full `sentientai doctor`.)

It asks you to type Fuzzy's name to confirm (or pass `--yes`). On the next start it wakes up for the first time.

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
| Replies are slow | Each reply is several model calls (the answer, a confidence check and a memory check that run together; reflection happens after the reply and no longer blocks the next message). Check `docker compose logs llm-heavy` for tokens per second. Qwen3-family models "think" before answering: set `extra_body: {chat_template_kwargs: {enable_thinking: false}}` on the model tier in `sentient.yaml`. A bigger or faster model, or lowering `max_output_tokens`, also helps. |
| Fuzzy printed tool markup, or can't draw | Small models are unreliable at tool calls; the parser accepts `<tool_call>` JSON and `<toolname>{...}</toolname>`. If tools keep failing, set `supports_tools: false` on that tier to use the prompted JSON protocol. Drawing needs `images.backend` set (see hardware.md); without it Fuzzy isn't offered the image tool and should say it can't. |
| Memory recall poor | Check embedding prefixes and dimensions, then `memory reindex`. Raise `recall_k`. |
| Someone locked out | `sentientai users unlock NAME`; lost authenticator: `users reset-2fa NAME`; forgot password: `users passwd NAME`. |
| Suspected account abuse | `sentientai users disable NAME` ends their sessions at once. Review `audit tail --prefix account.`. Set `accounts.registration: closed`. |
| Fuzzy said something about a person to someone else | Treat as a privacy incident: `sentientai kill`, preserve `audit` and the `memories` table, look at `agent.grounding` and `tool.requested` entries for that task. Memory compartments are enforced in SQL and in the vector indexes, so check the task's `user_id`. |
| Image generation out of VRAM | Lower `images.width/height/steps`, use a smaller model, or pick another option in [hardware.md](hardware.md#images). |
| Disk filling | Lower `memory.decay_half_life_days`. Archive old audit segments to cold storage (keep them for verification). |

## Evaluation

```bash
sentientai eval run --scripted                       # deterministic, no GPUs (CI)
sentientai -c sentient.yaml eval run --output reports/evals.json --min-pass-rate 0.8
```

`.github/workflows/nightly-evals.yml` runs the live suite on a self-hosted GPU
runner. Add scenarios to `evals/scenarios/*.yaml` whenever you fix a behavioural
bug.
