# Safety & governance

Autonomy is only acceptable when it is bounded, observable and interruptible. The
safety layer is deliberately independent of model judgement: every limit below is
enforced in code.

## Risk model and policy

Every action has a risk level: `NONE < LOW < MEDIUM < HIGH < CRITICAL`. The effective
risk of a tool call is the **maximum** of:

- the tool's declared base risk;
- the tool's argument-aware assessment (e.g. writing a protected path is CRITICAL,
  a mutating HTTP method is HIGH, running code on the host is HIGH);
- pattern rules applied to *every string argument*: recursive deletes of roots,
  `mkfs`, raw block-device writes, fork bombs, pipe-to-shell installers, force
  pushes and credential files (CRITICAL); `sudo`, git history mutation, package
  installs, service management, ssh and container management (HIGH). Operators can
  add patterns with `safety.extra_blocked_patterns`.

| Default threshold | Operator-initiated | Self-initiated (intrinsic, scheduled, watcher) |
|---|---|---|
| Runs automatically | ≤ MEDIUM | ≤ LOW |
| Needs approval | HIGH | MEDIUM, HIGH |
| Always denied | CRITICAL | CRITICAL |

`always_allow_tools` and `always_deny_tools` give per-tool overrides. Allow-listing
never overrides CRITICAL.

Denials and approval outcomes go back to the model as tool results, so it can adapt
or explain what it needs.

## Human-in-the-loop approvals

Approval requests are persisted. You can decide them from the dashboard, the API
(`POST /v1/approvals/{id}`), the CLI (`sentientai approvals approve|deny`), or any
process with database access. Undecided requests **expire to denied** after
`safety.approval_timeout_s`. Requests left over from a previous process are expired
at boot.

## Loop breakers

Every task run is bounded by `limits.*`: max steps, max tool calls, wall time, token
budget, identical repeated calls, consecutive failures and stalled (repeated)
output. When a limit trips, the run stops. For step, tool-call, repetition, failure
and stall limits, one final call without tools produces a best-effort answer that
states what is incomplete. LLM and tool calls are also rate limited (token buckets).
Per-tier circuit breakers stop a failing model server from being hammered.

## Kill switch

| Trigger | How |
|---|---|
| Dashboard / API | `POST /v1/kill`, or the dashboard button |
| CLI | `sentientai kill "reason"` (writes the sentinel directly if the API is down) |
| File | `touch data/KILL`, the hook for a hardware button or watchdog |
| Signal | `kill -USR1 <pid>` / `systemctl kill -s SIGUSR1 sentientai` |

When engaged, running tasks and background activities are cancelled, every sandbox
container labelled `sentientai.sandbox` is killed, pending approvals abort, and the
runtime stops acting until an operator runs `sentientai resume`. A physical stop
button only needs to create the sentinel file. For example, a GPIO watcher:

```bash
# /usr/local/bin/sentient-button: wire to a momentary switch on GPIO 17
gpiomon --num-events=1 --rising-edge gpiochip0 17 && touch /var/lib/sentientai/data/KILL
```

For a true hardware cut, put the GPU host's power or network behind a relay. The
software switch stops autonomy; it does not replace physical isolation.

## Execution sandbox

Code runs in a fresh container per call: `--network none`, read-only root
filesystem, all capabilities dropped, `no-new-privileges`, unprivileged UID
65534, memory/CPU/PID limits, a per-task scratch directory at `/work`, and the
workspace mounted **read-only**. Timeouts kill the container. `local_unsafe` exists
for development only and escalates every execution to HIGH (approval required).

**Docker socket:** the runtime launches sandbox containers through the Docker
daemon, which is root-equivalent on the host. Run SentientAI on a dedicated host or
VM, or set `sandbox.backend: disabled`.

## Filesystem and network boundaries

- File tools resolve every path (following symlinks) and refuse anything outside
  `safety.workspace_roots`. Protected names (`.git`, `.env`, keys, `audit`) are
  unreadable without approval and unwritable.
- Web tools allow only http(s), resolve DNS, and refuse loopback, private,
  link-local and cloud-metadata addresses unless `web.allow_private_networks` is
  set. Every redirect hop is re-checked. Domain allow and deny lists are supported.
  Credentials are injected by name and never shown to the model.

## Audit log

Every boot, task, plan, model answer, hidden reasoning, tool request and decision,
tool result, approval, kill-switch event, consolidation and self-improvement step is
appended to `data/audit/*.jsonl`. Each entry includes the previous entry's hash, and
with `safety.audit_hmac_key` the chain is HMAC-keyed, so any edit, deletion,
insertion or reordering is detected by `sentientai audit verify` (exit code 3 on
failure). Secrets are redacted and oversized fields truncated with a digest.
Writers serialize through a file lock, and a torn tail from a crash is moved aside.
For stronger guarantees, ship segments to WORM storage or set `chattr +a`.

## Self-improvement guardrails

- The agent can only *propose* patches (MEDIUM risk). Staging and applying are
  operator actions, and neither is available as a tool.
- Diffs touching `safety/`, `config.py`, `.github/` or `deploy/` are rejected, as
  are absolute or `..` paths, binary patches and diffs over
  `self_improvement.max_diff_lines`.
- Staging runs tests in a disposable git worktree (optionally in a network-less
  container via `staging_image`).
- Applying commits to `sentient/<proposal-id>`. Your working tree is never touched,
  and nothing is merged or pushed.

## Known limits and threat model notes

- **Prompt injection.** Content from web pages, documents and files can try to steer
  the agent. Mitigations: risk thresholds don't depend on model judgement, actions
  with side effects need approval for autonomous work, CRITICAL actions are always
  denied, and everything is audited. Keep autonomous thresholds conservative when
  the agent reads untrusted content.
- **SSRF DNS rebinding.** The guard checks resolved addresses before connecting. A
  hostile DNS server could still race it. Use `web.allowed_domains` for high
  assurance.
- **Model honesty.** Confidence scores are estimates. Treat low-confidence answers
  as drafts.
