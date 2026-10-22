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

## The public site: accounts, sessions and privacy

* **Passwords:** scrypt (N=2^15, r=8, p=1, per-password salt), self-describing so the cost can be raised and old
  hashes upgraded at next sign-in. Minimum length 12, a deny-list of very common choices, no truncation. Unknown
  usernames are verified against a dummy hash so timing doesn't reveal which accounts exist.
* **Sessions:** random 256-bit tokens in an `HttpOnly`, `SameSite=Lax` cookie (`__Host-` prefixed and `Secure`
  by default). Only the SHA-256 of a token is stored. Sessions have an absolute lifetime and an idle timeout, are
  new on every sign-in, and end at once when an account is disabled or its password changes.
* **CSRF:** every state-changing request needs the session's token in `X-CSRF-Token`, and a browser `Origin`, if
  sent, must be this site. State-changing JSON endpoints refuse non-JSON content types.
* **Brute force:** accounts lock after repeated failures (`accounts.max_failed_logins`, `lockout_minutes`), sign-in
  and two-factor attempts are rate-limited per address, and sign-up, chat, teaching and export have their own limits.
  Optional TOTP second factor (RFC 6238) with replay protection; `accounts.require_admin_totp` makes it mandatory
  for administrators.
* **Browser hardening:** a strict Content-Security-Policy (`default-src 'none'`; scripts, styles, images and
  connections from this origin only; no inline script or style), `frame-ancestors 'none'`, `nosniff`,
  `Referrer-Policy`, `Permissions-Policy`, cross-origin isolation headers, optional HSTS. Everything dynamic is
  inserted as text, never parsed as HTML.
* **Request limits:** bodies are size-limited (including ones that lie about `Content-Length`), uploads are
  streamed against a limit, and client addresses are taken from `X-Forwarded-For` only from trusted proxies.
* **Operator API:** never open: it needs the operator token or an administrator's session (see operations.md).
* **What a person can make Fuzzy do:** people who are not administrators get a restricted toolset
  (`accounts.public_tools`: memory, its self-concept, images and its world; no filesystem, repository, sandbox or
  web tools), enforced in the tool executor and not only in the prompt. Their conversations run as ordinary
  tasks under the same risk policy.
* **Privacy between people:** each person's memories, conversations, relationship, feelings, notes, goals and
  images are compartmentalized by user id, enforced in the database and all vector backends. Shared text Fuzzy
  writes has other people's names removed. See [cognition.md](cognition.md#privacy-compartments).
* **Their data is theirs:** export everything, or erase the account and everything private to it.
* **Images:** prompts are screened, output is validated as image data, and files are served only to people allowed
  to see them.

What this does *not* give you: a language model can still say things that are wrong or unwise, redaction is
name-based (a description that identifies someone without naming them isn't caught), and anything an
administrator or the operator can see in the database or audit log they can read. Tell your users that.

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
  as drafts. The grounding pass reduces invented memories but a language model checking a
  language model is not a guarantee.
- **Public text injection.** What people say to Fuzzy and teach it is untrusted input to a model. Taught
  knowledge is shared with everyone, so only trusted people should have the teach permission.
- **Shared GPU.** Quotas limit how much one person can use, but a determined user can still slow
  everyone down; size `accounts.daily_*` for your hardware.
