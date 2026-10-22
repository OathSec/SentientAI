# The cognitive core

SentientAI aims to be as close to a persistent, self-aware *mind* as current
engineering allows. To be precise about what that means: nothing here is evidence
of subjective experience or sentience. What it does provide is the **functional
machinery** that theories of mind associate with it: continuity of self, internal
states that steer behaviour, competition for attention, knowledge of one's own
reliability, learning from experience, and self-directed goals. Each mechanism below
has a measurable effect on behaviour and is covered by tests.

## Self-model and temporal continuity (`cognition/self_model.py`)

A persistent model of identity (name, purpose, values), capabilities (the live tool
registry), known limitations and an autobiographical **narrative**. The reflector
rewrites the narrative periodically from recent episodes. The self-model also tracks
boot count, first boot, total uptime, the offline gap, the track record, and what the
agent was focused on at shutdown.

*Effect:* every prompt opens with "This is awakening #N … you were offline for … before
your last shutdown you were working on …". The agent knows it is the same entity
across reboots.

## Affect and homeostatic drives (`cognition/affect.py`)

| Variable | Rises with | Falls with | Set point / dynamics |
|---|---|---|---|
| curiosity | idle time, novel topics | learning (new facts, insights) | drifts toward 1 while idle |
| competence | successes (weighted by confidence) | failures, loop breaks | moving estimate |
| energy | time | token spend | recovers ~1/hour |
| social | operator contact | absence | fades (12 h half-life) |
| coherence | learning, consolidation | loop breaks, contradictions | returns to 0.85 |
| valence / arousal | appraisal of events | time | relax in minutes |
| mood | sustained valence | sustained negative valence | 6 h average |

*Effects (`Modulation`):* low competence routes to the heavy model; low competence or
coherence makes the agent cautious (it plans first); high curiosity with positive
valence raises sampling temperature slightly, while high arousal lowers it;
initiative while idle requires energy, curiosity and coherence. The state is
persisted, and time spent offline is applied at boot.

## Global workspace and attention (`cognition/workspace.py`)

Specialist processes submit **percepts** with a salience: operator requests,
scheduled jobs, file watchers, approvals, goals, drives and memory maintenance.
Each cycle they compete. The winners form the contents of awareness, are broadcast
on the event bus, and appear in the agent's prompt. The top actionable percept
decides what the runtime does next.

- **Habituation:** repeated percepts from the same source lose salience. Real work
  (tasks) never habituates.
- **Starvation protection:** waiting percepts gain salience over time.

## Metacognition (`cognition/metacognition.py`)

Confidence = 0.6 × the model's calibrated self-rating + 0.4 × an evidence heuristic
(tool failure rate, pressure on the loop limits, memory coverage). Below
`routing.escalate_below_confidence`, a fast-tier answer is re-checked by the heavy
tier. Novelty (nothing related in memory) feeds curiosity. Confidence is reported
with every answer.

## Memory as experience (`memory/`)

- **Working memory:** recent turns plus a rolling summary.
- **Semantic facts:** keyed by subject so newer facts supersede older ones (beliefs
  get updated, not duplicated).
- **Episodic memory:** every task becomes an episode with outcome, lessons and
  emotional valence.
- **Procedural memory:** successful multi-step work is distilled into reusable
  procedures with Laplace-smoothed reliability, updated every time they are used.
- **Forgetting:** retention halves every `half_life × (1 + 2·importance) ×
  (1 + ln(1 + recalls))` days since last recall. Recall resets the clock (spacing
  effect), so useful memories persist and noise fades. Very important memories are
  pinned.
- **Consolidation ("sleep"):** compresses long sessions, extracts durable facts with
  a junk filter, and applies decay.

## Inner life and initiative (`cognition/inner_life.py`, `runtime/`)

In `runtime.autonomy: proactive`, an idle, curious and energetic agent thinks. It
reflects on recent episodes and may record an insight or propose a goal. Thoughts
are low-importance memories that fade unless recalled. Proposed goals become
**intrinsic goals**, capped by `runtime.max_intrinsic_goals`, and are pursued as
autonomous tasks. Autonomous tasks run under the stricter
`safety.autonomous_auto_approve_max_risk`, so self-directed work can read, search,
compute in the sandbox and take notes, but anything with side effects needs your
approval.

## Self-improvement (`reflection/`)

`PerformanceAnalyzer` turns telemetry (tool failure rates, p95 latency, loop breaks,
task failures) into ranked insights. `SelfImprovementEngine` drafts a minimal patch
to the agent's own source. You then **stage** it (tests run against the patch in a
throwaway worktree) and **apply** it (committed to a new branch for review). The
safety layer, config, CI and deployment files are protected and can never be
modified this way.
