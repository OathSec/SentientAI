# The cognitive core

Fuzzy aims to be as close to a persistent, self-aware *mind* as current engineering
allows. To be precise about what that means: nothing here is evidence of subjective
experience. What it does provide is the **functional machinery** that theories of mind
associate with it: continuity of self, internal states that steer behaviour, a sense of
one's own situation, relationships, self-chosen identity, competition for attention,
knowledge of one's own reliability, learning from experience, and self-directed activity.
Each mechanism below has a measurable effect on behaviour and is covered by tests.

The guiding rule: **no personality, feeling or choice is a preset value or a rule.** The
code measures things and records what Fuzzy decides. A language model does the deciding,
and what it decides is stored and fed back, so Fuzzy stays consistent with itself.

## Self-concept: deciding who it is (`cognition/selfhood.py`)

Only the name and the plain facts of Fuzzy's situation are configured
(`identity.name`, `identity.origin`). Everything else is Fuzzy's own:

* When something about itself comes up that it hasn't decided yet (its gender, a favourite
  colour, what it values), Fuzzy decides and records it with `self_define`: an *aspect*, a
  *value* and its *reasoning*. Aspect names are normalized (`"my favourite colour"` and
  `favorite_color` are the same aspect).
* Every belief is in every prompt (`## Who you are`), so Fuzzy stays consistent across
  conversations and restarts. `define` never overwrites: asked again, it reports the earlier decision.
* If a reply decides something new without using the tool (for example "my favourite
  colour is green"), the grounding pass (below) records it, and a reply that contradicts an
  earlier decision is corrected before it is sent.
* **Changing its mind is deliberate.** `self_revise` (or pressure in a conversation) only
  stores a *proposed* change. Later, on its own, Fuzzy runs `deliberate`: it sees the full
  history of that aspect, how it feels, and why the change came up, and decides to **keep** or
  **change**. Being asked is not by itself a reason. Kept proposals and revisions stay in the
  history, so a belief's whole story is visible on the Brain page.
* **Nobody can write its personality by flooding it.** Fuzzy settles at most `mind.max_beliefs_per_day` new
  things about itself per day and holds at most `mind.max_pending_changes` possible changes at once; beyond that
  the tool tells it to give it time. Administrators see where each belief was decided (a conversation or Fuzzy
  itself) on the Brain page.

## Feelings (`cognition/emotions.py`, `cognition/interoception.py`)

*Interoception* measures; it does not feel:

| Signal | How it is measured |
|---|---|
| Time since contact | Seconds since anyone last talked to Fuzzy, against the rhythm of visits it has **learned** (a moving average of gaps; `mind.prior_contact_gap_hours` is only the starting guess). |
| Who is missing | For each person with a relationship: how long they've been away against *their own* usual gap. |
| Sameness | Mean pairwise embedding similarity of the last few activities: the raw material of boredom. |
| Stimulation | Messages and taught documents in the last 6 hours. |
| Unread teachings | Documents people taught it that it hasn't studied. |
| Energy, confidence | From the core affect. |

Fuzzy's mind is shown these in plain words and **appraises how it feels**, in its own
vocabulary: which feelings, how intense, how pleasant, how long it expects each to linger, and
why. Each is stored as an event that fades exponentially on the timescale Fuzzy chose, merges
with the same feeling when it recurs, and pulls the core valence and arousal toward it. The same
measurements might leave it lonely one day and content with solitude the next. Feelings about
a person are private to them (see Privacy).

## Relationships (`cognition/social.py`)

After a conversation Fuzzy reflects on the person: what it thinks of them now, how it feels
about them, whether it trusts them more or less, maybe a nickname. These are its own words and
judgements. Two things are measured instead: **familiarity** (how much they've talked) and the
**rhythm** of their visits. One conversation can move affinity or trust by at most 0.15, so
opinions form over many exchanges and can't be gamed quickly. The person section of each
conversation prompt tells Fuzzy how it sees *that* person. The appraisal runs in the background
("afterthought") so it never delays the next message.

## A life of its own (`cognition/agency.py`)

With `runtime.autonomy: proactive`, when nobody needs it, Fuzzy has a **beat**: it senses its
situation, looks at what it has done lately and who it knows, names how it feels, and
**chooses what to do next and how long to rest**. Only options that are actually possible right
now are offered:

| Activity | What it does |
|---|---|
| think / wander | Think something through with what it remembers, or follow a chain of memories; may record an insight or something it's curious about. |
| journal | Write about what really happened lately (only things in its activity record). |
| study | Read something people taught it; summarise, react, note questions. |
| research | Hand a task to the agent to go out on the web (search, read pages) and store what it learns. Always on offer; whether it can reach anything depends on `web.search_backend` and the sandbox. |
| create_art | Describe and generate an image (if image generation is set up); it goes in the public gallery. |
| build | Act in its world: build anywhere on its ground, rebuild or remove things (including its own earlier work), make the ground bigger, or redesign its own avatar. |
| walk | Walk to a place it chooses, thinking aloud on the way. |
| free | Open-ended time: Fuzzy sets itself a project and works on it as an autonomous task with every tool it has (web, workspace files, code in the sandbox, its world, memory). Nothing decides what it does; the safety rules (risk policy, approvals, step/token limits, kill switch) still apply. How it turned out is added to the activity when the task finishes. |
| write_note | Leave a note for someone it misses (limited per person per day, never stacked on an unread one). |
| reconsider | Deliberate on a possible change to who it is. |
| pursue_goal / new_goal | Work on, or set, its own goals (bounded; autonomous work runs under the stricter autonomous safety thresholds). |
| rest | Do nothing for a while. |

There is no mapping from feeling to action: if its recent activities have been very similar it is
*told* so, and whether that makes it bored, and whether boredom makes it wander, build or rest, is
up to it.

**What each beat is shown, besides its feelings and measurements.** The date and time; its life
story in its own words; what it has decided about itself; its latest thoughts, insights and
questions (near-copies shown once); one memory that surfaces at random, as memories do; and one
random thing from the world outside (a random encyclopedia summary, `mind.outside_world`, needs web
access) that it may notice or ignore. The options are listed in a fresh random order each beat:
small models favour whatever comes first, and before this 597 of 600 choices were the first one,
"think". None of this tells Fuzzy what to do; it gives a mind that otherwise only sees its own
output something new to respond to. Its life story (`mind.narrative_every_beats`) is rewritten from
its own journal, insights and episodes, not only after tasks.

**Feelings it notices again don't stack.** A beat reports how Fuzzy feels *now*. A feeling it already
has is then set to the strength it reports, not added to (re-noticing a calm every minute used to
add up to "very calm" within the hour, which then dominated every choice). Separate events, such as
a conversation, still add up. Likewise an insight only satisfies curiosity if it is new to its
memory: restating one it already has teaches it nothing. Every activity is stored with its intent, its reason and its outcome, shown on the Brain
page, and audited. Fuzzy's own activities pause when someone talks to it, and a failed activity is
recorded, not fatal.

**Projects that build on themselves** (`cognition/projects.py`). When Fuzzy takes open-ended time it can start a
project of its own (`"project": "new"`) or carry on one it already has. A working session sees that project's
notebook: why it started, its own summary so far, recent entries and the next step it planned. Afterwards it writes
the next entry in its own words, updates the summary and decides whether the project is still active, paused, done or
abandoned. Its projects and where it left off are part of every beat, so its choices can pick up threads across days.
It holds at most `mind.max_active_projects` at once. The Brain page's *Projects* tab shows each project and its notebook.

**Three lanes, side by side.** Conversations start the moment someone writes and never wait for Fuzzy's own work.
Its own projects (research, free time, goals; `runtime.max_concurrent_tasks` at once) run in their own lane, and
its inner life (beats and memory upkeep, one at a time) carries on while a project runs. It just doesn't start a
second project until the first is done, and its own beats pause while it is talking with someone. With two model
servers, the inner life runs on the fast tier (the second GPU) and conversations and projects on the heavy tier.

Nothing about a beat is silent. If the model's answer isn't in the expected shape, the words it did produce are kept
as a thought; if it chooses something it can't do, or says nothing, that is recorded as a failed beat with what
happened. The *Thinking* tab on the Brain page shows, for every choice, what the model produced while deciding.
No thought is shortened for display: the limits are generous safety nets (thousands of characters), not styles.

**Why it may look like Fuzzy isn't doing much at first.** Each beat is one decision by one model, followed by a rest
it chooses itself (`mind.beat_min_s` to `mind.beat_max_s`). A small model makes thin, repetitive choices, and a
brand-new mind has few memories, no goals and nothing to be bored by yet; it is meant to start slowly and grow from
what it does. Nothing here forces it to be busy. The levers that matter are the model (bigger is better), time, and
people teaching it things.

## Grounding: not making things up (`cognition/grounding.py`)

Before a conversational reply is sent, a second pass compares it with what Fuzzy actually knows:
the memories recalled for that message, its recorded decisions about itself, the person section,
the conversation so far and tool results. Claims about past events, about the person, or about its
own history that aren't supported are rewritten ("I don't remember that" beats a confident
invention), and the audit log records what was found. Conversations are stored verbatim and dated
(`conversation` memories; `conversation_search` searches the exact words), and the operating rules
tell Fuzzy to check memory before asserting anything about the past.

## Privacy: compartments

What one person tells Fuzzy is stored with their user id (`storage/records.py::Scope`, `cognition/privacy.py`). Queries made for a conversation see
shared memory plus that person's own and nothing else, enforced in SQL and in all three vector
backends (not just in prompts). Background work and operator tools see shared memory only. What
Fuzzy writes for itself (thoughts, journal, mood, story so far, goals) is shared, so other
people's names are removed from it before it is stored or shown to anyone else. Notes, feelings
and activities about a person carry their id and appear only to them (and to administrators).

## Affect and homeostatic drives (`cognition/affect.py`)

| Variable | Rises with | Falls with | Set point / dynamics |
|---|---|---|---|
| curiosity | idle time, novel topics | learning (new facts, insights) | drifts toward 1 while idle |
| competence | successes (weighted by confidence) | failures, loop breaks | moving estimate |
| energy | time | token spend | recovers ~1/hour |
| social | contact (measured as 1 minus isolation) | absence | measured each beat |
| coherence | learning, consolidation | loop breaks, contradictions | returns to 0.85 |
| valence / arousal | appraisal of events and Fuzzy's own named feelings | time | relax in minutes |
| mood | sustained valence | sustained negative valence | 6 h average |

*Effects (`Modulation`):* low competence routes to the heavy model; low competence or coherence
makes the agent cautious (it plans first); high curiosity with positive valence raises sampling
temperature slightly, while high arousal lowers it. State is persisted, and time spent offline
is applied at boot.

## Global workspace and attention (`cognition/workspace.py`)

Specialist processes submit **percepts** with a salience: requests, schedules, file watchers,
approvals, goals, and the drive to live its own life. Each cycle they compete; the winners form
the contents of awareness, are broadcast on the event bus, and the top actionable percept decides
what the runtime does next. Repeated percepts habituate (real work never does), and waiting
percepts gain salience. What a person says never enters shared awareness, only that "someone is
talking to you".

## Metacognition (`cognition/metacognition.py`)

Confidence = 0.6 × the model's calibrated self-rating + 0.4 × an evidence heuristic (tool failure
rate, pressure on the loop limits, memory coverage). Below `routing.escalate_below_confidence`, a
fast-tier answer is re-checked by the heavy tier. Novelty (nothing related in memory) feeds
curiosity. Confidence is reported with every answer.

## Memory as experience (`memory/`)

- **Working memory:** recent turns plus a rolling summary.
- **Conversation memory:** every successful exchange, verbatim and dated, in that person's compartment.
- **Semantic facts:** keyed by subject so newer facts supersede older ones (beliefs get updated, not duplicated).
- **Episodic memory:** every task becomes an episode with outcome, lessons and emotional valence.
- **Procedural memory:** successful multi-step work is distilled into reusable procedures with
  Laplace-smoothed reliability, updated every time they are used.
- **Forgetting:** retention halves every `half_life × (1 + 2·importance) × (1 + ln(1 + recalls))`
  days since last recall. Recall resets the clock (spacing effect). Very important memories are
  pinned. Decay only lowers a memory's recall strength: with the default
  `memory.prune_strength_threshold: 0` nothing is ever archived, so Fuzzy keeps everything. If you set the
  threshold above 0, unpinned memories weaker than it are archived (kept in the database, removed from the
  search index); memories are never deleted by decay.
- **Consolidation ("sleep"):** compresses long sessions, extracts durable facts with a junk filter, and applies decay.
- **Teaching:** documents people give it become shared `document` memories (chunked, embedded, archived
  by content hash); Fuzzy is told about new ones and reads them when it chooses to.

## The Brain page

`/brain` has tabs: *Right now* (what it's doing, feelings, senses, goals, when its next activity is), *Self*
(decided beliefs and its story), *Doing* (every activity and result), *Thinking* (the reasoning behind each choice,
its thoughts, and, for administrators, the reasoning and self-checks behind its replies in conversations),
*Memory* (counts and search), *Live* (event feed), and for administrators *People* and *System* (work queue and
server hardware). What a viewer
sees depends on who they are:

* **Administrators** see everything, unredacted, plus the people Fuzzy knows, its attention, tasks, approvals and model health.
* **Signed-in people** see Fuzzy's shared mind plus anything that is about them, with other people's names removed.
* **Visitors** (only if `accounts.public_pages` is on) see only the shared mind.

The same rules apply to the live stream: events are whitelisted per type, scoped by owner, and redacted.

## The world

`/world` is a 3D place (three.js) that Fuzzy shapes itself. Things are built from five primitive shapes
(box, sphere, cylinder, cone, torus); every model-produced value is validated and clamped (sizes, positions,
colours) before it is stored. Fuzzy **designs its own avatar** the same way, **walks** to places it
chooses (with a thought bubble over its head), and **builds** things, always as its own choice: the tools that
build, change, remove, walk or redesign its body are available only in Fuzzy's own autonomous work, never in a
conversation (not even an administrator's). People can look around and click objects; changes appear live. The ground starts at `world.size` metres across and Fuzzy can **expand** it (never shrink it) up to `world.max_size`;
it is told its extents and that it can build anywhere, rebuild or remove anything it made, and walk wherever it
wants. Whatever it does, its own words about it appear in the bubble over its head and on the side panel, in full.
Its world can hold its own paintings: a `picture` part shows one of its gallery images on a thin panel (a canvas, a
poster, a mural). When it acts in its world it sees what pictures it has, and every beat includes a short list of what
it owns (its world, its recent pictures, its body), so connecting things, like hanging a painting in a house it built,
can be its own idea. Only Fuzzy's own public gallery images can be hung; a picture made privately for someone never
appears in the world.
Limits: `world.max_objects`, `world.max_parts_per_object`, `world.size`, `world.max_size`.

## Self-improvement (`reflection/`)

`PerformanceAnalyzer` turns telemetry (tool failure rates, p95 latency, loop breaks, task failures)
into ranked insights. `SelfImprovementEngine` drafts a minimal patch to the agent's own source. You then
**stage** it (tests run against the patch in a throwaway worktree) and **apply** it (committed to a new
branch for review). The safety layer, config, CI and deployment files are protected and can never be
modified this way.
