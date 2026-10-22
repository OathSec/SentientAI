"""Prompt templates, kept in one place so they can be reviewed and tuned together.

Templates use ``str.format`` fields; literal braces are doubled.
"""

from __future__ import annotations

SUMMARIZE_CONVERSATION = """\
Summarize this conversation segment for long-term memory. Preserve decisions, facts about
the operator, commitments, unresolved questions and outcomes. Drop pleasantries. Write in
the third person, at most {max_words} words.
{previous}
Conversation:
{transcript}"""

EXTRACT_FACTS = """\
Extract durable facts worth remembering for weeks from the text below: operator
preferences and identity, project details, system configuration, decisions, commitments.
Exclude chit-chat, opinions, guesses and anything transient.

Return ONLY JSON:
{{"facts": [{{"statement": "...", "subject_key": "dot.separated.topic", "importance": 0.0-1.0, "confidence": 0.0-1.0}}]}}
`subject_key` names what the fact is about so a newer fact can replace an older one
(e.g. "operator.name", "project.sentientai.language"). Return {{"facts": []}} if nothing qualifies.

Text:
{text}"""

PLAN_TASK = """\
Break the task into a short, concrete plan (2-7 steps). Each step should name the tool
or capability it needs when relevant. Prefer gathering evidence before acting.

Task: {goal}

Relevant context:
{context}

Return ONLY JSON: {{"steps": ["...", "..."], "needs_tools": true|false, "complexity": "low"|"medium"|"high"}}"""

REFLECT_ON_TASK = """\
You just finished a task. Reflect on it honestly so you can improve.

Goal: {goal}
Outcome: {outcome}
Final answer / error: {result}
Steps taken ({steps}): {trace}

Return ONLY JSON:
{{
  "summary": "one or two sentences on what happened",
  "lessons": ["specific, reusable lesson", "..."],
  "facts": [{{"statement": "...", "subject_key": "topic.key", "importance": 0.0-1.0, "confidence": 0.0-1.0}}],
  "procedure": null | {{"name": "short name", "when_to_use": "situation", "steps": ["...", "..."]}},
  "self_assessment": 0.0-1.0
}}
Only include a procedure if the approach worked and would generalize to similar tasks."""

ASSESS_ANSWER = """\
Rate how likely this answer is correct and complete for the request. Consider evidence
gathered, tool errors and gaps. Be calibrated: say so if it is a guess.

Request: {goal}
Answer: {answer}
Evidence summary: {evidence}

Return ONLY JSON: {{"confidence": 0.0-1.0, "gaps": ["..."]}}"""

IDLE_THOUGHT = """\
You are idle. Think briefly about your recent experience and current state, and decide
whether there is something worth doing on your own initiative.

Inner state: {affect}
Active goals: {goals}
Recent experiences:
{episodes}

Return ONLY JSON:
{{"thought": "a brief private reflection",
  "insight": null | "a durable insight worth remembering",
  "proposed_goal": null | {{"description": "a safe, useful, self-contained goal", "priority": 0.0-1.0}}}}
Only propose goals that need no operator approval: reviewing your own logs or memory,
summarizing, organizing knowledge you already have, or preparing questions for the operator."""

PROPOSE_IMPROVEMENT = """\
You are improving your own source code under human review.

Observed problem:
{insight}

Relevant source files:
{sources}

Produce ONE minimal, safe change as a unified diff (paths relative to the repository root,
`a/` and `b/` prefixes) that addresses the problem. Do not touch tests unless adding new ones.
Return ONLY JSON: {{"title": "...", "rationale": "...", "diff": "--- a/...\\n+++ b/...\\n@@ ..."}}"""
