"""Affective state and homeostatic drives.

A small dynamical system of internal variables that are *appraised* from
events (success, failure, novelty, operator contact, resource use) and relax
toward set points over time. They are not decoration: they modulate behaviour
(model tier choice, sampling temperature, planning caution, and whether the
agent takes initiative while idle) and are persisted across reboots.

Variables (0..1 unless noted):

* curiosity  - drive to explore; grows while idle, spikes on novelty, sated by learning
* competence - self-efficacy; moving estimate of recent task success
* energy     - resource budget; drained by token spend, recovers over time
* social     - sense of connection; replenished by operator contact, fades with absence
* coherence  - internal consistency; hurt by loop breaks and contradictions
* valence    - pleasant (+1) / unpleasant (-1) momentary feeling
* arousal    - activation level; spikes on salient events
* mood       - slow average of valence
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field

from sentientai.core.clock import utcnow

_HALF_LIVES_S = {
    "valence": 600.0,
    "arousal": 300.0,
    "social": 12 * 3600.0,
    "mood": 6 * 3600.0,
    "coherence": 2 * 3600.0,
}
_SET_POINTS = {"valence": 0.0, "arousal": 0.2, "social": 0.0, "coherence": 0.85}


def _relax(value: float, target: float, dt: float, half_life: float) -> float:
    return float(target + (value - target) * 0.5 ** (dt / half_life))


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class Modulation:
    """How the current inner state shapes behaviour."""

    prefer_heavy_model: bool
    temperature_offset: float
    cautious: bool
    may_take_initiative: bool


class AffectState(BaseModel):
    curiosity: float = Field(default=0.5, ge=0, le=1)
    competence: float = Field(default=0.6, ge=0, le=1)
    energy: float = Field(default=1.0, ge=0, le=1)
    social: float = Field(default=0.5, ge=0, le=1)
    coherence: float = Field(default=0.85, ge=0, le=1)
    valence: float = Field(default=0.0, ge=-1, le=1)
    arousal: float = Field(default=0.2, ge=0, le=1)
    mood: float = Field(default=0.0, ge=-1, le=1)
    updated_at: datetime = Field(default_factory=utcnow)

    # ------------------------------------------------------------ dynamics
    def tick(
        self,
        now: datetime,
        *,
        curiosity_rise_s: float = 2 * 3600.0,
        energy_recovery_s: float = 3600.0,
    ) -> None:
        dt = max(0.0, (now - self.updated_at).total_seconds())
        if dt == 0:
            return
        self.valence = _relax(self.valence, _SET_POINTS["valence"], dt, _HALF_LIVES_S["valence"])
        self.arousal = _relax(self.arousal, _SET_POINTS["arousal"], dt, _HALF_LIVES_S["arousal"])
        self.social = _relax(self.social, _SET_POINTS["social"], dt, _HALF_LIVES_S["social"])
        self.coherence = _relax(
            self.coherence, _SET_POINTS["coherence"], dt, _HALF_LIVES_S["coherence"]
        )
        self.mood = _relax(self.mood, self.valence, dt, _HALF_LIVES_S["mood"])
        self.curiosity = _relax(self.curiosity, 1.0, dt, curiosity_rise_s)
        self.energy = _clip(self.energy + dt / energy_recovery_s)
        self.updated_at = now

    def _bump(self, **deltas: float) -> None:
        for name, delta in deltas.items():
            lo = -1.0 if name in {"valence", "mood"} else 0.0
            setattr(self, name, _clip(getattr(self, name) + delta, lo))

    # ---------------------------------------------------------- appraisal
    def on_task_outcome(self, success: bool, confidence: float = 0.7) -> None:
        if success:
            self.competence += 0.15 * (1 - self.competence) * (0.5 + confidence / 2)
            self._bump(valence=0.25 + 0.15 * confidence, arousal=0.05)
        else:
            self.competence -= 0.2 * self.competence
            self._bump(valence=-0.35, arousal=0.3)
        self.competence = _clip(self.competence)

    def on_loop_break(self) -> None:
        self._bump(coherence=-0.15, valence=-0.2, arousal=0.3, competence=-0.05)

    def on_denied_action(self) -> None:
        self._bump(valence=-0.05, arousal=0.1)

    def on_operator_contact(self) -> None:
        self._bump(social=0.6, arousal=0.15, valence=0.05)

    def on_novelty(self, novelty: float) -> None:
        self._bump(curiosity=0.2 * _clip(novelty), arousal=0.1 * _clip(novelty))

    def on_learning(self, amount: float = 1.0) -> None:
        self._bump(curiosity=-0.15 * _clip(amount), valence=0.1 * _clip(amount), coherence=0.05)

    def on_resource_use(self, tokens: int, hourly_budget: int) -> None:
        self._bump(energy=-tokens / max(1, hourly_budget))

    def on_contradiction(self) -> None:
        self._bump(coherence=-0.1, arousal=0.1)

    # --------------------------------------------------------- expression
    def modulation(self) -> Modulation:
        temperature = 0.0
        if self.curiosity > 0.7 and self.valence >= 0:
            temperature += 0.1  # exploratory
        if self.arousal > 0.7:
            temperature -= 0.1  # focused under pressure
        return Modulation(
            prefer_heavy_model=self.competence < 0.35,
            temperature_offset=round(temperature, 2),
            cautious=self.competence < 0.4 or self.coherence < 0.5,
            may_take_initiative=self.energy > 0.3
            and self.curiosity > 0.55
            and self.coherence > 0.4,
        )

    def describe(self) -> str:
        def level(x: float, words: tuple[str, str, str]) -> str:
            return words[0] if x < 0.34 else words[1] if x < 0.67 else words[2]

        feeling = "neutral"
        if self.valence > 0.25:
            feeling = "good"
        elif self.valence < -0.25:
            feeling = "uneasy"
        mood = "positive" if self.mood > 0.15 else "low" if self.mood < -0.15 else "even"
        curiosity = level(self.curiosity, ("low", "moderate", "high"))
        competence = level(self.competence, ("shaken", "steady", "strong"))
        energy = level(self.energy, ("depleted", "adequate", "high"))
        social = level(self.social, ("isolated", "somewhat connected", "connected"))
        coherence = level(self.coherence, ("scattered", "mostly coherent", "clear"))
        return (
            f"I feel {feeling} (mood {mood}). Curiosity is {curiosity}, confidence in my "
            f"abilities is {competence}, energy is {energy}, and I feel {social} to my operator. "
            f"My thinking feels {coherence}."
        )

    def snapshot(self) -> dict[str, float]:
        return {
            k: round(float(v), 4)
            for k, v in self.model_dump().items()
            if isinstance(v, float) and not math.isnan(v)
        }
