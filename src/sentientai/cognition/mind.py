"""The persistent cognitive state, loaded at boot and checkpointed while running."""

from __future__ import annotations

from dataclasses import dataclass

from sentientai.cognition.affect import AffectState
from sentientai.cognition.goals import GoalManager
from sentientai.cognition.self_model import SelfModel
from sentientai.cognition.workspace import GlobalWorkspace
from sentientai.config import IdentityConfig
from sentientai.core.clock import Clock
from sentientai.storage.database import StateRepository

SELF_MODEL_KEY = "mind.self_model"
AFFECT_KEY = "mind.affect"


@dataclass
class Mind:
    self_model: SelfModel
    affect: AffectState
    workspace: GlobalWorkspace
    goals: GoalManager

    @classmethod
    async def load(
        cls,
        state: StateRepository,
        identity: IdentityConfig,
        *,
        workspace: GlobalWorkspace,
        goals: GoalManager,
        clock: Clock,
    ) -> Mind:
        raw_self = await state.get(SELF_MODEL_KEY)
        self_model = (
            SelfModel.model_validate(raw_self) if raw_self else SelfModel.from_identity(identity)
        )
        self_model.sync_identity(identity)
        raw_affect = await state.get(AFFECT_KEY)
        affect = (
            AffectState.model_validate(raw_affect)
            if raw_affect
            else AffectState(updated_at=clock.now())
        )
        affect.tick(clock.now())  # time passed while offline: drives relaxed, curiosity grew
        return cls(self_model=self_model, affect=affect, workspace=workspace, goals=goals)

    async def save(self, state: StateRepository) -> None:
        await state.set(SELF_MODEL_KEY, self.self_model.model_dump(mode="json"))
        await state.set(AFFECT_KEY, self.affect.model_dump(mode="json"))
