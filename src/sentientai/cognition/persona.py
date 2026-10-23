"""What a conversation's prompt needs to know about who Fuzzy is, seen from one person."""

from __future__ import annotations

from dataclasses import dataclass

from sentientai.cognition.emotions import Emotions
from sentientai.cognition.intentions import Intentions
from sentientai.cognition.privacy import Redact, Redactor
from sentientai.cognition.selfhood import SelfConcept
from sentientai.cognition.social import Relationships
from sentientai.storage.database import Database
from sentientai.storage.records import Scope


@dataclass
class Perspective:
    beliefs: str
    person: str | None
    feelings: str
    redact: Redact
    promises: str = ""  # what it told this person it would do, and whether it has


class Persona:
    def __init__(
        self,
        db: Database,
        *,
        selfhood: SelfConcept,
        emotions: Emotions,
        relationships: Relationships,
        redactor: Redactor,
        intentions: Intentions | None = None,
    ) -> None:
        self.db = db
        self.intentions = intentions
        self.selfhood = selfhood
        self.emotions = emotions
        self.relationships = relationships
        self.redactor = redactor

    async def perspective(self, user_id: str | None, *, privileged: bool) -> Perspective:
        redact = await self.redactor.for_viewer(user_id, privileged=privileged)
        beliefs = SelfConcept.render(await self.selfhood.beliefs())
        person = None
        if user_id is not None:
            user = await self.db.users.get(user_id)
            if user is not None:
                person = Relationships.render(await self.relationships.get(user_id), user)
        scope = Scope.system() if privileged else Scope.for_user(user_id)
        feelings = self.emotions.render(
            await self.emotions.current(scope), viewer=user_id, everything=privileged
        )
        promises = ""
        if user_id is not None and self.intentions is not None:
            promises = self.intentions.render_for_person(await self.intentions.for_person(user_id))
        return Perspective(
            beliefs=redact(beliefs),
            person=person,
            feelings=redact(feelings),
            redact=redact,
            promises=promises,
        )
