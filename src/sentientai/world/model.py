"""Shapes, avatars and Fuzzy's body in its world.

Everything in the world is built from a few primitive shapes. Part data comes from
Fuzzy's model output, so it is validated and clamped here: odd values are fixed or
dropped rather than trusted.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from sentientai.core.clock import utcnow

Shape = Literal["box", "sphere", "cylinder", "cone", "torus", "picture"]
_MEDIA_ID = re.compile(r"^img_[0-9a-z]{6,40}$")
_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_NAMED = {
    "white": "#f2f2f2",
    "black": "#1e1e1e",
    "grey": "#8a8a8a",
    "gray": "#8a8a8a",
    "red": "#c0392b",
    "orange": "#e67e22",
    "yellow": "#f1c40f",
    "green": "#4c9a4c",
    "blue": "#3a6fb0",
    "purple": "#7d4fa0",
    "pink": "#e58fb4",
    "brown": "#7a5230",
    "beige": "#d8c8a8",
}
MAX_EXTENT = 20.0  # metres, for any single dimension
MIN_EXTENT = 0.02


def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return max(lo, min(hi, x))


def _vec3(value: Any, lo: float, hi: float, default: float) -> list[float]:
    items = list(value)[:3] if isinstance(value, (list, tuple)) else []
    items += [default] * (3 - len(items))
    return [round(_clamp(v, lo, hi, default), 3) for v in items]


class Part(BaseModel):
    shape: Shape
    color: str = "#8a8a8a"
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    size: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    radius: float = 0.5
    height: float = 1.0
    tube: float = 0.1
    # For a "picture": one of Fuzzy's own gallery images, shown on the front of a thin panel
    # (size = width, height, depth); ``color`` is the frame.
    media_id: str | None = None

    @field_validator("media_id", mode="before")
    @classmethod
    def _media(cls, value: Any) -> str | None:
        text = str(value or "").strip().strip("[]")
        return text if _MEDIA_ID.match(text) else None

    @field_validator("color", mode="before")
    @classmethod
    def _color(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        if _COLOR.match(text):
            return text
        if re.match(r"^#[0-9a-f]{3}$", text):
            return "#" + "".join(ch * 2 for ch in text[1:])
        return _NAMED.get(text, "#8a8a8a")

    @field_validator("position", mode="before")
    @classmethod
    def _position(cls, value: Any) -> list[float]:
        return _vec3(value, -MAX_EXTENT, MAX_EXTENT, 0.0)

    @field_validator("rotation", mode="before")
    @classmethod
    def _rotation(cls, value: Any) -> list[float]:
        return _vec3(value, -360.0, 360.0, 0.0)

    @field_validator("size", mode="before")
    @classmethod
    def _size(cls, value: Any) -> list[float]:
        return _vec3(value, MIN_EXTENT, MAX_EXTENT, 1.0)

    @field_validator("radius", "height", mode="before")
    @classmethod
    def _extent(cls, value: Any) -> float:
        return round(_clamp(value, MIN_EXTENT, MAX_EXTENT, 0.5), 3)

    @field_validator("tube", mode="before")
    @classmethod
    def _tube(cls, value: Any) -> float:
        return round(_clamp(value, MIN_EXTENT, MAX_EXTENT / 4, 0.1), 3)


def clean_parts(raw: Any, max_parts: int) -> list[dict[str, Any]]:
    """Validate model-produced parts; invalid ones are dropped, values clamped."""

    parts = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            part = Part.model_validate(item)
        except ValidationError:
            continue
        if part.shape == "picture" and part.media_id is None:
            continue  # a picture needs a picture
        if part.shape != "picture":
            part.media_id = None
        parts.append(part.model_dump(exclude_none=True))
        if len(parts) >= max_parts:
            break
    return parts


def clean_text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


class Avatar(BaseModel):
    description: str = ""
    parts: list[dict[str, Any]] = Field(default_factory=list)
    designed_at: datetime | None = None
    versions: int = 0


Action = Literal["standing", "walking", "building", "thinking", "resting", "talking", "creating"]


class Body(BaseModel):
    """Where Fuzzy is in its world and what it's doing there."""

    x: float = 0.0
    z: float = 0.0
    heading: float = 0.0  # degrees, 0 = facing +z
    target: list[float] | None = None
    action: Action = "standing"
    thought: str = ""
    updated_at: datetime = Field(default_factory=utcnow)
