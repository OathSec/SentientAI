"""Tools for Fuzzy's world and for making images."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from sentientai.accounts.quotas import QuotaExceeded, Quotas
from sentientai.media.images import ImageError, ImageService
from sentientai.tools.base import Tool, ToolArgs, ToolContext, ToolResult
from sentientai.world.service import World, WorldError

_PARTS_HELP = (
    "List of shapes. Each: {shape: box|sphere|cylinder|cone|torus, color: '#rrggbb', "
    "position: [x,y,z] relative to the object's centre on the ground (y up), rotation: [rx,ry,rz] "
    "degrees, and size: [w,h,d] for box, radius for sphere, radius+height for cylinder/cone, "
    "radius+tube for torus}. Metres."
)


def _who(ctx: ToolContext) -> str:
    return f"user:{ctx.user_id}" if ctx.user_id else "agent"


class LookArgs(ToolArgs):
    pass


class WorldLook(Tool[LookArgs]):
    name = "world_look"
    description = "Look around your world: what you've built, where you are, how you look."
    args_model = LookArgs
    tags = ("world", "read")

    def __init__(self, world: World) -> None:
        self.world = world

    async def run(self, args: LookArgs, ctx: ToolContext) -> ToolResult:
        body = self.world.body
        return ToolResult.success(
            f"Your world is {self.world.size:.0f} m across (it can grow to "
            f"{self.world.config.max_size:.0f} m). You are at "
            f"({body.x:.1f}, {body.z:.1f}), {body.action}.\n{self.world.describe_avatar()}\n"
            f"Things here:\n{await self.world.describe()}"
        )


class BuildArgs(ToolArgs):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    x: float = 0.0
    z: float = 0.0
    rotation: float = 0.0
    parts: list[dict[str, Any]] = Field(min_length=1, description=_PARTS_HELP)


class WorldBuild(Tool[BuildArgs]):
    name = "world_build"
    description = (
        "Build something in your world from simple shapes. It's your home, and what goes in it "
        "is your own choice."
    )
    args_model = BuildArgs
    autonomous_only = True  # Fuzzy's world and body are Fuzzy's own choices
    tags = ("world",)

    def __init__(self, world: World) -> None:
        self.world = world

    async def run(self, args: BuildArgs, ctx: ToolContext) -> ToolResult:
        try:
            obj = await self.world.build(
                args.name,
                args.parts,
                x=args.x,
                z=args.z,
                rotation=args.rotation,
                description=args.description,
                by=_who(ctx),
            )
        except WorldError as exc:
            return ToolResult.failure(str(exc))
        return ToolResult.success(
            f"Built {obj.name} [{obj.id}] with {len(obj.parts)} parts at ({obj.x}, {obj.z})."
        )


class ModifyArgs(ToolArgs):
    object_id: str
    name: str | None = Field(default=None, max_length=80)
    description: str | None = Field(default=None, max_length=500)
    x: float | None = None
    z: float | None = None
    rotation: float | None = None
    parts: list[dict[str, Any]] | None = Field(default=None, description=_PARTS_HELP)


class WorldModify(Tool[ModifyArgs]):
    name = "world_modify"
    description = "Change, move or rebuild something in your world."
    args_model = ModifyArgs
    autonomous_only = True  # Fuzzy's world and body are Fuzzy's own choices
    tags = ("world",)

    def __init__(self, world: World) -> None:
        self.world = world

    async def run(self, args: ModifyArgs, ctx: ToolContext) -> ToolResult:
        try:
            obj = await self.world.modify(
                args.object_id,
                name=args.name,
                description=args.description,
                parts=args.parts,
                x=args.x,
                z=args.z,
                rotation=args.rotation,
                by=_who(ctx),
            )
        except WorldError as exc:
            return ToolResult.failure(str(exc))
        return ToolResult.success(f"Changed {obj.name}.")


class RemoveArgs(ToolArgs):
    object_id: str


class WorldRemove(Tool[RemoveArgs]):
    name = "world_remove"
    description = "Remove something from your world."
    args_model = RemoveArgs
    autonomous_only = True  # Fuzzy's world and body are Fuzzy's own choices
    tags = ("world",)

    def __init__(self, world: World) -> None:
        self.world = world

    async def run(self, args: RemoveArgs, ctx: ToolContext) -> ToolResult:
        try:
            obj = await self.world.remove(args.object_id, by=_who(ctx))
        except WorldError as exc:
            return ToolResult.failure(str(exc))
        return ToolResult.success(f"Removed {obj.name}.")


class WalkArgs(ToolArgs):
    x: float
    z: float


class WorldWalk(Tool[WalkArgs]):
    name = "world_walk"
    description = "Walk somewhere in your world."
    args_model = WalkArgs
    autonomous_only = True  # Fuzzy's world and body are Fuzzy's own choices
    tags = ("world",)

    def __init__(self, world: World) -> None:
        self.world = world

    async def run(self, args: WalkArgs, ctx: ToolContext) -> ToolResult:
        x, z = await self.world.walk_to(args.x, args.z)
        return ToolResult.success(f"Walking to ({x}, {z}).")


class ExpandArgs(ToolArgs):
    size: float = Field(gt=0, description="New side length of the square ground, in metres.")


class WorldExpand(Tool[ExpandArgs]):
    name = "world_expand"
    description = "Make the ground of your world bigger, to make room."
    args_model = ExpandArgs
    autonomous_only = True  # Fuzzy's world and body are Fuzzy's own choices
    tags = ("world",)

    def __init__(self, world: World) -> None:
        self.world = world

    async def run(self, args: ExpandArgs, ctx: ToolContext) -> ToolResult:
        try:
            size = await self.world.expand(args.size, by=_who(ctx))
        except WorldError as exc:
            return ToolResult.failure(str(exc))
        return ToolResult.success(f"Your world is now {size:.0f} m across.")


class AvatarArgs(ToolArgs):
    description: str = Field(min_length=3, max_length=600, description="How you look, and why.")
    parts: list[dict[str, Any]] = Field(
        min_length=1, description=_PARTS_HELP + " Relative to your feet; about 1-2 m tall."
    )


class AvatarDesign(Tool[AvatarArgs]):
    name = "avatar_design"
    description = (
        "Change how YOU look in your 3D world (your own body). This is not a picture and not a "
        "gift for anyone: don't use it to answer a request to draw or make an image."
    )
    args_model = AvatarArgs
    autonomous_only = True  # Fuzzy's world and body are Fuzzy's own choices
    tags = ("world",)

    def __init__(self, world: World) -> None:
        self.world = world

    async def run(self, args: AvatarArgs, ctx: ToolContext) -> ToolResult:
        try:
            avatar = await self.world.design_avatar(args.description, args.parts, by=_who(ctx))
        except WorldError as exc:
            return ToolResult.failure(str(exc))
        return ToolResult.success(
            f"Done: your own body in your world now looks like that (version {avatar.versions}). "
            "No picture was made."
        )


class ImageArgs(ToolArgs):
    prompt: str = Field(min_length=3, max_length=1000, description="What the image shows.")
    caption: str = Field(default="", max_length=500, description="A line about it, from you.")


class ImageGenerate(Tool[ImageArgs]):
    name = "image_generate"
    description = (
        "Make an image. In a conversation it's made for (and private to) the person you're "
        "talking with; on your own it goes in your public gallery."
    )
    args_model = ImageArgs
    timeout_s = 600.0
    tags = ("media",)
    unavailable_note = (
        "make pictures or images of any kind: no image generator is set up on this server yet"
    )

    def __init__(self, images: ImageService, quotas: Quotas) -> None:
        self.images = images
        self.quotas = quotas

    def available(self) -> bool:
        return self.images.available  # no image server, no tool: Fuzzy knows it can't draw

    async def run(self, args: ImageArgs, ctx: ToolContext) -> ToolResult:
        if not self.images.available:
            return ToolResult.failure("image generation isn't set up on this server")
        try:
            await self.quotas.take(ctx.user_id, images=1, privileged=ctx.privileged)
        except QuotaExceeded as exc:
            return ToolResult.failure(str(exc))
        try:
            record = await self.images.create(
                args.prompt,
                user_id=ctx.user_id,
                public=ctx.user_id is None,
                caption=args.caption,
                by=_who(ctx),
            )
        except ImageError as exc:
            await self.quotas.take(ctx.user_id, images=-1, privileged=True)
            return ToolResult.failure(str(exc))
        return ToolResult.success(
            f"Made the image [{record.id}]."
            + (" It will be shown with your reply." if ctx.user_id else " It's in your gallery."),
            data={"media_id": record.id},
        )


def world_tools(world: World) -> list[Tool[Any]]:
    return [
        WorldLook(world),
        WorldBuild(world),
        WorldModify(world),
        WorldRemove(world),
        WorldWalk(world),
        WorldExpand(world),
        AvatarDesign(world),
    ]


def media_tools(images: ImageService, quotas: Quotas) -> list[Tool[Any]]:
    return [ImageGenerate(images, quotas)]
