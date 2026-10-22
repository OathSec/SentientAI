"""Vector indexes for semantic recall.

The relational store is the source of truth; a vector index only maps memory
ids to embeddings, so any backend can be rebuilt from SQL (``sentientai
memory reindex``).

* :class:`InMemoryVectorStore` - numpy, persisted to ``.npz`` (default, zero infra)
* :class:`PgVectorStore` - PostgreSQL + pgvector with an HNSW cosine index
* :class:`QdrantVectorStore` - Qdrant server, embedded file mode, or ``:memory:``
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncEngine

from sentientai.errors import ConfigError, StorageError
from sentientai.storage.records import GLOBAL_SCOPE


@dataclass(frozen=True)
class VectorItem:
    id: str
    vector: list[float]
    kind: str
    scope: str = GLOBAL_SCOPE  # "global" or "u:<user id>" (private to one person)


@dataclass(frozen=True)
class VectorHit:
    id: str
    score: float  # cosine similarity in [-1, 1]


@runtime_checkable
class VectorStore(Protocol):
    dimensions: int

    async def open(self) -> None: ...

    async def upsert(self, items: Sequence[VectorItem]) -> None: ...

    async def search(
        self,
        vector: list[float],
        k: int,
        kinds: Sequence[str] | None = None,
        scopes: Sequence[str] | None = None,
    ) -> list[VectorHit]: ...

    async def delete(self, ids: Sequence[str]) -> None: ...

    async def count(self) -> int: ...

    async def flush(self) -> None: ...

    async def aclose(self) -> None: ...


class InMemoryVectorStore:
    def __init__(self, dimensions: int, path: Path | None = None) -> None:
        self.dimensions = dimensions
        self.path = path
        self._ids: list[str] = []
        self._kinds: list[str] = []
        self._scopes: list[str] = []
        self._index: dict[str, int] = {}
        self._matrix = np.zeros((0, dimensions), dtype=np.float32)
        self._dirty = False
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        if self.path is None or not self.path.exists():
            return
        data = np.load(self.path, allow_pickle=False)
        matrix = data["matrix"].astype(np.float32)
        if matrix.shape[1] != self.dimensions and matrix.shape[0] > 0:
            raise ConfigError(
                f"vector index {self.path} has {matrix.shape[1]} dims, expected {self.dimensions}; "
                "run `sentientai memory reindex` after changing embedding models"
            )
        self._ids = [str(x) for x in data["ids"]]
        self._kinds = [str(x) for x in data["kinds"]]
        self._scopes = (
            [str(x) for x in data["scopes"]]
            if "scopes" in data.files
            else [GLOBAL_SCOPE] * len(self._ids)  # index written before scoping existed
        )
        self._matrix = matrix.reshape(len(self._ids), self.dimensions)
        self._index = {id_: i for i, id_ in enumerate(self._ids)}

    async def upsert(self, items: Sequence[VectorItem]) -> None:
        async with self._lock:
            new_rows: list[np.ndarray] = []
            for item in items:
                vec = np.asarray(item.vector, dtype=np.float32)
                if vec.shape != (self.dimensions,):
                    raise ConfigError(
                        f"vector has shape {vec.shape}, expected ({self.dimensions},)"
                    )
                if item.id in self._index:
                    i = self._index[item.id]
                    self._matrix[i] = vec
                    self._kinds[i] = item.kind
                    self._scopes[i] = item.scope
                else:
                    self._index[item.id] = len(self._ids) + len(new_rows)
                    new_rows.append(vec)
                    self._ids.append(item.id)
                    self._kinds.append(item.kind)
                    self._scopes.append(item.scope)
            if new_rows:
                self._matrix = np.vstack([self._matrix, np.stack(new_rows)])
            self._dirty = True

    async def search(
        self,
        vector: list[float],
        k: int,
        kinds: Sequence[str] | None = None,
        scopes: Sequence[str] | None = None,
    ) -> list[VectorHit]:
        if not self._ids or k <= 0:
            return []
        query = np.asarray(vector, dtype=np.float32)
        scores = self._matrix @ query
        if kinds is not None:
            allowed = set(kinds)
            mask = np.fromiter(
                (kind in allowed for kind in self._kinds), dtype=bool, count=len(self._kinds)
            )
            scores = np.where(mask, scores, -np.inf)
        if scopes is not None:
            visible = set(scopes)
            mask = np.fromiter(
                (scope in visible for scope in self._scopes), dtype=bool, count=len(self._scopes)
            )
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(self._ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [VectorHit(self._ids[i], float(scores[i])) for i in top if np.isfinite(scores[i])]

    async def delete(self, ids: Sequence[str]) -> None:
        async with self._lock:
            remove = {self._index[i] for i in ids if i in self._index}
            if not remove:
                return
            keep = [i for i in range(len(self._ids)) if i not in remove]
            self._matrix = self._matrix[keep]
            self._ids = [self._ids[i] for i in keep]
            self._kinds = [self._kinds[i] for i in keep]
            self._scopes = [self._scopes[i] for i in keep]
            self._index = {id_: i for i, id_ in enumerate(self._ids)}
            self._dirty = True

    async def count(self) -> int:
        return len(self._ids)

    async def flush(self) -> None:
        if self.path is None or not self._dirty:
            return
        async with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp.npz")
            np.savez(
                tmp,
                matrix=self._matrix,
                ids=np.array(self._ids, dtype=str),
                kinds=np.array(self._kinds, dtype=str),
                scopes=np.array(self._scopes, dtype=str),
            )
            os.replace(tmp, self.path)
            self._dirty = False

    async def aclose(self) -> None:
        await self.flush()


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


class PgVectorStore:
    TABLE = "memory_vectors"

    def __init__(self, engine: AsyncEngine, dimensions: int) -> None:
        if engine.dialect.name != "postgresql":
            raise ConfigError("pgvector backend requires storage.database_url to be PostgreSQL")
        self.engine = engine
        self.dimensions = dimensions

    async def open(self) -> None:
        async with self.engine.begin() as conn:
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except Exception as exc:
                raise StorageError(
                    "pgvector extension unavailable; install it (e.g. the pgvector/pgvector image) "
                    f"or choose another vector backend: {exc}"
                ) from exc
            await conn.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS {self.TABLE} ("
                    "id TEXT PRIMARY KEY, kind TEXT NOT NULL, "
                    f"scope TEXT NOT NULL DEFAULT '{GLOBAL_SCOPE}', "
                    f"embedding vector({self.dimensions}) NOT NULL, "
                    "updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"
                )
            )
            await conn.execute(
                text(
                    f"ALTER TABLE {self.TABLE} ADD COLUMN IF NOT EXISTS "
                    f"scope TEXT NOT NULL DEFAULT '{GLOBAL_SCOPE}'"
                )
            )
            dims = (
                await conn.execute(
                    text(
                        "SELECT atttypmod FROM pg_attribute "
                        f"WHERE attrelid = '{self.TABLE}'::regclass AND attname = 'embedding'"
                    )
                )
            ).scalar_one()
            if int(dims) != self.dimensions:
                raise ConfigError(
                    f"{self.TABLE}.embedding has {dims} dims, expected {self.dimensions}; "
                    "run `sentientai memory reindex` after changing embedding models"
                )
            await conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {self.TABLE}_hnsw ON {self.TABLE} "
                    "USING hnsw (embedding vector_cosine_ops)"
                )
            )
            await conn.execute(
                text(f"CREATE INDEX IF NOT EXISTS {self.TABLE}_kind ON {self.TABLE} (kind)")
            )
            await conn.execute(
                text(f"CREATE INDEX IF NOT EXISTS {self.TABLE}_scope ON {self.TABLE} (scope)")
            )

    async def upsert(self, items: Sequence[VectorItem]) -> None:
        if not items:
            return
        stmt = text(
            f"INSERT INTO {self.TABLE} (id, kind, scope, embedding, updated_at) "
            "VALUES (:id, :kind, :scope, CAST(:embedding AS vector), now()) "
            "ON CONFLICT (id) DO UPDATE SET kind = EXCLUDED.kind, scope = EXCLUDED.scope, "
            "embedding = EXCLUDED.embedding, updated_at = now()"
        )
        async with self.engine.begin() as conn:
            await conn.execute(
                stmt,
                [
                    {
                        "id": i.id,
                        "kind": i.kind,
                        "scope": i.scope,
                        "embedding": _vector_literal(i.vector),
                    }
                    for i in items
                ],
            )

    async def search(
        self,
        vector: list[float],
        k: int,
        kinds: Sequence[str] | None = None,
        scopes: Sequence[str] | None = None,
    ) -> list[VectorHit]:
        if (kinds is not None and not kinds) or (scopes is not None and not scopes):
            return []
        conditions = []
        if kinds is not None:
            conditions.append("kind IN :kinds")
        if scopes is not None:
            conditions.append("scope IN :scopes")
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        stmt = text(
            f"SELECT id, 1 - (embedding <=> CAST(:q AS vector)) AS score FROM {self.TABLE} {where} "
            "ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k"
        )
        params: dict[str, Any] = {"q": _vector_literal(vector), "k": k}
        if kinds is not None:
            stmt = stmt.bindparams(bindparam("kinds", expanding=True))
            params["kinds"] = list(kinds)
        if scopes is not None:
            stmt = stmt.bindparams(bindparam("scopes", expanding=True))
            params["scopes"] = list(scopes)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt, params)).fetchall()
        return [VectorHit(str(r[0]), float(r[1])) for r in rows]

    async def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        stmt = text(f"DELETE FROM {self.TABLE} WHERE id IN :ids").bindparams(
            bindparam("ids", expanding=True)
        )
        async with self.engine.begin() as conn:
            await conn.execute(stmt, {"ids": list(ids)})

    async def count(self) -> int:
        async with self.engine.connect() as conn:
            return int(
                (await conn.execute(text(f"SELECT count(*) FROM {self.TABLE}"))).scalar_one()
            )

    async def flush(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


_QDRANT_NS = uuid.UUID("6f1c3a52-6a53-4f53-9c2b-5e3f7d1a9b10")


class QdrantVectorStore:
    def __init__(
        self,
        dimensions: int,
        *,
        collection: str,
        url: str | None = None,
        path: Path | None = None,
    ) -> None:
        from qdrant_client import AsyncQdrantClient

        self.dimensions = dimensions
        self.collection = collection
        self._remote = bool(url)
        if url:
            self.client = AsyncQdrantClient(url=url)
        elif path:
            self.client = AsyncQdrantClient(path=str(path))
        else:
            self.client = AsyncQdrantClient(location=":memory:")

    @staticmethod
    def _point_id(memory_id: str) -> str:
        return str(uuid.uuid5(_QDRANT_NS, memory_id))

    async def open(self) -> None:
        from qdrant_client import models

        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(
                    size=self.dimensions, distance=models.Distance.COSINE
                ),
            )
            if self._remote:  # payload indexes only exist on a Qdrant server
                for field in ("kind", "scope"):
                    await self.client.create_payload_index(
                        self.collection, field, field_schema=models.PayloadSchemaType.KEYWORD
                    )
        else:
            info = await self.client.get_collection(self.collection)
            params = info.config.params.vectors
            size = getattr(params, "size", None)
            if size is not None and int(size) != self.dimensions:
                raise ConfigError(
                    f"qdrant collection {self.collection} has {size} dims, "
                    f"expected {self.dimensions}"
                )

    async def upsert(self, items: Sequence[VectorItem]) -> None:
        from qdrant_client import models

        if not items:
            return
        points = [
            models.PointStruct(
                id=self._point_id(i.id),
                vector=list(i.vector),
                payload={"mid": i.id, "kind": i.kind, "scope": i.scope},
            )
            for i in items
        ]
        await self.client.upsert(self.collection, points=points, wait=True)

    async def search(
        self,
        vector: list[float],
        k: int,
        kinds: Sequence[str] | None = None,
        scopes: Sequence[str] | None = None,
    ) -> list[VectorHit]:
        from qdrant_client import models

        if (kinds is not None and not kinds) or (scopes is not None and not scopes):
            return []
        must: list[Any] = []
        if kinds is not None:
            must.append(models.FieldCondition(key="kind", match=models.MatchAny(any=list(kinds))))
        if scopes is not None:
            must.append(models.FieldCondition(key="scope", match=models.MatchAny(any=list(scopes))))
        query_filter = models.Filter(must=must) if must else None
        result = await self.client.query_points(
            self.collection,
            query=list(vector),
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [VectorHit(str((p.payload or {})["mid"]), float(p.score)) for p in result.points]

    async def delete(self, ids: Sequence[str]) -> None:
        from qdrant_client import models

        if not ids:
            return
        await self.client.delete(
            self.collection,
            points_selector=models.PointIdsList(points=[self._point_id(i) for i in ids]),
            wait=True,
        )

    async def count(self) -> int:
        return int((await self.client.count(self.collection, exact=True)).count)

    async def flush(self) -> None:
        return None

    async def aclose(self) -> None:
        await self.client.close()
