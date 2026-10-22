"""Assemble the memory system from configuration."""

from __future__ import annotations

from sentientai.config import Settings
from sentientai.core.clock import Clock
from sentientai.core.events import EventBus
from sentientai.core.metrics import Metrics
from sentientai.memory.system import MemorySystem
from sentientai.memory.vector import (
    InMemoryVectorStore,
    PgVectorStore,
    QdrantVectorStore,
    VectorStore,
)
from sentientai.memory.working import build_working_memory
from sentientai.models.embeddings import Embedder, build_embedder
from sentientai.storage.database import Database


def build_vector_store(settings: Settings, db: Database, dimensions: int) -> VectorStore:
    storage = settings.storage
    if storage.vector_backend == "pgvector":
        return PgVectorStore(db.engine, dimensions)
    if storage.vector_backend == "qdrant":
        return QdrantVectorStore(
            dimensions,
            collection=storage.qdrant_collection,
            url=storage.qdrant_url,
            path=storage.qdrant_path,
        )
    return InMemoryVectorStore(dimensions, storage.data_dir / "vectors.npz")


def build_memory(
    settings: Settings,
    db: Database,
    *,
    bus: EventBus | None = None,
    metrics: Metrics | None = None,
    embedder: Embedder | None = None,
    clock: Clock | None = None,
) -> MemorySystem:
    embedder = embedder or build_embedder(settings.models.embedding)
    return MemorySystem(
        db,
        build_vector_store(settings, db, embedder.dimensions),
        embedder,
        build_working_memory(
            settings.storage.redis_url,
            max_turns=settings.memory.working_memory_turns,
            ttl_s=settings.memory.working_memory_ttl_s,
        ),
        settings.memory,
        bus=bus,
        metrics=metrics,
        clock=clock,
        query_prefix=settings.models.embedding.query_prefix,
        document_prefix=settings.models.embedding.document_prefix,
    )
