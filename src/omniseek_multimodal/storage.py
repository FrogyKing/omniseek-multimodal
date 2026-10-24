"""SQLite stores canonical float32 vectors and metadata in one transaction."""

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .config import INDEX_VERSION, MODEL_REVISION
from .vectors import normalize


def new_id() -> str:
    return uuid.uuid4().hex


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS collections (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
                    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY, collection_id TEXT NOT NULL REFERENCES collections(id)
                    ON DELETE CASCADE, name TEXT NOT NULL, modality TEXT NOT NULL,
                    sha256 TEXT NOT NULL, size INTEGER NOT NULL, metadata TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'uploaded', UNIQUE(collection_id,sha256));
                CREATE TABLE IF NOT EXISTS segments (
                    id TEXT PRIMARY KEY, file_id TEXT NOT NULL REFERENCES files(id)
                    ON DELETE CASCADE, metadata TEXT NOT NULL, vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL, revision TEXT NOT NULL,
                    index_version INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, collection_id TEXT NOT NULL REFERENCES collections(id)
                    ON DELETE CASCADE, status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '', created_at TEXT DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
                CREATE TABLE IF NOT EXISTS query_cache (
                    key TEXT PRIMARY KEY, vector BLOB NOT NULL, revision TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS searches (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def rows(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with self.connection() as db:
            return [dict(row) for row in db.execute(sql, args)]

    def execute(self, sql: str, args: tuple = ()) -> None:
        with self.connection() as db:
            db.execute(sql, args)

    def collections(self) -> list[dict[str, Any]]:
        return self.rows("""SELECT c.*, (SELECT count(*) FROM files f WHERE f.collection_id=c.id)
            AS file_count, (SELECT count(*) FROM segments s JOIN files f ON f.id=s.file_id
            WHERE f.collection_id=c.id) AS segment_count FROM collections c ORDER BY c.created_at""")

    def files(self, collection_id: str | None = None) -> list[dict[str, Any]]:
        rows = self.rows(
            "SELECT * FROM files"
            + (" WHERE collection_id=?" if collection_id else "")
            + " ORDER BY rowid DESC",
            (collection_id,) if collection_id else (),
        )
        for row in rows:
            row["metadata"] = json.loads(row["metadata"])
        return rows

    def replace_segments(
        self, file_id: str, segments: list[dict], vectors: np.ndarray, encoder_config: str = "full"
    ) -> None:
        vectors = normalize(vectors)
        if len(segments) != len(vectors):
            raise ValueError("Segment/vector count mismatch")
        with self.connection() as db:
            db.execute("DELETE FROM segments WHERE file_id=?", (file_id,))
            for segment, vector in zip(segments, vectors, strict=True):
                db.execute(
                    "INSERT INTO segments VALUES (?,?,?,?,?,?,?)",
                    (
                        segment.get("id", new_id()),
                        file_id,
                        json.dumps({**segment, "encoder_config": encoder_config}),
                        vector.tobytes(),
                        768,
                        MODEL_REVISION,
                        INDEX_VERSION,
                    ),
                )
            db.execute("UPDATE files SET status='indexed' WHERE id=?", (file_id,))

    def corpus(self, collection_id: str | None = None, modality: str | None = None):
        sql = """SELECT s.*, f.name, f.modality, f.collection_id, f.metadata AS file_metadata
                 FROM segments s JOIN files f ON f.id=s.file_id WHERE 1=1"""
        args = []
        if collection_id:
            sql += " AND f.collection_id=?"
            args.append(collection_id)
        if modality:
            sql += " AND f.modality=?"
            args.append(modality)
        rows = self.rows(sql + " ORDER BY s.id", tuple(args))
        for row in rows:
            if (
                row["revision"] != MODEL_REVISION
                or row["dimension"] != 768
                or row["index_version"] != INDEX_VERSION
            ):
                raise ValueError("Incompatible model revision or index version; rebuild required")
        vectors = np.array(
            [np.frombuffer(row.pop("vector"), dtype=np.float32) for row in rows], dtype=np.float32
        ).reshape(-1, 768)
        for row in rows:
            row["metadata"] = json.loads(row["metadata"])
            row["file_metadata"] = json.loads(row["file_metadata"])
        return rows, vectors

    @staticmethod
    def cache_key(text: str, prompt: str = "SearchQuery") -> str:
        return hashlib.sha256(f"{MODEL_REVISION}:{prompt}:{text.strip()}".encode()).hexdigest()

    def cache_query(self, text: str, vector: np.ndarray, prompt: str = "SearchQuery") -> None:
        self.execute(
            "INSERT OR REPLACE INTO query_cache VALUES (?,?,?)",
            (self.cache_key(text, prompt), normalize(vector).tobytes(), MODEL_REVISION),
        )

    def cached_query(self, text: str, prompt: str = "SearchQuery") -> np.ndarray | None:
        rows = self.rows(
            "SELECT vector FROM query_cache WHERE key=? AND revision=?",
            (self.cache_key(text, prompt), MODEL_REVISION),
        )
        return np.frombuffer(rows[0]["vector"], dtype=np.float32).copy() if rows else None
