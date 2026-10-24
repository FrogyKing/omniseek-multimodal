import hashlib
import json
import logging
import threading
from pathlib import Path

import numpy as np

from .cloud import ModalEmbedder
from .config import MODEL_REVISION, Settings
from .media import modality_for, prepare, safe_name
from .storage import Store, new_id
from .vectors import search

logger = logging.getLogger(__name__)


class Service:
    def __init__(self, settings: Settings, embedder=None):
        self.settings = settings
        self.store = Store(settings.data_dir / "omniseek.sqlite3")
        self.upload_dir = settings.data_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or ModalEmbedder(settings)
        self.job_lock = threading.Lock()

    def file_path(self, file_id: str) -> Path:
        rows = self.store.rows("SELECT name FROM files WHERE id=?", (file_id,))
        if not rows:
            raise KeyError("File not found")
        return self.upload_dir / (file_id + Path(rows[0]["name"]).suffix.lower())

    def upload(self, collection_id: str, filename: str, content: bytes) -> dict:
        if not self.store.rows("SELECT id FROM collections WHERE id=?", (collection_id,)):
            raise KeyError("Collection not found")
        if not content or len(content) > self.settings.max_upload_bytes:
            raise ValueError("Empty file or file exceeds 20 MiB upload limit")
        filename = safe_name(filename)
        modality = modality_for(filename)
        digest = hashlib.sha256(content).hexdigest()
        duplicate = self.store.rows(
            "SELECT id FROM files WHERE collection_id=? AND sha256=?", (collection_id, digest)
        )
        if duplicate:
            return {"id": duplicate[0]["id"], "duplicate": True}
        if len(self.store.files(collection_id)) >= self.settings.max_files:
            raise ValueError("Collection file limit reached")
        key = new_id()
        path = self.upload_dir / (key + Path(filename).suffix.lower())
        path.write_bytes(content)
        try:
            metadata, segments, _ = prepare(path, filename, self.settings)
            metadata["segments"] = len(segments)
            self.store.execute(
                "INSERT INTO files (id,collection_id,name,modality,sha256,size,metadata) "
                "VALUES (?,?,?,?,?,?,?)",
                (key, collection_id, filename, modality, digest, len(content), json.dumps(metadata)),
            )
            return {"id": key, "duplicate": False, "name": filename, "modality": modality}
        except Exception as exc:
            path.unlink(missing_ok=True)
            if isinstance(exc, ValueError):
                raise
            raise ValueError("File is corrupt or cannot be decoded") from exc

    def index(self, job_id: str, collection_id: str, rebuild: bool = False) -> None:
        if not self.job_lock.acquire(blocking=False):
            self.store.execute(
                "UPDATE jobs SET status='failed',message=? WHERE id=?",
                ("Another indexing job is running", job_id),
            )
            return
        try:
            self.store.execute("UPDATE jobs SET status='running' WHERE id=?", (job_id,))
            files = [f for f in self.store.files(collection_id) if rebuild or f["status"] != "indexed"]
            segments_by_file = []
            payloads = []
            for file in files:
                _, segments, prepared = prepare(self.file_path(file["id"]), file["name"], self.settings)
                segments_by_file.append((file["id"], segments))
                payloads.extend(prepared)
            if len(payloads) > self.settings.max_segments:
                raise ValueError("Job exceeds 128 segments; use smaller collections")
            if payloads:
                self.store.execute(
                    "UPDATE jobs SET progress=0.2,message='Embedding on Modal' WHERE id=?", (job_id,)
                )
                vectors, measurements = self.embedder.embed(payloads, label="index collection")
                offset = 0
                for file_id, segments in segments_by_file:
                    self.store.replace_segments(
                        file_id,
                        segments,
                        vectors[offset : offset + len(segments)],
                        measurements.get("config", "full"),
                    )
                    offset += len(segments)
            self.store.execute(
                "UPDATE jobs SET status='completed',progress=1,message=? WHERE id=?",
                (f"Indexed {len(payloads)} segments", job_id),
            )
        except Exception as exc:
            logger.warning("Indexing failed (%s)", type(exc).__name__)
            message = (
                str(exc)
                if isinstance(exc, (ValueError, RuntimeError))
                else "GPU service unavailable; try again later"
            )
            self.store.execute("UPDATE jobs SET status='failed',message=? WHERE id=?", (message, job_id))
        finally:
            self.job_lock.release()

    def search(self, request) -> dict:
        vector: np.ndarray | None
        if not request.query.strip() and not request.image_file_id:
            raise ValueError("Enter a query or select an image")
        if request.image_file_id:
            files = self.store.rows("SELECT modality FROM files WHERE id=?", (request.image_file_id,))
            if not files or files[0]["modality"] != "image":
                raise ValueError("Image query must reference an uploaded image")
            cached = self.store.rows(
                "SELECT vector FROM segments WHERE file_id=? AND revision=? LIMIT 1",
                (request.image_file_id, MODEL_REVISION),
            )
            if cached:
                vector = np.frombuffer(cached[0]["vector"], dtype=np.float32).copy()
            else:
                _, _, payloads = prepare(self.file_path(request.image_file_id), "query.png", self.settings)
                vectors, _ = self.embedder.embed(payloads, label="image query")
                vector = vectors[0]
            cache_hit = bool(cached)
        else:
            prompt = "CodeRetrieval" if request.modality == "code" else "SearchQuery"
            vector = self.store.cached_query(request.query, prompt)
            cache_hit = vector is not None
            if vector is None:
                vectors, _ = self.embedder.embed(
                    [{"kind": "text", "text": request.query.strip(), "prompt": prompt}], label="search query"
                )
                vector = vectors[0]
                self.store.cache_query(request.query, vector, prompt)
        rows, vectors = self.store.corpus(request.collection_id, request.modality)
        import time

        start = time.perf_counter()
        results = []
        if rows:
            scores, indices = search(vectors, vector, request.dimension, request.k)
            for score, i in zip(scores, indices, strict=True):
                row = rows[int(i)]
                row["score"] = float(np.clip(score, -1, 1))
                row["url"] = f"/api/files/{row['file_id']}/content"
                results.append(row)
        key = new_id()
        response = {
            "id": key,
            "results": results,
            "dimension": request.dimension,
            "cache_hit": cache_hit,
            "retrieval_ms": (time.perf_counter() - start) * 1000,
            "total_segments": len(rows),
        }
        self.store.execute("INSERT INTO searches VALUES (?,?)", (key, json.dumps(response)))
        # Bound retained search history.
        self.store.execute(
            "DELETE FROM searches WHERE rowid NOT IN (SELECT rowid FROM searches ORDER BY rowid DESC LIMIT 100)"
        )
        return response
