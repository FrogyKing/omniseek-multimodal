"""Local-only API; serve UI and originals from the same origin."""

import json
import mimetypes
import threading
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .cloud import CloudDisabled
from .config import MODEL_ID, MODEL_REVISION, ROOT, Settings
from .costs import Ledger
from .middleware import BodyLimitMiddleware
from .schemas import CollectionCreate, IndexRequest, SearchRequest
from .service import Service
from .storage import new_id


def create_app(settings: Settings | None = None, embedder=None) -> FastAPI:
    settings = settings or Settings()
    service = Service(settings, embedder)
    store = service.store
    write_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        store.execute(
            "UPDATE jobs SET status='failed',message='Server restarted; retry indexing' "
            "WHERE status IN ('queued','running')"
        )
        yield

    app = FastAPI(title="OmniSeek", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_upload_bytes + 1024 * 1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        # Prevent drive-by browser pages from spending cloud credits through localhost.
        allowed = {"localhost", "127.0.0.1", "testserver"}
        if request.url.hostname not in allowed:
            return JSONResponse({"error": "Unrecognized host"}, status_code=403)
        if request.method in ("POST", "DELETE", "PUT", "PATCH"):
            origin = request.headers.get("origin")
            if origin and origin not in {f"http://{h}:{p}" for h in allowed for p in (5173, 8000)}:
                return JSONResponse({"error": "Unrecognized request origin"}, status_code=403)
            length = request.headers.get("content-length")
            if length and (not length.isdigit() or int(length) > settings.max_upload_bytes + 1024 * 1024):
                return JSONResponse({"error": "Request exceeds upload limit"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=422)

    @app.exception_handler(KeyError)
    async def key_error(request, exc):
        return JSONResponse({"error": str(exc).strip("'")}, status_code=404)

    @app.exception_handler(CloudDisabled)
    async def cloud_disabled(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=409)

    @app.exception_handler(RuntimeError)
    async def runtime_error(request, exc):
        return JSONResponse(
            {
                "error": "Embedding service unavailable or cloud budget exhausted. "
                "Check local cost-report and Modal status."
            },
            status_code=503,
        )

    @app.get("/health")
    @app.get("/api/health")
    def health():
        return {"status": "ok", "model": MODEL_ID}

    @app.get("/api/collections")
    def collections():
        return store.collections()

    @app.post("/api/collections", status_code=201)
    def create_collection(body: CollectionCreate):
        key = new_id()
        store.execute(
            "INSERT INTO collections (id,name,description) VALUES (?,?,?)", (key, body.name, body.description)
        )
        return {"id": key, **body.model_dump()}

    @app.delete("/api/collections/{key}", status_code=204)
    def delete_collection(key: str):
        with write_lock:
            if service.job_lock.locked():
                raise HTTPException(409, "Wait for the active indexing job")
            if not store.rows("SELECT id FROM collections WHERE id=?", (key,)):
                raise KeyError("Collection not found")
            paths = [service.file_path(f["id"]) for f in store.files(key)]
            store.execute("DELETE FROM collections WHERE id=?", (key,))
            for path in paths:
                path.unlink(missing_ok=True)

    @app.post("/api/uploads", status_code=201)
    def upload(collection_id: str = Form(...), file: UploadFile = File(...)):
        content = file.file.read(settings.max_upload_bytes + 1)
        with write_lock:
            return service.upload(collection_id, file.filename or "", content)

    @app.get("/api/files")
    def files(
        collection_id: str | None = None, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)
    ):
        rows = store.files(collection_id)
        return {"items": rows[offset : offset + limit], "total": len(rows)}

    @app.get("/api/files/{key}/content")
    def content(key: str):
        path = service.file_path(key)
        file = store.rows("SELECT name,modality FROM files WHERE id=?", (key,))[0]
        if not path.exists():
            raise KeyError("Original file is missing")
        media_type = mimetypes.guess_type(file["name"])[0] or "application/octet-stream"
        if file["modality"] in ("text", "code"):
            media_type = "text/plain; charset=utf-8"
        return FileResponse(
            path, filename=file["name"], media_type=media_type, content_disposition_type="inline"
        )

    @app.delete("/api/files/{key}", status_code=204)
    def delete_file(key: str):
        with write_lock:
            if service.job_lock.locked():
                raise HTTPException(409, "Wait for the active indexing job")
            path = service.file_path(key)
            store.execute("DELETE FROM files WHERE id=?", (key,))
            path.unlink(missing_ok=True)

    @app.post("/api/search")
    def query(body: SearchRequest):
        return service.search(body)

    @app.get("/api/search/results/{key}")
    def search_result(key: str):
        rows = store.rows("SELECT payload FROM searches WHERE id=?", (key,))
        if not rows:
            raise KeyError("Search not found")
        return json.loads(rows[0]["payload"])

    @app.post("/api/indexing/jobs", status_code=202)
    def index(body: IndexRequest, tasks: BackgroundTasks):
        if not store.rows("SELECT id FROM collections WHERE id=?", (body.collection_id,)):
            raise KeyError("Collection not found")
        if not settings.allow_cloud and embedder is None:
            raise CloudDisabled("Enable OMNISEEK_ALLOW_CLOUD=1 to start GPU indexing")
        key = new_id()
        store.execute(
            "INSERT INTO jobs (id,collection_id,status) VALUES (?,?,'queued')", (key, body.collection_id)
        )
        tasks.add_task(service.index, key, body.collection_id, body.rebuild)
        return {"id": key, "status": "queued"}

    @app.get("/api/indexing/jobs")
    def jobs():
        return store.rows("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 100")

    @app.get("/api/indexing/jobs/{key}")
    def job(key: str):
        rows = store.rows("SELECT * FROM jobs WHERE id=?", (key,))
        if not rows:
            raise KeyError("Job not found")
        return rows[0]

    @app.get("/api/experiments")
    def experiments():
        path = ROOT / "experiments/results/retrieval.json"
        return json.loads(path.read_text()) if path.exists() else {"status": "not_run", "results": []}

    @app.get("/api/demo/queries")
    def demo_queries():
        path = ROOT / "assets/demo/manifest.json"
        return json.loads(path.read_text()).get("queries", []) if path.exists() else []

    @app.get("/api/system/status")
    def status():
        return {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "dimension": 768,
            "cloud_enabled": settings.allow_cloud,
            "gpu": settings.gpu,
            "collections": len(store.collections()),
            "files": len(store.files()),
            "segments": store.rows("SELECT count(*) AS n FROM segments")[0]["n"],
            "cost": Ledger(settings.data_dir / "cost-ledger.json").report(),
            "limits": {
                "upload_mib": 20,
                "files_per_collection": settings.max_files,
                "media_seconds": settings.max_duration,
                "segments_per_job": 128,
            },
            "search": "FAISS IndexFlatIP on CPU",
            "index_version": 1,
        }

    frontend = ROOT / "frontend/dist"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()
