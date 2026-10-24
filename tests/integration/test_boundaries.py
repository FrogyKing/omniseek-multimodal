import subprocess
import sys

from fastapi.testclient import TestClient

from omniseek_multimodal.api import create_app
from omniseek_multimodal.config import Settings


def test_chunked_requests_cannot_bypass_size_limit(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, max_upload_bytes=16))) as client:
        chunks = (b"x" * 4096 for _ in range(300))
        response = client.post(
            "/api/uploads", content=chunks, headers={"Content-Type": "application/octet-stream"}
        )
        assert response.status_code == 413


def test_collection_file_count_limit(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, max_files=1))) as client:
        key = client.post("/api/collections", json={"name": "Bounded"}).json()["id"]
        first = client.post("/api/uploads", data={"collection_id": key}, files={"file": ("one.md", b"first")})
        assert first.status_code == 201
        second = client.post(
            "/api/uploads", data={"collection_id": key}, files={"file": ("two.md", b"second")}
        )
        assert second.status_code == 422
        assert client.get("/api/files").json()["total"] == 1


def test_modal_definitions_import_outside_repository_root(tmp_path):
    code = (
        "from omniseek_multimodal.modal_apps.embedding_service import embed_t4; "
        "from omniseek_multimodal.modal_apps.gpu_benchmark import research; "
        'print("packaged imports ok")'
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=30, check=True
    )
    assert result.stdout.strip() == "packaged imports ok"
