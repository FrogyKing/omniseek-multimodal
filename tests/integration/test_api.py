import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from omniseek_multimodal.api import create_app
from omniseek_multimodal.config import ROOT


def collection(client):
    response = client.post("/api/collections", json={"name": "Test notes"})
    assert response.status_code == 201
    return response.json()["id"]


def test_collection_upload_duplicate_validation_and_delete(client):
    key = collection(client)
    response = client.post(
        "/api/uploads", data={"collection_id": key}, files={"file": ("../../hello.md", b"# Hello")}
    )
    assert response.status_code == 201
    file_id = response.json()["id"]
    duplicate = client.post(
        "/api/uploads", data={"collection_id": key}, files={"file": ("hello.md", b"# Hello")}
    )
    assert duplicate.json()["duplicate"]
    assert duplicate.json()["id"] == file_id
    files = client.get("/api/files").json()
    assert files["total"] == 1
    assert files["items"][0]["name"] == "hello.md"
    assert client.get(f"/api/files/{file_id}/content").text == "# Hello"
    assert client.delete(f"/api/collections/{key}").status_code == 204
    assert client.get(f"/api/files/{file_id}/content").status_code == 404
    assert client.get("/api/files").json()["total"] == 0


@pytest.mark.parametrize(
    "name,content",
    [
        ("bad.exe", b"run"),
        ("bad.png", b"not an image"),
        ("bad.wav", b"bad"),
        ("bad.mp4", b"bad"),
        ("empty.md", b""),
        ("binary.txt", b"\x00data"),
    ],
)
def test_reject_invalid_files(client, name, content):
    key = collection(client)
    response = client.post("/api/uploads", data={"collection_id": key}, files={"file": (name, content)})
    assert response.status_code == 422
    assert client.get("/api/files").json()["total"] == 0


def test_validation_origin_guard_and_cloud_disabled(client):
    assert client.post("/api/collections", json={"name": "   "}).status_code == 422
    assert client.post("/api/search", json={"query": ""}).status_code == 422
    assert client.post("/api/search", json={"query": "unknown", "dimension": 200}).status_code == 422
    assert client.post("/api/search", json={"query": "unknown"}).status_code == 409
    assert (
        client.post(
            "/api/collections", json={"name": "driveby"}, headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    assert client.get("/api/system/status", headers={"Host": "evil.example"}).status_code == 403
    key = collection(client)
    assert client.post("/api/indexing/jobs", json={"collection_id": key}).status_code == 409
    assert client.get("/api/openapi.json").status_code == 404
    assert client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize("dimension", [768, 512, 256, 128])
def test_real_recorded_multimodal_search_and_persistent_results(demo_client, dimension):
    response = demo_client.post(
        "/api/search", json={"query": "A sailboat on the blue ocean", "dimension": dimension}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["cache_hit"] and data["dimension"] == dimension
    assert data["results"][0]["name"] in ("ocean.md", "ocean.png", "sailboat.mp4")
    assert {"text", "image", "video"} <= {r["modality"] for r in data["results"][:5]}
    assert demo_client.get("/api/search/results/" + data["id"]).json() == data
    assert all(-1 <= r["score"] <= 1 for r in data["results"])


def test_image_query_and_video_timestamp(demo_client):
    files = demo_client.get("/api/files?limit=100").json()["items"]
    image = next(f for f in files if f["name"] == "ocean.png")
    result = demo_client.post("/api/search", json={"image_file_id": image["id"]}).json()
    assert result["cache_hit"]
    assert result["results"][0]["name"] == "ocean.png"
    assert "sailboat.mp4" in [r["name"] for r in result["results"][:3]]
    result = demo_client.post(
        "/api/search", json={"query": "A red ball bouncing", "modality": "video"}
    ).json()
    top = result["results"][0]
    assert top["name"] == "bouncing-ball.mp4"
    assert top["metadata"]["start"] == 0 and top["metadata"]["end"] == 4
    original = demo_client.get(top["url"], headers={"Range": "bytes=0-19"})
    assert original.status_code == 206
    assert len(original.content) == 20


def test_code_prompt_and_collection_filter(demo_client):
    result = demo_client.post(
        "/api/search",
        json={"query": "Find the function that retries failed API requests", "modality": "code"},
    ).json()
    assert result["cache_hit"]
    assert result["results"][0]["name"] == "retry.py"
    assert result["results"][0]["metadata"]["line_start"] >= 1
    empty = collection(demo_client)
    assert (
        demo_client.post("/api/search", json={"query": "A red door", "collection_id": empty}).json()[
            "results"
        ]
        == []
    )


def test_no_credential_values_in_status(demo_client, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-value-must-never-appear")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "modal-private-value")
    text = demo_client.get("/api/system/status").text
    assert "secret-value-must-never-appear" not in text
    assert "modal-private-value" not in text
    assert demo_client.get("/api/indexing/jobs").json() == []


def test_gpu_failure_does_not_destroy_previous_index(settings):
    class FailingEmbedder:
        def embed(self, *args, **kwargs):
            raise ConnectionError("Do not expose backend internals")

    app = create_app(settings, embedder=FailingEmbedder())
    with TestClient(app) as client:
        key = collection(client)
        file = client.post(
            "/api/uploads", data={"collection_id": key}, files={"file": ("a.txt", b"content")}
        ).json()
        app.state.service.store.replace_segments(
            file["id"], [{"text": "content"}], np.ones((1, 768), dtype=np.float32)
        )
        job = client.post("/api/indexing/jobs", json={"collection_id": key, "rebuild": True}).json()
        response = client.get("/api/indexing/jobs/" + job["id"]).json()
        assert response["status"] == "failed"
        assert "internals" not in json.dumps(response)
        assert app.state.service.store.corpus()[1].shape == (1, 768)


def test_missing_demo_artifact_has_clear_error(tmp_path, settings, monkeypatch):
    import omniseek_multimodal.demo as demo

    monkeypatch.setattr(demo, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="missing"):
        demo.install_demo(settings)


def test_demo_assets_recorded_from_real_gpu():
    run = json.loads((ROOT / "experiments/results/demo-run.json").read_text())
    assert run["measurements"]["gpu"] in ("Tesla T4", "NVIDIA L4")
    assert run["measurements"]["parameters"] == 744371488
    assert run["corpus_count"] == 14
