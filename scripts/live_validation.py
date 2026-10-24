"""Explicitly billable API integration check using original disposable assets."""

import json

from fastapi.testclient import TestClient

from omniseek_multimodal.api import create_app
from omniseek_multimodal.config import ROOT, Settings


def main():
    settings = Settings(allow_cloud=True, gpu="T4")
    app = create_app(settings)
    collection_id = None
    with TestClient(app) as client:
        try:
            response = client.post(
                "/api/collections", json={"name": "Disposable live integration validation"}
            )
            response.raise_for_status()
            collection_id = response.json()["id"]
            names = ["aurora.md", "retry.py", "door.png", "bell.wav", "bouncing-ball.mp4"]
            for name in names:
                result = client.post(
                    "/api/uploads",
                    data={"collection_id": collection_id},
                    files={"file": (name, (ROOT / "assets/demo" / name).read_bytes())},
                )
                result.raise_for_status()
            job = client.post("/api/indexing/jobs", json={"collection_id": collection_id})
            job.raise_for_status()
            job_result = client.get("/api/indexing/jobs/" + job.json()["id"]).json()
            assert job_result["status"] == "completed", job_result
            query = "Show a red door in a wall"
            found = client.post("/api/search", json={"query": query, "collection_id": collection_id})
            found.raise_for_status()
            result = found.json()
            assert not result["cache_hit"]
            assert result["results"][0]["name"] == "door.png"
            # Restart with GPU access disabled and prove canonical vectors + cached query persist.
            with TestClient(create_app(Settings(allow_cloud=False))) as restarted:
                repeated = restarted.post(
                    "/api/search", json={"query": query, "collection_id": collection_id}
                ).json()
                assert repeated["cache_hit"]
                assert repeated["results"][0]["name"] == "door.png"
            receipt = {
                "status": "passed",
                "uploaded_modalities": ["text", "code", "image", "audio", "video"],
                "indexed_segments": result["total_segments"],
                "new_query_gpu_executed": True,
                "top_result": result["results"][0]["name"],
                "score": result["results"][0]["score"],
                "restart_without_gpu_passed": True,
                "job_message": job_result["message"],
            }
            (ROOT / "experiments/results/live-api-validation.json").write_text(json.dumps(receipt, indent=2))
            print(json.dumps(receipt, indent=2))
        finally:
            if collection_id:
                client.delete("/api/collections/" + collection_id).raise_for_status()


if __name__ == "__main__":
    main()
