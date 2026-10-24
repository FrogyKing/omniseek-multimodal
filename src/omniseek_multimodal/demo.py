import hashlib
import json

import numpy as np

from .cloud import ModalEmbedder
from .config import MODEL_REVISION, ROOT, Settings
from .media import prepare
from .service import Service
from .storage import new_id


def gpu_demo(settings: Settings, benchmark: bool = False):
    manifest = json.loads((ROOT / "assets/demo/manifest.json").read_text())
    payloads: list[dict] = []
    records = []
    for name in manifest["files"]:
        metadata, segments, items = prepare(ROOT / "assets/demo" / name, name, settings)
        records.append(
            {
                "name": name,
                "metadata": metadata,
                "segments": segments,
                "offset": len(payloads),
                "count": len(items),
                "sha256": hashlib.sha256((ROOT / "assets/demo" / name).read_bytes()).hexdigest(),
            }
        )
        payloads.extend(items)
    corpus_count = len(payloads)
    queries = [{"kind": "text", "text": q["text"], "prompt": "SearchQuery"} for q in manifest["queries"]]
    code_queries = [
        {"kind": "text", "text": q["text"], "prompt": "CodeRetrieval"}
        for q in manifest["queries"]
        if q["modality"] == "code"
    ]
    payloads.extend(queries + code_queries)
    if benchmark:
        # Compare prompt choice on the same corpus and the same GPU execution.
        payloads.extend({"kind": "text", "text": q["text"]} for q in manifest["queries"])
    print(
        f"Running {len(payloads)} real embeddings on Modal {settings.gpu}; bounded at 600 s + 300 s startup."
    )
    import modal

    with modal.enable_output():
        vectors, measurements = ModalEmbedder(settings).embed(
            payloads, batch_size=4, label="demo benchmark" if benchmark else "multimodal smoke"
        )
    out = ROOT / "experiments/results"
    out.mkdir(parents=True, exist_ok=True)
    label = settings.gpu.lower()
    np.savez_compressed(
        out / f"{label}-embeddings.npz",
        corpus=vectors[:corpus_count],
        queries=vectors[corpus_count : corpus_count + len(queries)],
        code_queries=vectors[corpus_count + len(queries) : corpus_count + len(queries) + len(code_queries)],
        unprompted=vectors[corpus_count + len(queries) + len(code_queries) :],
    )
    data = {
        "model_revision": MODEL_REVISION,
        "records": records,
        "measurements": measurements,
        "query_count": len(queries),
        "corpus_count": corpus_count,
        "gpu": settings.gpu,
        "benchmark": benchmark,
    }
    (out / f"{label}-run.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    # The least expensive verified run is the portable demo source.
    if settings.gpu == "T4" or not (out / "demo-run.json").exists():
        (out / "demo-run.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        np.savez_compressed(
            out / "demo-embeddings.npz",
            corpus=vectors[:corpus_count],
            queries=vectors[corpus_count : corpus_count + len(queries)],
            code_queries=vectors[
                corpus_count + len(queries) : corpus_count + len(queries) + len(code_queries)
            ],
        )
    print(json.dumps(measurements, indent=2))
    print(json.dumps(install_demo(settings), indent=2))


def install_demo(settings: Settings) -> dict:
    out = ROOT / "experiments/results"
    if not (out / "demo-embeddings.npz").exists():
        raise RuntimeError(
            "Verified demo embeddings are missing; run an explicitly budgeted gpu-smoke-test first"
        )
    run = json.loads((out / "demo-run.json").read_text())
    if run["model_revision"] != MODEL_REVISION:
        raise ValueError("Demo model revision mismatch")
    manifest = json.loads((ROOT / "assets/demo/manifest.json").read_text())
    data = np.load(out / "demo-embeddings.npz", allow_pickle=False)
    service = Service(settings)
    collections = [c for c in service.store.collections() if c["name"] == "Signal / demo collection"]
    if collections:
        collection_id = collections[0]["id"]
    else:
        collection_id = new_id()
        service.store.execute(
            "INSERT INTO collections (id,name,description) VALUES (?,?,?)",
            (
                collection_id,
                "Signal / demo collection",
                "12 original CC0 assets · real Modal embeddings · cached example queries",
            ),
        )
    for record in run["records"]:
        name = record["name"]
        if (
            record.get("sha256")
            and hashlib.sha256((ROOT / "assets/demo" / name).read_bytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"Demo asset differs from recorded embedding source: {name}")
        file = service.upload(collection_id, name, (ROOT / "assets/demo" / name).read_bytes())
        start, count = record["offset"], record["count"]
        service.store.replace_segments(file["id"], record["segments"], data["corpus"][start : start + count])
    code_offset = 0
    for q, vector in zip(manifest["queries"], data["queries"], strict=True):
        service.store.cache_query(q["text"], vector)
        if q["modality"] == "code":
            service.store.cache_query(q["text"], data["code_queries"][code_offset], "CodeRetrieval")
            code_offset += 1
    return {
        "collection_id": collection_id,
        "files": len(run["records"]),
        "segments": run["corpus_count"],
        "cached_queries": len(manifest["queries"]),
        "cloud_spend": 0,
    }
