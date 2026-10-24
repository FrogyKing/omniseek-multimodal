"""Private ephemeral GPU functions; no HTTP endpoint and no warm containers."""

import os
import time
from pathlib import Path
from typing import Any

import modal

from omniseek_multimodal.config import MODEL_ID, MODEL_REVISION

app = modal.App("omniseek-embeddings")
cache = modal.Volume.from_name("omniseek-model-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libsndfile1")
    .uv_pip_install(
        requirements=[str(Path(__file__).resolve().parents[3] / "requirements/gpu.txt")], uv_version="0.12.5"
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_python_source("omniseek_multimodal")
)
secret = modal.Secret.from_name("huggingface")


def hf_token():
    # The existing secret's actual key is inspected by preflight. Never return values.
    names = [
        name
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")
        if os.getenv(name)
    ]
    if not names:
        raise RuntimeError("huggingface secret has no supported token variable")
    return os.environ[names[0]], names


@app.function(
    image=image,
    secrets=[secret],
    volumes={"/cache": cache},
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=300,
    startup_timeout=300,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    retries=0,
    single_use_containers=True,
)
def preflight():
    from huggingface_hub import HfApi, snapshot_download

    token, names = hf_token()
    api = HfApi(token=token)
    api.whoami()  # Validate access without returning account or token data.
    info = api.model_info(MODEL_ID, revision=MODEL_REVISION)
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, token=token)
    cache.commit()
    return {"authenticated": True, "secret_keys": names, "revision": info.sha, "model_id": MODEL_ID}


def run_embedding(payloads: list[dict], config: str = "full", batch_size: int = 4) -> dict:
    import io

    import numpy as np
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    if not 1 <= len(payloads) <= 128 or batch_size not in (1, 2, 4, 8):
        raise ValueError("Payload count or batch size exceeds limit")
    total = sum(
        len(p.get("content", b"")) + sum(map(len, p.get("frames", []))) + len(p.get("text", "").encode())
        for p in payloads
    )
    if total > 64 * 1024 * 1024:
        raise ValueError("Decoded payload exceeds 64 MiB")
    configs = {
        "full": {},
        "text": {"vision_config": None, "audio_config": None},
        "vision": {"audio_config": None},
        "audio": {"vision_config": None},
    }
    if config not in configs:
        raise ValueError("Unknown encoder configuration")
    start = time.perf_counter()
    token, _ = hf_token()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32
    torch.set_num_threads(2)
    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=token,
        device="cuda",
        config_kwargs=configs[config],
        model_kwargs={"dtype": dtype},
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    outputs = []
    timings = []
    # Homogeneous prompt/modality groups preserve input order and allow batching.
    i = 0
    while i < len(payloads):
        kind = payloads[i]["kind"]
        prompt = payloads[i].get("prompt")
        group: list[Any] = []
        while (
            i < len(payloads)
            and len(group) < batch_size
            and payloads[i]["kind"] == kind
            and payloads[i].get("prompt") == prompt
        ):
            item = payloads[i]
            if kind == "text":
                group.append(item["text"])
            elif kind == "image":
                group.append({"image": Image.open(io.BytesIO(item["content"])).convert("RGB")})
            elif kind == "audio":
                group.append({"audio": np.frombuffer(item["content"], dtype=np.float32).copy()})
            elif kind == "video":
                group.append({"video": [Image.open(io.BytesIO(b)).convert("RGB") for b in item["frames"]]})
            else:
                raise ValueError("Unsupported payload modality")
            i += 1
        before = time.perf_counter()
        with torch.inference_mode():
            vectors = model.encode(
                group,
                batch_size=batch_size,
                prompt_name=prompt,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        torch.cuda.synchronize()
        timings.append({"kind": kind, "count": len(group), "seconds": time.perf_counter() - before})
        outputs.extend(np.asarray(vectors, dtype=np.float32).tolist())
    vectors = np.array(outputs, dtype=np.float32)
    if vectors.shape != (len(payloads), 768) or not np.isfinite(vectors).all():
        raise RuntimeError("Model returned invalid embeddings")
    result = {
        "vectors": outputs,
        "revision": MODEL_REVISION,
        "dimension": 768,
        "config": config,
        "dtype": str(model.dtype),
        "gpu": torch.cuda.get_device_name(),
        "load_seconds": load_seconds,
        "total_seconds": time.perf_counter() - start,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "batches": timings,
        "batch_size": batch_size,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    del model
    torch.cuda.empty_cache()
    return result


options: dict[str, Any] = dict(
    image=image,
    secrets=[secret],
    volumes={"/cache": cache},
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=600,
    startup_timeout=300,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    retries=0,
    single_use_containers=True,
)
embed_t4 = app.function(gpu="T4", **options)(run_embedding)
embed_l4 = app.function(gpu="L4", name="embed_l4", **options)(run_embedding)
