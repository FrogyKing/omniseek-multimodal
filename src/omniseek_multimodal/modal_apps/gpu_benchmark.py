"""Small encoder, batching and temporal diagnostics in one bounded GPU input."""

from .embedding_service import MODEL_ID, MODEL_REVISION, app, hf_token, options


@app.function(gpu="T4", name="research_t4", **options)
def research(payloads: list[dict], video_bytes: bytes) -> dict:
    import io
    import tempfile
    import time
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    token, _ = hf_token()
    dtype = torch.float32
    configs = {
        "full": {},
        "text": {"vision_config": None, "audio_config": None},
        "vision": {"audio_config": None},
        "audio": {"vision_config": None},
    }

    def load(config):
        return SentenceTransformer(
            MODEL_ID,
            revision=MODEL_REVISION,
            token=token,
            device="cuda",
            config_kwargs=configs[config],
            model_kwargs={"dtype": dtype},
        )

    def encode(model, items, **kwargs):
        with torch.inference_mode():
            result = np.asarray(
                model.encode(items, normalize_embeddings=True, show_progress_bar=False, **kwargs),
                dtype=np.float32,
            )
        torch.cuda.synchronize()
        return result

    start = time.perf_counter()
    model = load("full")
    texts = [
        "task: search result | query: A red ball bouncing",
        "task: search result | query: A sailboat on the blue ocean",
        "task: search result | query: A ringing bell",
        "task: search result | query: Random static white noise",
    ]
    image_item = next(p for p in payloads if p["kind"] == "image")
    image_input = {"image": Image.open(io.BytesIO(image_item["content"])).convert("RGB")}
    audios = [np.frombuffer(p["content"], dtype=np.float32).copy() for p in payloads if p["kind"] == "audio"]
    audio_input = {"audio": audios[0]}
    full_text = encode(model, texts)
    full_image = encode(model, [image_input])
    full_audio = encode(model, [audio_input])
    batches = []
    for batch_size in (1, 2, 4, 8):
        encode(model, texts * 2, batch_size=batch_size)
        times = []
        for _ in range(5):
            before = time.perf_counter()
            encode(model, texts * 2, batch_size=batch_size)
            times.append(time.perf_counter() - before)
        batches.append(
            {
                "batch_size": batch_size,
                "items": 8,
                "repeats": 5,
                "median_seconds": float(np.median(times)),
                "items_per_second": 8 / float(np.median(times)),
            }
        )

    frames = [
        Image.open(io.BytesIO(b)).convert("RGB")
        for p in payloads
        if p["kind"] == "video"
        for b in p["frames"]
    ][:4]
    # Deliberately join two visible scenes to measure localization at a known transition.
    video_segments = []
    temporal = []
    if frames:
        # Input video is a separately generated original: ball [0,2), boat [2,4).
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transition.mp4"
            path.write_bytes(video_bytes)
            native = encode(model, [{"video": str(path)}])
            # Use the model's native file decoder above; compare prepared frames below.
            import av

            decoded: list[Image.Image] = []
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    if float(frame.time or 0) + 1e-6 >= len(decoded):
                        decoded.append(frame.to_image())
            scene_starts = [0]
            for i in range(1, len(decoded)):
                left = np.asarray(decoded[i - 1].resize((16, 16)), dtype=float) / 255
                right = np.asarray(decoded[i].resize((16, 16)), dtype=float) / 255
                if float(np.mean(np.abs(left - right))) > 0.18:
                    scene_starts.append(i)
            scene_starts.append(len(decoded))
            strategies = {
                "native_full_clip": [(0, len(decoded))],
                "fixed_1s": [(i, i + 1) for i in range(len(decoded))],
                "fixed_2s": [(i, min(i + 2, len(decoded))) for i in range(0, len(decoded), 2)],
                "scene_mean_rgb": list(zip(scene_starts[:-1], scene_starts[1:], strict=True)),
            }
            for name, intervals in strategies.items():
                vec = (
                    native
                    if name == "native_full_clip"
                    else encode(
                        model,
                        [{"video": decoded[a:b]} for a, b in intervals],
                        processing_kwargs={"video": {"fps": None, "add_timestamps": False}},
                    )
                )
                similarities = full_text[:2] @ vec.T
                matches = [
                    {
                        "query": texts[i],
                        "start": intervals[int(np.argmax(row))][0],
                        "end": intervals[int(np.argmax(row))][1],
                        "score": float(max(row)),
                    }
                    for i, row in enumerate(similarities)
                ]
                video_segments.append({"strategy": name, "intervals": intervals, "matches": matches})
    composite = np.concatenate(audios)
    for window in (1.5, 3.0, 6.0):
        step = int(window * 16000)
        audio_intervals = [
            (i / 16000, min(i + step, len(composite)) / 16000) for i in range(0, len(composite), step)
        ]
        vec = encode(model, [{"audio": composite[i : i + step]} for i in range(0, len(composite), step)])
        scores = full_text[2:] @ vec.T
        temporal.append(
            {
                "window_seconds": window,
                "intervals": audio_intervals,
                "matches": [
                    {
                        "start": audio_intervals[int(np.argmax(row))][0],
                        "end": audio_intervals[int(np.argmax(row))][1],
                        "score": float(max(row)),
                    }
                    for row in scores
                ],
            }
        )
    actual_dtype = str(model.dtype)
    del model
    torch.cuda.empty_cache()
    compatibility = []
    for config, inputs, reference in [
        ("text", texts, full_text),
        ("vision", [image_input], full_image),
        ("audio", [audio_input], full_audio),
    ]:
        before = time.perf_counter()
        model = load(config)
        vectors = encode(model, inputs)
        compatibility.append(
            {
                "config": config,
                "parameters": sum(p.numel() for p in model.parameters()),
                "max_abs_delta": float(np.max(np.abs(vectors - reference))),
                "minimum_cosine_with_full": float(np.min(np.sum(vectors * reference, axis=1))),
                "load_and_encode_seconds": time.perf_counter() - before,
            }
        )
        del model
        torch.cuda.empty_cache()
    return {
        "revision": MODEL_REVISION,
        "gpu": torch.cuda.get_device_name(),
        "dtype": actual_dtype,
        "total_seconds": time.perf_counter() - start,
        "batching": batches,
        "encoder_compatibility": compatibility,
        "video": video_segments,
        "audio": temporal,
        "notes": "One synthetic transition and two generated sounds; diagnostic, not a validated segmentation benchmark.",
    }
