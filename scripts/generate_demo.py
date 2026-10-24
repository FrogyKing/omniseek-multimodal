"""Generate tiny original CC0 fixtures; no downloaded third-party media."""

import json
import math
from pathlib import Path

import av
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/demo"


def scene(kind: str, progress: float = 0) -> Image.Image:
    image = Image.new("RGB", (640, 400), "#d3e8ed")
    d = ImageDraw.Draw(image)
    if kind == "mountain":
        d.rectangle((0, 260, 640, 400), fill="#2f624e")
        d.polygon([(60, 290), (265, 40), (480, 290)], fill="#687d8a")
        d.polygon([(265, 40), (195, 125), (325, 125)], fill="#f5f7f5")
        d.polygon([(300, 310), (470, 100), (640, 310)], fill="#506777")
        d.ellipse((520, 30, 580, 90), fill="#f8d575")
    elif kind == "door":
        d.rectangle((0, 0, 640, 400), fill="#e4ded2")
        d.rectangle((200, 40, 430, 400), fill="#554c48")
        d.rectangle((215, 50, 415, 395), fill="#bb342c")
        d.rectangle((235, 75, 390, 190), outline="#ee7766", width=5)
        d.rectangle((235, 225, 390, 355), outline="#ee7766", width=5)
        d.ellipse((373, 202, 389, 218), fill="#e5bb68")
    elif kind == "ocean":
        d.rectangle((0, 185, 640, 400), fill="#2673a0")
        for y in range(200, 400, 28):
            for x in range(-80, 700, 80):
                d.arc((x, y, x + 80, y + 20), 0, 180, fill="#8ad0db", width=3)
        x = int(80 + progress * 350)
        d.polygon([(x, 230), (x + 130, 230), (x + 105, 265), (x + 25, 265)], fill="#234258")
        d.line((x + 65, 110, x + 65, 230), fill="#30434c", width=5)
        d.polygon([(x + 60, 115), (x + 60, 215), (x - 2, 215)], fill="#faf5e8")
        d.polygon([(x + 70, 120), (x + 70, 213), (x + 120, 213)], fill="#d9c9aa")
    else:
        d.rectangle((0, 0, 640, 400), fill="#eee8df")
        d.rectangle((0, 320, 640, 400), fill="#9cac85")
        x = int(60 + 480 * progress)
        y = 265 - int(120 * abs(math.sin(progress * math.pi * 3)))
        d.ellipse((x - 36, y - 36, x + 36, y + 36), fill="#d53d35", outline="#942924", width=3)
    return image


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    texts = {
        "aurora.md": "# The northern lights\nSolar particles collide with gases in Earth's atmosphere. "
        "These collisions create green and purple auroras near the magnetic poles.\n",
        "ocean.md": "# Ocean field notes\nA white sailboat crosses a calm blue ocean. Wind fills its sails. "
        "Waves ripple across the sea below a clear blue sky.\n",
        "deployment.md": "# Release procedure\nBuild and test before deploying. Keep a previous release "
        "available for rollback. Monitor errors and latency after a production release.\n",
        "retry.py": '"""Retry failed API requests with exponential backoff."""\nimport time\n\n'
        "def retry_request(request, attempts=3):\n"
        "    for attempt in range(attempts):\n"
        "        try:\n            return request()\n"
        "        except ConnectionError:\n"
        "            if attempt == attempts - 1:\n                raise\n"
        "            time.sleep(2 ** attempt)\n",
        "normalize.py": '"""Normalize vectors for cosine similarity."""\nimport numpy as np\n\n'
        "def normalize_vector(vector):\n"
        "    norm = np.linalg.norm(vector)\n"
        '    if norm == 0:\n        raise ValueError("zero vector")\n'
        "    return vector / norm\n",
    }
    for name, text in texts.items():
        (OUT / name).write_text(text, encoding="utf-8", newline="\r\n")
    for kind in ("mountain", "door", "ocean"):
        scene(kind).save(OUT / f"{kind}.png")
    sr = 16000
    t = np.arange(sr * 3) / sr
    rng = np.random.default_rng(42)
    sf.write(
        OUT / "bell.wav",
        (np.sin(2 * np.pi * 880 * t) * np.exp(-t * 2) + 0.5 * np.sin(2 * np.pi * 1760 * t) * np.exp(-t * 3))
        * 0.5,
        sr,
    )
    sf.write(OUT / "noise.wav", rng.normal(0, 0.12, len(t)), sr)
    for kind, name in (("ball", "bouncing-ball.mp4"), ("ocean", "sailboat.mp4")):
        with av.open(str(OUT / name), "w") as container:
            stream = container.add_stream("libx264", rate=12)
            stream.width, stream.height, stream.pix_fmt = 640, 400, "yuv420p"
            for i in range(48):
                frame = av.VideoFrame.from_image(scene(kind, i / 47))
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    with av.open(str(OUT / "transition.mp4"), "w") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width, stream.height, stream.pix_fmt = 640, 400, "yuv420p"
        for i in range(48):
            frame = av.VideoFrame.from_image(scene("ball" if i < 24 else "ocean", (i % 24) / 23))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    queries = [
        {"text": "What causes the northern lights?", "relevant": ["aurora.md"], "modality": "text"},
        {
            "text": "A sailboat on the blue ocean",
            "relevant": ["ocean.md", "ocean.png", "sailboat.mp4"],
            "modality": "mixed",
        },
        {"text": "A red door", "relevant": ["door.png"], "modality": "image"},
        {"text": "Snow covered mountains", "relevant": ["mountain.png"], "modality": "image"},
        {"text": "A red ball bouncing", "relevant": ["bouncing-ball.mp4"], "modality": "video"},
        {"text": "A ringing bell", "relevant": ["bell.wav"], "modality": "audio"},
        {"text": "Random static white noise", "relevant": ["noise.wav"], "modality": "audio"},
        {
            "text": "Find the function that retries failed API requests",
            "relevant": ["retry.py"],
            "modality": "code",
        },
        {"text": "Normalize a vector to unit length", "relevant": ["normalize.py"], "modality": "code"},
        {
            "text": "How to roll back a production deployment",
            "relevant": ["deployment.md"],
            "modality": "text",
        },
    ]
    files = [
        *texts,
        "mountain.png",
        "door.png",
        "ocean.png",
        "bell.wav",
        "noise.wav",
        "bouncing-ball.mp4",
        "sailboat.mp4",
    ]
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "license": "CC0-1.0",
                "seed": 42,
                "description": "Original programmatically generated diagnostic fixtures; not a general retrieval benchmark.",
                "files": files,
                "queries": queries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(files)} original demo assets")


if __name__ == "__main__":
    main()
