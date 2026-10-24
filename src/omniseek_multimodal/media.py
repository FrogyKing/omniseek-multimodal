"""Bounded decoding and timestamp-preserving media preparation."""

import ast
import io
import warnings
from pathlib import Path

import av
import numpy as np
from PIL import Image

from .config import Settings

EXTENSIONS = {
    "text": {".txt", ".md", ".csv", ".json", ".rst"},
    "code": {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".cpp", ".c", ".sql"},
    "image": {".png", ".jpg", ".jpeg", ".webp"},
    "audio": {".wav", ".flac", ".mp3", ".ogg", ".m4a"},
    "video": {".mp4", ".webm", ".mov"},
}
Image.MAX_IMAGE_PIXELS = 20_000_000


def modality_for(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    for modality, extensions in EXTENSIONS.items():
        if ext in extensions:
            return modality
    raise ValueError("Unsupported file type")


def safe_name(name: str) -> str:
    name = name.replace("\\", "/").split("/")[-1].strip()
    if not name or len(name) > 200 or any(ord(c) < 32 for c in name):
        raise ValueError("Invalid filename")
    return name


def image_bytes(image: Image.Image) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((768, 768))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def text_chunks(text: str, filename: str) -> list[dict]:
    lines = text.splitlines()
    boundaries = [0, len(lines)]
    if filename.endswith(".py"):
        try:
            tree = ast.parse(text)
            boundaries = sorted(
                {0, len(lines)}
                | {
                    node.lineno - 1
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                }
            )
        except SyntaxError:
            pass
    chunks = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        for start in range(left, right, 40):
            end = min(start + 48, right)
            content = "\n".join(lines[start:end])
            # Bound very long minified lines as well as ordinary source files.
            for offset in range(0, len(content), 6000):
                part = content[offset : offset + 6000]
                if part.strip():
                    chunks.append({"text": part, "line_start": start + 1, "line_end": end})
    return chunks


def prepare(path: Path, filename: str, settings: Settings) -> tuple[dict, list[dict], list[dict]]:
    modality = modality_for(filename)
    payloads: list[dict] = []
    segments: list[dict] = []
    metadata: dict = {}
    if modality in ("text", "code"):
        content = path.read_text(encoding="utf-8")
        if "\x00" in content:
            raise ValueError("Binary content is not a UTF-8 text document")
        segments = text_chunks(content, filename)
        payloads = [{"kind": "text", "text": f"title: {filename} | text: {s['text']}"} for s in segments]
        metadata = {"lines": len(content.splitlines())}
    elif modality == "image":
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image.load()
                metadata = {"width": image.width, "height": image.height}
                payloads = [{"kind": "image", "content": image_bytes(image)}]
        segments = [{"text": filename}]
    elif modality == "audio":
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            duration = (
                float(stream.duration * stream.time_base)
                if stream.duration
                else (container.duration / av.time_base if container.duration else 0)
            )
            if duration > settings.max_duration:
                raise ValueError(f"Audio exceeds {settings.max_duration:g} seconds")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
            pieces = []
            count = 0
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    piece = converted.to_ndarray().flatten()
                    count += len(piece)
                    if count > settings.max_duration * 16000:
                        raise ValueError("Decoded audio exceeds duration limit")
                    pieces.append(piece)
            for converted in resampler.resample(None):
                pieces.append(converted.to_ndarray().flatten())
            waveform = np.concatenate(pieces).astype(np.float32) if pieces else np.empty(0)
            duration = len(waveform) / 16000
            metadata = {"duration": duration, "sample_rate": 16000}
            step = int(settings.segment_seconds * 16000)
            for start in range(0, len(waveform), step):
                chunk = waveform[start : start + step]
                segments.append(
                    {"text": filename, "start": start / 16000, "end": (start + len(chunk)) / 16000}
                )
                payloads.append({"kind": "audio", "content": chunk.tobytes()})
    else:
        with av.open(str(path)) as container:
            video_stream = container.streams.video[0]
            if video_stream.width * video_stream.height > 20_000_000:
                raise ValueError("Video frame exceeds 20 million pixel limit")
            duration = (
                float(video_stream.duration * video_stream.time_base)
                if video_stream.duration
                else (container.duration / av.time_base if container.duration else 0)
            )
            if duration > settings.max_duration:
                raise ValueError(f"Video exceeds {settings.max_duration:g} seconds")
            frames: dict[int, list[bytes]] = {}
            next_second = 0.0
            decoded = 0
            last_time = 0.0
            for video_frame in container.decode(video_stream):
                decoded += 1
                if decoded > 30000:
                    raise ValueError("Video exceeds decoded-frame limit")
                timestamp = float(video_frame.time or 0)
                if timestamp > settings.max_duration:
                    raise ValueError("Video exceeds duration limit")
                last_time = timestamp
                if timestamp + 1e-6 < next_second:
                    continue
                next_second = timestamp + 1.0
                window = int(timestamp // settings.segment_seconds)
                frames.setdefault(window, []).append(image_bytes(video_frame.to_image()))
            duration = duration or last_time + 1 / float(video_stream.average_rate or 1)
            metadata = {
                "duration": duration,
                "width": video_stream.width,
                "height": video_stream.height,
                "sampling_fps": 1,
                "segmentation": "fixed windows",
            }
            for window, images in sorted(frames.items()):
                window_start = window * settings.segment_seconds
                segments.append(
                    {
                        "text": filename,
                        "start": window_start,
                        "end": min(duration, window_start + settings.segment_seconds),
                    }
                )
                payloads.append({"kind": "video", "frames": images})
    if not segments:
        raise ValueError("File contains no decodable content")
    if len(segments) > settings.max_segments:
        raise ValueError("File exceeds segment limit")
    return metadata, segments, payloads
