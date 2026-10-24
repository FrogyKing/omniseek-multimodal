"""Reproducible model identity and bounded local-first configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
INDEX_VERSION = 1
DIMENSIONS = (768, 512, 256, 128)
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("OMNISEEK_DATA_DIR", "data")))
    allow_cloud: bool = field(default_factory=lambda: os.getenv("OMNISEEK_ALLOW_CLOUD") == "1")
    gpu: str = field(default_factory=lambda: os.getenv("OMNISEEK_GPU", "T4"))
    max_upload_bytes: int = 20 * 1024 * 1024
    max_files: int = 100
    max_duration: float = 180.0
    segment_seconds: float = 10.0
    max_segments: int = 128
