import json

import numpy as np
import pytest
import soundfile as sf

from omniseek_multimodal.config import ROOT, Settings
from omniseek_multimodal.costs import Ledger, estimate
from omniseek_multimodal.evaluation import metrics
from omniseek_multimodal.media import modality_for, prepare, safe_name, text_chunks


def test_path_traversal_is_reduced_to_name():
    assert safe_name("../../secret.py") == "secret.py"
    assert safe_name("C:\\Users\\private\\x.md") == "x.md"
    for name in ("", "x\x00.png", "x" * 201):
        with pytest.raises(ValueError):
            safe_name(name)


def test_language_aware_python_chunking():
    chunks = text_chunks(
        "import time\n\ndef retry():\n    return 1\n\ndef normalize():\n    return 2\n", "a.py"
    )
    assert len(chunks) == 3
    assert chunks[1]["line_start"] == 3
    assert "retry" in chunks[1]["text"]


def test_all_original_demo_modalities_decode(settings):
    manifest = json.loads((ROOT / "assets/demo/manifest.json").read_text())
    modalities = set()
    for name in manifest["files"]:
        modalities.add(modality_for(name))
        _, segments, payloads = prepare(ROOT / "assets/demo" / name, name, settings)
        assert segments and len(segments) == len(payloads)
    assert modalities == {"text", "code", "audio", "video", "image"}


def test_audio_segments_keep_exact_offsets_and_16khz(tmp_path, settings):
    path = tmp_path / "a.wav"
    sf.write(path, np.ones(44100 * 23, dtype=np.float32) * 0.1, 44100)
    meta, segments, payloads = prepare(path, "a.wav", settings)
    assert meta["sample_rate"] == 16000
    assert [s["start"] for s in segments] == [0, 10, 20]
    assert segments[-1]["end"] == pytest.approx(23)
    assert len(payloads[-1]["content"]) == 3 * 16000 * 4
    with pytest.raises(ValueError, match="exceeds"):
        prepare(path, "a.wav", Settings(data_dir=tmp_path, max_duration=20))


def test_video_windows_keep_timestamp_bounds(settings):
    _, segments, payloads = prepare(ROOT / "assets/demo/sailboat.mp4", "clip.mp4", settings)
    assert segments[0]["start"] == 0
    assert segments[0]["end"] == pytest.approx(4)
    assert len(payloads[0]["frames"]) == 4


def test_budget_reservations_are_atomic_and_failed_calls_stay_reserved(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    key = ledger.reserve("test", "T4")
    with pytest.raises(RuntimeError, match="already active"):
        Ledger(tmp_path / "ledger.json").reserve("parallel", "L4")
    ledger.finish(key, "failed")
    assert ledger.report()["reserved_usd"] == pytest.approx(0.5 + estimate("T4"))
    with ledger.locked() as state:
        state["storage_reserve_usd"] = 7.9
    with pytest.raises(RuntimeError, match="exceeded"):
        ledger.reserve("too expensive", "L4")


def test_ir_metrics_have_known_values():
    result = metrics(["b", "a", "c", "d"], {"a", "d"})
    assert result["recall_at_3"] == 0.5
    assert result["precision_at_3"] == pytest.approx(1 / 3)
    assert result["mrr"] == 0.5
    assert result["ndcg_at_3"] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
