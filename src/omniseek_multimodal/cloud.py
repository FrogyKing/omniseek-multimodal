"""Authenticated Modal calls are budget-reserved and explicitly enabled."""

import time

import modal
import numpy as np

from .config import MODEL_REVISION, Settings
from .costs import RATES, Ledger


class CloudDisabled(RuntimeError):
    pass


def check_billing(gpu: str):
    """Fail closed when price information is unavailable or has increased."""
    workspace = modal.Workspace.from_context()
    summary = workspace.billing.summary()
    rates = workspace.billing.rates()
    for name, expected in (
        (f"gpu_hour_cost_{gpu.lower()}", RATES[gpu]),
        ("cpu_hour_cost", RATES["CPU"]),
        ("mem_gib_hour_cost", RATES["RAM"]),
        ("volume_storage_gib_month_cost", 0.09),
        ("egress_gib_cost", 0.04),
    ):
        rate = rates.get(name)
        if rate is None or float(rate) > expected * 1.01:
            raise RuntimeError(
                "Provider prices unavailable or increased; update cost estimates before execution"
            )
    return summary


class ModalEmbedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ledger = Ledger(settings.data_dir / "cost-ledger.json")

    def embed(
        self, payloads: list[dict], config: str = "auto", batch_size: int = 4, label: str = "embedding"
    ) -> tuple[np.ndarray, dict]:
        if not self.settings.allow_cloud:
            raise CloudDisabled(
                "Cloud inference is disabled. Use a cached demo query, or enable "
                "OMNISEEK_ALLOW_CLOUD=1 to index files and embed new queries."
            )
        if config == "auto":
            kinds = {p["kind"] for p in payloads}
            vision = bool(kinds & {"image", "video"})
            audio = "audio" in kinds
            config = "full" if vision and audio else "vision" if vision else "audio" if audio else "text"
        from .modal_apps.embedding_service import app, embed_l4, embed_t4

        # Check current workspace billing before every reservation. Values stay local.
        summary = check_billing(self.settings.gpu)
        key = self.ledger.reserve(label, self.settings.gpu)
        start = time.perf_counter()
        try:
            with app.run():
                function = embed_t4 if self.settings.gpu == "T4" else embed_l4
                result = function.remote(payloads, config, batch_size)
                app_id = app.app_id
            if result["revision"] != MODEL_REVISION:
                raise ValueError("Remote model revision mismatch")
            measurements = {k: v for k, v in result.items() if k != "vectors"}
            measurements["wall_seconds"] = time.perf_counter() - start
            measurements["modal_app_id"] = app_id
            measurements["billing_before_usd"] = float(summary.billed_cost)
            self.ledger.finish(key, "succeeded", measurements)
            return np.asarray(result["vectors"], dtype=np.float32), measurements
        except Exception:
            self.ledger.finish(key, "failed", {"wall_seconds": time.perf_counter() - start})
            raise
