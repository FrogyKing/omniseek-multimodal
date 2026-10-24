"""Atomic conservative reservations; never assumes credits are available."""

import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import portalocker

# Modal billing rates verified 2026-10-24; CPU and RAM include resource ceilings.
RATES = {"T4": 0.59, "L4": 0.80, "CPU": 0.0473, "RAM": 0.008}
CEILING = 8.0  # Preserve USD 2 of the user's USD 10 authorization.


def estimate(gpu: str | None, seconds: int = 930, cpu: int = 2, ram: int = 8) -> float:
    if gpu is not None and gpu not in ("T4", "L4"):
        raise ValueError("Only priced T4/L4 configurations are permitted")
    # Include 10 minutes of CPU image/build overhead and up to 1 GiB egress.
    return round(
        (
            (RATES.get(gpu or "", 0) + cpu * RATES["CPU"] + ram * RATES["RAM"]) * seconds / 3600
            + 600 * (cpu * RATES["CPU"] + ram * RATES["RAM"]) / 3600
            + 0.04
        )
        * 1.25,
        5,
    )


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with portalocker.Lock(str(self.path) + ".lock", timeout=5):
            state = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else {"ceiling_usd": CEILING, "storage_reserve_usd": 0.50, "jobs": []}
            )
            yield state
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temp.replace(self.path)

    def report(self) -> dict:
        with self.locked() as state:
            reserved = state["storage_reserve_usd"] + sum(j["reserved_usd"] for j in state["jobs"])
            return {
                **state,
                "reserved_usd": round(reserved, 5),
                "remaining_usd": round(CEILING - reserved, 5),
            }

    def reserve(self, label: str, gpu: str | None, seconds: int = 930) -> str:
        amount = estimate(gpu, seconds)
        with self.locked() as state:
            if any(j["status"] == "running" for j in state["jobs"]):
                raise RuntimeError("A cloud reservation is already active; reconcile it first")
            reserved = state["storage_reserve_usd"] + sum(j["reserved_usd"] for j in state["jobs"])
            if reserved + amount > CEILING:
                raise RuntimeError("Conservative cloud budget would be exceeded")
            key = str(time.time_ns())
            state["jobs"].append(
                {
                    "id": key,
                    "label": label,
                    "gpu": gpu,
                    "max_seconds": seconds,
                    "reserved_usd": amount,
                    "status": "running",
                    "started_at": datetime.now(UTC).isoformat(),
                }
            )
            return key

    def finish(self, key: str, status: str, measured: dict | None = None) -> None:
        with self.locked() as state:
            job = next(j for j in state["jobs"] if j["id"] == key)
            job.update(status=status, measurements=measured or {})
            # Keep full reservation, even after failures, rather than reclaim uncertain charges.
