"""Export only this project's provider receipts, without account credentials."""

import dataclasses
import json
from datetime import UTC, datetime, timedelta

import modal

from omniseek_multimodal.config import ROOT, Settings
from omniseek_multimodal.costs import Ledger


def main():
    ledger = Ledger(Settings().data_dir / "cost-ledger.json").report()
    start = min(datetime.fromisoformat(j["started_at"]) for j in ledger["jobs"]).replace(
        minute=0, second=0, microsecond=0
    )
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    workspace = modal.Workspace.from_context()
    rows = workspace.billing.report(start=start, end=end, resolution="h")
    project_rows = [dataclasses.asdict(row) for row in rows if row.description == "omniseek-embeddings"]
    total = sum(float(row["cost"]) for row in project_rows)
    summary = workspace.billing.summary()
    snapshot = {
        "captured_at": datetime.now(UTC).isoformat(),
        "provider": "Modal authenticated workspace billing API",
        "gross_project_app_cost_usd": total,
        "workspace_net_billed_usd_at_snapshot": float(summary.billed_cost),
        "conservative_reserved_usd": ledger["reserved_usd"],
        "storage_reserve_usd": ledger["storage_reserve_usd"],
        "note": "Provider snapshot may be revised; app receipts include CPU, memory, GPU and egress. "
        "Ongoing model-volume storage is separate. Net billing includes workspace credits.",
        "receipts": project_rows,
    }
    path = ROOT / "experiments/results/costs.json"
    path.write_text(json.dumps(snapshot, default=str, indent=2), encoding="utf-8")
    print(
        f"Project app cost: ${total:.8f}; workspace billed: ${summary.billed_cost}; "
        f"conservative reservation: ${ledger['reserved_usd']:.5f}"
    )


if __name__ == "__main__":
    main()
