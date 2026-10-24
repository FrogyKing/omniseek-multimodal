import argparse
import json

from .config import ROOT, Settings
from .costs import Ledger, estimate


def main():
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="omniseek")
    parser.add_argument(
        "command",
        choices=[
            "serve",
            "cost-report",
            "benchmark-dry-run",
            "preflight",
            "gpu-smoke-test",
            "gpu-benchmark",
            "gpu-research",
            "demo",
            "evaluate",
        ],
    )
    parser.add_argument("--gpu", choices=["T4", "L4"], default="T4")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = Settings(gpu=args.gpu)
    ledger = Ledger(settings.data_dir / "cost-ledger.json")
    if args.command == "serve":
        import uvicorn

        uvicorn.run("omniseek_multimodal.api:app", host="127.0.0.1", port=args.port)
    elif args.command == "cost-report":
        report = ledger.report()
        report["jobs"] = [
            {k: j[k] for k in ("label", "gpu", "status", "reserved_usd")} for j in report["jobs"]
        ]
        print(json.dumps(report, indent=2))
    elif args.command == "benchmark-dry-run":
        print(
            json.dumps(
                {
                    "gpu": args.gpu,
                    "max_seconds": 930,
                    "reserved_usd": estimate(args.gpu),
                    "budget": {
                        k: ledger.report()[k] for k in ("ceiling_usd", "reserved_usd", "remaining_usd")
                    },
                    "billable_jobs_launched": 0,
                },
                indent=2,
            )
        )
    elif args.command == "preflight":
        import modal

        from .cloud import check_billing
        from .modal_apps.embedding_service import app, preflight

        check_billing("T4")

        key = ledger.reserve("HF authentication and cached model download", None, 630)
        try:
            with modal.enable_output(), app.run():
                result = preflight.remote()
            ledger.finish(key, "succeeded", result)
            print(json.dumps(result, indent=2))
        except Exception:
            ledger.finish(key, "failed")
            raise
    elif args.command == "gpu-research":
        import modal

        from .cloud import check_billing
        from .media import prepare
        from .modal_apps.embedding_service import app
        from .modal_apps.gpu_benchmark import research

        check_billing("T4")
        manifest = json.loads((ROOT / "assets/demo/manifest.json").read_text())
        payloads = [
            p for name in manifest["files"] for p in prepare(ROOT / "assets/demo" / name, name, settings)[2]
        ]
        key = ledger.reserve("encoder, batching and temporal research", "T4")
        try:
            with modal.enable_output(), app.run():
                result = research.remote(payloads, (ROOT / "assets/demo/transition.mp4").read_bytes())
            ledger.finish(key, "succeeded", result)
            (ROOT / "experiments/results/gpu-research.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
        except Exception:
            ledger.finish(key, "failed")
            raise
    elif args.command == "demo":
        from .demo import install_demo

        print(json.dumps(install_demo(settings), indent=2))
    elif args.command in ("gpu-smoke-test", "gpu-benchmark"):
        from .demo import gpu_demo

        gpu_demo(
            Settings(data_dir=settings.data_dir, allow_cloud=True, gpu=args.gpu),
            benchmark=args.command == "gpu-benchmark",
        )
    elif args.command == "evaluate":
        from .evaluation import evaluate_demo

        evaluate_demo(ROOT)


if __name__ == "__main__":
    main()
