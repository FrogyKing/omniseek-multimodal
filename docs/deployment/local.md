# Running locally

Use Python 3.12, uv, and Node.js 22.12+ (Node 24 is used in CI). FFmpeg is provided
by the PyAV wheel locally and installed in the Modal image. No separate local
FFmpeg executable or CUDA installation is required.

From the repository root:

```sh
uv sync --locked
npm --prefix frontend ci
npm --prefix frontend run build
uv run python -m omniseek_multimodal.cli demo
uv run python -m omniseek_multimodal.cli serve
```

Open <http://127.0.0.1:8000>. OpenAPI documentation is at
<http://127.0.0.1:8000/docs>. The API serves the production frontend from the same
origin. The demo installation imports real GPU-recorded embeddings; it does not
run a GPU or access credentials. Cached example queries and queries using indexed
images work with cloud inference disabled. Arbitrary new text/image embeddings
require the on-demand backend; no fabricated fallback exists.

For frontend development, start the API and then run
`npm --prefix frontend run dev` in another terminal. Vite proxies `/api` to port
8000. Both localhost and 127.0.0.1 on ports 8000/5173 are allowed write origins.

## Enable indexing and new queries

Reuse an authenticated Modal profile and the existing secret named `huggingface`.
Do not place a Hugging Face token in `.env`, source code or shell arguments.

PowerShell:

```powershell
$env:OMNISEEK_ALLOW_CLOUD = '1'
$env:OMNISEEK_GPU = 'T4'
uv run python -m omniseek_multimodal.cli serve
```

POSIX:

```sh
OMNISEEK_ALLOW_CLOUD=1 OMNISEEK_GPU=T4 uv run python -m omniseek_multimodal.cli serve
```

Creating a collection or uploading files never invokes a GPU. Press **Index** to
index pending files, or the rebuild button to replace all vectors. New uncached
queries also invoke the model and are billed. Indexing and query embedding each
use the same local ledger. The server must be restarted to change cloud settings.
Stop it with Ctrl+C. Keep the `data/` folder private and persistent across runs.

`OMNISEEK_DATA_DIR` selects the local data directory; its default is `data/`.
The `.env.example` is explanatory; the application reads process environment
variables, not `.env` automatically. Run commands from the repository root.

## Reproduce research

```sh
uv run python -m omniseek_multimodal.cli benchmark-dry-run
uv run python -m omniseek_multimodal.cli cost-report
# The following commands explicitly launch billable cloud work:
uv run python -m omniseek_multimodal.cli preflight
uv run python -m omniseek_multimodal.cli gpu-smoke-test --gpu T4
uv run python -m omniseek_multimodal.cli gpu-benchmark --gpu L4
uv run python -m omniseek_multimodal.cli gpu-research
# CPU-only evaluation of recorded vectors:
uv run python -m omniseek_multimodal.cli evaluate
```

`gpu-benchmark` adds a no-prefix query ablation to the same 14-segment corpus.
`gpu-research` measures selective encoders, batch sizes and temporal granularity
in one bounded T4 invocation. It is a separate billable command. GNU Make targets
wrap these commands; on Windows, use the commands above if Make is unavailable.
`benchmark-dry-run` estimates one standard GPU job and launches nothing.

## Checks

```sh
uv run pytest
uv run ruff check src modal_apps scripts tests
uv run ruff format --check src modal_apps scripts tests
uv run mypy
npm --prefix frontend run lint
npm --prefix frontend run build
uv run python -m omniseek_multimodal.cli demo
npm --prefix frontend exec -- playwright install chromium
npm --prefix frontend test
```

Browser tests need the production frontend built and demo installed. They can
start the API automatically. For an isolated test run, choose a separate
`OMNISEEK_DATA_DIR` for both demo installation and test server. The default pytest
command excludes marked GPU tests. Only
`uv run pytest -m gpu -o addopts=''` invokes the explicitly billable live test.
CI never provides cloud credentials or enables GPU execution.
Browser screenshots normally go to ignored test output. Set
`OMNISEEK_CAPTURE_DOCS=1` when running the browser suite to refresh the README
screenshots deliberately.

`uv run python scripts/live_validation.py` is a separate, explicitly billable
integration check. It uploads disposable original fixtures, indexes five
modalities through the API, embeds a new query, restarts with cloud access off,
verifies persistence, then deletes its own collection. It shares the normal local
cost ledger. GPU dependencies, including transitive versions and hashes, are
fixed in `requirements/gpu.txt`; the Modal image uses this lock and uv 0.12.5.
The demo verifies source-byte hashes. Regenerating media with another FFmpeg build
may change those bytes; rerun the explicit GPU benchmark before replacing its
recorded vectors. Git preserves the committed fixture bytes across platforms.

## Troubleshooting

- **Cloud inference is disabled:** use an exact cached example, or opt in as above.
- **Missing demo embeddings:** restore the committed `experiments/results` fixtures.
- **Authentication or gated-model failure:** check `uv run modal secret list` and
  the existing profile; do not print credentials. Run the explicit preflight.
- **Price/budget reservation failure:** inspect `cost-report`. Never erase or edit
  ledger history to bypass a budget check. Reconcile interrupted work with Modal
  before changing a stale running reservation.
- **Interrupted job:** previous indexed files remain; restart the API and retry
  the affected collection explicitly.
- **Large media:** split recordings before upload; the 180-second limit is
  intentional. Temporal matches describe windows, rather than exact action times.
