"""Explicit file-level relevance; exhaustive rankings; reproducible CPU metrics."""

import json
import platform
import time
from pathlib import Path

import faiss
import numpy as np

from .config import DIMENSIONS
from .vectors import normalize


def metrics(ranking: list[str], relevant: set[str], k: int = 3) -> dict:
    hits = np.array([int(item in relevant) for item in ranking[:k]], dtype=float)
    dcg = float(np.sum(hits / np.log2(np.arange(len(hits)) + 2)))
    ideal = float(np.sum(1 / np.log2(np.arange(min(len(relevant), k)) + 2)))
    reciprocal = next((1 / (i + 1) for i, name in enumerate(ranking) if name in relevant), 0.0)
    return {
        "recall_at_3": float(hits.sum() / len(relevant)),
        "precision_at_3": float(hits.sum() / k),
        "mrr": reciprocal,
        "ndcg_at_3": dcg / ideal if ideal else 0.0,
    }


def unique_ranking(indices, names):
    return list(dict.fromkeys(names[int(i)] for i in indices if i >= 0))


def evaluate_demo(root: Path):
    faiss.omp_set_num_threads(1)
    results_dir = root / "experiments/results"
    run = json.loads((results_dir / "demo-run.json").read_text())
    manifest = json.loads((root / "assets/demo/manifest.json").read_text())
    data = np.load(results_dir / "demo-embeddings.npz", allow_pickle=False)
    names = [record["name"] for record in run["records"] for _ in range(record["count"])]
    baseline_rankings = None
    results = []
    for dimension in DIMENSIONS:
        corpus = normalize(data["corpus"], dimension)
        queries = normalize(data["queries"], dimension)
        for approximate in (False, True):
            start = time.perf_counter()
            index = (
                faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_INNER_PRODUCT)
                if approximate
                else faiss.IndexFlatIP(dimension)
            )
            index.add(corpus)
            build_ms = (time.perf_counter() - start) * 1000
            index.search(queries, len(corpus))
            times = []
            for _ in range(100):
                start = time.perf_counter()
                _, ids = index.search(queries, len(corpus))
                times.append((time.perf_counter() - start) * 1000 / len(queries))
            rankings = [unique_ranking(row, names) for row in ids]
            if baseline_rankings is None:
                baseline_rankings = rankings
            per_query = [
                {
                    "query": q["text"],
                    "modality": q["modality"],
                    "ranking": ranking,
                    **metrics(ranking, set(q["relevant"])),
                    "top3_overlap_with_768": len(set(ranking[:3]) & set(base[:3])) / 3,
                }
                for q, ranking, base in zip(manifest["queries"], rankings, baseline_rankings, strict=True)
            ]
            keys = ["recall_at_3", "precision_at_3", "mrr", "ndcg_at_3", "top3_overlap_with_768"]
            aggregate = {key: float(np.mean([q[key] for q in per_query])) for key in keys}
            by_modality = {}
            for modality in ("text", "code", "image", "audio", "video", "mixed"):
                group = [q for q in per_query if q["modality"] == modality]
                by_modality[modality] = {key: float(np.mean([q[key] for q in group])) for key in keys}
            results.append(
                {
                    "dimension": dimension,
                    "index": "HNSW" if approximate else "FlatIP",
                    **aggregate,
                    "vector_bytes": corpus.nbytes,
                    "index_bytes": len(faiss.serialize_index(index)),
                    "build_ms": build_ms,
                    "query_ms_median": float(np.median(times)),
                    "query_ms_p95": float(np.percentile(times, 95)),
                    "by_modality": by_modality,
                    "per_query": per_query,
                }
            )
    # Within-modality image query diagnostics, including cross-modal targets, exclude self.
    corpus = normalize(data["corpus"])
    image_diagnostics = []
    for record in run["records"]:
        if record["name"].endswith(".png"):
            query = corpus[record["offset"]]
            ids = np.argsort(-(corpus @ query))
            ranking = [name for name in unique_ranking(ids, names) if name != record["name"]]
            image_diagnostics.append({"query": record["name"], "top3_excluding_self": ranking[:3]})
    prompt_ablation = []
    t4_data = np.load(results_dir / "t4-embeddings.npz", allow_pickle=False)
    if len(t4_data["unprompted"]) == len(manifest["queries"]):
        for prompt, query_vectors in [("SearchQuery", data["queries"]), ("no prefix", t4_data["unprompted"])]:
            scores = normalize(query_vectors) @ corpus.T
            ranks = [unique_ranking(np.argsort(-row), names) for row in scores]
            values = [
                metrics(rank, set(query["relevant"]))
                for rank, query in zip(ranks, manifest["queries"], strict=True)
            ]
            prompt_ablation.append(
                {"prompt": prompt, **{key: float(np.mean([v[key] for v in values])) for key in values[0]}}
            )
    code_prompt_ablation = []
    offset = 0
    for i, query in enumerate(manifest["queries"]):
        if query["modality"] == "code":
            ranked = unique_ranking(np.argsort(-(corpus @ normalize(data["code_queries"][offset]))), names)
            code_prompt_ablation.append(
                {"query": query["text"], "ranking": ranked, **metrics(ranked, set(query["relevant"]))}
            )
            offset += 1
    report = {
        "status": "measured",
        "dataset": "12 original CC0 diagnostic assets, 10 hand-authored queries",
        "relevance_unit": "file, max segment score; all indexed modalities searched",
        "hardware": platform.processor(),
        "platform": platform.platform(),
        "faiss_version": faiss.__version__,
        "threads": 1,
        "repeats": 100,
        "revision": run["model_revision"],
        "results": results,
        "image_queries": image_diagnostics,
        "prompt_ablation": prompt_ablation,
        "code_prompt_ablation": code_prompt_ablation,
    }
    (results_dir / "retrieval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            [{k: v for k, v in row.items() if k not in ("by_modality", "per_query")} for row in results],
            indent=2,
        )
    )
    return report
