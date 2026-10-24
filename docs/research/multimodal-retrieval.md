# Multimodal retrieval under a small compute budget

Measured on 2026-10-24. Results and vectors are committed in
[`experiments/results`](../../experiments/results). These are independent, small
diagnostic measurements; no model-vendor benchmark scores are reproduced as
OmniSeek results.

## Research question

How do Matryoshka truncation, query instructions, encoder selection and temporal
segmentation affect retrieval in a compact, mixed-modality collection? Can a
temporary inexpensive GPU build a useful index that remains searchable on CPU?

## Architecture and integration

The pinned model is `google/embeddinggemma-2` at revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`. It combines a text backbone with optional
vision and audio encoders, mean pooling and a shared 768-dimensional output.
Text includes source code. Video consumes sampled visual frames. The measured
full configuration contains 744,371,488 parameters. The application uses
Sentence Transformers 6.2.0, Transformers 5.20.2, PyTorch 2.14.1, Torchvision
0.29.1 and TorchCodec 0.17.0. The complete local environment is fixed in `uv.lock`.

Integration and task instructions follow the [official multimodal
guide](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)
and [model card](https://huggingface.co/google/embeddinggemma-2). Images, audio and
video receive no textual task prefix. Titled text documents use
`title: filename | text: content`; text queries use `SearchQuery`, with
`CodeRetrieval` available when searching code. Local queries cache their prompt
identity and revision, rather than reusing vectors across instruction changes.

## Mathematical formulation

Let the encoder output be \(x\in\mathbb{R}^{768}\). For a supported dimension
\(d\in\{768,512,256,128\}\), define \(P_dx=(x_1,\ldots,x_d)\) and

\[
z_d(x)=\frac{P_dx}{\lVert P_dx\rVert_2},\qquad
s_d(q,x)=z_d(q)^Tz_d(x).
\]

For nonzero finite vectors, this inner product is cosine similarity. A full
unit vector generally satisfies \(\lVert P_dx\rVert_2<1\). Using a raw truncated
dot product introduces a factor \(\lVert P_dq\rVert_2\lVert P_dx\rVert_2\),
which varies by document and can reorder neighbors. Both query and corpus must
be renormalized after slicing. Zero or nonfinite representations are rejected.
Unit tests include a case where omitting renormalization reverses the correct
ranking. This follows the flexible nested representation idea in
[Kusupati et al., Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147).

Canonical vectors remain 768-dimensional on disk. Experiments materialize
shortened arrays and measure raw float32 and serialized index bytes. Selecting
128d in the interface changes the retrieval computation; it does not reclaim
space from the canonical database.

## Dataset and relevance judgments

The main corpus has 12 original CC0 files and 14 segments: three short text
documents, two Python examples (two chunks each), three geometric illustrations,
two synthesized three-second sounds, and two four-second animations. All are
reproducible with `scripts/generate_demo.py`, seed 42. No downloaded copyrighted
media is included. The extra `transition.mp4` is a four-second diagnostic with a
red ball in [0,2) and a sailboat in [2,4).

Ten manually authored queries and their binary relevant filenames are fixed in
`assets/demo/manifest.json`. They cover two text, two code, two image, two audio,
one video and one mixed query. The sailboat query judges the prose, image and
animation relevant. All other queries have one relevant file. Judgments are
declared before ranking; no learned reranker or relevance fitting is used.

Evaluation searches all modalities together. Segment scores are sorted and
filenames deduplicated, equivalent to each file's best segment score. The UI can
show several segment results per file; reported metrics are file-level. This is
a deliberately easy diagnostic corpus, without a train/test split, independent
annotators, natural photographic variation, long-form speech or difficult code
repositories. Confidence intervals on broad retrieval quality would be misleading.

## Hardware and protocol

The local machine has 16 logical CPUs and approximately 63.5 GiB RAM on Windows
11. Its Intel Arc integrated GPU is not used for model inference. FAISS CPU
experiments fix one thread, build a fresh index for each setting, warm it once,
then perform 100 searches of the ten-query batch. The reported latency is batch
elapsed time divided by ten; it excludes model inference, HTTP, decoding,
normalization and corpus loading. The API reports a broader retrieval interval,
so its displayed latency will differ.

Modal workers request two physical CPU cores, an 8 GiB RAM ceiling, one GPU,
zero retries, zero warm containers and a 600-second execution timeout plus a
300-second startup timeout. T4 uses float32; L4 uses native bfloat16. Float16 is
not used. A preliminary T4 bfloat16 run succeeded through emulation, then was
superseded by float32 for the canonical demo. Torchvision was added after an
initial processor-loading failure. Both failed attempts remain in the cost ledger.

## Metrics

- Recall@3: relevant files retrieved in the first three, divided by total relevant files.
- Precision@3: relevant files in the first three, divided by three.
- MRR: mean reciprocal position of the first relevant file over the full ranking.
- nDCG@3: binary-relevance discounted gain, with log2(rank+1) discounts and ideal normalization.
- Top-3 stability: set intersection with the corresponding 768d top three, divided by three; it is not an order-sensitive correlation.
- Storage: array bytes and actual FAISS serialization length.
- Build/latency: wall-clock timings of index construction and warm CPU search.

All denominators are explicit. In this corpus the maximum average Precision@3
is 0.4 because nine queries have one relevant file and one has three.

## Measured results

### Matryoshka compression, exact retrieval

| Dimensions | Vector bytes | Flat index bytes | Recall@3 | Precision@3 | MRR | nDCG@3 | Top-3 stability |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 768 | 43,008 | 43,053 | 1.000 | 0.400 | 1.000 | 1.000 | 1.000 |
| 512 | 28,672 | 28,717 | 1.000 | 0.400 | 1.000 | 1.000 | 0.933 |
| 256 | 14,336 | 14,381 | 1.000 | 0.400 | 0.950 | 0.963 | 0.833 |
| 128 | 7,168 | 7,213 | 1.000 | 0.400 | 1.000 | 1.000 | 0.800 |

Raw vector storage falls exactly 6× between 768d and 128d; headers and metadata
reduce the total-file ratio. The 256d representation moves `noise.wav` ahead of
`mountain.png` for “Snow covered mountains.” Image-query MRR therefore falls to
0.75 in that group (two queries). This nonmonotonic result is retained; it does
not establish that 128d is generally better than 256d. Per-modality metrics and
every ranking are in `retrieval.json`.

Warm exact searches were approximately 0.0006–0.0014 ms per query in the recorded
run. At this tiny scale, timer overhead, CPU scheduling and batching dominate;
the numbers do not support a production speedup claim. HNSW produced the same
relevance scores on these 14 vectors, while using 45,346 bytes at 768d versus
43,053 for FlatIP. Its recorded 768d build was 0.356 ms versus 0.040 ms for
FlatIP. There is no reason to prefer approximation for this collection. The
FAISS implementation builds on [Johnson et al.](https://arxiv.org/abs/1702.08734);
this project does not reproduce that paper's large-scale GPU experiments.

### Prompt ablation

| Query instruction | Recall@3 | MRR | nDCG@3 |
| --- | ---: | ---: | ---: |
| SearchQuery | 1.000 | 1.000 | 1.000 |
| No prefix | 1.000 | 0.933 | 0.950 |

Both explicit code queries also ranked their relevant source file first with
`CodeRetrieval`. There are only two code questions, so no general instruction
advantage can be inferred for code. The prompt experiment compares the same
corpus embeddings, full 768d, and the same T4 float32 run.

### GPU execution and encoder footprint

| GPU / precision | 36 embeddings, load included | Local invocation wall time | Peak allocated VRAM | App receipt, all resources |
| --- | ---: | ---: | ---: | ---: |
| T4 / float32 | 10.694 s | 32.902 s | 3.29 GiB | $0.004991 |
| L4 / bfloat16 | 10.906 s | 41.752 s | 1.68 GiB | $0.008521 |

Cold loading, container allocation and client overhead matter more than kernel
time here. These are single runs with different precision, rather than controlled
GPU throughput comparisons. T4 is the default because complete execution is
verified and its posted hourly rate is lower; “fastest GPU” is not established.

| Encoder configuration | Measured parameters | Maximum absolute vector difference from full |
| --- | ---: | ---: |
| Text | 271,002,624 | 0.0 |
| Text + vision | 438,760,448 | 0.0 |
| Text + audio | 576,613,664 | 0.0 |
| Full | 744,371,488 | Reference |

The compatibility check used four text inputs, one illustration and one sound
with float32 on the same T4 invocation. It validates those samples and the common
projection, without proving bitwise equality for every possible input. The app
therefore selects the required encoders for new query/index requests while keeping
the same checkpoint revision and output representation.

### Warm text batching

Five repetitions after warm-up, eight short texts, full model on T4 float32:

| Batch size | Median seconds / 8 texts | Texts / second |
| ---: | ---: | ---: |
| 1 | 0.2857 | 28.0 |
| 2 | 0.1485 | 53.9 |
| 4 | 0.0864 | 92.6 |
| 8 | 0.0498 | 160.6 |

This supports batching short texts on that configuration. It does not predict
image/audio/video throughput or cold-start latency. Production batches default
to four because mixed media have different token and memory costs. Peak allocated
memory is measured; GPU occupancy/utilization percentages were not collected.

### Temporal granularity

For the generated two-scene clip, native MP4 encoding returned one [0,4) interval
for both queries. The red-ball and sailboat cosine scores were 0.691 and 0.741.
One-second windows matched [1,2) and [3,4); two-second windows matched [0,2) and
[2,4). A simple mean-RGB scene-change heuristic also found the boundary at two
seconds and produced the same intervals and vectors as fixed two-second windows.
That agreement is a consequence of this constructed clip; it is not evidence
that the heuristic is a robust scene detector.

Concatenating the synthesized bell and noise produced a six-second recording.
Three-second windows matched the correct [0,3) bell and [3,6) noise portions,
with scores 0.734 and 0.700. A single six-second window matched both queries to
the whole recording, with scores 0.725 and 0.678. The 1.5-second experiment also
located the correct portions. Window length controls localization and index
size; higher similarity alone is not a validated measure of segmentation quality.
The application retains simple ten-second windows as a bounded default, with
coarse-match labels in the media viewer.

## Error analysis

The 256d mountain/noise inversion shows that Recall@3 can hide a harmful top-1
change. Cross-modal score distributions also admit semantically unrelated sounds
among low-ranked results for code or visual queries. Scores are uncalibrated
cosines, not probabilities, and no global confidence threshold is claimed.
Image-to-video retrieval works for the sailboat drawing and its animation; a
static door cannot establish whether a door opens. A sampled single-frame window
cannot establish motion direction. Original media and line/timestamp metadata
remain available for human inspection.

## Cost and reproducibility

Authenticated Modal billing receipts show **$0.05357479 gross app usage** across
the initial download, failed dependency attempt, preliminary T4 run, canonical
T4 run, L4 run, temporal/encoder research, two live API calls and the final
locked-image text test. Seven GPU invocations succeeded; one stopped on the missing
Torchvision dependency. The workspace's net billed cost was **$0.00 after credits**
at that snapshot. The final `costs.json` is authoritative for the recorded total.
A local conservative reservation of **$3.40153** includes startup, bounded
execution, failures, build overhead, egress and $0.50 for model-cache storage.
Ongoing storage is separate from completed app charges. Provider data may be
revised and local estimates are not a provider-level cap.

Reproduction commands, including the explicit billable boundary, are in
[`docs/deployment/local.md`](../deployment/local.md). `evaluate` consumes only
committed real vectors and makes no cloud calls. CPU tests never impersonate
GPU validation. Native MP4 decoding, all five input modalities, selective encoder
loading and the dimensionality experiments were actually executed on Modal.

## Limitations and conclusions

The end-to-end system proves persistent cross-modal retrieval with a small GPU
budget. It provides reproducible vector-compression, prompting, batching and
temporal diagnostics. The main benchmark is too small and synthetic for broad
quality claims. There are no multilingual, long-recording, PDF/OCR, speech
transcription, captioning, approximate-search-at-scale, or ranking-calibration
evaluations. Scene-aware segmentation is research-only. Larger independent
datasets, harder negatives, interval relevance judgments, repeated GPU trials
and GPU utilization profiling are the next experiments. Compression should be
chosen per workload; the full representation remains the default.
