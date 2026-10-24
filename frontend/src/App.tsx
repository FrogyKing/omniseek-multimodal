import { useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import {
  Activity,
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  AudioLines,
  Box,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Code2,
  Database,
  FileText,
  FlaskConical,
  Folder,
  Image as ImageIcon,
  Layers3,
  LoaderCircle,
  Plus,
  Radio,
  Search,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Upload,
  Video,
  X,
} from "lucide-react";
import { api, post } from "./api";
import type {
  Collection,
  DemoQuery,
  Experiment,
  Experiments,
  FileItem,
  Job,
  Modality,
  Result,
  SearchResponse,
  SystemStatus,
} from "./api";

type Page =
  | "Search"
  | "Collections"
  | "Media explorer"
  | "Indexing jobs"
  | "Embedding lab"
  | "System status";
const pages = [
  { name: "Search", icon: Search },
  { name: "Collections", icon: Folder },
  { name: "Media explorer", icon: Layers3 },
  { name: "Indexing jobs", icon: Activity },
  { name: "Embedding lab", icon: FlaskConical },
  { name: "System status", icon: Radio },
] as const;
const modalities = ["all", "text", "image", "audio", "video", "code"] as const;
const icons = {
  text: FileText,
  image: ImageIcon,
  audio: AudioLines,
  video: Video,
  code: Code2,
};
const plural: Record<string, string> = {
  all: "Everything",
  text: "Text",
  image: "Images",
  audio: "Audio",
  video: "Video",
  code: "Code",
};
const timeLabel = (seconds = 0) =>
  `${Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0")}:${Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0")}`;
const bytes = (n: number) =>
  n > 1048576
    ? `${(n / 1048576).toFixed(1)} MB`
    : `${(n / 1024).toFixed(1)} KB`;

function Media({ result }: { result: Result }) {
  const source = `${result.url}${result.metadata.start !== undefined ? `#t=${result.metadata.start},${result.metadata.end}` : ""}`;
  if (result.modality === "image")
    return <img src={source} alt={result.name} />;
  if (result.modality === "video")
    return <video src={source} controls preload="metadata" />;
  if (result.modality === "audio")
    return (
      <div className="audio-preview">
        <AudioLines size={64} strokeWidth={1} />
        <audio src={source} controls preload="metadata" />
      </div>
    );
  return <pre>{result.metadata.text}</pre>;
}

export default function App() {
  const [page, setPage] = useState<Page>("Search");
  const [collections, setCollections] = useState<Collection[]>([]);
  const [files, setFiles] = useState<FileItem[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [experiments, setExperiments] = useState<Experiments>({
    status: "not_run",
    results: [],
  });
  const [queries, setQueries] = useState<DemoQuery[]>([]);
  const [query, setQuery] = useState("A sailboat on the blue ocean");
  const [collectionId, setCollectionId] = useState("");
  const [modality, setModality] = useState<(typeof modalities)[number]>("all");
  const [dimension, setDimension] = useState(768);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [selected, setSelected] = useState<Result | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [uploading, setUploading] = useState(false);
  const uploadRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLInputElement>(null);
  const queryRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [c, f, j, s, e, q] = await Promise.all([
        api<Collection[]>("/collections"),
        api<{ items: FileItem[] }>("/files?limit=100"),
        api<Job[]>("/indexing/jobs"),
        api<SystemStatus>("/system/status"),
        api<Experiments>("/experiments"),
        api<DemoQuery[]>("/demo/queries"),
      ]);
      setCollections(c);
      setFiles(f.items);
      setJobs(j);
      setStatus(s);
      setExperiments(e);
      setQueries(q);
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      void refresh();
    }, 0);
    const poll = setInterval(() => {
      void refresh();
    }, 5000);
    return () => {
      clearTimeout(timer);
      clearInterval(poll);
    };
  }, [refresh]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setPage("Search");
        queryRef.current?.focus();
      }
      if (e.key === "Escape") {
        setSelected(null);
        setShowCreate(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const runSearch = async (
    text = query,
    imageId?: string,
    searchModality = modality,
  ) => {
    setBusy(true);
    setError("");
    setPage("Search");
    try {
      const result = await post<SearchResponse>("/search", {
        query: imageId ? "" : text,
        image_file_id: imageId,
        collection_id: collectionId || null,
        modality: searchModality === "all" ? null : searchModality,
        dimension,
        k: 18,
      });
      setResponse(result);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const createCollection = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      const c = await post<Collection>("/collections", { name, description });
      setCollectionId(c.id);
      setShowCreate(false);
      setName("");
      setDescription("");
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const upload = async (list: FileList | null, imageQuery = false) => {
    if (!list?.length) return;
    let target = collectionId || collections[0]?.id;
    setUploading(true);
    setError("");
    try {
      if (!target) {
        const c = await post<Collection>("/collections", {
          name: "My collection",
          description: "",
        });
        target = c.id;
      }
      for (const file of Array.from(list)) {
        const body = new FormData();
        body.append("collection_id", target);
        body.append("file", file);
        const result = await api<{ id: string; duplicate: boolean }>(
          "/uploads",
          { method: "POST", body },
        );
        if (imageQuery) {
          setQuery(`Image: ${file.name}`);
          await runSearch("", result.id);
        }
      }
      if (!imageQuery) {
        setNotice(
          `${list.length} file${list.length === 1 ? "" : "s"} received. Start indexing to make them searchable.`,
        );
        setCollectionId(target);
        setPage("Collections");
      }
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setUploading(false);
      if (uploadRef.current) uploadRef.current.value = "";
      if (imageRef.current) imageRef.current.value = "";
    }
  };

  const deleteFile = async (id: string) => {
    if (!confirm("Delete this file and its indexed segments?")) return;
    try {
      await api(`/files/${id}`, { method: "DELETE" });
      await refresh();
      setResponse(null);
    } catch (err) {
      setError((err as Error).message);
    }
  };
  const deleteCollection = async (id: string) => {
    if (
      !confirm(
        "Delete this collection, all original files, and all indexed segments?",
      )
    )
      return;
    try {
      await api(`/collections/${id}`, { method: "DELETE" });
      if (collectionId === id) setCollectionId("");
      await refresh();
      setResponse(null);
    } catch (err) {
      setError((err as Error).message);
    }
  };
  const startIndex = async (id: string, rebuild = false) => {
    setError("");
    try {
      await post("/indexing/jobs", { collection_id: id, rebuild });
      setPage("Indexing jobs");
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  };
  const openFile = async (file: FileItem) => {
    let text = "";
    if (file.modality === "text" || file.modality === "code") {
      try {
        const res = await fetch(`/api/files/${file.id}/content`);
        text = await res.text();
      } catch {
        setError("Could not open original");
        return;
      }
    }
    setSelected({
      id: file.id,
      file_id: file.id,
      name: file.name,
      modality: file.modality,
      score: 0,
      url: `/api/files/${file.id}/content`,
      metadata: { text },
    });
  };
  const shownFiles = files.filter(
    (f) =>
      (!collectionId || f.collection_id === collectionId) &&
      (modality === "all" || f.modality === modality),
  );
  const realExperiments = experiments.results.filter(
    (e) => e.index === "FlatIP",
  );
  const selectedExperiment = realExperiments.find(
    (e) => e.dimension === dimension,
  );
  const activeJobs = jobs.filter(
    (j) => j.status === "running" || j.status === "queued",
  ).length;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage("Search");
          }}
          aria-label="OmniSeek home"
        >
          <span className="brand-mark">
            <Search size={23} strokeWidth={2.5} />
          </span>
          <span>
            omni<span className="brand-light">seek</span>
            <sup> / </sup>
          </span>
        </a>
        <div className="workspace">
          <div className="workspace-icon">O</div>
          <div>
            <strong>Personal workspace</strong>
            <span>Local environment</span>
          </div>
          <ChevronDown size={14} />
        </div>
        <p className="nav-label">WORKSPACE</p>
        <nav>
          {pages.map(({ name: item, icon: Icon }) => (
            <button
              key={item}
              aria-label={item}
              className={`nav-item ${page === item ? "active" : ""}`}
              onClick={() => {
                setPage(item);
                setNotice("");
              }}
            >
              <Icon size={18} />
              <span>{item}</span>
              {item === "Indexing jobs" && activeJobs > 0 ? (
                <span className="nav-count">{activeJobs}</span>
              ) : item === "Embedding lab" ? (
                <span className="tiny-tag">LAB</span>
              ) : null}
            </button>
          ))}
        </nav>
        <div className="sidebar-collections">
          <div className="section-label">
            <span>COLLECTIONS</span>
            <button
              onClick={() => setShowCreate(true)}
              aria-label="New collection"
            >
              <Plus size={15} />
            </button>
          </div>
          {collections.slice(0, 4).map((c) => (
            <button
              className="collection-nav"
              key={c.id}
              onClick={() => {
                setCollectionId(c.id);
                setPage("Collections");
              }}
            >
              <span className="collection-dot" />
              <span>{c.name.replace(" / demo collection", "")}</span>
              <small>{c.file_count}</small>
            </button>
          ))}
          {!collections.length && (
            <p className="muted small">Your collections appear here.</p>
          )}
        </div>
        <div className="sidebar-bottom">
          <div className="engine-card">
            <span className="green-dot" />
            <strong>EmbeddingGemma 2</strong>
            <span className="engine-meta">
              768 dimensions · Modal {status?.gpu || "T4"}
            </span>
            <div className="engine-line">
              <span>
                {status?.cloud_enabled
                  ? "Cloud inference enabled"
                  : "Local retrieval ready"}
              </span>
              <ArrowUpRight size={13} />
            </div>
          </div>
          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="help"
          >
            <CircleHelp size={16} /> API documentation{" "}
            <ArrowUpRight size={13} />
          </a>
          <div className="profile">
            <span>LG</span>
            <div>
              <strong>Personal workspace</strong>
              <small>OmniSeek · v0.1</small>
            </div>
            <Box size={16} />
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div>
            <span className="muted">Workspace</span>
            <ChevronRight size={13} />
            <span>{page}</span>
          </div>
          <div className="topbar-right">
            <span className="local-chip">
              <span className="green-dot" /> Local-first
            </span>
            <button
              className="primary compact"
              onClick={() => uploadRef.current?.click()}
              disabled={uploading}
            >
              <Upload size={14} /> {uploading ? "Uploading…" : "Upload files"}
            </button>
          </div>
        </header>
        <main>
          {error && (
            <div className="alert error" role="alert">
              <span>{error}</span>
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {notice && (
            <div className="alert notice" role="status">
              <Check size={16} />
              <span>{notice}</span>
              <button aria-label="Dismiss notice" onClick={() => setNotice("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {page === "Search" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">
                    <span className="line" /> MULTIMODAL INTELLIGENCE
                  </div>
                  <h1>
                    Find the signal<span>.</span>
                  </h1>
                  <p>One search. Every kind of knowledge.</p>
                </div>
                <div className="heading-index">
                  <Database size={16} />
                  <div>
                    <strong>
                      {status?.segments ?? "—"} <span>segments indexed</span>
                    </strong>
                    <small>
                      Across {status?.collections ?? "—"} collections
                    </small>
                  </div>
                </div>
              </div>
              <section className="search-panel">
                <form
                  className="search-form"
                  onSubmit={(e) => {
                    e.preventDefault();
                    void runSearch();
                  }}
                >
                  <Search size={22} strokeWidth={1.6} />
                  <input
                    ref={queryRef}
                    aria-label="Search your knowledge"
                    placeholder="Describe what you're looking for…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                  <button
                    className="icon-button image-query"
                    type="button"
                    aria-label="Search with an image"
                    title="Search with an image"
                    onClick={() => imageRef.current?.click()}
                  >
                    <ImageIcon size={19} />
                  </button>
                  <button
                    className="search-submit"
                    aria-label="Run search"
                    type="submit"
                    disabled={busy || !query.trim()}
                  >
                    {busy ? (
                      <LoaderCircle className="spin" size={18} />
                    ) : (
                      <ArrowRight size={20} />
                    )}
                  </button>
                </form>
                <div className="search-controls">
                  <div className="collection-select">
                    <Folder size={14} />
                    <select
                      aria-label="Search collection"
                      value={collectionId}
                      onChange={(e) => setCollectionId(e.target.value)}
                    >
                      <option value="">All collections</option>
                      {collections.map((c) => (
                        <option key={c.id} value={c.id}>
                          {c.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <span className="control-divider" />
                  <span className="model-label">
                    <span className="green-dot" /> EmbeddingGemma 2
                  </span>
                  <div className="dimension-select">
                    <SlidersHorizontal size={13} />
                    <select
                      aria-label="Embedding dimension"
                      value={dimension}
                      onChange={(e) => setDimension(Number(e.target.value))}
                    >
                      {[768, 512, 256, 128].map((d) => (
                        <option value={d} key={d}>
                          {d} dimensions
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </section>
              <div className="suggestions">
                <span>TRY A SIGNAL</span>
                {queries.slice(1, 4).map((q) => (
                  <button
                    key={q.text}
                    onClick={() => {
                      setQuery(q.text);
                      void runSearch(q.text);
                    }}
                  >
                    {q.text}
                    <ArrowUpRight size={12} />
                  </button>
                ))}
              </div>
              <div className="results-toolbar">
                <div className="modality-tabs">
                  {modalities.map((m) => {
                    const Icon = m === "all" ? Layers3 : icons[m];
                    return (
                      <button
                        key={m}
                        className={modality === m ? "selected" : ""}
                        onClick={() => setModality(m)}
                      >
                        <Icon size={15} />
                        {plural[m]}
                      </button>
                    );
                  })}
                </div>
                <span className="result-count">
                  {response
                    ? `${response.results.filter((r) => modality === "all" || r.modality === modality).length} results`
                    : `${status?.files ?? 0} files available`}
                </span>
              </div>
              {response ? (
                <>
                  <div className="results-meta">
                    <span>
                      SEMANTIC MATCHES <ArrowDown size={11} />{" "}
                      <small>Ranked by cosine similarity</small>
                    </span>
                    <span>
                      {response.retrieval_ms.toFixed(2)} ms retrieval ·{" "}
                      {response.dimension}d{" "}
                      <span className="cache-label">
                        {response.cache_hit
                          ? "Cached embedding"
                          : "New GPU embedding"}
                      </span>
                    </span>
                  </div>
                  <div className="results-grid">
                    {response.results
                      .filter(
                        (r) => modality === "all" || r.modality === modality,
                      )
                      .map((r, i) => (
                        <ResultCard
                          key={r.id}
                          result={r}
                          rank={i + 1}
                          onClick={() => setSelected(r)}
                        />
                      ))}
                  </div>
                  {!response.results.length && (
                    <Empty
                      title="No indexed segments yet"
                      body="Upload files to a collection and run indexing to start searching."
                    />
                  )}
                </>
              ) : (
                <>
                  <div className="section-title">
                    <h2>Your knowledge, connected</h2>
                    <span>Five modalities. One embedding space.</span>
                  </div>
                  <div className="modality-grid">
                    {(["text", "image", "audio", "video", "code"] as const).map(
                      (m) => {
                        const Icon = icons[m];
                        const count = files.filter(
                          (f) => f.modality === m,
                        ).length;
                        return (
                          <button
                            key={m}
                            className={`modality-card ${m}`}
                            onClick={() => {
                              setModality(m);
                              setPage("Media explorer");
                            }}
                          >
                            <div className="modality-card-top">
                              <Icon size={20} strokeWidth={1.5} />
                              <ArrowUpRight size={15} />
                            </div>
                            <div className="modality-art" aria-hidden="true">
                              {m === "audio" ? (
                                <div className="wave">
                                  {Array.from({ length: 28 }, (_, i) => (
                                    <i
                                      key={i}
                                      style={{
                                        height: `${12 + ((i * 17 + 5) % 45)}px`,
                                      }}
                                    />
                                  ))}
                                </div>
                              ) : m === "image" ? (
                                <div className="mini-mountain" />
                              ) : m === "video" ? (
                                <div className="mini-film">
                                  <span />
                                  <span />
                                  <span />
                                </div>
                              ) : m === "code" ? (
                                <div className="mini-code">
                                  <i />
                                  <i />
                                  <i />
                                  <i />
                                </div>
                              ) : (
                                <div className="mini-document">
                                  <i />
                                  <i />
                                  <i />
                                  <i />
                                </div>
                              )}
                            </div>
                            <h3>{plural[m]}</h3>
                            <span>
                              {count} file{count !== 1 ? "s" : ""}
                              <ChevronRight size={13} />
                            </span>
                          </button>
                        );
                      },
                    )}
                  </div>
                  <div className="demo-strip">
                    <span className="demo-icon">
                      <Sparkles size={19} />
                    </span>
                    <div>
                      <strong>Start with a small discovery.</strong>
                      <p>
                        Original demo media, real GPU embeddings. Explore
                        without a cloud call.
                      </p>
                    </div>
                    <button
                      onClick={() => {
                        setQuery(queries[0]?.text || query);
                        void runSearch(queries[0]?.text || query);
                      }}
                    >
                      Explore the demo <ArrowRight size={15} />
                    </button>
                  </div>
                  <div className="section-title query-title">
                    <h2>Explore a different angle</h2>
                    <span>Cached example queries · no inference cost</span>
                  </div>
                  <div className="query-grid">
                    {queries.slice(4, 10).map((q) => {
                      const Icon =
                        q.modality in icons
                          ? icons[q.modality as Modality]
                          : Search;
                      return (
                        <button
                          key={q.text}
                          onClick={() => {
                            setQuery(q.text);
                            setModality("all");
                            void runSearch(q.text, undefined, "all");
                          }}
                        >
                          <Icon size={16} />
                          <span>{q.text}</span>
                          <ArrowUpRight size={14} />
                        </button>
                      );
                    })}
                  </div>
                </>
              )}
              <footer>
                <span>
                  <span className="green-dot" />{" "}
                  {status?.cloud_enabled
                    ? "On-demand GPU inference enabled"
                    : "GPU sleeps. Your index stays searchable."}
                </span>
                <span>
                  Powered by EmbeddingGemma 2{" "}
                  <span className="footer-slash">/</span> Built for discovery
                </span>
              </footer>
            </>
          )}
          {page === "Collections" && (
            <>
              <PageHeading
                eyebrow="YOUR KNOWLEDGE BASE"
                title="A place for everything."
                description="Organize your files into searchable collections."
                action={
                  <button
                    className="primary"
                    onClick={() => setShowCreate(true)}
                  >
                    <Plus size={16} /> New collection
                  </button>
                }
              />
              <div className="collection-grid">
                {collections.map((c) => (
                  <article
                    className={`collection-card ${c.id === collectionId ? "focused" : ""}`}
                    key={c.id}
                  >
                    <div className="collection-card-top">
                      <Folder size={23} />
                      <button
                        aria-label={`Delete ${c.name}`}
                        onClick={() => void deleteCollection(c.id)}
                      >
                        <Trash2 size={15} />
                      </button>
                    </div>
                    <button
                      className="collection-card-name"
                      onClick={() =>
                        setCollectionId(c.id === collectionId ? "" : c.id)
                      }
                    >
                      {c.name}
                    </button>
                    <p>{c.description || "Your next discovery starts here."}</p>
                    <div className="collection-stats">
                      <span>{c.file_count} files</span>
                      <span>{c.segment_count} segments</span>
                    </div>
                    <div className="collection-actions">
                      <button
                        onClick={() => {
                          setCollectionId(c.id);
                          uploadRef.current?.click();
                        }}
                      >
                        <Upload size={14} /> Add files
                      </button>
                      <button
                        onClick={() => void startIndex(c.id)}
                        disabled={!status?.cloud_enabled}
                      >
                        <Activity size={14} /> Index
                      </button>
                      <button
                        title="Rebuild all embeddings"
                        aria-label={`Rebuild ${c.name}`}
                        onClick={() => void startIndex(c.id, true)}
                        disabled={!status?.cloud_enabled}
                      >
                        <Layers3 size={14} />
                      </button>
                    </div>
                  </article>
                ))}
              </div>
              {!collections.length && (
                <Empty
                  title="Create your first collection"
                  body="Collect documents, photos, recordings, clips and code in one place."
                />
              )}
              <div
                className="upload-zone"
                role="button"
                tabIndex={0}
                onClick={() => uploadRef.current?.click()}
                onKeyDown={(e) => {
                  if (e.key === "Enter") uploadRef.current?.click();
                }}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  void upload(e.dataTransfer.files);
                }}
              >
                <Upload size={26} />
                <strong>Drop files here, or click to browse</strong>
                <span>
                  Text, code, PNG, JPEG, WebP, WAV, FLAC, MP3, MP4 & more · 20
                  MB per file · 3 minutes per recording
                </span>
              </div>
              {!status?.cloud_enabled && (
                <p className="info-line">
                  <CircleHelp size={14} /> Cloud indexing is off. Set
                  OMNISEEK_ALLOW_CLOUD=1 when starting the API to use your Modal
                  account.
                </p>
              )}
              <FileTable
                files={shownFiles}
                onOpen={openFile}
                onDelete={deleteFile}
              />
            </>
          )}
          {page === "Media explorer" && (
            <>
              <PageHeading
                eyebrow="LOOK CLOSER"
                title="The whole picture."
                description="Every original file, ready to explore."
              />
              <div className="results-toolbar">
                <div className="modality-tabs">
                  {modalities.map((m) => (
                    <button
                      key={m}
                      className={modality === m ? "selected" : ""}
                      onClick={() => setModality(m)}
                    >
                      {plural[m]}
                    </button>
                  ))}
                </div>
                <select
                  className="standalone-select"
                  aria-label="Explorer collection"
                  value={collectionId}
                  onChange={(e) => setCollectionId(e.target.value)}
                >
                  <option value="">All collections</option>
                  {collections.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="media-grid">
                {shownFiles.map((f) => {
                  const Icon = icons[f.modality];
                  return (
                    <button
                      key={f.id}
                      className="media-tile"
                      onClick={() => void openFile(f)}
                    >
                      <div className={`media-tile-preview ${f.modality}`}>
                        {f.modality === "image" ? (
                          <img
                            src={`/api/files/${f.id}/content`}
                            alt={f.name}
                          />
                        ) : f.modality === "video" ? (
                          <video
                            src={`/api/files/${f.id}/content`}
                            preload="metadata"
                            muted
                          />
                        ) : (
                          <Icon size={46} strokeWidth={1} />
                        )}
                      </div>
                      <div>
                        <Icon size={15} />
                        <strong>{f.name}</strong>
                        <span>{bytes(f.size)}</span>
                      </div>
                    </button>
                  );
                })}
              </div>
              {!shownFiles.length && (
                <Empty
                  title="No files in this view"
                  body="Try a different filter, or upload something new."
                />
              )}
            </>
          )}
          {page === "Indexing jobs" && (
            <>
              <PageHeading
                eyebrow="FROM FILES TO SIGNAL"
                title="Work in motion."
                description="On-demand GPU jobs, with progress you can follow."
              />
              <div className="summary-cards">
                <Stat
                  label="ACTIVE JOBS"
                  value={String(activeJobs)}
                  detail="One GPU job at a time"
                />
                <Stat
                  label="COMPLETED"
                  value={String(
                    jobs.filter((j) => j.status === "completed").length,
                  )}
                  detail="Stored indexing jobs"
                />
                <Stat
                  label="SEGMENTS"
                  value={String(status?.segments ?? 0)}
                  detail="Persistent vectors on disk"
                />
              </div>
              <div className="panel">
                <div className="panel-heading">
                  <h2>Indexing activity</h2>
                  <span className="muted small">Refreshes every 5 seconds</span>
                </div>
                {jobs.length ? (
                  jobs.map((j) => (
                    <div key={j.id} className="job-row">
                      <div className={`job-icon ${j.status}`}>
                        <Activity size={19} />
                      </div>
                      <div className="job-main">
                        <strong>
                          {collections.find((c) => c.id === j.collection_id)
                            ?.name || "Collection"}
                        </strong>
                        <p>{j.message || "Waiting for worker"}</p>
                        <div className="progress">
                          <span style={{ width: `${j.progress * 100}%` }} />
                        </div>
                      </div>
                      <span className={`status-badge ${j.status}`}>
                        {j.status}
                      </span>
                    </div>
                  ))
                ) : (
                  <Empty
                    title="A quiet queue"
                    body="Index a collection to see its GPU work here. The included demo is already indexed."
                  />
                )}
              </div>
            </>
          )}
          {page === "Embedding lab" && (
            <>
              <PageHeading
                eyebrow="MEASURE, THEN DECIDE"
                title="Less space. More insight."
                description="Explore Matryoshka compression with actual retrieval measurements."
              />
              <div className="lab-banner">
                <FlaskConical size={22} />
                <div>
                  <strong>A small, transparent experiment.</strong>
                  <p>
                    {experiments.dataset ||
                      "No measurements yet. Run the evaluation after installing the verified demo."}{" "}
                    · Results describe this diagnostic collection.
                  </p>
                </div>
                <span className="tiny-tag">
                  {experiments.status === "measured" ? "MEASURED" : "NOT RUN"}
                </span>
              </div>
              <div className="dimension-tabs">
                {[768, 512, 256, 128].map((d) => (
                  <button
                    key={d}
                    className={dimension === d ? "selected" : ""}
                    onClick={() => setDimension(d)}
                  >
                    <strong>
                      {d}
                      <small>d</small>
                    </strong>
                    <span>
                      {d === 768
                        ? "Full representation"
                        : `${(768 / d).toFixed(1)}× vector compression`}
                    </span>
                  </button>
                ))}
              </div>
              <div className="summary-cards">
                <Stat
                  label="RECALL @ 3"
                  value={
                    selectedExperiment
                      ? `${(selectedExperiment.recall_at_3 * 100).toFixed(1)}%`
                      : "—"
                  }
                  detail="File-level relevance judgments"
                />
                <Stat
                  label="VECTOR STORAGE"
                  value={
                    selectedExperiment
                      ? bytes(selectedExperiment.vector_bytes)
                      : "—"
                  }
                  detail="Raw float32 vectors"
                />
                <Stat
                  label="CPU RETRIEVAL"
                  value={
                    selectedExperiment
                      ? `${selectedExperiment.query_ms_median.toFixed(4)} ms`
                      : "—"
                  }
                  detail="Median per query · warm index"
                />
              </div>
              <div className="lab-chart panel">
                <div className="panel-heading">
                  <h2>The storage / quality tradeoff</h2>
                  <span className="small muted">
                    Exact FAISS · all modalities
                  </span>
                </div>
                <div className="chart-legend">
                  <span>
                    <i /> Vector storage relative to 768d
                  </span>
                  <span>
                    <i /> Recall@3
                  </span>
                </div>
                {realExperiments.map((e) => (
                  <div className="chart-row" key={e.dimension}>
                    <strong>{e.dimension}d</strong>
                    <div className="chart-bars">
                      <div style={{ width: `${(e.dimension / 768) * 100}%` }} />
                      <div style={{ width: `${e.recall_at_3 * 100}%` }} />
                    </div>
                    <span>{(e.recall_at_3 * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
              <div className="panel experiment-table">
                <div className="panel-heading">
                  <h2>Retrieval measurements</h2>
                  <span className="small muted">
                    10 queries · original demo fixtures
                  </span>
                </div>
                <table>
                  <thead>
                    <tr>
                      <th>Dimension</th>
                      <th>Recall@3</th>
                      <th>Precision@3</th>
                      <th>MRR</th>
                      <th>nDCG@3</th>
                      <th>Top-3 stability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {realExperiments.map((e) => (
                      <ExperimentRow key={e.dimension} row={e} />
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="info-line">
                <CircleHelp size={14} /> Vectors are truncated, then normalized
                to unit length again. Similarity is cosine, never a probability.
              </p>
            </>
          )}
          {page === "System status" && (
            <>
              <PageHeading
                eyebrow="UNDER THE SURFACE"
                title="Built to stay lightweight."
                description="Your local index and on-demand inference, at a glance."
              />
              <div className="summary-cards">
                <Stat
                  label="FILES"
                  value={String(status?.files ?? 0)}
                  detail={`${status?.collections ?? 0} collections`}
                />
                <Stat
                  label="INDEXED SEGMENTS"
                  value={String(status?.segments ?? 0)}
                  detail="768-dimensional canonical vectors"
                />
                <Stat
                  label="RESERVED BUDGET"
                  value={`$${status?.cost.reserved_usd.toFixed(2) ?? "0.00"}`}
                  detail="Conservative estimates, including storage"
                />
              </div>
              <div className="system-grid">
                <div className="panel">
                  <div className="panel-heading">
                    <h2>Inference engine</h2>
                    <span className="status-badge completed">
                      {status?.cloud_enabled ? "enabled" : "offline demo"}
                    </span>
                  </div>
                  <dl>
                    <dt>Model</dt>
                    <dd>{status?.model}</dd>
                    <dt>Revision</dt>
                    <dd className="revision">{status?.revision}</dd>
                    <dt>GPU provider</dt>
                    <dd>Modal · {status?.gpu} · scale to zero</dd>
                    <dt>Vector retrieval</dt>
                    <dd>{status?.search}</dd>
                    <dt>Storage</dt>
                    <dd>Local SQLite · transactional persistence</dd>
                    <dt>Upload limits</dt>
                    <dd>20 MB / file · 180 seconds / recording</dd>
                  </dl>
                </div>
                <div className="panel budget-panel">
                  <div className="panel-heading">
                    <h2>Cloud budget</h2>
                    <span className="tiny-tag">USD</span>
                  </div>
                  <div className="budget-number">
                    ${status?.cost.remaining_usd.toFixed(2) ?? "—"}
                    <span>available in local ledger</span>
                  </div>
                  <div className="progress">
                    <span
                      style={{
                        width: `${Math.min(100, ((status?.cost.reserved_usd || 0) / 8) * 100)}%`,
                      }}
                    />
                  </div>
                  <p>
                    $8 operational ceiling leaves $2 below the authorized $10
                    maximum. Reservations include failed attempts and a storage
                    allowance.
                  </p>
                  <small>
                    Estimated tracking is not a provider billing cap. No GPU
                    runs automatically when opening this app.
                  </small>
                </div>
              </div>
              <div className="panel">
                <div className="panel-heading">
                  <h2>Cloud execution ledger</h2>
                  <span className="small muted">
                    Local reservations · costs are estimates
                  </span>
                </div>
                {status?.cost.jobs.map((j, i) => (
                  <div key={i} className="ledger-row">
                    <span>{j.label}</span>
                    <span>{j.gpu || "CPU"}</span>
                    <span
                      className={`status-badge ${j.status === "succeeded" ? "completed" : j.status}`}
                    >
                      {j.status}
                    </span>
                    <strong>${j.reserved_usd.toFixed(3)}</strong>
                  </div>
                ))}
                {!status?.cost.jobs.length && (
                  <Empty
                    title="No cloud executions on this installation"
                    body="Installing and exploring the demo spends no GPU credits."
                  />
                )}
              </div>
            </>
          )}
        </main>
      </div>
      <input
        hidden
        ref={uploadRef}
        type="file"
        multiple
        onChange={(e) => void upload(e.target.files)}
      />
      <input
        hidden
        ref={imageRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        onChange={(e) => void upload(e.target.files, true)}
      />
      {showCreate && (
        <div className="modal-backdrop" onClick={() => setShowCreate(false)}>
          <section
            className="dialog create-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="create-title"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="dialog-close"
              aria-label="Close dialog"
              onClick={() => setShowCreate(false)}
            >
              <X size={20} />
            </button>
            <Folder size={28} className="accent" />
            <h2 id="create-title">Make room for discovery.</h2>
            <p className="muted">
              Create a collection for your next set of signals.
            </p>
            <form onSubmit={(e) => void createCollection(e)}>
              <label>
                Collection name
                <input
                  autoFocus
                  required
                  maxLength={80}
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Research, field notes, inspiration…"
                />
              </label>
              <label>
                Description
                <textarea
                  maxLength={500}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="What belongs here?"
                />
              </label>
              <button className="primary" disabled={!name.trim()}>
                <Plus size={16} /> Create collection
              </button>
            </form>
          </section>
        </div>
      )}
      {selected && (
        <div className="modal-backdrop" onClick={() => setSelected(null)}>
          <section
            className="dialog preview-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="preview-title"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="preview-heading">
              <div>
                <span className={`type-label ${selected.modality}`}>
                  {selected.modality}
                </span>
                <h2 id="preview-title">{selected.name}</h2>
              </div>
              <button
                aria-label="Close preview"
                onClick={() => setSelected(null)}
              >
                <X size={20} />
              </button>
            </div>
            <div className="preview-media">
              <Media result={selected} />
            </div>
            <div className="preview-footer">
              <span>
                {selected.metadata.start !== undefined
                  ? `${timeLabel(selected.metadata.start)} – ${timeLabel(selected.metadata.end)} · coarse segment match`
                  : selected.metadata.line_start
                    ? `Lines ${selected.metadata.line_start}–${selected.metadata.line_end}`
                    : "Original file"}
                {selected.score !== 0 &&
                  ` · cosine ${selected.score.toFixed(4)}`}
              </span>
              <a href={selected.url} target="_blank" rel="noreferrer">
                Open original <ArrowUpRight size={14} />
              </a>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

function PageHeading({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="page-heading secondary">
      <div>
        <div className="eyebrow">
          <span className="line" />
          {eyebrow}
        </div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </div>
  );
}
function Empty({ title, body }: { title: string; body: string }) {
  return (
    <div className="empty">
      <Database size={26} strokeWidth={1} />
      <h3>{title}</h3>
      <p>{body}</p>
    </div>
  );
}
function Stat({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}
function ExperimentRow({ row: e }: { row: Experiment }) {
  return (
    <tr>
      <td>{e.dimension}d</td>
      <td>{(e.recall_at_3 * 100).toFixed(1)}%</td>
      <td>{(e.precision_at_3 * 100).toFixed(1)}%</td>
      <td>{e.mrr.toFixed(3)}</td>
      <td>{e.ndcg_at_3.toFixed(3)}</td>
      <td>{(e.top3_overlap_with_768 * 100).toFixed(1)}%</td>
    </tr>
  );
}
function FileTable({
  files,
  onOpen,
  onDelete,
}: {
  files: FileItem[];
  onOpen: (f: FileItem) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}) {
  return (
    <div className="panel file-table">
      <div className="panel-heading">
        <h2>Files</h2>
        <span className="small muted">{files.length} in this view</span>
      </div>
      {files.length ? (
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Size</th>
              <th>Status</th>
              <th>
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {files.map((f) => {
              const Icon = icons[f.modality];
              return (
                <tr key={f.id}>
                  <td>
                    <button className="filename" onClick={() => void onOpen(f)}>
                      <Icon size={16} />
                      {f.name}
                    </button>
                  </td>
                  <td>{f.modality}</td>
                  <td>{bytes(f.size)}</td>
                  <td>
                    <span
                      className={`status-badge ${f.status === "indexed" ? "completed" : ""}`}
                    >
                      {f.status}
                    </span>
                  </td>
                  <td>
                    <button
                      aria-label={`Delete ${f.name}`}
                      onClick={() => void onDelete(f.id)}
                    >
                      <Trash2 size={15} />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : (
        <Empty
          title="A fresh start"
          body="Upload a file to add it to this collection."
        />
      )}
    </div>
  );
}
function ResultCard({
  result: r,
  rank,
  onClick,
}: {
  result: Result;
  rank: number;
  onClick: () => void;
}) {
  const Icon = icons[r.modality];
  return (
    <button className={`result-card ${r.modality}`} onClick={onClick}>
      <div className="result-preview">
        {r.modality === "image" ? (
          <img src={r.url} alt={r.name} />
        ) : r.modality === "video" ? (
          <>
            <video
              src={`${r.url}#t=${r.metadata.start || 0}`}
              preload="metadata"
              muted
            />
            <span className="clip-time">
              <Video size={12} />
              {timeLabel(r.metadata.start)} – {timeLabel(r.metadata.end)}
            </span>
          </>
        ) : r.modality === "audio" ? (
          <div className="wave large">
            {Array.from({ length: 40 }, (_, i) => (
              <i key={i} style={{ height: `${10 + ((i * 23) % 65)}px` }} />
            ))}
          </div>
        ) : (
          <pre>{r.metadata.text.slice(0, 420)}</pre>
        )}
        <span className="result-rank">{String(rank).padStart(2, "0")}</span>
      </div>
      <div className="result-body">
        <div className="result-title">
          <Icon size={15} />
          <strong>{r.name}</strong>
          <ArrowUpRight size={14} />
        </div>
        <div className="result-details">
          <span>
            {r.modality}
            {r.metadata.line_start
              ? ` · L${r.metadata.line_start}–${r.metadata.line_end}`
              : ""}
          </span>
          <span className="similarity">
            <i />
            {r.score.toFixed(3)} <small>cosine</small>
          </span>
        </div>
      </div>
    </button>
  );
}
