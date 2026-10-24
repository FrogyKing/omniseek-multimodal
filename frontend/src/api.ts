export type Modality = "text" | "image" | "audio" | "video" | "code";
export type Collection = {
  id: string;
  name: string;
  description: string;
  file_count: number;
  segment_count: number;
};
export type FileItem = {
  id: string;
  collection_id: string;
  name: string;
  modality: Modality;
  status: string;
  size: number;
  metadata: { duration?: number; segments?: number };
};
export type Result = {
  id: string;
  file_id: string;
  name: string;
  modality: Modality;
  score: number;
  url: string;
  metadata: {
    text: string;
    start?: number;
    end?: number;
    line_start?: number;
    line_end?: number;
  };
};
export type SearchResponse = {
  id: string;
  results: Result[];
  dimension: number;
  cache_hit: boolean;
  retrieval_ms: number;
  total_segments: number;
};
export type DemoQuery = { text: string; modality: string };
export type Job = {
  id: string;
  collection_id: string;
  status: string;
  progress: number;
  message: string;
  created_at: string;
};
export type Cost = {
  reserved_usd: number;
  remaining_usd: number;
  ceiling_usd: number;
  jobs: { label: string; gpu: string; status: string; reserved_usd: number }[];
};
export type SystemStatus = {
  model: string;
  revision: string;
  dimension: number;
  cloud_enabled: boolean;
  gpu: string;
  collections: number;
  files: number;
  segments: number;
  cost: Cost;
  limits: { upload_mib: number; media_seconds: number };
  search: string;
};
export type Experiment = {
  dimension: number;
  index: string;
  recall_at_3: number;
  ndcg_at_3: number;
  mrr: number;
  precision_at_3: number;
  vector_bytes: number;
  index_bytes: number;
  query_ms_median: number;
  top3_overlap_with_768: number;
};
export type Experiments = {
  status: string;
  dataset?: string;
  results: Experiment[];
};

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(
      body.error ||
        (typeof body.detail === "string"
          ? body.detail
          : "Request could not be completed"),
    );
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

export function post<T>(path: string, body: unknown): Promise<T> {
  return api<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
