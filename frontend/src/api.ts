export type Video = {
  id: string;
  name: string;
  duration: number;
  fps: number;
  width: number;
  height: number;
  status: string;
  indexed_modalities: string[];
  created_at: string;
};

export type Job = {
  id: string;
  video_id: string;
  status: string;
  stage: string;
  progress: number;
  modalities: string[];
  metrics?: {
    total_elapsed_seconds?: number | null;
    stages?: Record<string, { elapsed_seconds?: number; status?: string; [key: string]: unknown }>;
  };
  error?: string;
};

export type IndexOptions = {
  visualModel?: string;
  visualSampleFps?: number;
  visualSegmentSeconds?: number;
  faceSampleFps?: number;
  ocrSampleFps?: number;
  asrModel?: string;
  asrLanguage?: string;
};

export type Entity = {
  id: string;
  name: string;
  reference_path: string;
  embedding_path?: string;
};

export type Evidence = {
  modality: string;
  score: number;
  raw_score?: number | null;
  robust_z?: number | null;
  percentile?: number | null;
  decision?: string;
  distribution_reliable?: boolean | null;
  distribution_median?: number | null;
  distribution_mad?: number | null;
  detail?: string;
  best_time?: number | null;
  unit_type?: string | null;
  unit_id?: number | null;
  best_ms?: number | null;
  text?: string | null;
  features?: Record<string, unknown>;
  visual_top1?: number | null;
  visual_top3?: number | null;
  visual_mean?: number | null;
  lexical_score?: number | null;
  semantic_score?: number | null;
  semantic_cosine?: number | null;
};

export type SearchResult = {
  video_id: string;
  video_name: string;
  start_time: number;
  end_time: number;
  score: number;
  modalities: string[];
  thumbnail_url?: string;
  media_url: string;
  clip_url?: string;
  decision?: string;
  above_threshold: boolean;
  evidence: Evidence[];
};

export type SearchResponse = {
  count: number;
  above_count?: number;
  elapsed_seconds?: number;
  results: SearchResult[];
};

async function json<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `请求失败 (${response.status})`);
  return payload as T;
}

export const api = {
  videos: () => json<Video[]>("/api/videos"),
  jobs: () => json<Job[]>("/api/jobs"),
  entities: () => json<Entity[]>("/api/entities"),
  uploadVideo: (video: File, transcript?: File) => {
    const form = new FormData();
    form.append("video", video);
    if (transcript) form.append("transcript", transcript);
    return json<Video>("/api/videos", { method: "POST", body: form });
  },
  indexVideo: (videoId: string, modalities: string[], options: IndexOptions = {}) =>
    json<Job>(`/api/videos/${videoId}/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        modalities,
        visual_model: options.visualModel,
        visual_sample_fps: options.visualSampleFps,
        visual_segment_seconds: options.visualSegmentSeconds,
        face_sample_fps: options.faceSampleFps,
        ocr_sample_fps: options.ocrSampleFps,
        asr_model: options.asrModel,
        asr_language: options.asrLanguage,
      }),
    }),
  renameVideo: (videoId: string, name: string) =>
    json<Video>(`/api/videos/${videoId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),
  deleteVideo: (videoId: string) =>
    json<{ status: string; id: string }>(`/api/videos/${videoId}`, { method: "DELETE" }),
  createEntity: (name: string, reference: File) => {
    const form = new FormData();
    form.append("name", name);
    form.append("reference", reference);
    return json<Entity>("/api/entities", { method: "POST", body: form });
  },
  search: (params: {
    queryText: string;
    queryImage?: File;
    modalities: string[];
    videoIds: string[];
    alpha: number;
    limit?: number;
  }) => {
    const form = new FormData();
    if (params.queryText) form.append("query_text", params.queryText);
    if (params.queryImage) form.append("query_image", params.queryImage);
    form.append("modalities", params.modalities.join(","));
    form.append("video_ids", JSON.stringify(params.videoIds));
    form.append("alpha", String(params.alpha));
    form.append("limit", String(params.limit ?? 50));
    return json<SearchResponse>("/api/search", { method: "POST", body: form });
  },
};
