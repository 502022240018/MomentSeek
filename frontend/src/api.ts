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
  error?: string;
};

export type Entity = {
  id: string;
  name: string;
  reference_path: string;
  embedding_path?: string;
};

export type Evidence = { modality: string; score: number; detail?: string };

export type SearchResult = {
  video_id: string;
  video_name: string;
  start_time: number;
  end_time: number;
  score: number;
  modalities: string[];
  thumbnail_url?: string;
  media_url: string;
  evidence: Evidence[];
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
  indexVideo: (videoId: string, modalities: string[]) =>
    json<Job>(`/api/videos/${videoId}/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ modalities }),
    }),
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
  }) => {
    const form = new FormData();
    if (params.queryText) form.append("query_text", params.queryText);
    if (params.queryImage) form.append("query_image", params.queryImage);
    form.append("modalities", params.modalities.join(","));
    form.append("video_ids", JSON.stringify(params.videoIds));
    form.append("alpha", String(params.alpha));
    return json<{ count: number; results: SearchResult[] }>("/api/search", { method: "POST", body: form });
  },
};

