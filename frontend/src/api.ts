export type IndexModality = "visual" | "face" | "asr" | "ocr";

export type Video = {
  id: string;
  name: string;
  duration: number;
  fps: number;
  width: number;
  height: number;
  status: string;
  indexed_modalities: IndexModality[];
  speaker_indexed?: boolean;
  created_at: string;
};

export type Job = {
  id: string;
  video_id: string;
  status: string;
  stage: string;
  progress: number;
  modalities: IndexModality[];
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
  visualSegmentStrategy?: "fixed" | "shot";
  visualMinSegmentSeconds?: number;
  visualMaxSegmentSeconds?: number;
  visualShotDetector?: "simple" | "pyscenedetect_content" | "pyscenedetect_adaptive";
  visualShotThreshold?: number;
  faceSampleFps?: number;
  ocrSampleFps?: number;
  asrModel?: string;
  asrLanguage?: string;
  asrSpeakerEnabled?: boolean;
};

export type Entity = {
  id: string;
  name: string;
  reference_path: string;
  embedding_path?: string;
  voice_sample_count?: number;
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

export type SpeakerUtterance = {
  index: number; start_ms: number; end_ms: number; asr_chunk_index: number;
  text: string; auto_track_id: number; track_id: number | null; searchable: boolean; clip_url: string;
};

export type VideoSpeaker = {
  track_id: number; label: string; display_name?: string; representative_utterance_index: number;
  utterance_indices: number[]; utterance_count: number; duration_ms: number; hidden: boolean; entity_id?: string;
};

export type SpeakerView = { video_id: string; tracks: VideoSpeaker[]; utterances: SpeakerUtterance[] };
export type VoiceHit = { video_id: string; video_name: string; utterance_index: number; track_id: number | null; start_ms: number; end_ms: number; score: number; text: string; clip_url: string };

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
  renameEntity: (entityId: string, name: string) =>
    json<Entity>(`/api/entities/${entityId}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) }),
  deleteEntity: (entityId: string) =>
    json<{ status: string; id: string }>(`/api/entities/${entityId}`, { method: "DELETE" }),
  speakers: (videoId: string) => json<SpeakerView>(`/api/videos/${videoId}/speakers`),
  updateSpeaker: (videoId: string, trackId: number, update: { display_name?: string; representative_utterance_index?: number; hidden?: boolean }) =>
    json<SpeakerView>(`/api/videos/${videoId}/speakers/${trackId}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(update) }),
  updateUtterance: (videoId: string, utteranceIndex: number, update: { corrected_track_id: number | null; searchable: boolean }) =>
    json<SpeakerView>(`/api/videos/${videoId}/utterances/${utteranceIndex}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(update) }),
  voiceSearch: (queryVideoId: string, queryUtteranceIndex: number, videoIds?: string[]) =>
    json<{ count: number; results: VoiceHit[] }>("/api/voice-search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query_video_id: queryVideoId, query_utterance_index: queryUtteranceIndex, video_ids: videoIds, limit: 50 }) }),
  voiceSearchUpload: (reference: File, videoIds?: string[]) => {
    const form = new FormData(); form.append("reference", reference);
    if (videoIds) form.append("video_ids", JSON.stringify(videoIds));
    form.append("limit", "50");
    return json<{ query_samples: number; count: number; results: VoiceHit[] }>("/api/voice-search/upload", { method: "POST", body: form });
  },
  uploadVideo: (video: File, transcript?: File) => {
    const form = new FormData();
    form.append("video", video);
    if (transcript) form.append("transcript", transcript);
    return json<Video>("/api/videos", { method: "POST", body: form });
  },
  indexVideo: (videoId: string, modalities: IndexModality[], options: IndexOptions = {}) => {
    const selected = new Set(modalities);
    return json<Job>(`/api/videos/${videoId}/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        modalities,
        visual_model: selected.has("visual") ? options.visualModel : undefined,
        visual_sample_fps: selected.has("visual") ? options.visualSampleFps : undefined,
        visual_segment_seconds: selected.has("visual") ? options.visualSegmentSeconds : undefined,
        visual_segment_strategy: selected.has("visual") ? options.visualSegmentStrategy : undefined,
        visual_min_segment_seconds: selected.has("visual") ? options.visualMinSegmentSeconds : undefined,
        visual_max_segment_seconds: selected.has("visual") ? options.visualMaxSegmentSeconds : undefined,
        visual_shot_detector: selected.has("visual") ? options.visualShotDetector : undefined,
        visual_shot_threshold: selected.has("visual") ? options.visualShotThreshold : undefined,
        face_sample_fps: selected.has("face") ? options.faceSampleFps : undefined,
        ocr_sample_fps: selected.has("ocr") ? options.ocrSampleFps : undefined,
        asr_model: selected.has("asr") ? options.asrModel : undefined,
        asr_language: selected.has("asr") ? options.asrLanguage : undefined,
        asr_speaker_enabled: selected.has("asr") ? options.asrSpeakerEnabled : false,
      }),
    });
  },
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
  createVoiceEntity: (name: string) => json<Entity>("/api/entities/voice-only", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) }),
  addVoiceSample: (entityId: string, videoId: string, utteranceIndex: number, bindTrackId?: number) =>
    json<Record<string, unknown>>(`/api/entities/${entityId}/voice-samples`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ video_id: videoId, utterance_index: utteranceIndex, bind_track_id: bindTrackId }) }),
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
