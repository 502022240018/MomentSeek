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
  folder_ids: string[];
  folders: FolderRef[];
  speaker_indexed?: boolean;
  created_at: string;
};

export type FolderRef = { id: string; name: string };
export type Folder = FolderRef & { kind: "default" | "user"; video_count: number; created_at?: string };

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

export type ColorGradingCapability = {
  enabled: boolean;
  available: boolean;
  reason?: string | null;
  model_loaded: boolean;
  database_connected: boolean;
  device?: string | null;
};

export type ColorGradingTask = {
  id: string;
  external_task_id?: string | null;
  input_video_id: string;
  input_video_name: string;
  reference_type: "image" | "video";
  reference_video_id?: string | null;
  reference_video_name?: string | null;
  reference_url?: string | null;
  ncc: boolean;
  status: string;
  stage: string;
  upstream_status?: string | null;
  queue_position?: number | null;
  media_url?: string | null;
  lut_url?: string | null;
  imported_video_id?: string | null;
  error_code?: string | null;
  error_message?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  created_at: string;
  updated_at: string;
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
  retrieval_score?: number;
  rerank_score?: number | null;
  original_rank?: number;
  planner_evidence?: {
    moment_id: string;
    raw_scores: Record<string, number>;
    source_contrib: Record<string, number>;
    source_count: number;
    primary_source_count: number;
    support_source_count: number;
    primary_contrib: Record<string, number>;
    support_contrib: Record<string, number>;
    verifier_contrib: Record<string, number>;
    constraint_results: Record<string, boolean>;
    primary_score: number;
    support_bonus: number;
    verifier_score?: number | null;
  };
};

export type PlannerMode = "guide" | "assist" | "auto";
export type PlannerCapability = {
  tool_id: string; label: string; modality: string; operations: string[];
  score_range: number[]; calibration: string; default_top_k: number;
  default_weight: number; cost: string; latency: string;
  temporal_granularity: string; description: string;
};
export type PlannerLabCapabilities = {
  enabled: boolean; llm_enabled: boolean; planner: string;
  capabilities: PlannerCapability[]; fusion_methods: string[]; modes: PlannerMode[];
  evidence_roles?: EvidenceRole[]; step_decisions?: string[];
  role_policy?: Record<string, string>;
};
export type EvidenceRole = "primary" | "support" | "constraint" | "verifier" | "fallback";
export type StepQualityGate = {
  min_results: number; min_score_spread: number; min_match_rate: number;
  min_survivors: number; max_top_k_disruption: number;
};
export type PlanStep = {
  step_id: string; tool_id: string; operation: "search" | "rerank" | "filter";
  role: EvidenceRole; target_id: string; depends_on: string[];
  query: string; weight: number; top_k: number; rationale: string;
  support_bonus_cap: number; fallback_for?: string | null;
  failure_policy: "skip" | "rollback" | "fallback" | "abort";
  quality_gate: StepQualityGate; enabled: boolean;
  parameters: Record<string, unknown>;
};
export type CandidatePlan = {
  plan_id: "fast" | "balanced" | "deep"; label: string; description: string;
  estimated_cost: "low" | "medium" | "high"; fusion: "rrf" | "combsum" | "combmnz";
  result_limit: number; early_stop_threshold: number; steps: PlanStep[];
};
export type IdentityClarification = {
  clarification_id: string; kind: "unregistered_identity"; name: string;
  message: string; visual_fallback_query: string; rationale?: string;
  options: Array<"upload_reference" | "generic_visual" | "keep_original">;
};
export type PlanSetResponse = {
  mode: PlannerMode; query_intent: string; constraints: string[]; negative_constraints: string[];
  available_modalities: string[]; plans: CandidatePlan[];
  matched_entity?: { id?: string; name?: string } | null;
  identity_mentions?: Array<{ name: string; visual_fallback_query: string; rationale?: string }>;
  clarifications?: IdentityClarification[];
  planner_trace: Record<string, any>; scope: Record<string, any>;
};
export type PlannerExecution = {
  execution_id: string; plan: CandidatePlan; executed_steps: number;
  stop_reason: string; elapsed_seconds: number; count: number; above_count: number;
  accepted_steps: number; skipped_steps: number; rolled_back_steps: number;
  results: SearchResult[]; trace: Array<Record<string, any>>; scope: Record<string, any>;
};

export type OrchestrationComponent = {
  provider: string;
  type: string;
  model: string;
  prompt_version: string;
};

export type OrchestrationProfile = {
  name: string;
  description: string;
  planner?: OrchestrationComponent | null;
  reranker?: OrchestrationComponent | null;
};

export type OrchestrationProfiles = {
  enabled: boolean;
  default_profile: string;
  profiles: OrchestrationProfile[];
};

export type SearchResponse = {
  count: number;
  above_count?: number;
  elapsed_seconds?: number;
  execution?: Record<string, any>;
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
  folders: () => json<Folder[]>("/api/folders"),
  jobs: () => json<Job[]>("/api/jobs"),
  colorGradingStatus: () =>
    json<ColorGradingCapability>("/api/color-grading/status"),
  colorGradingTasks: () =>
    json<ColorGradingTask[]>("/api/color-grading/tasks"),
  createColorGradingTask: (params: {
    inputVideoId: string;
    referenceType: "image" | "video";
    referenceImage?: File;
    referenceVideoId?: string;
    ncc?: boolean;
  }) => {
    const form = new FormData();
    form.append("input_video_id", params.inputVideoId);
    form.append("reference_type", params.referenceType);
    form.append("ncc", String(params.ncc ?? false));
    if (params.referenceImage) form.append("ref_image", params.referenceImage);
    if (params.referenceVideoId) form.append("ref_video_id", params.referenceVideoId);
    return json<ColorGradingTask>("/api/color-grading/tasks", { method: "POST", body: form });
  },
  importColorGradingResult: (taskId: string) =>
    json<Video>(`/api/color-grading/tasks/${taskId}/import`, { method: "POST" }),
  cancelJob: (jobId: string) =>
    json<Job>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
  entities: () => json<Entity[]>("/api/entities"),
  orchestrationProfiles: () => json<OrchestrationProfiles>("/api/orchestration/profiles"),
  plannerLabCapabilities: () => json<PlannerLabCapabilities>("/api/planner-lab/capabilities"),
  plannerLabPlans: (params: {
    queryText: string; queryImage?: File; videoIds?: string[]; folderIds?: string[];
    mode: PlannerMode; orchestrationProfile?: string;
  }) => {
    const form = new FormData();
    form.append("query_text", params.queryText);
    if (params.queryImage) form.append("query_image", params.queryImage);
    if (params.videoIds?.length) form.append("video_ids", JSON.stringify(params.videoIds));
    if (params.folderIds?.length) form.append("folder_ids", JSON.stringify(params.folderIds));
    form.append("mode", params.mode);
    if (params.orchestrationProfile) form.append("orchestration_profile", params.orchestrationProfile);
    return json<PlanSetResponse>("/api/planner-lab/plans", { method: "POST", body: form });
  },
  plannerLabExecute: (params: {
    queryText: string; queryImage?: File; plan: CandidatePlan;
    videoIds?: string[]; folderIds?: string[]; maxSteps?: number;
  }) => {
    const form = new FormData();
    form.append("query_text", params.queryText);
    form.append("plan", JSON.stringify(params.plan));
    if (params.queryImage) form.append("query_image", params.queryImage);
    if (params.videoIds?.length) form.append("video_ids", JSON.stringify(params.videoIds));
    if (params.folderIds?.length) form.append("folder_ids", JSON.stringify(params.folderIds));
    if (params.maxSteps != null) form.append("max_steps", String(params.maxSteps));
    return json<PlannerExecution>("/api/planner-lab/execute", { method: "POST", body: form });
  },
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
  uploadVideo: (video: File, transcript?: File, folderIds: string[] = []) => {
    const form = new FormData();
    form.append("video", video);
    if (transcript) form.append("transcript", transcript);
    if (folderIds.length) form.append("folder_ids", JSON.stringify(folderIds));
    return json<Video>("/api/videos", { method: "POST", body: form });
  },
  createFolder: (name: string) => json<Folder>("/api/folders", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) }),
  renameFolder: (folderId: string, name: string) => json<Folder>(`/api/folders/${folderId}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) }),
  deleteFolder: (folderId: string) => json<{ status: string; id: string; released_video_count: number }>(`/api/folders/${folderId}`, { method: "DELETE" }),
  updateVideoFolders: (videoIds: string[], folderIds: string[], operation: "add" | "remove" | "replace") => json<{ status: string }>("/api/videos/folders", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ video_ids: videoIds, folder_ids: folderIds, operation }) }),
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
    videoIds?: string[];
    folderIds?: string[];
    alpha: number;
    limit?: number;
    orchestrationProfile?: string;
    plannerMode?: "auto" | "off" | "force";
    rerankerMode?: "auto" | "off" | "force";
  }) => {
    const form = new FormData();
    if (params.queryText) form.append("query_text", params.queryText);
    if (params.queryImage) form.append("query_image", params.queryImage);
    form.append("modalities", params.modalities.join(","));
    if (params.videoIds?.length) form.append("video_ids", JSON.stringify(params.videoIds));
    if (params.folderIds?.length) form.append("folder_ids", JSON.stringify(params.folderIds));
    form.append("alpha", String(params.alpha));
    form.append("limit", String(params.limit ?? 50));
    if (params.orchestrationProfile) form.append("orchestration_profile", params.orchestrationProfile);
    form.append("planner_mode", params.plannerMode ?? "auto");
    form.append("reranker_mode", params.rerankerMode ?? "auto");
    return json<SearchResponse>("/api/search", { method: "POST", body: form });
  },
};
