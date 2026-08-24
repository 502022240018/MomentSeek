import type {
  CandidatePlan,
  ColorGradingCapability,
  ColorGradingTask,
  Entity,
  FaceGalleryView,
  Folder,
  Job,
  OrchestrationProfiles,
  PlanSetResponse,
  PlannerExecution,
  PlannerLabCapabilities,
  PlanStep,
  SearchResponse,
  SearchResult,
  SpeakerView,
  Video,
  VoiceHit,
} from "./api";

export type Normalizer<T> = (value: unknown) => T;
type JsonRecord = Record<string, any>;

function record(value: unknown, label: string): JsonRecord {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} 必须是对象`);
  }
  return value as JsonRecord;
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${label} 必须是数组`);
  return value;
}

function optionalArray(value: unknown, label: string): unknown[] {
  if (value == null) return [];
  return array(value, label);
}

function requiredText(value: unknown, label: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${label} 缺失`);
  return value;
}

function text(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function number(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function requiredNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} 必须是有限数值`);
  }
  return value;
}

function stringArray(value: unknown, label: string): string[] {
  return optionalArray(value, label).filter((item): item is string => typeof item === "string");
}

function errorDetail(value: unknown): string | undefined {
  if (!value || typeof value !== "object") return undefined;
  const detail = (value as JsonRecord).detail;
  if (typeof detail === "string") return detail;
  if (detail != null) return JSON.stringify(detail);
  return undefined;
}

export async function readJsonResponse<T>(response: Response, normalize?: Normalizer<T>): Promise<T> {
  const body = await response.text();
  let payload: unknown;
  try {
    payload = body ? JSON.parse(body) : null;
  } catch {
    throw new Error(`服务返回了无法解析的响应 (${response.status})`);
  }
  if (!response.ok) throw new Error(errorDetail(payload) || `请求失败 (${response.status})`);
  try {
    return normalize ? normalize(payload) : payload as T;
  } catch (error) {
    const message = error instanceof Error ? error.message : "未知结构错误";
    throw new Error(`服务响应结构不兼容：${message}`);
  }
}

export function normalizeFolder(value: unknown, label = "folder"): Folder {
  const item = record(value, label);
  return {
    ...item,
    id: requiredText(item.id, `${label}.id`),
    name: requiredText(item.name, `${label}.name`),
    kind: item.kind === "default" ? "default" : "user",
    video_count: number(item.video_count),
  } as Folder;
}

export function normalizeFolderList(value: unknown): Folder[] {
  return array(value, "folders").map((item, index) => normalizeFolder(item, `folders[${index}]`));
}

export function normalizeVideo(value: unknown, label = "video"): Video {
  const item = record(value, label);
  const folders = optionalArray(item.folders, `${label}.folders`).map((folder, index) => {
    const entry = record(folder, `${label}.folders[${index}]`);
    return {
      id: requiredText(entry.id, `${label}.folders[${index}].id`),
      name: requiredText(entry.name, `${label}.folders[${index}].name`),
    };
  });
  return {
    ...item,
    id: requiredText(item.id, `${label}.id`),
    name: text(item.name, "未命名视频"),
    duration: number(item.duration),
    fps: number(item.fps),
    width: number(item.width),
    height: number(item.height),
    status: text(item.status, "uploaded"),
    indexed_modalities: stringArray(item.indexed_modalities, `${label}.indexed_modalities`) as Video["indexed_modalities"],
    folder_ids: stringArray(item.folder_ids, `${label}.folder_ids`),
    folders,
    speaker_indexed: Boolean(item.speaker_indexed),
    created_at: text(item.created_at),
  } as Video;
}

export function normalizeVideoList(value: unknown): Video[] {
  return array(value, "videos").map((item, index) => normalizeVideo(item, `videos[${index}]`));
}

export function normalizeJob(value: unknown, label = "job"): Job {
  const item = record(value, label);
  return {
    ...item,
    id: requiredText(item.id, `${label}.id`),
    video_id: requiredText(item.video_id, `${label}.video_id`),
    status: text(item.status, "queued"),
    stage: text(item.stage),
    progress: number(item.progress),
    modalities: stringArray(item.modalities, `${label}.modalities`) as Job["modalities"],
    metrics: item.metrics && typeof item.metrics === "object" ? item.metrics : undefined,
  } as Job;
}

export function normalizeJobList(value: unknown): Job[] {
  return array(value, "jobs").map((item, index) => normalizeJob(item, `jobs[${index}]`));
}

export function normalizeEntity(value: unknown, label = "entity"): Entity {
  const item = record(value, label);
  return {
    ...item,
    id: requiredText(item.id, `${label}.id`),
    name: text(item.name, "未命名人物"),
    reference_path: text(item.reference_path),
    voice_sample_count: number(item.voice_sample_count),
  } as Entity;
}

export function normalizeEntityList(value: unknown): Entity[] {
  return array(value, "entities").map((item, index) => normalizeEntity(item, `entities[${index}]`));
}

export function normalizeColorGradingCapability(value: unknown): ColorGradingCapability {
  const item = record(value, "color_grading_status");
  return {
    ...item,
    enabled: Boolean(item.enabled),
    available: Boolean(item.available),
    model_loaded: Boolean(item.model_loaded),
    database_connected: Boolean(item.database_connected),
  } as ColorGradingCapability;
}

export function normalizeColorGradingTask(value: unknown, label = "color_grading_task"): ColorGradingTask {
  const item = record(value, label);
  return { ...item, id: requiredText(item.id, `${label}.id`) } as ColorGradingTask;
}

export function normalizeColorGradingTaskList(value: unknown): ColorGradingTask[] {
  return array(value, "color_grading_tasks").map((item, index) => normalizeColorGradingTask(item, `color_grading_tasks[${index}]`));
}

export function normalizeSearchResult(value: unknown, label = "result"): SearchResult {
  const item = record(value, label);
  const startTime = requiredNumber(item.start_time, `${label}.start_time`);
  const endTime = requiredNumber(item.end_time, `${label}.end_time`);
  if (startTime < 0 || endTime <= startTime) {
    throw new Error(`${label} 时间范围无效`);
  }
  const evidence = optionalArray(item.evidence, `${label}.evidence`).map((entry, index) => ({
    ...record(entry, `${label}.evidence[${index}]`),
  }));
  return {
    ...item,
    video_id: requiredText(item.video_id, `${label}.video_id`),
    video_name: text(item.video_name, "未命名视频"),
    start_time: startTime,
    end_time: endTime,
    score: number(item.score),
    modalities: stringArray(item.modalities, `${label}.modalities`),
    evidence,
    media_url: text(item.media_url),
    above_threshold: item.above_threshold !== false,
  } as SearchResult;
}

export function normalizeSearchResponse(value: unknown): SearchResponse {
  const item = record(value, "search_response");
  const results = array(item.results, "search_response.results").map((result, index) =>
    normalizeSearchResult(result, `search_response.results[${index}]`),
  );
  return {
    ...item,
    count: number(item.count, results.length),
    above_count: number(item.above_count, results.filter(result => result.above_threshold !== false).length),
    elapsed_seconds: number(item.elapsed_seconds),
    results,
  } as SearchResponse;
}

function normalizePlanStep(value: unknown, label: string): PlanStep {
  const item = record(value, label);
  return {
    ...item,
    step_id: requiredText(item.step_id, `${label}.step_id`),
    tool_id: requiredText(item.tool_id, `${label}.tool_id`),
    operation: item.operation === "rerank" || item.operation === "filter" ? item.operation : "search",
    role: text(item.role, "primary") as PlanStep["role"],
    target_id: text(item.target_id),
    depends_on: stringArray(item.depends_on, `${label}.depends_on`),
    query: text(item.query),
    weight: number(item.weight, 1),
    top_k: Math.max(1, number(item.top_k, 20)),
    rationale: text(item.rationale),
    support_bonus_cap: number(item.support_bonus_cap, 0.4),
    failure_policy: text(item.failure_policy, "skip") as PlanStep["failure_policy"],
    quality_gate: item.quality_gate && typeof item.quality_gate === "object" ? item.quality_gate : {},
    enabled: item.enabled !== false,
    parameters: item.parameters && typeof item.parameters === "object" ? item.parameters : {},
  } as PlanStep;
}

function normalizeCandidatePlan(value: unknown, label: string): CandidatePlan {
  const item = record(value, label);
  const steps = array(item.steps, `${label}.steps`).map((step, index) => normalizePlanStep(step, `${label}.steps[${index}]`));
  return {
    ...item,
    plan_id: requiredText(item.plan_id, `${label}.plan_id`) as CandidatePlan["plan_id"],
    label: text(item.label, "未命名策略"),
    description: text(item.description),
    estimated_cost: text(item.estimated_cost, "medium") as CandidatePlan["estimated_cost"],
    fusion: text(item.fusion, "rrf") as CandidatePlan["fusion"],
    result_limit: Math.max(1, number(item.result_limit, 50)),
    early_stop_threshold: number(item.early_stop_threshold, 0.9),
    steps,
  } as CandidatePlan;
}

export function normalizePlannerCapabilities(value: unknown): PlannerLabCapabilities {
  const item = record(value, "planner_capabilities");
  const capabilities = optionalArray(item.capabilities, "planner_capabilities.capabilities").map((entry, index) => {
    const capability = record(entry, `planner_capabilities.capabilities[${index}]`);
    return {
      ...capability,
      tool_id: requiredText(capability.tool_id, `planner_capabilities.capabilities[${index}].tool_id`),
      label: text(capability.label, capability.tool_id),
      operations: stringArray(capability.operations, `planner_capabilities.capabilities[${index}].operations`),
      score_range: optionalArray(capability.score_range, `planner_capabilities.capabilities[${index}].score_range`).map(value => number(value)),
    };
  });
  return {
    ...item,
    enabled: Boolean(item.enabled),
    llm_enabled: Boolean(item.llm_enabled),
    planner: text(item.planner),
    capabilities,
    fusion_methods: stringArray(item.fusion_methods, "planner_capabilities.fusion_methods"),
    modes: stringArray(item.modes, "planner_capabilities.modes") as PlannerLabCapabilities["modes"],
  } as PlannerLabCapabilities;
}

export function normalizePlanSetResponse(value: unknown): PlanSetResponse {
  const item = record(value, "plan_set");
  const plans = array(item.plans, "plan_set.plans").map((plan, index) => normalizeCandidatePlan(plan, `plan_set.plans[${index}]`));
  return {
    ...item,
    mode: text(item.mode, "assist") as PlanSetResponse["mode"],
    query_intent: text(item.query_intent),
    constraints: stringArray(item.constraints, "plan_set.constraints"),
    negative_constraints: stringArray(item.negative_constraints, "plan_set.negative_constraints"),
    available_modalities: stringArray(item.available_modalities, "plan_set.available_modalities"),
    plans,
    clarifications: optionalArray(item.clarifications, "plan_set.clarifications") as PlanSetResponse["clarifications"],
    planner_trace: item.planner_trace && typeof item.planner_trace === "object" ? item.planner_trace : { status: "fallback" },
    scope: item.scope && typeof item.scope === "object" ? item.scope : {},
  } as PlanSetResponse;
}

export function normalizePlannerExecution(value: unknown): PlannerExecution {
  const item = record(value, "planner_execution");
  const results = array(item.results, "planner_execution.results").map((result, index) =>
    normalizeSearchResult(result, `planner_execution.results[${index}]`),
  );
  const trace = optionalArray(item.trace, "planner_execution.trace").map((entry, index) => {
    const traceItem = record(entry, `planner_execution.trace[${index}]`);
    return {
      ...traceItem,
      step: normalizePlanStep(traceItem.step, `planner_execution.trace[${index}].step`),
      top_k_jaccard: number(traceItem.top_k_jaccard),
      rank_stability: number(traceItem.rank_stability),
      elapsed_seconds: number(traceItem.elapsed_seconds),
    };
  });
  return {
    ...item,
    execution_id: requiredText(item.execution_id, "planner_execution.execution_id"),
    plan: normalizeCandidatePlan(item.plan, "planner_execution.plan"),
    executed_steps: number(item.executed_steps),
    stop_reason: text(item.stop_reason, "completed"),
    elapsed_seconds: number(item.elapsed_seconds),
    count: number(item.count, results.length),
    above_count: number(item.above_count),
    accepted_steps: number(item.accepted_steps),
    skipped_steps: number(item.skipped_steps),
    rolled_back_steps: number(item.rolled_back_steps),
    results,
    trace,
    scope: item.scope && typeof item.scope === "object" ? item.scope : {},
  } as PlannerExecution;
}

export function normalizeOrchestrationProfiles(value: unknown): OrchestrationProfiles {
  const item = record(value, "orchestration_profiles");
  return {
    ...item,
    enabled: Boolean(item.enabled),
    default_profile: text(item.default_profile),
    profiles: optionalArray(item.profiles, "orchestration_profiles.profiles") as OrchestrationProfiles["profiles"],
  } as OrchestrationProfiles;
}

export function normalizeSpeakerView(value: unknown): SpeakerView {
  const item = record(value, "speaker_view");
  return {
    ...item,
    video_id: requiredText(item.video_id, "speaker_view.video_id"),
    tracks: optionalArray(item.tracks, "speaker_view.tracks") as SpeakerView["tracks"],
    utterances: optionalArray(item.utterances, "speaker_view.utterances") as SpeakerView["utterances"],
  } as SpeakerView;
}

export function normalizeFaceGalleryView(value: unknown): FaceGalleryView {
  const item = record(value, "face_gallery");
  return {
    ...item,
    video_id: requiredText(item.video_id, "face_gallery.video_id"),
    groups: optionalArray(item.groups, "face_gallery.groups") as FaceGalleryView["groups"],
  } as FaceGalleryView;
}

export function normalizeVoiceSearchResponse(value: unknown): { count: number; results: VoiceHit[] } {
  const item = record(value, "voice_search_response");
  const results = optionalArray(item.results, "voice_search_response.results") as VoiceHit[];
  return { ...item, count: number(item.count, results.length), results };
}
